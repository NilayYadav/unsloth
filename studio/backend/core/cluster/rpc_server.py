# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import socket
import stat
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from loggers import get_logger

logger = get_logger(__name__)

BINARY_NAMES = ("ggml-rpc-server", "rpc-server")
MIN_FREE_DISK_BYTES = 10 * 1024**3
DEFAULT_CACHE_FRACTION = 0.25
MAX_DEFAULT_CACHE_BYTES = 256 * 1024**3
BANNER_GRACE_S = 5.0

_CACHE_NAME = re.compile(r"^[0-9a-f]{16}$")
_DEVICE_LINE = re.compile(r"^\s+([A-Za-z][A-Za-z0-9_\-]*):\s(.+?)\s\((\d+) MiB, (\d+) MiB free\)\s*$")
_VERSION_LINE = re.compile(r"Starting RPC server v(\d+)\.(\d+)\.(\d+)")
_TRANSPORT_LINE = re.compile(r"^\s*transport\s*:\s*(.+?)\s*$")


def cluster_root() -> Path:
    from utils.paths.storage_roots import cache_root

    return cache_root() / "cluster"


def llama_cache_dir() -> Path:
    return cluster_root() / "llama"


def atomic_write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents = True, exist_ok = True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(payload), encoding = "utf-8")
    os.replace(tmp, path)


def find_llama_server_binary() -> Optional[str]:
    try:
        from core.inference.llama_cpp import LlamaCppBackend

        return LlamaCppBackend._find_llama_server_binary()
    except Exception:
        return None


def find_rpc_server_binary(llama_server: Optional[str] = None) -> Optional[str]:
    override = os.environ.get("UNSLOTH_RPC_SERVER_PATH")
    if override and Path(override).is_file():
        return override
    server = llama_server or find_llama_server_binary()
    if not server:
        return None
    suffix = ".exe" if sys.platform == "win32" else ""
    launched = Path(server).parent
    resolved = Path(server).resolve().parent
    candidates = dict.fromkeys(
        [launched, resolved, launched / "build" / "bin", resolved / "build" / "bin", resolved.parent / "build" / "bin"]
    )
    for directory in candidates:
        for name in BINARY_NAMES:
            path = directory / f"{name}{suffix}"
            if path.is_file() and _ensure_executable(path):
                return str(path)
    return None


def _ensure_executable(path: Path) -> bool:
    if os.access(path, os.X_OK):
        return True
    if sys.platform == "win32":
        return False
    try:
        path.chmod(path.stat().st_mode | 0o111)
    except OSError:
        return False
    return os.access(path, os.X_OK)


def install_marker(binary: Optional[str]) -> dict:
    try:
        from utils.llama_cpp_freshness import read_install_marker

        marker = read_install_marker(binary)
    except Exception:
        marker = None
    return marker if isinstance(marker, dict) else {}


def build_identity(marker: dict) -> Optional[str]:
    for key in ("source_commit", "release_tag", "tag"):
        value = marker.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def detect_local_family(binary: Optional[str] = None) -> Optional[str]:
    from core.cluster.planner import family_from_backend_label

    marker = install_marker(binary)
    for key in ("backend", "bundle_profile", "asset"):
        family = family_from_backend_label(marker.get(key))
        if family:
            return family
    if sys.platform == "darwin":
        return "metal" if platform.machine() == "arm64" else None
    try:
        from core.inference.llama_cpp import LlamaCppBackend

        if binary and LlamaCppBackend._is_vulkan_backend(binary):
            return "vulkan"
    except Exception:
        pass
    if shutil.which("nvidia-smi"):
        return "cuda"
    if shutil.which("rocm-smi") or shutil.which("amd-smi") or Path("/opt/rocm").is_dir():
        return "rocm"
    return None


def free_local_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def port_is_listening(host: str, port: int, timeout: float = 0.3) -> bool:
    try:
        with socket.create_connection((host, port), timeout = timeout):
            return True
    except OSError:
        return False


def _child_env(binary: str) -> dict:
    try:
        from core.inference.llama_cpp import LlamaCppBackend

        env = dict(LlamaCppBackend._llama_server_env_for_binary(binary))
    except Exception:
        env = dict(os.environ)
    for name in [k for k in env if k.startswith("LLAMA_ARG_")]:
        env.pop(name, None)
    return env


def spawn_kwargs() -> dict:
    kwargs: dict = {}
    try:
        from utils.subprocess_compat import windows_hidden_subprocess_kwargs

        kwargs.update(windows_hidden_subprocess_kwargs())
    except Exception:
        pass
    try:
        from utils.process_lifetime import child_popen_kwargs

        kwargs.update(child_popen_kwargs())
    except Exception:
        pass
    return kwargs


def spawn_child(factory: Callable[[], subprocess.Popen]) -> subprocess.Popen:
    # PDEATHSIG follows the forking thread, and sharing starts from short-lived threads
    try:
        from utils.process_lifetime import spawn_on_lifetime_thread
    except Exception:
        return factory()
    return spawn_on_lifetime_thread(factory)


def adopt_child(pid: Optional[int]) -> None:
    try:
        from utils.process_lifetime import adopt_pid

        adopt_pid(pid)
    except Exception:
        pass


def forget_child(pid: Optional[int]) -> None:
    try:
        from utils.process_lifetime import forget_pid

        forget_pid(pid)
    except Exception:
        pass


@dataclass
class RpcBanner:
    version: Optional[tuple[int, int, int]] = None
    devices: list[dict] = field(default_factory = list)
    transport: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "version": list(self.version) if self.version else None,
            "devices": [dict(d) for d in self.devices],
            "transport": self.transport,
        }


def parse_banner_line(banner: RpcBanner, line: str) -> None:
    match = _VERSION_LINE.search(line)
    if match:
        banner.version = tuple(int(g) for g in match.groups())
        return
    match = _DEVICE_LINE.match(line)
    if match:
        name, description, total, free = match.groups()
        if all(d["name"] != name for d in banner.devices):
            banner.devices.append(
                {"name": name, "description": description, "total_mib": int(total), "free_mib": int(free)}
            )
        return
    match = _TRANSPORT_LINE.match(line)
    if match:
        banner.transport = match.group(1)


class RpcServerProcess:
    def __init__(
        self,
        binary: str,
        host: str,
        port: int,
        llama_cache: Path,
        on_event: Optional[Callable[[str], None]] = None,
    ):
        self.binary = binary
        self.host = host
        self.port = port
        self.llama_cache = Path(llama_cache)
        self.banner = RpcBanner()
        self.active_session = False
        self.sessions = 0
        self._on_event = on_event
        self._proc: Optional[subprocess.Popen] = None
        self._lines: deque[str] = deque(maxlen = 400)

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def tail(self, count: int = 20) -> list[str]:
        return list(self._lines)[-count:]

    def start(self, timeout: float = 180.0) -> RpcBanner:
        self.llama_cache.mkdir(parents = True, exist_ok = True)
        env = _child_env(self.binary)
        # ggml-rpc-server stores its tensor cache under $LLAMA_CACHE/rpc
        env["LLAMA_CACHE"] = str(self.llama_cache)
        cmd = [self.binary, "-H", self.host, "-p", str(self.port), "-c"]
        logger.info("Starting ggml-rpc-server: %s", " ".join(cmd))
        self._proc = spawn_child(
            lambda: subprocess.Popen(
                cmd,
                stdin = subprocess.DEVNULL,
                stdout = subprocess.PIPE,
                stderr = subprocess.STDOUT,
                text = True,
                encoding = "utf-8",
                errors = "replace",
                bufsize = 1,
                env = env,
                **spawn_kwargs(),
            )
        )
        adopt_child(self._proc.pid)
        threading.Thread(target = self._read_output, name = "ggml-rpc-server-output", daemon = True).start()
        probe_host = "127.0.0.1" if self.host in ("0.0.0.0", "") else self.host
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                detail = "; ".join(line for line in self.tail(6) if line.strip())
                self.stop()
                raise RuntimeError(f"ggml-rpc-server exited during startup: {detail or 'no output'}")
            # the banner is printf'd into a pipe and only flushed when a client connects,
            # so the port probe comes first and is what releases the device list
            if port_is_listening(probe_host, self.port):
                banner_deadline = time.monotonic() + BANNER_GRACE_S
                while not self.banner.devices and time.monotonic() < banner_deadline and self.alive:
                    time.sleep(0.02)
                return self.banner
            time.sleep(0.05)
        self.stop()
        raise RuntimeError("ggml-rpc-server did not open its port in time")

    def _read_output(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        try:
            for raw in proc.stdout:
                line = raw.rstrip("\n")
                self._lines.append(line)
                parse_banner_line(self.banner, line)
                event = None
                if "Accepted client connection" in line:
                    self.active_session = True
                    self.sessions += 1
                    event = "session_open"
                elif "Client connection closed" in line:
                    self.active_session = False
                    event = "session_closed"
                if event and self._on_event is not None:
                    try:
                        self._on_event(event)
                    except Exception:
                        logger.debug("rpc-server event handler failed", exc_info = True)
        except (OSError, ValueError):
            pass

    def stop(self, timeout: float = 5.0) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout = timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout = timeout)
                except subprocess.TimeoutExpired:
                    pass
            except OSError:
                pass
        forget_child(proc.pid)
        self.active_session = False


def default_cache_cap_bytes(path: Path) -> int:
    try:
        return int(min(shutil.disk_usage(path).total * DEFAULT_CACHE_FRACTION, MAX_DEFAULT_CACHE_BYTES))
    except OSError:
        return 64 * 1024**3


class TensorCache:
    def __init__(self, llama_cache: Path):
        self.root = Path(llama_cache)
        self.directory = self.root / "rpc"
        self.manifest = self.root / "sealed.json"

    def _entries(self) -> list[tuple[Path, os.stat_result]]:
        try:
            scanned = list(os.scandir(self.directory))
        except OSError:
            return []
        entries = []
        for entry in scanned:
            if not _CACHE_NAME.match(entry.name):
                continue
            try:
                st = entry.stat(follow_symlinks = False)
            except OSError:
                continue
            if stat.S_ISREG(st.st_mode):
                entries.append((Path(entry.path), st))
        return entries

    def usage(self) -> dict:
        entries = self._entries()
        return {"bytes": sum(st.st_size for _, st in entries), "files": len(entries)}

    def _sealed(self) -> dict:
        try:
            data = json.loads(self.manifest.read_text(encoding = "utf-8"))
            files = data.get("files")
            return files if isinstance(files, dict) else {}
        except (OSError, ValueError, AttributeError):
            return {}

    def seal(self, quiet_seconds: float = 0.0) -> None:
        horizon = time.time_ns() - int(quiet_seconds * 1e9)
        records = {
            path.name: [st.st_size, st.st_mtime_ns]
            for path, st in self._entries()
            if not quiet_seconds or st.st_mtime_ns <= horizon
        }
        atomic_write_json(self.manifest, {"version": 1, "files": records})

    def sanitize(self) -> int:
        # a write cut short by a crash keeps its hash name, and ggml-rpc-server would
        # later serve the truncated bytes as that tensor; only files sealed after a
        # finished session are trusted
        sealed = self._sealed()
        removed = 0
        for path, st in self._entries():
            record = sealed.get(path.name)
            if isinstance(record, list) and record == [st.st_size, st.st_mtime_ns]:
                continue
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
        if removed:
            self.seal()
        return removed

    def prune(self, cap_bytes: Optional[int], min_free_bytes: int = MIN_FREE_DISK_BYTES) -> int:
        self.directory.mkdir(parents = True, exist_ok = True)
        entries = sorted(self._entries(), key = lambda e: max(e[1].st_atime_ns, e[1].st_mtime_ns))
        total = sum(st.st_size for _, st in entries)
        try:
            free = shutil.disk_usage(self.directory).free
        except OSError:
            free = None
        freed = 0
        for path, st in entries:
            over_cap = cap_bytes is not None and total > cap_bytes
            low_disk = free is not None and free < min_free_bytes
            if not over_cap and not low_disk:
                break
            try:
                path.unlink()
            except OSError:
                continue
            total -= st.st_size
            freed += st.st_size
            if free is not None:
                free += st.st_size
        if freed:
            self.seal()
        return freed

    def clear(self) -> int:
        freed = 0
        for path, st in self._entries():
            try:
                path.unlink()
                freed += st.st_size
            except OSError:
                pass
        self.seal()
        return freed
