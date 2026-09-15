# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import hashlib
import hmac
import json
import os
import platform
import secrets
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from core.cluster import store
from core.cluster.netinfo import local_addresses
from core.cluster.planner import family_from_devices
from core.cluster.relay import normalize_ip
from core.cluster.rpc_server import (
    RpcServerProcess,
    TensorCache,
    adopt_child,
    atomic_write_json,
    build_identity,
    cluster_root,
    default_cache_cap_bytes,
    detect_local_family,
    find_llama_server_binary,
    find_rpc_server_binary,
    forget_child,
    free_local_port,
    install_marker,
    llama_cache_dir,
    spawn_child,
    spawn_kwargs,
)
from loggers import get_logger

logger = get_logger(__name__)

CONTROL_API_VERSION = 1
PAIRING_TTL_S = 15 * 60
PAIRING_MAX_ATTEMPTS = 20
LEASE_TTL_S = 90.0
LEASE_MAX_TTL_S = 300.0
SEAL_QUIET_S = 2.0
WATCH_INTERVAL_S = 5.0
CRASH_WINDOW_S = 600.0
MAX_RESTARTS = 3
_CODE_ALPHABET = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"
_CODE_LENGTH = 8
_LAST_SEEN_WRITE_S = 60.0


def host_name() -> str:
    name = socket.gethostname() or "Unsloth"
    return name[:-6] if name.endswith(".local") else name


def token_digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def normalize_code(code: str) -> str:
    return "".join(ch for ch in str(code).upper() if ch.isalnum())


class ClusterServiceError(RuntimeError):
    def __init__(self, code: str, status: int = 400):
        super().__init__(code)
        self.code = code
        self.status = status


class RelayProcess:
    def __init__(
        self,
        listen_port: int,
        target_port: int,
        allow_file: Path,
        on_event: Callable[[dict], None],
    ):
        self.listen_port = listen_port
        self.target_port = target_port
        self.allow_file = allow_file
        self._on_event = on_event
        self._proc: Optional[subprocess.Popen] = None
        self._lines: deque[str] = deque(maxlen = 100)
        self._listening = threading.Event()
        self._failure: Optional[str] = None

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self, timeout: float = 15.0) -> None:
        script = Path(__file__).with_name("relay.py")
        cmd = [
            sys.executable,
            "-u",
            str(script),
            "--listen-host",
            "0.0.0.0",
            "--listen-port",
            str(self.listen_port),
            "--target-host",
            "127.0.0.1",
            "--target-port",
            str(self.target_port),
            "--allow-file",
            str(self.allow_file),
        ]
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        self._proc = spawn_child(
            lambda: subprocess.Popen(
                cmd,
                stdin = subprocess.PIPE,
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
        threading.Thread(target = self._read_output, name = "cluster-relay-output", daemon = True).start()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._listening.is_set():
                return
            if self._failure or self._proc.poll() is not None:
                break
            time.sleep(0.02)
        failure = self._failure or "; ".join(list(self._lines)[-3:]) or "relay did not start"
        self.stop()
        raise ClusterServiceError(failure, 500)

    def _read_output(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        try:
            for raw in proc.stdout:
                line = raw.strip()
                if not line:
                    continue
                self._lines.append(line)
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                kind = event.get("event")
                if kind == "listening":
                    self._listening.set()
                elif kind == "bind_failed":
                    self._failure = f"port_in_use:{self.listen_port}"
                try:
                    self._on_event(event)
                except Exception:
                    logger.debug("relay event handler failed", exc_info = True)
        except (OSError, ValueError):
            pass

    def stop(self, timeout: float = 3.0) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except OSError:
            pass
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout = timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
            except OSError:
                pass
        forget_child(proc.pid)


class ControlServer:
    def __init__(self, app: FastAPI, port: int):
        self.app = app
        self.port = port
        self._server = None
        self._thread: Optional[threading.Thread] = None

    def start(self, timeout: float = 10.0) -> None:
        import uvicorn

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            if sys.platform != "win32":
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", self.port))
            sock.listen(64)
            sock.set_inheritable(False)
        except OSError as exc:
            sock.close()
            raise ClusterServiceError(f"port_in_use:{self.port}", 409) from exc
        config = uvicorn.Config(
            self.app,
            lifespan = "off",
            log_config = None,
            access_log = False,
            server_header = False,
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(
            target = self._server.run,
            kwargs = {"sockets": [sock]},
            name = "cluster-control",
            daemon = True,
        )
        self._thread.start()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._server.started:
                return
            if not self._thread.is_alive():
                break
            time.sleep(0.02)
        self.stop()
        raise ClusterServiceError("control_listener_failed", 500)

    def stop(self, timeout: float = 3.0) -> None:
        server, thread = self._server, self._thread
        self._server, self._thread = None, None
        if server is not None:
            server.should_exit = True
        if thread is not None:
            thread.join(timeout)


class SharingService:
    def __init__(self):
        self._lock = threading.RLock()
        self._op_lock = threading.Lock()
        self.state = "off"
        self.error: Optional[str] = None
        self._rpc: Optional[RpcServerProcess] = None
        self._relay: Optional[RelayProcess] = None
        self._control: Optional[ControlServer] = None
        self._cache = TensorCache(llama_cache_dir())
        self._llama_server: Optional[str] = None
        self._family: Optional[str] = None
        self._build: Optional[str] = None
        self._direct = False
        self._ports: dict[str, int] = {}
        self._pairing: Optional[dict] = None
        self._leases: dict[str, dict] = {}
        self._session: Optional[dict] = None
        self._last_transfer: Optional[dict] = None
        self._rejected: Optional[dict] = None
        self._started_at: Optional[float] = None
        self._last_seen_written: dict[str, float] = {}
        self._available: Optional[tuple[float, bool]] = None
        self._generation = 0
        self._crash_times: list[float] = []

    @property
    def allow_file(self) -> Path:
        return cluster_root() / "relay-allow.json"

    def available(self) -> bool:
        now = time.monotonic()
        cached = self._available
        if cached is not None and now - cached[0] < 30.0:
            return cached[1]
        found = find_rpc_server_binary() is not None
        self._available = (now, found)
        return found

    def start_async(self) -> dict:
        with self._lock:
            if self.state in ("starting", "online"):
                return self.status()
            self.state, self.error = "starting", None
        threading.Thread(target = self.start, name = "cluster-share-start", daemon = True).start()
        return self.status()

    def start(self, persist: bool = True) -> dict:
        with self._op_lock:
            with self._lock:
                if self.state == "online":
                    return self.status()
                self.state, self.error = "starting", None
            try:
                self._start_locked()
                if persist:
                    store.update_share_settings(auto_start = True)
            except Exception as exc:
                code = exc.code if isinstance(exc, ClusterServiceError) else str(exc)
                logger.warning("GPU sharing failed to start: %s", code)
                self._teardown(seal = False)
                with self._lock:
                    self.state, self.error = "error", code
        return self.status()

    def _start_locked(self) -> None:
        llama_server = find_llama_server_binary()
        binary = find_rpc_server_binary(llama_server)
        if not binary:
            raise ClusterServiceError("rpc_server_missing", 409)
        settings = store.share_settings()
        direct = bool(settings.get("direct"))
        control_port = int(settings.get("control_port") or store.DEFAULT_CONTROL_PORT)
        rpc_port = int(settings.get("rpc_port") or store.DEFAULT_RPC_PORT)

        removed = self._cache.sanitize()
        if removed:
            logger.info("Removed %d unsealed tensor cache files", removed)
        self._cache.prune(self._cache_cap_bytes(settings))

        control = ControlServer(build_control_app(self), control_port)
        control.start()
        self._control = control
        relay = None
        if direct:
            rpc = RpcServerProcess(binary, "0.0.0.0", rpc_port, llama_cache_dir(), on_event = self._on_rpc_event)
        else:
            atomic_write_json(self.allow_file, {"allow": {}})
            internal = free_local_port()
            relay = RelayProcess(rpc_port, internal, self.allow_file, self._on_relay_event)
            relay.start()
            self._relay = relay
            rpc = RpcServerProcess(binary, "127.0.0.1", internal, llama_cache_dir(), on_event = self._on_rpc_event)
        self._rpc = rpc
        banner = rpc.start()
        if banner.version:
            store.set_local_rpc_proto(list(banner.version))
        marker = install_marker(llama_server)
        family = family_from_devices(d["name"] for d in banner.devices) or detect_local_family(llama_server)
        with self._lock:
            self._llama_server = llama_server
            self._family = family
            self._build = build_identity(marker)
            self._direct = direct
            self._ports = {"control": control_port, "rpc": rpc_port}
            self._started_at = time.time()
            self._pairing = self._new_pairing()
            self._generation += 1
            generation = self._generation
            self.state, self.error = "online", None
        threading.Thread(target = self._watch, args = (generation,), name = "cluster-share-watch", daemon = True).start()
        logger.info(
            "GPU sharing online: control :%d, rpc :%d (%s), devices %s",
            control_port,
            rpc_port,
            "direct" if direct else "relay",
            ", ".join(d["name"] for d in banner.devices),
        )

    def stop(self, persist: bool = True) -> dict:
        with self._op_lock:
            self._teardown(seal = True)
            with self._lock:
                self.state, self.error = "off", None
                self._leases.clear()
                self._session = None
                self._pairing = None
            if persist:
                store.update_share_settings(auto_start = False)
        return self.status()

    def _teardown(self, seal: bool) -> None:
        control, relay, rpc = self._control, self._relay, self._rpc
        self._control = self._relay = self._rpc = None
        busy = bool(rpc and rpc.active_session)
        for part in (control, relay, rpc):
            if part is None:
                continue
            try:
                part.stop()
            except Exception:
                logger.debug("cluster teardown step failed", exc_info = True)
        if seal and not busy:
            try:
                self._cache.seal()
            except OSError:
                pass

    def restart(self) -> dict:
        self.stop(persist = False)
        return self.start(persist = False)

    def _cache_cap_bytes(self, settings: Optional[dict] = None) -> int:
        settings = settings or store.share_settings()
        cap_gib = settings.get("cache_cap_gib")
        if isinstance(cap_gib, (int, float)) and cap_gib > 0:
            return int(cap_gib * 1024**3)
        probe = llama_cache_dir()
        while not probe.exists() and probe.parent != probe:
            probe = probe.parent
        return default_cache_cap_bytes(probe)

    def _on_relay_event(self, event: dict) -> None:
        kind = event.get("event")
        with self._lock:
            if kind == "opened":
                peer = event.get("peer")
                self._session = {"peer": peer, "since": event.get("ts"), "head": self._head_name_for(peer)}
            elif kind == "closed":
                self._session = None
                self._last_transfer = {
                    "peer": event.get("peer"),
                    "head": self._head_name_for(event.get("peer")),
                    "bytes_in": event.get("bytes_in"),
                    "bytes_out": event.get("bytes_out"),
                    "seconds": event.get("seconds"),
                    "at": event.get("ts"),
                }
            elif kind == "rejected":
                self._rejected = {"peer": event.get("peer"), "reason": event.get("reason"), "at": event.get("ts")}

    def _on_rpc_event(self, event: str) -> None:
        if event == "session_open" and self._relay is None:
            with self._lock:
                self._session = {"peer": None, "since": time.time(), "head": None}
        elif event == "session_closed":
            if self._relay is None:
                with self._lock:
                    self._session = None
            threading.Thread(target = self._after_session, name = "cluster-cache-seal", daemon = True).start()

    def _after_session(self) -> None:
        rpc = self._rpc
        time.sleep(SEAL_QUIET_S)
        if rpc is None or rpc.active_session:
            return
        try:
            self._cache.seal(SEAL_QUIET_S)
            if not rpc.active_session:
                self._cache.prune(self._cache_cap_bytes())
        except OSError:
            logger.debug("tensor cache seal failed", exc_info = True)

    def _head_name_for(self, peer: Optional[str]) -> Optional[str]:
        for lease in self._leases.values():
            if lease["ip"] == peer:
                return lease["name"]
        return None

    def _new_pairing(self) -> dict:
        code = "".join(secrets.choice(_CODE_ALPHABET) for _ in range(_CODE_LENGTH))
        return {"code": code, "expires_at": time.time() + PAIRING_TTL_S, "attempts": 0}

    def pairing(self) -> Optional[dict]:
        with self._lock:
            if self.state != "online":
                return None
            if self._pairing is None or self._pairing["expires_at"] <= time.time():
                self._pairing = self._new_pairing()
            code = self._pairing["code"]
            return {"code": f"{code[:4]}-{code[4:]}", "expires_at": self._pairing["expires_at"]}

    def regenerate_pairing(self) -> Optional[dict]:
        with self._lock:
            if self.state == "online":
                self._pairing = self._new_pairing()
        return self.pairing()

    def pair(self, code: str, head_id: str, head_name: str, peer: str) -> dict:
        with self._lock:
            if self.state != "online":
                raise ClusterServiceError("not_sharing", 503)
            if head_id == store.identity()["node_id"]:
                raise ClusterServiceError("self_pairing", 400)
            current = self._pairing
            if current is None or current["expires_at"] <= time.time():
                raise ClusterServiceError("code_expired", 403)
            current["attempts"] += 1
            if not hmac.compare_digest(normalize_code(code), current["code"]):
                if current["attempts"] >= PAIRING_MAX_ATTEMPTS:
                    self._pairing = None
                raise ClusterServiceError("code_invalid", 403)
            token = secrets.token_urlsafe(32)
            now = time.time()
            heads = [h for h in store.paired_heads() if h["id"] != head_id]
            heads.append(
                {
                    "id": head_id,
                    "name": (head_name or "Unsloth")[:80],
                    "token_sha256": token_digest(token),
                    "paired_at": now,
                    "last_seen": now,
                    "last_address": peer,
                }
            )
            store.save_paired_heads(heads)
            self._pairing = self._new_pairing()
        logger.info("Paired cluster head %s from %s", head_name, peer)
        return {"token": token, **self.describe()}

    def authenticate(self, token: str) -> Optional[dict]:
        if not token:
            return None
        digest = token_digest(token)
        for head in store.paired_heads():
            stored = head.get("token_sha256")
            if isinstance(stored, str) and hmac.compare_digest(stored, digest):
                return head
        return None

    def _touch_head(self, head: dict, peer: str) -> None:
        now = time.time()
        if now - self._last_seen_written.get(head["id"], 0.0) < _LAST_SEEN_WRITE_S:
            return
        self._last_seen_written[head["id"]] = now
        heads = store.paired_heads()
        for record in heads:
            if record["id"] == head["id"]:
                record["last_seen"] = now
                record["last_address"] = peer
        store.save_paired_heads(heads)

    def revoke_head(self, head_id: str) -> bool:
        heads = store.paired_heads()
        kept = [h for h in heads if h["id"] != head_id]
        store.save_paired_heads(kept)
        with self._lock:
            self._leases.pop(head_id, None)
            self._write_allowlist()
        return len(kept) != len(heads)

    def _expire_leases(self, now: float) -> None:
        expired = [hid for hid, lease in self._leases.items() if lease["expires_at"] <= now]
        for hid in expired:
            self._leases.pop(hid, None)
        if expired:
            self._write_allowlist()

    def _write_allowlist(self) -> None:
        if self._relay is None:
            return
        allow: dict[str, float] = {}
        for lease in self._leases.values():
            allow[lease["ip"]] = max(allow.get(lease["ip"], 0.0), lease["expires_at"])
        try:
            atomic_write_json(self.allow_file, {"allow": allow})
        except OSError:
            logger.warning("Could not write the relay allowlist", exc_info = True)

    def busy_for(self, head_id: Optional[str], peer: Optional[str] = None) -> bool:
        now = time.time()
        with self._lock:
            self._expire_leases(now)
            if any(hid != head_id for hid in self._leases):
                return True
            session = self._session
            return bool(session and session.get("peer") and peer and session["peer"] != peer)

    def lease(self, head: dict, peer: str, ttl: float = LEASE_TTL_S) -> dict:
        ttl = min(max(float(ttl), 10.0), LEASE_MAX_TTL_S)
        with self._lock:
            if self.state != "online":
                raise ClusterServiceError("not_sharing", 503)
            if self.busy_for(head["id"], peer):
                raise ClusterServiceError("busy", 409)
            self._leases[head["id"]] = {
                "ip": peer,
                "expires_at": time.time() + ttl,
                "name": head.get("name"),
                "head_id": head["id"],
            }
            self._write_allowlist()
        self._touch_head(head, peer)
        return {"lease_ttl": ttl, **self.describe(head["id"], peer)}

    def release(self, head: dict) -> None:
        with self._lock:
            if self._leases.pop(head["id"], None) is not None:
                self._write_allowlist()

    def _memory(self) -> tuple[int, int]:
        rpc = self._rpc
        devices = rpc.banner.devices if rpc else []
        banner_free = sum(d["free_mib"] for d in devices)
        banner_total = sum(d["total_mib"] for d in devices)
        try:
            from core.inference.llama_cpp import LlamaCppBackend

            rows = LlamaCppBackend._get_gpu_memory(self._llama_server)
        except Exception:
            rows = []
        total = sum(max(0, int(r[2])) for r in rows) if rows else 0
        if total > 0:
            return sum(max(0, int(r[1])) for r in rows), total
        return banner_free, banner_total

    def describe(self, head_id: Optional[str] = None, peer: Optional[str] = None) -> dict:
        rpc = self._rpc
        free_mib, total_mib = self._memory()
        return {
            "api": CONTROL_API_VERSION,
            "node_id": store.identity()["node_id"],
            "name": host_name(),
            "platform": sys.platform,
            "machine": platform.machine(),
            "family": self._family,
            "build": self._build,
            "rpc_proto": list(rpc.banner.version) if rpc and rpc.banner.version else None,
            "transport": rpc.banner.transport if rpc else None,
            "control_port": self._ports.get("control"),
            "rpc_port": self._ports.get("rpc"),
            "direct": self._direct,
            "addresses": local_addresses(),
            "devices": [dict(d) for d in (rpc.banner.devices if rpc else [])],
            "free_mib": free_mib,
            "total_mib": total_mib,
            "busy": self.busy_for(head_id, peer),
        }

    def status(self) -> dict:
        settings = store.share_settings()
        with self._lock:
            state, error = self.state, self.error
            session = dict(self._session) if self._session else None
            last_transfer = dict(self._last_transfer) if self._last_transfer else None
            rejected = dict(self._rejected) if self._rejected else None
            leases = [
                {"head_id": l["head_id"], "name": l["name"], "address": l["ip"], "expires_at": l["expires_at"]}
                for l in self._leases.values()
            ]
        online = state == "online"
        rpc = self._rpc
        payload = {
            "available": self.available(),
            "state": state,
            "error": error,
            "auto_start": bool(settings.get("auto_start")),
            "direct": bool(settings.get("direct")),
            "control_port": settings.get("control_port"),
            "rpc_port": settings.get("rpc_port"),
            "cache": {**self._cache.usage(), "cap_bytes": self._cache_cap_bytes(settings)},
            "cache_cap_gib": settings.get("cache_cap_gib"),
            "family": self._family if online else None,
            "build": self._build if online else None,
            "devices": [dict(d) for d in rpc.banner.devices] if online and rpc else [],
            "transport": rpc.banner.transport if online and rpc else None,
            "addresses": local_addresses(),
            "pairing": self.pairing() if online else None,
            "session": session,
            "last_transfer": last_transfer,
            "rejected": rejected,
            "leases": leases,
            "started_at": self._started_at if online else None,
            "paired_heads": [
                {
                    "id": h["id"],
                    "name": h.get("name"),
                    "paired_at": h.get("paired_at"),
                    "last_seen": h.get("last_seen"),
                    "last_address": h.get("last_address"),
                }
                for h in store.paired_heads()
            ],
        }
        if online and not self._parts_alive():
            payload["state"] = "error"
            payload["error"] = "rpc_server_exited"
        return payload

    def _parts_alive(self) -> bool:
        rpc, relay = self._rpc, self._relay
        if rpc is not None and not rpc.alive:
            return False
        return not (isinstance(relay, RelayProcess) and not relay.alive)

    def _watch(self, generation: int) -> None:
        while True:
            time.sleep(WATCH_INTERVAL_S)
            with self._lock:
                if self.state != "online" or self._generation != generation:
                    return
            if self._parts_alive():
                continue
            rpc = self._rpc
            detail = "; ".join(line for line in (rpc.tail(4) if rpc else []) if line.strip())
            now = time.monotonic()
            recent = [t for t in self._crash_times if now - t < CRASH_WINDOW_S]
            if len(recent) >= MAX_RESTARTS:
                logger.warning("GPU sharing stopped: ggml-rpc-server keeps exiting (%s)", detail)
                self._teardown(seal = False)
                with self._lock:
                    self.state, self.error = "error", "rpc_server_exited"
                return
            self._crash_times = recent + [now]
            logger.warning("GPU sharing: restarting ggml-rpc-server after it exited (%s)", detail)
            self.restart()
            return

    def update_settings(self, **changes) -> dict:
        allowed = {k: v for k, v in changes.items() if k in ("direct", "cache_cap_gib", "control_port", "rpc_port")}
        before = store.share_settings()
        after = store.update_share_settings(**allowed)
        restart_keys = ("direct", "control_port", "rpc_port")
        if self.state == "online" and any(before.get(k) != after.get(k) for k in restart_keys):
            if self._session:
                raise ClusterServiceError("busy", 409)
            self.restart()
        elif before.get("cache_cap_gib") != after.get("cache_cap_gib"):
            rpc = self._rpc
            if rpc is None or not rpc.active_session:
                self._cache.prune(self._cache_cap_bytes(after))
        return self.status()

    def clear_cache(self) -> dict:
        rpc = self._rpc
        if rpc is not None and rpc.active_session:
            raise ClusterServiceError("busy", 409)
        self._cache.clear()
        return self.status()


class PairBody(BaseModel):
    code: str = Field(min_length = 4, max_length = 32)
    head_id: str = Field(min_length = 8, max_length = 64)
    head_name: str = Field(default = "", max_length = 120)


class LeaseBody(BaseModel):
    ttl: float = Field(default = LEASE_TTL_S, ge = 10.0, le = LEASE_MAX_TTL_S)


def _peer(request: Request) -> str:
    return normalize_ip(request.client.host) if request.client else ""


def build_control_app(service: SharingService) -> FastAPI:
    app = FastAPI(docs_url = None, redoc_url = None, openapi_url = None)

    def head_for(authorization: Optional[str]) -> dict:
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(status_code = 401, detail = "unauthorized")
        head = service.authenticate(authorization[7:].strip())
        if head is None:
            raise HTTPException(status_code = 401, detail = "unauthorized")
        return head

    def fail(exc: ClusterServiceError):
        raise HTTPException(status_code = exc.status, detail = exc.code)

    @app.get("/cluster/v1/hello")
    def hello():
        return {"service": "unsloth-cluster", "api": CONTROL_API_VERSION, "sharing": service.state == "online"}

    @app.post("/cluster/v1/pair")
    def pair(body: PairBody, request: Request):
        try:
            return service.pair(body.code, body.head_id, body.head_name, _peer(request))
        except ClusterServiceError as exc:
            fail(exc)

    @app.get("/cluster/v1/info")
    def info(request: Request, authorization: Optional[str] = Header(default = None)):
        head = head_for(authorization)
        return service.describe(head["id"], _peer(request))

    @app.post("/cluster/v1/lease")
    def lease(body: LeaseBody, request: Request, authorization: Optional[str] = Header(default = None)):
        head = head_for(authorization)
        try:
            return service.lease(head, _peer(request), body.ttl)
        except ClusterServiceError as exc:
            fail(exc)

    @app.post("/cluster/v1/release")
    def release(authorization: Optional[str] = Header(default = None)):
        service.release(head_for(authorization))
        return {"released": True}

    @app.post("/cluster/v1/unpair")
    def unpair(authorization: Optional[str] = Header(default = None)):
        head = head_for(authorization)
        service.revoke_head(head["id"])
        return {"unpaired": True}

    return app


_service: Optional[SharingService] = None
_service_lock = threading.Lock()


def get_sharing_service() -> SharingService:
    global _service
    with _service_lock:
        if _service is None:
            _service = SharingService()
        return _service


def _announce(status: dict) -> None:
    pairing = status.get("pairing") or {}
    addresses = [a["address"] for a in status.get("addresses") or [] if not a.get("public")]
    if status.get("state") != "online":
        print(f"GPU sharing could not start: {status.get('error')}", flush = True)
        return
    print(
        "GPU sharing is on. To add this computer, open Settings > Cluster on the other Unsloth "
        f"and enter {addresses[0] if addresses else host_name()} with code {pairing.get('code')}",
        flush = True,
    )


def maybe_auto_start_sharing() -> bool:
    requested = os.environ.get("UNSLOTH_CLUSTER_SHARE") == "1"
    try:
        remembered = bool(store.share_settings().get("auto_start"))
    except Exception:
        remembered = False
    if not requested and not remembered:
        return False

    def run():
        status = get_sharing_service().start(persist = requested)
        if requested:
            _announce(status)

    threading.Thread(target = run, name = "cluster-share-autostart", daemon = True).start()
    return True


def shutdown_sharing() -> None:
    service = _service
    if service is not None and service.state != "off":
        try:
            service.stop(persist = False)
        except Exception:
            logger.debug("GPU sharing shutdown failed", exc_info = True)
