# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

# Talking to the remote: SSH exec/rsync/tunnels and the agent HTTP/SSE client.

from __future__ import annotations

import json
import os
import shlex
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

from unsloth_cli.remote import RemoteError
from unsloth_cli.remote.state import RemoteRecord, known_hosts_file, ssh_dir

_STDERR_TAIL_LINES = 15
# Unix sockets cap paths at ~104 bytes; leave room for ssh's ctl-<hash>.
_MAX_CONTROL_DIR_LEN = 55


def _bin(name: str) -> str:
    return os.environ.get(f"UNSLOTH_REMOTE_{name.upper().replace('-', '_')}", name)


def _supports_control_master() -> bool:
    return sys.platform != "win32"


def _control_dir() -> Path:
    preferred = ssh_dir()
    if len(str(preferred)) <= _MAX_CONTROL_DIR_LEN:
        return preferred
    fallback = Path(tempfile.gettempdir()) / f"unsloth-ssh-{os.getuid()}"
    try:
        if fallback.exists() and fallback.stat().st_uid != os.getuid():
            return Path(tempfile.mkdtemp(prefix = "unsloth-ssh-"))
    except OSError:
        pass
    return fallback


def _remote_shell_path(path: str) -> str:
    # Quote for the remote shell, keeping ~ expansion alive.
    if path.startswith("~/"):
        return "~/" + shlex.quote(path[2:])
    return shlex.quote(path)


def _stderr_tail(stderr: str) -> str:
    lines = (stderr or "").strip().splitlines()
    return "\n".join(lines[-_STDERR_TAIL_LINES:])


def free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def scan_host_key(host: str, port: int = 22, timeout: int = 10) -> Tuple[str, str]:
    scan = subprocess.run(
        [_bin("ssh-keyscan"), "-T", str(timeout), "-p", str(port), host],
        capture_output = True, text = True, timeout = timeout + 10,
    )
    keys = [l for l in scan.stdout.splitlines() if l.strip() and not l.startswith("#")]
    if not keys:
        raise RemoteError(
            f"Could not fetch SSH host key from {host}:{port}.",
            hint = _stderr_tail(scan.stderr) or "Check that the host is reachable and sshd is running.",
        )
    preference = ["ssh-ed25519", "ecdsa", "ssh-rsa"]
    keys.sort(key = lambda l: next((i for i, p in enumerate(preference) if p in l), len(preference)))
    fingerprint_proc = subprocess.run(
        [_bin("ssh-keygen"), "-lf", "-"],
        input = keys[0] + "\n", capture_output = True, text = True, timeout = 30,
    )
    fields = fingerprint_proc.stdout.split()
    fingerprint = next((f for f in fields if f.startswith("SHA256:")), None)
    if fingerprint is None:
        raise RemoteError(f"Could not fingerprint the host key of {host}:{port}.")
    return fingerprint, "\n".join(keys) + "\n"


def pin_host_key(known_hosts_text: str) -> None:
    path = known_hosts_file()
    path.parent.mkdir(parents = True, exist_ok = True)
    existing = path.read_text(encoding = "utf-8") if path.exists() else ""
    new_lines = [l for l in known_hosts_text.splitlines() if l and l not in existing]
    if new_lines:
        with open(path, "a", encoding = "utf-8") as f:
            f.write("\n".join(new_lines) + "\n")
    path.chmod(0o600)


class SSHRunner:

    def __init__(
        self,
        destination: str,
        port: int = 22,
        identity_file: Optional[str] = None,
        connect_timeout: int = 15,
    ):
        self.destination = destination
        self.port = port
        self.identity_file = identity_file
        self.connect_timeout = connect_timeout
        self.use_master = _supports_control_master()

    @classmethod
    def for_remote(cls, record: RemoteRecord) -> "SSHRunner":
        return cls(
            destination = record.destination,
            port = record.port,
            identity_file = record.identity_file,
        )

    def base_options(self) -> List[str]:
        options = [
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=yes",
            "-o", f"UserKnownHostsFile={known_hosts_file()}",
            "-o", f"ConnectTimeout={self.connect_timeout}",
            "-p", str(self.port),
        ]
        if self.use_master:
            control_dir = _control_dir()
            control_dir.mkdir(parents = True, exist_ok = True, mode = 0o700)
            options += [
                "-o", "ControlMaster=auto",
                "-o", f"ControlPath={control_dir}/ctl-%C",
                "-o", "ControlPersist=120",
            ]
        if self.identity_file:
            options += ["-i", str(Path(self.identity_file).expanduser())]
        return options

    def ssh_argv(self, args: Sequence[str]) -> List[str]:
        return [_bin("ssh"), *self.base_options(), self.destination, *args]

    def control_argv(self, args: Sequence[str]) -> List[str]:
        return [_bin("ssh"), *self.base_options(), *args, self.destination]

    def _raise(self, action: str, proc: subprocess.CompletedProcess) -> None:
        detail = _stderr_tail(proc.stderr)
        message = f"{action} failed on {self.destination} (exit {proc.returncode})."
        if "Host key verification failed" in detail or "REMOTE HOST IDENTIFICATION HAS CHANGED" in detail:
            raise RemoteError(
                f"Host key of {self.destination} does not match the pinned fingerprint.",
                hint = "If the machine was legitimately reinstalled, remove and re-add the remote.",
            )
        if "support PTY" in detail:
            raise RemoteError(
                f"{self.destination} is a restricted SSH gateway that cannot run commands.",
                hint = (
                    "Use the machine's direct SSH endpoint instead. On RunPod, use the "
                    "pod's public IP and TCP port mapping ('SSH over exposed TCP' in the "
                    "Connect dialog), not ssh.runpod.io."
                ),
            )
        raise RemoteError(message, hint = detail)

    def run(
        self,
        command: str,
        timeout: int = 120,
        check: bool = True,
        input_text: Optional[str] = None,
    ) -> subprocess.CompletedProcess:
        try:
            proc = subprocess.run(
                self.ssh_argv([command]),
                capture_output = True, text = True, timeout = timeout, input = input_text,
            )
        except subprocess.TimeoutExpired:
            raise RemoteError(
                f"Command timed out after {timeout}s on {self.destination}: {command[:80]}"
            ) from None
        if check and proc.returncode != 0:
            self._raise(f"`{command[:80]}`", proc)
        return proc

    def stream(self, command: str, input_text: Optional[str] = None) -> int:
        # Streams to this terminal; returns the exit code (130 on Ctrl-C).
        # input_text feeds stdin so secrets stay off the command line.
        try:
            if input_text is not None:
                proc = subprocess.run(self.ssh_argv([command]), input = input_text, text = True)
            else:
                proc = subprocess.run(self.ssh_argv([command]))
            return proc.returncode
        except KeyboardInterrupt:
            return 130

    def close_master(self) -> None:
        if self.use_master:
            subprocess.run(
                self.control_argv(["-O", "exit"]), capture_output = True, text = True,
            )


    def _rsync_rsh(self) -> str:
        return shlex.join([_bin("ssh"), *self.base_options()])

    def rsync(
        self,
        source: str,
        dest: str,
        excludes: Sequence[str] = (),
        timeout: int = 3600,
    ) -> None:
        # Remote endpoints are '<dest>:<path>'. Falls back to tar-over-ssh
        # when rsync is missing on either end.
        argv = [_bin("rsync"), "-az", "--partial", "-e", self._rsync_rsh()]
        for pattern in excludes:
            argv += ["--exclude", pattern]
        argv += [source, dest]
        try:
            proc = subprocess.run(argv, capture_output = True, text = True, timeout = timeout)
        except FileNotFoundError:
            self._tar_transfer(source, dest, excludes = excludes, timeout = timeout)
            return
        except subprocess.TimeoutExpired:
            raise RemoteError(f"rsync timed out after {timeout}s.") from None
        if proc.returncode != 0:
            if proc.returncode == 127 or "command not found" in (proc.stderr or ""):
                self._tar_transfer(source, dest, excludes = excludes, timeout = timeout)
                return
            self._raise("rsync", proc)

    def rsync_up(self, local: str, remote_path: str, excludes: Sequence[str] = ()) -> None:
        self.rsync(local, f"{self.destination}:{remote_path}", excludes = excludes)

    def rsync_down(self, remote_path: str, local: str, excludes: Sequence[str] = ()) -> None:
        self.rsync(f"{self.destination}:{remote_path}", local, excludes = excludes)

    def _tar_transfer(
        self,
        source: str,
        dest: str,
        excludes: Sequence[str] = (),
        timeout: int = 3600,
    ) -> None:
        remote_prefix = f"{self.destination}:"
        if dest.startswith(remote_prefix):
            local = Path(source).expanduser()
            remote_dir = dest[len(remote_prefix):].rstrip("/")
            exclude_flags = " ".join(f"--exclude={shlex.quote(p)}" for p in excludes)
            pack = subprocess.Popen(
                [_bin("tar"), "-C", str(local.parent), "-cf", "-", local.name],
                stdout = subprocess.PIPE,
            )
            unpack_cmd = (
                f"mkdir -p {_remote_shell_path(remote_dir)} && "
                f"tar -C {_remote_shell_path(remote_dir)} {exclude_flags} -xf -"
            )
            proc = subprocess.run(
                self.ssh_argv([unpack_cmd]),
                stdin = pack.stdout, capture_output = True, text = True, timeout = timeout,
            )
            pack.stdout.close()
            pack_code = pack.wait()
            if pack_code != 0:
                raise RemoteError(f"tar failed reading {local} (exit {pack_code}).")
            if proc.returncode != 0:
                self._raise("tar-over-ssh upload", proc)
            return

        if source.startswith(remote_prefix):
            remote_dir = source[len(remote_prefix):].rstrip("/")
            local = Path(dest).expanduser()
            local.mkdir(parents = True, exist_ok = True)
            exclude_flags = " ".join(f"--exclude={shlex.quote(p)}" for p in excludes)
            pack_cmd = f"tar -C {_remote_shell_path(remote_dir)} {exclude_flags} -cf - ."
            pack = subprocess.Popen(
                self.ssh_argv([pack_cmd]),
                stdout = subprocess.PIPE, stderr = subprocess.PIPE,
            )
            proc = subprocess.run(
                [_bin("tar"), "-C", str(local), "-xf", "-"],
                stdin = pack.stdout, capture_output = True, text = True, timeout = timeout,
            )
            pack.stdout.close()
            pack_stderr = pack.stderr.read().decode("utf-8", errors = "replace")
            pack.stderr.close()
            pack_code = pack.wait()
            if pack_code != 0:
                raise RemoteError(
                    f"tar-over-ssh download failed on {self.destination} (exit {pack_code}).",
                    hint = _stderr_tail(pack_stderr),
                )
            if proc.returncode != 0:
                raise RemoteError(
                    f"tar failed extracting into {local} (exit {proc.returncode}).",
                    hint = _stderr_tail(proc.stderr),
                )
            return

        raise RemoteError("tar fallback needs one remote endpoint ('<dest>:<path>').")


    def open_tunnel(self, remote_port: int) -> "Tunnel":
        local_port = free_local_port()
        forward = f"{local_port}:127.0.0.1:{remote_port}"
        if self.use_master:
            proc = subprocess.run(
                self.control_argv(["-O", "forward", "-L", forward]),
                capture_output = True, text = True,
            )
            if proc.returncode != 0:
                proc = subprocess.run(
                    self.control_argv(["-fN", "-L", forward]),
                    capture_output = True, text = True,
                )
                if proc.returncode != 0:
                    self._raise("Opening SSH tunnel", proc)
            return Tunnel(self, local_port, forward, child = None)
        child = subprocess.Popen(
            self.control_argv(["-N", "-L", forward]),
            stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL,
        )
        return Tunnel(self, local_port, forward, child = child)


class Tunnel:

    def __init__(self, runner: SSHRunner, local_port: int, forward: str, child):
        self.runner = runner
        self.local_port = local_port
        self.forward = forward
        self.child = child

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.local_port}"

    def close(self) -> None:
        if self.child is not None:
            self.child.terminate()
            return
        subprocess.run(
            self.runner.control_argv(["-O", "cancel", "-L", self.forward]),
            capture_output = True, text = True,
        )

    def __enter__(self) -> "Tunnel":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


_DEFAULT_TIMEOUT = 20
_SSE_READ_TIMEOUT = 90
_SSE_MAX_RECONNECTS = 30
_SSE_RECONNECT_DELAY = 3.0


class RemoteBusyError(RemoteError):

    def __init__(self, message: str, active_job_id: str = ""):
        super().__init__(
            message,
            hint = (
                f"Watch it with: unsloth logs {active_job_id} -f" if active_job_id
                else "Check it with: unsloth remote status <name>"
            ),
        )
        self.active_job_id = active_job_id


@dataclass
class SSEEvent:
    event: str
    data: dict = field(default_factory = dict)
    event_id: Optional[str] = None


class AgentClient:
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def _headers(self, streaming: bool = False, bearer: Optional[str] = None) -> dict:
        headers = {"Content-Type": "application/json"}
        token = bearer or self.api_key
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if streaming:
            headers["Accept"] = "text/event-stream"
        return headers

    def _request(
        self,
        method: str,
        path: str,
        body: Optional[dict] = None,
        timeout: int = _DEFAULT_TIMEOUT,
        bearer: Optional[str] = None,
    ) -> dict:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data = data,
            headers = self._headers(bearer = bearer),
            method = method,
        )
        try:
            with urllib.request.urlopen(request, timeout = timeout) as response:
                text = response.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors = "replace")
            try:
                detail = json.loads(detail).get("detail", detail)
            except (json.JSONDecodeError, AttributeError):
                pass
            if e.code in (401, 403):
                raise RemoteError(
                    f"The agent rejected our credentials: {detail}",
                    hint = "Re-run `unsloth remote add` to refresh them, or check remotes.toml.",
                ) from None
            raise RemoteError(f"Agent returned HTTP {e.code} for {path}: {detail}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            raise RemoteError(
                f"Could not reach the agent at {self.base_url}{path}: {e}",
                hint = "Is the remote up? Check with: unsloth remote doctor <name>",
            ) from None
        return json.loads(text) if text else {}


    def health(self) -> dict:
        return self._request("GET", "/api/health")

    def hardware(self) -> dict:
        return self._request("GET", "/api/train/hardware")


    def login(self, username: str, password: str) -> dict:
        return self._request(
            "POST", "/api/auth/login",
            body = {"username": username, "password": password},
        )

    def change_password(
        self, current_password: str, new_password: str, bearer: str,
    ) -> dict:
        return self._request(
            "POST", "/api/auth/change-password",
            body = {"current_password": current_password, "new_password": new_password},
            bearer = bearer,
        )

    def mint_api_key(self, name: str, bearer: str) -> str:
        response = self._request(
            "POST", "/api/auth/api-keys", body = {"name": name}, bearer = bearer,
        )
        key = response.get("key")
        if not key:
            raise RemoteError("The agent did not return an API key.")
        return key


    def train_status(self) -> dict:
        return self._request("GET", "/api/train/status")

    def start_training(self, payload: dict) -> dict:
        try:
            response = self._request("POST", "/api/train/start", body = payload, timeout = 60)
        except RemoteError as e:
            if "HTTP 409" in str(e):
                raise RemoteBusyError(str(e)) from None
            raise
        # A busy trainer is reported as a 200 with a body-level error.
        if response.get("status") == "error":
            if "already" in (response.get("error") or "").lower() + (response.get("message") or "").lower():
                raise RemoteBusyError(
                    response.get("message") or "Training is already in progress.",
                    active_job_id = response.get("job_id") or "",
                )
            raise RemoteError(response.get("message") or response.get("error") or "Training failed to start.")
        return response

    def stop_training(self, save: bool = True) -> dict:
        return self._request("POST", "/api/train/stop", body = {"save": save}, timeout = 120)

    def list_runs(self, limit: int = 50, offset: int = 0) -> dict:
        return self._request("GET", f"/api/train/runs?offset={offset}&limit={limit}")

    def run_detail(self, run_id: str) -> dict:
        return self._request("GET", f"/api/train/runs/{run_id}")


    def export_load_checkpoint(self, checkpoint_path: str) -> dict:
        return self._request(
            "POST", "/api/export/load-checkpoint",
            body = {"checkpoint_path": checkpoint_path}, timeout = 600,
        )

    def export_gguf(self, save_directory: str, quantization: str) -> dict:
        return self._request(
            "POST", "/api/export/gguf",
            body = {"save_directory": save_directory, "quantization": quantization},
            timeout = 60,
        )

    def export_status(self) -> dict:
        return self._request("GET", "/api/export/status")


    def stream_progress(
        self,
        last_event_id: Optional[str] = None,
        max_reconnects: int = _SSE_MAX_RECONNECTS,
    ) -> Iterator[SSEEvent]:
        # Reconnects with Last-Event-ID; ends on a complete/error event or
        # when reconnects are exhausted.
        reconnects = 0
        while True:
            headers = self._headers(streaming = True)
            if last_event_id is not None:
                headers["Last-Event-ID"] = str(last_event_id)
            request = urllib.request.Request(
                f"{self.base_url}/api/train/progress", headers = headers,
            )
            try:
                with urllib.request.urlopen(request, timeout = _SSE_READ_TIMEOUT) as response:
                    reconnects = 0
                    for event in _parse_sse(response):
                        if event.event_id is not None:
                            last_event_id = event.event_id
                        yield event
                        if event.event in ("complete", "error"):
                            return
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
                pass
            reconnects += 1
            if reconnects > max_reconnects:
                raise RemoteError(
                    "Lost the progress stream and could not reconnect.",
                    hint = "The job keeps running on the remote. Reattach with: unsloth logs <job> -f",
                )
            time.sleep(_SSE_RECONNECT_DELAY)


def _parse_sse(response) -> Iterator[SSEEvent]:
    event_type = "message"
    event_id: Optional[str] = None
    data_lines: list = []
    for raw in response:
        line = raw.decode("utf-8", errors = "replace").rstrip("\r\n")
        if line == "":
            if data_lines:
                text = "\n".join(data_lines)
                try:
                    data = json.loads(text)
                except json.JSONDecodeError:
                    data = {"raw": text}
                yield SSEEvent(event = event_type, data = data, event_id = event_id)
            event_type = "message"
            data_lines = []
            continue
        if line.startswith(":"):
            continue
        if ":" in line:
            fieldname, _, value = line.partition(":")
            value = value.lstrip(" ")
        else:
            fieldname, value = line, ""
        if fieldname == "event":
            event_type = value
        elif fieldname == "data":
            data_lines.append(value)
        elif fieldname == "id":
            event_id = value
