# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""SSH transport for remotes: command execution, file sync, and port-forward tunnels.

Uses the system ``ssh``/``rsync`` binaries (no paramiko). Connections are
multiplexed through an OpenSSH ControlMaster socket where supported, so probe,
bootstrap, rsync, and the API tunnel all share one authenticated connection.
Host keys are pinned in ~/.unsloth/ssh/known_hosts with strict checking.
"""

from __future__ import annotations

import os
import shlex
import socket
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from unsloth_cli.remote import RemoteError
from unsloth_cli.remote.state import RemoteRecord, known_hosts_file, ssh_dir

_STDERR_TAIL_LINES = 15


def _bin(name: str) -> str:
    """Binary path, overridable via UNSLOTH_REMOTE_<NAME> (test seam)."""
    return os.environ.get(f"UNSLOTH_REMOTE_{name.upper().replace('-', '_')}", name)


def _supports_control_master() -> bool:
    return sys.platform != "win32"


def _stderr_tail(stderr: str) -> str:
    lines = (stderr or "").strip().splitlines()
    return "\n".join(lines[-_STDERR_TAIL_LINES:])


def free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def scan_host_key(host: str, port: int = 22, timeout: int = 10) -> Tuple[str, str]:
    """Fetch the host's public keys; return (fingerprint, known_hosts_text).

    The fingerprint is of the strongest offered key (ed25519 > ecdsa > rsa).
    """
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
    """Append the host's keys to our private known_hosts file."""
    path = known_hosts_file()
    path.parent.mkdir(parents = True, exist_ok = True)
    existing = path.read_text(encoding = "utf-8") if path.exists() else ""
    new_lines = [l for l in known_hosts_text.splitlines() if l and l not in existing]
    if new_lines:
        with open(path, "a", encoding = "utf-8") as f:
            f.write("\n".join(new_lines) + "\n")
    path.chmod(0o600)


class SSHRunner:
    """Runs commands and file transfers against one remote host."""

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
            ssh_dir().mkdir(parents = True, exist_ok = True)
            options += [
                "-o", "ControlMaster=auto",
                "-o", f"ControlPath={ssh_dir()}/ctl-%C",
                "-o", "ControlPersist=120",
            ]
        if self.identity_file:
            options += ["-i", str(Path(self.identity_file).expanduser())]
        return options

    def ssh_argv(self, args: Sequence[str]) -> List[str]:
        return [_bin("ssh"), *self.base_options(), self.destination, *args]

    def control_argv(self, args: Sequence[str]) -> List[str]:
        """argv for ssh control/tunnel operations (flags before the destination)."""
        return [_bin("ssh"), *self.base_options(), *args, self.destination]

    def _raise(self, action: str, proc: subprocess.CompletedProcess) -> None:
        detail = _stderr_tail(proc.stderr)
        message = f"{action} failed on {self.destination} (exit {proc.returncode})."
        if "Host key verification failed" in detail or "REMOTE HOST IDENTIFICATION HAS CHANGED" in detail:
            raise RemoteError(
                f"Host key of {self.destination} does not match the pinned fingerprint.",
                hint = "If the machine was legitimately reinstalled, remove and re-add the remote.",
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

    def stream(self, command: str) -> int:
        """Run a remote command with output flowing straight to this terminal.

        Returns the exit code; Ctrl-C maps to 130 (the stream stops, the
        remote process keeps running unless it dies with the pipe).
        """
        try:
            proc = subprocess.run(self.ssh_argv([command]))
            return proc.returncode
        except KeyboardInterrupt:
            return 130

    def close_master(self) -> None:
        if self.use_master:
            subprocess.run(
                self.control_argv(["-O", "exit"]), capture_output = True, text = True,
            )

    # -- file transfer -------------------------------------------------------

    def _rsync_rsh(self) -> str:
        return shlex.join([_bin("ssh"), *self.base_options()])

    def rsync(
        self,
        source: str,
        dest: str,
        excludes: Sequence[str] = (),
        timeout: int = 3600,
    ) -> None:
        """Transfer with rsync; remote endpoints are '<dest>:<path>' strings."""
        argv = [_bin("rsync"), "-az", "--partial", "-e", self._rsync_rsh()]
        for pattern in excludes:
            argv += ["--exclude", pattern]
        argv += [source, dest]
        try:
            proc = subprocess.run(argv, capture_output = True, text = True, timeout = timeout)
        except subprocess.TimeoutExpired:
            raise RemoteError(f"rsync timed out after {timeout}s.") from None
        if proc.returncode != 0:
            self._raise("rsync", proc)

    def rsync_up(self, local: str, remote_path: str, excludes: Sequence[str] = ()) -> None:
        self.rsync(local, f"{self.destination}:{remote_path}", excludes = excludes)

    def rsync_down(self, remote_path: str, local: str, excludes: Sequence[str] = ()) -> None:
        self.rsync(f"{self.destination}:{remote_path}", local, excludes = excludes)

    # -- tunnels ---------------------------------------------------------------

    def open_tunnel(self, remote_port: int) -> "Tunnel":
        local_port = free_local_port()
        forward = f"{local_port}:127.0.0.1:{remote_port}"
        if self.use_master:
            proc = subprocess.run(
                self.control_argv(["-O", "forward", "-L", forward]),
                capture_output = True, text = True,
            )
            if proc.returncode != 0:
                # No master yet: establish one implicitly with the forward attached.
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
    """A live local port-forward to the remote agent."""

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
