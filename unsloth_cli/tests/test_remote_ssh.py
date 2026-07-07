# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for unsloth_cli.remote.ssh — stubbed binaries, no real SSH."""

from __future__ import annotations

import stat
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from unsloth_cli.remote import RemoteError
from unsloth_cli.remote.ssh import SSHRunner, free_local_port, pin_host_key, scan_host_key
from unsloth_cli.remote.state import known_hosts_file


@pytest.fixture(autouse = True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_HOME", str(tmp_path / "unsloth-home"))
    return tmp_path


def _stub_binary(tmp_path, name, script):
    path = tmp_path / f"stub-{name}"
    path.write_text("#!/bin/sh\n" + script, encoding = "utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


class TestArgv:
    def test_ssh_argv_shape(self):
        runner = SSHRunner("ubuntu@1.2.3.4", port = 2222, identity_file = "~/.ssh/id_ed25519")
        argv = runner.ssh_argv(["echo hi"])
        joined = " ".join(argv)
        assert argv[0] == "ssh"
        assert "BatchMode=yes" in joined
        assert "StrictHostKeyChecking=yes" in joined
        assert str(known_hosts_file()) in joined
        assert ["-p", "2222"] == argv[argv.index("-p"):argv.index("-p") + 2]
        # destination comes before the remote command
        assert argv.index("ubuntu@1.2.3.4") == len(argv) - 2
        assert argv[-1] == "echo hi"

    def test_control_argv_puts_destination_last(self):
        runner = SSHRunner("ubuntu@1.2.3.4")
        argv = runner.control_argv(["-O", "forward", "-L", "1:127.0.0.1:2"])
        assert argv[-1] == "ubuntu@1.2.3.4"
        assert "-O" in argv and argv.index("-O") < argv.index("ubuntu@1.2.3.4")

    @pytest.mark.skipif(sys.platform == "win32", reason = "ControlMaster is POSIX-only")
    def test_control_master_options_present(self):
        argv = SSHRunner("u@h").ssh_argv(["true"])
        assert any(a.startswith("ControlPath=") for a in argv)
        assert "ControlMaster=auto" in argv

    def test_rsync_rsh_quotes_options(self):
        runner = SSHRunner("u@h", identity_file = None)
        rsh = runner._rsync_rsh()
        assert rsh.startswith("ssh ")
        assert "StrictHostKeyChecking=yes" in rsh


class TestRun:
    def test_run_success(self, tmp_path, monkeypatch):
        stub = _stub_binary(tmp_path, "ssh", 'echo "stdout from remote"\nexit 0\n')
        monkeypatch.setenv("UNSLOTH_REMOTE_SSH", stub)
        proc = SSHRunner("u@h").run("echo hi")
        assert proc.stdout.strip() == "stdout from remote"

    def test_run_failure_maps_to_remote_error_with_stderr(self, tmp_path, monkeypatch):
        stub = _stub_binary(tmp_path, "ssh", 'echo "nvidia-smi: command not found" >&2\nexit 127\n')
        monkeypatch.setenv("UNSLOTH_REMOTE_SSH", stub)
        with pytest.raises(RemoteError) as exc:
            SSHRunner("u@h").run("nvidia-smi")
        assert "exit 127" in str(exc.value)
        assert "command not found" in exc.value.hint

    def test_run_no_check_returns_process(self, tmp_path, monkeypatch):
        stub = _stub_binary(tmp_path, "ssh", "exit 3\n")
        monkeypatch.setenv("UNSLOTH_REMOTE_SSH", stub)
        proc = SSHRunner("u@h").run("false", check = False)
        assert proc.returncode == 3

    def test_host_key_mismatch_message(self, tmp_path, monkeypatch):
        stub = _stub_binary(
            tmp_path, "ssh",
            'echo "Host key verification failed." >&2\nexit 255\n',
        )
        monkeypatch.setenv("UNSLOTH_REMOTE_SSH", stub)
        with pytest.raises(RemoteError, match = "pinned fingerprint"):
            SSHRunner("u@h").run("true")


class TestRsync:
    def test_rsync_up_argv(self, tmp_path, monkeypatch):
        captured = {}

        def fake_run(argv, **kwargs):
            captured["argv"] = argv
            import subprocess
            return subprocess.CompletedProcess(argv, 0, "", "")

        monkeypatch.setattr("unsloth_cli.remote.ssh.subprocess.run", fake_run)
        runner = SSHRunner("u@h")
        runner.rsync_up("/local/data.jsonl", "/remote/dir/", excludes = ["checkpoint-*"])
        argv = captured["argv"]
        assert argv[0] == "rsync"
        assert "--partial" in argv
        assert ["--exclude", "checkpoint-*"] == argv[argv.index("--exclude"):argv.index("--exclude") + 2]
        assert argv[-2:] == ["/local/data.jsonl", "u@h:/remote/dir/"]

    def test_rsync_down_argv(self, monkeypatch):
        captured = {}

        def fake_run(argv, **kwargs):
            captured["argv"] = argv
            import subprocess
            return subprocess.CompletedProcess(argv, 0, "", "")

        monkeypatch.setattr("unsloth_cli.remote.ssh.subprocess.run", fake_run)
        SSHRunner("u@h").rsync_down("/remote/out/", "/local/dest/")
        assert captured["argv"][-2:] == ["u@h:/remote/out/", "/local/dest/"]


class TestHostKeys:
    ED25519_LINE = "1.2.3.4 ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIBase64Base64Base64Base64Base64Base64Base6"
    RSA_LINE = "1.2.3.4 ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQCBase64Base64Base64Base64Base64Base64"

    def test_scan_prefers_ed25519_and_extracts_fingerprint(self, tmp_path, monkeypatch):
        keyscan = _stub_binary(
            tmp_path, "keyscan",
            f'echo "{self.RSA_LINE}"\necho "{self.ED25519_LINE}"\n',
        )
        keygen = _stub_binary(
            tmp_path, "keygen",
            'cat > /dev/null\necho "256 SHA256:abcdefFINGERPRINT 1.2.3.4 (ED25519)"\n',
        )
        monkeypatch.setenv("UNSLOTH_REMOTE_SSH_KEYSCAN", keyscan)
        monkeypatch.setenv("UNSLOTH_REMOTE_SSH_KEYGEN", keygen)
        fingerprint, known_hosts = scan_host_key("1.2.3.4")
        assert fingerprint == "SHA256:abcdefFINGERPRINT"
        # ed25519 sorted first
        assert known_hosts.splitlines()[0] == self.ED25519_LINE

    def test_scan_unreachable_host_raises(self, tmp_path, monkeypatch):
        keyscan = _stub_binary(tmp_path, "keyscan", 'echo "connection refused" >&2\nexit 1\n')
        monkeypatch.setenv("UNSLOTH_REMOTE_SSH_KEYSCAN", keyscan)
        with pytest.raises(RemoteError, match = "host key"):
            scan_host_key("10.0.0.1")

    def test_pin_host_key_dedupes_and_is_private(self):
        pin_host_key(self.ED25519_LINE + "\n")
        pin_host_key(self.ED25519_LINE + "\n" + self.RSA_LINE + "\n")
        content = known_hosts_file().read_text(encoding = "utf-8")
        assert content.count(self.ED25519_LINE) == 1
        assert content.count(self.RSA_LINE) == 1
        mode = stat.S_IMODE(known_hosts_file().stat().st_mode)
        assert mode == 0o600


def test_free_local_port_is_bindable():
    import socket
    port = free_local_port()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", port))
