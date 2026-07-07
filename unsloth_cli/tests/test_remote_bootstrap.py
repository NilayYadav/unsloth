# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for bootstrap sequencing and the `unsloth remote` command group — mocked SSH."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import unsloth_cli.commands.remote as remotemod
from unsloth_cli.commands.remote import remote_app
from unsloth_cli.remote import RemoteError
from unsloth_cli.remote.bootstrap import run_bootstrap
from unsloth_cli.remote.probe import ProbeCheck, ProbeResult
from unsloth_cli.remote.state import RemoteCapabilities, RemoteRecord, load_remotes, upsert_remote

runner = CliRunner()


@pytest.fixture(autouse = True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_HOME", str(tmp_path / "unsloth-home"))
    return tmp_path


class FakeSSHRunner:
    """Records commands; returns canned stdout by substring match."""

    def __init__(self, responses = None, destination = "ubuntu@1.2.3.4"):
        self.destination = destination
        self.commands = []
        self.responses = responses or {}

    def run(self, command, timeout = 120, check = True, input_text = None):
        self.commands.append(command if input_text is None else f"{command}\n{input_text}")
        stdout = ""
        for needle, value in self.responses.items():
            if needle in command:
                stdout = value
                break
        return subprocess.CompletedProcess([command], 0, stdout, "")

    def open_tunnel(self, remote_port):
        return FakeTunnel()

    def close_master(self):
        pass


class FakeTunnel:
    url = "http://127.0.0.1:1"

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def close(self):
        pass


def _caps(systemd = True):
    return RemoteCapabilities(
        gpu_names = ["NVIDIA A100"], vram_gb = [80.0], driver = "550.54",
        has_rsync = True, has_tmux = True, has_systemd_user = systemd,
    )


class TestBootstrapSequence:
    RESPONSES = {
        "create-api-key": "sk-unsloth-newkey\n",
        "importlib.metadata": "2026.6.1\n",
    }

    def test_systemd_path_order(self):
        ssh = FakeSSHRunner(self.RESPONSES)
        result = run_bootstrap(ssh, _caps(systemd = True), 8877)
        joined = "\n---\n".join(ssh.commands)

        def index_of(needle):
            return next(i for i, c in enumerate(ssh.commands) if needle in c)

        assert index_of("astral.sh/uv/install.sh") < index_of("uv venv")
        assert index_of("uv venv") < index_of("unsloth[huggingface]")
        # NVIDIA machines (probe saw a driver) also get bitsandbytes for 4-bit.
        assert "bitsandbytes" in ssh.commands[index_of("unsloth[huggingface]")]
        assert index_of("unsloth[huggingface]") < index_of("studio.txt")
        assert index_of("studio.txt") < index_of("create-api-key")
        assert index_of("create-api-key") < index_of("systemctl --user daemon-reload")
        assert "unsloth-agent.service" in joined
        assert "enable-linger" in joined
        assert "--api-only --silent --no-cloudflare" in joined
        assert "--port 8877" in joined
        assert "tmux new-session" not in joined
        assert result.api_key == "sk-unsloth-newkey"
        assert result.supervision == "systemd"
        assert result.remote_unsloth_version == "2026.6.1"

    def test_tmux_fallback(self):
        ssh = FakeSSHRunner(self.RESPONSES)
        result = run_bootstrap(ssh, _caps(systemd = False), 9000)
        joined = "\n".join(ssh.commands)
        assert "tmux new-session -d -s unsloth-agent" in joined
        assert "systemctl --user enable" not in joined
        assert "--port 9000" in joined
        assert result.supervision == "tmux"

    def test_skip_install_only_configures_agent(self):
        ssh = FakeSSHRunner(self.RESPONSES)
        run_bootstrap(ssh, _caps(), 8877, skip_install = True)
        joined = "\n".join(ssh.commands)
        assert "uv venv" not in joined
        assert "unsloth[huggingface]" not in joined
        assert "create-api-key" in joined

    def test_empty_api_key_output_fails_with_step_label(self):
        ssh = FakeSSHRunner({"create-api-key": "\n"})
        with pytest.raises(RemoteError, match = r"\[Mint agent API key\]"):
            run_bootstrap(ssh, _caps(), 8877)

    def test_requirements_lookup_survives_a_real_shell(self):
        # Regression: nested single quotes inside the python -c payload broke
        # out of the shell quoting and produced `-r ""` on real machines.
        import re
        import subprocess

        ssh = FakeSSHRunner(self.RESPONSES)
        run_bootstrap(ssh, _caps(), 8877)
        command = next(c for c in ssh.commands if "studio.txt" in c)
        substitution = re.search(r"\$\((.*)\)", command).group(1)
        substitution = substitution.replace("~/.unsloth/env/bin/python", sys.executable)
        result = subprocess.run(
            ["bash", "-c", substitution],
            capture_output = True, text = True, cwd = str(_REPO_ROOT),
        )
        assert result.returncode == 0, result.stderr
        expected = str(_REPO_ROOT / "studio" / "backend" / "requirements" / "studio.txt")
        assert result.stdout.strip() == expected


def _healthy_probe():
    return ProbeResult(
        checks = [ProbeCheck(name = "GPU", ok = True, value = "NVIDIA A100 (80 GB)")],
        capabilities = _caps(),
    )


class FakeAgentClient:
    def __init__(self, url, api_key = None):
        self.api_key = api_key

    def health(self):
        return {"status": "ok"}

    def train_status(self):
        return {"phase": "idle", "job_id": ""}

    def hardware(self):
        return {"gpus": [{"name": "NVIDIA A100"}]}


@pytest.fixture()
def stubbed_add(monkeypatch):
    fake_ssh = FakeSSHRunner({"create-api-key": "sk-unsloth-added\n"})
    monkeypatch.setattr(remotemod, "scan_host_key", lambda host, port = 22: ("SHA256:fp", "1.2.3.4 ssh-ed25519 KEY\n"))
    monkeypatch.setattr(remotemod, "pin_host_key", lambda text: None)
    monkeypatch.setattr(remotemod, "SSHRunner", lambda *a, **k: fake_ssh)
    monkeypatch.setattr(remotemod.SSHRunner, "for_remote", classmethod(lambda cls, r: fake_ssh), raising = False)
    monkeypatch.setattr(remotemod, "run_probe", lambda r: _healthy_probe())
    monkeypatch.setattr(remotemod, "AgentClient", FakeAgentClient)
    return fake_ssh


class TestRemoteAddCommand:
    def test_add_saves_record(self, stubbed_add):
        result = runner.invoke(
            remote_app,
            ["add", "gpu1", "ubuntu@1.2.3.4", "--yes"],
        )
        assert result.exit_code == 0, result.output
        remotes = load_remotes()
        assert "gpu1" in remotes
        record = remotes["gpu1"]
        assert record.host == "1.2.3.4"
        assert record.user == "ubuntu"
        assert record.api_key == "sk-unsloth-added"
        assert record.host_key_fingerprint == "SHA256:fp"
        assert record.supervision == "systemd"
        assert "ready" in result.output

    def test_add_rejects_bad_destination(self, stubbed_add):
        result = runner.invoke(remote_app, ["add", "gpu1", "nouserhost", "--yes"])
        assert result.exit_code == 1
        assert "user@host" in result.output

    def test_add_aborts_on_fatal_probe(self, stubbed_add, monkeypatch):
        fatal = ProbeResult(
            checks = [ProbeCheck(name = "GPU", ok = False, value = "none", hint = "install driver", fatal = True)],
            capabilities = _caps(),
        )
        monkeypatch.setattr(remotemod, "run_probe", lambda r: fatal)
        result = runner.invoke(remote_app, ["add", "gpu1", "ubuntu@1.2.3.4", "--yes"])
        assert result.exit_code == 1
        assert "not ready" in result.output
        assert "gpu1" not in load_remotes()


class TestRemoteListRm:
    def _seed(self):
        upsert_remote(RemoteRecord(
            name = "gpu1", host = "1.2.3.4", user = "ubuntu",
            capabilities = _caps(), supervision = "systemd",
        ))

    def test_list_offline(self):
        self._seed()
        result = runner.invoke(remote_app, ["list", "--offline"])
        assert result.exit_code == 0, result.output
        assert "gpu1" in result.output
        assert "ubuntu@1.2.3.4" in result.output

    def test_list_empty(self):
        result = runner.invoke(remote_app, ["list", "--offline"])
        assert "No remotes configured" in result.output

    def test_rm(self):
        self._seed()
        result = runner.invoke(remote_app, ["rm", "gpu1", "--yes"])
        assert result.exit_code == 0, result.output
        assert load_remotes() == {}

    def test_rm_unknown(self):
        result = runner.invoke(remote_app, ["rm", "nope", "--yes"])
        assert result.exit_code == 1
