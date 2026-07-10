# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for probe parsing, bootstrap sequencing, and `unsloth remote` — mocked SSH."""

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
from unsloth_cli.remote.bootstrap import (
    ProbeCheck,
    ProbeResult,
    parse_probe_output,
    run_bootstrap,
)
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


def _caps(systemd = True, **overrides):
    fields = dict(
        gpu_names = ["NVIDIA A100"], vram_gb = [80.0], driver = "550.54",
        has_rsync = True, has_tmux = True, has_systemd_user = systemd,
        has_curl = True,
    )
    fields.update(overrides)
    return RemoteCapabilities(**fields)


class TestBootstrapSequence:
    RESPONSES = {
        "importlib.metadata": "2026.6.1\n",
    }

    def test_systemd_path_order(self):
        ssh = FakeSSHRunner(self.RESPONSES)
        result = run_bootstrap(ssh, _caps(systemd = True), 8877)
        joined = "\n---\n".join(ssh.commands)

        def index_of(needle):
            return next(i for i, c in enumerate(ssh.commands) if needle in c)

        assert index_of("astral.sh/uv/install.sh") < index_of("uv venv")
        assert index_of("uv venv") < index_of("constraints.txt")
        assert index_of("constraints.txt") < index_of("unsloth[huggingface]")
        assert index_of("unsloth[huggingface]") < index_of("studio.txt")
        assert index_of("studio.txt") < index_of("systemctl --user daemon-reload")
        # NVIDIA machines (probe saw a driver) also get bitsandbytes for 4-bit.
        assert "bitsandbytes" in ssh.commands[index_of("unsloth[huggingface]")]
        # Both installs resolve under the uploaded single-env constraints so
        # they cannot pick conflicting versions (transformers vs studio pins).
        assert "-c ~/.unsloth/constraints.txt" in ssh.commands[index_of("unsloth[huggingface]")]
        assert "-c ~/.unsloth/constraints.txt" in ssh.commands[index_of("studio.txt")]
        assert "unsloth-agent.service" in joined
        assert "enable-linger" in joined
        assert "--api-only --silent --no-cloudflare" in joined
        assert "--port 8877" in joined
        assert "tmux new-session" not in joined
        assert result.supervision == "systemd"
        assert result.remote_unsloth_version == "2026.6.1"

    def test_constraints_step_uploads_single_env_pins(self):
        ssh = FakeSSHRunner(self.RESPONSES)
        run_bootstrap(ssh, _caps(), 8877)
        upload = next(c for c in ssh.commands if "cat > ~/.unsloth/constraints.txt" in c)
        # The uploaded text is the repo's single-env constraints file.
        assert "transformers==" in upload
        assert "huggingface-hub==" in upload

    def test_tmux_fallback(self):
        ssh = FakeSSHRunner(self.RESPONSES)
        result = run_bootstrap(ssh, _caps(systemd = False), 9000)
        joined = "\n".join(ssh.commands)
        assert "tmux new-session -d -s unsloth-agent" in joined
        assert "systemctl --user enable" not in joined
        assert "--port 9000" in joined
        assert result.supervision == "tmux"

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


class TestSystemPackageInstall:
    """Bare boxes: the user gives user@host + key; bootstrap installs the rest."""

    def test_root_container_installs_tmux_and_rsync_first(self):
        # Fresh RunPod/Vast container: root, no tmux/rsync/systemd.
        ssh = FakeSSHRunner(TestBootstrapSequence.RESPONSES)
        result = run_bootstrap(
            ssh,
            _caps(systemd = False, has_tmux = False, has_rsync = False, is_root = True),
            8877,
        )
        install = next(c for c in ssh.commands if "apt-get" in c)
        assert "tmux" in install and "rsync" in install
        assert "DEBIAN_FRONTEND=noninteractive" in install
        assert "sudo" not in install  # already root
        # dnf/yum fallbacks for Amazon Linux / RHEL images.
        assert "dnf install" in install and "yum install" in install
        # Package install precedes everything else.
        assert ssh.commands.index(install) < next(
            i for i, c in enumerate(ssh.commands) if "astral.sh/uv" in c
        )
        assert result.supervision == "tmux"

    def test_sudo_user_gets_noninteractive_sudo_prefix(self):
        # Ubuntu on AWS/GCP/Lambda: unprivileged user with passwordless sudo.
        ssh = FakeSSHRunner(TestBootstrapSequence.RESPONSES)
        run_bootstrap(
            ssh,
            _caps(systemd = True, has_rsync = False, has_passwordless_sudo = True),
            8877,
        )
        install = next(c for c in ssh.commands if "apt-get" in c)
        assert "sudo -n " in install

    def test_nothing_missing_means_no_install_step(self):
        ssh = FakeSSHRunner(TestBootstrapSequence.RESPONSES)
        run_bootstrap(ssh, _caps(is_root = True), 8877)
        assert not any("apt-get" in c for c in ssh.commands)

    def test_unprivileged_box_gets_no_install_step(self):
        # Missing rsync but no way to install it: rely on the tar fallback.
        ssh = FakeSSHRunner(TestBootstrapSequence.RESPONSES)
        run_bootstrap(ssh, _caps(has_rsync = False), 8877)
        assert not any("apt-get" in c for c in ssh.commands)

    def test_uv_install_falls_back_to_wget(self):
        ssh = FakeSSHRunner(TestBootstrapSequence.RESPONSES)
        run_bootstrap(ssh, _caps(), 8877)
        uv_step = next(c for c in ssh.commands if "astral.sh/uv" in c)
        assert "curl -LsSf" in uv_step and "wget -qO-" in uv_step


class FakeAuthClient:
    """Fake AgentClient covering the public first-run auth endpoints."""

    def __init__(self, must_change_password = True):
        self.must_change = must_change_password
        self.logins = []
        self.password_changes = []
        self.minted = []

    def login(self, username, password):
        self.logins.append((username, password))
        return {"access_token": "jwt-login", "must_change_password": self.must_change}

    def change_password(self, current_password, new_password, bearer):
        assert bearer == "jwt-login"
        self.password_changes.append((current_password, new_password))
        return {"access_token": "jwt-fresh", "must_change_password": False}

    def mint_api_key(self, name, bearer):
        self.minted.append((name, bearer))
        return "sk-unsloth-minted"


class TestInitializeAgentAuth:
    def test_first_run_flow(self):
        from unsloth_cli.remote.bootstrap import initialize_agent_auth

        ssh = FakeSSHRunner({".bootstrap_password": "diceware pass phrase\n"})
        client = FakeAuthClient(must_change_password = True)
        key = initialize_agent_auth(ssh, client)
        assert key == "sk-unsloth-minted"
        assert client.logins == [("unsloth", "diceware pass phrase")]
        # The bootstrap password is replaced with a random one, never reused.
        (current, new), = client.password_changes
        assert current == "diceware pass phrase"
        assert new != current and len(new) >= 8
        # The key is minted with the fresh token and a recognizable label.
        (label, bearer), = client.minted
        assert label.startswith("remote-")
        assert bearer == "jwt-fresh"

    def test_already_initialized_password_skips_change(self):
        from unsloth_cli.remote.bootstrap import initialize_agent_auth

        ssh = FakeSSHRunner({".bootstrap_password": "diceware pass phrase\n"})
        client = FakeAuthClient(must_change_password = False)
        key = initialize_agent_auth(ssh, client)
        assert key == "sk-unsloth-minted"
        assert client.password_changes == []
        assert client.minted[0][1] == "jwt-login"

    def test_missing_bootstrap_password_gives_reset_hint(self):
        from unsloth_cli.remote.bootstrap import initialize_agent_auth

        ssh = FakeSSHRunner()  # empty stdout for the cat
        with pytest.raises(RemoteError, match = "already set up") as exc:
            initialize_agent_auth(ssh, FakeAuthClient())
        assert "reset-password" in exc.value.hint


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
    fake_ssh = FakeSSHRunner()
    monkeypatch.setattr(remotemod, "scan_host_key", lambda host, port = 22: ("SHA256:fp", "1.2.3.4 ssh-ed25519 KEY\n"))
    monkeypatch.setattr(remotemod, "pin_host_key", lambda text: None)
    monkeypatch.setattr(remotemod, "SSHRunner", lambda *a, **k: fake_ssh)
    monkeypatch.setattr(remotemod.SSHRunner, "for_remote", classmethod(lambda cls, r: fake_ssh), raising = False)
    monkeypatch.setattr(remotemod, "run_probe", lambda r: _healthy_probe())
    monkeypatch.setattr(remotemod, "AgentClient", FakeAgentClient)
    monkeypatch.setattr(remotemod, "initialize_agent_auth", lambda runner, client: "sk-unsloth-added")
    return fake_ssh


class TestProbeOutput:
    """Healthy probes render as one line; the table appears only on problems."""

    def _render(self, result):
        import io

        from rich.console import Console

        console = Console(file = io.StringIO(), width = 120)
        remotemod._print_probe(console, result)
        return console.file.getvalue()

    def test_healthy_probe_is_one_line(self):
        out = self._render(_healthy_probe())
        assert "Probe ok:" in out
        assert "Remote machine probe" not in out
        assert out.strip().count("\n") == 0

    def test_problems_show_the_full_table_with_hints(self):
        result = ProbeResult(
            checks = [ProbeCheck(
                name = "GPU", ok = False, value = "none", hint = "install driver", fatal = True,
            )],
            capabilities = _caps(),
        )
        out = self._render(result)
        assert "Remote machine probe" in out
        assert "install driver" in out


class TestRemoteAddCommand:
    def test_add_saves_record(self, stubbed_add):
        result = runner.invoke(
            remote_app,
            ["add", "gpu1", "ubuntu@1.2.3.4"],
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
        result = runner.invoke(remote_app, ["add", "gpu1", "nouserhost"])
        assert result.exit_code == 1
        assert "user@host" in result.output

    def test_add_aborts_on_fatal_probe(self, stubbed_add, monkeypatch):
        fatal = ProbeResult(
            checks = [ProbeCheck(name = "GPU", ok = False, value = "none", hint = "install driver", fatal = True)],
            capabilities = _caps(),
        )
        monkeypatch.setattr(remotemod, "run_probe", lambda r: fatal)
        result = runner.invoke(remote_app, ["add", "gpu1", "ubuntu@1.2.3.4"])
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
        result = runner.invoke(remote_app, ["list"])
        assert result.exit_code == 0, result.output
        assert "gpu1" in result.output
        assert "ubuntu@1.2.3.4" in result.output

    def test_list_empty(self):
        result = runner.invoke(remote_app, ["list"])
        assert "No remotes configured" in result.output

    def test_rm(self):
        self._seed()
        result = runner.invoke(remote_app, ["rm", "gpu1"])
        assert result.exit_code == 0, result.output
        assert load_remotes() == {}

    def test_rm_unknown(self):
        result = runner.invoke(remote_app, ["rm", "nope"])
        assert result.exit_code == 1


# -- probe parsing (canned outputs, no SSH) --------------------------------------

def _output(
    gpu = "NVIDIA A100-SXM4-80GB, 81920 MiB, 550.54.15",
    cuda = "12.4",
    python = "3.10.12",
    disk_kb = 400 * 1024 * 1024,
    arch = "x86_64",
    tools = "rsync=yes\ntmux=yes\nuv=no\ncurl=yes",
    priv = "uid=1000\nsudo=no",
    systemd = "running",
    os_name = "Ubuntu 22.04.4 LTS",
):
    gpu_lines = "\n".join(f"gpu={l}" for l in gpu.splitlines() if l)
    return "\n".join(filter(None, [
        gpu_lines,
        f"cuda={cuda}",
        f"python={python}",
        f"disk_kb={disk_kb}",
        f"arch={arch}",
        tools,
        priv,
        f"systemd={systemd}",
        f"os={os_name}",
    ]))


class TestHealthyBox:
    def test_all_green(self):
        result = parse_probe_output(_output())
        assert result.fatal_failures == []
        caps = result.capabilities
        assert caps.gpu_names == ["NVIDIA A100-SXM4-80GB"]
        assert caps.vram_gb == [80.0]
        assert caps.driver == "550.54.15"
        assert caps.cuda == "12.4"
        assert caps.python == "3.10.12"
        assert caps.arch == "x86_64"
        assert caps.disk_free_gb == 400.0
        assert caps.has_rsync and caps.has_tmux and caps.has_systemd_user

    def test_multi_gpu(self):
        gpu = (
            "NVIDIA A100-SXM4-80GB, 81920 MiB, 550.54.15\n"
            "NVIDIA A100-SXM4-80GB, 81920 MiB, 550.54.15"
        )
        caps = parse_probe_output(_output(gpu = gpu)).capabilities
        assert len(caps.gpu_names) == 2
        assert caps.vram_gb == [80.0, 80.0]


class TestFailures:
    def test_no_gpu_is_fatal_with_driver_hint(self):
        result = parse_probe_output(_output(gpu = "", cuda = ""))
        fatal = result.fatal_failures
        assert len(fatal) == 1
        assert fatal[0].name == "GPU"
        assert "nvidia-driver-550" in fatal[0].hint

    def test_old_driver_is_fatal_with_upgrade_hint(self):
        result = parse_probe_output(
            _output(gpu = "NVIDIA T4, 15360 MiB, 470.82.01", cuda = "CUDA Version: 11.4")
        )
        fatal = result.fatal_failures
        assert [c.name for c in fatal] == ["NVIDIA driver"]
        assert "525" in fatal[0].hint
        assert "sudo apt install nvidia-driver-550" in fatal[0].hint

    def test_low_disk_is_warning_not_fatal(self):
        result = parse_probe_output(_output(disk_kb = 10 * 1024 * 1024))
        assert result.fatal_failures == []
        warning = [c for c in result.warnings if c.name.startswith("Disk")]
        assert len(warning) == 1
        assert "free up space" in warning[0].hint.lower()

    def test_missing_python_is_warning_only(self):
        result = parse_probe_output(_output(python = ""))
        assert result.fatal_failures == []
        py = [c for c in result.warnings if c.name == "python3"]
        assert py and "uv" in py[0].hint

    def test_no_supervision_is_fatal_without_privilege(self):
        result = parse_probe_output(_output(
            tools = "rsync=yes\ntmux=no\nuv=no\ncurl=yes",
            systemd = "none",
        ))
        assert [c.name for c in result.fatal_failures] == ["Agent supervision"]
        assert "tmux" in result.fatal_failures[0].hint

    def test_tmux_fallback_when_no_user_systemd(self):
        result = parse_probe_output(_output(systemd = "none"))
        assert result.fatal_failures == []
        supervision = [c for c in result.checks if c.name == "Agent supervision"]
        assert supervision[0].value == "tmux"

    def test_missing_rsync_warns_without_privilege(self):
        result = parse_probe_output(_output(
            tools = "rsync=no\ntmux=yes\nuv=no\ncurl=yes",
        ))
        assert result.capabilities.has_rsync is False
        assert any(c.name == "rsync" for c in result.warnings)


class TestAutoInstall:
    """Root or passwordless sudo: missing tools become bootstrap's job, not the user's."""

    BARE_CONTAINER_TOOLS = "rsync=no\ntmux=no\nuv=no\ncurl=yes\nwget=no"

    def test_root_container_missing_tmux_is_not_fatal(self):
        # A fresh RunPod/Vast container: root, no tmux, no rsync, no systemd.
        result = parse_probe_output(_output(
            tools = self.BARE_CONTAINER_TOOLS, priv = "uid=0\nsudo=no", systemd = "none",
        ))
        assert result.fatal_failures == []
        assert result.warnings == []
        packages = next(c for c in result.checks if c.name == "System packages")
        assert "tmux" in packages.value and "rsync" in packages.value
        supervision = next(c for c in result.checks if c.name == "Agent supervision")
        assert supervision.ok and supervision.value == "tmux (auto-install)"
        assert result.capabilities.is_root

    def test_passwordless_sudo_counts_as_privileged(self):
        result = parse_probe_output(_output(
            tools = self.BARE_CONTAINER_TOOLS, priv = "uid=1000\nsudo=yes", systemd = "none",
        ))
        assert result.fatal_failures == []
        assert result.capabilities.has_passwordless_sudo
        assert result.capabilities.can_install_packages

    def test_unprivileged_bare_container_still_fatal(self):
        result = parse_probe_output(_output(
            tools = self.BARE_CONTAINER_TOOLS, priv = "uid=1000\nsudo=no", systemd = "none",
        ))
        assert [c.name for c in result.fatal_failures] == ["Agent supervision"]

    def test_missing_curl_auto_installs_when_root(self):
        result = parse_probe_output(_output(
            tools = "rsync=yes\ntmux=yes\nuv=no\ncurl=no\nwget=no",
            priv = "uid=0\nsudo=no",
        ))
        assert result.fatal_failures == []
        packages = next(c for c in result.checks if c.name == "System packages")
        assert "curl" in packages.value

    def test_wget_satisfies_the_uv_download(self):
        # No curl but wget present: nothing to install, bootstrap uses wget.
        result = parse_probe_output(_output(
            tools = "rsync=yes\ntmux=yes\nuv=no\ncurl=no\nwget=yes",
        ))
        assert result.fatal_failures == []
        assert not any(c.name in ("curl", "System packages") for c in result.checks)

    def test_probe_without_priv_lines_defaults_unprivileged(self):
        output = _output(tools = self.BARE_CONTAINER_TOOLS, priv = "", systemd = "none")
        result = parse_probe_output(output)
        assert not result.capabilities.can_install_packages
        assert [c.name for c in result.fatal_failures] == ["Agent supervision"]
