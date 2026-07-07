# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for `unsloth export --remote` and `unsloth remote exec` — mocked SSH."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from unsloth_cli import app
from unsloth_cli.remote import RemoteError
from unsloth_cli.remote.jobs import JobRecord, save_job
from unsloth_cli.remote.state import RemoteCapabilities, RemoteRecord, upsert_remote

runner = CliRunner()

JOB = "job_20260707_120000_deadbeef"


@pytest.fixture(autouse = True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_HOME", str(tmp_path / "unsloth-home"))
    return tmp_path


class FakeTunnel:
    url = "http://127.0.0.1:1"

    def close(self):
        pass


class FakeSSH:
    stream_exit = 0

    def __init__(self):
        self.streamed = []
        self.downloads = []
        FakeSSH.last = self

    def stream(self, command):
        self.streamed.append(command)
        return FakeSSH.stream_exit

    def rsync_down(self, remote_path, local, excludes = ()):
        self.downloads.append((remote_path, local))
        Path(local).mkdir(parents = True, exist_ok = True)
        (Path(local) / "model-q8_0.gguf").write_text("gguf", encoding = "utf-8")

    def open_tunnel(self, port):
        return FakeTunnel()


class FakeAgent:
    phase = "idle"

    def __init__(self, url, api_key = None):
        pass

    def train_status(self):
        return {"phase": FakeAgent.phase, "job_id": "job_busy"}


@pytest.fixture()
def wired(monkeypatch):
    FakeSSH.stream_exit = 0
    FakeAgent.phase = "idle"
    fake_ssh = FakeSSH()
    upsert_remote(RemoteRecord(
        name = "gpu1", host = "1.2.3.4", user = "ubuntu",
        api_key = "sk-x", capabilities = RemoteCapabilities(),
    ))
    import unsloth_cli.commands.export as exportmod  # ensure module import for patching path
    from unsloth_cli.remote import ssh as sshmod
    monkeypatch.setattr(sshmod.SSHRunner, "for_remote", classmethod(lambda cls, r: fake_ssh))
    monkeypatch.setattr("unsloth_cli.remote.agent.AgentClient", FakeAgent)
    monkeypatch.setattr(
        "unsloth_cli.commands.export.AgentClient", FakeAgent, raising = False,
    )
    return fake_ssh


def _patch_agent(monkeypatch):
    # _remote_export imports AgentClient inside the function from remote.agent.
    import unsloth_cli.remote.agent as agentmod
    monkeypatch.setattr(agentmod, "AgentClient", FakeAgent)


class TestRemoteExport:
    def _seed_job(self):
        save_job(JobRecord(
            job_id = JOB, remote = "gpu1", status = "completed",
            output_dir = "/remote/outputs/run1", model_name = "unsloth/Qwen3-0.6B",
        ))

    def test_job_id_export_runs_cli_and_pulls(self, wired, monkeypatch, tmp_path):
        _patch_agent(monkeypatch)
        self._seed_job()
        out = tmp_path / "exported"
        result = runner.invoke(app, [
            "export", JOB, str(out), "--format", "gguf", "-q", "q8_0", "--remote", "gpu1",
        ])
        assert result.exit_code == 0, result.output
        command = wired.streamed[0]
        assert "unsloth export" in command
        assert "/remote/outputs/run1" in command
        assert "--format gguf" in command
        assert "-q q8_0" in command
        assert wired.downloads
        assert (out / "model-q8_0.gguf").exists()
        from unsloth_cli.remote.registry import get_artifact
        artifact = get_artifact(JOB)
        assert artifact is not None and artifact.gguf

    def test_refuses_while_training(self, wired, monkeypatch, tmp_path):
        _patch_agent(monkeypatch)
        FakeAgent.phase = "training"
        self._seed_job()
        result = runner.invoke(app, [
            "export", JOB, str(tmp_path / "out"), "-f", "gguf", "--remote", "gpu1",
        ])
        assert result.exit_code == 1
        assert "busy training" in result.output
        assert wired.streamed == []

    def test_wrong_remote_for_job(self, wired, monkeypatch, tmp_path):
        _patch_agent(monkeypatch)
        save_job(JobRecord(job_id = JOB, remote = "other", status = "completed",
                           output_dir = "/x"))
        result = runner.invoke(app, [
            "export", JOB, str(tmp_path / "out"), "-f", "gguf", "--remote", "gpu1",
        ])
        assert result.exit_code == 1
        assert "ran on remote 'other'" in result.output

    def test_remote_failure_propagates(self, wired, monkeypatch, tmp_path):
        _patch_agent(monkeypatch)
        FakeSSH.stream_exit = 1
        self._seed_job()
        result = runner.invoke(app, [
            "export", JOB, str(tmp_path / "out"), "-f", "gguf", "--remote", "gpu1",
        ])
        assert result.exit_code == 1
        assert "exit 1" in result.output

    def test_local_export_untouched(self, tmp_path):
        # Without --remote the command must reach the local backend import path.
        result = runner.invoke(app, [
            "export", str(tmp_path / "nope"), str(tmp_path / "out"), "-f", "bogus",
        ])
        # Invalid format is rejected before any backend import — proves the
        # local validation path still runs first.
        assert result.exit_code == 2
        assert "Invalid format" in result.output


class TestRemoteExec:
    def test_exec_streams_unsloth_command(self, wired):
        result = runner.invoke(app, ["remote", "exec", "gpu1", "--", "train", "--help"])
        assert result.exit_code == 0, result.output
        assert wired.streamed == ["~/.unsloth/env/bin/unsloth train --help"]

    def test_exec_strips_leading_unsloth(self, wired):
        runner.invoke(app, ["remote", "exec", "gpu1", "--", "unsloth", "jobs"])
        assert wired.streamed == ["~/.unsloth/env/bin/unsloth jobs"]

    def test_exec_requires_command(self, wired):
        result = runner.invoke(app, ["remote", "exec", "gpu1"])
        assert result.exit_code == 1
        assert "No command given" in result.output

    def test_exec_propagates_exit_code(self, wired):
        FakeSSH.stream_exit = 3
        result = runner.invoke(app, ["remote", "exec", "gpu1", "--", "jobs"])
        assert result.exit_code == 3
