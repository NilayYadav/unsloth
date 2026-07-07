# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for `unsloth train --remote` orchestration — fully mocked transport."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml
from rich.console import Console
from typer.testing import CliRunner

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import unsloth_cli.remote.run_remote as runmod
from unsloth_cli.config import Config
from unsloth_cli.remote import RemoteError
from unsloth_cli.remote.agent import RemoteBusyError
from unsloth_cli.remote.jobs import load_job
from unsloth_cli.remote.registry import get_artifact
from unsloth_cli.remote.run_remote import run_remote_training
from unsloth_cli.remote.state import RemoteCapabilities, RemoteRecord

runner = CliRunner()

JOB_ID = "job_20260707_120000_deadbeef"


@pytest.fixture(autouse = True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_HOME", str(tmp_path / "unsloth-home"))
    return tmp_path


class FakeTunnel:
    url = "http://127.0.0.1:1"

    def close(self):
        pass


class FakeSSH:
    def __init__(self):
        self.commands = []
        self.uploads = []
        self.downloads = []

    def run(self, command, **kwargs):
        import subprocess
        self.commands.append(command)
        return subprocess.CompletedProcess([command], 0, "", "")

    def rsync_up(self, local, remote_path, excludes = ()):
        self.uploads.append((local, remote_path))

    def rsync_down(self, remote_path, local, excludes = ()):
        self.downloads.append((remote_path, local, tuple(excludes)))
        Path(local).mkdir(parents = True, exist_ok = True)
        (Path(local) / "adapter_config.json").write_text("{}", encoding = "utf-8")

    def open_tunnel(self, port):
        return FakeTunnel()


class FakeAgentClient:
    busy = False
    run_status = "completed"

    def __init__(self, url, api_key = None):
        self.api_key = api_key
        FakeAgentClient.last = self
        self.stopped = False
        self.started_payloads = []

    def start_training(self, payload):
        if FakeAgentClient.busy:
            raise RemoteBusyError("Training is already in progress.", active_job_id = "job_other")
        self.started_payloads.append(payload)
        return {"job_id": JOB_ID, "status": "queued", "message": "ok"}

    def stop_training(self, save = True):
        self.stopped = True
        return {}

    def run_detail(self, run_id):
        return {
            "run": {
                "id": run_id,
                "status": FakeAgentClient.run_status,
                "model_name": "unsloth/Qwen3-0.6B",
                "output_dir": "/home/ubuntu/.unsloth/studio/outputs/run1",
                "final_step": 30,
                "total_steps": 30,
                "final_loss": 0.81,
                "error_message": "boom" if FakeAgentClient.run_status == "error" else None,
            },
            "config": {},
            "metrics": {},
        }


@pytest.fixture()
def wired(monkeypatch):
    """Wire run_remote's collaborators to fakes; returns the FakeSSH."""
    FakeAgentClient.busy = False
    FakeAgentClient.run_status = "completed"
    fake_ssh = FakeSSH()
    record = RemoteRecord(
        name = "gpu1", host = "1.2.3.4", user = "ubuntu",
        api_key = "sk-unsloth-x", capabilities = RemoteCapabilities(),
    )
    monkeypatch.setattr(runmod, "get_remote", lambda name: record)
    monkeypatch.setattr(
        runmod.SSHRunner, "for_remote", classmethod(lambda cls, r: fake_ssh),
    )
    monkeypatch.setattr(runmod, "AgentClient", FakeAgentClient)
    monkeypatch.setattr(runmod, "attach_progress", lambda client, console, job_id, **kw: ("completed", {}))
    return fake_ssh


def _config(**kwargs):
    data = {
        "model": "unsloth/Qwen3-0.6B",
        "data": {"dataset": "yahma/alpaca-cleaned"},
        "training": {"max_steps": 30, "save_steps": 10},
    }
    data.update(kwargs)
    return Config(**data)


class TestRunRemoteTraining:
    def test_detach_submits_and_returns(self, wired):
        job_id, adapter = run_remote_training(
            _config(), "gpu1", Console(), detach = True,
        )
        assert job_id == JOB_ID
        assert adapter is None
        record = load_job(JOB_ID)
        assert record.status == "running"
        assert record.remote == "gpu1"
        assert "hf_token" not in record.payload

    def test_busy_raises_with_active_job(self, wired):
        FakeAgentClient.busy = True
        with pytest.raises(RemoteBusyError):
            run_remote_training(_config(), "gpu1", Console(), detach = True)

    def test_keyboard_interrupt_detaches_without_stopping(self, wired, monkeypatch):
        def interrupt(*args, **kwargs):
            raise KeyboardInterrupt

        monkeypatch.setattr(runmod, "attach_progress", interrupt)
        job_id, adapter = run_remote_training(_config(), "gpu1", Console())
        assert job_id == JOB_ID
        assert adapter is None
        assert FakeAgentClient.last.stopped is False
        # Job record stays live for later `unsloth logs/jobs`.
        assert load_job(JOB_ID).status == "running"

    def test_completed_run_auto_pulls_and_registers(self, wired):
        job_id, adapter = run_remote_training(_config(), "gpu1", Console())
        assert adapter is not None and adapter.exists()
        remote_path, local, excludes = wired.downloads[0]
        assert remote_path.endswith("/outputs/run1/")
        assert "checkpoint-*" in excludes
        record = load_job(job_id)
        assert record.status == "completed"
        assert record.final_loss == 0.81
        assert record.pulled is True
        artifact = get_artifact(job_id)
        assert artifact.base_model == "unsloth/Qwen3-0.6B"
        assert artifact.remote == "gpu1"

    def test_error_run_raises_with_log_hint(self, wired, monkeypatch):
        FakeAgentClient.run_status = "error"
        monkeypatch.setattr(runmod, "attach_progress", lambda *a, **k: ("error", {}))
        with pytest.raises(RemoteError, match = "boom") as exc:
            run_remote_training(_config(), "gpu1", Console())
        assert "unsloth logs" in exc.value.hint
        assert load_job(JOB_ID).status == "error"

    def test_local_datasets_uploaded_and_rewritten(self, wired, tmp_path):
        data_file = tmp_path / "data.jsonl"
        data_file.write_text('{"text": "hi"}\n', encoding = "utf-8")
        cfg = _config(data = {"local_dataset": [str(data_file)]})
        run_remote_training(cfg, "gpu1", Console(), detach = True)
        assert wired.uploads and wired.uploads[0][0] == str(data_file)
        assert any("mkdir -p" in c for c in wired.commands)
        payload = load_job(JOB_ID).payload
        assert len(payload["local_datasets"]) == 1
        assert payload["local_datasets"][0].startswith("uploads/remote/up_")
        assert payload["local_datasets"][0].endswith("/data.jsonl")

    def test_missing_local_dataset_fails_before_submit(self, wired):
        cfg = _config(data = {"local_dataset": ["/nope/missing.jsonl"]})
        with pytest.raises(RemoteError, match = "not found"):
            run_remote_training(cfg, "gpu1", Console(), detach = True)
        assert load_job(JOB_ID) is None


class TestTrainCommandRemoteFlag:
    def _write_config(self, tmp_path):
        config_path = tmp_path / "train.yaml"
        config_path.write_text(yaml.dump({
            "model": "unsloth/Qwen3-0.6B",
            "data": {"dataset": "yahma/alpaca-cleaned"},
            "training": {"max_steps": 30, "save_steps": 10},
        }), encoding = "utf-8")
        return config_path

    def test_train_remote_detach_cli(self, wired, tmp_path):
        from unsloth_cli import app

        config_path = self._write_config(tmp_path)
        result = runner.invoke(
            app, ["train", "--config", str(config_path), "--remote", "gpu1", "--detach"],
        )
        assert result.exit_code == 0, result.output
        assert JOB_ID in result.output
        assert load_job(JOB_ID) is not None

    def test_train_remote_busy_cli(self, wired, tmp_path):
        from unsloth_cli import app

        FakeAgentClient.busy = True
        config_path = self._write_config(tmp_path)
        result = runner.invoke(
            app, ["train", "--config", str(config_path), "--remote", "gpu1"],
        )
        assert result.exit_code == 1
        assert "already in progress" in result.output

    def test_detach_without_remote_rejected(self, tmp_path):
        from unsloth_cli import app

        config_path = self._write_config(tmp_path)
        result = runner.invoke(app, ["train", "--config", str(config_path), "--detach"])
        assert result.exit_code == 2
        assert "--detach requires --remote" in result.output
