# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for `unsloth jobs/logs/cancel/resume/pull` — mocked transport."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import unsloth_cli.commands.remote as jobsmod
from unsloth_cli import app
from unsloth_cli.remote import RemoteError
from unsloth_cli.remote.state import JobRecord, load_job, save_job
from unsloth_cli.remote.state import RemoteCapabilities, RemoteRecord

runner = CliRunner()

JOB = "job_20260707_120000_deadbeef"


@pytest.fixture(autouse = True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_HOME", str(tmp_path / "unsloth-home"))
    return tmp_path


class FakeTunnel:
    url = "http://127.0.0.1:1"

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def close(self):
        pass


class FakeClient:
    phase = "idle"
    current_job = ""
    can_resume = True
    run_status = "stopped"

    def __init__(self):
        self.stop_calls = []
        self.start_payloads = []
        FakeClient.last = self

    def train_status(self):
        return {
            "job_id": FakeClient.current_job,
            "phase": FakeClient.phase,
            "details": {"step": 12, "total_steps": 30},
        }

    def list_runs(self, limit = 200, offset = 0):
        return {"runs": [{
            "id": JOB, "status": FakeClient.run_status,
            "model_name": "unsloth/Qwen3-0.6B",
            "output_dir": "/remote/outputs/run1",
            "final_step": 12, "total_steps": 30, "final_loss": 1.5,
            "can_resume": FakeClient.can_resume,
        }], "total": 1}

    def run_detail(self, run_id):
        return {"run": self.list_runs()["runs"][0], "config": {}, "metrics": {}}

    def stream_progress(self, last_event_id = None, max_reconnects = None):
        class _Ev:
            event = "complete"
            data = {}
            event_id = None
        yield _Ev()

    def stop_training(self, save = True):
        self.stop_calls.append(save)
        return {}

    def start_training(self, payload):
        self.start_payloads.append(payload)
        return {"job_id": "job_20260707_130000_cafebabe", "status": "queued", "message": "ok"}


class FakeSSH:
    def __init__(self):
        self.streamed = []
        self.downloads = []

    def stream(self, command):
        self.streamed.append(command)
        return 0

    def rsync_down(self, remote_path, local, excludes = ()):
        self.downloads.append((remote_path, local))
        Path(local).mkdir(parents = True, exist_ok = True)
        (Path(local) / "adapter_config.json").write_text("{}", encoding = "utf-8")

    def open_tunnel(self, port):
        return FakeTunnel()


@pytest.fixture()
def wired(monkeypatch):
    FakeClient.phase = "idle"
    FakeClient.current_job = ""
    FakeClient.can_resume = True
    FakeClient.run_status = "stopped"
    fake_ssh = FakeSSH()
    record = RemoteRecord(
        name = "gpu1", host = "1.2.3.4", user = "ubuntu",
        api_key = "sk-x", capabilities = RemoteCapabilities(),
    )
    monkeypatch.setattr(jobsmod, "get_remote", lambda name: record)
    monkeypatch.setattr(jobsmod, "_connect", lambda r: (fake_ssh, FakeTunnel(), FakeClient()))
    monkeypatch.setattr(jobsmod.SSHRunner, "for_remote", classmethod(lambda cls, r: fake_ssh))
    # run_remote's pull uses its own SSHRunner.for_remote
    import unsloth_cli.remote.run_remote as runmod
    monkeypatch.setattr(runmod.SSHRunner, "for_remote", classmethod(lambda cls, r: fake_ssh), raising = False)
    monkeypatch.setattr(runmod, "get_remote", lambda name: record, raising = False)
    return fake_ssh


def _seed_job(**overrides):
    fields = dict(
        job_id = JOB, remote = "gpu1", status = "running",
        model_name = "unsloth/Qwen3-0.6B",
        payload = {"model_name": "unsloth/Qwen3-0.6B", "format_type": "auto"},
        submitted_at = "2026-07-07T12:00:00+00:00",
    )
    fields.update(overrides)
    save_job(JobRecord(**fields))


class TestJobs:
    def test_lists_and_refreshes_from_remote(self, wired, monkeypatch):
        monkeypatch.setenv("COLUMNS", "200")
        _seed_job()
        result = runner.invoke(app, ["jobs"])
        assert result.exit_code == 0, result.output
        assert JOB in result.output
        assert "stopped" in result.output  # refreshed from FakeClient.run_status
        assert load_job(JOB).final_loss == 1.5  # mirror updated

    def test_unreachable_remote_shows_cached(self, wired, monkeypatch):
        _seed_job()

        def broken_connect(record):
            raise RemoteError("down")

        monkeypatch.setattr(jobsmod, "_connect", broken_connect)
        result = runner.invoke(app, ["jobs"])
        assert result.exit_code == 0, result.output
        assert "(cached)" in result.output

    def test_empty(self, wired):
        result = runner.invoke(app, ["jobs"])
        assert "No remote jobs yet" in result.output


class TestLogs:
    def test_raw_streams_tail(self, wired):
        _seed_job()
        result = runner.invoke(app, ["logs", JOB, "--raw"])
        assert result.exit_code == 0, result.output
        assert wired.streamed and "tail -n 200" in wired.streamed[0]

    def test_raw_follow_uses_capital_f(self, wired):
        _seed_job()
        runner.invoke(app, ["logs", JOB, "--raw", "-f"])
        assert "-F" in wired.streamed[0]

    def test_finished_job_prints_summary(self, wired):
        _seed_job()
        result = runner.invoke(app, ["logs", JOB])
        assert result.exit_code == 0, result.output
        assert "stopped" in result.output

    def test_prefix_resolution(self, wired):
        _seed_job()
        result = runner.invoke(app, ["logs", "job_20260707_12"])
        assert result.exit_code == 0, result.output

    def test_unknown_job(self, wired):
        result = runner.invoke(app, ["logs", "job_1999"])
        assert result.exit_code == 1


class TestCancel:
    def test_cancels_active_job(self, wired):
        _seed_job()
        FakeClient.current_job = JOB
        FakeClient.phase = "training"
        result = runner.invoke(app, ["cancel", JOB])
        assert result.exit_code == 0, result.output
        assert FakeClient.last.stop_calls == [True]
        assert "resume" in result.output

    def test_refuses_non_active_job(self, wired):
        _seed_job()
        FakeClient.current_job = "job_other"
        FakeClient.phase = "training"
        result = runner.invoke(app, ["cancel", JOB])
        assert result.exit_code == 1
        assert "not the active job" in result.output


class TestResume:
    def test_resubmits_with_checkpoint_and_env_tokens(self, wired, monkeypatch):
        _seed_job(status = "stopped", output_dir = "/remote/outputs/run1")
        monkeypatch.setenv("HF_TOKEN", "hf_fresh")
        result = runner.invoke(app, ["resume", JOB])
        assert result.exit_code == 0, result.output
        payload = FakeClient.last.start_payloads[0]
        assert payload["resume_from_checkpoint"] == "/remote/outputs/run1"
        assert payload["hf_token"] == "hf_fresh"
        new_job = load_job("job_20260707_130000_cafebabe")
        assert new_job.resumed_from == JOB
        # Secrets never persist in the new record either.
        assert "hf_token" not in new_job.payload

    def test_refuses_unresumable(self, wired):
        _seed_job(status = "error")
        FakeClient.can_resume = False
        result = runner.invoke(app, ["resume", JOB])
        assert result.exit_code == 1
        assert "cannot be resumed" in result.output

    def test_resume_waits_out_async_stop(self, wired, monkeypatch):
        # Right after `cancel`, the run row briefly still says "running"
        # while the worker checkpoints. resume must poll past that.
        _seed_job(status = "running", output_dir = "/remote/outputs/run1")
        statuses = iter(["running", "running", "stopped"])

        original = FakeClient.run_detail

        def settling_run_detail(self, run_id):
            FakeClient.run_status = next(statuses, "stopped")
            return original(self, run_id)

        monkeypatch.setattr(FakeClient, "run_detail", settling_run_detail)
        monkeypatch.setattr(jobsmod.time, "sleep", lambda s: None)
        result = runner.invoke(app, ["resume", JOB])
        assert result.exit_code == 0, result.output
        assert FakeClient.last.start_payloads


class TestPull:
    def test_pull_downloads_and_registers(self, wired):
        _seed_job(status = "completed", output_dir = "/remote/outputs/run1")
        result = runner.invoke(app, ["pull", JOB])
        assert result.exit_code == 0, result.output
        assert wired.downloads
        job = load_job(JOB)
        assert job.pulled is True
        from unsloth_cli.remote.state import resolve_job_artifact
        assert resolve_job_artifact(JOB) is not None

    def test_pull_is_idempotent(self, wired):
        _seed_job(status = "completed", output_dir = "/remote/outputs/run1")
        runner.invoke(app, ["pull", JOB])
        result = runner.invoke(app, ["pull", JOB])
        assert result.exit_code == 0, result.output
        assert len(wired.downloads) == 1
        assert "Already pulled" in result.output

    def test_pull_force_repulls(self, wired):
        _seed_job(status = "completed", output_dir = "/remote/outputs/run1")
        runner.invoke(app, ["pull", JOB])
        runner.invoke(app, ["pull", JOB, "--force"])
        assert len(wired.downloads) == 2


class TestLatestJobDefault:
    def test_logs_defaults_to_most_recent(self, wired):
        _seed_job()
        _seed_job(
            job_id = "job_20260708_120000_aaaaaaaa",
            submitted_at = "2026-07-08T12:00:00+00:00",
        )
        result = runner.invoke(app, ["logs"])
        assert result.exit_code == 0, result.output
        assert "job_20260708_120000_aaaaaaaa" in result.output

    def test_no_jobs_yet_hint(self, wired):
        result = runner.invoke(app, ["cancel"])
        assert result.exit_code == 1
        assert "No remote jobs yet" in result.output
