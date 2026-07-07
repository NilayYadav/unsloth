# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for job-id → pulled-adapter resolution in chat/inference."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from unsloth_cli.remote import RemoteError
from unsloth_cli.remote.jobs import JobRecord, save_job
from unsloth_cli.remote.registry import (
    ADAPTER_SUBDIR,
    job_registry_dir,
    register_artifact,
    resolve_model_identifier,
)

JOB = "job_20260707_120000_deadbeef"


@pytest.fixture(autouse = True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_HOME", str(tmp_path / "unsloth-home"))
    return tmp_path


def _pulled_job():
    adapter = job_registry_dir(JOB) / ADAPTER_SUBDIR
    adapter.mkdir(parents = True)
    (adapter / "adapter_config.json").write_text("{}", encoding = "utf-8")
    register_artifact(JOB, base_model = "unsloth/Qwen3-0.6B", remote = "gpu1")
    return str(adapter)


class TestResolveModelIdentifier:
    def test_pulled_job_resolves_to_adapter_path(self):
        adapter = _pulled_job()
        assert resolve_model_identifier(JOB) == adapter

    def test_job_prefix_resolves(self):
        adapter = _pulled_job()
        assert resolve_model_identifier("job_20260707") == adapter

    def test_hf_id_passes_through(self):
        assert resolve_model_identifier("unsloth/Qwen3-0.6B") == "unsloth/Qwen3-0.6B"

    def test_local_path_passes_through(self):
        assert resolve_model_identifier("./outputs/checkpoint-30") == "./outputs/checkpoint-30"

    def test_none_passes_through(self):
        assert resolve_model_identifier(None) is None

    def test_known_unpulled_job_errors_with_pull_hint(self):
        save_job(JobRecord(job_id = JOB, remote = "gpu1", status = "completed"))
        with pytest.raises(RemoteError, match = "has not been pulled") as exc:
            resolve_model_identifier(JOB)
        assert f"unsloth pull {JOB}" in exc.value.hint

    def test_unknown_job_id_passes_through(self):
        # Not in the registry, not in the job cache: let model resolution try
        # (it might legitimately be a directory named job_...).
        assert resolve_model_identifier("job_19990101_000000_aaaaaaaa") == \
            "job_19990101_000000_aaaaaaaa"


class TestChatCommandHook:
    def test_chat_unpulled_job_exits_with_hint(self, monkeypatch):
        from typer.testing import CliRunner
        import unsloth_cli.commands.chat as chatmod

        save_job(JobRecord(job_id = JOB, remote = "gpu1", status = "completed"))
        # Guard: resolution must fail before any model loading is attempted.
        monkeypatch.setattr(
            chatmod, "resolve_model_config",
            lambda *a, **k: pytest.fail("resolve_model_config should not be reached"),
        )
        import typer
        app = typer.Typer()
        app.command()(chatmod.chat)
        result = CliRunner().invoke(app, [JOB])
        assert result.exit_code == 1
        assert "unsloth pull" in result.output
