# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for unsloth_cli.remote state/jobs/registry — filesystem only, no SSH."""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from unsloth_cli.remote import RemoteError, looks_like_job_id
from unsloth_cli.remote.jobs import JobRecord, list_jobs, load_job, resolve_job, save_job
from unsloth_cli.remote.registry import (
    ADAPTER_SUBDIR,
    get_artifact,
    job_registry_dir,
    register_artifact,
    resolve_job_artifact,
)
from unsloth_cli.remote.state import (
    RemoteCapabilities,
    RemoteRecord,
    get_remote,
    load_remotes,
    remotes_file,
    remove_remote,
    save_remotes,
    upsert_remote,
)


@pytest.fixture(autouse = True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_HOME", str(tmp_path / "unsloth-home"))
    return tmp_path


def _record(name = "gpu1", **kwargs):
    defaults = dict(
        name = name,
        host = "1.2.3.4",
        user = "ubuntu",
        api_key = "sk-unsloth-test",
        supervision = "systemd",
        capabilities = RemoteCapabilities(
            gpu_names = ["NVIDIA A100-SXM4-80GB"],
            vram_gb = [80.0],
            driver = "550.54",
            has_rsync = True,
        ),
    )
    defaults.update(kwargs)
    return RemoteRecord(**defaults)


class TestRemotesStore:
    def test_round_trip(self):
        upsert_remote(_record())
        loaded = load_remotes()
        assert set(loaded) == {"gpu1"}
        gpu1 = loaded["gpu1"]
        assert gpu1.destination == "ubuntu@1.2.3.4"
        assert gpu1.port == 22
        assert gpu1.api_key == "sk-unsloth-test"
        assert gpu1.capabilities.gpu_names == ["NVIDIA A100-SXM4-80GB"]
        assert gpu1.capabilities.has_rsync is True

    def test_file_is_private(self):
        upsert_remote(_record())
        mode = stat.S_IMODE(os.stat(remotes_file()).st_mode)
        assert mode == 0o600

    def test_none_fields_omitted_from_toml(self):
        upsert_remote(_record(identity_file = None))
        text = remotes_file().read_text(encoding = "utf-8")
        assert "identity_file" not in text

    def test_get_unknown_remote_lists_known(self):
        upsert_remote(_record())
        with pytest.raises(RemoteError) as exc:
            get_remote("nope")
        assert "gpu1" in str(exc.value)

    def test_remove(self):
        upsert_remote(_record())
        upsert_remote(_record(name = "gpu2", host = "5.6.7.8"))
        remove_remote("gpu1")
        assert set(load_remotes()) == {"gpu2"}
        with pytest.raises(RemoteError):
            remove_remote("gpu1")


class TestJobRecords:
    def test_save_strips_secrets(self):
        save_job(JobRecord(
            job_id = "job_20260706_120000_deadbeef",
            remote = "gpu1",
            payload = {"model_name": "m", "hf_token": "hf_secret", "wandb_token": "w"},
        ))
        loaded = load_job("job_20260706_120000_deadbeef")
        assert loaded.payload == {"model_name": "m"}

    def test_resolve_exact_and_prefix(self):
        save_job(JobRecord(job_id = "job_20260706_120000_deadbeef", remote = "gpu1"))
        save_job(JobRecord(job_id = "job_20260707_120000_cafebabe", remote = "gpu1"))
        assert resolve_job("job_20260706_120000_deadbeef").job_id.endswith("deadbeef")
        assert resolve_job("job_20260707").job_id.endswith("cafebabe")

    def test_resolve_ambiguous_prefix(self):
        save_job(JobRecord(job_id = "job_20260706_120000_deadbeef", remote = "gpu1"))
        save_job(JobRecord(job_id = "job_20260706_130000_cafebabe", remote = "gpu1"))
        with pytest.raises(RemoteError, match = "ambiguous"):
            resolve_job("job_20260706")

    def test_resolve_rejects_non_job_identifier(self):
        with pytest.raises(RemoteError, match = "does not look like a job id"):
            resolve_job("unsloth/Qwen3-0.6B")

    def test_resolve_missing(self):
        with pytest.raises(RemoteError, match = "No job matching"):
            resolve_job("job_19990101")

    def test_list_skips_corrupt_files(self):
        save_job(JobRecord(job_id = "job_20260706_120000_deadbeef", remote = "gpu1"))
        bad = load_job("job_20260706_120000_deadbeef")  # ensure dir exists
        assert bad is not None
        from unsloth_cli.remote.jobs import jobs_dir
        (jobs_dir() / "job_corrupt.json").write_text("{not json", encoding = "utf-8")
        assert [r.job_id for r in list_jobs()] == ["job_20260706_120000_deadbeef"]


class TestRegistry:
    JOB = "job_20260706_120000_deadbeef"

    def _pull(self, job_id = None):
        job_id = job_id or self.JOB
        adapter = job_registry_dir(job_id) / ADAPTER_SUBDIR
        adapter.mkdir(parents = True)
        (adapter / "adapter_config.json").write_text("{}", encoding = "utf-8")
        return register_artifact(job_id, base_model = "unsloth/Qwen3-0.6B", remote = "gpu1")

    def test_register_and_resolve(self):
        self._pull()
        path = resolve_job_artifact(self.JOB)
        assert path is not None and path.endswith(f"{self.JOB}/{ADAPTER_SUBDIR}")
        assert get_artifact(self.JOB).base_model == "unsloth/Qwen3-0.6B"

    def test_resolve_by_unique_prefix(self):
        self._pull()
        assert resolve_job_artifact("job_20260706") is not None

    def test_non_job_identifiers_fall_through(self):
        self._pull()
        assert resolve_job_artifact("unsloth/Qwen3-0.6B") is None
        assert resolve_job_artifact("./outputs/checkpoint-100") is None
        assert resolve_job_artifact(None) is None

    def test_unpulled_job_returns_none(self):
        assert resolve_job_artifact(self.JOB) is None

    def test_register_merges_gguf(self):
        self._pull()
        register_artifact(self.JOB, gguf = ["exports/model-q8_0.gguf"])
        record = get_artifact(self.JOB)
        assert record.gguf == ["exports/model-q8_0.gguf"]
        assert record.base_model == "unsloth/Qwen3-0.6B"


def test_looks_like_job_id():
    assert looks_like_job_id("job_20260706_120000_deadbeef")
    assert looks_like_job_id("job_2026")
    assert not looks_like_job_id("unsloth/Qwen3-0.6B")
    assert not looks_like_job_id("")
