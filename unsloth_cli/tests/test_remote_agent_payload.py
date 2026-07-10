# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for unsloth_cli.remote.run_remote and .agent — fake HTTP, no server."""

from __future__ import annotations

import io
import json
import sys
import urllib.error
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_BACKEND_DIR = _REPO_ROOT / "studio" / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from unsloth_cli.config import Config
from unsloth_cli.remote import RemoteError
from unsloth_cli.remote.ssh import (
    AgentClient,
    RemoteBusyError,
    SSEEvent,
    _parse_sse,
)
from unsloth_cli.remote.run_remote import (
    build_training_payload,
    remote_dataset_dir,
    training_payload_warnings,
)


def _config(**overrides):
    data = {
        "model": "unsloth/Qwen3-0.6B",
        "data": {"dataset": "yahma/alpaca-cleaned", "format_type": "auto"},
        "training": {"max_steps": 30, "save_steps": 10, "learning_rate": 2e-4},
    }
    for key, value in overrides.items():
        section, _, fieldname = key.partition(".")
        if fieldname:
            data.setdefault(section, {})[fieldname] = value
        else:
            data[section] = value
    return Config(**data)


class TestPayload:
    def test_validates_against_real_training_start_request(self):
        from models.training import TrainingStartRequest

        payload = build_training_payload(
            _config(),
            project_name = "my-run",
            hf_token = "hf_test",
        )
        request = TrainingStartRequest(**payload)
        assert request.model_name == "unsloth/Qwen3-0.6B"
        assert request.training_type == "LoRA/QLoRA"
        assert request.use_lora is True
        assert request.max_steps == 30

    def test_learning_rate_stringified(self):
        payload = build_training_payload(_config())
        assert payload["learning_rate"] == "0.0002"
        assert isinstance(payload["learning_rate"], str)

    def test_target_modules_split(self):
        payload = build_training_payload(_config(**{"lora.target_modules": "q_proj, v_proj"}))
        assert payload["target_modules"] == ["q_proj", "v_proj"]

    def test_full_finetuning_mapping(self):
        payload = build_training_payload(_config(**{"training.training_type": "full"}))
        assert payload["training_type"] == "Full Finetuning"
        assert payload["use_lora"] is False

    def test_zero_max_steps_becomes_absent(self):
        payload = build_training_payload(_config(**{"training.max_steps": 0}))
        assert "max_steps" not in payload

    def test_local_datasets_use_uploaded_paths(self):
        remote_paths = [f"{remote_dataset_dir('up_1')}/data.jsonl"]
        payload = build_training_payload(
            _config(data = {"local_dataset": ["/home/me/data.jsonl"], "format_type": "auto"}),
            local_dataset_remote_paths = remote_paths,
        )
        assert payload["local_datasets"] == ["uploads/remote/up_1/data.jsonl"]
        assert "hf_dataset" not in payload

    def test_missing_model_raises(self):
        with pytest.raises(RemoteError, match = "No model"):
            build_training_payload(Config())

    def test_no_token_key_when_absent(self):
        payload = build_training_payload(_config())
        assert "hf_token" not in payload
        assert "wandb_token" not in payload

    def test_resume_checkpoint_passthrough(self):
        payload = build_training_payload(
            _config(), resume_from_checkpoint = "/remote/outputs/run1",
        )
        assert payload["resume_from_checkpoint"] == "/remote/outputs/run1"

    def test_warns_on_no_save_steps(self):
        warnings = training_payload_warnings(_config(**{"training.save_steps": 0}))
        assert any("cannot be resumed" in w for w in warnings)

    def test_warns_on_custom_output_dir(self):
        warnings = training_payload_warnings(_config(**{"training.output_dir": "my_runs"}))
        assert any("ignored for remote runs" in w for w in warnings)

    def test_no_warnings_for_defaults(self):
        assert training_payload_warnings(_config()) == []


def _sse_bytes(text: str):
    return io.BytesIO(text.encode("utf-8"))


class TestSSEParser:
    def test_parses_events_with_id_and_type(self):
        stream = _sse_bytes(
            "retry: 3000\n\n"
            "id: 5\nevent: progress\ndata: {\"step\": 5, \"loss\": 1.2}\n\n"
            "event: heartbeat\ndata: {}\n\n"
            "id: 10\nevent: complete\ndata: {\"step\": 10}\n\n"
        )
        events = list(_parse_sse(stream))
        assert [e.event for e in events] == ["progress", "heartbeat", "complete"]
        assert events[0].data["loss"] == 1.2
        assert events[0].event_id == "5"
        assert events[2].event_id == "10"

    def test_ignores_comments_and_non_json(self):
        stream = _sse_bytes(": ping\n\nevent: progress\ndata: not json\n\n")
        events = list(_parse_sse(stream))
        assert events[0].data == {"raw": "not json"}


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestAgentClient:
    def _client(self):
        return AgentClient("http://127.0.0.1:9999", api_key = "sk-unsloth-x")

    def _patch_urlopen(self, monkeypatch, handler):
        calls = []

        def fake_urlopen(request, timeout = None):
            calls.append(request)
            return handler(request)

        monkeypatch.setattr("unsloth_cli.remote.ssh.urllib.request.urlopen", fake_urlopen)
        return calls

    def test_bearer_header_sent(self, monkeypatch):
        calls = self._patch_urlopen(
            monkeypatch, lambda req: _FakeResponse(b'{"status": "ok"}')
        )
        self._client().health()
        assert calls[0].get_header("Authorization") == "Bearer sk-unsloth-x"

    def test_busy_body_raises_remote_busy(self, monkeypatch):
        body = json.dumps({
            "job_id": "job_20260706_120000_deadbeef",
            "status": "error",
            "message": "Training is already in progress. Stop current training before starting a new one.",
            "error": "Training already active",
        }).encode()
        self._patch_urlopen(monkeypatch, lambda req: _FakeResponse(body))
        with pytest.raises(RemoteBusyError) as exc:
            self._client().start_training({"model_name": "m"})
        assert exc.value.active_job_id == "job_20260706_120000_deadbeef"

    def test_http_409_raises_remote_busy(self, monkeypatch):
        def handler(req):
            raise urllib.error.HTTPError(
                req.full_url, 409, "Conflict", {}, io.BytesIO(b'{"detail": "busy"}')
            )

        self._patch_urlopen(monkeypatch, handler)
        with pytest.raises(RemoteBusyError):
            self._client().start_training({"model_name": "m"})

    def test_other_body_error_raises_remote_error(self, monkeypatch):
        body = json.dumps({
            "job_id": "", "status": "error", "message": "Bad config", "error": "boom",
        }).encode()
        self._patch_urlopen(monkeypatch, lambda req: _FakeResponse(body))
        with pytest.raises(RemoteError, match = "Bad config"):
            self._client().start_training({"model_name": "m"})

    def test_success_returns_job(self, monkeypatch):
        body = json.dumps({
            "job_id": "job_20260706_130000_cafebabe", "status": "queued", "message": "ok",
        }).encode()
        self._patch_urlopen(monkeypatch, lambda req: _FakeResponse(body))
        response = self._client().start_training({"model_name": "m"})
        assert response["job_id"] == "job_20260706_130000_cafebabe"

    def test_auth_failure_hint(self, monkeypatch):
        def handler(req):
            raise urllib.error.HTTPError(
                req.full_url, 401, "Unauthorized", {},
                io.BytesIO(b'{"detail": "Invalid API key"}'),
            )

        self._patch_urlopen(monkeypatch, handler)
        with pytest.raises(RemoteError, match = "rejected our credentials") as exc:
            self._client().train_status()
        assert "Invalid API key" in str(exc.value)
        assert "remote add" in exc.value.hint

    def test_connection_error_hint(self, monkeypatch):
        def handler(req):
            raise urllib.error.URLError("connection refused")

        self._patch_urlopen(monkeypatch, handler)
        with pytest.raises(RemoteError, match = "Could not reach") as exc:
            self._client().train_status()
        assert "doctor" in exc.value.hint

    def test_stream_reconnects_with_last_event_id(self, monkeypatch):
        first = (
            "retry: 3000\n\n"
            "id: 7\nevent: progress\ndata: {\"step\": 7}\n\n"
        )
        second = "id: 9\nevent: complete\ndata: {\"step\": 9}\n\n"
        responses = [_FakeResponse(first.encode()), _FakeResponse(second.encode())]
        seen_headers = []

        def fake_urlopen(request, timeout = None):
            seen_headers.append(request.headers)
            return responses.pop(0)

        monkeypatch.setattr("unsloth_cli.remote.ssh.urllib.request.urlopen", fake_urlopen)
        monkeypatch.setattr("unsloth_cli.remote.ssh._SSE_RECONNECT_DELAY", 0)
        events = list(self._client().stream_progress())
        assert [e.event for e in events] == ["progress", "complete"]
        # Second connection resumed from the last seen event id.
        assert seen_headers[1].get("Last-event-id") == "7"

    def test_stream_gives_up_after_max_reconnects(self, monkeypatch):
        def fake_urlopen(request, timeout = None):
            raise urllib.error.URLError("down")

        monkeypatch.setattr("unsloth_cli.remote.ssh.urllib.request.urlopen", fake_urlopen)
        monkeypatch.setattr("unsloth_cli.remote.ssh._SSE_RECONNECT_DELAY", 0)
        with pytest.raises(RemoteError, match = "could not reconnect") as exc:
            list(self._client().stream_progress(max_reconnects = 2))
        assert "keeps running" in exc.value.hint
