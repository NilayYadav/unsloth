# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A/B probe for unslothai/unsloth PR #10217.

Identical on both branches. It asserts only what the PR claims to change: the
`model` field of an audio API request must reach the auto-switch hook, and the
hook must be told the target has to be a speech model. The two "no model"
cases are controls: they must pass on both sides.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import routes.inference as inference_route
from models.inference import AudioSpeechRequest, ChatCompletionRequest


class _Reached(Exception):
    pass


def _request():
    """Enough Request for these routes: the API-monitor row is opt-out via state."""
    return SimpleNamespace(state = SimpleNamespace(skip_api_monitor = True))


def _capture(monkeypatch):
    captured = {}

    async def _fake(model, request, subject, **kwargs):
        captured["model"] = model
        captured.update(kwargs)
        raise _Reached()

    monkeypatch.setattr(inference_route, "_maybe_auto_switch_model", _fake)
    return captured


def _speech(monkeypatch, **fields):
    captured = _capture(monkeypatch)
    body = AudioSpeechRequest(input = "hello sloth", **fields)
    with pytest.raises(_Reached):
        asyncio.run(inference_route.openai_audio_speech(body, _request(), "tester"))
    return captured


def _generate(monkeypatch, **fields):
    captured = _capture(monkeypatch)
    payload = ChatCompletionRequest(messages = [{"role": "user", "content": "say hi"}], **fields)
    with pytest.raises(_Reached):
        asyncio.run(inference_route.generate_audio(payload, _request(), "tester"))
    return captured


def test_A_speech_named_model_reaches_the_switch_hook(monkeypatch):
    got = _speech(monkeypatch, model = "org/B-GGUF")
    print(f"PROBE A /v1/audio/speech model='org/B-GGUF' -> hook saw {got['model']!r}")
    assert got["model"] == "org/B-GGUF"


def test_B_speech_without_a_model_is_reload_only(monkeypatch):
    got = _speech(monkeypatch)
    print(f"PROBE B /v1/audio/speech no model -> hook saw {got['model']!r}")
    assert got["model"] == inference_route._RELOAD_ONLY_MODEL


def test_C_generate_named_model_reaches_the_switch_hook(monkeypatch):
    got = _generate(monkeypatch, model = "org/B-GGUF")
    print(f"PROBE C /audio/generate model='org/B-GGUF' -> hook saw {got['model']!r}")
    assert got["model"] == "org/B-GGUF"


def test_D_generate_without_a_model_is_reload_only(monkeypatch):
    got = _generate(monkeypatch)
    print(f"PROBE D /audio/generate no model -> hook saw {got['model']!r}")
    assert got["model"] == inference_route._RELOAD_ONLY_MODEL


def test_E_a_named_audio_target_must_be_a_speech_model(monkeypatch):
    got = _speech(monkeypatch, model = "org/B-GGUF")
    print(f"PROBE E /v1/audio/speech model='org/B-GGUF' -> require_speech={got.get('require_speech')!r}")
    assert got.get("require_speech") is True
