# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Disposable A/B repro for unslothai/unsloth#10220. Identical on both branches."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import routes.inference as inference_route
from core import research_runs

_LOCAL_CTX = 4096
_PROMPT_CHARS = 6000
_MESSAGES = [{"role": "user", "content": "e" * _PROMPT_CHARS}]
_CLOUD = {
    "providerType": "anthropic",
    "connectionId": "conn-1",
    "model": "claude-sonnet-4",
    "maxTokens": 16384,
}
_LOCAL = {"model": "local-gguf", "maxTokens": 16384}


@pytest.fixture
def small_local_model(monkeypatch):
    """A 4096-token GGUF is loaded locally, and nothing else is."""
    llama = SimpleNamespace(is_loaded = True, context_length = _LOCAL_CTX)
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: llama)
    monkeypatch.setattr(research_runs, "_peek_inference_backend", lambda: None)
    assert research_runs._loaded_context_length() == _LOCAL_CTX


def test_cloud_run_is_not_sized_by_the_loaded_local_model(small_local_model):
    resolved = research_runs._resolve_max_tokens(16384, _CLOUD, _MESSAGES)
    print(f"REPRO cloud_max_output_tokens={resolved} requested=16384 local_ctx={_LOCAL_CTX}")
    assert resolved == 16384, (
        f"cloud run asked for 16384 output tokens and got {resolved}: "
        f"sized against the {_LOCAL_CTX}-token local model it does not run on"
    )


def test_local_run_is_still_clamped_to_the_loaded_context(small_local_model):
    resolved = research_runs._resolve_max_tokens(16384, _LOCAL, _MESSAGES)
    print(f"REPRO local_max_output_tokens={resolved} requested=16384 local_ctx={_LOCAL_CTX}")
    assert 0 < resolved < 16384, (
        f"local run must stay inside the {_LOCAL_CTX}-token window, got {resolved}"
    )
