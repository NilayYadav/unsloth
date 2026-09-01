# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""PR 10096 reproduction probe: non-streaming /v1/messages and a client that leaves.

Identical on both A/B branches. It drives the real ``anthropic_messages`` route
with the same fake llama backend the admission tests use, so the only thing that
differs between the negative and positive runs is the implementation under test.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
from types import SimpleNamespace

import pytest

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

import routes.inference as inf_mod
from routes.inference import anthropic_messages
from models.inference import AnthropicMessagesRequest
from core.inference.api_monitor import ApiMonitor
from core.inference.llama_admission import reset_llama_admission_queues

TOTAL_TOKENS = 400
STEP_S = 0.005


class _Request:
    """A client that leaves once the model has actually started answering.

    Reporting the disconnect before admission would only exercise the pre-admission
    499 guard, which already existed; the gap this probe targets is a client that
    goes away *after* the slot was granted and generation is under way.
    """

    def __init__(self, leaves_after = None):
        self.state = SimpleNamespace()
        self.url = SimpleNamespace(path = "/v1/messages")
        self.method = "POST"
        self._leaves_after = leaves_after

    async def is_disconnected(self):
        return self._leaves_after is not None and self._leaves_after.is_set()


@pytest.fixture(autouse = True)
def _isolate(monkeypatch):
    reset_llama_admission_queues()
    monkeypatch.setattr(inf_mod, "api_monitor", ApiMonitor(max_entries = 64))
    monkeypatch.setattr(inf_mod, "_CANCEL_REGISTRY", {})
    yield
    reset_llama_admission_queues()


def _install_counting_backend(monkeypatch, emitted, started):
    def _gen_plain(**kwargs):
        cancel_event = kwargs.get("cancel_event")
        text = ""
        for _ in range(TOTAL_TOKENS):
            if cancel_event is not None and cancel_event.wait(STEP_S):
                return
            emitted["n"] += 1
            started.set()
            text += "x"
            yield text

    def _gen_tools(**kwargs):
        cancel_event = kwargs.get("cancel_event")
        for _ in range(TOTAL_TOKENS):
            if cancel_event is not None and cancel_event.wait(STEP_S):
                return
            emitted["n"] += 1
            started.set()
            yield {"type": "content", "text": "x"}

    backend = SimpleNamespace(
        is_loaded = True,
        is_vision = False,
        supports_tools = True,
        supports_tool_passthrough = False,
        model_identifier = "test-model",
        context_length = 2048,
        count_chat_tokens = lambda *a, **k: 2,
        generate_chat_completion = _gen_plain,
        generate_chat_completion_with_tools = _gen_tools,
        effective_parallel_slots = 2,
        base_url = "http://llama.pr10096.test:9999",
    )
    monkeypatch.setattr(inf_mod, "get_llama_cpp_backend", lambda: backend)
    return backend


def _payload(**fields) -> AnthropicMessagesRequest:
    base = {"max_tokens": 1024, "messages": [{"role": "user", "content": "hi"}]}
    base.update(fields)
    return AnthropicMessagesRequest(**base)


def _drive(monkeypatch, *, leaves, payload):
    emitted = {"n": 0}
    started = threading.Event()
    _install_counting_backend(monkeypatch, emitted, started)

    async def _run():
        return await anthropic_messages(
            payload,
            request = _Request(started if leaves else None),
            current_subject = "t",
        )

    response = asyncio.run(_run())
    rows = [r for r in inf_mod.api_monitor.snapshot() if r.get("kind") != "lifecycle"]
    return response, emitted["n"], rows


@pytest.mark.parametrize(
    "payload_kwargs",
    [
        pytest.param({}, id = "plain"),
        pytest.param(
            {"tools": [{"name": "web_search", "input_schema": {"type": "object"}}]},
            id = "tools",
        ),
    ],
)
def test_connected_client_still_gets_the_whole_answer(monkeypatch, payload_kwargs):
    """Control: nothing about the normal path may change."""
    response, emitted, rows = _drive(
        monkeypatch, leaves = False, payload = _payload(**payload_kwargs)
    )
    print(f"PROBE control emitted={emitted} status={[r.get('status') for r in rows]}")
    assert response.status_code == 200
    assert emitted == TOTAL_TOKENS
    assert rows and rows[0]["status"] == "completed"


@pytest.mark.parametrize(
    "payload_kwargs",
    [
        pytest.param({}, id = "plain"),
        pytest.param(
            {"tools": [{"name": "web_search", "input_schema": {"type": "object"}}]},
            id = "tools",
        ),
    ],
)
def test_disconnected_client_stops_the_generation(monkeypatch, payload_kwargs):
    """The repro: a client that leaves must not hold the model for the full answer."""
    response, emitted, rows = _drive(
        monkeypatch, leaves = True, payload = _payload(**payload_kwargs)
    )
    print(f"PROBE disconnect emitted={emitted} status={[r.get('status') for r in rows]}")
    assert response.status_code == 200
    assert emitted < TOTAL_TOKENS, (
        f"generation ran to completion after the client left: {emitted}/{TOTAL_TOKENS} tokens"
    )
    assert rows, "no api-monitor row was recorded"
    assert rows[0]["status"] == "cancelled", (
        f"api-monitor recorded a disconnected run as {rows[0]['status']!r}"
    )
