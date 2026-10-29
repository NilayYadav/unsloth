# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json

import httpx
import pytest

from core.inference import external_provider as ep
from core.inference.providers import get_base_url, list_available_providers


def _capture(monkeypatch, provider_type, model, thread_id = None, **kwargs) -> dict:
    captured: dict = {}

    def handler(request):
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        captured["headers"] = request.headers
        return httpx.Response(
            200, headers = {"content-type": "text/event-stream"}, content = "data: [DONE]\n\n"
        )

    async def run():
        async with httpx.AsyncClient(transport = httpx.MockTransport(handler)) as http:
            monkeypatch.setattr(ep, "_client", lambda: http)
            client = ep.ExternalProviderClient(provider_type, get_base_url(provider_type), "sk-test")
            async for _ in client.stream_chat_completion(
                messages = [{"role": "user", "content": "hi"}],
                model = model,
                temperature = 0.7,
                top_p = 0.95,
                max_tokens = 256,
                thread_id = thread_id,
                **kwargs,
            ):
                pass

    asyncio.run(run())
    return captured


def test_new_providers_listed():
    types = {p["provider_type"] for p in list_available_providers()}
    assert {"opencode", "opencode_go", "minimax"} <= types


def test_opencode_go_sends_session(monkeypatch):
    cap = _capture(monkeypatch, "opencode_go", "kimi-k3", thread_id = "thread-abc")
    assert cap["url"] == "https://opencode.ai/inference/go/v1/chat/completions"
    assert cap["headers"]["x-opencode-session"]
    assert cap["headers"]["user-agent"] == "unsloth-studio"
    assert cap["headers"]["authorization"] == "Bearer sk-test"


def test_opencode_zen_identifies_studio(monkeypatch):
    cap = _capture(monkeypatch, "opencode", "glm-5.3")
    assert cap["url"] == "https://opencode.ai/inference/v1/chat/completions"
    assert cap["headers"]["user-agent"] == "unsloth-studio"


def test_minimax_body_shape(monkeypatch):
    cap = _capture(monkeypatch, "minimax", "MiniMax-M3")
    body = cap["body"]
    assert cap["url"] == "https://api.minimax.io/v1/chat/completions"
    assert body["max_completion_tokens"] == 256
    assert "max_tokens" not in body
    assert body["reasoning_split"] is True
    for field in ("temperature", "top_p", "presence_penalty", "thinking"):
        assert field not in body


def test_minimax_m3_thinking_off(monkeypatch):
    cap = _capture(monkeypatch, "minimax", "MiniMax-M3", enable_thinking = False)
    assert cap["body"]["thinking"] == {"type": "disabled"}


@pytest.mark.parametrize(("effort", "sent"), [("xhigh", "xhigh"), ("none", None)])
def test_minimax_m31_flash_effort_and_never_disabled(monkeypatch, effort, sent):
    cap = _capture(
        monkeypatch,
        "minimax",
        "MiniMax-M3.1-Flash",
        enable_thinking = False,
        reasoning_effort = effort,
    )
    assert cap["body"].get("reasoning_effort") == sent
    assert "thinking" not in cap["body"]
