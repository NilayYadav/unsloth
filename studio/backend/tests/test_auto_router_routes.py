# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import core.inference.auto_router as auto_router
import routes.inference as inference
from core.inference.auto_router import RouterDecision, RouterProfile


def profile():
    return RouterProfile.model_validate(
        {
            "models": [
                {"id": "coder", "tasks": ["code"], "tools": True},
                {"id": "writer", "tasks": ["writing"], "tools": True},
                {"id": "general", "tasks": ["general"], "tools": True},
            ],
            "default_model": "general",
        }
    )


def fake_request(headers=None):
    return SimpleNamespace(headers=headers or {}, state=SimpleNamespace())


def payload(**fields):
    return SimpleNamespace(model="auto", tools=None, thread_id=None, model_extra={}, **fields)


def catalog(*ids):
    async def objects():
        return [{"id": model_id, "object": "model", "loaded": False} for model_id in ids]

    return objects


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    monkeypatch.setattr(auto_router, "_sessions", {})
    monkeypatch.setattr(auto_router, "get_profile", lambda: profile())
    monkeypatch.setattr(inference, "_openai_model_objects", lambda: [])
    monkeypatch.setattr(inference, "_openai_catalog_objects", catalog("coder", "writer", "general"))


def resolve(body, messages, request=None, **kwargs):
    request = request or fake_request()
    asyncio.run(inference._resolve_auto_model(body, request, messages, subject="owner", **kwargs))
    return request.state.auto_router_decision


def test_api_conversation_keeps_its_model_without_a_session_header(monkeypatch):
    monkeypatch.setattr(auto_router, "_classify", lambda prompt, choices: ("code", 0.55))
    first = [{"role": "system", "content": "You are an agent"}, {"role": "user", "content": "Fix my Python bug"}]
    assert resolve(payload(), first).model == "coder"

    monkeypatch.setattr(auto_router, "_classify", lambda prompt, choices: ("writing", 0.6))
    later = [*first, {"role": "assistant", "content": "Done."}, {"role": "user", "content": "make it nicer"}]
    decision = resolve(payload(), later)
    assert decision.model == "coder"
    assert decision.reason == "continuing with current model"


def test_auto_pools_every_downloaded_model_without_setup(monkeypatch):
    import core.inference.auto_router_capabilities as capabilities

    monkeypatch.setattr(auto_router, "get_profile", lambda: RouterProfile())
    monkeypatch.setattr(
        inference, "_openai_catalog_objects",
        catalog_with_task({"Qwen3-Coder-30B": None, "Llama-3.1-8B": None, "Whisper": "speech"}),
    )
    monkeypatch.setattr(
        capabilities, "detect_capabilities",
        lambda model_id: {"vision": False, "tools": True, "context_length": 32768},
    )
    monkeypatch.setattr(inference, "_openai_model_objects", lambda: [{"id": "llama-3.1-8b"}])
    monkeypatch.setattr(auto_router, "_classify", lambda prompt, choices: ("code", 0.9))

    profile, automatic = asyncio.run(inference._effective_auto_profile({"llama-3.1-8b"}))
    assert automatic
    assert [model.id for model in profile.models] == ["Qwen3-Coder-30B", "Llama-3.1-8B"]
    assert profile.default_model == "Llama-3.1-8B"

    decision = resolve(payload(), [{"role": "user", "content": "Fix my Python bug"}])
    assert decision.model == "Llama-3.1-8B"
    assert decision.reason == "code model is not loaded, keeping the current model"


def test_auto_without_downloaded_models_explains_itself(monkeypatch):
    monkeypatch.setattr(auto_router, "get_profile", lambda: RouterProfile())
    monkeypatch.setattr(inference, "_openai_catalog_objects", catalog())
    with pytest.raises(HTTPException, match="no downloaded models"):
        resolve(payload(), [{"role": "user", "content": "hi"}])


def test_saved_profile_missing_its_default_falls_back_to_nothing(monkeypatch):
    monkeypatch.setattr(inference, "_openai_catalog_objects", catalog("coder", "writer"))
    profile, automatic = asyncio.run(inference._effective_auto_profile())
    assert not automatic
    assert profile.models == []


def catalog_with_task(tasks):
    async def objects():
        return [{"id": model_id, "object": "model", "loaded": False, "task": task} for model_id, task in tasks.items()]

    return objects


def test_different_conversations_do_not_share_a_session(monkeypatch):
    monkeypatch.setattr(auto_router, "_classify", lambda prompt, choices: ("code", 0.55))
    resolve(payload(), [{"role": "user", "content": "Fix my Python bug"}])
    monkeypatch.setattr(auto_router, "_classify", lambda prompt, choices: ("writing", 0.55))
    assert resolve(payload(), [{"role": "user", "content": "Write a poem"}]).model == "writer"


def test_anthropic_tool_result_turn_never_reclassifies(monkeypatch):
    monkeypatch.setattr(auto_router, "_classify", lambda prompt, choices: ("code", 0.9))
    first = [{"role": "user", "content": "Refactor the parser"}]
    assert resolve(payload(system="agent"), first).model == "coder"

    def must_not_classify(prompt, choices):
        raise AssertionError("tool results must not be classified")

    monkeypatch.setattr(auto_router, "_classify", must_not_classify)
    turn = [
        *first,
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "read", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "file text"}]},
    ]
    decision = resolve(payload(system="agent"), turn)
    assert decision.model == "coder"
    assert decision.reason == "continuing a tool call"


def test_count_only_uses_the_loaded_model_without_classifying(monkeypatch):
    monkeypatch.setattr(inference, "_openai_model_objects", lambda: [{"id": "Writer"}])

    def must_not_classify(prompt, choices):
        raise AssertionError("counting must not classify")

    monkeypatch.setattr(auto_router, "_classify", must_not_classify)
    request = fake_request()
    decision = resolve(payload(), [{"role": "user", "content": "hi"}], request, count_only=True)
    assert decision.model == "writer"
    assert request.state.auto_router_session is None


def test_body_pin_is_honoured_and_not_forwarded(monkeypatch):
    monkeypatch.setattr(auto_router, "_classify", lambda prompt, choices: ("code", 0.9))
    body = payload()
    body.model_extra["router_pin"] = "writer"
    decision = resolve(body, [{"role": "user", "content": "Fix my Python bug"}])
    assert decision.model == "writer"
    assert decision.reason == "pinned by user"
    assert "router_pin" not in body.model_extra


def test_failed_load_falls_back_to_the_next_model(monkeypatch):
    calls = []

    async def switch(model, request, subject, **kwargs):
        calls.append((model, kwargs.get("alongside")))
        if model == "coder":
            raise HTTPException(status_code=409, detail="does not fit")

    monkeypatch.setattr(inference, "_maybe_auto_switch_model", switch)
    request = fake_request()
    body = payload()
    body.model = "coder"
    decision = RouterDecision("coder", "code task", "code", ("general", "writer"))
    request.state.auto_router_decision = decision
    request.state.auto_router_payload = body
    request.state.auto_router_session = ("owner", "thread-1")
    asyncio.run(inference._switch_with_auto_fallback(decision, request, "owner", {}))

    assert calls == [("coder", True), ("general", True)]
    assert body.model == "general"
    assert request.state.auto_router_decision.model == "general"
    assert "coder could not load" in request.state.auto_router_decision.reason
    assert auto_router.session_model("owner", "thread-1") == "general"


def test_bad_request_is_not_retried_on_another_model(monkeypatch):
    async def switch(model, request, subject, **kwargs):
        raise HTTPException(status_code=400, detail="bad image")

    monkeypatch.setattr(inference, "_maybe_auto_switch_model", switch)
    request = fake_request()
    decision = RouterDecision("coder", "code task", "code", ("general",))
    request.state.auto_router_decision = decision
    request.state.auto_router_payload = payload()
    request.state.auto_router_session = None
    with pytest.raises(HTTPException):
        asyncio.run(inference._switch_with_auto_fallback(decision, request, "owner", {}))


def test_stream_starts_with_the_router_decision():
    async def body():
        yield "data: {\"choices\": []}\n\n"

    response = SimpleNamespace(body_iterator=body())

    async def produce(*args, **kwargs):
        return response

    request = fake_request({"X-Unsloth-Events": "1"})
    request.state.auto_router_decision = RouterDecision("coder", "code task", "code")

    async def collect():
        result = inference.with_router_decision_frame(await produce(), request)
        return [chunk async for chunk in result.body_iterator]

    frames = asyncio.run(collect())
    first = json.loads(frames[0].removeprefix("data: "))
    assert first == {"type": "router_decision", "model": "coder", "reason": "code task", "task": "code"}
    assert frames[1] == "data: {\"choices\": []}\n\n"


def test_plain_openai_stream_has_no_router_frame():
    async def body():
        yield "data: {}\n\n"

    response = SimpleNamespace(body_iterator=body())

    async def produce(*args, **kwargs):
        return response

    request = fake_request()
    request.state.auto_router_decision = RouterDecision("coder", "code task", "code")
    assert inference.with_router_decision_frame(asyncio.run(produce()), request) is response
