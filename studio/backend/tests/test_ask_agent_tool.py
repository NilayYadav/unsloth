# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import asyncio
import json
import threading
import uuid
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from core.inference import ask_agent_tool, studio_tool_loop as loop_mod, tools
from core.inference.studio_tool_loop import (
    ToolLoopPolicy,
    ToolLoopRun,
    stream_with_studio_tools,
)
from core.inference.tool_loop_controller import status_for_tool
from core.scheduled_tasks import scheduler
from models.inference import ChatCompletionRequest
from routes import inference
from state import tool_approvals
from storage import studio_db

SECRET = "The vault code is 4417."
DONE = "data: [DONE]"


def _save(id, name, **fields):
    studio_db.upsert_chat_agent(
        {
            "id": id,
            "name": name,
            "tools": {"webSearch": False, "code": False, "images": False},
            "createdAt": 1,
            "updatedAt": 1,
            **fields,
        }
    )


@pytest.fixture(autouse = True)
def agents(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", set())
    monkeypatch.setattr(inference, "_enabled_agent_skills", lambda: [])
    monkeypatch.setattr(inference, "_thread_has_conversation_archive", lambda _: False)
    monkeypatch.setattr(loop_mod, "build_rag_autoinject", lambda *a, **k: None)
    _save(
        "reader",
        "Reader",
        description = "Reads attached files and reports what they say.",
        systemPrompt = "Answer from the file only.",
        canAsk = ["desk"],
    )
    _save("coder", "Coder", tools = {"webSearch": True, "code": True, "images": False})
    _save("desk", "Desk", canAsk = ["reader", "coder", "gone", "desk"], filesForAskedOnly = True)
    _save("solo", "Solo")


def _chat(**fields) -> ChatCompletionRequest:
    return ChatCompletionRequest.model_validate(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "agent_id": "desk",
            "provider_id": "cloud",
            "external_model": "cloud-model",
            **fields,
        }
    )


async def _bind(payload):
    await ask_agent_tool.bind_request(payload, SimpleNamespace(headers = {}, app = None), "unsloth")


def _content(text: str) -> str:
    return "data: " + json.dumps(
        {"model": "reader-model", "choices": [{"delta": {"content": text}}]}
    )


def _reply(*lines: str):
    async def body():
        for line in lines:
            yield line + "\n\n"

    return StreamingResponse(body(), media_type = "text/event-stream")


class Asked:
    """Stands in for the asked agent's model: records the request it was sent."""

    def __init__(
        self,
        monkeypatch,
        respond = None,
    ):
        self.payloads: list[ChatCompletionRequest] = []
        self.requests = []
        self.offered = []
        self.respond = respond or (lambda payload: _reply(_content("It says 4417."), DONE))
        monkeypatch.setattr(inference, "produce_openai_chat_completions", self)

    async def __call__(self, payload, request, subject, *, cancel_on_disconnect):
        self.payloads.append(payload)
        self.requests.append(request)
        self.offered.append(ask_agent_tool.offered_tool())
        return self.respond(payload)


def _ask(
    arguments,
    payload = None,
    **kwargs,
) -> str:
    async def go():
        await _bind(payload or _chat())
        return await asyncio.to_thread(tools.execute_tool, "ask_agent", arguments, **kwargs)

    return asyncio.run(go())


def _selected(payload) -> dict:
    async def go():
        await _bind(payload)
        chosen = await inference._select_request_tools(
            SimpleNamespace(enabled_tools = [], rag_scope = None, bypass_permissions = False),
            tools_on = True,
            mcp_allowed = False,
        )
        return {tool["function"]["name"]: tool["function"] for tool in chosen}

    return asyncio.run(go())


def test_a_chat_whose_agent_can_ask_is_offered_one_tool_built_from_those_agents():
    ask = _selected(_chat())["ask_agent"]

    assert ask["parameters"]["properties"]["agent"]["enum"] == ["Reader", "Coder"]
    assert "- Reader: Reads attached files and reports what they say." in ask["description"]
    assert "- Coder" in ask["description"]
    assert "files" not in ask["parameters"]["properties"]


def test_attached_files_kept_for_asked_agents_can_be_passed_by_name():
    payload = _chat(ask_agent_files = [{"name": "notes.txt", "text": SECRET}])

    ask = _selected(payload)["ask_agent"]

    assert ask["parameters"]["properties"]["files"]["items"]["enum"] == ["notes.txt"]
    assert SECRET not in json.dumps(ask)


@pytest.mark.parametrize(
    "fields",
    [{"agent_id": "solo"}, {"agent_id": "nobody"}, {"agent_id": None}, {"stream": False}],
)
def test_a_chat_with_nobody_to_ask_is_not_offered_the_tool(fields):
    assert "ask_agent" not in _selected(_chat(**fields))


def test_a_request_that_can_ask_keeps_the_tool_loop_when_only_hosted_tools_are_on():
    hosted = SimpleNamespace(enabled_tools = ["web_search"], mcp_enabled = False)

    async def go(payload):
        await _bind(payload)
        return inference._selects_only_provider_hosted_tools(hosted, "anthropic", [])

    assert asyncio.run(go(_chat(agent_id = "solo"))) is True
    assert asyncio.run(go(_chat())) is False


def test_the_asked_agent_starts_fresh_with_its_own_prompt_and_only_the_task(monkeypatch):
    asked = Asked(monkeypatch)

    result = _ask({"agent": "reader", "task": "What is the code?"})

    assert result == "Reader (reader-model) replied:\n\nIt says 4417."
    sent = asked.payloads[0]
    assert [(m.role, m.content) for m in sent.messages] == [
        ("system", "Answer from the file only."),
        ("user", "What is the code?"),
    ]
    assert (sent.provider_id, sent.external_model) == ("cloud", "cloud-model")
    assert sent.enable_tools is False and sent.enabled_tools == []


def test_an_agent_that_is_being_asked_cannot_ask_anyone(monkeypatch):
    asked = Asked(monkeypatch)

    _ask({"agent": "Reader", "task": "Go."})

    assert asked.offered == [None]
    assert asked.payloads[0].agent_id is None


def test_a_named_file_is_read_by_the_asked_agent_and_never_by_the_chats_model(monkeypatch):
    asked = Asked(monkeypatch)
    call = json.dumps({"agent": "Reader", "task": "What is the code?", "files": ["notes.txt"]})
    delta = {"index": 0, "id": "c1", "type": "function"}
    turns = [
        [
            "data: "
            + json.dumps(
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {**delta, "function": {"name": "ask_agent", "arguments": call}}
                                ]
                            },
                        }
                    ]
                }
            ),
            "data: "
            + json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}),
        ],
        ["data: " + json.dumps({"choices": [{"index": 0, "delta": {"content": "4417"}}]}), DONE],
    ]
    provider_requests: list[list[dict]] = []

    class Provider:
        heals_text_tool_calls = False
        sanitizes_provider_frames = True

        def stream(self, *, messages, tools, tool_choice, cancel_event):
            provider_requests.append([dict(message) for message in messages])
            lines = turns.pop(0) if turns else [DONE]

            async def body():
                for line in lines:
                    yield line

            return body()

    async def go():
        await _bind(_chat(ask_agent_files = [{"name": "notes.txt", "text": SECRET}]))
        stream = stream_with_studio_tools(
            Provider(),
            run = ToolLoopRun(
                messages = [{"role": "user", "content": "[notes.txt (23 B) is attached.]"}],
                session_id = "s1",
                model = "cloud-model",
            ),
            policy = ToolLoopPolicy(
                tools = [ask_agent_tool.offered_tool()],
                max_calls = 25,
                timeout = 300,
                permission_mode = "auto",
                confirm_calls = True,
                bypass_permissions = False,
                rag_scope = None,
            ),
            cancel_event = threading.Event(),
        )
        return [line async for line in stream]

    asyncio.run(go())

    assert asked.payloads[0].messages[-1].content == (
        f"What is the code?\n<attachment name=notes.txt>\n{SECRET}\n</attachment>"
    )
    assert len(provider_requests) == 2
    assert SECRET not in json.dumps(provider_requests)
    assert provider_requests[1][-1]["role"] == "tool"
    assert provider_requests[1][-1]["content"] == "Reader (reader-model) replied:\n\nIt says 4417."


def test_only_what_the_asked_agent_says_after_its_last_tool_call_comes_back(monkeypatch):
    frames = [
        _content("Let me look that up."),
        "data: " + json.dumps({"type": "tool_start", "tool_name": "web_search", "arguments": {}}),
        "data: "
        + json.dumps({"type": "tool_end", "tool_name": "web_search", "result": "raw page"}),
        _content("The answer "),
        _content("is 42."),
        DONE,
    ]
    Asked(monkeypatch, lambda payload: _reply(*frames))

    result = _ask({"agent": "Coder", "task": "Find it."})

    assert result == "Coder (reader-model) replied:\n\nThe answer is 42."


def test_a_call_of_the_asked_agent_that_needs_approval_is_declined(monkeypatch):
    decisions = []

    def respond(payload):
        approval_id = uuid.uuid4().hex
        slot = tool_approvals.begin_tool_decision(payload.session_id, approval_id)

        async def body():
            start = {
                "type": "tool_start",
                "tool_name": "python",
                "approval_id": approval_id,
                "awaiting_confirmation": True,
            }
            yield "data: " + json.dumps(start) + "\n\n"
            decisions.append(
                await asyncio.to_thread(tool_approvals.wait_tool_decision, slot, approval_id)
            )
            yield _content("Done without it.") + "\n\n" + DONE + "\n\n"

        return StreamingResponse(body())

    Asked(monkeypatch, respond)

    result = _ask({"agent": "Coder", "task": "Run it."}, _chat(session_id = "s1"))

    assert decisions == ["deny"]
    assert result.endswith("Done without it.")


def test_the_asked_agent_runs_under_the_chats_permissions_with_its_own_tool_switches(monkeypatch):
    asked = Asked(monkeypatch)
    chat = _chat(
        permission_mode = "ask",
        sandbox_level = "low",
        session_id = "s1",
        max_tool_calls_per_message = 7,
        tool_call_timeout = 120,
    )

    _ask({"agent": "Coder", "task": "Go."}, chat)

    sent = asked.payloads[0]
    assert sent.enable_tools is True
    assert sent.enabled_tools == ["web_search", "python", "terminal", "edit_file", "view_image"]
    assert (sent.permission_mode, sent.sandbox_level, sent.session_id) == ("ask", "low", "s1")
    assert (sent.max_tool_calls_per_message, sent.tool_call_timeout) == (7, 120)
    assert sent.thread_id is None and sent.mcp_enabled is not True
    assert asked.requests[0].headers.get("x-unsloth-events") == "1"


def test_a_chat_on_a_connection_loads_the_asked_agents_local_model_when_it_is_asked(monkeypatch):
    _save("local", "Local", modelId = "org/Model-GGUF", modelGgufVariant = "Q4_K_M")
    _save("desk", "Desk", canAsk = ["local"])
    monkeypatch.setattr(scheduler, "_model_is_loaded", lambda model: False)
    asked = Asked(monkeypatch)

    _ask({"agent": "Local", "task": "Go."})

    assert asked.payloads[0].model == "org/Model-GGUF:Q4_K_M"
    assert asked.payloads[0].provider_id is None
    assert inference._model_switch_allowed_for_request(asked.requests[0])
    assert inference._load_alongside_for_request(asked.requests[0])


def test_a_chat_on_a_local_model_never_swaps_models_to_ask(monkeypatch):
    _save("local", "Local", modelId = "org/Other-GGUF")
    _save("desk", "Desk", canAsk = ["local", "reader"])
    monkeypatch.setattr(inference, "get_llama_cpp_backend", lambda: SimpleNamespace(is_loaded = True))
    loaded = {"org/Mine-GGUF"}
    monkeypatch.setattr(scheduler, "_model_is_loaded", lambda model: model in loaded)
    asked = Asked(monkeypatch)
    chat = _chat(provider_id = None, external_model = None, model = "org/Mine-GGUF")

    refused = _ask({"agent": "Local", "task": "Go."}, chat)

    assert refused.startswith("Error: Local runs on org/Other-GGUF, which is not loaded")
    assert asked.payloads == []

    loaded.add("org/Other-GGUF")
    _ask({"agent": "Local", "task": "Go."}, chat)
    _ask({"agent": "Reader", "task": "Go."}, chat)

    assert [payload.model for payload in asked.payloads] == ["org/Other-GGUF", "org/Mine-GGUF"]
    assert not any(inference._model_switch_allowed_for_request(r) for r in asked.requests)


def _refuses(detail):
    def respond(payload):
        raise HTTPException(status_code = 409, detail = detail)

    return respond


@pytest.mark.parametrize(
    ("arguments", "respond", "expected"),
    [
        ({"agent": "Nobody", "task": "x"}, None, "Error: there is no agent named 'Nobody' to ask."),
        ({"agent": "Reader"}, None, "Error: ask_agent needs a task."),
        (
            {"agent": "Reader", "task": "x", "files": ["other.txt"]},
            None,
            "Error: no attached file named other.txt. Attached files: none.",
        ),
        (
            {"agent": "Reader", "task": "x"},
            _refuses({"error": {"message": "model_busy"}}),
            "Error: Reader could not answer: model_busy",
        ),
        (
            {"agent": "Reader", "task": "x"},
            lambda payload: _reply('data: {"error": {"message": "out of context"}}'),
            "Error: out of context",
        ),
        (
            {"agent": "Reader", "task": "x"},
            lambda payload: _reply(DONE),
            "Error: Reader returned an empty reply.",
        ),
    ],
)
def test_a_failed_ask_is_an_error_result_the_chat_can_read(
    monkeypatch, arguments, respond, expected
):
    Asked(monkeypatch, respond)

    assert _ask(arguments).startswith(expected)


def test_stopping_the_chat_stops_the_asked_agent(monkeypatch):
    monkeypatch.setattr(ask_agent_tool, "_STOP_WAIT_S", 5.0)
    started = threading.Event()

    def respond(payload):
        async def body():
            started.set()
            await asyncio.sleep(60)
            yield DONE

        return StreamingResponse(body())

    asked = Asked(monkeypatch, respond)
    stop = threading.Event()
    threading.Thread(target = lambda: (started.wait(20), stop.set()), daemon = True).start()

    result = _ask({"agent": "Reader", "task": "x"}, cancel_event = stop)

    assert result == "Error: asking Reader was stopped."
    assert asked.requests[0].state.generation_cancel_event.is_set()


def test_the_ask_runs_unprompted_and_says_who_is_being_asked():
    assert tools.is_high_risk_tool_call("ask_agent", {"agent": "Reader", "task": "x"}) is False
    assert status_for_tool("ask_agent", {"agent": "Reader"}) == "Asking Reader..."
