# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Chat tool: ask another saved agent. It runs in a fresh context and only its final reply comes back"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

ASK_AGENT_TOOL_NAME = "ask_agent"

_DESCRIPTION = (
    "Ask another agent to do a task and return its reply. It starts with no memory of this "
    "conversation, so put everything it needs to know into the task. Agents you can ask:"
)
_TOOL_NAMES = (
    ("webSearch", ("web_search",)),
    ("code", ("python", "terminal", "edit_file", "view_image")),
    ("images", ("generate_image", "generate_video")),
)
_MODEL_FIELDS = {
    "model",
    "provider_id",
    "provider_type",
    "external_model",
    "encrypted_api_key",
    "provider_base_url",
    "provider_api_type",
    "provider_reasoning_config",
}
_RUN_FIELDS = {
    "permission_mode",
    "confirm_tool_calls",
    "bypass_permissions",
    "sandbox_level",
    "max_tool_calls_per_message",
    "tool_call_timeout",
    "auto_heal_tool_calls",
    "session_id",
}
# A chat with no tool time limit still gets a ceiling, so a stuck run cannot hold the turn forever.
_MAX_WAIT_S = 3600.0
_STOP_WAIT_S = 30.0
_POLL_S = 0.25


@dataclass(frozen = True)
class AskAgentRequest:
    """What one chat request may ask: set per request, read by the tool thread that copies its context."""

    agents: tuple[dict, ...]
    files: dict[str, str]
    model_fields: dict
    run_fields: dict
    local: bool
    app: Any
    subject: str
    timezone_headers: dict[str, str]
    loop: asyncio.AbstractEventLoop


_bound: contextvars.ContextVar[Optional[AskAgentRequest]] = contextvars.ContextVar(
    "unsloth_ask_agent", default = None
)


class _Refused(Exception):
    pass


def askable_agents(agent_id: str) -> list[dict]:
    """The saved agents ``agent_id`` lists under Can ask that still exist, one per name."""
    from storage.studio_db import list_chat_agents

    agents = {agent["id"]: agent for agent in list_chat_agents()}
    asker = agents.get(agent_id)
    if asker is None:
        return []
    by_name: dict[str, dict] = {}
    for listed in asker["canAsk"]:
        agent = agents.get(listed)
        if agent is not None and listed != agent_id:
            by_name.setdefault(agent["name"].casefold(), agent)
    return list(by_name.values())


async def bind_request(payload, request, subject: str) -> None:
    from utils.current_date_prompt_settings import CURRENT_DATE_TIMEZONE_HEADER

    _bound.set(None)
    if not payload.agent_id or payload.stream is not True:
        return
    agents = await asyncio.to_thread(askable_agents, payload.agent_id)
    if not agents:
        return
    timezone = request.headers.get(CURRENT_DATE_TIMEZONE_HEADER)
    _bound.set(
        AskAgentRequest(
            agents = tuple(agents),
            files = {file.name: file.text for file in payload.ask_agent_files or []},
            model_fields = payload.model_dump(include = _MODEL_FIELDS, exclude_unset = True),
            run_fields = payload.model_dump(include = _RUN_FIELDS, exclude_none = True),
            local = not (payload.provider_id or payload.provider_type),
            app = request.app,
            subject = subject,
            timezone_headers = {CURRENT_DATE_TIMEZONE_HEADER: timezone} if timezone else {},
            loop = asyncio.get_running_loop(),
        )
    )


def bound() -> bool:
    return _bound.get() is not None


def offered_tool() -> Optional[dict]:
    """The ask_agent schema for this request, or None when its chat has nobody to ask."""
    request = _bound.get()
    if request is None:
        return None
    lines = [
        f"- {agent['name']}: {agent['description']}"
        if agent["description"]
        else f"- {agent['name']}"
        for agent in request.agents
    ]
    properties: dict[str, Any] = {
        "agent": {
            "type": "string",
            "enum": [agent["name"] for agent in request.agents],
            "description": "The agent to ask.",
        },
        "task": {
            "type": "string",
            "description": "What it should do, with everything it needs to know.",
        },
    }
    if request.files:
        properties["files"] = {
            "type": "array",
            "items": {"type": "string", "enum": list(request.files)},
            "description": (
                "Names of attached files it should read. Their content goes to that agent, not to you."
            ),
        }
    return {
        "type": "function",
        "function": {
            "name": ASK_AGENT_TOOL_NAME,
            "description": "\n".join([_DESCRIPTION, *lines]),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": ["agent", "task"],
            },
        },
    }


def _error_text(detail: Any) -> str:
    if isinstance(detail, dict):
        nested = detail.get("error")
        if isinstance(nested, dict):
            detail = nested
        detail = detail.get("message") or detail.get("detail") or detail
    return detail if isinstance(detail, str) else json.dumps(detail, default = str)


def _model_label(model: Optional[str]) -> str:
    from routes.inference import _connection_model_ref
    connection = _connection_model_ref(model)
    return connection[1] if connection is not None else (model or "")


def _model_fields(request: AskAgentRequest, agent: dict) -> tuple[dict, bool]:
    """The request fields that pick the asked agent's model, and whether it has to be loaded first."""
    from core.scheduled_tasks.scheduler import _model_is_loaded
    from routes.inference import _connection_model_ref, get_llama_cpp_backend
    from storage import providers_db

    model = agent.get("modelId")
    connection = _connection_model_ref(model)
    if connection is not None:
        provider_id, external_model = connection
        config = providers_db.get_provider(provider_id)
        if config is None or not config["is_enabled"]:
            raise _Refused(f"the connection {agent['name']} runs on is not available.")
        return {"model": model, "provider_id": provider_id, "external_model": external_model}, False
    variant = agent.get("modelGgufVariant")
    requested = f"{model}:{variant}" if variant else model
    if not request.local:
        if not model:
            return dict(request.model_fields), False
        return {"model": requested}, not _model_is_loaded(requested)
    if not getattr(get_llama_cpp_backend(), "is_loaded", False):
        raise _Refused(
            "a chat on a local model can ask an agent only while that model is a GGUF model."
        )
    if not model:
        return dict(request.model_fields), False
    if _model_is_loaded(requested):
        return {"model": requested}, False
    # Loading it would unload the model this chat's turn is running on, and the switch waits for that turn.
    raise _Refused(
        f"{agent['name']} runs on {requested}, which is not loaded, and loading it now would unload "
        "this chat's model. Load both models first (Settings > System > Keep multiple models "
        f"loaded) or give {agent['name']} this chat's model."
    )


async def _final_reply(chunks) -> tuple[str, Optional[str]]:
    """The text after the last tool call, and the model that wrote it. A call that asks for approval is declined."""
    from core.inference.chat_generation_runs import _SSEDecoder
    from state.tool_approvals import resolve_tool_decision

    decoder = _SSEDecoder()
    text, model = "", None
    try:
        async for raw in chunks:
            block = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)
            for data in decoder.feed(block):
                if data == "[DONE]":
                    return text, model
                try:
                    chunk = json.loads(data)
                except ValueError:
                    continue
                if not isinstance(chunk, dict):
                    continue
                if chunk.get("error"):
                    raise _Refused(_error_text(chunk))
                if chunk.get("type") == "tool_start":
                    text = ""
                    if chunk.get("awaiting_confirmation"):
                        resolve_tool_decision(chunk.get("approval_id"), "deny")
                    continue
                model = model or chunk.get("model")
                for choice in chunk.get("choices") or []:
                    text += (choice.get("delta") or {}).get("content") or ""
    finally:
        aclose = getattr(chunks, "aclose", None)
        if aclose is not None:
            await aclose()
    return text, model


async def _ask(
    request: AskAgentRequest,
    agent: dict,
    task: str,
    files: list[str],
    cancel_event: threading.Event,
) -> str:
    from core.inference.chat_generation_runs import _background_request
    from core.inference.llama_keepwarm import InferenceActivityReservation, end_in_process_request
    from models.inference import ChatCompletionRequest
    from routes.inference import (
        _extract_responses_reasoning,
        allow_model_switch_for_request,
        disable_auto_download_for_request,
        load_alongside_for_request,
        produce_openai_chat_completions,
    )

    model_fields, load_first = await asyncio.to_thread(_model_fields, request, agent)
    tools = [name for switch, names in _TOOL_NAMES if agent["tools"].get(switch) for name in names]
    content = "\n".join(
        [
            task,
            *(f"<attachment name={name}>\n{request.files[name]}\n</attachment>" for name in files),
        ]
    )
    messages = [{"role": "user", "content": content}]
    if agent["systemPrompt"].strip():
        messages.insert(0, {"role": "system", "content": agent["systemPrompt"]})
    run_id = f"ask-{uuid.uuid4().hex}"
    payload = ChatCompletionRequest.model_validate(
        {
            **request.run_fields,
            **model_fields,
            "messages": messages,
            "stream": True,
            "cancel_id": run_id,
            "enable_tools": bool(tools),
            "enabled_tools": tools,
        }
    )
    inner = _background_request(
        request.app, run_id, cancel_event, request.timezone_headers, via_api_key = False
    )
    disable_auto_download_for_request(inner.scope)
    if load_first:
        allow_model_switch_for_request(inner.scope)
        load_alongside_for_request(inner.scope)
    activity = InferenceActivityReservation()
    activity.reserve()
    answered = False
    try:
        await activity.start(cancel_event)
        response = await produce_openai_chat_completions(
            payload, inner, request.subject, cancel_on_disconnect = False
        )
        chunks = getattr(response, "body_iterator", None)
        if chunks is None:
            raise _Refused(_error_text(json.loads(bytes(response.body))))
        text, model = await _final_reply(chunks)
        answered = True
    finally:
        end_in_process_request(inner.scope, answered = answered)
        activity.finish(inner.scope)
    if model_fields.get("provider_id") or model_fields.get("provider_type"):
        text = _extract_responses_reasoning(text, parse_think_markers = True)[1]
    if not text.strip():
        raise _Refused(f"{agent['name']} returned an empty reply.")
    label = _model_label(model or model_fields.get("model"))
    return f"{agent['name']} ({label}) replied:\n\n{text.strip()}"


def _run(arguments: dict, cancel_event, timeout: Optional[float]) -> str:
    from fastapi import HTTPException

    from utils.account_context import arun_as, current_account

    request = _bound.get()
    if request is None:
        return "Error: this chat has no agents to ask."
    names = ", ".join(agent["name"] for agent in request.agents)
    name = str(arguments.get("agent") or "").strip()
    agent = next((a for a in request.agents if a["name"].casefold() == name.casefold()), None)
    if agent is None:
        return f"Error: there is no agent named '{name}' to ask. Agents you can ask: {names}."
    task = str(arguments.get("task") or "").strip()
    if not task:
        return "Error: ask_agent needs a task."
    files = arguments.get("files") or []
    files = [files] if isinstance(files, str) else [str(file) for file in files]
    missing = [file for file in files if file not in request.files]
    if missing:
        available = ", ".join(request.files) or "none"
        return f"Error: no attached file named {', '.join(missing)}. Attached files: {available}."

    stop = threading.Event()
    # An empty context: the asked agent's run must not inherit this chat's ask_agent, slot or tool selection.
    future = contextvars.Context().run(
        asyncio.run_coroutine_threadsafe,
        arun_as(current_account(), _ask(request, agent, task, files, stop)),
        request.loop,
    )
    limit = _MAX_WAIT_S if timeout is None or timeout >= 9999 else max(1.0, float(timeout))
    deadline = time.monotonic() + limit
    stopped = None
    while not future.done():
        if cancel_event is not None and cancel_event.is_set():
            stopped = f"Error: asking {agent['name']} was stopped."
        elif time.monotonic() >= deadline:
            stopped = (
                f"Error: {agent['name']} did not finish within {int(limit)} seconds, this chat's "
                "time limit for one tool call."
            )
        if stopped:
            stop.set()
            concurrent.futures.wait([future], timeout = _STOP_WAIT_S)
            future.cancel()
            return stopped
        concurrent.futures.wait([future], timeout = _POLL_S)
    try:
        return future.result()
    except concurrent.futures.CancelledError:
        return f"Error: asking {agent['name']} was stopped."
    except _Refused as exc:
        return f"Error: {exc}"
    except HTTPException as exc:
        return f"Error: {agent['name']} could not answer: {_error_text(exc.detail)}"


def run_ask_agent(
    arguments: Any,
    *,
    cancel_event = None,
    timeout: Optional[float] = None,
) -> str:
    """Ask one agent; always a string, an ``Error: ...`` one on failure."""
    try:
        return _run(arguments if isinstance(arguments, dict) else {}, cancel_event, timeout)
    except Exception as exc:  # noqa: BLE001 - a tool result, never a crashed turn
        logger.warning("ask_agent failed: %s", exc, exc_info = True)
        return f"Error: asking the agent failed: {exc}"
