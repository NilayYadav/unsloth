# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import threading
from types import SimpleNamespace

from core.inference.safetensors_agentic import run_safetensors_tool_loop
from core.inference.subagents import (
    SUBAGENT_TOOL_NAME,
    build_subagent_tool,
    is_project_session,
    make_subagent_runner,
)


def _tool(name):
    return {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}


PARENT_TOOLS = [_tool("web_search"), _tool("python"), _tool("terminal")]


def _drain(gen):
    events = []
    try:
        while True:
            events.append(next(gen))
    except StopIteration as stop:
        return events, stop.value


class FakeLoop:
    def __init__(self, events):
        self.events = events
        self.calls = []

    def __call__(self, messages, tools, effort, budget):
        self.calls.append(
            {
                "messages": messages,
                "tools": [t["function"]["name"] for t in tools],
                "effort": effort,
                "budget": budget,
            }
        )
        yield from self.events


def test_is_project_session():
    assert is_project_session("project-abc")
    assert not is_project_session("thread-abc")
    assert not is_project_session(None)


def test_tool_schema_scopes_to_parent_tools_without_recursion():
    spec = build_subagent_tool(PARENT_TOOLS + [_tool(SUBAGENT_TOOL_NAME)])
    params = spec["function"]["parameters"]
    assert spec["function"]["name"] == SUBAGENT_TOOL_NAME
    assert params["properties"]["tools"]["items"]["enum"] == ["web_search", "python", "terminal"]
    assert params["properties"]["reasoning_effort"]["enum"] == ["low", "medium", "high"]
    assert params["required"] == ["task"]


def test_fixed_effort_hides_effort_param():
    spec = build_subagent_tool(PARENT_TOOLS, fixed_effort = "high")
    assert "reasoning_effort" not in spec["function"]["parameters"]["properties"]


def test_runner_scopes_tools_and_effort():
    loop = FakeLoop(
        [
            {"type": "status", "text": "Searching: x"},
            {"type": "tool_end", "tool_name": "web_search", "result": "r"},
            {"type": "status", "text": ""},
            {"type": "content", "text": "<think>hmm</think>Found"},
            {"type": "content", "text": "<think>hmm</think>Found it."},
        ]
    )
    runner = make_subagent_runner(
        run_loop = loop, parent_tools = PARENT_TOOLS + [_tool(SUBAGENT_TOOL_NAME)]
    )
    events, result = _drain(
        runner(
            {
                "task": "look up x",
                "name": "researcher",
                "tools": ["web_search", "spawn_subagent", "rm_rf"],
                "reasoning_effort": "low",
            }
        )
    )
    call = loop.calls[0]
    assert call["tools"] == ["web_search"]
    assert call["effort"] == "low"
    assert call["budget"] == 6
    assert call["messages"][0]["role"] == "system"
    assert call["messages"][1] == {"role": "user", "content": "look up x"}
    assert [e["text"] for e in events] == [
        "Sub-agent researcher (low): starting",
        "Sub-agent researcher (low): Searching: x",
    ]
    assert "Tools: web_search" in result
    assert "Denied tools (not in planner scope): spawn_subagent, rm_rf" in result
    assert "Tool calls: web_search" in result
    assert result.endswith("Report:\nFound it.")


def test_runner_defaults_and_fixed_effort():
    loop = FakeLoop([{"type": "content", "text": "ok"}])
    runner = make_subagent_runner(run_loop = loop, parent_tools = PARENT_TOOLS, fixed_effort = "high")
    _, result = _drain(runner({"task": "think", "reasoning_effort": "low"}))
    assert loop.calls[0]["tools"] == []
    assert loop.calls[0]["effort"] == "high"
    assert loop.calls[0]["budget"] == 25
    assert "Tools: none" in result

    loop = FakeLoop([{"type": "content", "text": "ok"}])
    runner = make_subagent_runner(run_loop = loop, parent_tools = PARENT_TOOLS)
    _drain(runner({"task": "think", "tools": "python, terminal", "reasoning_effort": "bogus"}))
    assert loop.calls[0]["tools"] == ["python", "terminal"]
    assert loop.calls[0]["effort"] == "medium"


def test_runner_rejects_empty_task_and_reports_errors():
    loop = FakeLoop([])
    runner = make_subagent_runner(run_loop = loop, parent_tools = PARENT_TOOLS)
    _, result = _drain(runner({"task": "  "}))
    assert result.startswith("Error:")
    assert loop.calls == []

    def boom(*_args):
        raise RuntimeError("server down")
        yield

    runner = make_subagent_runner(run_loop = boom, parent_tools = PARENT_TOOLS)
    _, result = _drain(runner({"task": "x"}))
    assert result.startswith("Error:") and "server down" in result


def test_runner_stops_on_cancel():
    cancel = threading.Event()

    def loop(*_args):
        cancel.set()
        yield {"type": "content", "text": "partial"}

    runner = make_subagent_runner(run_loop = loop, parent_tools = PARENT_TOOLS, cancel_event = cancel)
    _, result = _drain(runner({"task": "x"}))
    assert result == "Error: sub-agent cancelled."


def test_safetensors_loop_routes_spawn_subagent_to_runner():
    turns = iter(
        [
            '<tool_call>{"name":"spawn_subagent","arguments":{"task":"compute","tools":["python"],"reasoning_effort":"high"}}</tool_call>',
            "Planner done.",
        ]
    )
    seen_tool_results = []

    def single_turn(messages, *, active_tools = None):
        seen_tool_results.extend(m["content"] for m in messages if m.get("role") == "tool")
        yield next(turns)

    executed = []

    def execute_tool(name, arguments, **_kwargs):
        executed.append(name)
        return "unused"

    sub_loop = FakeLoop([{"type": "content", "text": "42"}])
    runner = make_subagent_runner(run_loop = sub_loop, parent_tools = PARENT_TOOLS)
    tools = PARENT_TOOLS + [build_subagent_tool(PARENT_TOOLS)]

    events = list(
        run_safetensors_tool_loop(
            single_turn = single_turn,
            messages = [{"role": "user", "content": "plan it"}],
            tools = tools,
            execute_tool = execute_tool,
            max_tool_iterations = 3,
            subagent_runner = runner,
        )
    )

    assert executed == []
    assert sub_loop.calls[0]["tools"] == ["python"]
    assert sub_loop.calls[0]["effort"] == "high"
    tool_end = next(e for e in events if e["type"] == "tool_end")
    assert tool_end["tool_name"] == SUBAGENT_TOOL_NAME
    assert "Report:\n42" in tool_end["result"]
    assert any("Report:\n42" in r for r in seen_tool_results)
    assert any(
        e["type"] == "status" and e["text"].startswith("Sub-agent executor (high)") for e in events
    )
    assert any(e["type"] == "content" and e["text"] == "Planner done." for e in events)


def test_with_subagent_tool_requires_project_session_and_opt_in():
    from routes.inference import _add_planner_nudge, _with_subagent_tool

    on = SimpleNamespace(
        enabled_tools = ["python", SUBAGENT_TOOL_NAME],
        session_id = "project-p1",
        subagent_effort = None,
    )
    tools = _with_subagent_tool(on, PARENT_TOOLS)
    assert tools[-1]["function"]["name"] == SUBAGENT_TOOL_NAME
    assert "planner" in _add_planner_nudge("", tools)

    for payload in (
        SimpleNamespace(
            enabled_tools = [SUBAGENT_TOOL_NAME], session_id = "thread-1", subagent_effort = None
        ),
        SimpleNamespace(enabled_tools = ["python"], session_id = "project-p1", subagent_effort = None),
        SimpleNamespace(enabled_tools = None, session_id = "project-p1", subagent_effort = None),
    ):
        assert _with_subagent_tool(payload, PARENT_TOOLS) == PARENT_TOOLS
    assert _add_planner_nudge("base", PARENT_TOOLS) == "base"
