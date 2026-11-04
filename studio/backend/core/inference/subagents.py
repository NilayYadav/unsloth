# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import re
import threading
from typing import Any, Callable, Generator, Optional

from core.tool_healing import strip_tool_call_markup
from loggers import get_logger

logger = get_logger(__name__)

SUBAGENT_TOOL_NAME = "spawn_subagent"
SUBAGENT_EFFORTS = ("low", "medium", "high")
PROJECT_SESSION_PREFIX = "project-"

_EFFORT_TOOL_BUDGET = {"low": 6, "medium": 12, "high": 25}
_MAX_TASK_CHARS = 8000
_MAX_REPORT_CHARS = 12000
_MAX_TRACE_LINES = 20
_THINK_RE = re.compile(r"<think>.*?(?:</think>|\Z)", re.DOTALL)

_EXECUTOR_PROMPT = (
    "You are an executor sub-agent spawned by a planner agent inside an Unsloth "
    "Studio project. Complete only the task you are given. Use your tools when "
    "they help; you cannot spawn other agents. When finished, reply with a "
    "concise report of what you did and what you found, including file names, "
    "numbers or code the planner needs. Do not ask the planner questions."
)

PLANNER_NUDGE = (
    "You are the planner for this project. For multi-step work, break the task "
    f"into focused pieces and delegate them with {SUBAGENT_TOOL_NAME}. Give each "
    "executor a self-contained task, only the tools it needs, and an effort "
    "level: low for quick lookups or simple edits, high for hard reasoning or "
    "long multi-step work. Executors share this project's workspace and return "
    "a report; combine their reports into your final answer. Handle trivial "
    "requests yourself."
)

RunLoop = Callable[[list[dict], list[dict], str, int], Generator[dict, None, None]]


def is_project_session(session_id: Optional[str]) -> bool:
    return bool(session_id) and session_id.startswith(PROJECT_SESSION_PREFIX)


def _tool_name(tool: dict) -> str:
    return (tool.get("function") or {}).get("name") or ""


def scopable_tools(tools: list[dict]) -> list[dict]:
    return [t for t in tools if _tool_name(t) and _tool_name(t) != SUBAGENT_TOOL_NAME]


def build_subagent_tool(parent_tools: list[dict], fixed_effort: Optional[str] = None) -> dict:
    names = [_tool_name(t) for t in scopable_tools(parent_tools)]
    properties: dict[str, Any] = {
        "task": {
            "type": "string",
            "description": "Self-contained instructions for the executor, including any context it needs.",
        },
        "name": {
            "type": "string",
            "description": "Short label for this executor, e.g. 'researcher' or 'test-writer'.",
        },
        "tools": {
            "type": "array",
            "items": {"type": "string", "enum": names} if names else {"type": "string"},
            "description": (
                "Tools the executor may use (its tool scope). Omit or pass [] for a "
                f"reasoning-only executor. Available: {', '.join(names) or 'none'}."
            ),
        },
    }
    if fixed_effort is None:
        properties["reasoning_effort"] = {
            "type": "string",
            "enum": list(SUBAGENT_EFFORTS),
            "description": "How hard the executor should think and how many tool calls it gets. Default medium.",
        }
    return {
        "type": "function",
        "function": {
            "name": SUBAGENT_TOOL_NAME,
            "description": (
                "Spawn an executor sub-agent with its own tool scope and reasoning "
                "effort to complete one focused task. Returns the executor's report."
            ),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": ["task"],
            },
        },
    }


def _normalize_effort(value: Any, default: str) -> str:
    effort = str(value or "").strip().lower()
    return effort if effort in SUBAGENT_EFFORTS else default


def _requested_tool_names(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [part.strip() for part in value.split(",")]
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip() and item.strip() not in out:
            out.append(item.strip())
    return out


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "\n... (truncated)"


def make_subagent_runner(
    *,
    run_loop: RunLoop,
    parent_tools: list[dict],
    fixed_effort: Optional[str] = None,
    cancel_event: Optional[threading.Event] = None,
) -> Callable[[dict], Generator[dict, None, str]]:
    allowed = {_tool_name(t): t for t in scopable_tools(parent_tools)}
    fixed = fixed_effort if fixed_effort in SUBAGENT_EFFORTS else None

    def run(arguments: dict) -> Generator[dict, None, str]:
        task = str(arguments.get("task") or "").strip()
        if not task:
            return "Error: spawn_subagent requires a non-empty task."
        label = str(arguments.get("name") or "").strip()[:40] or "executor"
        effort = fixed or _normalize_effort(arguments.get("reasoning_effort"), "medium")
        requested = _requested_tool_names(arguments.get("tools"))
        granted = [name for name in requested if name in allowed]
        denied = [name for name in requested if name not in allowed]
        sub_tools = [allowed[name] for name in granted]

        messages = [
            {"role": "system", "content": _EXECUTOR_PROMPT},
            {"role": "user", "content": _clip(task, _MAX_TASK_CHARS)},
        ]
        prefix = f"Sub-agent {label} ({effort})"
        yield {"type": "status", "text": f"{prefix}: starting"}
        logger.info(
            "Spawning sub-agent %s effort=%s tools=%s denied=%s", label, effort, granted, denied
        )

        report = ""
        trace: list[str] = []
        try:
            for event in run_loop(messages, sub_tools, effort, _EFFORT_TOOL_BUDGET[effort]):
                if cancel_event is not None and cancel_event.is_set():
                    return "Error: sub-agent cancelled."
                kind = event.get("type")
                if kind == "content":
                    report = event.get("text") or ""
                elif kind == "status" and event.get("text"):
                    yield {"type": "status", "text": f"{prefix}: {event['text']}"}
                elif kind == "tool_end":
                    trace.append(str(event.get("tool_name") or "tool"))
        except Exception as exc:
            logger.exception("Sub-agent %s failed", label)
            return f"Error: sub-agent {label} failed: {exc}"

        report = (
            _THINK_RE.sub("", strip_tool_call_markup(report, final = True)).strip() or "(no report)"
        )
        header = [
            f"Sub-agent: {label}",
            f"Effort: {effort}",
            f"Tools: {', '.join(granted) or 'none'}",
        ]
        if denied:
            header.append(f"Denied tools (not in planner scope): {', '.join(denied)}")
        if trace:
            shown = trace[:_MAX_TRACE_LINES]
            more = f" (+{len(trace) - len(shown)} more)" if len(trace) > len(shown) else ""
            header.append(f"Tool calls: {', '.join(shown)}{more}")
        return "\n".join(header) + "\n\nReport:\n" + _clip(report, _MAX_REPORT_CHARS)

    return run
