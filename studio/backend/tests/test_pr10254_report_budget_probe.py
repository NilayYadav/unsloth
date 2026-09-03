# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""PR 10254 A/B probe: what output budget Deep Research asks a saved connection for.

Identical on both A/B branches. It drives the real ``ResearchSupervisor._research``
synthesis block against a stand-in connection that stops at whatever limit it is given,
and prints the real values the run produced.
"""

from __future__ import annotations

import asyncio
import json
import os
from types import SimpleNamespace

import pytest

from storage import research_runs_db as research_db
from storage import studio_db


# The stand-in connection's resolved per-model ceiling, and what the report needs to finish.
CONNECTION_CEILING = 32_768
REPORT_NEEDS = 24_000
OLD_FIXED_BUDGET = 16_384

FULL_REPORT = (
    "## Findings\n\n"
    + ("The evidence supports the conclusion. " * 200).strip()
    + "\n\n```python\ndef summarise(rows):\n    return len(rows)\n```\n\n## Conclusion\n\nDone."
)
# Where a model that ran out of budget actually stops: inside the fence.
TRUNCATED_REPORT = FULL_REPORT.split("```python\n")[0] + "```python\ndef summarise(rows):\n"

FACTS = {}


@pytest.fixture
def research_home(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    studio_db.upsert_chat_thread(
        {
            "id": "thread-1",
            "title": "Research",
            "modelType": "base",
            "modelId": "local-model",
            "createdAt": 1,
        }
    )
    studio_db.upsert_chat_message(
        {
            "id": "user-1",
            "threadId": "thread-1",
            "role": "user",
            "content": [{"type": "text", "text": "what happened?"}],
            "createdAt": 2,
        }
    )
    return tmp_path


def _claimed_run(supervisor) -> dict:
    research_db.create_run(
        run_id = "run-1",
        owner_subject = "alice",
        thread_id = "thread-1",
        user_message_id = "user-1",
        assistant_message_id = None,
        config = {
            "model": "local-model",
            # A saved connection whose model accepts far more than the old fixed number.
            "inferenceRequest": {
                "model": "local-model",
                "providerId": "conn-1",
                "providerType": "gemini",
                "externalModel": "gemini-3.6-flash",
                "maxOutputTokens": CONNECTION_CEILING,
            },
            "ragScope": None,
            "instructions": "",
            "question": "what happened?",
            "budgets": {
                "maxSteps": 1,
                "maxSources": 5,
                "modelTimeoutSeconds": 30,
                "toolTimeoutSeconds": 10,
            },
        },
    )
    planned = research_db.set_plan("run-1", {"title": "Plan", "steps": []})
    research_db.approve("run-1", planned["planRevision"], planned["planHash"])
    return research_db.claim_next(supervisor.worker_id)


def test_deep_research_asks_the_connection_for_the_report_budget(research_home, monkeypatch):
    from core import research_runs as worker

    supervisor = worker.ResearchSupervisor(
        SimpleNamespace(state = SimpleNamespace(server_port = 1))
    )
    claimed = _claimed_run(supervisor)
    calls: list[dict] = []

    async def stand_in_connection(run, messages, **kwargs):
        phase = kwargs.get("phase")
        if phase not in {"synthesis", "synthesis_recovery"}:
            # Unparseable, and an empty plan has no seed action, so the step loop breaks.
            return "not json", "", "stop", None
        asked = kwargs.get("max_tokens")
        # The number that would actually leave the backend on the wire for this call.
        wire = worker._resolve_max_tokens(asked, run["config"]["inferenceRequest"], messages)
        calls.append({"phase": phase, "asked": asked, "wire": wire})
        if wire >= REPORT_NEEDS:
            return FULL_REPORT, "", "stop", {
                "prompt_tokens": 5_000,
                "completion_tokens": REPORT_NEEDS,
            }
        return TRUNCATED_REPORT, "", "length", {
            "prompt_tokens": 5_000,
            "completion_tokens": wire,
        }

    monkeypatch.setattr(supervisor, "_stream_completion", stand_in_connection)
    asyncio.run(supervisor._research(claimed))

    finished = research_db.get_run("run-1")
    report = finished["report"]
    synthesis = next(call for call in calls if call["phase"] == "synthesis")

    FACTS.update(
        {
            "connection_ceiling": CONNECTION_CEILING,
            "report_needs_tokens": REPORT_NEEDS,
            "synthesis_max_tokens_asked": synthesis["asked"],
            "synthesis_max_tokens_on_wire": synthesis["wire"],
            "phases": [call["phase"] for call in calls],
            "recovery_ran": any(call["phase"] == "synthesis_recovery" for call in calls),
            "run_status": finished["status"],
            "report_chars": len(report),
            "incomplete_banner": "Incomplete report." in report,
            "report_first_line": report.splitlines()[0] if report else "",
            "report_last_line": report.rstrip().splitlines()[-1] if report else "",
        }
    )
    print("PR10254_SYNTHESIS_FACTS " + json.dumps(FACTS, sort_keys = True))

    assert finished["status"] == "completed"
    # The claim under test: the report is written against the connection's real ceiling,
    # so it is not cut off and carries no "Incomplete report." banner.
    assert synthesis["wire"] == CONNECTION_CEILING, (
        f"synthesis asked for {synthesis['wire']} tokens, not the connection's "
        f"{CONNECTION_CEILING}"
    )
    assert not FACTS["incomplete_banner"], "the report was delivered cut short"
    assert "## Conclusion" in report


def test_the_route_accepts_the_connection_ceiling():
    from routes.research_runs import CreateResearchRun, _sanitize_config

    payload = CreateResearchRun(
        threadId = "thread-1",
        userMessageId = "user-1",
        question = "what happened?",
        inferenceRequest = {"model": "local-model", "maxOutputTokens": CONNECTION_CEILING},
    )
    config = _sanitize_config(payload, {"modelId": "local-model"})
    resolved = config["inferenceRequest"].get("maxOutputTokens")
    print(f"PR10254_ROUTE_FACT maxOutputTokens={resolved!r}")
    assert resolved == CONNECTION_CEILING


def teardown_module(module):
    path = os.environ.get("PR10254_FACTS_OUT")
    if path and FACTS:
        with open(path, "w", encoding = "utf-8") as handle:
            json.dump(FACTS, handle, indent = 2, sort_keys = True)
