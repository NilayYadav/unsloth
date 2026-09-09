# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Throwaway CI evidence probe for unslothai/unsloth#10663.

Runs one Deep Research run whose every search step is throttled and PRINTS what the run
actually became. It asserts nothing about the fix, so it is green on both sides of the A/B
and the two logs can be compared value for value.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from storage import research_runs_db as research_db
from storage import studio_db


REPORT = "## Findings\n\nWritten from memory, because nothing came back."
THROTTLED = "Search failed: the search engines are rate limiting this machine."


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


def test_evidence_two_throttled_searches(research_home, monkeypatch, capsys):
    from core import research_runs as worker

    supervisor = worker.ResearchSupervisor(SimpleNamespace(state = SimpleNamespace(server_port = 1)))
    research_db.create_run(
        run_id = "run-1",
        owner_subject = "alice",
        thread_id = "thread-1",
        user_message_id = "user-1",
        assistant_message_id = None,
        config = {
            "model": "local-model",
            "inferenceRequest": {"model": "local-model"},
            "ragScope": None,
            "instructions": "",
            "question": "what happened?",
            "websitePolicy": None,
            "budgets": {
                "maxSteps": 2,
                "maxSources": 5,
                "modelTimeoutSeconds": 900,
                "toolTimeoutSeconds": 10,
            },
        },
    )
    plan_steps = [
        {"title": "Step 0", "query": "what happened 0"},
        {"title": "Step 1", "query": "what happened 1"},
    ]
    planned = research_db.set_plan("run-1", {"title": "Plan", "steps": plan_steps})
    research_db.approve("run-1", planned["planRevision"], planned["planHash"])
    claimed = research_db.claim_next(supervisor.worker_id)

    def fake_execute_tool(name, arguments, **kwargs):
        return THROTTLED

    async def fake_stream_completion(run, messages, **kwargs):
        if kwargs.get("phase") == "synthesis":
            return REPORT, "", "stop", None
        return "not json", "", "stop", None

    monkeypatch.setattr(worker, "execute_tool", fake_execute_tool)
    monkeypatch.setattr(supervisor, "_stream_completion", fake_stream_completion)
    asyncio.run(supervisor._process(claimed))

    finished = research_db.get_run("run-1")
    messages = studio_db.list_chat_messages("thread-1")
    assistant = [message for message in messages if message["role"] == "assistant"]
    assistant = assistant[-1] if assistant else {}

    with capsys.disabled():
        print("\n===== PR10663 EVIDENCE: every search step throttled =====")
        print(f"run.status                 = {finished['status']!r}")
        print(f"run.error                  = {str(finished.get('error') or '')[:300]!r}")
        print(f"run.report length          = {len(finished.get('report') or '')}")
        print(f"run.report                 = {str(finished.get('report') or '')[:300]!r}")
        print(f"run.sources count          = {len(finished.get('sources') or [])}")
        print(f"run.documentSources count  = {len(finished.get('documentSources') or [])}")
        print(f"chat.researchStatus        = {(assistant.get('metadata') or {}).get('researchStatus')!r}")
        print(f"chat.content               = {json.dumps(assistant.get('content'))[:300]}")
        delivered = bool(finished.get("report")) and finished["status"] == "completed"
        print(f"VERDICT                    = {'FABRICATED REPORT DELIVERED (bug present)' if delivered else 'RUN FAILED WITH THE SEARCH ERROR (fixed)'}")
        print("========================================================\n")
