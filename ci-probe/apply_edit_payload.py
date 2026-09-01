# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hands the PUT body the frontend just built to the real studio_db.

Seeds the thread the way the durable generation path leaves it -- a completed run whose
assistant row is settled and carries the length-limit note, its timing and its context usage --
then replays the captured edit through the same entry point the route uses.
"""

import json
import os
import sys
import tempfile

os.environ["UNSLOTH_STUDIO_HOME"] = tempfile.mkdtemp(prefix = "pr10161_")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "studio", "backend"))

from storage import chat_generation_runs_db as runs_db  # noqa: E402
from storage import studio_db  # noqa: E402

CREATED_AT = int(os.environ["PROBE_CREATED_AT"])
DISPLAY = {
    "timing": {"tokensPerSecond": 42.5, "durationMs": 1200},
    "contextUsage": {"promptTokens": 900, "contextLength": 4096},
}


def seed():
    studio_db.upsert_chat_thread(
        {"id": "thread-1", "title": "Chat", "modelType": "base", "modelId": "local", "createdAt": 1}
    )
    studio_db.upsert_chat_message(
        {
            "id": "user-1",
            "threadId": "thread-1",
            "role": "user",
            "content": [{"type": "text", "text": "hello"}],
            "createdAt": 2,
        }
    )
    # The placeholder the composer puts down before the run claims it, so the turn carries the
    # createdAt the captured PUT body was built with.
    studio_db.upsert_chat_message(
        {
            "id": "assistant-1",
            "threadId": "thread-1",
            "parentId": "user-1",
            "role": "assistant",
            "content": [{"type": "text", "text": ""}],
            "createdAt": CREATED_AT,
        }
    )
    runs_db.create_run(
        run_id = "run-1",
        owner_subject = "alice",
        thread_id = "thread-1",
        user_message_id = "user-1",
        assistant_message_id = "assistant-1",
        request_payload = {"model": "local", "messages": [], "stream": True},
    )
    token = runs_db.get_worker_token("run-1")
    assert runs_db.mark_running("run-1", token)
    seq = runs_db.append_events("run-1", token, [("chunk", {"i": 1})])[-1]
    streaming = studio_db.get_chat_message("thread-1", "assistant-1")
    streaming["content"] = [{"type": "text", "text": "the original reply"}]
    streaming["metadata"] = {
        **streaming["metadata"],
        "generationSeq": seq,
        "generationStatus": "running",
        **DISPLAY,
    }
    studio_db.upsert_chat_message(streaming)
    assert runs_db.finish_run(
        "run-1", worker_token = token, status = "completed", finish_reason = "length"
    )
    run = runs_db.get_run("run-1", "alice")
    settled = studio_db.get_chat_message("thread-1", "assistant-1")
    settled["metadata"] = {
        **settled["metadata"],
        "generationSeq": run["lastEventSeq"],
        "generationStatus": "completed",
        "generationSettled": True,
    }
    studio_db.upsert_chat_message(settled)
    return studio_db.get_chat_message("thread-1", "assistant-1")


def main():
    stored = seed()
    print("stored before the edit:", json.dumps(stored["metadata"], sort_keys = True))
    assert stored["metadata"]["incomplete"] == {"reason": "length"}, stored["metadata"]

    with open(sys.argv[1], encoding = "utf-8") as handle:
        payload = json.load(handle)
    print("PUT body from the frontend:", json.dumps(payload, sort_keys = True))

    try:
        saved = studio_db.upsert_chat_message(payload, allow_generation_edit = True)
    except studio_db.ChatMessageProtectedError as error:
        print(f"FAIL: the route answers 409 and the edit is rolled back -- {error}")
        return 1

    problems = []
    if saved["content"] != [{"type": "text", "text": "an edited reply"}]:
        problems.append(f"content was not rewritten: {saved['content']}")
    metadata = saved.get("metadata") or {}
    if metadata.get("incomplete") != {"reason": "length"}:
        problems.append(f"the length-limit note was lost: {metadata.get('incomplete')}")
    for key, value in DISPLAY.items():
        if metadata.get(key) != value:
            problems.append(f"{key} was lost: {metadata.get(key)}")
    if runs_db.get_run("run-1", "alice") is not None:
        problems.append("the generation run was not detached")
    if problems:
        for problem in problems:
            print(f"FAIL: {problem}")
        return 1

    print("saved metadata:", json.dumps(metadata, sort_keys = True))
    print("PASS: the edit saved, the run detached, and the reply kept its details")
    return 0


if __name__ == "__main__":
    sys.exit(main())
