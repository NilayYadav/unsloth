# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Drive the real Studio llama.cpp backend against a real llama-server and
report what the final-answer continuation actually put on the wire."""

import argparse
import contextlib
import copy
import json
import os
import sys
import threading

p = argparse.ArgumentParser()
p.add_argument("--backend-dir", required=True)
p.add_argument("--port", type=int, required=True)
p.add_argument("--ctx", type=int, required=True)
p.add_argument("--history", type=int, default=2)
p.add_argument("--out", required=True)
a = p.parse_args()

sys.path.insert(0, a.backend_dir)
os.chdir(a.backend_dir)

from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402

backend = LlamaCppBackend.__new__(LlamaCppBackend)
backend._process = object()
backend._healthy = True
backend._port = a.port
backend._api_key = None
backend._effective_context_length = a.ctx
backend._supports_reasoning = False
backend._reasoning_always_on = False
backend._reasoning_style = "enable_thinking"
backend._supports_preserve_thinking = False
backend._cancel_event = threading.Event()
# As studio/backend/tests/test_truncated_answer_continuation.py does: a crash
# recovery attempt here would answer a question this probe is not asking.
backend._maybe_recover_from_mtp_crash = lambda *_a, **_k: False

payloads: list = []
_real_stream = backend._stream_with_retry


@contextlib.contextmanager
def _recording_stream(client, url, payload, cancel_event, headers=None, first_token_deadline=None):
    payloads.append(copy.deepcopy(payload))
    with _real_stream(
        client, url, payload, cancel_event,
        headers=headers, first_token_deadline=first_token_deadline,
    ) as response:
        yield response


backend._stream_with_retry = _recording_stream

_FILLER = (
    "Earlier in this session we went through the bakery's inventory in detail: "
    "flour grades, hydration ratios, proofing times, oven temperatures, crumb "
    "structure, scoring patterns, starter feeding schedules and shaping technique. "
)


def _conversation():
    messages = []
    for i in range(a.history):
        messages.append({"role": "user", "content": f"Notes part {i}. " + _FILLER * 4})
        messages.append({"role": "assistant", "content": f"Recorded part {i}. " + _FILLER * 4})
    messages.append({
        "role": "user",
        "content": "Now write a long, detailed story about a dragon who learns to bake "
                   "bread. Write at least 3000 words. Do not stop early.",
    })
    return messages


cut_off_at = {"chars": None}
events: list = []
raised = None
try:
    for event in backend.generate_chat_completion_with_tools(
        messages=_conversation(),
        tools=None,
        temperature=0.0,
        seed=1234,
        max_tokens=None,
        max_tool_iterations=1,
        context_overflow="truncate_oldest",
        cancel_event=threading.Event(),
    ):
        events.append(event)
        if event.get("type") == "status" and event.get("text") and cut_off_at["chars"] is None:
            shown = [e["text"] for e in events if e.get("type") == "content"]
            cut_off_at["chars"] = len(shown[-1]) if shown else 0
except Exception as exc:  # the defect surfaces here
    raised = f"{type(exc).__name__}: {exc}"

shown = [e["text"] for e in events if e.get("type") == "content"]
final = shown[-1] if shown else ""

report = {
    "requests_to_llama_server": len(payloads),
    "request_flags": [
        {
            "n": i + 1,
            "last_message_role": (pl.get("messages") or [{}])[-1].get("role"),
            "continue_final_message": pl.get("continue_final_message"),
            "add_generation_prompt": pl.get("add_generation_prompt"),
        }
        for i, pl in enumerate(payloads)
    ],
    "answer_chars_when_cut_off": cut_off_at["chars"],
    "answer_chars_at_end": len(final),
    "answer_tail": final[-120:],
    "continuations_started": len([
        e for e in events
        if e.get("type") == "status" and e.get("text") == "Continuing the answer..."
    ]),
    "raised": raised,
}
with open(a.out, "w") as fh:
    json.dump(report, fh, indent=2)
print(json.dumps(report, indent=2))

failures = []
if report["requests_to_llama_server"] < 2:
    failures.append("the answer was never cut off, so the continuation path never ran")
if raised:
    failures.append(f"the turn ended in an error: {raised}")
if cut_off_at["chars"] is not None and len(final) <= (cut_off_at["chars"] or 0):
    failures.append("the answer never grew past where it was cut off")
for flags in report["request_flags"][1:]:
    if flags["continue_final_message"] and flags["add_generation_prompt"] is not False:
        failures.append(
            f"request {flags['n']} sent continue_final_message without "
            f"add_generation_prompt=false (llama-server rejects that)"
        )

if failures:
    print("\nFAIL")
    for line in failures:
        print(f"  - {line}")
    sys.exit(1)
print(
    f"\nPASS: cut off at {cut_off_at['chars']} chars, resumed "
    f"{report['continuations_started']}x, ended at {len(final)} chars"
)
