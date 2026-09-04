#!/usr/bin/env python3
"""PR 10255: every content-part shape the routes now answer for, with the real status.

Identical on every leg. Run with cwd = studio/backend and PYTHONPATH=.
"""
import sys
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
import routes.inference as ri
from auth.authentication import get_current_subject

AUD = {"type": "input_audio", "input_audio": {"data": "UklGRiQAAABXQVZF", "format": "wav"}}
TXT = {"type": "text", "text": "what is said here?"}

app = FastAPI()
app.include_router(ri.router)
app.include_router(ri.router, prefix = "/v1")
app.dependency_overrides[get_current_subject] = lambda: "probe"


def post(path, body):
    with TestClient(app, raise_server_exceptions = False) as client:
        r = client.post(path, json = body)
    detail = r.text
    for key in ('"message":"', '"detail":"'):
        if key in detail:
            detail = detail.split(key, 1)[1].split('"', 1)[0]
            break
    return r.status_code, detail[:95]


def msg(content, role = "user"):
    return {"role": role, "content": content}


CASES = [
    ("local: text + audio",       "/v1/chat/completions", {"model": "local", "messages": [msg([TXT, AUD])]}),
    ("external: text + audio",    "/v1/chat/completions", {"model": "gpt-4o", "provider_type": "openai", "messages": [msg([TXT, AUD])]}),
    ("external: unknown part",    "/v1/chat/completions", {"model": "gpt-4o", "provider_type": "openai", "messages": [msg([TXT, {"type": "file", "file": {"file_id": "f1"}}])]}),
    ("audio on assistant turn",   "/v1/chat/completions", {"model": "local", "messages": [msg("hi"), msg([TXT, AUD], role = "assistant"), msg("and?")]}),
    ("two recordings",            "/v1/chat/completions", {"model": "local", "messages": [msg([TXT, AUD, {"type": "input_audio", "input_audio": {"data": "c2Vjb25k", "format": "wav"}}])]}),
    ("empty audio data",          "/v1/chat/completions", {"model": "local", "messages": [msg([TXT, {"type": "input_audio", "input_audio": {"data": "", "format": "wav"}}])]}),
    ("malformed part type",       "/v1/chat/completions", {"model": "local", "messages": [msg([{"type": [{"a": 1}], "x": 1}])]}),
    ("count_tokens: audio part",  "/chat/count_tokens",   {"model": "default", "messages": [msg([TXT, AUD])]}),
    ("TTS: audio part",           "/audio/generate",      {"model": "default", "messages": [msg([TXT, AUD])]}),
]

print(f"{'case':28s} {'status':>6s}  detail")
print("-" * 100)
results = {}
for label, path, body in CASES:
    status, detail = post(path, body)
    results[label] = status
    print(f"{label:28s} {status:>6d}  {detail}")

# durable runs are not an HTTP surface here; drive the sanitizer directly
try:
    from routes.chat_generation_runs import CreateChatGenerationRun, _sanitize_request
    run = CreateChatGenerationRun(
        runId = "r", threadId = "t", userMessageId = "u", assistantMessageId = "a",
        requestPayload = {"model": "default", "messages": [msg([TXT, AUD])]},
    )
    try:
        out = _sanitize_request(run)
        results["durable run: audio part"] = 202
        print(f"{'durable run: audio part':28s} {202:>6d}  queued; blob persisted: {'UklGRiQAAABXQVZF' in str(out)}")
    except HTTPException as exc:
        results["durable run: audio part"] = exc.status_code
        print(f"{'durable run: audio part':28s} {exc.status_code:>6d}  {str(exc.detail)[:95]}")
except Exception as exc:
    print(f"{'durable run: audio part':28s} {'n/a':>6s}  {type(exc).__name__}: {exc}")

mode = sys.argv[1] if len(sys.argv) > 1 else "report"
if mode == "expect-final":
    want = {
        "external: text + audio": 400, "external: unknown part": 400,
        "audio on assistant turn": 400, "two recordings": 400,
        "empty audio data": 422, "malformed part type": 422,
        "count_tokens: audio part": 503, "TTS: audio part": 400,
        "durable run: audio part": 400,
    }
    bad = {k: (results.get(k), v) for k, v in want.items() if results.get(k) != v}
    print()
    if bad:
        print("MISMATCH:", bad)
        sys.exit(1)
    if results["local: text + audio"] == 422:
        print("MISMATCH: the local path still rejects the documented part")
        sys.exit(1)
    print("ALL REFUSALS AS INTENDED, and the local path still takes the part")
