#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""PR 10314 repro probe: /v1/messages tool handling on a GGUF whose chat
template does not advertise tools (gemma-3-270m-it).

Asserts the contract the PR is meant to establish. Identical bytes on every
branch; only the implementation under test differs.
"""
import json
import os
import sys
import urllib.error
import urllib.request

PORT = os.environ["STUDIO_PORT"]
TOKEN = os.environ["API_KEY"]
BASE = f"http://127.0.0.1:{PORT}"
OUT = os.environ.get("PROBE_OUT", "probe-result.json")

TOOL_ANTHROPIC = {
    "name": "get_weather",
    "description": "Get the current weather in a city.",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}
TOOL_OPENAI = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}
ASK = "What is the weather in Paris? Use the get_weather tool."

# A replayed client tool loop: assistant tool_use + user tool_result, no catalog.
HISTORY = [
    {"role": "user", "content": "What is the weather in Paris?"},
    {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_probe1",
                "name": "get_weather",
                "input": {"city": "Paris"},
            }
        ],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_probe1",
                "content": "18C and sunny",
            }
        ],
    },
    # The realistic continuation: the client asks the model to use the tool result.
    {"role": "user", "content": "Given that result, what is the weather in Paris?"},
]


def post(path, body):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode(),
        headers={
            "content-type": "application/json",
            "Authorization": f"Bearer {TOKEN}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, {"_raw": raw[:2000]}


def get(path):
    req = urllib.request.Request(
        BASE + path, headers={"Authorization": f"Bearer {TOKEN}"}
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, {"_raw": e.read().decode()[:2000]}


def msg(**fields):
    body = {"model": "probe", "max_tokens": 64, "messages": [{"role": "user", "content": ASK}]}
    body.update(fields)
    return body


def has_tool_use(payload):
    for block in (payload or {}).get("content") or []:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            return True
    return False


def text_of(payload):
    parts = []
    for block in (payload or {}).get("content") or []:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text") or "")
    return "".join(parts)


def err_message(payload):
    if not isinstance(payload, dict):
        return ""
    err = payload.get("error")
    if isinstance(err, dict):
        return err.get("message") or ""
    detail = payload.get("detail")
    if isinstance(detail, dict):
        inner = detail.get("error")
        if isinstance(inner, dict):
            return inner.get("message") or ""
    return json.dumps(payload)[:400]


def err_type(payload):
    if not isinstance(payload, dict):
        return None
    err = payload.get("error")
    if isinstance(err, dict):
        return err.get("type")
    detail = payload.get("detail")
    if isinstance(detail, dict):
        inner = detail.get("error")
        if isinstance(inner, dict):
            return inner.get("type")
    return None


results = {}
failures = []


def record(name, status, payload, extra=None):
    entry = {"http_status": status}
    if extra:
        entry.update(extra)
    if status >= 400:
        entry["error_type"] = err_type(payload)
        entry["error_message"] = err_message(payload)
    else:
        entry["has_tool_use_block"] = has_tool_use(payload)
        entry["text"] = text_of(payload)[:300]
        entry["stop_reason"] = payload.get("stop_reason")
    results[name] = entry
    return entry


def check(name, condition, expected, actual):
    line = {"assertion": name, "expected": expected, "actual": actual,
            "result": "PASS" if condition else "FAIL"}
    results.setdefault("_assertions", []).append(line)
    print(f"[{line['result']}] {name}: expected {expected}; actual {actual}", flush=True)
    if not condition:
        failures.append(name)


# ── R0: what the loaded model actually advertises ─────────────────────────
st, models = get("/v1/models")
results["R0_models"] = {"http_status": st, "body": models}
print("R0 /v1/models:", json.dumps(models)[:600], flush=True)

# ── R1: client tool catalogue on a template with no tool support ──────────
st, body = post("/v1/messages", msg(tools=[TOOL_ANTHROPIC]))
e = record("R1_messages_with_tool_catalog", st, body)
check(
    "R1 /v1/messages with a client tool catalogue is rejected, not silently answered in prose",
    st == 400 and "does not advertise tools" in (e.get("error_message") or ""),
    "HTTP 400 with 'does not advertise tools'",
    f"HTTP {st} err={e.get('error_message', '')[:120]!r} "
    f"tool_use={e.get('has_tool_use_block')} text={e.get('text', '')[:80]!r}",
)

# ── R2: tool_choice none withdraws the catalogue, so it must still answer ─
st, body = post(
    "/v1/messages", msg(tools=[TOOL_ANTHROPIC], tool_choice={"type": "none"})
)
e = record("R2_messages_tool_choice_none", st, body)
check(
    "R2 /v1/messages with tool_choice none still answers",
    st == 200,
    "HTTP 200",
    f"HTTP {st} {e.get('error_message', '')[:120]}",
)

# ── R3: replayed tool history, no catalogue -> folded into user text ──────
st, body = post("/v1/messages", msg(messages=HISTORY))
e = record("R3_messages_tool_history_no_catalog", st, body)
check(
    "R3 /v1/messages with replayed tool history and no catalogue still answers "
    "(fold_tool_results_into_user path)",
    st == 200 and bool(e.get("text")),
    "HTTP 200 with non-empty text",
    f"HTTP {st} text={e.get('text', '')[:100]!r} {e.get('error_message', '')[:120]}",
)

# ── R4: a plain request is untouched ──────────────────────────────────────
st, body = post("/v1/messages", msg())
e = record("R4_messages_plain", st, body)
check(
    "R4 /v1/messages without tools still answers",
    st == 200 and bool(e.get("text")),
    "HTTP 200 with non-empty text",
    f"HTTP {st} text={e.get('text', '')[:80]!r}",
)

# ── R5: control -- the sibling OpenAI route already rejects this ──────────
st, body = post(
    "/v1/chat/completions",
    {
        "model": "probe",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": ASK}],
        "tools": [TOOL_OPENAI],
    },
)
results["R5_chat_completions_with_tool_catalog"] = {
    "http_status": st,
    "error_message": err_message(body) if st >= 400 else None,
    "body": None if st >= 400 else json.dumps(body)[:400],
}
check(
    "R5 control: /v1/chat/completions already rejects the same catalogue",
    st == 400,
    "HTTP 400",
    f"HTTP {st}",
)

results["_summary"] = {
    "failures": failures,
    "passed": len(results.get("_assertions", [])) - len(failures),
    "total": len(results.get("_assertions", [])),
}

with open(OUT, "w") as fh:
    json.dump(results, fh, indent=2)

print("\n=== PROBE JSON ===")
print(json.dumps(results, indent=2)[:8000])
print("=== END PROBE JSON ===")

if failures:
    print(f"\nPROBE FAILED on: {', '.join(failures)}", flush=True)
    sys.exit(1)
print("\nPROBE PASSED: all assertions held.", flush=True)
