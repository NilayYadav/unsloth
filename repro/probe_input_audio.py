#!/usr/bin/env python3
"""PR 10255 repro probe: does OpenAI's documented `input_audio` content part reach the audio path?

Identical on both A/B sides. Run with cwd = studio/backend and PYTHONPATH=.

The OpenAI SDK documents an audio turn as
    {"type": "input_audio", "input_audio": {"data": <b64>, "format": "wav"}}

Step 1 posts exactly that to the REAL /v1/chat/completions router through a TestClient and
prints the real HTTP status and body.
Step 2 checks the same payload at the model layer and, where the symbol exists, that the part
is lifted onto `audio_base64` (the field every downstream audio check reads).
"""

import json
import os
import sys

AUDIO_B64 = "UklGRiQAAABXQVZF"  # a real RIFF/WAVE header prefix

PAYLOAD = {
    "model": "local",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is said here?"},
                {"type": "input_audio", "input_audio": {"data": AUDIO_B64, "format": "wav"}},
            ],
        }
    ],
    "max_tokens": 8,
}

result = {"step1": {}, "step2": {}}


def step1_real_route():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import routes.inference as ri
    from auth.authentication import get_current_subject

    app = FastAPI()
    app.include_router(ri.router, prefix = "/v1")
    app.dependency_overrides[get_current_subject] = lambda: "probe"

    with TestClient(app, raise_server_exceptions = False) as client:
        r = client.post("/v1/chat/completions", json = PAYLOAD)
    body = r.text[:1200]
    result["step1"] = {
        "status": r.status_code,
        "body": body,
        "union_tag_invalid": "union_tag_invalid" in body,
    }
    print(f"[step1] REAL ROUTE POST /v1/chat/completions -> HTTP {r.status_code}")
    print(f"[step1] body: {body}")


def step2_model_layer():
    from models.inference import ChatCompletionRequest

    try:
        req = ChatCompletionRequest(**PAYLOAD)
    except Exception as exc:
        result["step2"] = {"validated": False, "error_type": type(exc).__name__, "error": str(exc)[:900]}
        print(f"[step2] ChatCompletionRequest REJECTED the part: {type(exc).__name__}")
        print(f"[step2] {str(exc)[:900]}")
        return

    part_types = [getattr(p, "type", None) for p in req.messages[0].content]
    result["step2"] = {"validated": True, "part_types_before": part_types}
    print(f"[step2] validated; content part types = {part_types}")

    import routes.inference as ri

    normalise = getattr(ri, "_normalise_chat_content_parts", None)
    if normalise is None:
        result["step2"]["lift"] = "absent"
        print("[step2] no _normalise_chat_content_parts on this tree")
        return
    normalise(req)
    result["step2"]["lift"] = "present"
    result["step2"]["audio_base64"] = req.audio_base64
    result["step2"]["part_types_after"] = [getattr(p, "type", None) for p in req.messages[0].content]
    print(f"[step2] after lift: audio_base64 = {req.audio_base64!r}")
    print(f"[step2] after lift: content part types = {result['step2']['part_types_after']}")


step1_real_route()
step2_model_layer()

out = os.environ.get("PROBE_JSON")
if out:
    with open(out, "w") as fh:
        json.dump(result, fh, indent = 2)

# --- verdict -------------------------------------------------------------------
mode = sys.argv[1] if len(sys.argv) > 1 else "report"
s1, s2 = result["step1"], result["step2"]
print()
if mode == "expect-broken":
    ok = s1["status"] == 422 and s1["union_tag_invalid"] and not s2.get("validated")
    print("REPRO CONFIRMED: the documented part is rejected before any model runs" if ok
          else "REPRO NOT CONFIRMED")
    sys.exit(0 if ok else 1)
if mode == "expect-fixed":
    ok = (
        s1["status"] != 422
        and not s1["union_tag_invalid"]
        and s2.get("validated")
        and s2.get("audio_base64") == AUDIO_B64
        and s2.get("part_types_after") == ["text"]
    )
    print("FIX CONFIRMED: the part validates, reaches the route, and lands on audio_base64" if ok
          else "FIX NOT CONFIRMED")
    sys.exit(0 if ok else 1)
