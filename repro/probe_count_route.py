#!/usr/bin/env python3
"""PR 10255 second probe: /chat/count_tokens refuses audio. Does that survive the new part?

The route guards images at the content-part level but audio only through `audio_base64`.
That was safe only while an `input_audio` part could not validate at all. Identical on every leg.
"""
import json, os, sys

AUDIO_B64 = "UklGRiQAAABXQVZF"
BODY = {
    "model": "default",
    "messages": [{"role": "user", "content": [
        {"type": "text", "text": "how many tokens?"},
        {"type": "input_audio", "input_audio": {"data": AUDIO_B64, "format": "wav"}},
    ]}],
}

from fastapi import FastAPI
from fastapi.testclient import TestClient
import routes.inference as ri
from auth.authentication import get_current_subject

app = FastAPI()
app.include_router(ri.router)
app.dependency_overrides[get_current_subject] = lambda: "probe"
with TestClient(app, raise_server_exceptions = False) as client:
    r = client.post("/chat/count_tokens", json = BODY)

body = r.text[:900]
print(f"[count] REAL ROUTE POST /chat/count_tokens (text + input_audio) -> HTTP {r.status_code}")
print(f"[count] body: {body}")

rejected_at_validation = r.status_code == 422 and "union_tag_invalid" in body
audio_guard_fired = r.status_code == 503 and "messages containing audio" in body
got_past_the_guard = not rejected_at_validation and not audio_guard_fired

if rejected_at_validation:
    verdict = "REJECTED AT VALIDATION - the part cannot exist, so the guard never has to fire"
elif audio_guard_fired:
    verdict = "GUARD FIRED - the route refuses the audio part exactly as it refuses audio_base64"
else:
    verdict = "GUARD BYPASSED - the request carries audio and the audio refusal did not fire"
print(f"[count] {verdict}")

out = os.environ.get("PROBE_JSON_COUNT")
if out:
    with open(out, "w") as fh:
        json.dump({"status": r.status_code, "body": body, "verdict": verdict}, fh, indent = 2)

mode = sys.argv[1] if len(sys.argv) > 1 else "report"
expected = {
    "expect-rejected": rejected_at_validation,
    "expect-bypassed": got_past_the_guard,
    "expect-guarded": audio_guard_fired,
}
if mode in expected:
    sys.exit(0 if expected[mode] else 1)
