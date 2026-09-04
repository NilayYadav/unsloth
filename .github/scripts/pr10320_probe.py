# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request

BASE = os.environ["UNSLOTH_STUDIO_URL"].rstrip("/")
KEY = os.environ["UNSLOTH_API_KEY"]
CHAT_MODEL = os.environ.get("UNSLOTH_MODEL_ID", "")
EMBED_MODEL = "unsloth/bge-small-en-v1.5"
result = {}
failures = []


def call(method, url, body = None, key = KEY, timeout = 300):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data = data, method = method,
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout = timeout) as r:
            return r.status, json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode() or "{}")
        except Exception:
            return e.code, {}


def check(name, ok, detail):
    result[name] = detail
    print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}")
    if not ok:
        failures.append(name)


def redact(text):
    return text.replace(KEY, "<REDACTED>")


status = call("GET", f"{BASE}/api/inference/status")[1]
loaded = status.get("loaded") or []
check("chat_gguf_loaded", bool(loaded), loaded)

env = {**os.environ, "OPENAI_API_KEY": "sk-probe-canary-0123456789"}
proc = subprocess.run(
    ["unsloth", "start", "openclaw", "--no-launch", "--api-key", KEY],
    capture_output = True, text = True, env = env, timeout = 600,
)
print(redact(proc.stdout)[-2500:])
check("unsloth_start_exit", proc.returncode == 0, proc.returncode)
m = re.search(r"^export OPENCLAW_CONFIG_PATH=(\S+)", proc.stdout, re.M)
config_path = m.group(1).strip("'\"") if m else ""
check("config_path_printed", bool(config_path), config_path)
search = {}
if config_path and os.path.exists(config_path):
    search = json.load(open(config_path)).get("memory", {}).get("search", {})
shown = {**search, "remote": {**search.get("remote", {}), "apiKey": "<present>" if search.get("remote", {}).get("apiKey") else "<missing>"}}
print("memory.search =", json.dumps(shown))
check("memory_provider", search.get("provider") == "openai-compatible", search.get("provider"))
check("memory_model", search.get("model") == EMBED_MODEL, search.get("model"))
check("memory_fallback", search.get("fallback") == "none", search.get("fallback"))
check("memory_base_url", search.get("remote", {}).get("baseUrl") == f"{BASE}/v1", search.get("remote", {}).get("baseUrl"))
check("memory_api_key_present", search.get("remote", {}).get("apiKey", "").startswith("sk-unsloth-"), "sk-unsloth-* present" if search.get("remote", {}).get("apiKey") else "missing")
result["recipe_unsets_openai_key"] = "unset OPENAI_API_KEY" in proc.stdout
print("INFO recipe_unsets_openai_key (needs #10316, not in this stack):", result["recipe_unsets_openai_key"])

embed_url = search.get("remote", {}).get("baseUrl", f"{BASE}/v1").rstrip("/") + "/embeddings"
embed_key = search.get("remote", {}).get("apiKey") or KEY
embed_model = search.get("model") or EMBED_MODEL
code, body = call("POST", embed_url, {"model": embed_model, "input": ["hello world"]}, key = embed_key)
dim = len(body["data"][0]["embedding"]) if code == 200 and body.get("data") else 0
reported = body.get("model") if code == 200 else body.get("error", {}).get("message", "")
check("embeddings_from_config", code == 200 and dim == 384 and EMBED_MODEL in str(reported), f"HTTP {code} dim={dim} model={reported}")

code, body = call("POST", f"{BASE}/v1/embeddings", {"model": f"{EMBED_MODEL}-GGUF", "input": "x"})
check("embeddings_by_gguf_name", code == 200, f"HTTP {code}")
for words, want in ((700, 400), (510, 200), (511, 400)):
    code, body = call("POST", f"{BASE}/v1/embeddings", {"model": "default", "input": " ".join(["token"] * words)})
    msg = body.get("error", {}).get("message", "") if code != 200 else "ok"
    ok = code == want and (want == 200 or "510-token limit" in msg)
    check(f"embeddings_{words}_words", ok, f"HTTP {code} {msg[:80]}")

code, body = call("POST", f"{BASE}/v1/chat/completions", {"model": CHAT_MODEL, "messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 8, "stream": False})
check("chat_control", code == 200, f"HTTP {code}")
after = call("GET", f"{BASE}/api/inference/status")[1].get("loaded") or []
check("chat_gguf_still_loaded", after == loaded, after)

print("RESULT_JSON " + json.dumps(result))
sys.exit(1 if failures else 0)
