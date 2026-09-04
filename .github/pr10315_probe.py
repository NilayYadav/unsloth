"""PR 10315 A/B probe: does POST /v1/embeddings serve a vector when the resident
GGUF cannot? Runs against a live `unsloth studio`. SIDE=before|after decides the
assertion; both sides run byte-identical code."""
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PORT = os.environ["STUDIO_PORT"]
BASE = f"http://127.0.0.1:{PORT}"
SIDE = os.environ["SIDE"]
GGUF_REPO = os.environ["GGUF_REPO"]
GGUF_VARIANT = os.environ.get("GGUF_VARIANT", "UD-Q4_K_XL")
AUTH_DIR = Path(os.path.expanduser("~/.unsloth/studio/auth"))

results = {}


def http(method, path, body=None, headers=None, timeout=60):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method)
    req.add_header("content-type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8", "replace")
            try:
                return r.status, json.loads(raw)
            except Exception:
                return r.status, raw
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, raw
    except Exception as e:
        return 0, f"{type(e).__name__}: {e}"


def shape(body, n=300):
    s = body if isinstance(body, str) else json.dumps(body)
    return s[:n]


print("=" * 70, flush=True)
print(f"PR 10315 probe -- SIDE={SIDE}", flush=True)
print("=" * 70, flush=True)

# ---- auth: bootstrap login, then rotate for a durable bearer -------------
pw = (AUTH_DIR / ".bootstrap_password").read_text(encoding="utf-8").strip()
code, body = http("POST", "/api/auth/login", {"username": "unsloth", "password": pw})
assert code == 200 and isinstance(body, dict) and body.get("access_token"), f"bootstrap login -> {code}: {shape(body)}"
tok = body["access_token"]
NEW = "Pr10315-Probe-Pw!x9"
code, body = http("POST", "/api/auth/change-password",
                  {"current_password": pw, "new_password": NEW},
                  {"Authorization": f"Bearer {tok}"})
if code == 200:
    code, body = http("POST", "/api/auth/login", {"username": "unsloth", "password": NEW})
    assert code == 200 and body.get("access_token"), f"login after rotate -> {code}"
    tok = body["access_token"]
AUTH = {"Authorization": f"Bearer {tok}"}
print(f"[auth] bearer acquired (rotate -> {code})", flush=True)

# ---- Phase A: nothing loaded -------------------------------------------
code, body = http("GET", "/api/inference/status", headers=AUTH)
print(f"[phaseA] inference status -> {code}: {shape(body, 200)}", flush=True)

code, body = http("POST", "/v1/embeddings",
                  {"model": "text-embedding-3-small", "input": "hello world"},
                  AUTH, timeout=600)
results["no_model"] = code
dim_a = len(body["data"][0]["embedding"]) if code == 200 and isinstance(body, dict) and body.get("data") else 0
results["no_model_dim"] = dim_a
print(f"[phaseA] POST /v1/embeddings (no model loaded) -> {code} dim={dim_a}", flush=True)
print(f"[phaseA] body: {shape(body)}", flush=True)

# ---- Phase B: chat GGUF resident ---------------------------------------
code, body = http("POST", "/api/inference/load",
                  {"model_path": GGUF_REPO, "gguf_variant": GGUF_VARIANT,
                   "is_lora": False, "max_seq_length": 2048},
                  AUTH, timeout=900)
print(f"[phaseB] load {GGUF_REPO} -> {code}: {shape(body, 200)}", flush=True)
results["load"] = code
if code != 200:
    print("HARNESS: model load failed; phase B is inconclusive", flush=True)

code, body = http("POST", "/v1/embeddings",
                  {"model": "text-embedding-3-small", "input": "hello world"},
                  AUTH, timeout=600)
results["chat_loaded"] = code
dim_b = len(body["data"][0]["embedding"]) if code == 200 and isinstance(body, dict) and body.get("data") else 0
results["chat_loaded_dim"] = dim_b
print(f"[phaseB] POST /v1/embeddings (chat GGUF resident) -> {code} dim={dim_b}", flush=True)
print(f"[phaseB] body: {shape(body)}", flush=True)

# chat must still work right after embeddings -- the PR claims it never touches the chat slot
code, body = http("POST", "/v1/chat/completions",
                  {"model": GGUF_REPO, "messages": [{"role": "user", "content": "Say hi"}],
                   "max_tokens": 8, "stream": False},
                  AUTH, timeout=600)
results["chat_after"] = code
print(f"[phaseB] POST /v1/chat/completions after embeddings -> {code}: {shape(body, 200)}", flush=True)

print("=" * 70, flush=True)
print("RESULT " + json.dumps(results, sort_keys=True), flush=True)
print("=" * 70, flush=True)

# ---- assertions ---------------------------------------------------------
failures = []
if SIDE == "before":
    if results["no_model"] == 200:
        failures.append(f"expected non-200 with no model on main, got 200")
    if results["load"] == 200 and results["chat_loaded"] == 200:
        failures.append("expected non-200 with a chat GGUF resident on main, got 200")
    print(f"NEGATIVE-SIDE: no_model={results['no_model']} chat_loaded={results['chat_loaded']} "
          f"(both non-200 = defect reproduced)", flush=True)
else:
    if results["no_model"] != 200 or results["no_model_dim"] <= 0:
        failures.append(f"no model loaded: expected 200 + vector, got {results['no_model']} dim={results['no_model_dim']}")
    if results["load"] == 200:
        if results["chat_loaded"] != 200 or results["chat_loaded_dim"] <= 0:
            failures.append(f"chat GGUF resident: expected 200 + vector, got {results['chat_loaded']} dim={results['chat_loaded_dim']}")
        if results["chat_after"] != 200:
            failures.append(f"chat completions broken after embeddings: {results['chat_after']}")

if failures:
    for f in failures:
        print(f"FAIL: {f}", flush=True)
    sys.exit(1)
print(f"PASS: {SIDE} side behaved as expected", flush=True)
