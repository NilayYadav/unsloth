"""Run the real llama-server against the GLM-5.3 template, before and after the repair.

BEFORE is the template exactly as unsloth/GLM-5.3-GGUF embeds it, which is what
llama-server received before this PR. Both legs use the same binary, the same tiny
model and the same request: a conversation that replays an assistant tool call and
its result, which is the turn the bug breaks.
"""

import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "studio" / "backend"))

from core.inference.chat_template_helpers import repair_numeric_member_access

SERVER, MODEL, OUT = sys.argv[1], sys.argv[2], Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)

REQUEST = {
    "model": "x",
    "max_tokens": 8,
    "messages": [
        {"role": "user", "content": "What is the weather in Paris?"},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
        }]},
        {"role": "tool", "tool_call_id": "call_1", "content": "18C and sunny"},
        {"role": "user", "content": "Thanks, summarise."},
    ],
    "tools": [{"type": "function", "function": {
        "name": "get_weather", "description": "Weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                       "required": ["city"]},
    }}],
}


def post(url, payload=None, timeout=30):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"} if data else {}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode()
        try:
            return exc.code, json.loads(body)
        except ValueError:
            return exc.code, {"raw": body[:400]}


def leg(label, template, port):
    path = OUT / f"{label}.jinja"
    path.write_text(template, encoding="utf-8")
    log = open(OUT / f"{label}-server.log", "w")
    proc = subprocess.Popen(
        [SERVER, "--model", MODEL, "--jinja", "--chat-template-file", str(path),
         "--port", str(port), "--host", "127.0.0.1", "-c", "4096", "--no-webui"],
        stdout=log, stderr=subprocess.STDOUT, preexec_fn=os.setsid,
    )
    try:
        for _ in range(90):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read()
                break
            except Exception:
                if proc.poll() is not None:
                    raise RuntimeError(f"{label}: llama-server exited rc={proc.returncode}")
                time.sleep(1)
        else:
            raise RuntimeError(f"{label}: llama-server never became healthy")

        _, props = post(f"http://127.0.0.1:{port}/props")
        caps = props.get("chat_template_caps", {})
        status, body = post(f"http://127.0.0.1:{port}/v1/chat/completions", REQUEST)
        (OUT / f"{label}-response.json").write_text(json.dumps(body, indent=1))
        return caps, status, body
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=20)
        except Exception:
            proc.kill()


failures = []


def expect(label, got, want):
    ok = got == want
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: {got!r} (expected {want!r})")
    if not ok:
        failures.append(label)


url = "https://huggingface.co/api/models/unsloth/GLM-5.3-GGUF?expand%5B%5D=gguf"
for attempt in range(4):
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            raw = json.load(r)["gguf"]["chat_template"]
        break
    except Exception as exc:
        if attempt == 3:
            raise
        print(f"  retrying model metadata after {exc}")
        time.sleep(5 * (attempt + 1))

fixed = repair_numeric_member_access(raw)

print("== BEFORE: llama-server with the template as unsloth/GLM-5.3-GGUF ships it ==")
caps, status, body = leg("before", raw, 18941)
print(f"  caps: {json.dumps(caps, sort_keys=True)}")
print(f"  HTTP {status}: {json.dumps(body)[:400]}")
expect("BEFORE supports_tools", caps.get("supports_tools"), False)
expect("BEFORE supports_object_arguments", caps.get("supports_object_arguments"), False)
expect("BEFORE the tool turn is rejected", status, 400)
expect("BEFORE the template's arguments.items() is what failed",
       "hint: 'items'" in json.dumps(body), True)

print()
print("== AFTER: same binary, same request, template through the repair ==")
caps, status, body = leg("after", fixed, 18942)
print(f"  caps: {json.dumps(caps, sort_keys=True)}")
print(f"  HTTP {status}: {json.dumps(body)[:300]}")
expect("AFTER supports_tools", caps.get("supports_tools"), True)
expect("AFTER supports_tool_calls", caps.get("supports_tool_calls"), True)
expect("AFTER supports_parallel_tool_calls", caps.get("supports_parallel_tool_calls"), True)
expect("AFTER supports_object_arguments", caps.get("supports_object_arguments"), True)
expect("AFTER the tool turn is answered", status, 200)
expect("AFTER no template error", "error" in body, False)

print()
if failures:
    print(f"FAILED {len(failures)} assertion(s): {failures}")
    sys.exit(1)
print("ALL ASSERTIONS PASSED")
