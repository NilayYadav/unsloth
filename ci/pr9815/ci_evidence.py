# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""PR 9815 repro: does the public /p preview page render a thinking model's reasoning?

Serves the real preview router with a scripted OpenAI-compatible stream and drives
the real page in Chromium. Two turns:

  thinking -- reasoning_content deltas, then content deltas, finish_reason "stop"
  capped   -- reasoning_content only, then finish_reason "length" and no content
              (what an always-thinking model does when reasoning spends the whole
              1024-token preview cap)

Asserts the reasoning is visible and that a content-free capped turn explains
itself. Exits non-zero when the page drops either, which is what the pre-fix page
does. Screenshots and facts land in --outdir for the run artifact.
"""

import argparse, asyncio, json, logging, sys, threading, time, types
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--backend", required=True)
ap.add_argument("--outdir", required=True)
ap.add_argument("--port", type=int, default=8971)
ap.add_argument("--label", default="")
args = ap.parse_args()

BACKEND = Path(args.backend).resolve()
OUT = Path(args.outdir).resolve()
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(BACKEND))

_loggers = types.ModuleType("loggers")
_loggers.get_logger = lambda name: logging.getLogger(name)
sys.modules.setdefault("loggers", _loggers)

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

import routes.preview as preview
import utils.preview_token as preview_token
import utils.preview_rate_limit as _rl
from utils.paths import storage_roots as _sr

SECRET = b"pr9815-repro-secret-0123456789"
preview_token.get_or_create_preview_link_secret = lambda: SECRET
preview.get_preview_sharing_enabled = lambda: True

outputs = OUT / "outputs"
run_dir = outputs / "qwen3-4b-thinking"
(run_dir / "checkpoint-1").mkdir(parents=True, exist_ok=True)
(run_dir / "adapter_config.json").write_text(
    json.dumps({"base_model_name_or_path": "unsloth/Qwen3-4B-Thinking-2507"})
)
(run_dir / "checkpoint-1" / "adapter_config.json").write_text("{}")
_sr.outputs_root = lambda: outputs
_rl.reset()


def chunk(delta=None, finish=None):
    return "data: " + json.dumps({
        "id": "chatcmpl-pr9815",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "preview",
        "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}],
    }) + "\n\n"


THINK_1 = ["The user asks for ", "the capital of France. ", "That is Paris. ",
           "I should answer briefly."]
ANSWER_1 = ["The capital of ", "France is ", "**Paris**."]
THINK_2 = ["Let me work through this step by step. ",
           "I need to consider every case carefully, ",
           "enumerate the possibilities, ",
           "and check each one against the constraints. ",
           "This is taking a while and I have not reached an answer yet."]


async def fake_load(load_req, request, subject):
    return None


async def fake_chat(payload, request, subject):
    last = ""
    for m in payload.messages:
        if getattr(m, "role", None) == "user":
            c = m.content
            last = c if isinstance(c, str) else str(c)
    capped = "puzzle" in last.lower()

    async def gen():
        yield chunk({"role": "assistant"})
        for t in (THINK_2 if capped else THINK_1):
            yield chunk({"reasoning_content": t})
            await asyncio.sleep(0.01)
        if capped:
            yield chunk({}, "length")
        else:
            for t in ANSWER_1:
                yield chunk({"content": t})
            yield chunk({}, "stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


preview.load_model_for_preview = fake_load
preview.openai_chat_completions = fake_chat

app = FastAPI()
app.include_router(preview.router, prefix="/p")

import uvicorn
server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=args.port, log_level="error"))
threading.Thread(target=server.run, daemon=True).start()
for _ in range(400):
    if server.started:
        break
    time.sleep(0.05)
if not server.started:
    print("FAIL: preview server did not start", flush=True)
    sys.exit(2)

token = preview_token.sign_preview_ref("qwen3-4b-thinking")
url = f"http://127.0.0.1:{args.port}/p/qwen3-4b-thinking?k={token}"

from playwright.sync_api import sync_playwright

facts = {"label": args.label}
SCENES = [("thinking", "What is the capital of France?"),
          ("capped", "Think through a hard puzzle step by step.")]

with sync_playwright() as p:
    browser = p.chromium.launch()
    for scene, prompt in SCENES:
        page = browser.new_page(viewport={"width": 900, "height": 720}, device_scale_factor=2)
        page.goto(url, wait_until="networkidle")
        page.fill("#i", prompt)
        page.click("#b")
        page.wait_for_function("() => !document.getElementById('b').disabled", timeout=30000)
        page.wait_for_timeout(400)

        bubble = page.locator(".msg.assistant").last
        think = bubble.locator("details.think")
        f = {
            "think_block_visible": think.count() > 0 and think.first.is_visible(),
            "reasoning_chars": 0,
            "cutoff_notice": "",
            "bubble_text_chars": len((bubble.inner_text() or "").strip()),
        }
        if f["think_block_visible"]:
            think.first.click()
            page.wait_for_timeout(200)
            f["reasoning_chars"] = len((think.first.locator("div").inner_text() or "").strip())
        cut = bubble.locator(".cutoff")
        if cut.count() > 0:
            f["cutoff_notice"] = (cut.first.inner_text() or "").strip()
        facts[scene] = f
        page.screenshot(path=str(OUT / f"{scene}.png"))
        page.close()
    browser.close()

(OUT / "facts.json").write_text(json.dumps(facts, indent=2))
print(json.dumps(facts, indent=2), flush=True)

CUTOFF = "Reply cut off at the preview length limit."
checks = [
    ("thinking turn shows the reasoning block", facts["thinking"]["think_block_visible"]),
    ("thinking turn kept the reasoning text", facts["thinking"]["reasoning_chars"] > 0),
    ("capped turn shows the reasoning block", facts["capped"]["think_block_visible"]),
    ("capped turn is not an empty bubble", facts["capped"]["bubble_text_chars"] > 0),
    ("capped turn explains the cutoff", facts["capped"]["cutoff_notice"] == CUTOFF),
]
bad = 0
for name, ok in checks:
    print(("PASS: " if ok else "FAIL: ") + name, flush=True)
    bad += 0 if ok else 1
print(f"repro result: {len(checks)-bad}/{len(checks)} checks passed", flush=True)
sys.exit(1 if bad else 0)
