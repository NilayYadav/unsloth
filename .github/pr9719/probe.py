#!/usr/bin/env python3
"""PR 9719 repro probe.

Drives the real /v1/chat/completions safetensors path with a scripted vision
backend. Run from the repo root.

  A  two images on ONE message   -> HTTP 400 naming the one-image limit
  B  one image per turn (control) -> not rejected, an image still reaches the model
  C  no monitor row left running  -> whatever the outcome, the entry is terminal

B and C must hold on both sides. B because the guard counts the message being
answered, so an ordinary one-image-per-turn chat has to keep working; C because a
row left running keeps Studio reporting the backend as generating.
"""

import asyncio
import base64
import io
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join("studio", "backend")))

from PIL import Image
from fastapi import HTTPException

from models.inference import ChatCompletionRequest
import routes.inference as inf
from routes.inference import openai_chat_completions
from core.inference.api_monitor import ApiMonitor
from state.tool_policy import reset_tool_policy


def _png(colour):
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), colour).save(buf, format = "PNG")
    return base64.b64encode(buf.getvalue()).decode()


RED = _png((255, 0, 0))
BLUE = _png((0, 0, 255))
NAMES = {(255, 0, 0): "RED", (0, 0, 255): "BLUE"}


class _Request:
    state = SimpleNamespace()
    url = SimpleNamespace(path = "/v1/chat/completions")
    method = "POST"
    scope: dict = {}

    async def is_disconnected(self):
        return False


class _VisionBackend:
    active_model_name = "vision-sf"

    def __init__(self):
        self.models = {
            "vision-sf": {
                "is_vision": True,
                "chat_template_info": {"template": "chatml"},
                "context_length": 4096,
            }
        }
        self.images = []

    def resize_image(self, image):
        return image

    def generate_chat_response(self, **kwargs):
        self.images.append(kwargs.get("image"))
        yield "ok"

    def reset_generation_state(self, caller_cancel_event = None):
        pass


def _install(backend):
    reset_tool_policy()
    monitor = ApiMonitor(max_entries = 8)
    inf.api_monitor = monitor
    inf.get_llama_cpp_backend = lambda: SimpleNamespace(
        is_loaded = False, supports_tools = False, is_vision = False, context_length = None
    )
    inf.get_inference_backend = lambda: backend
    inf._detect_safetensors_features = lambda *a, **k: {"supports_tools": False}
    return monitor


def _turn(text, *images):
    content = [{"type": "text", "text": text}]
    for encoded in images:
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}
        )
    return {"role": "user", "content": content}


def _send(messages):
    backend = _VisionBackend()
    monitor = _install(backend)
    payload = ChatCompletionRequest(model = "default", messages = messages, stream = False)

    async def _run():
        return await openai_chat_completions(payload, request = _Request(), current_subject = "u")

    try:
        asyncio.run(_run())
    except HTTPException as exc:
        return backend, exc, monitor
    return backend, None, monitor


def _colour(image):
    if image is None:
        return "NONE"
    return NAMES.get(tuple(image.convert("RGB").getpixel((0, 0))), "OTHER")


def main():
    failures = []

    # A -- two images attached to a single message.
    backend, exc, monitor = _send([_turn("compare these two", RED, BLUE)])
    if exc is not None and exc.status_code == 400 and "one image per message" in str(exc.detail):
        print(f"A PASS  two images on one message -> HTTP 400: {exc.detail}")
    elif exc is not None:
        failures.append("A")
        print(f"A FAIL  two images on one message -> HTTP {exc.status_code}: {exc.detail}")
    else:
        failures.append("A")
        got = _colour(backend.images[0]) if backend.images else "NONE"
        print(
            f"A FAIL  two images on one message -> HTTP 200, backend was handed the {got} "
            "image only; the other one was dropped and nothing said so"
        )

    # B -- control: one image per turn is not a multi-image call.
    backend, exc, _ = _send(
        [
            _turn("what is this?", RED),
            {"role": "assistant", "content": "a red square"},
            _turn("and this one?", BLUE),
        ]
    )
    forwarded = bool(backend.images) and backend.images[0] is not None
    if exc is None and forwarded:
        print("B PASS  one image per turn -> not rejected, an image still reached the model")
    else:
        failures.append("B")
        detail = f"HTTP {exc.status_code}: {exc.detail}" if exc else "no image reached the model"
        print(f"B FAIL  one image per turn -> {detail}")

    # C -- the two-image request must not strand its API-monitor row.
    running = [e for e in list(getattr(monitor, "_entries")) if getattr(e, "status", None) == "running"]
    if not running:
        print("C PASS  the two-image request left no monitor row running")
    else:
        failures.append("C")
        print(
            f"C FAIL  the two-image request left {len(running)} monitor row(s) at "
            f"status='running'; Studio keeps reporting the backend as generating"
        )

    print(f"\nRESULT: {3 - len(failures)}/3 assertions passed", flush = True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
