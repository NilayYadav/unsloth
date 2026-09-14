#!/usr/bin/env python3
"""PR 10933 live Studio probe: an OpenRouter connection's reasoning control, what the backend
puts on the provider wire, and whether a text-only model accepts an image.

The connection points at a stand-in OpenRouter on loopback, so the real Studio backend builds
and sends the provider request and the stand-in records it. No key or network model is used.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from studio_test_kit.auth import login, seed_init_script  # noqa: E402
from studio_test_kit.ui import open_chat  # noqa: E402

EFFORT_MODEL = "deepseek/deepseek-v4-pro"
TEXT_ONLY_MODEL = "deepseek/deepseek-r1"
REPLY = "OK from the stand-in."

provider_bodies: list[dict] = []


class StandIn(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def _json(self, payload, status=200):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        # An empty live catalog keeps both sides on the bundled snapshot, so the pair is deterministic.
        self._json({"data": []})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        try:
            provider_bodies.append(json.loads(raw or b"{}"))
        except ValueError:
            provider_bodies.append({"__unparsable__": raw.decode(errors="replace")})
        chunks = [
            {"id": "x", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": REPLY}, "finish_reason": None}]},
            {"id": "x", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        ]
        body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        data = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def log(message: str) -> None:
    print(message, flush=True)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def find_unsloth_bin(home: Path) -> Path:
    for candidate in [home / "bin" / "unsloth", home / "unsloth_studio" / "bin" / "unsloth", *home.glob(".venv*/*/unsloth")]:
        if candidate.is_file():
            return candidate
    raise SystemExit(f"FAIL could not find unsloth CLI under {home}")


def read_bootstrap_password(home: Path, log_path: Path) -> str | None:
    for rel in ("auth/.bootstrap_password", ".bootstrap_password"):
        try:
            text = (home / rel).read_text(encoding="utf-8").strip()
            if text:
                return text
        except OSError:
            pass
    try:
        log_text = log_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    match = re.search(r"(?i)(?:bootstrap|initial|generated)\s*password(?:\s+is)?\s*[:=]?\s+(\S+)", log_text)
    return match.group(1).strip().strip(".,") if match else None


def wait_for_health(base_url: str, timeout_s: int = 240) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/api/health", timeout=3) as resp:
                if resp.status < 500:
                    return
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        time.sleep(2)
    raise SystemExit(f"FAIL Studio did not become healthy within {timeout_s}s")


def png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), (200, 40, 40)).save(buffer, format="PNG")
    return buffer.getvalue()


async def overlay(page, title: str, payload) -> None:
    await page.evaluate(
        """([title, payload]) => {
          const panel = document.createElement('div');
          panel.style.cssText = 'position:fixed;left:24px;top:24px;z-index:2147483647;max-width:560px;'
            + 'padding:14px 18px;border-radius:12px;background:#0b1020;color:#e6edf3;'
            + 'font:13px/1.5 ui-monospace,Menlo,monospace;box-shadow:0 10px 30px rgba(0,0,0,.35);white-space:pre-wrap;';
          panel.textContent = title + '\\n\\n' + JSON.stringify(payload, null, 2);
          document.body.appendChild(panel);
        }""",
        [title, payload],
    )


async def effort_scene(base_url: str, init: str, artifact_dir: Path, facts: dict) -> None:
    async with open_chat(base_url, init_scripts=[init], viewport=(1366, 900)) as sp:
        page = sp.page
        frontend_bodies: list[dict] = []
        page.on(
            "request",
            lambda r: frontend_bodies.append(json.loads(r.post_data or "{}"))
            if r.method == "POST" and r.url.endswith("/chat/completions")
            else None,
        )
        await page.goto(f"{base_url}/chat", wait_until="domcontentloaded", timeout=90_000)
        composer = page.locator("form:has(textarea) textarea").first
        await composer.wait_for(state="visible", timeout=90_000)
        await page.wait_for_timeout(5_000)
        facts["effort_model_on_screen"] = EFFORT_MODEL.split("/")[-1] in (await page.locator("body").inner_text())

        dropdown = page.locator("button.unsloth-thinking-pill").first
        toggle = page.locator('[data-pill-label="Thinking"]').first
        if await dropdown.count():
            facts["think_control"] = "effort dropdown"
            facts["think_label_before_pick"] = (await dropdown.inner_text()).strip()
            await dropdown.click()
            items = page.locator('.unsloth-thinking-menu [role="menuitem"]')
            await items.first.wait_for(state="visible", timeout=10_000)
            facts["think_menu_rows"] = [t.strip() for t in await items.all_inner_texts()]
            await page.screenshot(path=str(artifact_dir / "01_think_control.png"))
            pick = page.get_by_role("menuitem", name="Extra High", exact=True)
            if await pick.count():
                await pick.click()
            else:
                await page.keyboard.press("Escape")
            await page.wait_for_timeout(500)
            facts["think_label_after_pick"] = (await dropdown.inner_text()).strip()
        elif await toggle.count():
            facts["think_control"] = "on/off toggle"
            facts["think_menu_rows"] = []
            facts["think_label_before_pick"] = (await toggle.inner_text()).strip()
            await page.screenshot(path=str(artifact_dir / "01_think_control.png"))
        else:
            facts["think_control"] = "absent"
            facts["think_menu_rows"] = []
            await page.screenshot(path=str(artifact_dir / "01_think_control.png"))

        await composer.click()
        await composer.fill("Reply with OK.")
        await composer.press("Enter")
        for _ in range(60):
            if provider_bodies:
                break
            await page.wait_for_timeout(1_000)
        try:
            await page.get_by_text(REPLY).first.wait_for(state="visible", timeout=60_000)
            facts["reply_painted"] = True
        except Exception:  # noqa: BLE001
            facts["reply_painted"] = False
        frontend = frontend_bodies[0] if frontend_bodies else {}
        provider = provider_bodies[0] if provider_bodies else {}
        facts["frontend_reasoning_fields"] = {k: frontend[k] for k in ("reasoning_effort", "thinking", "enable_thinking") if k in frontend}
        facts["provider_model"] = provider.get("model")
        facts["provider_reasoning"] = provider.get("reasoning")
        await overlay(
            page,
            f"Sent for {EFFORT_MODEL}",
            {"Studio request": facts["frontend_reasoning_fields"], "OpenRouter wire reasoning": facts["provider_reasoning"]},
        )
        await page.wait_for_timeout(300)
        await page.screenshot(path=str(artifact_dir / "02_request_sent.png"))


async def image_scene(base_url: str, init: str, artifact_dir: Path, facts: dict) -> None:
    image_path = artifact_dir / "probe.png"
    image_path.write_bytes(png_bytes())
    async with open_chat(base_url, init_scripts=[init], viewport=(1366, 900)) as sp:
        page = sp.page
        await page.goto(f"{base_url}/chat", wait_until="domcontentloaded", timeout=90_000)
        composer = page.locator("form:has(textarea) textarea").first
        await composer.wait_for(state="visible", timeout=90_000)
        await page.wait_for_timeout(5_000)
        facts["text_only_model_on_screen"] = TEXT_ONLY_MODEL.split("/")[-1] in (await page.locator("body").inner_text())
        inputs = page.locator('input[type="file"]')
        target = None
        for index in range(await inputs.count()):
            accept = (await inputs.nth(index).get_attribute("accept")) or ""
            if "image" in accept:
                target = inputs.nth(index)
                break
        if target is None:
            facts["image_input_found"] = False
            await page.screenshot(path=str(artifact_dir / "03_image_attach.png"))
            return
        facts["image_input_found"] = True
        await target.set_input_files(str(image_path))
        refusal = page.get_by_text(re.compile(r"cannot accept images")).first
        try:
            await refusal.wait_for(state="visible", timeout=8_000)
            facts["image_refusal_toast"] = (await refusal.inner_text()).strip()
        except Exception:  # noqa: BLE001
            facts["image_refusal_toast"] = None
        facts["image_previews_in_composer"] = await page.locator(
            'form:has(textarea) img[src^="blob:"], form:has(textarea) img[src^="data:image"]'
        ).count()
        await page.wait_for_timeout(400)
        await page.screenshot(path=str(artifact_dir / "03_image_attach.png"))


async def main() -> int:
    home = Path(os.environ["UNSLOTH_STUDIO_HOME"]).resolve()
    artifact_dir = Path(os.environ.get("STUDIO_ARTIFACT_DIR", "studio-live-artifacts")).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    stand_in_port = free_port()
    server = ThreadingHTTPServer(("127.0.0.1", stand_in_port), StandIn)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    log_path = artifact_dir / "studio.log"
    env = os.environ.copy()
    env["UNSLOTH_STUDIO_HOME"] = str(home)
    handle = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [str(find_unsloth_bin(home)), "studio", "-H", "127.0.0.1", "-p", str(port)],
        stdout=handle, stderr=subprocess.STDOUT, env=env, start_new_session=True,
    )
    handle.close()
    facts: dict = {"head": os.environ.get("GITHUB_SHA")}
    try:
        wait_for_health(base_url)
        password = read_bootstrap_password(home, log_path)
        if not password:
            raise SystemExit("FAIL could not read the Studio bootstrap password")
        first = await login(base_url, "unsloth", password)
        # Every authenticated route answers 403 "Password change required" until the bootstrap password is rotated.
        async with httpx.AsyncClient(base_url=base_url, timeout=30) as client:
            rotated = await client.post(
                "/api/auth/change-password",
                json={"current_password": password, "new_password": f"probe-{os.urandom(12).hex()}"},
                headers={"Authorization": f"Bearer {first.access_token}"},
            )
            rotated.raise_for_status()
            tokens = rotated.json()
        auth = type("Auth", (), {"access_token": tokens["access_token"], "refresh_token": tokens.get("refresh_token", "")})()
        async with httpx.AsyncClient(base_url=base_url, headers={"Authorization": f"Bearer {auth.access_token}"}, timeout=30) as client:
            made = await client.post("/api/providers/", json={
                "provider_type": "openrouter",
                "display_name": "OpenRouter stand-in",
                "base_url": f"http://127.0.0.1:{stand_in_port}/api/v1",
                "models": [EFFORT_MODEL, TEXT_ONLY_MODEL],
                "available_models": [EFFORT_MODEL, TEXT_ONLY_MODEL],
                "encrypted_api_key": "",
            })
            made.raise_for_status()
            provider_id = made.json()["id"]

        def init_for(model: str) -> str:
            return seed_init_script(auth, [], extra_local_storage={
                "unsloth_chat_last_external_checkpoint": f"external::{provider_id}::{model}",
                "unsloth_chat_external_provider_keys": {provider_id: "stand-in-key"},
            })

        await effort_scene(base_url, init_for(EFFORT_MODEL), artifact_dir, facts)
        await image_scene(base_url, init_for(TEXT_ONLY_MODEL), artifact_dir, facts)
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        server.shutdown()

    checks = {
        "effort ladder offered for an OpenRouter reasoning model": facts.get("think_control") == "effort dropdown"
        and {"None", "High", "Extra High"} <= set(facts.get("think_menu_rows", [])),
        "OpenRouter wire carries reasoning.effort=xhigh": facts.get("provider_reasoning") == {"effort": "xhigh"},
        "text-only OpenRouter model refuses an image": bool(facts.get("image_refusal_toast")),
    }
    facts["checks"] = checks
    (artifact_dir / "facts.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")
    log("FACTS " + json.dumps(facts, sort_keys=True))
    for name, ok in checks.items():
        log(("PASS " if ok else "FAIL ") + name)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
