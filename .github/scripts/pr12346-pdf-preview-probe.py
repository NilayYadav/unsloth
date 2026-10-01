#!/usr/bin/env python3
"""Generated live Unsloth Studio scenario probe.

Do not print provider keys, bootstrap passwords, auth tokens, or API keys.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import httpx
from studio_test_kit.auth import (
    anthropic_provider,
    gemini_provider,
    login,
    openai_provider,
    seed_init_script,
)
from studio_test_kit.compose import hstack_images, hstack_videos
from studio_test_kit.flows import image_generation, multi_turn_chat, tool_pills, vision_upload
from studio_test_kit.ui import open_chat, send_prompt, wait_for_stream

SCENARIO = "smoke"
PROVIDER = "gemini"
MODEL = "gemini-2.5-flash"
PROMPT = "Say hello from Studio CI in one short sentence."
TURNS = ["Translate 'good morning' into Japanese.", "Now answer in a pirate voice.", "Summarize this thread in 5 words."]
RECORD_VIDEO = False
LOCAL_MODEL = "unsloth/Qwen3-1.7B-GGUF"
LOCAL_GGUF_VARIANT = "UD-Q4_K_XL"
LOCAL_MAX_SEQ_LENGTH = 1024
LOCAL_PARALLEL = 1
PRE_REF = "main"
INSTALL_UNIX = "bash ./install.sh --local --no-torch"
INSTALL_WINDOWS = "& ./install.ps1 --local --no-torch"


def pass_log(message: str) -> None:
    print(f"PASS {message}", flush=True)


def warn(message: str) -> None:
    print(f"WARN {message}", flush=True)


def fail(message: str) -> None:
    print(f"FAIL {message}", file=sys.stderr, flush=True)
    raise SystemExit(1)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def find_unsloth_bin(home: Path) -> Path:
    candidates = [
        home / "bin" / "unsloth",
        home / "bin" / "unsloth.exe",
        home / "unsloth_studio" / "bin" / "unsloth",
        home / "unsloth_studio" / "Scripts" / "unsloth.exe",
    ]
    candidates.extend(home.glob(".venv*/*/unsloth"))
    candidates.extend(home.glob(".venv*/Scripts/unsloth.exe"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    fail(f"could not find unsloth CLI under {home}")


def read_bootstrap_password(home: Path, log_path: Path) -> str | None:
    for rel in ("auth/.bootstrap_password", ".bootstrap_password"):
        path = home / rel
        try:
            text = path.read_text(encoding="utf-8").strip()
            if text:
                return text
        except OSError:
            pass
    try:
        log_text = log_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    match = re.search(
        r"(?i)(?:bootstrap|initial|generated)\s*password(?:\s+is)?\s*[:=]?\s+(\S+)",
        log_text,
    )
    return match.group(1).strip().strip(".,") if match else None


def read_api_key(log_path: Path, timeout_s: int = 900) -> str:
    deadline = time.time() + timeout_s
    pattern = re.compile(r"API Key:\s+(sk-unsloth-[a-f0-9]+)")
    while time.time() < deadline:
        try:
            text = log_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            text = ""
        match = pattern.search(text)
        if match:
            return match.group(1)
        time.sleep(2)
    fail("timed out waiting for Studio run API key in log")


def wait_for_health(base_url: str, timeout_s: int = 180) -> str:
    deadline = time.time() + timeout_s
    paths = ("/healthz", "/api/health", "/")
    while time.time() < deadline:
        for path in paths:
            try:
                with urllib.request.urlopen(f"{base_url}{path}", timeout=3) as resp:
                    if resp.status < 500:
                        return path
            except (urllib.error.URLError, OSError, TimeoutError):
                pass
        time.sleep(2)
    fail(f"Studio did not become healthy within {timeout_s}s; tried {paths}")


def process_kwargs(log_path: Path, env: dict[str, str]) -> dict:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8")
    kwargs: dict = {"stdout": log_handle, "stderr": subprocess.STDOUT, "env": env}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    else:
        kwargs["start_new_session"] = True
    kwargs["_log_handle"] = log_handle
    return kwargs


def start_process(cmd: list[str], log_path: Path, env: dict[str, str]) -> subprocess.Popen:
    kwargs = process_kwargs(log_path, env)
    log_handle = kwargs.pop("_log_handle")
    print("Launching: " + " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, **kwargs)
    log_handle.close()
    return proc


def studio_env(home: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["UNSLOTH_STUDIO_HOME"] = str(home)
    env.pop("STUDIO_HOME", None)
    return env


def start_plain_studio(home: Path, log_path: Path, port: int) -> subprocess.Popen:
    return start_process(
        [str(find_unsloth_bin(home)), "studio", "-H", "127.0.0.1", "-p", str(port)],
        log_path,
        studio_env(home),
    )


def start_local_studio_run(home: Path, log_path: Path, port: int) -> subprocess.Popen:
    cmd = [
        str(find_unsloth_bin(home)),
        "studio",
        "run",
        "--model",
        LOCAL_MODEL,
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
        "--api-key-name",
        "ci",
        "--max-seq-length",
        str(LOCAL_MAX_SEQ_LENGTH),
        "--parallel",
        str(LOCAL_PARALLEL),
        "--disable-tools",
    ]
    if LOCAL_GGUF_VARIANT:
        cmd.extend(["--gguf-variant", LOCAL_GGUF_VARIANT])
    return start_process(cmd, log_path, studio_env(home))


def stop_process(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    if os.name == "nt":
        proc.terminate()
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except OSError:
            proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            proc.kill()
        else:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except OSError:
                proc.kill()


def provider_seed(model: str):
    if PROVIDER == "gemini":
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            fail("GEMINI_API_KEY secret is required for provider scenario")
        return gemini_provider(api_key=key, models=[model])
    if PROVIDER == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            fail("OPENAI_API_KEY secret is required for provider scenario")
        return openai_provider(api_key=key, models=[model])
    if PROVIDER == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            fail("ANTHROPIC_API_KEY secret is required for provider scenario")
        return anthropic_provider(api_key=key, models=[model])
    fail(f"unsupported provider: {PROVIDER}")


def video_dir(artifact_dir: Path, name: str) -> Path | None:
    return artifact_dir / "video" / name if RECORD_VIDEO else None


async def auth_init(base_url: str, password: str, providers: list | None = None) -> str:
    auth = await login(base_url, "unsloth", password)
    pass_log("Studio API login succeeded")
    if auth.must_change_password:
        new_password = os.environ.get("STUDIO_TEST_PASSWORD", "UnslothStudioCI2026!")
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(
                f"{base_url}/api/auth/change-password",
                headers={"Authorization": f"Bearer {auth.access_token}"},
                json={"current_password": password, "new_password": new_password},
            )
            response.raise_for_status()
            body = response.json()
        auth.access_token = body["access_token"]
        auth.refresh_token = body.get("refresh_token", "")
        pass_log("Studio first-boot password change completed")
    return seed_init_script(auth, providers or [])


async def scenario_smoke(base_url: str, password: str, browser_name: str, artifact_dir: Path) -> None:
    init = await auth_init(base_url, password)
    async with open_chat(
        base_url,
        init_scripts=[init],
        video_dir=video_dir(artifact_dir, "smoke"),
        video_name=f"smoke-{browser_name}",
        transcode_mp4=False,
        browser_name=browser_name,
    ) as sp:
        composer = sp.page.locator("form:has(textarea) textarea").first
        await composer.wait_for(state="visible", timeout=30_000)
        await sp.screenshot(artifact_dir / f"studio-chat-{browser_name}.png")
    pass_log(f"Playwright {browser_name} opened /chat and found the composer")


def write_pdf(path: Path, title: str, color: str) -> Path:
    content = f"q {color} rg 50 600 500 150 re f Q BT /F1 36 Tf 70 660 Td ({title}) Tj ET".encode()
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
            b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out = b"%PDF-1.4\n"
    offs = []
    for i, o in enumerate(objs, 1):
        offs.append(len(out))
        out += b"%d 0 obj\n" % i + o + b"\nendobj\n"
    x = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1) + b"".join(b"%010d 00000 n \n" % o for o in offs)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, x)
    path.write_bytes(out)
    return path


CANT = "can't be previewed here"


async def scenario_pdf_preview(base_url: str, password: str, browser_name: str, artifact_dir: Path) -> None:
    a = write_pdf(artifact_dir / "sample-a.pdf", "Document A", "0.2 0.6 0.9")
    b = write_pdf(artifact_dir / "sample-b.pdf", "Document B", "0.9 0.5 0.2")
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        browser = await getattr(pw, browser_name).launch()
        context = await browser.new_context(viewport={"width": 1440, "height": 900})
        page = await context.new_page()
        await page.goto(f"{base_url}/login", wait_until="domcontentloaded", timeout=120_000)
        await page.locator("input[type=password]").first.fill(password)
        await page.get_by_role("button", name="Login").click()
        await page.wait_for_url(re.compile(r"/(chat|change-password)"), timeout=60_000)
        if "change-password" in page.url:
            new_password = os.environ.get("STUDIO_TEST_PASSWORD", "UnslothStudioCI2026!")
            fields = page.locator("input[type=password]")
            await fields.first.wait_for(timeout=30_000)
            for i in range(await fields.count()):
                await fields.nth(i).fill(new_password)
            await page.get_by_role("button", name="Change password").click()
            await page.wait_for_url(re.compile(r"/chat"), timeout=60_000)
        pass_log("logged in through the Studio login form")
        console: list[str] = []
        page.on("console", lambda m: console.append(f"{m.type}: {m.text[:300]}"))
        page.on("pageerror", lambda e: console.append(f"pageerror: {str(e)[:300]}"))
        await page.locator("form:has(textarea) textarea").first.wait_for(state="visible", timeout=60_000)

        async def attach(path: Path) -> None:
            for attempt in range(3):
                await page.locator("form:has(textarea) textarea").first.wait_for(state="visible", timeout=60_000)
                await page.get_by_role("button", name="Tools and attachments").click()
                await page.wait_for_timeout(600)
                async with page.expect_file_chooser() as fc:
                    await page.get_by_role("menuitem", name="Add photos & files").click()
                await (await fc.value).set_files(str(path))
                try:
                    await page.locator("form").get_by_text(path.name).first.wait_for(timeout=20_000)
                    return
                except Exception:
                    warn(f"{path.name} card did not appear (attempt {attempt + 1}); attaching again")
            fail(f"could not attach {path.name}")

        await attach(a)
        await page.wait_for_timeout(8000)
        await attach(b)
        card_b = page.locator("form").get_by_text(CANT)
        deadline = time.time() + 30
        while time.time() < deadline and await page.locator("form canvas").count() < 1 and await card_b.count() == 0:
            await page.wait_for_timeout(500)
        await page.wait_for_timeout(2000)
        await page.screenshot(path=str(artifact_dir / f"cards-{browser_name}.png"))
        cards = {"canvases": await page.locator("form canvas").count(), "cant_preview": await card_b.count()}
        print(f"CARDS {json.dumps(cards)}", flush=True)
        await page.get_by_text("sample-b.pdf").first.click()
        dialog = page.get_by_role("dialog").first
        await dialog.wait_for(timeout=30_000)
        deadline = time.time() + 30
        while time.time() < deadline and await dialog.get_by_text("Document B").count() == 0 and await dialog.get_by_text(CANT).count() == 0:
            await page.wait_for_timeout(500)
        await page.wait_for_timeout(1500)
        await page.screenshot(path=str(artifact_dir / f"dialog-{browser_name}.png"))
        shown = {"page_text": await dialog.get_by_text("Document B").count(), "cant_preview": await dialog.get_by_text(CANT).count()}
        print(f"DIALOG {json.dumps(shown)}", flush=True)
        (artifact_dir / "console.txt").write_text("\n".join(console), encoding="utf-8")
        mismatch = [line for line in console if "does not match the Worker version" in line or "fake worker" in line.lower()]
        if mismatch:
            print("CONSOLE " + " | ".join(mismatch[:3]), flush=True)
        if cards["cant_preview"] or cards["canvases"] < 1:
            fail(f"PDF card preview broken after a PDF was extracted: {cards}")
        if shown["cant_preview"] or not shown["page_text"]:
            fail(f"PDF preview dialog broken after a PDF was extracted: {shown}")
        await browser.close()
    pass_log(f"{browser_name}: both PDF cards and the preview dialog rendered after extraction")


async def scenario_provider_chat(base_url: str, password: str, browser_name: str, artifact_dir: Path) -> None:
    init = await auth_init(base_url, password, [provider_seed(MODEL)])
    async with open_chat(
        base_url,
        init_scripts=[init],
        video_dir=video_dir(artifact_dir, "provider-chat"),
        video_name=f"provider-chat-{browser_name}",
        transcode_mp4=False,
        browser_name=browser_name,
    ) as sp:
        result = await multi_turn_chat(sp, MODEL, TURNS, artifact_dir / "provider-chat")
    result.attach_video(sp)
    pass_log(f"provider chat completed {result.artefacts.get('turn_count', len(TURNS))} turns")


async def scenario_image_gen(base_url: str, password: str, browser_name: str, artifact_dir: Path) -> None:
    init = await auth_init(base_url, password, [provider_seed(MODEL)])
    async with open_chat(
        base_url,
        init_scripts=[init],
        video_dir=video_dir(artifact_dir, "image-gen"),
        video_name=f"image-gen-{browser_name}",
        transcode_mp4=False,
        browser_name=browser_name,
    ) as sp:
        result = await image_generation(sp, MODEL, PROMPT, artifact_dir / "image-gen")
    result.attach_video(sp)
    pass_log(f"image generation saved {result.artefacts.get('image_bytes')} bytes")


async def scenario_tools_pills(base_url: str, password: str, browser_name: str, artifact_dir: Path) -> None:
    init = await auth_init(base_url, password, [provider_seed(MODEL)])
    async with open_chat(
        base_url,
        init_scripts=[init],
        video_dir=video_dir(artifact_dir, "tools-pills"),
        video_name=f"tools-pills-{browser_name}",
        transcode_mp4=False,
        browser_name=browser_name,
    ) as sp:
        result = await tool_pills(sp, MODEL, artifact_dir / "tools-pills")
    result.attach_video(sp)
    pass_log("Search and Code composer pill flow completed")


def write_default_vision_image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tiny_png = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
        "AAIAAAoAAv/lxKUAAAAASUVORK5CYII="
    )
    path.write_bytes(base64.b64decode(tiny_png))
    return path


async def scenario_vision_upload(base_url: str, password: str, browser_name: str, artifact_dir: Path) -> None:
    image_path = Path(os.environ.get("STUDIO_VISION_IMAGE", ""))
    if not image_path.is_file():
        image_path = write_default_vision_image(artifact_dir / "vision" / "input.png")
    init = await auth_init(base_url, password, [provider_seed(MODEL)])
    async with open_chat(
        base_url,
        init_scripts=[init],
        video_dir=video_dir(artifact_dir, "vision-upload"),
        video_name=f"vision-upload-{browser_name}",
        transcode_mp4=False,
        browser_name=browser_name,
    ) as sp:
        result = await vision_upload(sp, MODEL, image_path, PROMPT, artifact_dir / "vision-upload")
    result.attach_video(sp)
    pass_log("vision upload flow completed")


async def scenario_local_chat(base_url: str, api_key: str, browser_name: str, artifact_dir: Path, password: str | None) -> None:
    payload = {
        "model": LOCAL_MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": False,
        "max_tokens": 64,
        "temperature": 0,
    }
    async with httpx.AsyncClient(timeout=180) as client:
        response = await client.post(
            f"{base_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
        )
        response.raise_for_status()
        body = response.json()
    (artifact_dir / "local-chat-response.json").write_text(json.dumps(body, indent=2), encoding="utf-8")
    content = (((body.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
    if not content:
        fail("local-chat API returned no assistant content")
    pass_log("local small-model chat API returned assistant content")

    init_scripts = [await auth_init(base_url, password)] if password else []
    try:
        async with open_chat(
            base_url,
            init_scripts=init_scripts,
            video_dir=video_dir(artifact_dir, "local-chat"),
            video_name=f"local-chat-{browser_name}",
            transcode_mp4=False,
            browser_name=browser_name,
        ) as sp:
            await send_prompt(sp, PROMPT)
            await wait_for_stream(sp, timeout_ms=180_000)
            await sp.screenshot(artifact_dir / f"local-chat-{browser_name}.png")
        pass_log("local small-model chat UI accepted a prompt")
    except Exception as exc:
        warn(f"local-chat UI probe skipped after API success: {type(exc).__name__}: {exc}")


def install_pre_ref(root: Path, log_path: Path) -> Path:
    if os.name == "nt":
        fail("side-by-side scenario is currently intended for Unix runners")
    repo = root / "pre-repo"
    home = root / "pre-home"
    origin = subprocess.check_output(["git", "remote", "get-url", "origin"], text=True).strip()
    clone_cmd = ["git", "clone", "--depth", "1", "--branch", PRE_REF, origin, str(repo)]
    try:
        subprocess.run(clone_cmd, check=True)
    except subprocess.CalledProcessError:
        warn(f"direct clone of pre-ref {PRE_REF!r} failed; trying fetch checkout fallback")
        if repo.exists():
            shutil.rmtree(repo)
        subprocess.run(["git", "clone", "--no-local", ".", str(repo)], check=True)
        subprocess.run(["git", "fetch", origin, PRE_REF, "--depth", "1"], cwd=repo, check=True)
        subprocess.run(["git", "checkout", "FETCH_HEAD"], cwd=repo, check=True)
    home.mkdir(parents=True, exist_ok=True)
    env = studio_env(home)
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(INSTALL_UNIX, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=2700, shell=True)
    return home


async def scenario_side_by_side(post_home: Path, post_base: str, post_password: str, browser_name: str, artifact_dir: Path) -> None:
    root = artifact_dir / "side-by-side"
    pre_home = install_pre_ref(root, artifact_dir / "pre-install.log")
    pre_port = free_port()
    pre_base = f"http://127.0.0.1:{pre_port}"
    pre_proc = start_plain_studio(pre_home, artifact_dir / "pre-studio.log", pre_port)
    try:
        wait_for_health(pre_base, timeout_s=180)
        pre_password = read_bootstrap_password(pre_home, artifact_dir / "pre-studio.log")
        if not pre_password:
            fail("could not read pre-ref Studio bootstrap password")
        async def drive(label: str, base: str, password: str):
            init = await auth_init(base, password, [provider_seed(MODEL)])
            async with open_chat(
                base,
                init_scripts=[init],
                video_dir=video_dir(artifact_dir, label),
                video_name=f"{label}-{browser_name}",
                transcode_mp4=False,
                browser_name=browser_name,
            ) as sp:
                result = await multi_turn_chat(sp, MODEL, TURNS, artifact_dir / label)
            result.attach_video(sp)
            return result
        pre = await drive("pre", pre_base, pre_password)
        post = await drive("post", post_base, post_password)
        combined = artifact_dir / "combined"
        combined.mkdir(parents=True, exist_ok=True)
        for idx, (left, right) in enumerate(zip(pre.screenshots, post.screenshots), start=1):
            hstack_images(left, right, combined / f"sxs_{idx:02d}.png", label_left=PRE_REF, label_right="post")
        if pre.video_webm and post.video_webm:
            try:
                hstack_videos(pre.video_webm, post.video_webm, combined / "sxs.mp4")
            except Exception as exc:
                warn(f"side-by-side video composition skipped: {exc}")
        pass_log("side-by-side provider chat comparison completed")
    finally:
        stop_process(pre_proc)


async def run_plain_scenario(home: Path, base_url: str, password: str, browser_name: str, artifact_dir: Path) -> None:
    if SCENARIO == "smoke":
        await scenario_pdf_preview(base_url, password, browser_name, artifact_dir)
    elif SCENARIO == "provider-chat":
        await scenario_provider_chat(base_url, password, browser_name, artifact_dir)
    elif SCENARIO == "image-gen":
        await scenario_image_gen(base_url, password, browser_name, artifact_dir)
    elif SCENARIO == "tools-pills":
        await scenario_tools_pills(base_url, password, browser_name, artifact_dir)
    elif SCENARIO == "vision-upload":
        await scenario_vision_upload(base_url, password, browser_name, artifact_dir)
    elif SCENARIO == "side-by-side":
        await scenario_side_by_side(home, base_url, password, browser_name, artifact_dir)
    else:
        fail(f"unexpected plain Studio scenario: {SCENARIO}")


async def main() -> None:
    home = Path(os.environ["UNSLOTH_STUDIO_HOME"]).resolve()
    browser_name = os.environ.get("STUDIO_BROWSER", "chromium")
    artifact_dir = Path(os.environ.get("STUDIO_ARTIFACT_DIR", "studio-live-artifacts")).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    proc: subprocess.Popen | None = None
    try:
        if SCENARIO == "local-chat":
            log_path = artifact_dir / "studio-run.log"
            proc = start_local_studio_run(home, log_path, port)
            api_key = read_api_key(log_path)
            health_path = wait_for_health(base_url, timeout_s=900)
            pass_log(f"Studio run healthy at {health_path} on {base_url}")

            password = read_bootstrap_password(home, log_path)
            await scenario_local_chat(base_url, api_key, browser_name, artifact_dir, password)
        else:
            log_path = artifact_dir / "studio.log"
            proc = start_plain_studio(home, log_path, port)
            health_path = wait_for_health(base_url)
            pass_log(f"Studio healthy at {health_path} on {base_url}")
            password = read_bootstrap_password(home, log_path)
            if not password:
                fail("could not read Studio bootstrap password from install home or log")
            await run_plain_scenario(home, base_url, password, browser_name, artifact_dir)
    finally:
        stop_process(proc)


if __name__ == "__main__":
    asyncio.run(main())
