"""Citation preview setting, hover behavior, and reload persistence."""

import json
import re
import urllib.request
import uuid
from pathlib import Path
from types import SimpleNamespace

from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from pr_ui_scenes._common import Session, api_get, api_post
from studio_test_kit.auth import seed_init_script
from studio_test_kit.ui import open_chat

THREAD = "issue16399-link-previews-" + uuid.uuid4().hex[:8]
TITLE = "Link preview preference"
SOURCE_TITLE = "Unsloth documentation"
DESCRIPTION = "Source details appear here when link previews are enabled."
SOURCE_URL = "https://unsloth.ai/docs"


def seed(session: Session) -> None:
    api_post(
        session,
        "/api/chat/threads",
        {
            "id": THREAD,
            "title": TITLE,
            "modelType": "base",
            "modelId": "",
            "createdAt": 1767225600000,
            "updatedAt": 1767225600000,
        },
    )
    turns = [
        ("question", None, "user", [{"type": "text", "text": "Where can I read more?"}]),
        (
            "answer",
            "question",
            "assistant",
            [
                {
                    "type": "text",
                    "text": "The documentation has more details. The source below stays clickable.",
                },
                {
                    "type": "source",
                    "sourceType": "url",
                    "id": "docs",
                    "url": SOURCE_URL,
                    "title": SOURCE_TITLE,
                    "metadata": {"description": DESCRIPTION},
                },
            ],
        ),
    ]
    for index, (mid, parent, role, content) in enumerate(turns):
        payload = {
            "id": f"{THREAD}-{mid}",
            "threadId": THREAD,
            "parentId": f"{THREAD}-{parent}" if parent else None,
            "role": role,
            "content": content,
            "createdAt": 1767225600001 + index,
        }
        request = urllib.request.Request(
            f"{session.base_url}/api/chat/threads/{THREAD}/messages/{THREAD}-{mid}",
            data = json.dumps(payload).encode(),
            method = "PUT",
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {session.access_token}",
            },
        )
        with urllib.request.urlopen(request, timeout = 60) as response:
            json.loads(response.read())
    if api_get(session, f"/api/chat/threads/{THREAD}")["title"] != TITLE:
        raise AssertionError("Seeded thread did not round-trip")


async def settings(page):
    await page.get_by_role("button", name = "Settings", exact = True).click()
    dialog = page.get_by_role("dialog")
    await dialog.wait_for(state = "visible")
    await dialog.get_by_role("button", name = re.compile(r"^Chat\b")).first.click()
    await dialog.get_by_text("Response model", exact = True).wait_for(state = "visible")
    return dialog


async def hover(page, expected: bool):
    source = page.locator('a[data-slot="source"]').filter(has_text = SOURCE_TITLE).last
    await source.wait_for(state = "visible")
    if await source.get_attribute("href") != SOURCE_URL:
        raise AssertionError("Source navigation target changed")
    preview = page.locator('[data-slot="hover-card-content"]')
    for attempt in range(3):
        await page.mouse.move(10, 10)
        await source.hover()
        try:
            await preview.wait_for(state = "visible" if expected else "hidden", timeout = 4_000)
            break
        except PlaywrightTimeoutError:
            if attempt == 2:
                raise
    if expected:
        await preview.get_by_text(DESCRIPTION, exact = True).wait_for(state = "visible")
    elif await page.locator('[data-slot="hover-card-trigger"]').count():
        raise AssertionError("Disabled preview retained a hover trigger")
    return {
        "preview_visible": await preview.is_visible(),
        "source_href": await source.get_attribute("href"),
    }


async def drive(session: Session, out_dir: Path, label: str, **_):
    seed(session)
    auth = SimpleNamespace(access_token = session.access_token, refresh_token = session.refresh_token)
    facts, shots = {}, []
    async with open_chat(
        session.base_url,
        init_scripts = [seed_init_script(auth, [])],
        viewport = (1440, 900),
        video_dir = out_dir / "video",
        video_name = label.lower(),
        transcode_mp4 = True,
    ) as sp:
        page = sp.page
        await page.goto(f"{session.base_url}/chat?thread={THREAD}", wait_until = "domcontentloaded")
        await page.get_by_text("The documentation has more details.", exact = False).first.wait_for(
            timeout = 60_000
        )
        facts["default"] = await hover(page, True)
        await page.mouse.move(10, 10)
        await page.locator('[data-slot="hover-card-content"]').wait_for(state = "hidden")
        dialog = await settings(page)
        switch = dialog.get_by_role("switch", name = "Link previews", exact = True)
        facts["setting_present"] = await switch.count() == 1
        if facts["setting_present"]:
            await switch.scroll_into_view_if_needed()
            if await switch.get_attribute("aria-checked") != "true":
                raise AssertionError("New preference is not enabled by default")
            await switch.click()
            if await switch.get_attribute("aria-checked") != "false":
                raise AssertionError("Preview switch did not turn off")
        else:
            await dialog.get_by_text("Response model", exact = True).scroll_into_view_if_needed()
        shot = out_dir / f"{label.lower()}_settings.png"
        await dialog.screenshot(path = str(shot))
        shots.append(shot)
        await page.keyboard.press("Escape")
        facts["after_disable"] = await hover(page, not facts["setting_present"])
        shot = out_dir / f"{label.lower()}_disabled_hover.png"
        await page.screenshot(path = str(shot), full_page = False)
        shots.append(shot)
        await page.mouse.move(10, 10)
        await page.reload(wait_until = "domcontentloaded")
        await page.get_by_text("The documentation has more details.", exact = False).first.wait_for(
            timeout = 60_000
        )
        facts["after_reload"] = await hover(page, not facts["setting_present"])
        if facts["setting_present"]:
            await page.mouse.move(10, 10)
            dialog = await settings(page)
            switch = dialog.get_by_role("switch", name = "Link previews", exact = True)
            await switch.scroll_into_view_if_needed()
            if await switch.get_attribute("aria-checked") != "false":
                raise AssertionError("Saved off preference was lost after reload")
            await switch.click()
            await page.keyboard.press("Escape")
            facts["reenabled"] = await hover(page, True)
    (out_dir / "checks.json").write_text(json.dumps(facts, indent = 2), encoding = "utf-8")
    return shots, facts
