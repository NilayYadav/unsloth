"""Real UI browser preference and uploaded background across fresh browsers."""

from __future__ import annotations

import asyncio
import re
import urllib.error
from pathlib import Path

from PIL import Image, ImageDraw
from pr_ui_scenes._common import Session, api_get
from studio_test_kit.auth import seed_init_script
from studio_test_kit.ui import open_chat


async def _settings(page, tab):
    if not await page.get_by_role("dialog").count():
        await page.get_by_role("button", name = "Settings", exact = True).click()
    dialog = page.get_by_role("dialog")
    await dialog.get_by_role("button", name = tab, exact = True).click()
    return dialog


async def drive(session: Session, out_dir: Path, label: str, **_):
    out_dir.mkdir(parents = True, exist_ok = True)
    image_path = out_dir / "uploaded-background.png"
    image = Image.new("RGB", (800, 500), "#065f80")
    draw = ImageDraw.Draw(image)
    for x in range(0, 800, 100):
        draw.rectangle((x, 0, x + 45, 500), fill = "#ef9347")
    image.save(image_path)
    init = seed_init_script(session, [])
    pref_path = "/api/settings/client-preferences"
    bg_path = "/api/settings/chat-background"
    try:
        api_get(session, pref_path)
        sync = True
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
        sync = False
    facts = {"sync_endpoint": sync, "selected_engine": "Google"}
    async with open_chat(session.base_url, init_scripts = [init], viewport = (1500, 1000)) as first:
        dialog = await _settings(first.page, "Browser")
        engine = dialog.get_by_role("combobox", name = "Search engine", exact = True)
        await engine.click()
        if sync:
            async with first.page.expect_response(
                lambda r: r.url.endswith(pref_path) and r.request.method == "PUT"
            ) as saved:
                await first.page.get_by_role("option", name = "Google", exact = True).click()
            assert (await saved.value).ok
        else:
            await first.page.get_by_role("option", name = "Google", exact = True).click()
        facts["first_browser_engine"] = await engine.inner_text()
        dialog = await _settings(first.page, "Appearance")
        upload = dialog.locator('input[type="file"][aria-label="Upload image"]')
        if sync:
            async with first.page.expect_response(
                lambda r: r.url.endswith(bg_path + "/image") and r.request.method == "PUT"
            ) as saved:
                await upload.set_input_files(str(image_path))
            assert (await saved.value).ok
        else:
            await upload.set_input_files(str(image_path))
        await dialog.get_by_role("button", name = "Replace image", exact = True).wait_for()
        await first.page.wait_for_function(
            "() => document.documentElement.hasAttribute('data-chat-wallpaper-ready')"
        )
        if sync:
            facts["saved_background"] = api_get(session, bg_path)
    requests = []
    async with open_chat(session.base_url, init_scripts = [init], viewport = (1500, 1000)) as second:
        second.page.on(
            "request",
            lambda r: requests.append(
                {"method": r.method, "path": r.url.split(session.base_url)[-1]}
            )
            if "/api/settings/client-preferences" in r.url
            or "/api/settings/chat-background" in r.url
            else None,
        )
        dialog = await _settings(second.page, "Browser")
        engine = dialog.get_by_role("combobox", name = "Search engine", exact = True)
        expected = "Google" if sync else "DuckDuckGo"
        await second.page.wait_for_function(
            "expected => document.querySelector('[role=combobox][aria-label=\"Search engine\"]')?.textContent.includes(expected)",
            arg = expected,
        )
        facts["fresh_browser_engine"] = await engine.inner_text()
        await engine.scroll_into_view_if_needed()
        browser_shot = out_dir / f"{label.lower()}_browser_preferences.png"
        await second.screenshot(browser_shot, full_page = False)
        dialog = await _settings(second.page, "Appearance")
        if sync:
            await dialog.get_by_role("button", name = "Replace image", exact = True).wait_for()
            await second.page.wait_for_function(
                "() => document.documentElement.hasAttribute('data-chat-wallpaper-ready')"
            )
        else:
            await dialog.locator("button").filter(has_text = re.compile("^Upload image$")).wait_for()
        facts["fresh_browser_background"] = (
            await dialog.get_by_role("button", name = "Replace image", exact = True).count() > 0
        )
        await dialog.locator("[data-wallpaper-previews]").scroll_into_view_if_needed()
        background_shot = out_dir / f"{label.lower()}_background.png"
        await second.screenshot(background_shot, full_page = False)
        if sync:
            before = api_get(session, bg_path)
            prefs_before = api_get(session, pref_path)
            for _ in range(2):
                await second.page.reload(wait_until = "domcontentloaded")
                await second.page.get_by_role("button", name = "Settings", exact = True).wait_for()
                await asyncio.sleep(1.2)
            after = api_get(session, bg_path)
            facts["repeat_load_background_unchanged"] = before == after
            facts["repeat_load_preferences_unchanged"] = prefs_before == api_get(session, pref_path)
            facts["repeat_load_write_requests"] = [r for r in requests if r["method"] != "GET"]
            facts["repeat_load_requests"] = requests
    return [browser_shot, background_shot], facts
