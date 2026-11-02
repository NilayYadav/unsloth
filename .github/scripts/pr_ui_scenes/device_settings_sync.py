"""Show interface scale in a fresh browser after an account edits it elsewhere."""

from __future__ import annotations

import urllib.error
from pathlib import Path

from pr_ui_scenes._common import Session, api_get
from studio_test_kit.auth import seed_init_script
from studio_test_kit.ui import open_chat


async def _scale_control(page):
    await page.get_by_role("button", name = "Settings", exact = True).click()
    dialog = page.get_by_role("dialog")
    await dialog.get_by_role("button", name = "Appearance", exact = True).click()
    control = dialog.get_by_role("spinbutton", name = "Interface scale", exact = True)
    await control.wait_for(state = "visible", timeout = 30_000)
    await control.scroll_into_view_if_needed()
    return control


async def drive(session: Session, out_dir: Path, label: str, **_) -> tuple[list[Path], dict]:
    init = seed_init_script(session, [])
    path = "/api/settings/client-preferences"
    try:
        api_get(session, path)
        sync_available = True
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
        sync_available = False
    facts = {"saved_scale": 125, "account_sync_endpoint": sync_available}
    async with open_chat(session.base_url, init_scripts = [init], viewport = (1500, 1000)) as first:
        scale = await _scale_control(first.page)
        await scale.fill("125")
        if sync_available:
            async with first.page.expect_response(
                lambda response: response.url.endswith(path) and response.request.method == "PUT"
            ) as saved:
                await scale.press("Enter")
            response = await saved.value
            assert response.ok, await response.text()
            facts["server_saved_scale"] = (await response.json())["interfaceScale"]
            assert facts["server_saved_scale"] == 125
        else:
            await scale.press("Enter")
        await first.page.wait_for_function(
            "JSON.parse(localStorage.getItem('unsloth_interface_scale'))?.state.scale === 125"
        )
        facts["first_browser_scale"] = await scale.input_value()
    async with open_chat(session.base_url, init_scripts = [init], viewport = (1500, 1000)) as second:
        scale = await _scale_control(second.page)
        expected = "125" if sync_available else "100"
        await second.page.wait_for_function(
            "expected => document.querySelector('input[aria-label=\"Interface scale\"]')?.value === expected",
            arg = expected,
        )
        facts["fresh_browser_scale"] = await scale.input_value()
        facts["fresh_browser_local_scale"] = await second.page.evaluate(
            "JSON.parse(localStorage.getItem('unsloth_interface_scale'))?.state.scale"
        )
        out_dir.mkdir(parents = True, exist_ok = True)
        shot = out_dir / f"{label.lower()}_fresh_browser_scale.png"
        await second.screenshot(shot, full_page = False)
    return [shot], facts
