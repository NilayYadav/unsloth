import json
from pathlib import Path

from pr_ui_scenes._common import api_get
from studio_test_kit.auth import seed_init_script
from studio_test_kit.ui import open_chat


async def drive(session, out_dir: Path, label: str, **kwargs):
    inventory = api_get(session, "/api/settings/caches?refresh=true")
    hub = next(row for row in inventory["caches"] if row["key"] == "hf_hub")
    breakdown = api_get(session, "/api/settings/caches/hf_hub/breakdown")
    facts = {
        "clear_bytes": hub["clear_size_bytes"],
        "clear_entries": hub["clear_entry_count"],
        "models": sorted(row.get("repo_id", row["name"]) for row in breakdown["entries"]),
    }
    (out_dir / "facts.json").write_text(
        json.dumps({"hub": hub, "breakdown": breakdown}), encoding = "utf-8"
    )
    init = seed_init_script(session, [])
    async with open_chat(
        session.base_url, init_scripts = [init], viewport = (1440, 1100), headless = True
    ) as sp:
        page = sp.page
        await page.get_by_role("button", name = "Settings", exact = True).click()
        dialog = page.get_by_role("dialog")
        await dialog.get_by_role("button", name = "System", exact = True).click()
        await dialog.get_by_role("button", name = "Details", exact = True).click()
        target = dialog.get_by_text("Hugging Face model cache", exact = True).locator(
            "xpath=ancestor::div[contains(@class,'border-t')][1]"
        )
        await target.get_by_role("button", name = "Show folders", exact = True).click()
        expected = "test/active" if facts["clear_entries"] else "new-downloads"
        await target.get_by_text(expected, exact = True).wait_for()
        facts["clear_enabled"] = await target.get_by_role(
            "button", name = "Clear", exact = True
        ).first.is_enabled()
        await target.scroll_into_view_if_needed()
        shot = out_dir / f"{label.lower()}_nested_cache.png"
        await target.screenshot(path = str(shot))
    return [shot], facts
