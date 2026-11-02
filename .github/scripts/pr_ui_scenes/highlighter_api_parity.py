"""Capture real highlighted API examples without loading a model."""

from pathlib import Path
import hashlib
import re
from types import SimpleNamespace

from studio_test_kit.auth import seed_init_script
from studio_test_kit.ui import open_chat
from studio_regress.journeys.settings import open_settings


async def drive(session, out_dir: Path, label: str, **kwargs):
    init = seed_init_script(
        SimpleNamespace(access_token = session.access_token, refresh_token = session.refresh_token), []
    )
    async with open_chat(
        session.base_url, init_scripts = [init], viewport = (1500, 1100), headless = True
    ) as sp:
        page = sp.page
        await open_settings(SimpleNamespace(page = page, base_url = session.base_url))
        dialog = page.get_by_role("dialog")
        await dialog.get_by_role("button", name = "API", exact = True).click()
        await dialog.get_by_role("button", name = "Python", exact = True).first.click()
        code = dialog.locator("pre code").first
        await code.wait_for(state = "visible", timeout = 30000)
        await page.wait_for_function(
            """() => Array.from(document.querySelectorAll('[role=dialog] pre code span[style]')).some(n => /#[0-9a-f]{3,8}/i.test(n.getAttribute('style')))""",
            timeout = 30000,
        )
        await code.scroll_into_view_if_needed()
        text = await code.inner_text()
        assert "curl" in text or "import" in text, text
        tokens = await code.locator("span[style]").evaluate_all(
            "nodes => nodes.map(n => ({text:n.textContent,style:n.getAttribute('style')}))"
        )
        assert len(tokens) > 5
        assert any("#" in t["style"] for t in tokens), tokens
        normalized = re.sub(r"127\.0\.0\.1:\d+|localhost:\d+", "localhost:PORT", text)
        facts = {
            "highlighted_token_count": len(tokens),
            "code_sha256": hashlib.sha256(normalized.encode()).hexdigest(),
            "has_real_code": True,
            "token_styles": [t["style"] for t in tokens],
        }
        shot = out_dir / f"{label}_api_highlight.png"
        await dialog.screenshot(path = str(shot))
        return [shot], facts
