"""Scene: a profile / appearance save in one Studio tab reaching an already open tab (PR 16580).

Two tabs of one browser context (storage events only cross tabs of one origin). Tab A writes
the persisted `unsloth_user_profile` and `unsloth_appearance_customization` payloads exactly as
its zustand persist would; tab B is then photographed without a reload.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

WORKSPACE = Path(
    os.environ.get("WORKSPACE")
    or os.environ.get("UNSLOTH_WORKSPACE")
    or Path(__file__).resolve().parents[2]
)
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "scripts"))

from playwright.async_api import async_playwright  # noqa: E402

from pr_ui_scenes._common import Session  # noqa: E402
from studio_test_kit.auth import seed_init_script  # noqa: E402

GREETING = "Hello from the other window"
ACCENT = "#d0183c"

_WRITE_JS = """([greeting, accent]) => {
  const pkey = "unsloth_user_profile";
  const p = JSON.parse(localStorage.getItem(pkey) || '{"state":{},"version":0}');
  p.state = {...p.state, greetingStyle: "text", greetingText: greeting, showGreetingSloth: true};
  localStorage.setItem(pkey, JSON.stringify(p));
  const akey = "unsloth_appearance_customization";
  const a = JSON.parse(localStorage.getItem(akey) || '{"state":{"customization":{}},"version":9}');
  const c = a.state.customization || {};
  const colors = c.colors || {};
  c.colors = {...colors, light: {...(colors.light || {}), accent}, dark: {...(colors.dark || {}), accent}};
  a.state.customization = c;
  localStorage.setItem(akey, JSON.stringify(a));
}"""

_READ_JS = """() => {
  const root = document.querySelector('.aui-thread-welcome-root');
  const cs = getComputedStyle(document.documentElement);
  return {welcome_text: root ? root.innerText.slice(0, 300) : null,
          primary: cs.getPropertyValue('--primary').trim(),
          control_accent: cs.getPropertyValue('--control-accent').trim()};
}"""


async def drive(
    session: Session,
    out_dir: Path,
    label: str,
    route: str = "/chat",
    **_: object,
) -> tuple[list[Path], dict]:
    out_dir.mkdir(parents = True, exist_ok = True)
    auth = type(
        "A", (), {"access_token": session.access_token, "refresh_token": session.refresh_token}
    )()
    shots: list[Path] = []
    facts: dict = {}
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless = True)
        ctx = await browser.new_context(viewport = {"width": 1400, "height": 900})
        await ctx.add_init_script(seed_init_script(auth, []))
        tab_b = await ctx.new_page()
        await tab_b.goto(f"{session.base_url}{route}", wait_until = "domcontentloaded")
        await tab_b.wait_for_selector(".aui-thread-welcome-root", timeout = 60000)
        await tab_b.wait_for_timeout(2000)
        facts["tab_b_before"] = await tab_b.evaluate(_READ_JS)
        tab_a = await ctx.new_page()
        await tab_a.goto(f"{session.base_url}{route}", wait_until = "domcontentloaded")
        await tab_a.wait_for_timeout(3000)
        await tab_a.evaluate(_WRITE_JS, [GREETING, ACCENT])
        await tab_b.bring_to_front()
        await tab_b.wait_for_timeout(2500)
        after = await tab_b.evaluate(_READ_JS)
        facts["tab_b_after_other_tab_saved"] = after
        facts["tab_b_shows_greeting"] = GREETING in (after["welcome_text"] or "")
        facts["tab_b_accent_changed"] = after["primary"] != facts["tab_b_before"]["primary"]
        p = out_dir / f"{label.lower()}_tab_b.png"
        await tab_b.screenshot(path = str(p))
        shots.append(p)
        await browser.close()
    (out_dir / f"{label.lower()}_facts.json").write_text(json.dumps(facts, indent = 2))
    return shots, facts
