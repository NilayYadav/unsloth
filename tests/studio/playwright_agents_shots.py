# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Deterministic before/after screenshots of Settings > Agents.

Drives smoke-settings.html (the real SettingsDialog, no backend) with every API
the Agents tab reads mocked, so the same fixtures render on main and on a
branch. Measures panel scroll height, the bloat signal, and captures the tab
top, the scrolled bottom, and -- when the page offers disclosure triggers --
the fully expanded state.

    PW_OUT=/tmp/agents-shots python tests/studio/playwright_agents_shots.py
"""

import json
import os
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _playwright_robust import (  # noqa: E402
    chromium_launch_args,
    start_vite,
    stop_process,
)

PORT = int(os.environ.get("PW_PORT", "5399"))
OUT = Path(os.environ.get("PW_OUT", "logs/agents_shots"))
VIEWPORT = {"width": 1440, "height": 900}

EXAMPLE_REPO = "unsloth/Qwen3.8-27B-GGUF"

VARIANTS = {
    "repo_id": EXAMPLE_REPO,
    "has_vision": False,
    "default_variant": "UD-Q4_K_XL",
    "variants": [
        {
            "filename": "Qwen3.8-27B-UD-Q4_K_XL.gguf",
            "quant": "UD-Q4_K_XL",
            "size_bytes": 16106127360,
            "download_size_bytes": 16106127360,
        },
        {
            "filename": "Qwen3.8-27B-Q4_K_M.gguf",
            "quant": "Q4_K_M",
            "size_bytes": 16428274483,
        },
        {
            "filename": "Qwen3.8-27B-Q8_0.gguf",
            "quant": "Q8_0",
            "size_bytes": 27380416512,
        },
    ],
}

STATUS_EMPTY = {
    "active_model": None,
    "model_identifier": None,
    "is_vision": False,
    "is_gguf": False,
    "loading": [],
    "loaded": [],
}

STATUS_ACTIVE_GGUF = {
    "active_model": EXAMPLE_REPO,
    "model_identifier": f"{EXAMPLE_REPO}:UD-Q4_K_XL",
    "is_vision": False,
    "is_gguf": True,
    "gguf_variant": "UD-Q4_K_XL",
    "loading": [],
    "loaded": [f"{EXAMPLE_REPO}:UD-Q4_K_XL"],
}

SCENES = {
    # Nothing loaded, the example model preselected: what a first-time user sees.
    "default": {"status": STATUS_EMPTY, "variants": VARIANTS},
    "active-model": {"status": STATUS_ACTIVE_GGUF, "variants": VARIANTS},
}

PANEL = 'div[role="dialog"] main div.hover-scrollbar'
DIALOG = 'div[role="dialog"]'
# The redesign mounts its disclosure triggers with this attribute; main has none.
DISCLOSURE = "[data-collapsible-section]"


def log(msg: str) -> None:
    print(f"[agents-shots] {msg}", flush=True)


def install_mocks(page, scene: dict) -> None:
    def json_of(payload):
        def handler(route):
            return route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(payload),
            )

        return handler

    page.route("**/api/models/list*", json_of({"models": []}))
    page.route("**/api/inference/status*", json_of(scene["status"]))
    page.route("**/api/models/local*", json_of({"models": []}))
    page.route("**/api/hub/cached-gguf*", json_of({"cached": []}))
    page.route("**/api/models/gguf-variants*", json_of(scene["variants"]))
    page.route("**/api/settings/coding-agents*", json_of({"agents": [], "detected": []}))
    page.route("https://huggingface.co/api/models**", json_of([]))


def facts(page) -> dict:
    return page.evaluate(
        """(sel) => {
            const panel = document.querySelector(sel.panel);
            if (!panel) return { present: false };
            const labels = [...panel.querySelectorAll('[data-settings-label]')]
                .map((el) => el.dataset.settingsLabel);
            return {
                present: true,
                scrollHeight: panel.scrollHeight,
                clientHeight: panel.clientHeight,
                overflowPx: panel.scrollHeight - panel.clientHeight,
                labels,
                codeBlocks: panel.querySelectorAll('pre code, code.block').length,
                disclosures: panel.querySelectorAll(sel.disclosure).length,
                textLength: (panel.innerText || '').trim().length,
            };
        }""",
        {"panel": PANEL, "disclosure": DISCLOSURE},
    )


def scroll_panel(page, top: bool) -> None:
    page.evaluate(
        """([sel, toTop]) => {
            const p = document.querySelector(sel);
            if (p) p.scrollTop = toTop ? 0 : p.scrollHeight;
        }""",
        [PANEL, top],
    )


def shoot_scene(page, name: str, scene: dict, report: dict) -> None:
    install_mocks(page, scene)
    page.goto(f"http://127.0.0.1:{PORT}/smoke-settings.html", wait_until="domcontentloaded")
    page.wait_for_function("() => !!window.__settingsSmoke", timeout=120000)
    page.wait_for_timeout(2500)
    page.evaluate("() => window.__settingsSmoke.open('agents')")
    page.wait_for_selector(DIALOG, timeout=15000)
    # the generated command is the last async piece (model discovery + variants)
    page.wait_for_function(
        """() => {
            const panel = document.querySelector('div[role="dialog"] main div.hover-scrollbar');
            return panel && panel.innerText.includes('unsloth');
        }""",
        timeout=30000,
    )
    page.wait_for_timeout(1500)

    entry: dict = {"top": facts(page)}
    page.locator(DIALOG).screenshot(path=str(OUT / f"{name}-top.png"))

    scroll_panel(page, top=False)
    page.wait_for_timeout(400)
    entry["bottom"] = facts(page)
    page.locator(DIALOG).screenshot(path=str(OUT / f"{name}-bottom.png"))
    scroll_panel(page, top=True)

    triggers = page.locator(DISCLOSURE)
    if triggers.count() > 0:
        for i in range(triggers.count()):
            triggers.nth(i).click()
        page.wait_for_timeout(500)
        entry["expanded"] = facts(page)
        page.locator(DIALOG).screenshot(path=str(OUT / f"{name}-expanded.png"))
        scroll_panel(page, top=False)
        page.wait_for_timeout(300)
        page.locator(DIALOG).screenshot(path=str(OUT / f"{name}-expanded-bottom.png"))
        scroll_panel(page, top=True)

    report[name] = entry
    top = entry["top"]
    log(
        f"{name}: overflow {top.get('overflowPx')}px, {len(top.get('labels', []))} labels, "
        f"{top.get('codeBlocks')} code blocks, {top.get('disclosures')} disclosures"
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {}
    vite = start_vite(PORT)
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True, args=chromium_launch_args())
            ctx = browser.new_context(viewport=VIEWPORT, reduced_motion="reduce")
            page = ctx.new_page()
            page.goto(f"http://127.0.0.1:{PORT}/smoke-settings.html", wait_until="domcontentloaded")
            page.wait_for_function("() => !!window.__settingsSmoke", timeout=120000)
            page.wait_for_timeout(3000)
            for name, scene in SCENES.items():
                try:
                    shoot_scene(page, name, scene, report)
                except Exception as exc:
                    report[name] = {"error": f"{type(exc).__name__}: {exc}"}
                    log(f"FAIL {name}: {report[name]['error']}")
            browser.close()
    finally:
        stop_process(vite)
    (OUT / "facts.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"facts -> {OUT / 'facts.json'}")
    return 1 if any("error" in v for v in report.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
