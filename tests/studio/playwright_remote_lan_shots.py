# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Deterministic before/after screenshots of Settings > Remote & LAN.

Drives smoke-settings.html (the real SettingsDialog, no backend) with every
Remote & LAN API mocked, so the same fixtures render on main and on a branch.

    PW_OUT=/tmp/shots python tests/studio/playwright_remote_lan_shots.py

Writes <scene>-top.png / <scene>-bottom.png per scene plus facts.json with the
panel scroll height and visible row counts -- the quantitative bloat signal.
"""

import json
import os
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _playwright_robust import (  # noqa: E402
    chromium_launch_args,
    start_vite,
    stop_process,
)

PORT = int(os.environ.get("PW_PORT", "5399"))
OUT = Path(os.environ.get("PW_OUT", "logs/remote_lan_shots"))
VIEWPORT = {"width": 1440, "height": 900}

LAN_ONLINE = {
    "state": "online",
    "urls": ["http://172.20.2.161:8000"],
    "public_urls": [],
    "error": None,
    "auto_start": False,
    "configured_port": None,
    "active_port": 8000,
    "managed_by": "launch",
    "can_start": False,
    "can_stop": True,
    "block_reason": "launch_managed",
    "bind_host": "0.0.0.0",
    "wildcard_bind": True,
    "serves_web_ui": True,
    "keyless_lan_eligible": True,
    "keyless_scope": "off",
    "keyless_tools": False,
}

LAN_OFF = {
    "state": "off",
    "urls": [],
    "public_urls": [],
    "error": None,
    "auto_start": False,
    "configured_port": None,
    "active_port": None,
    "managed_by": None,
    "can_start": True,
    "can_stop": False,
    "block_reason": None,
    "bind_host": None,
    "wildcard_bind": False,
    "serves_web_ui": True,
    "keyless_lan_eligible": True,
    "keyless_scope": "off",
    "keyless_tools": False,
}

REMOTE_OFF = {
    "state": "off",
    "url": None,
    "error": None,
    "auto_start": False,
    "default_auto_start": False,
    "available": True,
    "managed_by": None,
    "can_start": True,
    "can_stop": False,
    "block_reason": None,
    "password_pending": False,
    "streaming_supported": True,
}

REMOTE_ONLINE = {
    **REMOTE_OFF,
    "state": "online",
    "url": "https://posing-violent-michael-preference.trycloudflare.com",
    "managed_by": "settings",
    "can_start": False,
    "can_stop": True,
}

KEYLESS_OFF = {"scope": "off", "tools": False, "exposure": None}
KEYLESS_INFERENCE = {"scope": "inference", "tools": False, "exposure": "private_lan"}

SCENES = {
    # Remote off, LAN online and launch-managed: the state that motivated the redesign.
    "mixed": {"lan": LAN_ONLINE, "remote": REMOTE_OFF, "keyless": KEYLESS_OFF},
    "all-off": {"lan": LAN_OFF, "remote": REMOTE_OFF, "keyless": KEYLESS_OFF},
    "both-on": {
        "lan": {**LAN_ONLINE, "managed_by": "settings", "block_reason": None},
        "remote": REMOTE_ONLINE,
        "keyless": KEYLESS_INFERENCE,
    },
}

PANEL = 'div[role="dialog"] main div.hover-scrollbar'
DIALOG = 'div[role="dialog"]'


def log(msg: str) -> None:
    print(f"[remote-lan-shots] {msg}", flush=True)


API_KEYS_LIST = {
    "api_keys": [
        {
            "id": "k1",
            "name": "laptop",
            "prefix": "us-abc123",
            "created_at": "2026-09-01T10:00:00Z",
            "last_used_at": None,
            "expires_at": None,
            "is_active": True,
        }
    ]
}

AUTO_SWITCH = {"enabled": True}


def install_mocks(page, scene: dict) -> None:
    def json_route(payload):
        def handler(route):
            if route.request.method in ("GET", "HEAD"):
                return route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps(payload),
                )
            return route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(payload),
            )

        return handler

    page.route("**/api/settings/lan-access*", json_route(scene["lan"]))
    page.route("**/api/settings/remote-access*", json_route(scene["remote"]))
    page.route("**/api/settings/keyless-api-access*", json_route(scene["keyless"]))
    page.route("**/api/auth/api-keys*", json_route(API_KEYS_LIST))
    page.route("**/api/settings/openai-auto-switch*", json_route(AUTO_SWITCH))


def facts(page) -> dict:
    return page.evaluate(
        """(sel) => {
            const panel = document.querySelector(sel);
            if (!panel) return { present: false };
            const rows = [...panel.querySelectorAll('[data-settings-label]')]
                .map((el) => el.dataset.settingsLabel);
            const sections = [...panel.querySelectorAll('section[data-settings-label]')]
                .map((el) => el.dataset.settingsLabel);
            return {
                present: true,
                scrollHeight: panel.scrollHeight,
                clientHeight: panel.clientHeight,
                overflowPx: panel.scrollHeight - panel.clientHeight,
                rows,
                sections,
                textLength: (panel.innerText || '').trim().length,
            };
        }""",
        PANEL,
    )


def shoot_scene(page, name: str, scene: dict, report: dict) -> None:
    install_mocks(page, scene)
    page.goto(f"http://127.0.0.1:{PORT}/smoke-settings.html", wait_until="domcontentloaded")
    page.wait_for_function("() => !!window.__settingsSmoke", timeout=120000)
    page.wait_for_timeout(2500)
    page.evaluate("() => window.__settingsSmoke.open('remote-lan')")
    page.wait_for_selector(DIALOG, timeout=15000)
    page.wait_for_selector('section[data-settings-label="LAN access"]', timeout=15000)
    # polling sections re-render after the first load; let them settle
    page.wait_for_timeout(1500)

    report[name] = {"top": facts(page)}
    page.locator(DIALOG).screenshot(path=str(OUT / f"{name}-top.png"))

    page.evaluate(
        """(sel) => { const p = document.querySelector(sel); if (p) p.scrollTop = p.scrollHeight; }""",
        PANEL,
    )
    page.wait_for_timeout(400)
    report[name]["bottom"] = facts(page)
    page.locator(DIALOG).screenshot(path=str(OUT / f"{name}-bottom.png"))
    page.evaluate(
        """(sel) => { const p = document.querySelector(sel); if (p) p.scrollTop = 0; }""",
        PANEL,
    )
    log(
        f"{name}: overflow {report[name]['top'].get('overflowPx')}px, "
        f"{len(report[name]['top'].get('rows', []))} rows, "
        f"sections {report[name]['top'].get('sections')}"
    )


def shoot_api_keys_keyless(page, scene: dict, report: dict) -> None:
    """The keyless panel's home after the redesign: Settings > API keys."""
    install_mocks(page, scene)
    page.goto(f"http://127.0.0.1:{PORT}/smoke-settings.html", wait_until="domcontentloaded")
    page.wait_for_function("() => !!window.__settingsSmoke", timeout=120000)
    page.wait_for_timeout(2500)
    page.evaluate("() => window.__settingsSmoke.open('api-keys')")
    section = page.locator('section[data-settings-label="Keyless API access"]')
    section.wait_for(timeout=15000)
    page.wait_for_timeout(1500)
    section.scroll_into_view_if_needed()
    page.wait_for_timeout(300)
    section.screenshot(path=str(OUT / "api-keys-keyless.png"))
    report["api-keys-keyless"] = {"present": True}
    log("api-keys-keyless: keyless panel intact in API keys tab")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {}
    vite = start_vite(PORT)
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True, args=chromium_launch_args())
            ctx = browser.new_context(viewport=VIEWPORT, reduced_motion="reduce")
            page = ctx.new_page()
            # warm the vite dep graph once so scene loads are clean
            page.goto(f"http://127.0.0.1:{PORT}/smoke-settings.html", wait_until="domcontentloaded")
            page.wait_for_function("() => !!window.__settingsSmoke", timeout=120000)
            page.wait_for_timeout(3000)
            for name, scene in SCENES.items():
                try:
                    shoot_scene(page, name, scene, report)
                except Exception as exc:
                    report[name] = {"error": f"{type(exc).__name__}: {exc}"}
                    log(f"FAIL {name}: {report[name]['error']}")
            try:
                shoot_api_keys_keyless(page, SCENES["both-on"], report)
            except Exception as exc:
                report["api-keys-keyless"] = {"error": f"{type(exc).__name__}: {exc}"}
                log(f"FAIL api-keys-keyless: {report['api-keys-keyless']['error']}")
            browser.close()
    finally:
        stop_process(vite)
    (OUT / "facts.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"facts -> {OUT / 'facts.json'}")
    return 1 if any("error" in v for v in report.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
