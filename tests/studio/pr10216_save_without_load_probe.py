# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""PR #10216 A/B probe: save a model's run settings WITHOUT loading it.

The subject is issue #10168: the run-settings panel only persisted settings as part of
loading the model, so setting a model up ahead of time meant loading it every time.

The model under test is primed into HF_HOME by the workflow and is never loaded, by this
probe or by anything else, so "not loaded" is the state the whole run starts in rather
than something undone afterwards.

Gates (all HARD):

  1. nothing is loaded before the interaction (GET /api/inference/status.active_model
     is empty). Without this the "still unloaded" gate below proves nothing.
  2. the panel offers a save that is not a load: with "Remember for this model" ticked
     and a distinctive Context Length staged, a button named exactly "Save settings"
     is present ALONGSIDE the primary "Load model". This is the assertion the
     pre-PR branch must fail on: there, the only committing control is Load.
  3. clicking it reports success ("Settings saved." toast).
  4. the settings reach localStorage: unsloth_model_configs has an entry keyed to
     (repo, quant) carrying customContextLength == the distinctive value.
  5. the settings reach the server copy an API load reads:
     GET /api/settings/openai-auto-switch/overrides carries the same value for a key
     naming this model.
  6. the model was NOT loaded: zero POST /api/inference/load left the page, and
     /api/inference/status.active_model is still empty.
  7. it survives a reload: reopening the panel after a full browser reload shows the
     distinctive Context Length, with the model still unloaded.

Runs as a plain script (not via pytest), mirroring tests/studio/playwright_model_config.py,
whose picker selectors and settle window this reuses: accumulate failures in `_failed`,
exit non-zero if any fired.
"""

import json
import re
import sys
import os
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _playwright_robust import (  # noqa: E402
    chromium_launch_args,
    click_and_wait_for_response,
    dump_diagnostics,
    evaluate_fetch,
    install_view_transition_killer,
    install_wall_clock_watchdog,
    is_benign_page_error,
    recover_or_replace_page,
    robust_evaluate,
    wait_for_health,
)

BASE = os.environ["BASE_URL"]
NEW = os.environ.get("STUDIO_NEW_PW", "SaveNoLoad-NEW-2026!")
GGUF_REPO = os.environ.get("GGUF_REPO", "unsloth/gemma-3-270m-it-GGUF")
GGUF_VARIANT = os.environ.get("GGUF_VARIANT", "UD-Q4_K_XL")
MODEL_HINT = os.environ.get("STUDIO_MODEL_HINT", "gemma-3-270m")
DISTINCT_CTX = int(os.environ.get("STUDIO_DISTINCT_CTX", "4096"))
ART_DIR = os.environ.get("PW_ART_DIR", "logs/playwright_save_without_load")
# Same measured settle as playwright_model_config.py: an edit staged in the panel's first
# moments is dropped when it re-derives its baseline, and the panel exposes no readiness
# signal to poll.
CONFIG_SETTLE_MS = int(os.environ.get("STUDIO_CONFIG_SETTLE_MS", "1000"))
ART = Path(ART_DIR)
ART.mkdir(parents = True, exist_ok = True)
WALL_TIMEOUT_S = float(os.environ.get("STUDIO_UI_WALL_TIMEOUT_S", "720"))
FETCH_TIMEOUT_MS = int(os.environ.get("STUDIO_UI_FETCH_TIMEOUT_MS", "30000"))
OVERRIDES_URL = "/api/settings/openai-auto-switch/overrides"

_n = [0]
_failed: list[str] = []


_LOCAL_PATH_PREFIX_RE = re.compile(
    r"^(?:/|\.{1,2}(?:$|[\\/])|~(?:$|[\\/])|~[^\\/]+[\\/]|[A-Za-z]:[\\/]|\\\\)"
)


def _normalize_model_identity(model_id: str) -> str:
    """Mirror of normalizeModelIdentity for the hub-id case, which is all this needs:
    the model under test is a repo id, never a local path."""
    trimmed = model_id.strip()
    if not (trimmed and _LOCAL_PATH_PREFIX_RE.match(trimmed)):
        return trimmed.lower()
    return trimmed


def step(s: str) -> None:
    print(f"[pr10216] STEP {s}", flush = True)


def info(s: str) -> None:
    print(f"[pr10216] {s}", flush = True)


def fail(m: str) -> None:
    print(f"[pr10216] FAIL: {m}", flush = True)
    _failed.append(m)


def _count(loc) -> int:
    try:
        return loc.count()
    except Exception as exc:
        info(f"WARN: locator raised (not a missing element): {type(exc).__name__}: {exc}")
        return 0


def _as_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value).replace(",", "").strip())
    except Exception:
        return None


with sync_playwright() as p:
    _watchdog = install_wall_clock_watchdog(WALL_TIMEOUT_S, label = "pr10216", info = info)
    wait_for_health(BASE, timeout = 30.0, info = info)
    browser = p.chromium.launch(headless = True, args = chromium_launch_args())
    ctx = browser.new_context(
        viewport = {"width": 1280, "height": 900},
        reduced_motion = "reduce",
    )
    install_view_transition_killer(ctx)
    page = ctx.new_page()
    page.set_default_timeout(60_000)
    page_errors: list[str] = []

    def _on_pageerror(e):
        msg = str(e)
        if is_benign_page_error(msg):
            info(f"WARN ignoring benign pageerror: {msg!r}")
            return
        page_errors.append(msg)

    page.on("pageerror", _on_pageerror)

    # Every load the page attempts, and every override it mirrors. The load list is the
    # gate: this feature's whole claim is that the list stays empty.
    load_posts: list[str] = []
    override_puts: list[str] = []

    def _on_request(req):
        try:
            if req.method != "POST" and req.method != "PUT":
                return
            if req.method == "POST" and "/api/inference/load" in req.url:
                load_posts.append(req.post_data or "")
            elif req.method == "PUT" and OVERRIDES_URL in req.url:
                override_puts.append(req.post_data or "")
        except Exception:
            pass

    page.on("request", _on_request)

    def shoot(name: str) -> None:
        _n[0] += 1
        try:
            page.screenshot(
                path = str(ART / f"{_n[0]:02d}-{name}.png"),
                full_page = True,
                timeout = 90_000,
                animations = "disabled",
            )
        except Exception as _shoot_err:
            info(f"WARN: screenshot {name} failed: {_shoot_err}")

    def read_configs() -> dict:
        raw = robust_evaluate(page, "() => localStorage.getItem('unsloth_model_configs')")
        if not raw:
            return {}
        try:
            data = json.loads(raw)
        except Exception as exc:
            fail(f"unsloth_model_configs is unreadable ({exc}); raw={str(raw)[:200]!r}")
            return {}
        return data if isinstance(data, dict) else {}

    def entries_for_model(cfg: dict) -> list[dict]:
        """Only the entries keyed to (repo, quant) under test, parsed from the key.

        Scanning every entry would let another model's value satisfy the save gate.
        """
        want = (_normalize_model_identity(GGUF_REPO), GGUF_VARIANT.strip().lower())
        recognised = [k for k in cfg if re.match(r"^v\d+:\[", str(k))]
        if not recognised:
            return [v for v in cfg.values() if isinstance(v, dict)]
        matched = []
        for key in recognised:
            try:
                parts = json.loads(str(key).split(":", 1)[1])
            except Exception:
                continue
            if not isinstance(parts, list) or not parts:
                continue
            raw = (list(parts) + [""])[:2]
            got = (_normalize_model_identity(str(raw[0])), str(raw[1]).strip().lower())
            if got == want and isinstance(cfg[key], dict):
                matched.append(cfg[key])
        return matched

    def api_get(path: str) -> dict:
        tok = robust_evaluate(page, "() => localStorage.getItem('unsloth_auth_token')")
        resp = evaluate_fetch(
            page,
            f"{BASE}{path}",
            method = "GET",
            headers = {"Authorization": f"Bearer {tok}"},
            timeout_ms = FETCH_TIMEOUT_MS,
        )
        if resp.get("error"):
            fail(f"GET {path} failed: {resp['error']!r}")
            return {}
        if resp.get("status") != 200:
            fail(f"GET {path} -> {resp.get('status')}: {str(resp.get('body'))[:200]!r}")
            return {}
        body = resp.get("body")
        return body if isinstance(body, dict) else {}

    def active_model() -> str:
        st = api_get("/api/inference/status")
        for key in ("active_model", "model_identifier"):
            val = st.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        return ""

    # ─────────────────────────────────────────────────────
    step("setup: change-password")
    page.goto(f"{BASE}/change-password", wait_until = "domcontentloaded", timeout = 60_000)
    try:
        page.wait_for_load_state("networkidle", timeout = 30_000)
    except Exception:
        pass
    pw_field = page.locator("#new-password")
    pw_field.wait_for(state = "visible", timeout = 60_000)
    pw_field.fill(NEW, timeout = 60_000)
    page.fill("#confirm-password", NEW, timeout = 60_000)
    status, _ = click_and_wait_for_response(
        page,
        url_substr = "/api/auth/change-password",
        method = "POST",
        do_click = lambda: page.locator('button[type="submit"]').click(),
        timeout_ms = 30_000,
        info = lambda m: print(f"[pr10216]   {m}", flush = True),
    )
    if status is not None and status >= 400:
        raise AssertionError(f"change-password POST returned {status}")

    try:
        page.wait_for_load_state("networkidle", timeout = 30_000)
    except Exception:
        pass
    composer = page.locator('textarea[aria-label="Message input"]')
    try:
        composer.wait_for(state = "visible", timeout = 60_000)
    except Exception:
        page = recover_or_replace_page(
            page, ctx, default_timeout_ms = 60_000, goto_url = BASE,
            settle_networkidle = True,
            info = lambda m: print(f"[pr10216]   recovery: {m}", flush = True),
        )
        page.on("request", _on_request)
        page.locator('textarea[aria-label="Message input"]').wait_for(
            state = "visible", timeout = 60_000
        )
    shoot("01-chat-loaded")

    # ─────────────────────────────────────────────────────
    # 1. Nothing is loaded to begin with (HARD).
    # ─────────────────────────────────────────────────────
    step("no model is loaded before the interaction")
    before_active = active_model()
    if before_active:
        fail(f"a model was already loaded before the probe touched anything: {before_active!r}")
    else:
        info("OK precondition: /api/inference/status reports no active model")

    # ─────────────────────────────────────────────────────
    # Picker helpers (same proven selectors as playwright_model_config.py).
    # ─────────────────────────────────────────────────────
    POPOVER = '[data-tour="chat-model-selector-popover"]'
    TRIGGER = '[data-tour="chat-model-selector"]'
    GEAR_ANY = 'button[aria-label^="Inference settings for" i]'
    SOLE_QUANT_SETTLE_MS = 30_000
    QUANT_GEAR_MS = 2_000

    def diagnose(name, selector):
        rows = []
        try:
            opts = page.locator("[data-model-picker-option]")
            rows = [
                (opts.nth(i).inner_text() or "").strip()[:60] for i in range(min(opts.count(), 12))
            ]
        except Exception:
            pass
        gears = []
        try:
            g = page.locator(GEAR_ANY)
            gears = [g.nth(i).get_attribute("aria-label") for i in range(min(g.count(), 12))]
        except Exception:
            pass
        dump_diagnostics(
            page, ART, name, info = info,
            extra = {"missed_selector": selector, "option_rows": rows, "gear_labels": gears},
        )

    def open_picker():
        popover = page.locator(POPOVER).first
        if _count(popover) == 0 or not popover.is_visible():
            page.locator(TRIGGER).first.click()
            page.wait_for_timeout(900)
            popover = page.locator(POPOVER).first
        popover.wait_for(state = "visible", timeout = 30_000)
        return popover

    def close_picker():
        try:
            page.keyboard.press("Escape")
            page.wait_for_timeout(400)
        except Exception:
            pass

    def reveal_on_device_row(popover, hint):
        od = page.get_by_role("tab", name = "On Device").first
        if _count(od):
            od.click()
            page.wait_for_timeout(700)
        row = popover.locator("[data-model-picker-option]", has_text = hint).first
        if _count(row) == 0:
            search = popover.locator("[data-model-picker-search-input]").first
            if _count(search):
                search.click()
                search.fill(hint)
                page.wait_for_timeout(700)
                row = popover.locator("[data-model-picker-option]", has_text = hint).first
        return row if _count(row) else None

    def config_is_open(popover):
        return _count(popover.get_by_role("button", name = "Back to model list")) > 0

    def row_gear(popover, hint, quant = None, timeout_ms = SOLE_QUANT_SETTLE_MS):
        pattern = f"^Inference settings for .*{re.escape(hint)}"
        if quant:
            pattern += f".* {re.escape(quant)}$"
        gear = popover.get_by_role("button", name = re.compile(pattern, re.IGNORECASE)).first
        try:
            gear.wait_for(state = "visible", timeout = timeout_ms)
        except Exception:
            return None
        return gear

    def open_config(popover, hint):
        """Open the run-settings panel WITHOUT selecting the row.

        Clicking the row is what playwright_model_config.py falls back to, and it is not
        available here: a sole-quant row selects the model on click, and this probe's
        subject is a model nothing has touched. The gear alone, or nothing.
        """
        if reveal_on_device_row(popover, hint) is None:
            return None
        gear = row_gear(popover, hint, quant = GGUF_VARIANT, timeout_ms = QUANT_GEAR_MS)
        if gear is None:
            gear = row_gear(popover, hint)
        if gear is None:
            diagnose("no-gear-for-model", GEAR_ANY)
            return None
        gear.click()
        for _ in range(20):
            if config_is_open(popover):
                page.wait_for_timeout(CONFIG_SETTLE_MS)
                return popover
            page.wait_for_timeout(250)
        diagnose("open-config-not-open", 'button[name="Back to model list"]')
        return None

    def context_input(popover):
        for role in ("textbox", "spinbutton"):
            loc = popover.get_by_role(role, name = "Context Length").first
            if _count(loc):
                return loc
        loc = popover.locator('input[aria-label="Context Length"]').first
        return loc if _count(loc) else None

    def button_named(popover, name):
        b = popover.get_by_role("button", name = name, exact = True).first
        return b if _count(b) else None

    # ─────────────────────────────────────────────────────
    # 2. A save that is not a load exists (HARD -- the A/B assertion).
    # ─────────────────────────────────────────────────────
    step(f"run-settings for an unloaded {MODEL_HINT} offers a save that is not a load")
    popover = open_picker()
    shoot("02-picker-open")
    if open_config(popover, MODEL_HINT) is None:
        fail(f"could not open run-settings for an unloaded model matching {MODEL_HINT!r}")
        sys.exit(1)
    shoot("03-config-open")

    remember = popover.get_by_label("Remember for this model").first
    if _count(remember) == 0:
        fail("'Remember for this model' checkbox not found")
        sys.exit(1)
    try:
        remember.check()
    except Exception:
        remember.click()
    ctx_in = context_input(popover)
    if ctx_in is None:
        fail("Context Length input not found in run-settings")
        sys.exit(1)
    ctx_in.click()
    ctx_in.fill(str(DISTINCT_CTX))
    page.wait_for_timeout(300)
    shoot("04-ctx-staged")

    load_btn = button_named(popover, "Load model") or button_named(popover, "Reload model")
    save_btn = button_named(popover, "Save settings")
    if load_btn is None:
        fail("no Load/Reload button in run-settings; the panel is not in the state under test")
    if save_btn is None:
        # This is the pre-PR state and the exact assertion the negative branch must fail on:
        # the panel's only committing control is the one that loads the model.
        labels = []
        try:
            btns = popover.get_by_role("button")
            labels = [
                (btns.nth(i).inner_text() or "").strip()[:40] for i in range(min(btns.count(), 20))
            ]
        except Exception:
            pass
        fail(
            "no 'Save settings' button next to Load: run settings cannot be saved without "
            f"loading the model (buttons present: {labels!r})"
        )
        diagnose("no-save-settings-button", 'button[name="Save settings"]')
        shoot("05-no-save-button")
        print(f"[pr10216] RESULT: FAIL ({len(_failed)} gate(s))", flush = True)
        for m in _failed:
            print(f"[pr10216]   - {m}", flush = True)
        sys.exit(1)
    if not save_btn.is_enabled():
        fail("'Save settings' is present but disabled with Remember ticked and a context staged")
        shoot("05-save-disabled")
        print(f"[pr10216] RESULT: FAIL ({len(_failed)} gate(s))", flush = True)
        sys.exit(1)
    info("OK save-control: 'Save settings' present and enabled alongside 'Load model'")

    # ─────────────────────────────────────────────────────
    # 3-6. It saves, and it does not load (HARD).
    # ─────────────────────────────────────────────────────
    step("Save settings persists the settings and leaves the model unloaded")
    save_btn.click()
    page.wait_for_timeout(3000)
    shoot("06-after-save")

    toast = page.get_by_text("Settings saved.", exact = False).first
    if _count(toast):
        info("OK toast: 'Settings saved.'")
    else:
        fail("no 'Settings saved.' toast after clicking Save settings")

    cfg = read_configs()
    entries = entries_for_model(cfg)
    if any(_as_int(e.get("customContextLength")) == DISTINCT_CTX for e in entries):
        info(f"OK local: unsloth_model_configs carries customContextLength={DISTINCT_CTX}")
    else:
        fail(
            f"context {DISTINCT_CTX} not stored in unsloth_model_configs for "
            f"{GGUF_REPO}/{GGUF_VARIANT} (entries={json.dumps(entries)[:400]})"
        )

    # The server copy is the half that makes an API load use these settings, which is the
    # part a purely-local save would silently skip.
    ov_body = api_get(OVERRIDES_URL)
    overrides = ov_body.get("overrides") if isinstance(ov_body.get("overrides"), dict) else {}
    want_repo = _normalize_model_identity(GGUF_REPO)
    server_hit = None
    for key, entry in (overrides or {}).items():
        if not isinstance(entry, dict):
            continue
        if want_repo not in str(key).lower():
            continue
        for field in ("custom_context_length", "max_seq_length"):
            if _as_int(entry.get(field)) == DISTINCT_CTX:
                server_hit = (key, field)
                break
        if server_hit:
            break
    if server_hit:
        info(f"OK server: {OVERRIDES_URL} has {server_hit[1]}={DISTINCT_CTX} under {server_hit[0]!r}")
    else:
        fail(
            f"server override missing {DISTINCT_CTX} for {GGUF_REPO}; "
            f"PUTs seen={len(override_puts)}; overrides={json.dumps(overrides)[:400]}"
        )

    if load_posts:
        fail(f"Save settings triggered {len(load_posts)} /api/inference/load POST(s): {load_posts!r}")
    else:
        info("OK no-load: zero POST /api/inference/load left the page")

    after_active = active_model()
    if after_active:
        fail(f"a model is loaded after Save settings: {after_active!r}")
    else:
        info("OK unloaded: /api/inference/status still reports no active model")

    # ─────────────────────────────────────────────────────
    # 7. It survives a reload, still unloaded (HARD).
    # ─────────────────────────────────────────────────────
    step("the saved context survives a browser reload with the model still unloaded")
    close_picker()
    page.reload()
    page.locator('textarea[aria-label="Message input"]').wait_for(state = "visible", timeout = 60_000)
    popover = open_picker()
    if open_config(popover, MODEL_HINT) is None:
        fail("could not reopen run-settings after reload")
    else:
        ctx_in = context_input(popover)
        val = ctx_in.input_value() if ctx_in else None
        if _as_int(val) == DISTINCT_CTX:
            info(f"OK reload: Context Length still {val!r}")
        else:
            fail(f"Context Length did not survive the reload (got {val!r})")
        shoot("07-after-reload")
    final_active = active_model()
    if final_active:
        fail(f"a model is loaded at the end of the probe: {final_active!r}")
    else:
        info("OK unloaded: model never loaded across the whole probe")

    if page_errors:
        fail(f"page errors: {page_errors[:3]!r}")

    try:
        ctx.close()
        browser.close()
    except Exception:
        pass
    _watchdog.cancel()

if _failed:
    print(f"[pr10216] RESULT: FAIL ({len(_failed)} gate(s))", flush = True)
    for m in _failed:
        print(f"[pr10216]   - {m}", flush = True)
    sys.exit(1)
print("[pr10216] RESULT: PASS -- run settings saved for an unloaded model, and it stayed unloaded", flush = True)
