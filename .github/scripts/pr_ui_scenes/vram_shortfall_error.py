from pathlib import Path
import json
import re

from studio_regress.journeys import model_picker
from studio_test_kit.auth import seed_init_script
from studio_test_kit.ui import open_chat


async def drive(session, out_dir, label, **kwargs):
    root = Path(__file__).resolve().parents[1]
    arm = "before" if label == "BEFORE" else "after"
    probe = json.loads((root / f"outputs/pr16453_{arm}_probe.json").read_text(encoding = "utf-8"))
    message = next(
        r["message"] for r in probe["rows"] if not r["settled"] and r["telemetry_available"]
    )
    init = seed_init_script(
        type(
            "Auth", (), dict(access_token = session.access_token, refresh_token = session.refresh_token)
        )(),
        [],
    )
    requests = []

    async def refuse(route):
        requests.append(route.request.post_data_json)
        await route.fulfill(status = 500, json = {"detail": message})

    async with open_chat(
        session.base_url, init_scripts = [init], viewport = (1440, 900), headless = True
    ) as sp:
        page = sp.page
        await page.route("**/api/system", model_picker._serve_system)
        await page.route("**/api/models/gguf-variants*", model_picker._serve_canned)
        await page.route("**/api/hub/gguf-variants*", model_picker._serve_canned)
        await page.route("**/api/inference/load", refuse)
        await page.goto(session.base_url + "/chat", wait_until = "domcontentloaded")
        await page.get_by_role("button", name = "Select model", exact = True).click(timeout = 60000)
        await page.get_by_text(model_picker.LEAF, exact = True).first.click(timeout = 30000)
        buttons = page.get_by_role(
            "button", name = re.compile(r"^(Load|Download|Use)( model)?$", re.I)
        )
        await buttons.last.click(timeout = 30000)
        error = page.get_by_role("button", name = message, exact = True)
        await error.wait_for(state = "visible", timeout = 60000)
        await error.click()
        await page.get_by_role("button", name = "Copy error message", exact = True).wait_for(
            timeout = 10000
        )
        assert requests, "No intercepted load request; the intended error did not engage"
        shot = out_dir / f"{arm}_load_error.png"
        await sp.screenshot(shot, full_page = False)
    return [shot], dict(
        message = message,
        load_request_count = len(requests),
        error_source = probe["imported_from"],
        scope = "UI rendering of a captured load-path simulation error; no GPU race reproduction",
    )
