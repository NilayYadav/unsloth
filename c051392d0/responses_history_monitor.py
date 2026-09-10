"""Real Responses requests and production API-monitor UI; no mocked responses."""
import asyncio
import json
from pathlib import Path

import httpx
from playwright.async_api import async_playwright
from pr_ui_scenes._common import api_get
from studio_test_kit.auth import seed_init_script

PROMPT = "What was my project code?"


async def drive(session, out_dir: Path, label: str, **kwargs):
    history_param = kwargs.get("history_param", "previous_response_id")
    assert history_param in {"previous_response_id", "conversation"}
    history_reference = "conv_previous" if history_param == "conversation" else "resp_previous"
    headers = {"Authorization": f"Bearer {session.access_token}"}
    async with httpx.AsyncClient(base_url=session.base_url, headers=headers, timeout=90) as client:
        initial = (await client.get("/api/inference/monitor")).json()
        assert initial["active_model"] is None, initial["active_model"]
        assert initial["logging_enabled"] is True
        (await client.delete("/api/inference/monitor")).raise_for_status()
        control = await client.post("/v1/responses", json={"input": PROMPT, "stream": False})
        control_monitor = (await client.get("/api/inference/monitor")).json()
        control_rows = [e for e in control_monitor["entries"] if e["endpoint"] == "/v1/responses"]
        assert len(control_rows) == 1, control_monitor
        assert control_rows[0]["status"] == "error", control_rows
        (await client.delete("/api/inference/monitor")).raise_for_status()
        response = await client.post("/v1/responses", json={
            "input": PROMPT, "stream": False, history_param: history_reference,
        })
        monitor = (await client.get("/api/inference/monitor")).json()
        rows = [e for e in monitor["entries"] if e["endpoint"] == "/v1/responses"]

    facts = {
        "history_parameter": history_param,
        "history_reference": history_reference,
        "control_http_status": control.status_code,
        "control_response": control.json(),
        "control_monitor_rows": len(control_rows),
        "control_row_status": control_rows[0]["status"],
        "probe_http_status": response.status_code,
        "probe_response": response.json(),
        "probe_monitor_rows": len(rows),
        "probe_row_status": rows[0]["status"] if rows else None,
        "active_model": monitor["active_model"],
        "logging_enabled": monitor["logging_enabled"],
        "ui_console_errors": [],
        "ui_page_errors": [],
    }
    (out_dir / f"{label.lower()}-api.json").write_text(json.dumps(facts, indent=2))
    if label == "BEFORE":
        assert len(rows) == 1 and rows[0]["status"] == "error", facts
    else:
        assert response.status_code == 400, facts
        assert response.json()["error"]["code"] == "unsupported_parameter", facts
        assert response.json()["error"]["param"] == history_param, facts
        assert len(rows) == 0, facts

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        context = await browser.new_context(viewport={"width": 1440, "height": 1000}, color_scheme="light")
        await context.add_init_script(seed_init_script(session, []))
        page = await context.new_page()
        page.on("pageerror", lambda error: facts["ui_page_errors"].append(str(error)))
        page.on("console", lambda msg: facts["ui_console_errors"].append(msg.text) if msg.type == "error" else None)
        try:
            await page.goto(session.base_url + "/api-monitor", wait_until="domcontentloaded")
            await page.get_by_label("Search API requests").wait_for(timeout=60000)
            if rows:
                row = page.get_by_role("button").filter(has_text="/responses").first
                await row.wait_for(timeout=30000)
                await row.click()
                await page.get_by_role("heading", name="POST /v1/responses").wait_for()
                await page.get_by_text(f"user: {PROMPT}", exact=True).last.wait_for()
                await page.locator("pre").filter(has_text="Loading…").wait_for(state="hidden", timeout=15000)
            else:
                await page.get_by_text("No API traffic yet.", exact=False).wait_for(timeout=30000)
            await page.evaluate("document.fonts.ready")
            facts["ui_response_row_count"] = await page.get_by_role("button").filter(has_text="/responses").count()
            facts["ui_no_traffic_message"] = await page.get_by_text("No API traffic yet.", exact=False).count()
            facts["ui_body_text"] = await page.locator("body").inner_text()
            shot = out_dir / f"{label.lower()}-api-monitor.png"
            await page.screenshot(path=str(shot), clip={"x": 310, "y": 35, "width": 1100, "height": 915})
            assert facts["ui_response_row_count"] == len(rows)
            assert not facts["ui_page_errors"], facts["ui_page_errors"]
        except BaseException:
            await page.screenshot(path=str(out_dir / f"{label.lower()}-debug.png"))
            (out_dir / f"{label.lower()}-debug.txt").write_text(await page.locator("body").inner_text())
            raise
        finally:
            await context.close()
            await browser.close()
    return [shot], facts
