"""GPU-layer keyword diagnostics with a fixed picker listing; no model is loaded."""

from pathlib import Path

from pr_ui_scenes._common import Session
from studio_regress.journeys.model_picker import LEAF, REPO, canned_listing
from studio_test_kit.auth import StudioAuth, seed_init_script
from studio_test_kit.ui import open_chat


async def drive(session: Session, out_dir: Path, label: str, **kwargs):
    async def hub(route):
        await route.fulfill(
            json = [
                {
                    "id": REPO,
                    "author": "unsloth",
                    "downloads": 1000,
                    "likes": 10,
                    "tags": ["gguf", "text-generation", "qwen3_5"],
                    "pipeline_tag": "text-generation",
                    "library_name": "gguf",
                    "lastModified": "2026-11-01T00:00:00Z",
                    "createdAt": "2026-10-01T00:00:00Z",
                    "gguf": {"total": 800000000},
                    "config": {"model_type": "qwen3_5"},
                }
            ],
            headers = {"access-control-allow-origin": "*"},
        )

    async def system(route):
        await route.fulfill(
            json = {
                "status": "ready",
                "platform": "Linux",
                "python_version": "3.13",
                "device_backend": "cpu",
                "uptime_seconds": 0,
                "cpu": {
                    "logical_count": 8,
                    "physical_count": 8,
                    "usage_percent": 0,
                    "frequency_mhz": None,
                },
                "memory": {
                    "total_gb": 64,
                    "available_gb": 56,
                    "percent_used": 12.5,
                    "process_used_mb": 0,
                },
                "disk": {"total_gb": 1000, "free_gb": 900, "percent_used": 10},
                "gpu": {"available": False, "devices": []},
                "ml_packages": {},
            }
        )

    async def variants(route):
        listing = canned_listing()
        for item in listing["variants"]:
            item["downloaded"] = True
        await route.fulfill(json = listing)

    auth = StudioAuth(session.access_token, session.refresh_token, session.base_url)
    shots, facts = [], {}
    async with open_chat(
        session.base_url,
        init_scripts = [seed_init_script(auth, [])],
        viewport = (1440, 1100),
        headless = True,
    ) as sp:
        page = sp.page
        await page.route("https://huggingface.co/api/models*", hub)
        await page.route("**/api/system", system)
        await page.route("**/api/models/gguf-variants*", variants)
        await page.route("**/api/hub/gguf-variants*", variants)
        await page.goto(session.base_url + "/chat", wait_until = "domcontentloaded")
        await page.get_by_role("button", name = "Select model", exact = True).click()
        await page.screenshot(path = str(out_dir / "picker_debug.png"))
        (out_dir / "picker_debug.txt").write_text(
            await page.locator("body").inner_text(), encoding = "utf-8"
        )
        await page.get_by_text(LEAF, exact = True).first.click()
        await page.get_by_role(
            "button", name = f"Inference settings for {REPO} UD-Q4_K_XL", exact = True
        ).click()
        editor = page.get_by_role("textbox", name = "Extra llama-server arguments")
        if not await editor.is_visible():
            await page.get_by_role("switch", name = "Show advanced settings", exact = True).click()
        for value in ("auto", "all"):
            await editor.fill(f"-ngl {value}")
            await page.wait_for_function(
                "v => document.querySelector('textarea[aria-label=\"Extra llama-server arguments\"]')?.value === '-ngl ' + v",
                arg = value,
            )
            await editor.evaluate("e => e.scrollIntoView({block: 'center'})")
            advice = page.locator('[id="' + await editor.get_attribute("aria-describedby") + '"]')
            text = await advice.inner_text()
            rejected = "takes a number" in text
            if rejected != (label == "BEFORE"):
                raise AssertionError(f"{label} {value}: unexpected diagnostics {text!r}")
            facts[value] = {
                "input": await editor.input_value(),
                "rejected": rejected,
                "advice": text,
            }
            shot = out_dir / f"{label.lower()}_{value}.png"
            box = await editor.bounding_box()
            x, y = max(0, box["x"] - 16), max(0, box["y"] - 32)
            await page.screenshot(
                path = str(shot),
                clip = {
                    "x": x,
                    "y": y,
                    "width": min(1440 - x, box["width"] + 32),
                    "height": min(1100 - y, 260),
                },
            )
            shots.append(shot)
    return shots, facts
