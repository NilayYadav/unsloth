"""Scene: the Hub GGUF card's preselected quant on a spoofed 24 GB Apple Silicon Mac (PR 15848).

This host is Linux, so `/api/system` is answered with a Mac payload (MLX backend, one shared
device, 24 GB total, `free_gb` free, 16 GB Metal working set). Both sides get the identical
payload; the route hit count proves the spoof engaged. Wiring evidence, not real Metal.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from pathlib import Path

WORKSPACE = Path(os.environ.get("WORKSPACE") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(WORKSPACE))

from pr_ui_scenes._common import Session  # noqa: E402
from pr_ui_scenes.gguf_picker_rows import QUANT_RE, ROW_CHROME_RE, _select_repo  # noqa: E402
from studio_test_kit.auth import seed_init_script  # noqa: E402
from studio_test_kit.ui import open_chat  # noqa: E402

DEFAULT_REPO = "unsloth/Qwen3-8B-GGUF"


def _mac(data: dict, total_gb: float, free_gb: float, working_set_gb: float) -> dict:
    device = {
        "index": 0,
        "index_kind": "physical",
        "name": "Apple M4 Pro",
        "memory_total_gb": total_gb,
        "vram_free_gb": free_gb,
        "vram_used_gb": 0.0,
        "shared_memory": True,
        "unified_memory": True,
        "metal_working_set_gb": working_set_gb,
    }
    gpu = {"available": True, "backend": "mlx", "devices": [device]}
    data = dict(data)
    data.update(device_backend = "mlx", gpu = gpu, inference_gpu = gpu)
    data["memory"] = {**(data.get("memory") or {}), "total_gb": total_gb, "available_gb": free_gb}
    return data


async def drive(
    session: Session,
    out_dir: Path,
    label: str,
    repo: str = DEFAULT_REPO,
    total_gb: float = 24.0,
    free_gb: float = 10.0,
    working_set_gb: float = 16.0,
    **_: object,
) -> tuple[list[Path], dict]:
    out_dir.mkdir(parents = True, exist_ok = True)
    facts: dict = {
        "spoof": {"total_gb": total_gb, "free_gb": free_gb, "working_set_gb": working_set_gb}
    }
    hits = 0

    async def spoof(route):
        nonlocal hits
        resp = await route.fetch()
        body = _mac(await resp.json(), total_gb, free_gb, working_set_gb)
        hits += 1
        await route.fulfill(
            response = resp,
            body = json.dumps(body),
            headers = {**resp.headers, "content-type": "application/json"},
        )

    init = seed_init_script(
        type(
            "A", (), {"access_token": session.access_token, "refresh_token": session.refresh_token}
        )(),
        [],
    )
    shots: list[Path] = []
    async with open_chat(
        session.base_url, init_scripts = [init], viewport = (1500, 1000), headless = True
    ) as sp:
        page = sp.page
        await page.context.route(re.compile(r".*/api/system(\?.*)?$"), spoof)
        await page.goto(f"{session.base_url}/hub", wait_until = "domcontentloaded")
        await _select_repo(page, repo)
        selector = (
            page.locator("button")
            .filter(has_text = re.compile(QUANT_RE))
            .filter(has_not_text = re.compile(ROW_CHROME_RE))
        )
        await selector.first.wait_for(state = "visible", timeout = 60_000)
        await page.wait_for_timeout(8_000)
        facts["preselected"] = (await selector.first.inner_text()).replace("\n", " ")
        card = out_dir / f"{label.lower()}_hub_card.png"
        await sp.screenshot(card, full_page = False)
        shots.append(card)
        await selector.first.click(timeout = 30_000)
        await page.wait_for_timeout(3_000)
        menu = out_dir / f"{label.lower()}_hub_menu.png"
        await sp.screenshot(menu, full_page = False)
        shots.append(menu)
    facts["system_route_hits"] = hits
    if hits == 0:
        raise RuntimeError("/api/system was never intercepted: the Mac spoof did not engage")
    return shots, facts
