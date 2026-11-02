import json
import os
from pathlib import Path
import subprocess
import uuid

from pr_ui_scenes._common import api_get, api_post
from pr_ui_scenes.chat_export_filename import _put
from studio_test_kit.auth import seed_init_script
from studio_test_kit.ui import open_chat


async def drive(session, out_dir, label, **kwargs):
    workspace = Path(os.environ["WORKSPACE"])
    repo = workspace / "temp" / f"uidiff16121_{label.lower()}"
    code = """
import json, sys
sys.path.insert(0, sys.argv[1])
from routes.inference import _sf_reasoning_prefill_mode, _ResponsesReasoningExtractor
template = open(sys.argv[2], encoding="utf-8").read()
prefilled = _sf_reasoning_prefill_mode({"reasoning_style":"enable_thinking", "supports_reasoning":True}, None, template)
extractor = _ResponsesReasoningExtractor(parse_think_markers=True, reasoning_prefilled=prefilled)
r, c = extractor.feed("Four")
tr, tc = extractor.finish()
print("RESULT=" + json.dumps({"reasoning":r+tr, "content":c+tc, "prefilled":prefilled}))
"""
    result = subprocess.run(
        [
            str(session.home / "unsloth_studio/bin/python"),
            "-c",
            code,
            str(repo / "studio/backend"),
            str(workspace / "temp/pr16121/templates/lfm3_omni/chat_template.jinja"),
        ],
        capture_output = True,
        text = True,
        check = True,
        env = dict(os.environ, CUDA_VISIBLE_DEVICES = "", UNSLOTH_STUDIO_HOME = str(session.home)),
    )
    facts = json.loads(
        next(line[7:] for line in result.stdout.splitlines() if line.startswith("RESULT="))
    )
    expected = (
        {"reasoning": "Four", "content": "", "prefilled": True}
        if label == "BEFORE"
        else {"reasoning": "", "content": "Four", "prefilled": False}
    )
    assert facts == expected, facts
    thread = "uidiff-reasoning-" + uuid.uuid4().hex[:8]
    stamp = 1767225600000
    api_post(
        session,
        "/api/chat/threads",
        {
            "id": thread,
            "title": "Generation tag response",
            "modelType": "base",
            "modelId": "LiquidAI/LFM3-Omni-2B",
            "createdAt": stamp,
            "updatedAt": stamp,
        },
    )
    parts = [
        {"type": kind, "text": facts[key]}
        for kind, key in [("reasoning", "reasoning"), ("text", "content")]
        if facts[key]
    ]
    for index, (role, content) in enumerate(
        [("user", [{"type": "text", "text": "What is 2+2?"}]), ("assistant", parts)]
    ):
        mid = f"{thread}-{index}"
        _put(
            session,
            f"/api/chat/threads/{thread}/messages/{mid}",
            {
                "id": mid,
                "threadId": thread,
                "parentId": None if index == 0 else f"{thread}-0",
                "role": role,
                "content": content,
                "createdAt": stamp + index + 1,
            },
        )
    init = seed_init_script(
        type(
            "Auth",
            (),
            {"access_token": session.access_token, "refresh_token": session.refresh_token},
        )(),
        [],
    )
    async with open_chat(
        session.base_url, init_scripts = [init], viewport = (1200, 800), headless = True
    ) as sp:
        await sp.page.goto(
            f"{session.base_url}/chat?thread={thread}", wait_until = "domcontentloaded"
        )
        reply = sp.page.locator('[data-role="assistant"]').last
        await reply.wait_for(state = "visible", timeout = 60000)
        reasoning = reply.locator('[data-slot="reasoning-root"]')
        if label == "BEFORE":
            await reasoning.wait_for(state = "visible", timeout = 15000)
            trigger = reasoning.locator("button").first
            if not await reasoning.locator('[data-slot="reasoning-content"]').is_visible():
                await trigger.click()
        await reply.get_by_text("Four", exact = True).wait_for(state = "visible", timeout = 15000)
        facts["reasoning_drawers"] = await reasoning.count()
        assert facts["reasoning_drawers"] == (1 if label == "BEFORE" else 0)
        facts["_evidence_scope"] = (
            "Each build classifies fixed generated text; result replayed as saved transcript. No model generation."
        )
        shot = out_dir / f"{label.lower()}_reasoning.png"
        await sp.page.screenshot(path = str(shot), full_page = False)
    return [shot], facts
