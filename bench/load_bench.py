"""Time how long a long thread takes to load (all messages mounted, windowing settled), per engine.

In-page recorder from navigation start (performance.now base): every 250 ms a sample of mounted messages and
parked messages; every rAF the frame gap. Output per rep:
  t_textarea, t_first_msg, t_all_msgs (all expected messages mounted), t_settled (last change of the
  mounted / parked counts), long_frames (>50 ms) count and sum and max until t_all_msgs, and the timeline.

  python load_bench.py --base-url URL --home HOME --threads threads.json --turns 150 --arm main-now \
      --engine webkit --material solid --reps 3 --out load.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--base-url", required=True)
ap.add_argument("--home", required=True)
ap.add_argument("--threads", required=True)
ap.add_argument("--turns", default="150")
ap.add_argument("--arm", required=True)
ap.add_argument("--engine", default="webkit")
ap.add_argument("--material", default="solid", choices=["solid", "glass"])
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--timeout-s", type=float, default=180)
ap.add_argument("--pack", default=os.environ.get("PACK", ""))
ap.add_argument("--out", required=True)
a = ap.parse_args()
sys.path.insert(0, a.pack)
import menu_bench as mb  # noqa: E402

REC = r"""
(() => {
  const S = window.__load = {samples: [], gaps: [], textarea: null};
  let last = null;
  const tick = (t) => { if (last !== null) S.gaps.push([Math.round(t), Math.round(t - last)]); last = t;
                        if (!S.stop) requestAnimationFrame(tick); };
  requestAnimationFrame(tick);
  const sample = () => {
    if (S.stop) return;
    const msgs = document.querySelectorAll('[data-role][data-message-id]').length;
    const parked = document.querySelectorAll('[data-message-parked]').length;
    if (S.textarea === null && document.querySelector('form textarea')) S.textarea = Math.round(performance.now());
    const prev = S.samples[S.samples.length - 1];
    if (!prev || prev[1] !== msgs || prev[2] !== parked) S.samples.push([Math.round(performance.now()), msgs, parked]);
    setTimeout(sample, 250);
  };
  setTimeout(sample, 0);
})();
"""


async def one(p, init, info, rep) -> dict:
    eng = a.engine
    browser = await (p.chromium.launch(channel="msedge") if eng == "msedge" else getattr(p, eng).launch())
    try:
        ctx = await browser.new_context(viewport={"width": 1440, "height": 900})
        await ctx.add_init_script(init)
        await ctx.add_init_script(REC)
        page = await ctx.new_page()
        expected = info["messages"]
        await page.goto(f"{a.base_url}/chat?thread={info['thread']}", wait_until="domcontentloaded")
        t0 = asyncio.get_event_loop().time()
        state = {}
        while asyncio.get_event_loop().time() - t0 < a.timeout_s:
            await page.wait_for_timeout(500)
            state = await page.evaluate("() => { const s = window.__load.samples; return {now: performance.now(), last: s[s.length - 1] || null}; }")
            last = state["last"]
            if last and last[1] >= expected and state["now"] - last[0] > 3000:
                break
        res = await page.evaluate("() => { window.__load.stop = true; return window.__load; }")
        shape = await page.evaluate("() => ({nodes: document.getElementsByTagName('*').length, webkitgtk: /linux/i.test(navigator.platform + ' ' + navigator.userAgent) && /AppleWebKit/.test(navigator.userAgent) && !/android|chrom|edg/i.test(navigator.userAgent)})")
    finally:
        await browser.close()
    samples = res["samples"]
    first = next((t for t, m, _ in samples if m > 0), None)
    allm = next((t for t, m, _ in samples if m >= expected), None)
    settled = samples[-1][0] if samples else None
    upto = allm or settled or 0
    gaps = [g for t, g in res["gaps"] if t <= upto]
    longg = [g for g in gaps if g > 50]
    out = {"rep": rep, "t_textarea": res["textarea"], "t_first_msg": first, "t_all_msgs": allm, "t_settled": settled,
           "final_msgs": samples[-1][1] if samples else 0, "final_parked": samples[-1][2] if samples else 0,
           "long_frames": len(longg), "long_frames_ms": sum(longg), "max_gap_ms": max(gaps) if gaps else None,
           "frames": len(gaps), "timeline": samples[:400], **shape}
    print("LOAD", json.dumps({"arm": a.arm, "engine": eng, "material": a.material,
                              **{k: v for k, v in out.items() if k != "timeline"}}), flush=True)
    return out


async def main() -> int:
    from playwright.async_api import async_playwright

    s = mb.studio_session(a.base_url, Path(a.home), mb.PASSWORD)
    auth = type("A", (), {"access_token": s.access_token, "refresh_token": s.refresh_token})()
    extra = {}
    if a.material == "glass":
        extra["unsloth_window_material"] = json.dumps(
            {"state": {"material": "liquid-3d", "style": "liquid-3d", "menus": True, "lens": "off"}, "version": 0})
    init = mb.seed_init_script(auth, [], extra_local_storage=extra) if extra else mb.seed_init_script(auth, [])
    info = json.loads(Path(a.threads).read_text())[a.turns]
    reps = []
    async with async_playwright() as p:
        for r in range(a.reps):
            try:
                reps.append(await one(p, init, info, r))
            except Exception as exc:  # noqa: BLE001
                reps.append({"rep": r, "error": repr(exc)[:300]})
                print("LOAD_ERROR", a.arm, a.material, r, repr(exc)[:300], flush=True)
    Path(a.out).write_text(json.dumps({"arm": a.arm, "engine": a.engine, "material": a.material,
                                       "turns": a.turns, "reps": reps}))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
