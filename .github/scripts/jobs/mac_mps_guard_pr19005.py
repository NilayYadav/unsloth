"""Apple Silicon evidence for unsloth PR 19005 (#18979 follow-up, MPS large-attention guard).

Base = merge base 02bed8883e, head = PR head 8e6a2f9b25, both fetched from unslothai/unsloth into $RUNNER_TEMP, same
runner. Under torch 2.12.x (no MPS prefill kernel; the runner's default torch 2.15 reports no usable Metal device), head arm first:
  1. UNSLOTH_MPS_LARGE_ATTENTION=1 pytest of the three MPS attention test files, per arm;
  2. a wider-V-batch call (Q/K batch 1, V batch 3, 32 heads x 2896 tokens x 128) through the guard vs an fp64 CPU
     oracle, per arm (base counts only Q x K and leaves 8.05e8 scores unsplit; head splits).
Writes --out JSON; exit 0 (verdict is read from the JSON / log, not the tick)."""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

BASE_SHA = "02bed8883ed4520b733d68854a2d9381a0b3be25"
HEAD_SHA = "8e6a2f9b259c3e1f9cdaded5b35ee4a391499ad7"
REPO = "https://github.com/unslothai/unsloth"
FILES = [
    "tests/test_mps_attention_families.py",
    "tests/test_mps_attention_guard_on_load.py",
    "tests/test_mps_large_attention_guard.py",
]

ORACLE = r"""
import json, math, sys, time
import torch
import torch.nn.functional as F
from core import _mps_sdpa_guard as G
res = {"torch": torch.__version__, "guard_file": G.__file__, "mps": torch.backends.mps.is_available()}
H, T, D = 32, 2896, 128
g = torch.Generator().manual_seed(7)
q = torch.randn(1, H, T, D, generator = g).to(torch.bfloat16)
k = torch.randn(1, H, T, D, generator = g).to(torch.bfloat16)
v = torch.randn(3, H, T, D, generator = g).to(torch.bfloat16)
res["scores_qk_only"] = H * T * T
res["scores_with_v_batch"] = 3 * H * T * T
res["cap"] = G.MPS_SCORE_ELEMENTS
try:
    res["chunk_rows_qkv"] = G.chunk_rows(q, k, v)
except TypeError:
    res["chunk_rows_qkv"] = "chunk_rows takes no value (base)"
res["chunk_rows_qk"] = G.chunk_rows(q, k)
try:
    from core.inference import diffusion_attention as A
    res["prefill_serves"] = bool(A.mps_prefill_attention(q.to("mps"), k.to("mps"), v.to("mps")))
except Exception as e:
    res["prefill_serves"] = f"error {type(e).__name__}: {e}"[:200]
guarded = G.guard_sdpa(F.scaled_dot_product_attention)
G.stats["chunked_calls"] = 0
t0 = time.time()
try:
    with torch.inference_mode():
        out = guarded(q.to("mps"), k.to("mps"), v.to("mps"))
        torch.mps.synchronize()
    out = out.cpu()
    res["seconds"] = round(time.time() - t0, 2)
    res["chunked_calls"] = G.stats["chunked_calls"]
    res["out_shape"] = list(out.shape)
    torch.save(out, sys.argv[1] + ".out.pt")
    heads = sorted({0, H // 2, H - 2, H - 1})
    rows = torch.linspace(0, T - 1, 192).long()
    qd = q[:, heads][:, :, rows].double()
    sc = torch.softmax(qd @ k[:, heads].double().transpose(-1, -2) / math.sqrt(D), dim = -1)[0]
    # which V batch did each output batch use: cosine of output batch ob vs the fp64 oracle with V batch b
    cos = {}
    for ob in range(out.shape[0]):
        for b in range(3):
            ref = sc @ v[b, heads].double()
            cos[f"out{ob}_vs_v{b}"] = round(min(
                float(F.cosine_similarity(out[ob, h, rows].double().flatten(), ref[i].flatten(), dim = 0))
                for i, h in enumerate(heads)), 6)
    res["worst_cos"] = cos
    res["finite"] = bool(torch.isfinite(out.float()).all())
    small = [x[..., :64, :] for x in (q, k, v)]
    res["cpu_stock_shape_same_inputs_small"] = list(F.scaled_dot_product_attention(*[x.float() for x in small]).shape)
    res["mps_stock_shape_same_inputs_small"] = list(F.scaled_dot_product_attention(*[x.float().to("mps") for x in small]).shape)
except Exception as e:
    res["error"] = f"{type(e).__name__}: {e}"[:600]
print("ORACLE " + json.dumps(res), flush = True)
json.dump(res, open(sys.argv[1], "w"))
"""


def sh(
    cmd,
    cwd = None,
    env = None,
    timeout = 3600,
):
    print("+", cmd, flush = True)
    t0 = time.time()
    try:
        p = subprocess.run(
            cmd, shell = True, cwd = cwd, env = env, capture_output = True, text = True, timeout = timeout
        )
        rc, out = p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired as e:
        rc, out = 124, str(e.stdout or "")
    print(out[-6000:], flush = True)
    print(f"rc={rc} {time.time() - t0:.0f}s", flush = True)
    return rc, out


def fetch(sha, dst):
    dst.mkdir(parents = True, exist_ok = True)
    rc, _ = sh(
        f"git init -q && git remote add o {REPO} && git fetch -q --depth 1 o {sha} && git checkout -q FETCH_HEAD "
        f"&& git rev-parse HEAD",
        cwd = dst,
        timeout = 900,
    )
    return rc == 0


def pytest_arm(backend, tag):
    env = dict(os.environ, UNSLOTH_MPS_LARGE_ATTENTION = "1")
    rc, out = sh(
        f"{sys.executable} -m pytest -q -rfEs -o addopts= -p no:cacheprovider " + " ".join(FILES),
        cwd = backend,
        env = env,
        timeout = 3000,
    )
    tail = [
        l for l in out.strip().splitlines() if re.search(r"\d+ (passed|failed|skipped|error)", l)
    ]
    return {
        "rc": rc,
        "summary": tail[-1] if tail else out[-400:],
        "failed": sorted(set(re.findall(r"^(?:FAILED|ERROR) (\S+)", out, re.M))),
        "skipped": re.findall(r"^SKIPPED .*$", out, re.M)[:40],
        "mps_lines": re.findall(r"^MPS-(?:ATTN|GUARD) .*$", out, re.M)[:60],
    }


def oracle_arm(backend, out_json):
    env = dict(os.environ, PYTHONPATH = str(backend))
    script = Path(out_json).with_suffix(".py")
    script.write_text(ORACLE, encoding = "utf-8")
    rc, out = sh(f"{sys.executable} {script} {out_json}", cwd = backend, env = env, timeout = 1800)
    data = (
        json.loads(Path(out_json).read_text())
        if Path(out_json).exists()
        else {"error": out[-1500:]}
    )
    data["rc"] = rc
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required = True)
    ap.add_argument("--tiny", action = "store_true")
    a = ap.parse_args()
    tmp = Path(os.environ.get("RUNNER_TEMP") or "/tmp") / "pr19005"
    res = {"platform": sys.platform, "machine": os.uname().machine}
    sh(f"{sys.executable} -m pip install -q pytest pytest-timeout structlog numpy rich anyio")
    arms = {}
    for arm, sha in (("base", BASE_SHA), ("head", HEAD_SHA)):
        ok = fetch(sha, tmp / arm)
        arms[arm] = tmp / arm / "studio" / "backend"
        res[f"{arm}_fetched"] = ok
    out_dir = Path(a.out).resolve().parent
    sh(f"{sys.executable} -m pip install -q 'torch==2.12.*'")
    rc, tv = sh(
        f"{sys.executable} -c 'import torch; print(torch.__version__, torch.backends.mps.is_available())'"
    )
    res["torch"] = tv.strip().splitlines()[-1] if rc == 0 else tv[-300:]
    single = "tests/test_mps_attention_families.py::test_past_the_upstream_threshold_on_the_prefill_kernel"
    for arm in ("head", "base"):
        backend = arms[arm]
        cell = res.setdefault(arm, {})
        if not res.get(f"{arm}_fetched"):
            cell["error"] = "fetch failed"
            continue
        cell["oracle"] = oracle_arm(backend, out_dir / f"oracle_{arm}.json")
        cell["pytest"] = pytest_arm(backend, arm)
        env = dict(os.environ, UNSLOTH_MPS_LARGE_ATTENTION = "1")
        rc, out = sh(
            f"{sys.executable} -m pytest -q -rfEs -o addopts= -p no:cacheprovider {single}",
            cwd = backend,
            env = env,
        )
        cell["upstream_threshold_alone"] = [
            l for l in out.splitlines() if re.search(r"\d+ (passed|failed|skipped)", l)
        ][-1:]
        Path(a.out).write_text(json.dumps(res, indent = 1, default = str))
    rc, out = sh(
        f"{sys.executable} -c \"import torch,sys; a=torch.load(sys.argv[1]); b=torch.load(sys.argv[2]); "
        f"print('HEADvsBASE', list(a.shape), list(b.shape), torch.equal(a,b), float((a.float()-b.float()).abs().max()))\" "
        f"{out_dir / 'oracle_head.json.out.pt'} {out_dir / 'oracle_base.json.out.pt'}"
    )
    res["head_vs_base_output"] = [l for l in out.splitlines() if l.startswith("HEADvsBASE")]
    Path(a.out).write_text(json.dumps(res, indent = 1, default = str))
    print("RESULT " + json.dumps(res, default = str)[:20000], flush = True)


if __name__ == "__main__":
    main()
