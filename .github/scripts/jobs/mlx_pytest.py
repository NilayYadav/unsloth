"""Run pytest files on Apple Silicon with the MLX backend live (UNSLOTH_FORCE_GPU_PATH unset).

    python jobs/mlx_pytest.py --files tests/a.py,tests/b.py [--base-sha SHA] [--repo unslothai/unsloth]
        [--expect-fail-on-base TEST_NAME] --out r.json

Staging pytest legs force the GPU path, so every `DEVICE_TYPE == "mlx"` test skips there.
`--base-sha` also runs each file as it was at SHA (fetched next to the original, so conftests
and relative imports match). Exit 0 only when head has no failure and no MLX skip.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import urllib.request
import xml.etree.ElementTree as ET


def _run(paths, xml):
    env = dict(os.environ)
    env.pop("UNSLOTH_FORCE_GPU_PATH", None)
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-rs",
        "--tb=short",
        "-p",
        "no:cacheprovider",
        f"--junitxml={xml}",
        *paths,
    ]
    rc = subprocess.run(cmd, env = env).returncode
    out = {}
    if os.path.exists(xml):
        for case in ET.parse(xml).getroot().iter("testcase"):
            kind = "passed"
            for child in case:
                if child.tag in ("failure", "error"):
                    kind = "failed"
                elif child.tag == "skipped":
                    kind = "skipped:" + (child.get("message") or "")[:200]
            out[case.get("name")] = kind
    return rc, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", required = True)
    ap.add_argument("--base-sha")
    ap.add_argument("--repo", default = "unslothai/unsloth")
    ap.add_argument("--expect-fail-on-base", default = "")
    ap.add_argument("--out", default = "mlx_pytest.json")
    a = ap.parse_args()
    files = [f for f in a.files.split(",") if f]
    rec = {
        "job": "mlx_pytest",
        "device": f"{platform.system()}-{platform.machine()}",
        "files": files,
        "base_sha": a.base_sha,
        "checks": {},
    }

    rc, head = _run(files, "head.xml")
    mlx_skips = [
        n
        for n, k in head.items()
        if k.startswith("skipped") and "only active on the MLX backend" in k
    ]
    failed = [n for n, k in head.items() if k == "failed"]
    rec["head"] = {"rc": rc, "tests": head}
    rec["checks"]["head_ran"] = {
        "ok": bool(head) and rc in (0, 1),
        "detail": f"rc={rc} n={len(head)}",
    }
    rec["checks"]["head_no_failures"] = {"ok": not failed, "detail": failed}
    rec["checks"]["head_no_mlx_skips"] = {"ok": not mlx_skips, "detail": mlx_skips}

    if a.base_sha:
        base_paths = []
        for f in files:
            url = f"https://raw.githubusercontent.com/{a.repo}/{a.base_sha}/{f}"
            dst = os.path.join(
                os.path.dirname(f), "test_base__" + os.path.basename(f).removeprefix("test_")
            )
            with urllib.request.urlopen(url, timeout = 60) as r, open(dst, "wb") as fh:
                fh.write(r.read())
            base_paths.append(dst)
        brc, base = _run(base_paths, "base.xml")
        rec["base"] = {"rc": brc, "tests": base}
        for name in filter(None, a.expect_fail_on_base.split(",")):
            rec["checks"][f"base_fails:{name}"] = {
                "ok": base.get(name) == "failed" and head.get(name) == "passed",
                "detail": f"base={base.get(name)} head={head.get(name)}",
            }

    rec["passed"] = all(c["ok"] for c in rec["checks"].values())
    with open(a.out, "w") as fh:
        json.dump(rec, fh, indent = 1)
    print("JOB_RESULT " + json.dumps({"passed": rec["passed"], "checks": rec["checks"]}))
    sys.exit(0 if rec["passed"] else 1)


if __name__ == "__main__":
    main()
