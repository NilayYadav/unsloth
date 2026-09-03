# SPDX-License-Identifier: AGPL-3.0-only
# PR 10256 A/B probe. Identical on both arms; nothing here names the fix.
#
# Invariant under test:
#   every size the image gallery can record must, after "Restore settings", produce a recipe
#   that PUT /api/settings/generation-presets/image accepts.
#
# The restored size is taken from THIS tree by executing its own restoreSettings source
# (restore_size_probe.mjs). The verdict comes from the real settings router.

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "studio" / "backend"))

from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
from routes import settings as settings_routes
from storage import studio_db

# Sizes the gallery can record, and the path that records each one.
RECORDED = [
    (4032, 3024, "Edit a 4032x3024 phone photo: the edit branch snaps to /16 and does not clamp"),
    (1024, 208, "Transform a 1920x400 source with the sliders at 1024: _fit_within then snap"),
    (4096, 4096, "decode_b64_image's 4096 ceiling, the largest record reachable"),
    (2048, 256, "already inside the schema"),
    (1024, 1024, "a plain Create"),
]


def restored_size(width, height):
    out = subprocess.run(
        [
            "node", "--experimental-strip-types", "--no-warnings",
            str(Path(__file__).with_name("restore_size_probe.mjs")),
            str(ROOT), str(width), str(height),
        ],
        capture_output = True, text = True, check = True,
    )
    return json.loads(out.stdout)


def client(stored):
    studio_db.get_app_setting = lambda key, default = None: stored.get(key, default)
    studio_db.upsert_app_settings = stored.update
    app = FastAPI()
    app.include_router(settings_routes.router, prefix = "/api/settings")
    app.dependency_overrides[get_current_subject] = lambda: "pr10256-probe"
    return TestClient(app)


def main():
    print(f"tree: {ROOT}")
    print(f"node: {subprocess.run(['node','--version'],capture_output=True,text=True).stdout.strip()}")
    print()
    failures = []
    for width, height, why in RECORDED:
        restored = restored_size(width, height)
        rw, rh = restored["width"], restored["height"]
        stored = {}
        api = client(stored)
        # A recipe the user is actively naming, so a refusal costs the preset choice too.
        state = {
            "activePreset": "Landscape",
            "currentParams": {
                "negativePrompt": "blurry",
                "width": rw, "height": rh,
                "steps": 9, "guidance": 0.0, "batchSize": 1, "runs": 1,
            },
        }
        put = api.put("/api/settings/generation-presets/image", json = state)
        read = api.get("/api/settings/generation-presets/image").json()
        ok = put.status_code == 200
        ratio_in = width / height
        ratio_out = rw / rh
        print(f"recorded {width}x{height}  ({why})")
        print(f"  restore settings puts  {rw}x{rh}   (aspect {ratio_in:.4f} -> {ratio_out:.4f})")
        print(f"  PUT /api/settings/generation-presets/image -> {put.status_code}")
        if not ok:
            detail = put.json().get("detail", [])
            for d in detail[:2]:
                print(f"    {'.'.join(str(x) for x in d.get('loc', []))}: {d.get('msg')}")
        print(f"  GET back: saved={read['saved']}  activePreset={read['activePreset']!r}  "
              f"currentParams.width={read['currentParams']['width']}")
        print(f"  => {'PASS' if ok else 'FAIL'}")
        print()
        if not ok:
            failures.append(f"{width}x{height} restored to {rw}x{rh}, rejected with {put.status_code}")

    print("=" * 72)
    if failures:
        print(f"FAIL: {len(failures)} of {len(RECORDED)} recorded sizes cannot be saved after Restore settings")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"PASS: all {len(RECORDED)} recorded sizes restore to a recipe the preset endpoint accepts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
