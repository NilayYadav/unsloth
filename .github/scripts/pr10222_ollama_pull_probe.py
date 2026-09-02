"""PR 10222 repro probe: does Studio list a model that `ollama pull` put on disk?

Runs against a REAL ollama store. Prints the manifest ollama actually wrote,
then the rows Studio's own scan_ollama_dir returns for it.
Exit 0 = model is listed (fixed). Exit 1 = model is hidden (defect reproduced).
"""

import json
import os
from pathlib import Path

from hub.services.models.ollama import (
    is_ollama_manifest_ref,
    materialize_ollama_model_ref,
    scan_ollama_dir,
)

MODELS_DIR = Path(os.environ["OLLAMA_MODELS"]).expanduser()
EXPECTED_TAG = os.environ.get("PROBE_MODEL", "qwen2.5:0.5b")


def banner(text):
    print(f"\n{'=' * 70}\n{text}\n{'=' * 70}", flush = True)


banner("1. What `ollama pull` actually wrote to disk")
print(f"OLLAMA_MODELS = {MODELS_DIR}")
manifests = sorted(p for p in (MODELS_DIR / "manifests").rglob("*") if p.is_file())
print(f"manifest files found: {len(manifests)}")
for m in manifests:
    print(f"  {m.relative_to(MODELS_DIR)}")

if not manifests:
    print("PROBE-ERROR: no manifest on disk, the pull did not happen")
    raise SystemExit(2)

layer_types = []
for m in manifests:
    data = json.loads(m.read_text(encoding = "utf-8-sig"))
    print(f"\nlayers in {m.name}:")
    for layer in data.get("layers", []):
        media_type = layer.get("mediaType")
        layer_types.append(media_type)
        print(f"  {media_type:<50} {layer.get('size', 0):>13,} bytes")

banner("2. Studio's inventory scan over that exact directory")
rows = scan_ollama_dir(MODELS_DIR)
print(f"scan_ollama_dir() returned {len(rows)} row(s)")
for r in rows:
    print(f"  display_name = {r.display_name!r}")
    print(f"  source       = {r.source}   format = {r.model_format}   runtime = {r.runtime}")
    print(f"  size_bytes   = {r.size_bytes:,}")
    print(f"  load_id      = {(r.load_id or r.id)[:100]}")

banner("3. Verdict")
if not rows:
    print(f"FAIL: {EXPECTED_TAG} is on disk but Studio lists 0 Ollama models.")
    print(f"      Withheld over these layer types: {sorted(set(layer_types))}")
    print("REPRO-RESULT=DEFECT-REPRODUCED")
    raise SystemExit(1)

row = rows[0]
ref = row.load_id or row.id
if not is_ollama_manifest_ref(ref):
    print(f"FAIL: row load_id is not an ollama-manifest ref: {ref[:80]}")
    print("REPRO-RESULT=UNEXPECTED")
    raise SystemExit(1)

resolved = materialize_ollama_model_ref(ref)
print(f"listed  : {row.display_name}")
print(f"resolved: {resolved}")
if not resolved.endswith(".gguf") or not Path(resolved).exists():
    print("FAIL: listed row did not resolve to a real .gguf the loader can open")
    print("REPRO-RESULT=UNEXPECTED")
    raise SystemExit(1)

print(f"gguf size on disk: {Path(resolved).stat().st_size:,} bytes")
print(f"PASS: {EXPECTED_TAG} is listed and resolves to a loadable GGUF.")
print("REPRO-RESULT=FIXED")
