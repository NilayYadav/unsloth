import json, os, sys, tempfile
from pathlib import Path

home = Path(tempfile.mkdtemp()) / "hf"
hub = home / "hub"
os.environ["HF_HOME"] = str(home)
os.environ["HF_HUB_CACHE"] = str(hub)
sys.path.insert(0, str(Path("studio/backend").resolve()))

import huggingface_hub
from huggingface_hub import hf_hub_download

REPO = "unsloth/SmolLM2-135M-Instruct-GGUF"
Q2 = "SmolLM2-135M-Instruct-Q2_K.gguf"


def disk_bytes():
    total = 0
    for root, dirs, files in os.walk(hub):
        for name in files:
            total += os.lstat(os.path.join(root, name)).st_size
    return total


def payloads():
    store = hub / "blobs"
    return sorted(p.name[:12] for p in store.glob("*/*") if p.is_file() and len(p.name) == 64) if store.is_dir() else []


for f in (Q2, "SmolLM2-135M-Instruct-Q3_K_M.gguf"):
    hf_hub_download(REPO, f, cache_dir = str(hub))
q2_size = 88201792
before, pay_before = disk_bytes(), payloads()

from hub.services.models import deletion

res = deletion._delete_cached_model_blocking(REPO, "Q2_K", None)
after, pay_after = disk_bytes(), payloads()
facts = {
    "platform": sys.platform, "huggingface_hub": huggingface_hub.__version__,
    "shared_store": bool(pay_before), "payloads_before": pay_before, "payloads_after": pay_after,
    "delete_status": res.get("status"), "bytes_before": before, "bytes_after": after,
    "bytes_freed": before - after, "q2_k_size": q2_size,
}
print(json.dumps(facts, indent = 2))
Path(os.environ.get("FACTS_OUT", "facts.json")).write_text(json.dumps(facts, indent = 2))
assert facts["shared_store"], "huggingface_hub did not use the shared blob store; the probe proves nothing"
assert res.get("status") == "deleted", res
assert facts["bytes_freed"] >= q2_size, f"REPRO: deleted Q2_K but only {facts['bytes_freed']} bytes left the disk (expected >= {q2_size})"
print(f"PASS: deleting Q2_K freed {facts['bytes_freed']} bytes")
