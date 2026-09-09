import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

BACKEND = Path(sys.argv[1]).resolve()
CONFIG = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(BACKEND))
os.chdir(BACKEND)

# Byte sizes from the Hub tree of meta-llama/Llama-3.1-8B-Instruct (2026-09-08).
SHARDS = {
    "model-00001-of-00004.safetensors": 4976698672,
    "model-00002-of-00004.safetensors": 4999802720,
    "model-00003-of-00004.safetensors": 4915916176,
    "model-00004-of-00004.safetensors": 1168138808,
}
ORIGINAL_PTH = 16060617592
PARAMS = 8030261248
OPTIMIZER_PT = PARAMS * 2 * 4  # two fp32 AdamW moments
SHARD_SUM = sum(SHARDS.values())
GIB = 1024 ** 3


def sparse(path: Path, size: int) -> None:
    path.parent.mkdir(parents = True, exist_ok = True)
    with open(path, "wb") as handle:
        handle.truncate(size)


def write_index(folder: Path, shards: dict) -> None:
    # The model.safetensors.index.json transformers writes beside sharded weights: every
    # tensor maps to its shard, and metadata.total_size is the shard sum.
    weight_map = {f"layer.{i}.weight": name for i, name in enumerate(sorted(shards))}
    (folder / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": sum(shards.values())}, "weight_map": weight_map})
    )


root = Path(tempfile.mkdtemp(prefix = "pr10558-"))
meta = root / "Llama-3.1-8B-Instruct-meta-download"
run = root / "Llama-3.1-8B-Instruct-full-finetune-run"
for folder in (meta, run):
    folder.mkdir(parents = True, exist_ok = True)
    shutil.copy(CONFIG, folder / "config.json")
    for name, size in SHARDS.items():
        sparse(folder / name, size)
    write_index(folder, SHARDS)
sparse(meta / "original" / "consolidated.00.pth", ORIGINAL_PTH)
sparse(run / "optimizer.pt", OPTIMIZER_PT)
sparse(run / "scheduler.pt", 1064)
sparse(run / "training_args.bin", 5752)
sparse(run / "rng_state.pth", 14244)
sparse(run / "checkpoint-100" / "model-00001-of-00001.safetensors", SHARD_SUM)

# mistralai/Mistral-7B-Instruct-v0.3 ships a whole-model consolidated.safetensors beside
# the sharded transformers weights, in the same directory (Hub tree, 2026-09-08).
MISTRAL_SHARDS = {
    "model-00001-of-00003.safetensors": 4949453792,
    "model-00002-of-00003.safetensors": 4999819336,
    "model-00003-of-00003.safetensors": 4546807800,
}
MISTRAL_CONSOLIDATED = 14496078512
mistral = root / "Mistral-7B-Instruct-v0.3-hf-download"
mistral.mkdir()
for name, size in MISTRAL_SHARDS.items():
    sparse(mistral / name, size)
write_index(mistral, MISTRAL_SHARDS)
sparse(mistral / "consolidated.safetensors", MISTRAL_CONSOLIDATED)

# A LLaVA-style folder: a safetensors language model plus a separately loaded projector.
llava = root / "llava-style-custom-folder"
sparse(llava / "model.safetensors", 13_400_000_000)
sparse(llava / "mm_projector.bin", 41_960_000)

from utils.hardware.hardware import (
    _get_local_weight_size_bytes,
    estimate_fp16_model_size_bytes,
    estimate_required_model_memory_gb,
)

EXPECTED = {
    "meta_download": SHARD_SUM,
    "full_finetune_run": SHARD_SUM,
    "mistral_consolidated_beside_shards": max(sum(MISTRAL_SHARDS.values()), MISTRAL_CONSOLIDATED),
    "llava_projector_beside_safetensors": 13_400_000_000 + 41_960_000,
}
rows = {}
for label, folder in (
    ("meta_download", meta),
    ("full_finetune_run", run),
    ("mistral_consolidated_beside_shards", mistral),
    ("llava_projector_beside_safetensors", llava),
):
    local = _get_local_weight_size_bytes(str(folder))
    fp16, source = estimate_fp16_model_size_bytes(str(folder))
    full_gb, full_meta = estimate_required_model_memory_gb(
        str(folder), training_type = "Full Finetuning", load_in_4bit = False
    )
    lora_gb, _ = estimate_required_model_memory_gb(str(folder), training_type = "LoRA", load_in_4bit = False)
    qlora_gb, _ = estimate_required_model_memory_gb(str(folder), training_type = "QLoRA", load_in_4bit = True)
    rows[label] = {
        "weight_files_bytes": local,
        "weight_files_gib": round(local / GIB, 2) if local else None,
        "fp16_estimate_gib": round(fp16 / GIB, 2) if fp16 else None,
        "fp16_source": source,
        "full_finetune_required_gb": round(full_gb, 1) if full_gb else None,
        "lora_required_gb": round(lora_gb, 1) if lora_gb else None,
        "qlora_required_gb": round(qlora_gb, 1) if qlora_gb else None,
        "estimation_mode": full_meta.get("estimation_mode"),
    }
    print(f"{label}: {json.dumps(rows[label])}")

failures = []
for label, expected in EXPECTED.items():
    got = rows[label]["weight_files_bytes"]
    verdict = "ok" if got == expected else "WRONG"
    print(f"{label}: expected {expected} bytes ({expected / GIB:.2f} GiB), got {got} ({got / GIB:.2f} GiB) -> {verdict}")
    if got != expected:
        failures.append(f"{label}: sized {got} bytes ({got / GIB:.2f} GiB), expected {expected}")
if failures:
    print("REPRO FAIL: " + "; ".join(failures))
    sys.exit(1)
print("REPRO PASS: every folder sizes to one copy of its weights")
