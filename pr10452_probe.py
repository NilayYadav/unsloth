"""PR 10452 A/B probe: does `unsloth train --config` notice a key it does not know?

Identical on both branches. Drives the real CLI (`python -m unsloth_cli train
--config <file> --dry-run`) and asserts the FIXED contract, so the negative
branch fails at the assertion and the positive branch passes.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent
SHIPPED = REPO / "studio" / "backend" / "assets" / "configs"

VALID = (
    "model: unsloth/Qwen2.5-0.5B\n"
    "data:\n"
    "  dataset: tatsu-lab/alpaca\n"
    "training:\n"
    "  num_epochs: 9\n"
    "  learning_rate: 5e-5\n"
    "lora:\n"
    "  lora_r: 8\n"
)


def run(body, suffix = ".yaml"):
    tmp = Path(tempfile.mkdtemp()) / ("config" + suffix)
    tmp.write_text(body, encoding = "utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "unsloth_cli", "train", "--config", str(tmp), "--dry-run"],
        cwd = REPO, capture_output = True, text = True,
    )
    return proc.returncode, (proc.stdout + proc.stderr)


def resolved(out, key):
    for line in out.splitlines():
        if line.strip().startswith(key + ":"):
            return line.strip()
    return f"<{key} not printed>"


CASES = []


def case(name, body, check, note, suffix = ".yaml"):
    CASES.append((name, body, check, note, suffix))


case(
    "top-level learning_rate (the CLI flag spelling)",
    "model: unsloth/Qwen2.5-0.5B\ndata:\n  dataset: tatsu-lab/alpaca\nlearning_rate: 5e-5\n",
    lambda code, out: code == 2 and "learning_rate" in out and "training:" in out,
    lambda code, out: f"exit={code} resolved -> {resolved(out, 'learning_rate')}",
)
case(
    "lora_r under training: instead of lora:",
    "model: unsloth/Qwen2.5-0.5B\ndata:\n  dataset: tatsu-lab/alpaca\ntraining:\n  lora_r: 8\n",
    lambda code, out: code == 2 and "lora_r" in out and "'lora:'" in out,
    lambda code, out: f"exit={code} resolved -> {resolved(out, 'lora_r')}",
)
case(
    "num-epochs hyphenated inside training:",
    "model: unsloth/Qwen2.5-0.5B\ndata:\n  dataset: tatsu-lab/alpaca\ntraining:\n  num-epochs: 9\n",
    lambda code, out: code == 2 and "num-epochs" in out and "num_epochs" in out,
    lambda code, out: f"exit={code} resolved -> {resolved(out, 'num_epochs')}",
)
case(
    "unparseable YAML (bad indentation)",
    "model: unsloth/Qwen2.5-0.5B\ntraining:\n  num_epochs: 3\n   learning_rate: 1\n",
    lambda code, out: code == 2 and "Could not parse config file" in out,
    lambda code, out: f"exit={code} first line -> {out.strip().splitlines()[0] if out.strip() else '<no output>'}",
)
case(
    "top level is a YAML list, not a mapping",
    "- model: unsloth/Qwen2.5-0.5B\n- model: unsloth/Qwen2.5-1.5B\n",
    lambda code, out: code == 2 and "must be a mapping" in out,
    lambda code, out: f"exit={code} first line -> {out.strip().splitlines()[0] if out.strip() else '<no output>'}",
)
case(
    "CONTROL: a valid config still resolves unchanged",
    VALID,
    lambda code, out: (
        code == 0
        and "num_epochs: 9" in out
        and "lora_r: 8" in out
        and ("learning_rate: 5.0e-05" in out or "learning_rate: 5e-05" in out)
    ),
    lambda code, out: (
        f"exit={code} {resolved(out, 'num_epochs')} | "
        f"{resolved(out, 'learning_rate')} | {resolved(out, 'lora_r')}"
    ),
)

for shipped in ("full_finetune.yaml", "lora_text.yaml", "vision_lora.yaml"):
    case(
        f"CONTROL: shipped {shipped} still loads",
        (SHIPPED / shipped).read_text(encoding = "utf-8"),
        lambda code, out: code == 0 and "model:" in out,
        lambda code, out: f"exit={code} {resolved(out, 'model')}",
    )

failures = []
print("=" * 78)
print("PR 10452 -- unsloth train --config, unknown-key handling")
print("=" * 78)
for name, body, check, note, suffix in CASES:
    code, out = run(body, suffix)
    ok = check(code, out)
    print(f"\n[{'PASS' if ok else 'FAIL'}] {name}")
    print(f"       {note(code, out)}")
    if not ok:
        failures.append(name)
        for line in out.strip().splitlines()[:6]:
            print(f"       | {line}")

print("\n" + "=" * 78)
print(f"RESULT: {len(CASES) - len(failures)} passed, {len(failures)} failed")
for name in failures:
    print(f"  FAILED: {name}")
print("=" * 78)
sys.exit(1 if failures else 0)
