"""Drive the real Studio GGUF row lister against a frozen copy of live Hub listings.

Prints one row per selectable variant, exactly as the picker would show it, and asserts
that every published build is reachable under its own row.
"""
import json, os, sys, types

BACKEND = os.path.join(os.environ["REPO"], "studio", "backend")
sys.path.insert(0, BACKEND)

# `loggers` is the only backend-wide dep the gguf module pulls that a bare runner lacks.
mod = types.ModuleType("loggers")
mod.get_logger = lambda *a, **k: types.SimpleNamespace(
    info=lambda *a, **k: None, debug=lambda *a, **k: None,
    warning=lambda *a, **k: None, error=lambda *a, **k: None,
)
sys.modules.setdefault("loggers", mod)

from hub.utils.gguf import group_gguf_variant_files, gguf_variant_key  # noqa: E402

# Frozen from https://huggingface.co/api/models/<repo> on 2026-09-09.
LISTINGS = json.load(open(os.environ["LISTINGS"]))

failures = []
for repo, files in LISTINGS.items():
    pairs = [(f["rfilename"], f["size"]) for f in files]
    rows = group_gguf_variant_files(pairs)
    print(f"\n=== {repo}")
    print(f"    {len(pairs)} published GGUF build(s) -> {len(rows)} picker row(s)")
    for key in sorted(rows):
        filename, size = rows[key]
        print(f"      row {key!r:52} fetches {filename}")
    listed = {rows[k][0] for k in rows}
    missing = sorted({p for p, _ in pairs} - listed)
    for path in missing:
        failures.append(f"{repo}: {path} is published but NO row fetches it")
    # A row must fetch the file whose name it is derived from.
    for key in sorted(rows):
        filename, _ = rows[key]
        if gguf_variant_key(filename) != key:
            failures.append(f"{repo}: row {key!r} fetches {filename}, whose own key is {gguf_variant_key(filename)!r}")

print("\n" + "=" * 72)
if failures:
    print(f"FAIL: {len(failures)} unreachable-or-mislabelled build(s)")
    for f in failures:
        print("  FAIL " + f)
    sys.exit(1)
print("PASS: every published build has its own row, and every row fetches its own file")
