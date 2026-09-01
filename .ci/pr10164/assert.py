"""The verdict. Reads the three budget tables and states, per scenario, what each
revision sent and what the backend then gave the document."""

import json
import sys

MAIN, PIN, HEAD = sys.argv[1], sys.argv[2], sys.argv[3]
rows = {rev: json.load(open(path)) for rev, path in (
    ("main", MAIN), ("pin", PIN), ("head", HEAD))}

HOSTED = "hosted turn, a 4K GGUF still resident"
REDUCED = "local turn, load asked 8192, llama-server serves 4096"
PLAIN = "local turn, unreduced 4096 GGUF"

failures = []


def check(label, ok, detail):
    print(f"{'PASS' if ok else 'FAIL'}  {label}: {detail}")
    if not ok:
        failures.append(label)


def cell(rev, scenario):
    row = rows[rev][scenario]
    return row["context_length"], row["whole_doc_budget"]


print("\n=== what the frontend sends, and what the document gets ===")
for scenario in (HOSTED, REDUCED, PLAIN):
    print(f"\n{scenario}")
    for rev, sha in (("main", "5c8c238e6"), ("pin", "a98ad6ba3"), ("head", "d8b22f1e0")):
        window, budget = cell(rev, scenario)
        sent = "no window" if window is None else f"{window}"
        print(f"  {rev:5} {sha}  sends {sent:>9}  ->  {budget} tokens of document")

print("\n=== assertions ===")

main_window, main_budget = cell("main", HOSTED)
head_window, head_budget = cell("head", HOSTED)
check(
    "the reported defect reproduces on main",
    main_window == 4096 and main_budget < 6000,
    f"a hosted turn sent the resident GGUF's {main_window} and the document was cut to {main_budget}",
)
check(
    "the head fixes it",
    head_window is None and head_budget == 6000,
    f"the same turn sends no window and the document gets {head_budget} "
    f"(+{head_budget - main_budget} tokens, {head_budget / main_budget:.2f}x)",
)

pin_window, pin_budget = cell("pin", REDUCED)
served_window, served_budget = cell("head", REDUCED)
check(
    "the first commit's regression reproduces",
    pin_window == 8192 and pin_budget > served_budget,
    f"a load asked for 8192 and serves 4096; it budgeted {pin_budget} tokens "
    f"against a window that holds {served_budget}",
)
check(
    "the head budgets the served window",
    served_window == 4096 and served_budget == cell("main", REDUCED)[1],
    f"sends {served_window} -> {served_budget} tokens, the same as before the PR",
)

check(
    "an ordinary local turn is untouched",
    len({cell(rev, PLAIN) for rev in rows}) == 1,
    f"all three revisions send {cell('head', PLAIN)[0]} -> {cell('head', PLAIN)[1]} tokens",
)

print()
if failures:
    print(f"REPRO FAILED: {len(failures)} assertion(s): {failures}")
    raise SystemExit(1)
print("REPRO OK: every assertion held")
