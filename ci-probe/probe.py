import importlib.util
import json
import sys
from pathlib import Path

BACKEND = Path(sys.argv[1]).resolve()
EXPECT = sys.argv[2]

CASES = {
    "squad_like": {"context": "x" * 600, "question": "y" * 120, "answer": "y" * 120},
    "squad_with_metadata": {"id": "1", "title": "t", "context": "x" * 600, "question": "y" * 120, "answers": "y" * 120},
    "dolly_like": {"instruction": "y" * 120, "context": "x" * 600, "response": "y" * 120, "category": "qa"},
    "rag_eval": {"retrieved_contexts": "x" * 600, "question": "y" * 120, "ground_truth_answer": "y" * 120},
    "two_column_context": {"context": "x" * 600, "response": "y" * 120},
    "alpaca_unchanged": {"instruction": "y" * 120, "input": "y" * 120, "output": "y" * 120},
    "input_text_unchanged": {"input_text": "y" * 120, "target_text": "y" * 120},
    "fulltext_unchanged": {"fulltext": "x" * 600, "answer": "y" * 120},
}

CORRECT = {
    "squad_like": {"question": "user", "context": "system", "answer": "assistant"},
    "squad_with_metadata": {"question": "user", "context": "system", "title": "system", "answers": "assistant"},
    "dolly_like": {"instruction": "user", "context": "system", "response": "assistant"},
    "rag_eval": {"question": "user", "retrieved_contexts": "system", "ground_truth_answer": "assistant"},
    "two_column_context": {"context": "user", "response": "assistant"},
    "alpaca_unchanged": {"instruction": "user", "input": "system", "output": "assistant"},
    "input_text_unchanged": {"input_text": "user", "target_text": "assistant"},
    "fulltext_unchanged": {"fulltext": "user", "answer": "assistant"},
}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


impls = {
    "utils": load(BACKEND / "utils" / "datasets" / "format_detection.py", "probe_format_detection"),
    "hub": load(BACKEND / "hub" / "utils" / "dataset_format.py", "probe_dataset_format"),
}

results = {}
failures = []
for impl_name, mod in impls.items():
    for case, row in CASES.items():
        got = mod.detect_custom_format_heuristic([row])
        want = CORRECT[case]
        ok = got == want
        results[f"{impl_name}/{case}"] = {"got": got, "want": want, "ok": ok}
        print(f"[{impl_name}/{case}] {'OK ' if ok else 'BAD'} got={got} want={want}")
        if not ok:
            failures.append(f"{impl_name}/{case}")

print()
print("FAILING_CASES=" + (",".join(failures) if failures else "<none>"))
Path("probe-result.json").write_text(json.dumps(results, indent=2, sort_keys=True))

if EXPECT == "broken":
    key_bugs = [f for f in failures if f.endswith(("/squad_like", "/squad_with_metadata", "/dolly_like", "/rag_eval"))]
    if len(key_bugs) != 8:
        print(f"REPRO FAILED: expected all 8 context-column cases wrong on base, got {len(key_bugs)}: {key_bugs}")
        raise SystemExit(1)
    print("REPRO CONFIRMED: base mis-maps every context/question/answer dataset in both heuristics.")
elif EXPECT == "fixed":
    if failures:
        print(f"FIX FAILED: still wrong on {failures}")
        raise SystemExit(1)
    print("FIX CONFIRMED: every case maps correctly in both heuristics.")
else:
    raise SystemExit(f"unknown expectation {EXPECT}")
