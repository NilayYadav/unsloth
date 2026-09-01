"""PR 10165 A/B: run the same tree twice, swapping only checkpoint.py.

Both sides share one runner, one interpreter and one dependency set, so the only
difference between them is the file under test. The gate is not "everything is green":
this suite has failures that depend on the environment rather than on the change (the
/proc reachability probe off Linux, the archive embedder probes without one). The gate
is that the carry tests move from red to green and that NOTHING ELSE turns red.
"""
import json
import subprocess
import sys

FILE = "studio/backend/core/inference/checkpoint.py"
SUITE = "tests/test_checkpoint_compaction.py"
BASE = "5c8c238e63eac8904b04af3cd21d45e762c8ceff"
HEAD_SHA = sys.argv[1]

# The tests that exist to pin this fix. Red on base, green on head, on every runner.
CARRY = [
    "test_an_instruction_typed_beside_an_image_is_still_carried",
    "test_an_image_turn_costs_the_same_as_the_words_it_carries",
    "test_a_text_only_turn_costs_the_same_whether_it_arrives_as_a_list_or_a_string",
    "test_the_block_is_priced_with_the_estimator_the_caller_passed_in",
    "test_a_thread_opened_with_a_screenshot_still_names_its_task_after_a_reset",
    "test_an_image_turn_is_judged_on_its_words_not_its_attachment",
]


def swap(sha):
    subprocess.run(["git", "checkout", sha, "--", FILE], check = True)
    print(f"\n### checkpoint.py now at {sha}", flush = True)


def probe():
    done = subprocess.run([sys.executable, ".github/repro/pr10165_probe.py"])
    return done.returncode


def suite():
    """The failing test names, and the raw output, for one side."""
    done = subprocess.run(
        [sys.executable, "-m", "pytest", SUITE, "-q", "--no-header", "-rf"],
        cwd = "studio/backend",
        capture_output = True,
        text = True,
    )
    print(done.stdout[-8000:], flush = True)
    failed = set()
    for line in done.stdout.splitlines():
        if line.startswith("FAILED "):
            name = line.split(" ", 1)[1].split(" ")[0]
            failed.add(name.rsplit("::", 1)[-1])
    return failed


def main():
    swap(BASE)
    before_probe = probe()
    before = suite()

    swap(HEAD_SHA)
    after_probe = probe()
    after = suite()

    fixed = sorted(before - after)
    broken = sorted(after - before)
    still = sorted(after & before)

    print("\n" + "=" * 72)
    print("A/B RESULT (same runner, same deps, only checkpoint.py swapped)")
    print("=" * 72)
    print(f"probe   before: exit {before_probe}   after: exit {after_probe}")
    print(f"failing before: {len(before)}   after: {len(after)}")
    print(f"\nfixed by the change ({len(fixed)}):")
    for name in fixed:
        print(f"  + {name}")
    print(f"\nbroken by the change ({len(broken)}):")
    for name in broken:
        print(f"  - {name}")
    print(f"\nfailing on BOTH sides, so not this change ({len(still)}):")
    for name in still:
        print(f"  = {name}")

    problems = []
    if before_probe == 0:
        problems.append("the probe passed BEFORE the fix: the defect did not reproduce")
    if after_probe != 0:
        problems.append("the probe failed AFTER the fix")
    for name in CARRY:
        if name not in before:
            problems.append(f"{name} did not fail before the fix")
        if name in after:
            problems.append(f"{name} still fails after the fix")
    for name in broken:
        problems.append(f"{name} was green before the fix and is red after it")

    print("\n" + "=" * 72)
    if problems:
        print(f"A/B VERDICT: FAIL ({len(problems)})")
        for line in problems:
            print(f"  FAIL {line}")
        return 1
    print("A/B VERDICT: PASS")
    print("  the defect reproduces before the fix, is gone after it,")
    print("  and no test that was green before the fix is red after it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
