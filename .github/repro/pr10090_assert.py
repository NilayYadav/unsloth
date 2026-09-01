"""Assert the expected outcome for one PR 10090 repro variant."""
import json
import sys

VARIANT = sys.argv[1]
d = json.load(open("probe_out.json"))
oversize = d["cases"]["oversize"]
starvation = d["cases"]["starvation"]
fails = []


def check(label, cond, got):
    print(("PASS  " if cond else "FAIL  ") + label + "  ->  " + repr(got))
    if not cond:
        fails.append(label)


print("variant=%s rendered_by=%s" % (VARIANT, d["rendered_by"]))

if VARIANT == "base-before-pr":
    # The defect PR 10090 exists to fix: the client saw a raw llama-server string.
    check("oversize is a raw llama-server error",
          oversize["message"].startswith("llama-server error:"), oversize["message"])
    check("oversize never says the prompt is too long",
          "too long" not in oversize["message"].lower(), oversize["message"])

elif VARIANT == "head-before-fix":
    # The PR's feature works...
    check("oversize is reworded with both token counts",
          "Prompt is too long: 214331 tokens > 131072 maximum" in oversize["message"],
          oversize["message"])
    # ...but Codex's P1: starvation is swept into the same rewrite.
    check("REGRESSION PRESENT: starvation is also called too long",
          "too long" in starvation["message"].lower(), starvation["message"])
    check("REGRESSION PRESENT: starvation is classified as an overflow",
          starvation["classify"] is True, starvation["classify"])
    check("REGRESSION PRESENT: starvation is sent to the client as 400",
          starvation["status_under_pr_formula"] == 400,
          starvation["status_under_pr_formula"])

elif VARIANT == "head-fixed":
    check("oversize is still reworded with both token counts",
          "Prompt is too long: 214331 tokens > 131072 maximum" in oversize["message"],
          oversize["message"])
    check("oversize is still classified as an overflow",
          oversize["classify"] is True, oversize["classify"])
    check("starvation is no longer called too long",
          "too long" not in starvation["message"].lower(), starvation["message"])
    check("starvation gets the shared-cache explanation",
          "shared pool of context" in starvation["message"], starvation["message"])
    check("starvation is no longer classified as an overflow",
          starvation["classify"] is None, starvation["classify"])
    check("starvation no longer becomes a 400",
          starvation["status_under_pr_formula"] == 500,
          starvation["status_under_pr_formula"])
else:
    raise SystemExit("unknown variant " + VARIANT)

print()
if fails:
    print("VARIANT %s: %d expectation(s) not met: %s" % (VARIANT, len(fails), fails))
    raise SystemExit(1)
print("VARIANT %s: all expectations met" % VARIANT)
