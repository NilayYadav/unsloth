"""Assert the expected outcome for one PR 10090 repro variant."""
import json
import sys

VARIANT = sys.argv[1]
d = json.load(open("probe_out.json"))
oversize = d["cases"]["oversize"]
starvation = d["cases"]["starvation"]
ib_nocount = d["in_band"]["oversize_no_counts"]
ib_starv = d["in_band"]["starvation"]
ib_struct = d["in_band"]["structured_only_counts"]
fails = []


def check(label, cond, got):
    print(("PASS  " if cond else "FAIL  ") + label + "  ->  " + repr(got))
    if not cond:
        fails.append(label)


print("variant=%s rendered_by=%s in_band_by=%s"
      % (VARIANT, d["rendered_by"], d["in_band_by"]))

if VARIANT == "base-before-pr":
    # The defect PR 10090 exists to fix: the client saw a raw llama-server string.
    check("oversize is a raw llama-server error",
          oversize["message"].startswith("llama-server error:"), oversize["message"])
    check("oversize never says the prompt is too long",
          "too long" not in oversize["message"].lower(), oversize["message"])
    check("in-band count-less oversize is an internal error",
          "internal error" in ib_nocount["message"].lower(), ib_nocount["message"])

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

elif VARIANT == "head-inband-gap":
    # Both non-streaming P1s are fixed here...
    check("non-stream oversize is reworded",
          "Prompt is too long: 214331 tokens > 131072 maximum" in oversize["message"],
          oversize["message"])
    check("non-stream starvation is excluded",
          starvation["classify"] is None, starvation["classify"])
    # ...but the in-band SSE surface still flattens the count-less wording.
    check("GAP PRESENT: in-band count-less oversize is an internal error",
          "internal error" in ib_nocount["message"].lower(), ib_nocount["message"])
    check("GAP PRESENT: and it is still sent as invalid_request_error",
          ib_nocount["type"] == "invalid_request_error", ib_nocount["type"])
    check("GAP PRESENT: in-band starvation is an internal error too",
          "internal error" in ib_starv["message"].lower(), ib_starv["message"])
    check("GAP PRESENT: structured counts are lost too",
          "70494" not in ib_struct["message"], ib_struct["message"])

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
    check("in-band count-less oversize is now readable",
          ib_nocount["message"].startswith("Prompt is too long"), ib_nocount["message"])
    check("in-band count-less oversize is still a client error",
          ib_nocount["type"] == "invalid_request_error", ib_nocount["type"])
    check("in-band starvation keeps the shared-cache explanation",
          "shared pool of context" in ib_starv["message"], ib_starv["message"])
    check("in-band starvation is not a client error",
          ib_starv["type"] == "api_error", ib_starv["type"])
    check("in-band recovers counts from the structured fields",
          "Prompt is too long: 70494 tokens > 67584 maximum" in ib_struct["message"],
          ib_struct["message"])
else:
    raise SystemExit("unknown variant " + VARIANT)

print()
if fails:
    print("VARIANT %s: %d expectation(s) not met: %s" % (VARIANT, len(fails), fails))
    raise SystemExit(1)
print("VARIANT %s: all expectations met" % VARIANT)
