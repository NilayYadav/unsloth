"""PR 10165 probe: what a compaction carries forward when the user typed beside an image.

Identical across variants. Only studio/backend/core/inference/checkpoint.py is swapped,
so every difference in the printed values comes from that one file.
"""
import sys

sys.path.insert(0, "studio/backend")

from core.inference.checkpoint import carried_forward_items, fit_checkpoint_context  # noqa: E402
from core.inference.context_window import estimate_message_tokens  # noqa: E402

INSTRUCTION = (
    "Standing instruction for the rest of this task: always report results as a markdown "
    "table, and end every reply with STATUS::ZQXVARA123-ALPHA."
)
PAYLOAD = 30000


def image_turn(text, payload = PAYLOAD):
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * payload}},
        ],
    }


def count(messages):
    return sum(max(1, len(str(m.get("content", ""))) // 4) for m in messages)


def fit(messages):
    return fit_checkpoint_context(
        messages,
        context_length = 1200,
        max_tokens = 200,
        count_tokens = count,
        can_reset = True,
    )


def main():
    failures = []

    turn = image_turn(INSTRUCTION)
    whole = estimate_message_tokens(turn)
    words = estimate_message_tokens({"role": "user", "content": INSTRUCTION})
    print("=" * 72)
    print("A. what one image turn is priced at, against the words it can actually carry")
    print("-" * 72)
    print(f"  image payload chars              : {PAYLOAD}")
    print(f"  estimate_message_tokens(turn)    : {whole}")
    print(f"  estimate_message_tokens(text)    : {words}")
    print(f"  overcharge factor                : {whole / max(1, words):.1f}x")
    print(f"  carried-forward budget in a reset: 240 tokens (MAX_FRACTION of a 1200 ctx)")

    print()
    print("=" * 72)
    print("B. carried_forward_items on that single turn, budget 1024")
    print("-" * 72)
    items = carried_forward_items([turn], max_tokens = 1024)
    print(f"  items carried : {len(items)}")
    for item in items:
        print(f"    - {item}")
    if items != [INSTRUCTION]:
        failures.append("B: the instruction typed beside the image was NOT carried")

    print()
    print("=" * 72)
    print("C. the same words with no image attached (the control)")
    print("-" * 72)
    plain = carried_forward_items([{"role": "user", "content": INSTRUCTION}], max_tokens = 1024)
    print(f"  items carried : {len(plain)}")
    for item in plain:
        print(f"    - {item}")
    print(f"  image turn == plain turn : {items == plain}")
    if items != plain:
        failures.append("C: an image turn is not treated like the words it carries")

    print()
    print("=" * 72)
    print("D. a real thread opened with a screenshot, compacted")
    print("-" * 72)
    messages = [
        {"role": "system", "content": "you are helpful"},
        image_turn(INSTRUCTION),
        {"role": "assistant", "content": "Understood."},
    ]
    for index in range(8):
        messages += [
            {"role": "user", "content": f"Section {index}. " + "x" * 600},
            {"role": "assistant", "content": f"Section {index} noted."},
        ]
    messages += [{"role": "user", "content": "continue"}]

    fitted, truncation = fit(messages)
    started = truncation and truncation.get("checkpoint_started")
    system_turn = fitted[0]["content"]
    print(f"  messages in  : {len(messages)}")
    print(f"  messages out : {len(fitted)}")
    print(f"  checkpoint_started : {started}")
    print("  system turn the model actually receives:")
    for line in system_turn.splitlines():
        print(f"    | {line}")
    if not started:
        failures.append("D: no checkpoint reset happened")
    if INSTRUCTION not in system_turn:
        failures.append("D: the user's standing instruction is MISSING from the system turn")

    print()
    print("=" * 72)
    print("E. guard: a bare nudge sent with an image must not be quoted as an instruction")
    print("-" * 72)
    for nudge in ("ok", "continue"):
        got = carried_forward_items([image_turn(nudge)], max_tokens = 1024)
        print(f"  {nudge!r:12} -> {got}")
        if got:
            failures.append(f"E: {nudge!r} was quoted as a standing instruction")

    print()
    print("=" * 72)
    print("F. guard: an instruction larger than the whole budget is still excluded")
    print("-" * 72)
    oversized = "Always " + "w " * 2000
    got = carried_forward_items([image_turn(oversized)], max_tokens = 64)
    print(f"  oversized instruction chars : {len(oversized)}")
    print(f"  items carried at cap 64     : {len(got)}")
    if got:
        failures.append("F: an oversized instruction was carried under a 64-token cap")

    print()
    print("=" * 72)
    print("G. no image at all: the same words as a list of text parts vs as a string")
    print("-" * 72)
    listed = {"role": "user", "content": [{"type": "text", "text": INSTRUCTION}]}
    string = {"role": "user", "content": INSTRUCTION}
    cost = estimate_message_tokens(string)
    print(f"  price of the list-shaped turn : {estimate_message_tokens(listed)}")
    print(f"  price of the identical string : {cost}")
    print(f"  cap used below                : {cost}")
    as_list = carried_forward_items([listed], max_tokens = cost)
    as_string = carried_forward_items([string], max_tokens = cost)
    print(f"  carried as a list  : {len(as_list)}")
    print(f"  carried as a string: {len(as_string)}")
    if as_list != as_string:
        failures.append("G: the same words are carried as a string and dropped as a list")

    print()
    print("=" * 72)
    if failures:
        print(f"PROBE RESULT: FAIL ({len(failures)})")
        for line in failures:
            print(f"  FAIL {line}")
        return 1
    print("PROBE RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
