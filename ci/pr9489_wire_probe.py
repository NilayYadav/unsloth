#!/usr/bin/env python3
"""#9484 A/B probe: what a Stop-before-output puts on the wire, from whatever tree is checked out.

Slices the real serialisers and the real outbound prune out of chat-adapter.ts and runs them
under node over the reported history (prompt -> Stop with no output -> same prompt again).
Resolves the prune the same way either tree spells it, so the negative job (chat-adapter.ts
reverted to main) and the positive job (the PR) run the SAME probe and differ only in the
implementation under test.

Then renders that wire through the repo's own vendored ``gemma3_template`` -- the strict
alternation check named in ``_coalesce_consecutive_user_turns`` -- so the failure is the real
Jinja exception the reporter saw, not a description of one.

Exit 0 when exactly one user turn reaches the model and the template renders; exit 1 with the
wire and the template's own error printed otherwise.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHAT_TEMPLATES = ROOT / "unsloth/chat_templates.py"
ADAPTER = ROOT / "studio/frontend/src/features/chat/api/chat-adapter.ts"
CODEX = ROOT / "studio/frontend/src/features/chat/codex-reasoning.ts"
CONT = ROOT / "studio/frontend/src/features/chat/utils/continuation.ts"

OLD_PRUNE_START = (
    "  const survivingMessages: RunMessage[] = [];\n  for (const message of messages) {"
)
OLD_PRUNE_END = "\n  const outboundMessages = survivingMessages"

STUBS = """
// @ts-nocheck
function readCodexReasoning(_m: any): any { return undefined; }
function codexReasoningForToolCalls(_l: any, _i: any): any { return undefined; }
function getToolReplayProvenance(p: any): any { return p?.provenance; }
function shouldFlushCompletedLocalToolPair(_p: any): boolean { return false; }
function canReplayToolCallWithoutRoleTool(_p: any): boolean { return false; }
function serializeAssistantToolCallPart(_p: any): any { return null; }
function serializeToolResultPart(_p: any): any { return null; }
// ---- verbatim studio source follows ----
"""


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def slice_between(text: str, start: str, end: str) -> str:
    at = text.index(start)
    return text[at : text.index(end, at + len(start))]


def prune_shim(adapter: str) -> tuple[str, str]:
    """The tree's own outbound prune, exposed under one name."""
    if "function pruneOutboundHistory(" in adapter:
        return (
            "export const prune = (m: any[]) => pruneOutboundHistory(m, true);\n",
            "pruneOutboundHistory (this PR)",
        )
    body = slice_between(adapter, OLD_PRUNE_START, OLD_PRUNE_END)
    return (
        "export function prune(messages: any[]): any[] {\n"
        + body
        + "\n  return survivingMessages;\n}\n",
        "the refusal-only loop in buildOutboundMessagesForTokenCount (main)",
    )


def harness() -> tuple[str, str]:
    adapter = read(ADAPTER)
    shim, which = prune_shim(adapter)
    source = (
        STUBS
        + slice_between(adapter, "function collectTextParts(", "function normalizeOpenAIReasoningItem(")
        + slice_between(adapter, "// Refusal flag stamped on assistant metadata", "type SerializedMessage = {")
        + slice_between(adapter, "function sanitizeAssistantReplayText(", "function serializeAssistantReplayMessages(")
        + slice_between(adapter, "function serializeAssistantReplayMessages(", "function extractImageBase64(")
        + slice_between(read(CODEX), "export function codexLocalToolRoundId(", "export function addCodexReasoning(")
        + slice_between(read(CONT), "/** Why a turn ended before the model was done. */", "const INCOMPLETE_LABELS")
        + shim
        + """
// The local backend's own contract (_normalize_local_assistant_message in routes/inference.py):
// an assistant turn with no content, no tool_calls and no reasoning_content is a Stop sentinel
// and is dropped. Applied here so the probe reports what the MODEL sees, not what was posted.
export function wire(messages: any[]): any[] {
  return messages
    .flatMap((m: any) => toOpenAIMessages(m, true))
    .filter((m: any) => !(m.role === "assistant" && !m.content && !m.tool_calls && !m.reasoning_content))
    .map((m: any) => ({ role: m.role, content: m.content }));
}
"""
    )
    return source, which


SCRIPT = textwrap.dedent(
    """
    // @ts-nocheck
    import { prune, wire } from "./harness.ts";
    const user = (t) => ({ role: "user", content: [{ type: "text", text: t }] });
    // #9484 verbatim: run a prompt, press Stop before a single token arrives, send it again.
    const stopped = {
      role: "assistant",
      content: [],
      status: { type: "incomplete" },
      metadata: { custom: { incomplete: { reason: "cancelled" } } },
    };
    const PROMPT = "Write a haiku about the sea";
    const history = [user(PROMPT), stopped, user(PROMPT)];
    console.log(JSON.stringify({ wire: wire(prune(history)) }));
    """
)


def render_through_gemma3(turns: list[dict]) -> tuple[bool, str]:
    """Render the wire through the vendored Gemma 3 template, exactly as tests/test_gemma4_chat_template.py does."""
    import re

    from jinja2 import Environment, StrictUndefined
    from jinja2.exceptions import TemplateError

    source = CHAT_TEMPLATES.read_text(encoding="utf-8")
    match = re.search(r'gemma3_template\s*=\s*\\\n"""(.*?)"""', source, flags=re.DOTALL)
    if not match:
        return False, "gemma3_template could not be read out of unsloth/chat_templates.py"
    env = Environment(undefined=StrictUndefined, trim_blocks=False, lstrip_blocks=False)
    env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(TemplateError(msg))
    try:
        rendered = env.from_string(match.group(1)).render(
            messages=turns, add_generation_prompt=True, bos_token="<bos>"
        )
    except TemplateError as err:
        return False, f"{type(err).__name__}: {err}"
    return True, rendered


def main() -> int:
    source, which = harness()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / "harness.ts").write_text(source, encoding="utf-8")
        (work / "run.mts").write_text(SCRIPT, encoding="utf-8")
        proc = subprocess.run(
            ["node", "--experimental-strip-types", "--no-warnings", "run.mts"],
            cwd=work,
            capture_output=True,
            text=True,
            timeout=120,
            env=dict(os.environ, NODE_NO_WARNINGS="1"),
        )
    if proc.returncode != 0:
        print(f"probe could not run:\n{proc.stderr}", file=sys.stderr)
        return 2

    payload = json.loads([ln for ln in proc.stdout.strip().splitlines() if ln.strip()][-1])
    turns = payload["wire"]
    roles = [t["role"] for t in turns]

    print(f"prune under test : {which}")
    print(f"history posted   : user('Write a haiku about the sea'), assistant(Stop, no output), user(same)")
    print(f"roles on the wire: {roles}")
    for index, turn in enumerate(turns):
        print(f"  [{index}] {turn['role']}: {turn['content']!r}")

    ok, detail = render_through_gemma3(turns)
    print()
    print("rendered through the vendored gemma3_template (strict role alternation):")
    for line in detail.splitlines() or [""]:
        print(f"  {line}")

    users = roles.count("user")
    adjacent = any(a == "user" and b == "user" for a, b in zip(roles, roles[1:]))
    if adjacent or users != 1 or not ok:
        print()
        print(
            f"FAIL: {users} user turns reach the model"
            f"{' and two of them are adjacent' if adjacent else ''}."
            " The strict template refuses to render them; llama-server answers 400 on this."
        )
        return 1
    print()
    print("PASS: the abandoned turn and the prompt it stranded are both gone; one user turn is")
    print("      sent and the strict template renders it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
