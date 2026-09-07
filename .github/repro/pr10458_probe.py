# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Print the chat_template_kwargs Studio actually sends for every Think-menu level.

Drives the real LlamaCppBackend._request_reasoning_kwargs on a local model whose
template advertises the full 'none'..'max' ladder, exactly the way the chat route
calls it. Prints one row per level and exits non-zero when a picked level does
not reach the request.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "studio", "backend"))

from core.inference.llama_cpp import (  # noqa: E402
    LlamaCppBackend,
    detect_reasoning_flags,
)

# A reasoning_effort-style template that branches on the whole ladder, plus the
# numeric dial such a model is served with.
WIDE_TEMPLATE = (
    "{%- if reasoning_effort == 'none' -%}N"
    "{%- elif reasoning_effort == 'minimal' -%}m"
    "{%- elif reasoning_effort == 'low' -%}l"
    "{%- elif reasoning_effort == 'medium' -%}M"
    "{%- elif reasoning_effort == 'high' -%}H"
    "{%- elif reasoning_effort == 'xhigh' -%}X"
    "{%- elif reasoning_effort == 'max' -%}Z"
    "{%- endif -%}"
)

GPT_OSS_TEMPLATE = "{%- set effort = reasoning_effort or 'medium' -%}{{- 'Reasoning: ' + effort -}}"


def backend(levels, architecture):
    b = object.__new__(LlamaCppBackend)
    b._supports_reasoning = True
    b._reasoning_always_on = False
    b._reasoning_style = "reasoning_effort"
    b._reasoning_effort_levels = levels
    b._supports_preserve_thinking = False
    b._architecture = architecture
    return b


def main() -> int:
    flags = detect_reasoning_flags(WIDE_TEMPLATE, "vendor/Wide-GGUF")
    levels = flags["reasoning_effort_levels"]
    print(f"detected reasoning_style      = {flags['reasoning_style']}")
    print(f"detected reasoning_effort_levels = {levels}")
    print(f"Think menu offers            = {levels}")
    print()

    b = backend(levels, "inkling")
    print("model: wide ladder, numeric dial (architecture='inkling')")
    print(f"{'picked in Think menu':<22} {'chat_template_kwargs on the wire':<40} verdict")
    failures = []
    for level in levels:
        kw = b._request_reasoning_kwargs(None, level, None)
        sent = (kw or {}).get("reasoning_effort")
        expected = {
            "none": 0.0, "minimal": 0.1, "low": 0.2, "medium": 0.7,
            "high": 0.9, "xhigh": 0.99, "max": 0.99,
        }[level]
        ok = sent == expected
        if not ok:
            failures.append((level, kw, expected))
        print(f"{level:<22} {json.dumps(kw):<40} {'ok' if ok else 'DROPPED -> ' + repr(sent)}")

    print()
    gpt = detect_reasoning_flags(GPT_OSS_TEMPLATE, "unsloth/gpt-oss-20b-GGUF")
    print(f"model: gpt-oss (advertises no levels): reasoning_effort_levels = {gpt['reasoning_effort_levels']}")
    g = backend(gpt["reasoning_effort_levels"], None)
    for level in ("none", "low", "medium", "high", "minimal", "max"):
        print(f"  {level:<8} -> {json.dumps(g._request_reasoning_kwargs(None, level, None))}")
    print(f"  thinking off (enable_thinking=False) -> {json.dumps(g._request_reasoning_kwargs(False, None, None))}")

    print()
    if failures:
        for level, kw, expected in failures:
            print(f"FAIL: Think menu offers '{level}' but the request carries {json.dumps(kw)} (expected {expected})")
        print(f"FAIL: {len(failures)}/{len(levels)} advertised level(s) never reach the model")
        return 1
    print(f"PASS: all {len(levels)} advertised level(s) reach the model")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
