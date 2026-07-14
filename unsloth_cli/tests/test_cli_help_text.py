# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for public CLI help text."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from typer.testing import CliRunner


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _help(*args: str) -> str:
    from unsloth_cli import app

    result = CliRunner().invoke(app, [*args, "--help"], terminal_width = 200)
    assert result.exit_code == 0, result.output
    return re.sub(r"[\s│]+", " ", result.output)


def test_train_help_describes_generated_config_options():
    output = _help("train")

    for description in (
        "Base model name or local path to fine-tune.",
        "Hugging Face dataset name.",
        "Optimizer learning rate.",
        "LoRA rank.",
        "Log training metrics to TensorBoard.",
    ):
        assert description in output


def test_studio_help_explains_wildcard_exposure():
    for args in (("studio",), ("studio", "run")):
        output = _help(*args)
        assert "a wildcard host (0.0.0.0 or ::) also exposes the raw port" in output


def test_studio_run_help_hides_internal_implementation_details():
    output = _help("studio", "run")

    assert "pre-PR" not in output
    assert "studio/backend/core/inference/llama_server_args.py" not in output
