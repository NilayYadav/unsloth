# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import inspect
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def test_studio_default_exposes_share_gpu_off_by_default():
    from unsloth_cli.commands import studio as studio_mod

    opt = inspect.signature(studio_mod.studio_default).parameters["share_gpu"].default
    assert "--share-gpu" in set(getattr(opt, "param_decls", []) or [])
    assert getattr(opt, "default", None) is False


def test_share_gpu_hands_the_request_to_the_backend_through_the_environment():
    from unsloth_cli.commands import studio as studio_mod

    source = inspect.getsource(studio_mod.studio_default)
    assert 'os.environ["UNSLOTH_CLUSTER_SHARE"] = "1"' in source
