# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

# Client-side support for running Unsloth on remote GPU machines over SSH.

from __future__ import annotations


class RemoteError(Exception):
    
    def __init__(self, message: str, hint: str = ""):
        super().__init__(message)
        self.hint = hint


def looks_like_job_id(identifier: str) -> bool:
    return bool(identifier) and identifier.startswith("job_")
