# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Client-side support for running Unsloth on remote GPU machines over SSH."""

from __future__ import annotations

import re

JOB_ID_RE = re.compile(r"^job_\d{8}_\d{6}_[0-9a-f]+$")
JOB_ID_PREFIX_RE = re.compile(r"^job_[0-9a-f_]*$")


class RemoteError(Exception):
    """Error in remote operations, carrying an optional remediation hint."""

    def __init__(self, message: str, hint: str = ""):
        super().__init__(message)
        self.hint = hint


def looks_like_job_id(identifier: str) -> bool:
    """True when `identifier` could be a job id or a job-id prefix."""
    return bool(identifier) and identifier.startswith("job_")
