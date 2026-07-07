# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Local job-record cache: one JSON file per remote job under ~/.unsloth/jobs.

The record on the VM (Studio run history) stays authoritative; these files are
mirrors so `unsloth jobs/logs/pull/resume` can resolve a job after the fact,
even when the remote is unreachable.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from unsloth_cli.remote import RemoteError, looks_like_job_id
from unsloth_cli.remote.state import _write_private, unsloth_home

# Never persist credentials in job records; resume re-injects them from env/flags.
_SECRET_PAYLOAD_KEYS = ("hf_token", "wandb_token", "s3_config")


def jobs_dir():
    return unsloth_home() / "jobs"


class JobRecord(BaseModel):
    job_id: str
    remote: str
    kind: Literal["train", "export"] = "train"
    submitted_at: Optional[str] = None
    config_file: Optional[str] = None
    payload: dict = Field(default_factory = dict)
    uploaded_datasets: List[str] = Field(default_factory = list)
    status: str = "unknown"
    model_name: Optional[str] = None
    output_dir: Optional[str] = None
    final_step: Optional[int] = None
    total_steps: Optional[int] = None
    final_loss: Optional[float] = None
    resumed_from: Optional[str] = None
    pulled: bool = False
    registry_path: Optional[str] = None


def sanitize_payload(payload: dict) -> dict:
    return {k: v for k, v in payload.items() if k not in _SECRET_PAYLOAD_KEYS}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec = "seconds")


def save_job(record: JobRecord) -> None:
    record.payload = sanitize_payload(record.payload)
    path = jobs_dir() / f"{record.job_id}.json"
    _write_private(path, json.dumps(record.model_dump(mode = "json"), indent = 2))


def load_job(job_id: str) -> Optional[JobRecord]:
    path = jobs_dir() / f"{job_id}.json"
    if not path.exists():
        return None
    return JobRecord(**json.loads(path.read_text(encoding = "utf-8")))


def list_jobs() -> List[JobRecord]:
    directory = jobs_dir()
    if not directory.exists():
        return []
    records = []
    for path in sorted(directory.glob("job_*.json")):
        try:
            records.append(JobRecord(**json.loads(path.read_text(encoding = "utf-8"))))
        except (json.JSONDecodeError, ValueError):
            continue
    return records


def resolve_job(identifier: str) -> JobRecord:
    """Resolve a job id or unique prefix to its local record."""
    if not looks_like_job_id(identifier):
        raise RemoteError(
            f"'{identifier}' does not look like a job id (expected job_...).",
            hint = "List jobs with: unsloth jobs",
        )
    exact = load_job(identifier)
    if exact is not None:
        return exact
    matches = [r for r in list_jobs() if r.job_id.startswith(identifier)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise RemoteError(
            f"No job matching '{identifier}' found locally.",
            hint = "List jobs with: unsloth jobs",
        )
    options = ", ".join(r.job_id for r in matches[:5])
    raise RemoteError(f"Job prefix '{identifier}' is ambiguous: {options}")
