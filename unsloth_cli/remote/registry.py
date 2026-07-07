# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Registry of artifacts pulled from remotes: ~/.unsloth/registry/<job_id>/."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

from unsloth_cli.remote import RemoteError, looks_like_job_id
from unsloth_cli.remote.jobs import utc_now
from unsloth_cli.remote.state import unsloth_home

ARTIFACT_FILE = "artifact.json"
ADAPTER_SUBDIR = "adapter"
EXPORTS_SUBDIR = "exports"


def registry_dir() -> Path:
    return unsloth_home() / "registry"


class ArtifactRecord(BaseModel):
    job_id: str
    type: str = "adapter"
    path: str = ADAPTER_SUBDIR
    base_model: Optional[str] = None
    remote: Optional[str] = None
    pulled_at: Optional[str] = None
    gguf: List[str] = Field(default_factory = list)


def job_registry_dir(job_id: str) -> Path:
    return registry_dir() / job_id


def get_artifact(job_id: str) -> Optional[ArtifactRecord]:
    path = job_registry_dir(job_id) / ARTIFACT_FILE
    if not path.exists():
        return None
    return ArtifactRecord(**json.loads(path.read_text(encoding = "utf-8")))


def register_artifact(
    job_id: str,
    base_model: Optional[str] = None,
    remote: Optional[str] = None,
    gguf: Optional[List[str]] = None,
) -> ArtifactRecord:
    """Record a pulled artifact, merging into an existing record if present."""
    record = get_artifact(job_id) or ArtifactRecord(job_id = job_id)
    record.base_model = base_model or record.base_model
    record.remote = remote or record.remote
    record.pulled_at = utc_now()
    if gguf:
        record.gguf = sorted(set(record.gguf) | set(gguf))
    directory = job_registry_dir(job_id)
    directory.mkdir(parents = True, exist_ok = True)
    (directory / ARTIFACT_FILE).write_text(
        json.dumps(record.model_dump(mode = "json"), indent = 2), encoding = "utf-8"
    )
    return record


def _match_job_dir(identifier: str) -> Optional[str]:
    directory = registry_dir()
    if not directory.exists():
        return None
    if (directory / identifier / ARTIFACT_FILE).exists():
        return identifier
    matches = [
        p.name for p in directory.iterdir()
        if p.name.startswith(identifier) and (p / ARTIFACT_FILE).exists()
    ]
    return matches[0] if len(matches) == 1 else None


def resolve_job_artifact(identifier: Optional[str]) -> Optional[str]:
    """Map a job id (or unique prefix) to the pulled adapter path, else None.

    Non-job identifiers (HF ids, local paths) fall through untouched so this
    is safe to call unconditionally in `unsloth chat` / `unsloth inference`.
    """
    if not identifier or not looks_like_job_id(identifier):
        return None
    job_id = _match_job_dir(identifier)
    if job_id is None:
        return None
    record = get_artifact(job_id)
    if record is None:
        return None
    artifact_path = job_registry_dir(job_id) / record.path
    return str(artifact_path) if artifact_path.exists() else None


def resolve_model_identifier(identifier: Optional[str]) -> Optional[str]:
    """Resolve job ids to their pulled adapter path for chat/inference.

    Non-job identifiers pass through unchanged. A known job whose adapter has
    not been pulled raises with a `unsloth pull` hint instead of letting model
    resolution fail cryptically.
    """
    if not identifier or not looks_like_job_id(identifier):
        return identifier
    path = resolve_job_artifact(identifier)
    if path is not None:
        return path
    from unsloth_cli.remote.jobs import resolve_job
    try:
        job = resolve_job(identifier)
    except RemoteError:
        return identifier  # not a known job; let normal model resolution try
    raise RemoteError(
        f"Job {job.job_id} exists but its adapter has not been pulled to this machine.",
        hint = f"Fetch it with: unsloth pull {job.job_id}",
    )
