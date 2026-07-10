# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

# Local state under ~/.unsloth: remotes.toml, job mirrors, pulled artifacts.

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from unsloth_cli.remote import RemoteError, looks_like_job_id

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

import tomli_w

DEFAULT_AGENT_PORT = 8877
REMOTE_ENV_PYTHON = "~/.unsloth/env/bin/python"
ARTIFACT_FILE = "artifact.json"
ADAPTER_SUBDIR = "adapter"
# Never persist credentials in job records; resume re-injects them from env/flags.
_SECRET_PAYLOAD_KEYS = ("hf_token", "wandb_token", "s3_config")


def unsloth_home() -> Path:
    return Path(os.environ.get("UNSLOTH_HOME", "~/.unsloth")).expanduser()


def remotes_file() -> Path:
    return unsloth_home() / "remotes.toml"


def ssh_dir() -> Path:
    return unsloth_home() / "ssh"


def known_hosts_file() -> Path:
    return ssh_dir() / "known_hosts"


def jobs_dir() -> Path:
    return unsloth_home() / "jobs"


def registry_dir() -> Path:
    return unsloth_home() / "registry"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec = "seconds")


def _write_private(path: Path, text: str) -> None:
    # Atomic write, 0600.
    path.parent.mkdir(parents = True, exist_ok = True)
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding = "utf-8") as f:
            f.write(text)
    except BaseException:
        tmp.unlink(missing_ok = True)
        raise
    os.replace(tmp, path)


class RemoteCapabilities(BaseModel):
    gpu_names: List[str] = Field(default_factory = list)
    vram_gb: List[float] = Field(default_factory = list)
    driver: Optional[str] = None
    cuda: Optional[str] = None
    python: Optional[str] = None
    arch: Optional[str] = None
    os_release: Optional[str] = None
    disk_free_gb: Optional[float] = None
    has_rsync: bool = False
    has_tmux: bool = False
    has_systemd_user: bool = False
    has_curl: bool = False
    has_wget: bool = False
    has_uv: bool = False
    is_root: bool = False
    has_passwordless_sudo: bool = False

    @property
    def can_install_packages(self) -> bool:
        return self.is_root or self.has_passwordless_sudo


class RemoteRecord(BaseModel):
    name: str
    host: str
    user: str
    port: int = 22
    identity_file: Optional[str] = None
    host_key_fingerprint: Optional[str] = None
    agent_port: int = DEFAULT_AGENT_PORT
    api_key: Optional[str] = None
    supervision: Literal["systemd", "tmux", "none"] = "none"
    env_python: str = REMOTE_ENV_PYTHON
    unsloth_version: Optional[str] = None
    added_at: Optional[str] = None
    capabilities: RemoteCapabilities = Field(default_factory = RemoteCapabilities)

    @property
    def destination(self) -> str:
        return f"{self.user}@{self.host}"


def load_remotes() -> Dict[str, RemoteRecord]:
    path = remotes_file()
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        data = tomllib.load(f)
    remotes = {}
    for name, entry in (data.get("remotes") or {}).items():
        remotes[name] = RemoteRecord(name = name, **{k: v for k, v in entry.items() if k != "name"})
    return remotes


def save_remotes(remotes: Dict[str, RemoteRecord]) -> None:
    data = {
        "remotes": {
            name: record.model_dump(mode = "json", exclude_none = True, exclude = {"name"})
            for name, record in sorted(remotes.items())
        }
    }
    _write_private(remotes_file(), tomli_w.dumps(data))


def get_remote(name: str) -> RemoteRecord:
    remotes = load_remotes()
    if name in remotes:
        return remotes[name]
    known = ", ".join(sorted(remotes)) if remotes else "none configured"
    raise RemoteError(
        f"Unknown remote '{name}' (known remotes: {known}).",
        hint = "Add one with: unsloth remote add <name> <user@host>",
    )


def upsert_remote(record: RemoteRecord) -> None:
    remotes = load_remotes()
    remotes[record.name] = record
    save_remotes(remotes)


def remove_remote(name: str) -> RemoteRecord:
    remotes = load_remotes()
    try:
        record = remotes.pop(name)
    except KeyError:
        raise RemoteError(f"Unknown remote '{name}'.") from None
    save_remotes(remotes)
    return record


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
    # Job id -> pulled adapter path; anything else passes through unchanged.
    if not identifier or not looks_like_job_id(identifier):
        return identifier
    path = resolve_job_artifact(identifier)
    if path is not None:
        return path
    try:
        job = resolve_job(identifier)
    except RemoteError:
        return identifier  # not a known job; let normal model resolution try
    raise RemoteError(
        f"Job {job.job_id} exists but its adapter has not been pulled to this machine.",
        hint = f"Fetch it with: unsloth pull {job.job_id}",
    )
