# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Local state for remotes: ~/.unsloth paths and the remotes.toml store."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from unsloth_cli.remote import RemoteError

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

import tomli_w

DEFAULT_AGENT_PORT = 8877
REMOTE_ENV_PYTHON = "~/.unsloth/env/bin/python"


def unsloth_home() -> Path:
    """Root for client-side remote state; UNSLOTH_HOME overrides for tests."""
    return Path(os.environ.get("UNSLOTH_HOME", "~/.unsloth")).expanduser()


def remotes_file() -> Path:
    return unsloth_home() / "remotes.toml"


def ssh_dir() -> Path:
    return unsloth_home() / "ssh"


def known_hosts_file() -> Path:
    return ssh_dir() / "known_hosts"


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


def _write_private(path: Path, text: str) -> None:
    """Atomically write `text` to `path` with 0600 permissions."""
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
