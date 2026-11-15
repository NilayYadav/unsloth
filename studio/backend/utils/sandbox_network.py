# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Settings > Sandbox network allowlist for sandboxed Python and Terminal runs."""

from __future__ import annotations

import ipaddress
import re
from typing import Any

ENABLED_SETTING_KEY = "sandbox_network_allowlist"
EXTRA_HOSTS_SETTING_KEY = "sandbox_network_extra_hosts"
MAX_EXTRA_HOSTS = 100
_LABEL = re.compile(r"^[a-z0-9_]([a-z0-9_-]{0,61}[a-z0-9_])?$")


def normalize_host(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    host = value.strip().lower().rstrip(".")
    if host == "*":
        return host
    name = host[2:] if host.startswith("*.") else host
    if not name or len(name) > 253:
        return None
    try:
        ipaddress.ip_address(name)
        return None
    except ValueError:
        pass
    labels = name.split(".")
    if len(labels) < 2 or not all(_LABEL.match(label) for label in labels):
        return None
    return host


def _read(key: str, default: Any) -> Any:
    try:
        from storage.studio_db import get_app_setting
        from utils.account_context import OWNER, run_as

        # Installation-wide: the owner's store, whichever account's tool call is asking.
        return run_as(OWNER, get_app_setting, key, default)
    except Exception:  # noqa: BLE001 - an unreadable store keeps the shipped default
        return default


def allowlist_enabled() -> bool:
    return _read(ENABLED_SETTING_KEY, False) is True


def extra_hosts() -> tuple[str, ...]:
    stored = _read(EXTRA_HOSTS_SETTING_KEY, [])
    if not isinstance(stored, list):
        return ()
    hosts = (normalize_host(item) for item in stored[:MAX_EXTRA_HOSTS])
    return tuple(dict.fromkeys(host for host in hosts if host))


def clean_hosts(hosts: list[Any]) -> list[str]:
    if len(hosts) > MAX_EXTRA_HOSTS:
        raise ValueError(f"Add at most {MAX_EXTRA_HOSTS} hosts.")
    cleaned = []
    for item in hosts:
        host = normalize_host(item)
        if host is None:
            raise ValueError(
                f"{str(item).strip()!r} is not a host name. Use a name like example.com or *.example.com."
            )
        cleaned.append(host)
    return list(dict.fromkeys(cleaned))


def save(enabled: bool | None = None, hosts: list[Any] | None = None) -> None:
    updates: dict[str, Any] = {}
    if enabled is not None:
        updates[ENABLED_SETTING_KEY] = bool(enabled)
    if hosts is not None:
        updates[EXTRA_HOSTS_SETTING_KEY] = clean_hosts(hosts)
    if not updates:
        return
    from storage.studio_db import upsert_app_settings
    from utils.account_context import OWNER, run_as

    run_as(OWNER, upsert_app_settings, updates)
