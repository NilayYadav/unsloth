# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlsplit

_GOOGLE_MCP_SCOPES: dict[str, list[str]] = {
    "gmailmcp.googleapis.com": [
        "https://www.googleapis.com/auth/gmail.readonly",
        "https://www.googleapis.com/auth/gmail.compose",
    ],
    "drivemcp.googleapis.com": [
        "https://www.googleapis.com/auth/drive.readonly",
        "https://www.googleapis.com/auth/drive.file",
    ],
    "calendarmcp.googleapis.com": [
        "https://www.googleapis.com/auth/calendar.calendarlist.readonly",
        "https://www.googleapis.com/auth/calendar.events.readonly",
        "https://www.googleapis.com/auth/calendar.events.freebusy",
    ],
}


@dataclass(frozen=True)
class ConnectorOAuth:
    client_id: str
    client_secret: str
    scopes: list[str]


def _host_of(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""


def _google_client() -> tuple[str | None, str | None]:
    return (
        os.environ.get("UNSLOTH_GOOGLE_MCP_CLIENT_ID") or None,
        os.environ.get("UNSLOTH_GOOGLE_MCP_CLIENT_SECRET") or None,
    )


def is_managed_connector(url: str) -> bool:
    return _host_of(url) in _GOOGLE_MCP_SCOPES


def connector_oauth_for(url: str) -> ConnectorOAuth | None:
    scopes = _GOOGLE_MCP_SCOPES.get(_host_of(url))
    if scopes is None:
        return None
    client_id, client_secret = _google_client()
    if not client_id or not client_secret:
        return None
    return ConnectorOAuth(client_id, client_secret, list(scopes))


def connector_needs_setup(url: str) -> bool:
    return is_managed_connector(url) and connector_oauth_for(url) is None
