# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Bare hosts ("google.com") must be fetched as https, not refused."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from core.inference import tools  # noqa: E402


@pytest.fixture
def resolved(monkeypatch):
    seen: dict = {}

    def fake_resolve(hostname, port, deadline, cancel_event):
        seen["hostname"] = hostname
        seen["port"] = port
        return False, "stopped", None

    monkeypatch.setattr(tools, "_resolve_with_budget", fake_resolve)
    return seen


@pytest.mark.parametrize(
    "url, hostname, port",
    [
        ("google.com", "google.com", 443),
        ("www.google.com/x", "www.google.com", 443),
        ("//google.com", "google.com", 443),
        ("https://google.com", "google.com", 443),
        ("http://google.com", "google.com", 80),
    ],
)
def test_schemeless_urls_are_fetched_as_https(resolved, url, hostname, port):
    err, _, _ = tools._fetch_url_raw(url)
    assert resolved["hostname"] == hostname
    assert resolved["port"] == port
    assert "only http/https" not in (err or "")


@pytest.mark.parametrize("url", ["ftp://x.com", "file:///etc/passwd", "javascript:alert(1)"])
def test_non_http_schemes_still_blocked(url):
    err, _, _ = tools._fetch_url_raw(url)
    assert err and "only http/https" in err


def test_blocked_message_shows_the_url_not_the_scheme():
    err, _, _ = tools._fetch_url_raw("ftp://x.com")
    assert "ftp://x.com" in err
