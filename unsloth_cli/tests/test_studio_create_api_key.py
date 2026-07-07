# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the hidden `unsloth studio create-api-key` command."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import unsloth_cli.commands.studio as studiomod
from unsloth_cli.commands.studio import studio_app

runner = CliRunner()


class _FakeStorage:
    DEFAULT_ADMIN_USERNAME = "unsloth"

    def __init__(self):
        self.admin_ensured = False
        self.created = []

    def ensure_default_admin(self):
        self.admin_ensured = True
        return True

    def create_api_key(self, username, name):
        assert self.admin_ensured, "admin must exist before minting a key"
        self.created.append((username, name))
        return "sk-unsloth-testkey123", {"id": 1, "name": name}


@pytest.fixture()
def fake_storage(monkeypatch):
    storage = _FakeStorage()
    monkeypatch.setattr(studiomod, "_load_backend_auth_storage", lambda: storage)
    return storage


def test_prints_only_the_raw_key(fake_storage):
    result = runner.invoke(studio_app, ["create-api-key", "--name", "remote-laptop"])
    assert result.exit_code == 0, result.output
    assert result.output.strip() == "sk-unsloth-testkey123"
    assert fake_storage.created == [("unsloth", "remote-laptop")]


def test_default_name(fake_storage):
    result = runner.invoke(studio_app, ["create-api-key"])
    assert result.exit_code == 0, result.output
    assert fake_storage.created == [("unsloth", "remote")]


def test_command_is_hidden():
    command = next(
        c for c in studio_app.registered_commands if c.name == "create-api-key"
    )
    assert command.hidden is True
