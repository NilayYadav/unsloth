# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for `unsloth update`: command registration, uv/pip command building
(including prereleases), --dry-run, the editable-install guard, and the
success / failure reporting paths."""

from __future__ import annotations

import sys
from pathlib import Path

from typer.testing import CliRunner

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _update():
    from unsloth_cli.commands import update as _update_mod
    return _update_mod


def _app():
    import typer

    app = typer.Typer()
    app.command()(_update().update)
    return app


def _neutralize_win_locks(monkeypatch):
    """Stop the lazily-imported Windows exe-lock helpers from touching a real
    unsloth.exe when these tests run on Windows CI."""
    import unsloth_cli.commands.studio as studio_mod

    for name in (
        "_release_self_exe_lock_windows",
        "_restore_self_exe_lock_windows",
        "_cleanup_self_exe_lock_windows",
    ):
        monkeypatch.setattr(studio_mod, name, lambda: None)


# ── registration ─────────────────────────────────────────────────────


def test_update_registered_on_app():
    from unsloth_cli import app

    names = {(ci.name or ci.callback.__name__) for ci in app.registered_commands}
    assert "update" in names


# ── command building ─────────────────────────────────────────────────


def test_upgrade_cmd_prefers_uv(monkeypatch):
    m = _update()
    monkeypatch.setattr(m.shutil, "which", lambda name: "/usr/bin/uv")
    cmd = m._upgrade_cmd(["unsloth", "unsloth_zoo"], pre = False)
    assert cmd == [
        "uv", "pip", "install", "--python", sys.executable, "--upgrade",
        "unsloth", "unsloth_zoo",
    ]


def test_upgrade_cmd_falls_back_to_pip(monkeypatch):
    m = _update()
    monkeypatch.setattr(m.shutil, "which", lambda name: None)
    cmd = m._upgrade_cmd(["unsloth", "unsloth_zoo"], pre = False)
    assert cmd == [
        sys.executable, "-m", "pip", "install", "--upgrade",
        "unsloth", "unsloth_zoo",
    ]


def test_pre_uses_prerelease_for_uv(monkeypatch):
    m = _update()
    monkeypatch.setattr(m.shutil, "which", lambda name: "/usr/bin/uv")
    cmd = m._upgrade_cmd(["unsloth"], pre = True)
    assert cmd[cmd.index("--prerelease") + 1] == "allow"
    assert "--pre" not in cmd


def test_pre_uses_pre_flag_for_pip(monkeypatch):
    m = _update()
    monkeypatch.setattr(m.shutil, "which", lambda name: None)
    cmd = m._upgrade_cmd(["unsloth"], pre = True)
    assert "--pre" in cmd
    assert "--prerelease" not in cmd


def test_query_versions_reads_installed_and_missing():
    # Runs the real subprocess: typer is installed (the suite imports it), the
    # other name is not. Guards the egg-info-shadowing fix.
    result = _update()._query_versions(["typer", "definitely-not-a-real-pkg-xyz"])
    assert result["typer"] is not None
    assert result["definitely-not-a-real-pkg-xyz"] is None


# ── dry run ──────────────────────────────────────────────────────────


def test_dry_run_prints_cmd_without_installing(monkeypatch):
    m = _update()
    monkeypatch.setattr(m.shutil, "which", lambda name: "/usr/bin/uv")

    def boom(*a, **k):
        raise AssertionError("subprocess.run must not run on --dry-run")

    monkeypatch.setattr(m.subprocess, "run", boom)
    result = CliRunner().invoke(_app(), ["--dry-run"])
    assert result.exit_code == 0, result.output
    assert "uv pip install" in result.output
    assert "unsloth" in result.output and "unsloth_zoo" in result.output


# ── run reporting ────────────────────────────────────────────────────


def test_update_runs_and_reports_transition(monkeypatch):
    m = _update()
    _neutralize_win_locks(monkeypatch)
    monkeypatch.setattr(m.shutil, "which", lambda name: None)  # pip path

    phase = {"v": "before"}
    versions = {
        "before": {"unsloth": "2026.6.3", "unsloth_zoo": "2026.6.3"},
        "after": {"unsloth": "2026.7.0", "unsloth_zoo": "2026.7.0"},
    }
    monkeypatch.setattr(m, "_query_versions", lambda packages: dict(versions[phase["v"]]))

    captured = {}

    class _Result:
        returncode = 0

    def fake_run(cmd, *a, **k):
        captured["cmd"] = cmd
        phase["v"] = "after"
        return _Result()

    monkeypatch.setattr(m.subprocess, "run", fake_run)
    result = CliRunner().invoke(_app(), [])
    assert result.exit_code == 0, result.output
    assert captured["cmd"] == [
        sys.executable, "-m", "pip", "install", "--upgrade",
        "unsloth", "unsloth_zoo",
    ]
    out = result.output
    assert "2026.6.3" in out and "2026.7.0" in out


def test_update_failure_propagates_exit_code(monkeypatch):
    m = _update()
    _neutralize_win_locks(monkeypatch)
    monkeypatch.setattr(m.shutil, "which", lambda name: None)
    monkeypatch.setattr(m, "_query_versions", lambda packages: {p: "2026.6.3" for p in packages})

    class _Result:
        returncode = 7
        stdout = ""
        stderr = "boom"

    monkeypatch.setattr(m.subprocess, "run", lambda *a, **k: _Result())
    result = CliRunner().invoke(_app(), [])
    assert result.exit_code == 7, result.output
