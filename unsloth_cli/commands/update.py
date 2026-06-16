# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, version

import typer


# unsloth and unsloth_zoo are released in lockstep; the documented upgrade is
# always `pip install -U unsloth unsloth_zoo`, so update both together.
PACKAGES = ["unsloth", "unsloth_zoo"]


def _installed_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _query_versions(packages: list[str]) -> dict[str, str | None]:
    """Read installed versions from the target interpreter's site-packages.

    Done in a subprocess from a neutral working directory (and without an
    inherited PYTHONPATH) so a stale `*.egg-info` in the project root -- which
    lands on sys.path when invoked via cli.py -- can't shadow the real
    installed distribution and report a pre-update version.
    """
    code = (
        "import json, sys, importlib.metadata as m\n"
        "out = {}\n"
        "for name in sys.argv[1:]:\n"
        "    try:\n"
        "        out[name] = m.version(name)\n"
        "    except m.PackageNotFoundError:\n"
        "        out[name] = None\n"
        "print(json.dumps(out))\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code, *packages],
            capture_output = True,
            text = True,
            cwd = tempfile.gettempdir(),
            env = env,
        )
        return json.loads(proc.stdout)
    except (OSError, ValueError):
        # Fall back to an in-process read if the helper couldn't run.
        importlib.invalidate_caches()
        return {pkg: _installed_version(pkg) for pkg in packages}


def _upgrade_cmd(packages: list[str], pre: bool) -> list[str]:
    """Build the upgrade command, preferring uv when on PATH (as the rest of
    the repo does). uv and pip spell prereleases differently."""
    if shutil.which("uv"):
        cmd = ["uv", "pip", "install", "--python", sys.executable, "--upgrade"]
        if pre:
            cmd += ["--prerelease", "allow"]
    else:
        cmd = [sys.executable, "-m", "pip", "install", "--upgrade"]
        if pre:
            cmd.append("--pre")
    return cmd + packages


def update(
    pre: bool = typer.Option(
        False, "--pre", help = "Include pre-release (nightly) versions."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help = "Print the install command without running it."
    ),
):
    """Update Unsloth to the latest version."""
    cmd = _upgrade_cmd(PACKAGES, pre = pre)
    if dry_run:
        typer.echo(" ".join(cmd))
        raise typer.Exit()

    from rich.console import Console
    from rich.table import Table

    console = Console()
    before = _query_versions(PACKAGES)

    # On Windows pip can't overwrite the running unsloth.exe launcher, so rename
    # it aside first and restore it if the install fails before pip writes a
    # replacement. These no-op off Windows; shared with `studio update`.
    from unsloth_cli.commands.studio import (
        _cleanup_self_exe_lock_windows,
        _release_self_exe_lock_windows,
        _restore_self_exe_lock_windows,
    )

    _release_self_exe_lock_windows()
    # Capture the installer's chatter and show a spinner instead; surface the
    # raw output only if it fails.
    with console.status(f"[bold cyan]Updating {', '.join(PACKAGES)}…", spinner = "dots"):
        try:
            result = subprocess.run(cmd, capture_output = True, text = True)
        except BaseException:
            _restore_self_exe_lock_windows()
            raise
    if result.returncode != 0:
        _restore_self_exe_lock_windows()
        console.print((result.stdout or "") + (result.stderr or ""))
        console.print("[bold red]Update failed.[/]")
        raise typer.Exit(code = result.returncode)
    _cleanup_self_exe_lock_windows()

    after = _query_versions(PACKAGES)
    table = Table(title = "Unsloth update", title_style = "bold", title_justify = "left")
    table.add_column("Package", style = "cyan", no_wrap = True)
    table.add_column("Version")
    bumped = False
    for pkg in PACKAGES:
        old, new = before[pkg], after[pkg]
        if new is None:
            continue
        if old == new:
            table.add_row(pkg, f"[dim]{new} (up to date)[/]")
        else:
            bumped = True
            table.add_row(pkg, f"[dim]{old or '—'}[/] [green]→ {new}[/]")
    console.print(table)
    console.print(
        "[bold green]✓ Updated to the latest version.[/]" if bumped
        else "[green]✓ Already on the latest version.[/]"
    )
