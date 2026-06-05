# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`unsloth launch notebook` -- fetch the right official Unsloth notebook
for a model and open it in JupyterLab.

The whole feature is resolve -> fetch -> launch: pick the official notebook
that matches the model (the official notebook for a model already loads that
model), download it, and hand the path to `jupyter lab` so the browser opens
directly on that notebook. No cell mutation, no auto-install.
"""

from __future__ import annotations

import shutil
import subprocess
import urllib.parse
import webbrowser
from pathlib import Path
from typing import Optional

import typer

from unsloth_cli.launch import (
    DEFAULT_NOTEBOOK,
    fetch_notebook,
    list_notebooks,
    notebook_colab_url,
    notebook_matches,
    notebook_raw_url,
    resolve_notebook_for_model,
)


launch_app = typer.Typer(
    help = "Launch a Jupyter notebook with Unsloth ready.",
    no_args_is_help = True,
)


def _fail(msg: str, code: int = 1):
    typer.echo(msg, err = True)
    raise typer.Exit(code)


def _pick_notebook(include_all: bool = False) -> str:
    """Search-first interactive picker over the official notebook library.

    Prompts for a search before listing (there are hundreds), shows the
    matches numbered, and returns the chosen name. At the pick prompt, a
    number selects; anything else is treated as a new search. Degrades to
    ``DEFAULT_NOTEBOOK`` if the library can't be listed (e.g. offline).
    """
    typer.echo("Fetching available Unsloth notebooks...")
    try:
        names = list_notebooks(include_all = include_all)
    except Exception as e:  # network / parse / rate-limit
        typer.echo(f"  couldn't list notebooks ({e}); using the default.", err = True)
        return DEFAULT_NOTEBOOK
    if not names:
        return DEFAULT_NOTEBOOK

    query = typer.prompt(
        f"Search {len(names)} notebooks by model or task",
        default = "", show_default = False,
    ).strip()

    while True:
        shown = [n for n in names if notebook_matches(query, n)] if query else names
        if not shown:
            query = typer.prompt(
                f"  no matches for {query!r}. Search again", default = "", show_default = False,
            ).strip()
            continue

        for i, name in enumerate(shown, start = 1):
            typer.echo(f"  {i:3d}. {name}")

        # Pick a number, or type anything else to search again. Out-of-range
        # numbers re-prompt without reprinting the list.
        while True:
            raw = typer.prompt("Pick a number (or type to search again)", default = "1").strip()
            if raw.isdigit():
                idx = int(raw)
                if 1 <= idx <= len(shown):
                    return shown[idx - 1]
                typer.echo(f"  enter a number between 1 and {len(shown)}.")
                continue
            query = raw
            break


@launch_app.command()
def notebook(
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-hf",
        "--hf-repo",
        help = "Model path or HF repo; selects the matching official notebook.",
    ),
    notebook_arg: Optional[str] = typer.Option(
        None,
        "--notebook",
        help = "Notebook name (e.g. 'Gemma3_(270M)') or a full raw/blob URL.",
    ),
    port: int = typer.Option(8888, "--port", "-p", help = "JupyterLab port."),
    dir: Path = typer.Option(
        Path("./unsloth-notebooks"),
        "--dir",
        help = "Directory to download into and launch JupyterLab from.",
    ),
    all_notebooks: bool = typer.Option(
        False, "--all",
        help = "In the picker, also show platform variants (AMD/Kaggle/HF Course).",
    ),
    colab: bool = typer.Option(
        False, "--colab",
        help = "Open the notebook in Google Colab instead of launching JupyterLab.",
    ),
):
    """Fetch a notebook and open it in JupyterLab (or in Google Colab with --colab)."""
    # 1. Resolve the notebook name/URL.
    if notebook_arg is not None:
        target = notebook_arg
    elif model is not None:
        target = resolve_notebook_for_model(model) or DEFAULT_NOTEBOOK
    else:
        target = _pick_notebook(include_all = all_notebooks)

    # 2. --colab: open in Google Colab (the standard) -- no fetch, no JupyterLab.
    if colab:
        try:
            colab_url = notebook_colab_url(target)
        except ValueError as e:
            _fail(str(e))
        typer.echo(f"Opening in Google Colab:\n  {colab_url}")
        webbrowser.open(colab_url)
        return

    try:
        url = notebook_raw_url(target)
    except ValueError as e:
        _fail(str(e))

    # 2. Fetch into a per-notebook subdir (skips if already present).
    #    Each notebook gets its own directory so JupyterLab -- which roots
    #    its file browser at the launch cwd -- shows only this notebook, not
    #    every one ever downloaded.
    filename = Path(urllib.parse.unquote(urllib.parse.urlparse(url).path)).name
    nb_dir = dir.expanduser() / Path(filename).stem
    cached = (nb_dir / filename).exists()
    path = fetch_notebook(url, nb_dir)
    typer.echo(f"  {'Using cached' if cached else 'Downloaded'} {path}")

    # 3. Preflight Jupyter -- do NOT auto-install.
    if shutil.which("jupyter") is None:
        _fail("Jupyter not found. Install it with:  pip install jupyterlab")

    # 4. Launch + auto-open. Passing the filename with cwd=dest_dir makes
    #    JupyterLab open the browser directly on that notebook.
    base_url = f"http://127.0.0.1:{port}/lab"
    typer.echo("")
    typer.echo("=" * 56)
    typer.echo(f"  Opening {path.name}")
    typer.echo(f"  JupyterLab at {base_url}")
    typer.echo("  Press Ctrl-C to stop.")
    typer.echo("=" * 56)
    typer.echo("")

    try:
        subprocess.run(
            ["jupyter", "lab", path.name, "--port", str(port)],
            cwd = str(nb_dir),
        )
    except KeyboardInterrupt:
        # Let JupyterLab shut itself down; exit cleanly without a traceback.
        pass
