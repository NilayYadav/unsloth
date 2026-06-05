# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from unsloth_cli import app
from unsloth_cli.commands import launch as launch_cmd
from unsloth_cli.launch import notebooks


# --- resolve_notebook_for_model -------------------------------------------

def test_resolve_primary_id():
    assert (
        notebooks.resolve_notebook_for_model("unsloth/gemma-3-270m-it")
        == "Gemma3_(270M).ipynb"
    )


def test_resolve_alias_id():
    # `# Also applies to:` alias should map to the same notebook.
    assert (
        notebooks.resolve_notebook_for_model("google/gemma-3-270m-it")
        == "Gemma3_(270M).ipynb"
    )


def test_resolve_is_case_insensitive():
    assert (
        notebooks.resolve_notebook_for_model("UNSLOTH/Gemma-3-270M-IT")
        == "Gemma3_(270M).ipynb"
    )


def test_resolve_unknown_model_returns_none():
    assert notebooks.resolve_notebook_for_model("does/not-exist") is None


def test_resolve_model_without_notebook_returns_none():
    # gemma-4 YAMLs have no `# Based on` header -> no notebook.
    assert notebooks.resolve_notebook_for_model("unsloth/gemma-4-E4B") is None


def test_resolve_py_notebook_falls_back():
    # Embedding/MoE headers point at `.py` notebooks (e.g. "Qwen3_Embedding_(0_6B).py
    # embedding notebook"), not `.ipynb` -> None so the caller uses DEFAULT_NOTEBOOK
    # instead of building a 404 URL.
    assert notebooks.resolve_notebook_for_model("unsloth/Qwen3-Embedding-0.6B") is None


def test_resolve_strips_trailing_prose():
    # "# Based on Gemma2_(9B)-Alpaca.ipynb (same defaults for larger models)" must
    # resolve to the clean notebook name, not the whole commented line.
    assert (
        notebooks.resolve_notebook_for_model("unsloth/gemma-2-27b-bnb-4bit")
        == "Gemma2_(9B)-Alpaca.ipynb"
    )


# --- notebook_raw_url ------------------------------------------------------

def test_raw_url_bare_name():
    url = notebooks.notebook_raw_url("Gemma3_(270M)")
    assert url == (
        "https://raw.githubusercontent.com/unslothai/notebooks/main/nb/"
        "Gemma3_%28270M%29.ipynb"
    )


def test_raw_url_adds_ipynb_suffix_only_once():
    with_suffix = notebooks.notebook_raw_url("Foo.ipynb")
    assert with_suffix.endswith("/Foo.ipynb")


def test_raw_url_blob_rewrite():
    url = notebooks.notebook_raw_url(
        "https://github.com/unslothai/notebooks/blob/main/nb/Foo.ipynb"
    )
    assert url == (
        "https://raw.githubusercontent.com/unslothai/notebooks/main/nb/Foo.ipynb"
    )


def test_raw_url_passes_allowlisted_raw_url():
    raw = "https://raw.githubusercontent.com/unslothai/notebooks/main/nb/Foo.ipynb"
    assert notebooks.notebook_raw_url(raw) == raw


def test_raw_url_rejects_non_allowlisted_host():
    with pytest.raises(ValueError):
        notebooks.notebook_raw_url("https://evil.example.com/x.ipynb")


# --- notebook_colab_url ----------------------------------------------------

def test_colab_url_bare_name():
    assert notebooks.notebook_colab_url("Gemma3_(270M)") == (
        "https://colab.research.google.com/github/unslothai/notebooks/"
        "blob/main/nb/Gemma3_%28270M%29.ipynb"
    )


def test_colab_url_from_blob_url():
    url = notebooks.notebook_colab_url(
        "https://github.com/unslothai/notebooks/blob/main/nb/Foo.ipynb"
    )
    assert url == (
        "https://colab.research.google.com/github/unslothai/notebooks/"
        "blob/main/nb/Foo.ipynb"
    )


# --- fetch_notebook --------------------------------------------------------

def test_fetch_downloads_and_writes(monkeypatch, tmp_path):
    calls = []

    def fake_urlopen(url, timeout = 60):
        calls.append(url)
        return io.BytesIO(b'{"cells": []}')

    monkeypatch.setattr(notebooks.urllib.request, "urlopen", fake_urlopen)
    url = "https://raw.githubusercontent.com/unslothai/notebooks/main/nb/Foo.ipynb"
    dest = notebooks.fetch_notebook(url, tmp_path)
    assert dest == tmp_path / "Foo.ipynb"
    assert dest.read_bytes() == b'{"cells": []}'
    assert calls == [url]


def test_fetch_skips_when_file_exists(monkeypatch, tmp_path):
    (tmp_path / "Foo.ipynb").write_bytes(b"cached")

    def boom(*a, **k):
        raise AssertionError("urlopen should not be called when cached")

    monkeypatch.setattr(notebooks.urllib.request, "urlopen", boom)
    url = "https://raw.githubusercontent.com/unslothai/notebooks/main/nb/Foo.ipynb"
    dest = notebooks.fetch_notebook(url, tmp_path)
    assert dest.read_bytes() == b"cached"


def test_fetch_rejects_non_allowlisted_host(monkeypatch, tmp_path):
    with pytest.raises(ValueError):
        notebooks.fetch_notebook("https://evil.example.com/x.ipynb", tmp_path)


# --- list_notebooks --------------------------------------------------------

def _stub_listing(monkeypatch, names):
    payload = json.dumps(
        [{"type": "file", "name": n} for n in names]
        + [{"type": "file", "name": "README.md"}, {"type": "dir", "name": "subdir"}]
    ).encode()
    monkeypatch.setattr(
        notebooks.urllib.request, "urlopen",
        lambda *a, **k: io.BytesIO(payload),
    )


def test_list_notebooks_parses_and_sorts(monkeypatch):
    _stub_listing(monkeypatch, ["B.ipynb", "A.ipynb"])  # non-ipynb/dirs dropped
    assert notebooks.list_notebooks() == ["A.ipynb", "B.ipynb"]


def test_list_notebooks_hides_platform_variants_by_default(monkeypatch):
    _stub_listing(monkeypatch, [
        "Gemma3_(270M).ipynb",
        "AMD-Gemma3_(270M).ipynb",
        "Kaggle-Gemma3_(270M).ipynb",
        "HuggingFace Course-Gemma3_(270M).ipynb",
    ])
    assert notebooks.list_notebooks() == ["Gemma3_(270M).ipynb"]


def test_list_notebooks_include_all_keeps_variants(monkeypatch):
    _stub_listing(monkeypatch, ["Gemma3_(270M).ipynb", "AMD-Gemma3_(270M).ipynb"])
    assert notebooks.list_notebooks(include_all = True) == [
        "AMD-Gemma3_(270M).ipynb", "Gemma3_(270M).ipynb",
    ]


# --- notebook_matches (separator-insensitive search) ----------------------

def test_matches_multiword_across_separators():
    # "gemma 3 270" must match "Gemma3_(270M).ipynb" despite the formatting.
    assert notebooks.notebook_matches("gemma 3 270", "Gemma3_(270M).ipynb")


def test_matches_requires_all_tokens():
    assert not notebooks.notebook_matches("gemma vision", "Gemma3_(270M).ipynb")
    assert notebooks.notebook_matches("gemma vision", "Gemma3_(4B)-Vision.ipynb")


# --- command wiring --------------------------------------------------------

def _stub_fetch(monkeypatch, tmp_path):
    """Avoid network: write a fake notebook and return its path."""

    def fake_fetch(url, dest_dir):
        dest_dir.mkdir(parents = True, exist_ok = True)
        name = notebooks.Path(
            notebooks.urllib.parse.unquote(
                notebooks.urllib.parse.urlparse(url).path
            )
        ).name
        path = dest_dir / name
        if not path.exists():
            path.write_bytes(b"{}")
        return path

    monkeypatch.setattr(launch_cmd, "fetch_notebook", fake_fetch)


def test_command_errors_when_jupyter_missing(monkeypatch, tmp_path):
    _stub_fetch(monkeypatch, tmp_path)
    monkeypatch.setattr(launch_cmd.shutil, "which", lambda name: None)
    # Explicit --notebook bypasses the picker so this isolates the preflight.
    result = CliRunner().invoke(
        app, ["launch", "notebook", "--notebook", "Foo", "--dir", str(tmp_path)]
    )
    assert result.exit_code == 1
    assert "pip install jupyterlab" in (result.output + (result.stderr or ""))


def test_command_launches_jupyter_with_resolved_path(monkeypatch, tmp_path):
    _stub_fetch(monkeypatch, tmp_path)
    monkeypatch.setattr(launch_cmd.shutil, "which", lambda name: "/usr/bin/jupyter")

    recorded = {}

    def fake_run(args, cwd = None):
        recorded["args"] = args
        recorded["cwd"] = cwd

    monkeypatch.setattr(launch_cmd.subprocess, "run", fake_run)
    result = CliRunner().invoke(
        app,
        ["launch", "notebook", "--model", "unsloth/gemma-3-270m-it",
         "--dir", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    assert recorded["args"] == [
        "jupyter", "lab", "Gemma3_(270M).ipynb", "--port", "8888",
    ]
    # Each notebook launches from its own subdir so JupyterLab's file browser
    # shows only this notebook, not every one downloaded before.
    assert recorded["cwd"] == str(tmp_path / "Gemma3_(270M)")


def test_command_picker_lists_and_launches(monkeypatch, tmp_path):
    # No --model / --notebook -> picker over the whole library.
    _stub_fetch(monkeypatch, tmp_path)
    monkeypatch.setattr(launch_cmd.shutil, "which", lambda name: "/usr/bin/jupyter")
    monkeypatch.setattr(
        launch_cmd, "list_notebooks",
        lambda include_all = False: ["Apple.ipynb", "Gemma3_(270M).ipynb", "Zebra.ipynb"],
    )
    recorded = {}
    monkeypatch.setattr(
        launch_cmd.subprocess, "run",
        lambda args, cwd = None: recorded.update(args = args),
    )
    # Search "gemma" -> single match, then pick #1.
    result = CliRunner().invoke(
        app, ["launch", "notebook", "--dir", str(tmp_path)],
        input = "gemma\n1\n",
    )
    assert result.exit_code == 0, result.output
    assert recorded["args"][2] == "Gemma3_(270M).ipynb"


def test_command_colab_opens_browser_and_skips_jupyter(monkeypatch, tmp_path):
    opened = {}
    monkeypatch.setattr(launch_cmd.webbrowser, "open", lambda url: opened.update(url = url))

    def no_jupyter(*a, **k):
        raise AssertionError("--colab must not launch JupyterLab")

    monkeypatch.setattr(launch_cmd.subprocess, "run", no_jupyter)
    result = CliRunner().invoke(
        app,
        ["launch", "notebook", "--colab", "--notebook", "Gemma3_(270M)",
         "--dir", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    assert opened["url"] == (
        "https://colab.research.google.com/github/unslothai/notebooks/"
        "blob/main/nb/Gemma3_%28270M%29.ipynb"
    )


def test_command_picker_falls_back_to_default_on_list_error(monkeypatch, tmp_path):
    _stub_fetch(monkeypatch, tmp_path)
    monkeypatch.setattr(launch_cmd.shutil, "which", lambda name: "/usr/bin/jupyter")

    def boom(include_all = False):
        raise RuntimeError("offline")

    monkeypatch.setattr(launch_cmd, "list_notebooks", boom)
    recorded = {}
    monkeypatch.setattr(
        launch_cmd.subprocess, "run",
        lambda args, cwd = None: recorded.update(args = args),
    )
    result = CliRunner().invoke(
        app, ["launch", "notebook", "--dir", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert recorded["args"][2] == notebooks.DEFAULT_NOTEBOOK
