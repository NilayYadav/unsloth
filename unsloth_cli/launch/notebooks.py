# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Resolve, build URLs for, and fetch official Unsloth notebooks.

The model -> notebook mapping is read straight from the model-defaults
YAML headers under ``studio/backend/assets/configs/model_defaults`` (each
model-specific file carries ``# Based on <Name>.ipynb`` on line 2 and a
``# Also applies to: <id>, ...`` alias line). We scan those headers
directly rather than importing the studio backend's ``load_model_defaults``
(which pulls heavy deps and drops comment lines via ``yaml.safe_load``).

The allowlist + blob->raw fetch logic mirrors
``scripts/notebook_to_python.py`` (``github_blob_to_raw`` /
``download_notebook``); kept self-contained here because ``scripts/`` is
not an importable package.
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional


# Hosts we are willing to fetch raw notebook JSON from. Anything else is
# rejected before `urlopen` so a typoed / hostile URL cannot pull code
# from arbitrary infrastructure. Mirrors scripts/notebook_to_python.py.
_ALLOWED_NOTEBOOK_HOSTS = {
    "raw.githubusercontent.com",
    "gist.githubusercontent.com",
}

# Raw base for the official notebook library (the README Colab links use
# the `blob` form of this same path).
_NOTEBOOKS_RAW_BASE = "https://raw.githubusercontent.com/unslothai/notebooks/main/nb"

# GitHub contents API for the same `nb/` folder -- used to list every
# notebook the library ships (the picker shown when no model/notebook is
# given). Unauthenticated; the folder is well under the 1000-entry page cap.
_NOTEBOOKS_API = "https://api.github.com/repos/unslothai/notebooks/contents/nb"

# Platform-specific variants of the base notebooks. Hidden by default in the
# picker (a local CLI user wants the base notebook, not the AMD/Kaggle/HF
# duplicates -- they're ~65% of the listing); `include_all` keeps them.
_PLATFORM_PREFIXES = ("AMD-", "Kaggle-", "HuggingFace Course-")

# Sensible starter when no model/notebook is given: a real, lightweight,
# widely-applicable notebook (confirmed present as a `# Based on` value).
DEFAULT_NOTEBOOK = "Llama3.2_(1B_and_3B)-Conversational.ipynb"

# repo / site-packages root that holds both `unsloth_cli/` and `studio/`.
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_MODEL_DEFAULTS_DIR = (
    _PACKAGE_ROOT / "studio" / "backend" / "assets" / "configs" / "model_defaults"
)


def _build_model_notebook_map() -> Dict[str, str]:
    """Scan model-defaults YAML headers -> {model_id.lower(): notebook}.

    Reads only the first few lines of each file (skipping ``default.yaml``):
    line 1 ``# Model defaults for <id>``, line 2 ``# Based on <Name>.ipynb``,
    line 3 ``# Also applies to: <id>, <id>, ...``. Files without a clean
    ``.ipynb`` ``# Based on`` value (e.g. the gemma-4 placeholders, or the
    ``.py`` embedding/MoE notes) contribute nothing -> caller falls back to
    ``DEFAULT_NOTEBOOK``.
    """
    mapping: Dict[str, str] = {}
    if not _MODEL_DEFAULTS_DIR.is_dir():
        return mapping

    for path in _MODEL_DEFAULTS_DIR.rglob("*.yaml"):
        if path.name == "default.yaml":
            continue
        try:
            with open(path, "r", encoding = "utf-8") as f:
                header = [next(f, "").rstrip("\n") for _ in range(3)]
        except OSError:
            continue

        ids: list[str] = []
        notebook: Optional[str] = None
        for line in header:
            if line.startswith("# Model defaults for "):
                ids.append(line[len("# Model defaults for "):].strip())
            elif line.startswith("# Based on "):
                # Headers are freeform human comments: some point at `.py`
                # notebooks ("Qwen3_MoE.py", "...py embedding notebook") or
                # add trailing prose ("Foo.ipynb (same defaults...)"). Take
                # the first token and only accept a clean `.ipynb` name;
                # anything else -> notebook stays None -> default fallback.
                candidate = line[len("# Based on "):].strip()
                token = candidate.split()[0] if candidate else ""
                if token.endswith(".ipynb"):
                    notebook = token
            elif line.startswith("# Also applies to: "):
                ids.extend(
                    part.strip()
                    for part in line[len("# Also applies to: "):].split(",")
                    if part.strip()
                )

        if notebook:
            for model_id in ids:
                mapping[model_id.lower()] = notebook

    return mapping


def resolve_notebook_for_model(model_id: str) -> Optional[str]:
    """Return the ``<Name>.ipynb`` for *model_id*, or None if unknown.

    Matches the model's primary id or any of its aliases
    (case-insensitive). None means the caller should fall back to
    ``DEFAULT_NOTEBOOK``.
    """
    return _build_model_notebook_map().get(model_id.lower())


def list_notebooks(include_all: bool = False) -> List[str]:
    """Return ``.ipynb`` names from the official notebook library, sorted.

    Hits the GitHub contents API for ``unslothai/notebooks`` ``nb/``. By
    default platform-specific variants (``AMD-`` / ``Kaggle-`` /
    ``HuggingFace Course-``) are dropped so the picker shows the base
    notebooks; pass ``include_all=True`` to keep them. Raises on a
    network/parse failure so the caller can fall back to ``DEFAULT_NOTEBOOK``.
    """
    request = urllib.request.Request(
        _NOTEBOOKS_API, headers = {"Accept": "application/vnd.github+json"}
    )
    with urllib.request.urlopen(request, timeout = 60) as response:  # noqa: S310 (fixed GitHub API URL)
        data = json.load(response)
    if not isinstance(data, list):
        raise ValueError(f"Unexpected notebook listing response: {data!r}")
    names = sorted(
        entry["name"]
        for entry in data
        if entry.get("type") == "file" and entry.get("name", "").endswith(".ipynb")
    )
    if include_all:
        return names
    return [n for n in names if not n.startswith(_PLATFORM_PREFIXES)]


def notebook_matches(query: str, name: str) -> bool:
    """True if every token of *query* appears in *name*, ignoring separators.

    Both sides are lowercased and split on non-alphanumerics, so a loose
    query like ``"gemma 3 270"`` matches ``"Gemma3_(270M).ipynb"``.
    """
    haystack = re.sub(r"[^a-z0-9]+", " ", name.lower())
    tokens = re.sub(r"[^a-z0-9]+", " ", query.lower()).split()
    return all(token in haystack for token in tokens)


def _github_blob_to_raw(url: str) -> str:
    """Convert a GitHub blob URL to its raw form (else return unchanged).

    Mirrors scripts/notebook_to_python.py:github_blob_to_raw. Compares the
    parsed host exactly (not as a substring) so a URL like
    ``https://attacker.example.com/github.com/blob/...`` is not rewritten.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.netloc != "github.com" or "/blob/" not in parsed.path:
        return url
    new_path = parsed.path.replace("/blob/", "/", 1)
    return urllib.parse.urlunparse(
        parsed._replace(netloc = "raw.githubusercontent.com", path = new_path)
    )


def notebook_raw_url(name_or_url: str) -> str:
    """Build a raw notebook URL from a bare name or pass through a URL.

    A bare name (e.g. ``Gemma3_(270M)``) is resolved against the official
    notebook library. A full URL is normalised (blob->raw) and its host is
    validated against the allowlist; anything else raises ``ValueError``.
    """
    parsed = urllib.parse.urlparse(name_or_url)
    if parsed.scheme in ("http", "https"):
        raw_url = _github_blob_to_raw(name_or_url)
        host = urllib.parse.urlparse(raw_url).hostname
        if host not in _ALLOWED_NOTEBOOK_HOSTS:
            raise ValueError(
                f"Refused notebook fetch from {host!r}: not in allowlist "
                f"{sorted(_ALLOWED_NOTEBOOK_HOSTS)}"
            )
        return raw_url

    name = name_or_url if name_or_url.endswith(".ipynb") else f"{name_or_url}.ipynb"
    return f"{_NOTEBOOKS_RAW_BASE}/{urllib.parse.quote(name)}"


def notebook_colab_url(name_or_url: str) -> str:
    """Build a Google Colab URL that opens the notebook from GitHub.

    Accepts the same inputs as :func:`notebook_raw_url` (a bare notebook name
    or a raw/blob GitHub URL) and maps it to
    ``https://colab.research.google.com/github/<owner>/<repo>/blob/<branch>/<path>``
    -- the same form as the "Open in Colab" badges in the README.
    """
    raw = notebook_raw_url(name_or_url)
    parts = urllib.parse.urlparse(raw).path.lstrip("/").split("/")
    if len(parts) < 4:
        raise ValueError(f"Can't build a Colab URL from {name_or_url!r}")
    owner, repo, branch, *rest = parts
    path = "/".join(rest)
    return f"https://colab.research.google.com/github/{owner}/{repo}/blob/{branch}/{path}"


def fetch_notebook(url: str, dest_dir: Path) -> Path:
    """Download *url* into *dest_dir*, returning the local path.

    Skips the download if the file already exists. Re-validates the host
    against the allowlist before fetching. Mirrors
    scripts/notebook_to_python.py:download_notebook.
    """
    parsed = urllib.parse.urlparse(url)
    filename = Path(urllib.parse.unquote(parsed.path)).name
    dest = dest_dir / filename
    if dest.exists():
        return dest

    if parsed.hostname not in _ALLOWED_NOTEBOOK_HOSTS:
        raise ValueError(
            f"Refused notebook fetch from {parsed.hostname!r}: not in allowlist "
            f"{sorted(_ALLOWED_NOTEBOOK_HOSTS)}"
        )

    dest_dir.mkdir(parents = True, exist_ok = True)
    with urllib.request.urlopen(url, timeout = 60) as response:  # noqa: S310 (host allowlisted)
        content = response.read()
    dest.write_bytes(content)
    return dest
