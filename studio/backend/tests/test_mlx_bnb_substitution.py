# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MLX loads the full-precision base of an unsloth bnb-4bit repo, never the bnb weights.

The swap happens inside ``unsloth_zoo.mlx.loader``, which only prints it, so a curated
list naming a bnb repo costs an Apple Silicon user a download that is then discarded for
a much larger one, and the load-time stall watchdog measures a repo nothing is writing.
"""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import core.inference.defaults as defaults_mod  # noqa: E402
import utils.hardware.hardware as hw  # noqa: E402
from core.inference.mlx_bnb import mlx_bnb_base_repo, mlx_bnb_substitutions  # noqa: E402


def test_unsloth_bnb_repos_resolve_to_their_base():
    assert (
        mlx_bnb_base_repo("unsloth/Qwen2-VL-2B-Instruct-bnb-4bit")
        == "unsloth/Qwen2-VL-2B-Instruct"
    )
    assert mlx_bnb_base_repo("unsloth/gemma-3-4b-it-unsloth-bnb-4bit") == "unsloth/gemma-3-4b-it"


def test_repos_mlx_loads_as_given_have_no_base():
    assert mlx_bnb_base_repo("unsloth/Qwen3-4B-Instruct-2507") is None
    assert mlx_bnb_base_repo("unsloth/Llama-3.2-1B-Instruct-GGUF") is None
    assert mlx_bnb_base_repo("someone-else/model-bnb-4bit") is None
    assert mlx_bnb_base_repo(None) is None


def test_a_local_directory_is_never_remapped(monkeypatch, tmp_path):
    """Mirrors the loader: an on-disk path that looks like a repo id is loaded as given."""
    (tmp_path / "unsloth" / "model-bnb-4bit").mkdir(parents = True)
    monkeypatch.chdir(tmp_path)

    assert mlx_bnb_base_repo("unsloth/model-bnb-4bit") is None


def test_substitutions_cover_a_loras_base_and_skip_repos_already_watched():
    swaps = mlx_bnb_substitutions(
        ["me/my-lora", "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"]
    )
    assert swaps == [
        ("unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit", "unsloth/Meta-Llama-3.1-8B-Instruct")
    ]

    already_watched = ["unsloth/gemma-3-4b-it-bnb-4bit", "unsloth/gemma-3-4b-it"]
    assert mlx_bnb_substitutions(already_watched) == []


def _standard_host(monkeypatch, device):
    monkeypatch.setattr(hw, "CHAT_ONLY", False)
    monkeypatch.setattr(hw, "get_device", lambda: device)


def test_mlx_defaults_recommend_the_repos_mlx_actually_loads(monkeypatch):
    _standard_host(monkeypatch, hw.DeviceType.MLX)

    models = defaults_mod.get_default_models()

    assert [model for model in models if model.endswith("bnb-4bit")] == []
    assert "unsloth/Qwen2-VL-2B-Instruct" in models
    assert len(models) == len(set(models))


def test_cuda_defaults_keep_the_bnb_repos(monkeypatch):
    _standard_host(monkeypatch, hw.DeviceType.CUDA)

    assert defaults_mod.get_default_models() == defaults_mod.DEFAULT_MODELS_STANDARD


def test_chat_only_hosts_are_untouched(monkeypatch):
    monkeypatch.setattr(hw, "CHAT_ONLY", True)
    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.MLX)

    assert defaults_mod.get_default_models() == defaults_mod.DEFAULT_MODELS_GGUF


def test_worker_watches_the_repo_mlx_downloads():
    source = (_BACKEND / "core" / "inference" / "worker.py").read_text(encoding = "utf-8")
    assert "mlx_bnb_substitutions(watch_repos)" in source
    assert "watch_repos.append(mlx_base)" in source


def test_the_host_rule_only_fires_on_mlx(monkeypatch):
    from core.inference.mlx_bnb import mlx_host_bnb_base_repo

    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.CUDA)
    assert mlx_host_bnb_base_repo("unsloth/Qwen2-VL-2B-Instruct-bnb-4bit") is None

    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.MLX)
    assert (
        mlx_host_bnb_base_repo("unsloth/Qwen2-VL-2B-Instruct-bnb-4bit")
        == "unsloth/Qwen2-VL-2B-Instruct"
    )


def test_diffusion_bnb_repos_are_loaded_as_named(monkeypatch):
    """Diffusion runs on diffusers/MPS, which reads bnb weights; only the MLX loader cannot."""
    from core.inference.mlx_bnb import mlx_host_bnb_base_repo

    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.MLX)
    assert mlx_host_bnb_base_repo("unsloth/Qwen-Image-2512-unsloth-bnb-4bit") is None
    assert mlx_host_bnb_base_repo("unsloth/Z-Image-Turbo-unsloth-bnb-4bit") is None


def test_validate_reports_the_repo_mlx_will_load(monkeypatch):
    from types import SimpleNamespace

    from routes.inference import _mlx_base_for_config

    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.MLX)
    pick = SimpleNamespace(identifier = "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit", base_model = None)
    assert _mlx_base_for_config(pick) == "unsloth/Qwen2-VL-2B-Instruct"

    adapter = SimpleNamespace(
        identifier = "me/my-lora", base_model = "unsloth/gemma-3-4b-it-bnb-4bit"
    )
    assert _mlx_base_for_config(adapter) == "unsloth/gemma-3-4b-it"

    plain = SimpleNamespace(identifier = "unsloth/Qwen3-4B-Instruct-2507", base_model = None)
    assert _mlx_base_for_config(plain) is None
