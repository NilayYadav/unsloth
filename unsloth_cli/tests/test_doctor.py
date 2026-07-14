# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest
import typer
from typer.testing import CliRunner

import unsloth_cli.commands.doctor as doctor

runner = CliRunner()

app = typer.Typer()
app.command()(doctor.doctor)

HEALTHY_PKGS = {
    "unsloth": "2026.7.2",
    "unsloth_zoo": "2026.7.1",
    "torch": "2.9.0+cu126",
    "triton": "3.4.0",
    "xformers": "0.0.32",
    "bitsandbytes": "0.49.2",
    "transformers": "4.56.2",
    "trl": "0.19.0",
    "peft": "0.15.0",
}

NVIDIA_GPU = {
    "kind": "nvidia",
    "gpus": [{"name": "NVIDIA GeForce RTX 4090", "memory": "24564 MiB"}],
    "driver": "560.35",
    "driver_cuda": "12.6",
}


def _patch_env(monkeypatch, pkgs = HEALTHY_PKGS, gpu = NVIDIA_GPU, torch_info = None, mlx_ready = False, import_ok = True, import_err = "", latest = None):
    if torch_info is None:
        torch_info = {"version": pkgs.get("torch"), "cuda": "12.6", "hip": None, "available": True, "mps": False, "xpu": False}
    monkeypatch.setattr(doctor, "_pkg", lambda name: pkgs.get(name))
    monkeypatch.setattr(doctor, "_latest_pypi", lambda name: latest)
    monkeypatch.setattr(doctor, "_gpu_info", lambda: dict(gpu))
    monkeypatch.setattr(doctor, "_torch_probe", lambda: torch_info)
    monkeypatch.setattr(doctor, "_mlx_probe", lambda: mlx_ready)
    monkeypatch.setattr(doctor, "_import_probe", lambda: (import_ok, import_err))


def test_healthy_env_exits_zero(monkeypatch):
    _patch_env(monkeypatch)
    result = runner.invoke(app)
    assert result.exit_code == 0, result.output
    assert "No issues found." in result.output
    assert "torch 2.9.0+cu126 (cu12.6)" in result.output
    assert "transformers 4.56.2" in result.output
    assert "`import unsloth` OK" in result.output


def test_cuda_newer_than_driver_fails(monkeypatch):
    torch_info = {"version": "2.9.0+cu130", "cuda": "13.0", "hip": None, "available": True, "mps": False, "xpu": False}
    _patch_env(monkeypatch, torch_info = torch_info)
    result = runner.invoke(app)
    assert result.exit_code == 1
    assert "built for CUDA 13.0 but driver supports ≤ 12.6" in result.output


def test_nvidia_gpu_without_torch_cuda_fails(monkeypatch):
    torch_info = {"version": "2.9.0+cu126", "cuda": "12.6", "hip": None, "available": False, "mps": False, "xpu": False}
    _patch_env(monkeypatch, torch_info = torch_info)
    result = runner.invoke(app)
    assert result.exit_code == 1
    assert "PyTorch cannot see it" in result.output


def test_report_mode_prints_fenced_block(monkeypatch):
    _patch_env(monkeypatch)
    result = runner.invoke(app, ["--report"])
    assert result.exit_code == 0
    assert "### Unsloth `doctor` report" in result.output
    assert "```text" in result.output
    assert "[ok] torch 2.9.0+cu126 (cu12.6)" in result.output


def test_import_failure_shows_error_tail(monkeypatch):
    _patch_env(monkeypatch, import_ok = False, import_err = "RuntimeError: Found no NVIDIA driver on your system.")
    result = runner.invoke(app)
    assert result.exit_code == 1
    assert "`import unsloth` failed: RuntimeError: Found no NVIDIA driver" in result.output


def test_update_available_warns(monkeypatch):
    _patch_env(monkeypatch, latest = "2026.8.0")
    result = runner.invoke(app)
    assert result.exit_code == 0
    assert "2026.8.0 is available" in result.output
    assert "pip install --upgrade unsloth unsloth_zoo" in result.output


def test_missing_unsloth_fails_and_skips_import(monkeypatch):
    pkgs = {k: v for k, v in HEALTHY_PKGS.items() if k != "unsloth"}
    _patch_env(monkeypatch, pkgs = pkgs)
    monkeypatch.setattr(doctor, "_import_probe", lambda: pytest.fail("unexpected import check"))
    result = runner.invoke(app)
    assert result.exit_code == 1
    assert "pip install unsloth" in result.output


def test_missing_gpu_package_warns(monkeypatch):
    pkgs = {k: v for k, v in HEALTHY_PKGS.items() if k != "bitsandbytes"}
    _patch_env(monkeypatch, pkgs = pkgs)
    result = runner.invoke(app)
    assert result.exit_code == 0
    assert "bitsandbytes not installed" in result.output


def test_intel_xpu_is_a_supported_backend(monkeypatch):
    torch_info = {"version": "2.10.0+xpu", "cuda": None, "hip": None, "available": False, "mps": False, "xpu": True}
    _patch_env(monkeypatch, gpu = {"kind": None, "gpus": [], "driver": None, "driver_cuda": None}, torch_info = torch_info)
    result = runner.invoke(app)
    assert result.exit_code == 0
    assert "Intel XPU available in PyTorch" in result.output
    assert "torch 2.10.0+xpu (xpu)" in result.output


def test_apple_requires_working_mlx(monkeypatch):
    apple_gpu = {"kind": "apple", "gpus": [{"name": "Apple M4 Max", "memory": "128 GB unified"}], "driver": None, "driver_cuda": None}
    _patch_env(monkeypatch, gpu = apple_gpu, mlx_ready = False, torch_info = {"version": None, "cuda": None, "hip": None, "available": False, "mps": True, "xpu": False})
    monkeypatch.setattr(doctor, "_torch_probe", lambda: None)
    result = runner.invoke(app)
    assert result.exit_code == 1
    assert "MLX backend is unavailable" in result.output

    _patch_env(monkeypatch, gpu = apple_gpu, mlx_ready = True)
    monkeypatch.setattr(doctor, "_torch_probe", lambda: None)
    monkeypatch.setattr(doctor, "_pkg", lambda name: {"unsloth": "2026.7.2", "unsloth_zoo": "2026.7.1", "mlx": "0.30.0"}.get(name))
    result = runner.invoke(app)
    assert result.exit_code == 0
    assert "MLX available" in result.output
    assert "torch not installed (not required with MLX)" in result.output


def test_visible_devices_env_var_is_shown(monkeypatch):
    _patch_env(monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    result = runner.invoke(app)
    assert "CUDA_VISIBLE_DEVICES=0,1" in result.output


def test_cuda_newer_than_driver():
    assert doctor._cuda_newer_than_driver("13.0", "12.6") is True
    assert doctor._cuda_newer_than_driver("12.6", "12.6") is False
    assert doctor._cuda_newer_than_driver("12.4", "12.6") is False
    assert doctor._cuda_newer_than_driver("bogus", "12.6") is False
