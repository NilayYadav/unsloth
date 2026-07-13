# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for `unsloth doctor` — no network, no real subprocesses."""

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


def _patch_env(monkeypatch, pkgs = HEALTHY_PKGS, gpu = NVIDIA_GPU, torch_info = None, import_ok = True, import_err = ""):
    if torch_info is None:
        torch_info = {"version": pkgs.get("torch"), "cuda": "12.6", "hip": None, "available": True, "mps": False}
    monkeypatch.setattr(doctor, "_pkg", lambda name: pkgs.get(name))
    monkeypatch.setattr(doctor, "_latest_pypi", lambda name: None)
    monkeypatch.setattr(doctor, "_gpu_info", lambda: dict(gpu))
    monkeypatch.setattr(doctor, "_torch_probe", lambda: torch_info)
    monkeypatch.setattr(doctor, "_import_probe", lambda: (import_ok, import_err))
    monkeypatch.setattr(doctor, "_torch_pin_ok", lambda package, torch_version: True)


def test_healthy_env_exits_zero(monkeypatch):
    _patch_env(monkeypatch)
    result = runner.invoke(app)
    assert result.exit_code == 0, result.output
    assert "No issues found." in result.output
    assert "torch 2.9.0+cu126" in result.output


def test_cuda_newer_than_driver_fails_with_fix(monkeypatch):
    torch_info = {"version": "2.9.0+cu130", "cuda": "13.0", "hip": None, "available": True, "mps": False}
    _patch_env(monkeypatch, torch_info = torch_info)
    result = runner.invoke(app)
    assert result.exit_code == 1
    assert "built for CUDA 13.0 but driver supports ≤ 12.6" in result.output
    assert 'unsloth[cu126-torch290]' in result.output


def test_import_failure_classified(monkeypatch):
    _patch_env(monkeypatch, import_ok = False, import_err = "blah\nLLVM ERROR: Cannot select: intrinsic")
    result = runner.invoke(app)
    assert result.exit_code == 1
    assert "Triton kernel compilation failed" in result.output


def test_report_mode_prints_fenced_block(monkeypatch):
    _patch_env(monkeypatch)
    result = runner.invoke(app, ["--report"])
    assert result.exit_code == 0
    assert "### Unsloth `doctor` report" in result.output
    assert "```text" in result.output
    assert "[ok] torch 2.9.0+cu126" in result.output


def test_missing_unsloth_fails(monkeypatch):
    pkgs = {k: v for k, v in HEALTHY_PKGS.items() if k != "unsloth"}
    _patch_env(monkeypatch, pkgs = pkgs)
    result = runner.invoke(app)
    assert result.exit_code == 1
    assert "pip install unsloth" in result.output


@pytest.mark.parametrize(
    "torch_version, cuda, expected",
    [
        ("2.9.0", "12.4", None),
        ("2.9.0+cu126", "12.6", "cu126-torch290"),
        ("2.5.0", "12.4", "cu124-torch250"),
        ("2.6.0", "12.4", "cu124-torch260"),
        ("2.10.0", "12.8", "cu128-torch2100"),
        ("2.10.0", "12.4", None),
        ("2.1.0", "12.1", None),
        ("2.9.0", "10.2", None),
        ("2.9.0", None, None),
    ],
)
def test_install_extra_mapping(torch_version, cuda, expected):
    assert doctor._install_extra(torch_version, cuda) == expected


@pytest.mark.parametrize(
    "stderr, expected",
    [
        ("LLVM ERROR: Cannot select", "Triton kernel compilation failed — torch/triton/GPU mismatch"),
        ("CUDA Setup failed despite GPU", "bitsandbytes could not find CUDA libraries"),
        ("ImportError: No module named 'unsloth'", "unsloth is not installed in this environment"),
        ("something.so: undefined symbol: xyz", "a package was built against a different torch — reinstall matching wheels"),
    ],
)
def test_classify_import_error(stderr, expected):
    message, _ = doctor._classify_import_error(stderr)
    assert message == expected


def test_classify_unknown_error_returns_tail():
    message, fix = doctor._classify_import_error("line1\nRuntimeError: something odd")
    assert message == "RuntimeError: something odd"
    assert fix is None


def test_best_cuda_for_driver():
    assert doctor._best_cuda_for_driver("12.6") == "12.6"
    assert doctor._best_cuda_for_driver("12.9") == "12.8"
    assert doctor._best_cuda_for_driver("13.1") == "13.0"
    assert doctor._best_cuda_for_driver("11.0") is None
    assert doctor._best_cuda_for_driver("garbage") is None


def test_cuda_newer_than_driver():
    assert doctor._cuda_newer_than_driver("13.0", "12.6") is True
    assert doctor._cuda_newer_than_driver("12.6", "12.6") is False
    assert doctor._cuda_newer_than_driver("12.4", "12.6") is False
    assert doctor._cuda_newer_than_driver("bogus", "12.6") is False
