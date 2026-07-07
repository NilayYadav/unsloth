# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for unsloth_cli.remote.probe parsing — canned outputs, no SSH."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from unsloth_cli.remote.probe import parse_probe_output

S = "===UNSLOTH:"


def _output(
    gpu = "NVIDIA A100-SXM4-80GB, 81920 MiB, 550.54.15",
    cuda = "CUDA Version: 12.4",
    python = "Python 3.10.12",
    disk_kb = 400 * 1024 * 1024,
    arch = "x86_64",
    tools = "rsync=yes\ntmux=yes\nsystemctl=yes\nuv=no\ncurl=yes",
    systemd = "running",
    os_name = "Ubuntu 22.04.4 LTS",
):
    return "\n".join([
        f"{S}GPU===", gpu,
        f"{S}CUDA===", cuda,
        f"{S}PYTHON===", python,
        f"{S}DISK===", f"/dev/root 500000000 90000000 {disk_kb} 20% /",
        f"{S}ARCH===", arch,
        f"{S}TOOLS===", tools,
        f"{S}SYSTEMD===", systemd,
        f"{S}OS===", os_name,
    ])


class TestHealthyBox:
    def test_all_green(self):
        result = parse_probe_output(_output())
        assert result.fatal_failures == []
        caps = result.capabilities
        assert caps.gpu_names == ["NVIDIA A100-SXM4-80GB"]
        assert caps.vram_gb == [80.0]
        assert caps.driver == "550.54.15"
        assert caps.cuda == "12.4"
        assert caps.python == "3.10.12"
        assert caps.arch == "x86_64"
        assert caps.disk_free_gb == 400.0
        assert caps.has_rsync and caps.has_tmux and caps.has_systemd_user

    def test_multi_gpu(self):
        gpu = (
            "NVIDIA A100-SXM4-80GB, 81920 MiB, 550.54.15\n"
            "NVIDIA A100-SXM4-80GB, 81920 MiB, 550.54.15"
        )
        caps = parse_probe_output(_output(gpu = gpu)).capabilities
        assert len(caps.gpu_names) == 2
        assert caps.vram_gb == [80.0, 80.0]


class TestFailures:
    def test_no_gpu_is_fatal_with_driver_hint(self):
        result = parse_probe_output(_output(gpu = "NO_GPU", cuda = ""))
        fatal = result.fatal_failures
        assert len(fatal) == 1
        assert fatal[0].name == "GPU"
        assert "nvidia-driver-550" in fatal[0].hint

    def test_old_driver_is_fatal_with_upgrade_hint(self):
        result = parse_probe_output(
            _output(gpu = "NVIDIA T4, 15360 MiB, 470.82.01", cuda = "CUDA Version: 11.4")
        )
        fatal = result.fatal_failures
        assert [c.name for c in fatal] == ["NVIDIA driver"]
        assert "525" in fatal[0].hint
        assert "sudo apt install nvidia-driver-550" in fatal[0].hint

    def test_low_disk_is_warning_not_fatal(self):
        result = parse_probe_output(_output(disk_kb = 10 * 1024 * 1024))
        assert result.fatal_failures == []
        warning = [c for c in result.warnings if c.name.startswith("Disk")]
        assert len(warning) == 1
        assert "free up space" in warning[0].hint.lower()

    def test_missing_python_is_warning_only(self):
        result = parse_probe_output(_output(python = "NO_PYTHON"))
        assert result.fatal_failures == []
        py = [c for c in result.warnings if c.name == "python3"]
        assert py and "uv" in py[0].hint

    def test_no_supervision_is_fatal(self):
        result = parse_probe_output(_output(
            tools = "rsync=yes\ntmux=no\nsystemctl=no\nuv=no\ncurl=yes",
            systemd = "none",
        ))
        assert [c.name for c in result.fatal_failures] == ["Agent supervision"]
        assert "tmux" in result.fatal_failures[0].hint

    def test_tmux_fallback_when_no_user_systemd(self):
        result = parse_probe_output(_output(systemd = "none"))
        assert result.fatal_failures == []
        supervision = [c for c in result.checks if c.name == "Agent supervision"]
        assert supervision[0].value == "tmux"

    def test_missing_rsync_warns(self):
        result = parse_probe_output(_output(
            tools = "rsync=no\ntmux=yes\nsystemctl=yes\nuv=no\ncurl=yes",
        ))
        assert result.capabilities.has_rsync is False
        assert any(c.name == "rsync" for c in result.warnings)
