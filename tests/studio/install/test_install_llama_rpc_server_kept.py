# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import importlib.util
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = PACKAGE_ROOT / "studio" / "install_llama_prebuilt.py"
SPEC = importlib.util.spec_from_file_location("studio_install_llama_prebuilt_rpc", MODULE_PATH)
INSTALL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = INSTALL
SPEC.loader.exec_module(INSTALL)


@pytest.mark.parametrize("kind", ["linux-cpu", "linux-cuda", "linux-rocm", "linux-vulkan", "macos-arm64"])
def test_posix_bundles_keep_the_rpc_server(kind):
    assert "ggml-rpc-server" in INSTALL.runtime_patterns_for_install_kind(kind)


@pytest.mark.parametrize("kind", ["windows-cpu", "windows-cuda", "windows-vulkan", "windows-rocm"])
def test_windows_bundles_keep_the_rpc_server(kind):
    assert "ggml-rpc-server.exe" in INSTALL.runtime_patterns_for_install_kind(kind)


def test_the_unix_install_marks_the_rpc_server_executable():
    import inspect

    source = inspect.getsource(INSTALL.install_from_archives)
    assert 'build_bin / "ggml-rpc-server"' in source and "os.chmod(rpc_server, 0o755)" in source


def test_source_builds_enable_rpc_and_build_the_server():
    setup_sh = (PACKAGE_ROOT / "studio" / "setup.sh").read_text()
    setup_ps1 = (PACKAGE_ROOT / "studio" / "setup.ps1").read_text()
    assert "-DGGML_RPC=ON" in setup_sh and "--target ggml-rpc-server" in setup_sh
    assert "-DGGML_RPC=ON" in setup_ps1 and "--target ggml-rpc-server" in setup_ps1
