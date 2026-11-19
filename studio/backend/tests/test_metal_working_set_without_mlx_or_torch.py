# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A Chat only Mac has neither MLX nor torch, so the Metal working set is read from Metal itself."""

from __future__ import annotations

import ctypes
import platform
import sys
import types

import pytest

from utils.hardware import hardware as hw

GIB = 1024**3
WORKING_SET = int(10.67 * GIB)
_REAL_READER = hw._metal_framework_working_set_bytes


class _Fn:
    def __init__(self, impl):
        self.impl, self.restype, self.argtypes = impl, None, None

    def __call__(self, *args):
        return self.impl(*args)


def _fake_metal(monkeypatch, sent):
    metal = types.SimpleNamespace(MTLCreateSystemDefaultDevice = _Fn(lambda: 0x1234))

    def msg_send(device, selector):
        sent.append((device, selector))
        return WORKING_SET if selector == b"recommendedMaxWorkingSetSize" else None

    objc = types.SimpleNamespace(
        sel_registerName = _Fn(lambda name: name), objc_msgSend = _Fn(msg_send)
    )
    libs = {
        "/System/Library/Frameworks/Metal.framework/Metal": metal,
        "/usr/lib/libobjc.A.dylib": objc,
    }
    monkeypatch.setattr(ctypes, "CDLL", lambda path, *a, **k: libs[path])


@pytest.fixture
def chat_only_mac(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    monkeypatch.setitem(sys.modules, "mlx", None)
    monkeypatch.setitem(sys.modules, "mlx.core", None)
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setattr(hw, "_metal_framework_working_set_bytes", _REAL_READER)


def test_the_working_set_comes_from_metal_without_mlx_or_torch(monkeypatch, chat_only_mac):
    sent = []
    _fake_metal(monkeypatch, sent)
    assert hw._metal_recommended_working_set_bytes() == WORKING_SET
    assert sent == [(0x1234, b"recommendedMaxWorkingSetSize"), (0x1234, b"release")]


@pytest.mark.usefixtures("real_apple_metal_budget")
def test_a_chat_only_mac_budgets_against_the_working_set_not_total_ram(monkeypatch, chat_only_mac):
    from core.inference.llama_cpp import _APPLE_UNIFIED_MEMORY_FRACTION, LlamaCppBackend

    _fake_metal(monkeypatch, [])
    fake = types.ModuleType("psutil")
    vm = types.SimpleNamespace(total = 16 * GIB, available = int(14.5 * GIB))
    fake.virtual_memory = lambda: vm
    monkeypatch.setitem(sys.modules, "psutil", fake)
    assert LlamaCppBackend._apple_metal_memory_budget_bytes() == int(
        WORKING_SET * _APPLE_UNIFIED_MEMORY_FRACTION
    )


def test_off_apple_silicon_metal_is_not_asked(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(hw, "_ROSETTA", False)
    monkeypatch.setattr(ctypes, "CDLL", lambda *a, **k: pytest.fail("Metal was loaded"))
    assert _REAL_READER() == 0
