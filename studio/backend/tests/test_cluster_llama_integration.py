# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest

from core.cluster import head as head_module
from core.cluster import planner
from core.inference.llama_cpp import LlamaCppBackend

GIB = 1024**3
BASE = dict(
    use_fit = True,
    rpc_supported = lambda: True,
    gpu_memory_mode = "auto",
    tensor_parallel = False,
    extra_args = None,
    model_size = 40 * GIB,
    effective_ctx = 8192,
    cache_type_kv = None,
    n_parallel = 1,
    local_gpus = [(0, 24000)],
)


class FakeHead:
    def __init__(self, enabled = True):
        self.enabled = enabled
        self.calls = []

    def has_enabled_nodes(self):
        return self.enabled

    def plan_for_load(self, deficit, alive):
        self.calls.append((deficit, alive()))
        return "attached"


@pytest.fixture
def backend(monkeypatch):
    instance = LlamaCppBackend.__new__(LlamaCppBackend)
    instance._context_length = 32768
    instance._process = None
    monkeypatch.setattr(instance, "_estimate_kv_cache_bytes", lambda ctx, cache_type_kv = None, **kw: 2 * GIB)
    monkeypatch.delenv("LLAMA_ARG_RPC", raising = False)
    return instance


@pytest.fixture
def fake_head(monkeypatch):
    fake = FakeHead()
    monkeypatch.setattr(head_module, "get_cluster_head", lambda: fake)
    return fake


def test_a_model_that_does_not_fit_asks_for_exactly_the_shortfall(backend, fake_head):
    assert backend._plan_cluster_for_load(**BASE) == "attached"
    assert fake_head.calls == [(planner.deficit_mib(42 * GIB, [24000]), False)]


@pytest.mark.parametrize(
    "change",
    [
        {"use_fit": False},
        {"gpu_memory_mode": "manual"},
        {"tensor_parallel": True},
        {"extra_args": ["--rpc", "10.0.0.2:50052"]},
        {"extra_args": ["--rpc=10.0.0.2:50052"]},
        {"rpc_supported": lambda: False},
        {"model_size": 4 * GIB},
        {"model_size": None},
        {"local_gpus": []},
    ],
)
def test_no_cluster_when_the_load_does_not_need_or_allow_one(backend, fake_head, change):
    assert backend._plan_cluster_for_load(**{**BASE, **change}) is None
    assert fake_head.calls == []


def test_an_inherited_llama_arg_rpc_owns_placement(backend, fake_head, monkeypatch):
    monkeypatch.setenv("LLAMA_ARG_RPC", "10.0.0.2:50052")
    assert backend._plan_cluster_for_load(**BASE) is None


def test_no_paired_computers_skips_the_capability_probe(backend, monkeypatch):
    monkeypatch.setattr(head_module, "get_cluster_head", lambda: FakeHead(enabled = False))

    def probe():
        raise AssertionError("probed llama-server with no cluster configured")

    assert backend._plan_cluster_for_load(**{**BASE, "rpc_supported": probe}) is None


def test_the_capability_catalog_reports_rpc_support():
    caps = LlamaCppBackend.probe_server_capabilities("/nonexistent/llama-server")
    assert caps.get("supports_rpc") is False
