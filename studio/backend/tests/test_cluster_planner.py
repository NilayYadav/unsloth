# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from core.cluster import planner


def _node(node_id, usable, kind = "ethernet", rtt = 0.4, mbps = None):
    link = {"kind": kind, "rtt_ms": rtt}
    if mbps:
        link["throughput_mbps"] = mbps
    return {"id": node_id, "name": node_id, "usable_mib": usable, "link": link}


def test_family_comes_from_the_rpc_device_names():
    assert planner.family_from_devices(["MTL0"]) == "metal"
    assert planner.family_from_devices(["CUDA0", "CUDA1"]) == "cuda"
    assert planner.family_from_devices(["ROCm0"]) == "rocm"
    assert planner.family_from_devices(["Vulkan0"]) == "vulkan"
    assert planner.family_from_devices(["CPU"]) == "cpu"
    assert planner.family_from_devices(["CUDA0", "Vulkan0"]) is None
    assert planner.family_from_backend_label("macos-metal-arm64") == "metal"
    assert planner.family_from_backend_label("llama-b1-bin-linux-x64-cuda12-newer.tar.gz") == "cuda"


def test_only_the_same_gpu_family_can_cluster():
    assert planner.compatibility({"family": "cuda"}, {"family": "cuda"})[0]
    assert planner.compatibility({"family": "metal"}, {"family": "cuda"})[1] == "family_mismatch"
    assert planner.compatibility({"family": "cpu"}, {"family": "cpu"})[1] == "unsupported_family"
    assert planner.compatibility({"family": None}, {"family": "cuda"})[1] == "family_unknown"


def test_rpc_protocol_rule_matches_the_llama_cpp_client_check():
    local = {"family": "cuda", "rpc_proto": [6, 1, 0]}
    assert planner.compatibility(local, {"family": "cuda", "rpc_proto": [6, 0, 3]})[0]
    assert planner.compatibility(local, {"family": "cuda", "rpc_proto": [6, 2, 0]})[1] == "rpc_version"
    assert planner.compatibility(local, {"family": "cuda", "rpc_proto": [5, 0, 0]})[1] == "rpc_version"


def test_a_different_build_warns_without_blocking():
    ok, reason, warning = planner.compatibility(
        {"family": "metal", "build": "a"}, {"family": "metal", "build": "b"}
    )
    assert ok and reason is None and warning == "build_mismatch"


def test_nothing_is_attached_when_this_computer_holds_the_model():
    assert planner.choose_nodes([_node("a", 50000)], 0) == []
    assert planner.choose_nodes([_node("a", 50000)], -100) == []


def test_one_big_node_beats_two_hops_even_when_the_other_link_is_faster():
    thunderbolt = _node("tb", 6000, kind = "thunderbolt", rtt = 0.1)
    ethernet = _node("eth", 60000)
    assert [n["id"] for n in planner.choose_nodes([thunderbolt, ethernet], 20000)] == ["eth"]


def test_the_better_link_wins_between_nodes_that_each_suffice():
    wifi = _node("wifi", 60000, kind = "wifi", rtt = 4.0)
    ethernet = _node("eth", 30000)
    assert [n["id"] for n in planner.choose_nodes([wifi, ethernet], 20000)] == ["eth"]


def test_more_nodes_only_when_needed_and_everything_when_still_short():
    a, b = _node("a", 20000), _node("b", 20000)
    slow = _node("slow", 20000, kind = "wifi", rtt = 5.0)
    assert {n["id"] for n in planner.choose_nodes([a, b, slow], 30000)} == {"a", "b"}
    assert len(planner.choose_nodes([a, b, slow], 500000)) == 3


def test_rpc_order_puts_the_best_link_next_to_the_local_gpu():
    wifi = _node("wifi", 10000, kind = "wifi", rtt = 4.0)
    ethernet = _node("eth", 10000)
    thunderbolt = _node("tb", 10000, kind = "thunderbolt", rtt = 0.1)
    assert [n["id"] for n in planner.rpc_order([thunderbolt, wifi, ethernet])] == ["wifi", "eth", "tb"]


def test_measured_throughput_overrides_the_nominal_link_speed():
    throttled = _node("e", 10000, mbps = 90)
    fast_wifi = _node("w", 10000, kind = "wifi", rtt = 0.4, mbps = 900)
    assert planner.node_score(fast_wifi) > planner.node_score(throttled)


def test_shortfall_keeps_local_headroom():
    need = 30 * 1024**3
    assert planner.deficit_mib(need, [40000]) < 0
    assert planner.deficit_mib(need, [24000]) > 0


def test_usable_memory_reserves_the_fit_margin_per_device():
    reserve = planner.FIT_MARGIN_MIB + planner.COMPUTE_RESERVE_MIB
    assert planner.usable_mib(24000, 1) == 24000 - reserve
    assert planner.usable_mib(48000, 2) == 48000 - 2 * reserve
    assert planner.usable_mib(1000, 1) == 0
    assert planner.usable_mib(None) == 0


def test_best_address_prefers_wired_and_ignores_unreachable():
    addresses = [
        {"address": "10.0.0.5", "kind": "wifi", "rtt_ms": 3.0},
        {"address": "10.0.1.5", "kind": "ethernet", "rtt_ms": 0.3},
        {"address": "169.254.1.5", "kind": "thunderbolt", "rtt_ms": None},
    ]
    assert planner.best_address(addresses)["address"] == "10.0.1.5"
    assert planner.best_address([{"address": "x", "kind": "ethernet", "rtt_ms": None}]) is None
