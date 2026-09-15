# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from itertools import combinations
from typing import Iterable, Mapping, Optional, Sequence

from core.cluster.netinfo import KIND_RANK, nominal_mbps

GPU_FAMILIES = ("metal", "cuda", "rocm", "vulkan")
FAMILY_LABELS = {
    "metal": "Apple Silicon",
    "cuda": "NVIDIA",
    "rocm": "AMD",
    "vulkan": "Vulkan",
    "cpu": "CPU",
}

# llama.cpp's --fit keeps 1 GiB free per device (common.h fit_params_target)
FIT_MARGIN_MIB = 1024
COMPUTE_RESERVE_MIB = 768
LOCAL_USABLE_FRACTION = 0.92
COVERAGE_HEADROOM = 1.10
MAX_NODES_CONSIDERED = 8

_DEVICE_PREFIXES = (
    ("MTL", "metal"),
    ("METAL", "metal"),
    ("CUDA", "cuda"),
    ("ROCM", "rocm"),
    ("HIP", "rocm"),
    ("VULKAN", "vulkan"),
    ("CPU", "cpu"),
)


def family_from_devices(names: Iterable[str]) -> Optional[str]:
    found = set()
    for name in names:
        upper = str(name).upper()
        for prefix, family in _DEVICE_PREFIXES:
            if upper.startswith(prefix):
                found.add(family)
                break
    gpu = found - {"cpu"}
    if len(gpu) == 1:
        return gpu.pop()
    if not gpu and found == {"cpu"}:
        return "cpu"
    return None


def family_from_backend_label(label: Optional[str]) -> Optional[str]:
    lowered = (label or "").casefold()
    if not lowered:
        return None
    if "metal" in lowered or "mlx" in lowered:
        return "metal"
    if "cuda" in lowered:
        return "cuda"
    if "rocm" in lowered or "hip" in lowered:
        return "rocm"
    if "vulkan" in lowered:
        return "vulkan"
    if "cpu" in lowered:
        return "cpu"
    return None


def compatibility(local: Mapping, remote: Mapping) -> tuple[bool, Optional[str], Optional[str]]:
    local_family, remote_family = local.get("family"), remote.get("family")
    if not local_family or not remote_family:
        return False, "family_unknown", None
    if local_family not in GPU_FAMILIES or remote_family not in GPU_FAMILIES:
        return False, "unsupported_family", None
    if local_family != remote_family:
        return False, "family_mismatch", None
    local_proto, remote_proto = local.get("rpc_proto"), remote.get("rpc_proto")
    if local_proto and remote_proto:
        # llama.cpp's client refuses another major, or a server minor newer than its own
        if remote_proto[0] != local_proto[0] or remote_proto[1] > local_proto[1]:
            return False, "rpc_version", None
    local_build, remote_build = local.get("build"), remote.get("build")
    if local_build and remote_build and local_build != remote_build:
        return True, None, "build_mismatch"
    return True, None, None


def link_mbps(link: Optional[Mapping]) -> float:
    link = link or {}
    measured = link.get("throughput_mbps")
    nominal = nominal_mbps(link.get("kind"), link.get("speed_mbps"))
    if measured:
        return float(measured)
    return nominal


def node_score(node: Mapping) -> float:
    link = node.get("link") or {}
    rtt = link.get("rtt_ms")
    rtt = 1.0 if rtt is None else max(0.05, float(rtt))
    # every token crosses each link twice, so latency weighs as much as throughput
    return link_mbps(link) / (1.0 + rtt / 1.5)


def usable_mib(free_mib: Optional[int], device_count: int = 1) -> int:
    if not free_mib:
        return 0
    reserve = max(1, device_count) * (FIT_MARGIN_MIB + COMPUTE_RESERVE_MIB)
    return max(0, int(free_mib) - reserve)


def local_usable_mib(free_mib_per_gpu: Sequence[int]) -> int:
    return int(sum(max(0, int(f)) for f in free_mib_per_gpu) * LOCAL_USABLE_FRACTION)


def deficit_mib(need_bytes: int, local_free_mib: Sequence[int]) -> int:
    need_mib = int(need_bytes / (1024 * 1024)) + COMPUTE_RESERVE_MIB
    return need_mib - local_usable_mib(local_free_mib)


def choose_nodes(candidates: Sequence[Mapping], deficit: int) -> list[Mapping]:
    pool = [c for c in candidates if int(c.get("usable_mib") or 0) > 0]
    pool.sort(key = lambda c: (-node_score(c), -int(c.get("usable_mib") or 0)))
    pool = pool[:MAX_NODES_CONSIDERED]
    if deficit <= 0 or not pool:
        return []
    target = deficit * COVERAGE_HEADROOM + 256
    # fewest nodes first: each extra machine is two more network crossings per token
    for size in range(1, len(pool) + 1):
        best = None
        best_key = None
        for combo in combinations(pool, size):
            capacity = sum(int(c["usable_mib"]) for c in combo)
            if capacity < target:
                continue
            scores = [node_score(c) for c in combo]
            key = (min(scores), sum(scores), capacity)
            if best_key is None or key > best_key:
                best, best_key = combo, key
        if best is not None:
            return list(best)
    return pool


def rpc_order(nodes: Sequence[Mapping]) -> list[Mapping]:
    # llama.cpp puts RPC devices first and --fit fills back to front, so the last entry
    # is filled right after the local GPU and should be the best link
    return sorted(nodes, key = lambda n: (node_score(n), int(n.get("usable_mib") or 0)))


def best_address(addresses: Sequence[Mapping]) -> Optional[Mapping]:
    reachable = [a for a in addresses if a.get("rtt_ms") is not None]
    if not reachable:
        return None
    return max(
        reachable,
        key = lambda a: (
            nominal_mbps(a.get("kind"), a.get("speed_mbps")) / (1.0 + float(a["rtt_ms"]) / 1.5),
            KIND_RANK.get(a.get("kind") or "unknown", 0),
        ),
    )
