# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import copy
import threading
import uuid
from typing import Any, Callable, Optional

MODE_KEY = "cluster_mode"
NODES_KEY = "cluster_nodes"
SHARE_KEY = "cluster_share"
HEADS_KEY = "cluster_paired_heads"
IDENTITY_KEY = "cluster_identity"
LOCAL_PROTO_KEY = "cluster_local_rpc_proto"

MODES = ("off", "auto")
DEFAULT_MODE = "auto"
DEFAULT_CONTROL_PORT = 50051
DEFAULT_RPC_PORT = 50052
DEFAULT_SHARE = {
    "auto_start": False,
    "direct": False,
    "cache_cap_gib": None,
    "control_port": DEFAULT_CONTROL_PORT,
    "rpc_port": DEFAULT_RPC_PORT,
}

_lock = threading.RLock()
_memory: Optional[dict] = None


def use_memory_backend(initial: Optional[dict] = None) -> dict:
    global _memory
    _memory = dict(initial or {})
    return _memory


def use_database_backend() -> None:
    global _memory
    _memory = None


def _get(key: str, fallback: Any = None) -> Any:
    if _memory is not None:
        return copy.deepcopy(_memory.get(key, fallback))
    from storage.studio_db import get_app_setting

    return get_app_setting(key, fallback)


def _set(values: dict) -> None:
    if _memory is not None:
        _memory.update(copy.deepcopy(values))
        return
    from storage.studio_db import upsert_app_settings

    upsert_app_settings(values, read_back = False)


def identity() -> dict:
    with _lock:
        current = _get(IDENTITY_KEY)
        if isinstance(current, dict) and isinstance(current.get("node_id"), str):
            return current
        current = {"node_id": uuid.uuid4().hex}
        _set({IDENTITY_KEY: current})
        return current


def mode() -> str:
    value = _get(MODE_KEY)
    return value if value in MODES else DEFAULT_MODE


def set_mode(value: str) -> str:
    if value not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    with _lock:
        _set({MODE_KEY: value})
    return value


def nodes() -> list[dict]:
    value = _get(NODES_KEY, [])
    return [n for n in value if isinstance(n, dict) and n.get("id")] if isinstance(value, list) else []


def save_nodes(value: list[dict]) -> None:
    with _lock:
        _set({NODES_KEY: value})


def update_node(node_id: str, change: Callable[[dict], Optional[dict]]) -> Optional[dict]:
    with _lock:
        current = nodes()
        updated = None
        for index, node in enumerate(current):
            if node["id"] == node_id:
                result = change(node)
                current[index] = result if result is not None else node
                updated = current[index]
                break
        if updated is not None:
            _set({NODES_KEY: current})
        return updated


def upsert_node(node: dict) -> dict:
    with _lock:
        current = [n for n in nodes() if n["id"] != node["id"]]
        current.append(node)
        _set({NODES_KEY: current})
        return node


def remove_node(node_id: str) -> Optional[dict]:
    with _lock:
        current = nodes()
        kept = [n for n in current if n["id"] != node_id]
        if len(kept) == len(current):
            return None
        _set({NODES_KEY: kept})
        return next(n for n in current if n["id"] == node_id)


def share_settings() -> dict:
    value = _get(SHARE_KEY, {})
    merged = dict(DEFAULT_SHARE)
    if isinstance(value, dict):
        merged.update({k: v for k, v in value.items() if k in DEFAULT_SHARE})
    return merged


def update_share_settings(**changes: Any) -> dict:
    with _lock:
        merged = share_settings()
        merged.update({k: v for k, v in changes.items() if k in DEFAULT_SHARE})
        _set({SHARE_KEY: merged})
        return merged


def paired_heads() -> list[dict]:
    value = _get(HEADS_KEY, [])
    return [h for h in value if isinstance(h, dict) and h.get("id")] if isinstance(value, list) else []


def save_paired_heads(value: list[dict]) -> None:
    with _lock:
        _set({HEADS_KEY: value})


def local_rpc_proto() -> Optional[list[int]]:
    value = _get(LOCAL_PROTO_KEY)
    if isinstance(value, list) and len(value) == 3 and all(isinstance(v, int) for v in value):
        return value
    return None


def set_local_rpc_proto(value: Optional[list[int]]) -> None:
    with _lock:
        _set({LOCAL_PROTO_KEY: value})
