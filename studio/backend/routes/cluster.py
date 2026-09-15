# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import time
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from auth import policy
from auth.authentication import authenticated_via_api_key, get_current_subject
from core.cluster import store
from core.cluster.head import ClusterError, get_cluster_head
from core.cluster.worker import ClusterServiceError, get_sharing_service


async def _require_owner(current_subject: str = Depends(get_current_subject)) -> None:
    await policy.require_owner()


def _require_ui_session(via_api_key: bool = Depends(authenticated_via_api_key)) -> None:
    if via_api_key:
        raise HTTPException(status_code = 403, detail = "Cluster settings require a signed-in session.")


router = APIRouter(dependencies = [Depends(_require_owner), Depends(_require_ui_session)])


class ModePayload(BaseModel):
    mode: Literal["off", "auto"]


class ShareSettingsPayload(BaseModel):
    direct: Optional[bool] = None
    cache_cap_gib: Optional[float] = Field(default = None, gt = 0, le = 100000)
    reset_cache_cap: bool = False
    control_port: Optional[int] = Field(default = None, ge = 1024, le = 65535)
    rpc_port: Optional[int] = Field(default = None, ge = 1024, le = 65535)


class AddNodePayload(BaseModel):
    address: str = Field(min_length = 1, max_length = 255)
    code: str = Field(min_length = 4, max_length = 32)


class NodePatchPayload(BaseModel):
    enabled: bool


def _state() -> dict:
    return {"now": time.time(), "head": get_cluster_head().status(), "share": get_sharing_service().status()}


def _fail(exc) -> None:
    raise HTTPException(status_code = exc.status, detail = exc.code) from exc


@router.get("")
def get_cluster() -> dict:
    return _state()


@router.put("/mode")
def update_mode(payload: ModePayload) -> dict:
    store.set_mode(payload.mode)
    return _state()


@router.post("/share/start")
def start_sharing() -> dict:
    get_sharing_service().start_async()
    return _state()


@router.post("/share/stop")
def stop_sharing() -> dict:
    get_sharing_service().stop()
    return _state()


@router.put("/share/settings")
def update_share_settings(payload: ShareSettingsPayload) -> dict:
    changes = payload.model_dump(exclude_none = True, exclude = {"reset_cache_cap"})
    if payload.reset_cache_cap:
        changes["cache_cap_gib"] = None
    try:
        get_sharing_service().update_settings(**changes)
    except ClusterServiceError as exc:
        _fail(exc)
    return _state()


@router.post("/share/pairing-code")
def regenerate_pairing_code() -> dict:
    get_sharing_service().regenerate_pairing()
    return _state()


@router.post("/share/cache/clear")
def clear_share_cache() -> dict:
    try:
        get_sharing_service().clear_cache()
    except ClusterServiceError as exc:
        _fail(exc)
    return _state()


@router.delete("/share/heads/{head_id}")
def revoke_head(head_id: str) -> dict:
    get_sharing_service().revoke_head(head_id)
    return _state()


@router.post("/nodes")
def add_node(payload: AddNodePayload) -> dict:
    try:
        get_cluster_head().pair(payload.address, payload.code)
    except ClusterError as exc:
        _fail(exc)
    return _state()


@router.patch("/nodes/{node_id}")
def update_node(node_id: str, payload: NodePatchPayload) -> dict:
    try:
        get_cluster_head().set_enabled(node_id, payload.enabled)
    except ClusterError as exc:
        _fail(exc)
    return _state()


@router.delete("/nodes/{node_id}")
def remove_node(node_id: str) -> dict:
    try:
        get_cluster_head().remove(node_id)
    except ClusterError as exc:
        _fail(exc)
    return _state()


@router.post("/nodes/{node_id}/test")
def test_node(node_id: str) -> dict:
    try:
        get_cluster_head().test(node_id)
    except ClusterError as exc:
        _fail(exc)
    return _state()
