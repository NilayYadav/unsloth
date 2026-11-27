# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GitHub OAuth App device authorization, scoped to an authenticated Studio account."""

import json
import os
import secrets
import sys
import time

import httpx
from fastapi import APIRouter, Depends, HTTPException

from auth.authentication import (
    authenticated_via_api_key,
    get_current_credential,
    get_current_subject,
)
from routes.provider_credentials import (
    current_credential_write,
    require_ui_session,
    serialize_provider_config,
)
from storage import credential_secrets, providers_db

router = APIRouter(dependencies = [Depends(get_current_subject)])
FLOW_KIND = "github_copilot_oauth_flow"


def _require_provider(provider_id: str, via_api_key: bool):
    require_ui_session(via_api_key)
    row = providers_db.get_own_provider(provider_id)
    if row is None or row["provider_type"] != "github_copilot":
        raise HTTPException(404, "GitHub Copilot connection not found.")
    return row


async def _github_post(path: str, data: dict) -> dict:
    try:
        async with httpx.AsyncClient(timeout = 20, follow_redirects = False) as client:
            response = await client.post(
                "https://github.com/" + path,
                data = data,
                headers = {"Accept": "application/json"},
            )
            response.raise_for_status()
            return response.json()
    except (httpx.HTTPError, ValueError):
        raise HTTPException(502, "Could not contact GitHub. Try signing in again.") from None


@router.post("/{provider_id}/github/start")
@serialize_provider_config
async def start(
    provider_id: str,
    credential: tuple = Depends(get_current_credential),
    via_api_key: bool = Depends(authenticated_via_api_key),
):
    _require_provider(provider_id, via_api_key)
    if sys.version_info < (3, 11):
        raise HTTPException(
            503, "GitHub Copilot subscriptions require Studio running on Python 3.11 or newer."
        )
    client_id = os.environ.get("UNSLOTH_GITHUB_COPILOT_CLIENT_ID", "").strip()
    if not client_id:
        raise HTTPException(
            503,
            "Set UNSLOTH_GITHUB_COPILOT_CLIENT_ID to your GitHub OAuth App's client ID and enable its Device Flow.",
        )
    result = await _github_post("login/device/code", {"client_id": client_id})
    if not all(result.get(key) for key in ("device_code", "user_code", "expires_in")):
        raise HTTPException(
            502, "GitHub refused device authorization. Check the OAuth App's Device Flow setting."
        )
    flow_id = secrets.token_urlsafe(24)
    interval = max(5, int(result.get("interval", 5)))
    flow = {
        "id": flow_id,
        "device_code": result["device_code"],
        "client_id": client_id,
        "expires_at": time.time() + int(result["expires_in"]),
        "interval": interval,
        "next_poll": time.time() + interval,
    }
    with current_credential_write(credential):
        credential_secrets.upsert_secret(FLOW_KIND, provider_id, json.dumps(flow))
    return {
        "flow_id": flow_id,
        "user_code": result["user_code"],
        "verification_uri": "https://github.com/login/device",
        "expires_in": result["expires_in"],
        "interval": interval,
    }


@router.post("/{provider_id}/github/flows/{flow_id}/poll")
@serialize_provider_config
async def poll(
    provider_id: str,
    flow_id: str,
    credential: tuple = Depends(get_current_credential),
    via_api_key: bool = Depends(authenticated_via_api_key),
):
    _require_provider(provider_id, via_api_key)
    raw = credential_secrets.get_secret(FLOW_KIND, provider_id)
    flow = json.loads(raw) if raw else None
    if not flow or not secrets.compare_digest(flow["id"], flow_id):
        raise HTTPException(404, "Sign-in attempt not found. Start again.")
    now = time.time()
    if now >= flow["expires_at"]:
        with current_credential_write(credential):
            credential_secrets.delete_secret(FLOW_KIND, provider_id)
        return {"status": "expired"}
    if now < flow["next_poll"]:
        return {"status": "pending", "interval": flow["interval"]}
    # Persist before network I/O, including on failures, to honor GitHub's polling limit.
    flow["next_poll"] = now + flow["interval"]
    with current_credential_write(credential):
        credential_secrets.upsert_secret(FLOW_KIND, provider_id, json.dumps(flow))
    result = await _github_post(
        "login/oauth/access_token",
        {
            "client_id": flow["client_id"],
            "device_code": flow["device_code"],
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
        },
    )
    error = result.get("error")
    if error in ("authorization_pending", "slow_down"):
        if error == "slow_down":
            flow["interval"] += 5
            flow["next_poll"] = time.time() + flow["interval"]
            with current_credential_write(credential):
                credential_secrets.upsert_secret(FLOW_KIND, provider_id, json.dumps(flow))
        return {"status": "pending", "interval": flow["interval"]}
    with current_credential_write(credential):
        credential_secrets.delete_secret(FLOW_KIND, provider_id)
        token = result.get("access_token")
        if not error and isinstance(token, str) and token:
            # Same encrypted, account-scoped vault used for provider keys. Never sent to the browser.
            credential_secrets.save_provider_api_key(provider_id, token)
            return {"status": "connected"}
    if error == "expired_token":
        return {"status": "expired"}
    if error == "access_denied":
        return {"status": "denied"}
    raise HTTPException(502, "GitHub refused authorization. Start sign-in again.")


@router.delete("/{provider_id}/github", status_code = 204)
@serialize_provider_config
async def disconnect(
    provider_id: str,
    credential: tuple = Depends(get_current_credential),
    via_api_key: bool = Depends(authenticated_via_api_key),
):
    _require_provider(provider_id, via_api_key)
    with current_credential_write(credential):
        credential_secrets.delete_secret(FLOW_KIND, provider_id)
        credential_secrets.delete_provider_api_key(provider_id)
