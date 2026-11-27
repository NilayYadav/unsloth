# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Official SDK transport and account-scoped OAuth device lifecycle, without a paid subscription."""

import asyncio
import json
from contextlib import asynccontextmanager, nullcontext
from types import SimpleNamespace as Obj

import pytest
from fastapi import HTTPException

from core.inference import github_copilot as copilot
from core.inference.external_provider import ExternalProviderClient
from core.inference.providers import get_provider_info, list_available_providers
from routes import github_copilot_auth as auth


@pytest.fixture
def flow_store(monkeypatch):
    vault = {}
    monkeypatch.setenv("UNSLOTH_GITHUB_COPILOT_CLIENT_ID", "our-oauth-app")
    monkeypatch.setattr(
        auth.providers_db, "get_own_provider", lambda _: {"provider_type": "github_copilot"}
    )
    monkeypatch.setattr(auth, "current_credential_write", lambda _: nullcontext())
    monkeypatch.setattr(
        auth.credential_secrets,
        "upsert_secret",
        lambda kind, scope, value: vault.__setitem__((kind, scope), value),
    )
    monkeypatch.setattr(
        auth.credential_secrets, "get_secret", lambda kind, scope: vault.get((kind, scope))
    )
    monkeypatch.setattr(
        auth.credential_secrets, "delete_secret", lambda kind, scope: vault.pop((kind, scope), None)
    )
    monkeypatch.setattr(
        auth.credential_secrets,
        "save_provider_api_key",
        lambda scope, value: vault.__setitem__(("token", scope), value),
    )
    monkeypatch.setattr(
        auth.credential_secrets,
        "delete_provider_api_key",
        lambda scope: vault.pop(("token", scope), None),
    )
    now = [1000.0]
    monkeypatch.setattr(auth.time, "time", lambda: now[0])
    return vault, now


async def begin(monkeypatch, result = None):
    async def post(path, data):
        assert path == "login/device/code"
        assert data == {"client_id": "our-oauth-app"}
        return result or {
            "device_code": "private-device-code",
            "user_code": "ABCD-EFGH",
            "expires_in": 900,
            "interval": 5,
        }

    monkeypatch.setattr(auth, "_github_post", post)
    return await auth.start("connection", ("user", None), False)


def test_device_token_stays_server_side_and_polling_is_throttled(monkeypatch, flow_store):
    vault, now = flow_store

    async def run():
        flow = await begin(monkeypatch)
        assert "private-device-code" not in json.dumps(flow)
        assert flow["verification_uri"] == "https://github.com/login/device"
        # The initial poll must not hit GitHub.
        assert (await auth.poll("connection", flow["flow_id"], ("user", None), False))[
            "status"
        ] == "pending"
        now[0] += 5

        async def post(path, data):
            assert path == "login/oauth/access_token"
            assert data["device_code"] == "private-device-code"
            return {"access_token": "gho_private-token"}

        monkeypatch.setattr(auth, "_github_post", post)
        result = await auth.poll("connection", flow["flow_id"], ("user", None), False)
        assert result == {"status": "connected"}
        assert vault[("token", "connection")] == "gho_private-token"
        assert (auth.FLOW_KIND, "connection") not in vault
        with pytest.raises(HTTPException) as exc:
            await auth.poll("connection", flow["flow_id"], ("user", None), False)
        assert exc.value.status_code == 404

    asyncio.run(run())


@pytest.mark.parametrize(
    "error,status", [("access_denied", "denied"), ("expired_token", "expired")]
)
def test_denied_and_expired_never_save_a_token(monkeypatch, flow_store, error, status):
    vault, now = flow_store

    async def run():
        flow = await begin(monkeypatch)
        now[0] += 5

        async def post(*_):
            return {"error": error}

        monkeypatch.setattr(auth, "_github_post", post)
        assert await auth.poll("connection", flow["flow_id"], ("user", None), False) == {
            "status": status
        }
        assert not vault

    asyncio.run(run())


def test_slow_down_and_local_expiry(monkeypatch, flow_store):
    vault, now = flow_store

    async def run():
        flow = await begin(monkeypatch)
        now[0] += 5

        async def post(*_):
            return {"error": "slow_down"}

        monkeypatch.setattr(auth, "_github_post", post)
        result = await auth.poll("connection", flow["flow_id"], ("user", None), False)
        assert result == {"status": "pending", "interval": 10}
        assert json.loads(vault[(auth.FLOW_KIND, "connection")])["next_poll"] == 1015
        now[0] += 900
        assert await auth.poll("connection", flow["flow_id"], ("user", None), False) == {
            "status": "expired"
        }
        assert not vault

    asyncio.run(run())


def test_replacement_cancel_and_cross_connection_flows(monkeypatch, flow_store):
    async def run():
        first = await begin(monkeypatch)
        second = await begin(monkeypatch)
        for provider, flow_id in [
            ("connection", first["flow_id"]),
            ("different", second["flow_id"]),
        ]:
            with pytest.raises(HTTPException) as exc:
                await auth.poll(provider, flow_id, ("user", None), False)
            assert exc.value.status_code == 404
        await auth.disconnect("connection", ("user", None), False)
        with pytest.raises(HTTPException):
            await auth.poll("connection", second["flow_id"], ("user", None), False)

    asyncio.run(run())


def test_api_keys_and_other_accounts_cannot_authorize(monkeypatch, flow_store):
    async def run():
        with pytest.raises(HTTPException) as exc:
            await auth.start("connection", ("user", None), True)
        assert exc.value.status_code == 403
        monkeypatch.setattr(auth.providers_db, "get_own_provider", lambda _: None)
        with pytest.raises(HTTPException) as exc:
            await auth.start("connection", ("user", None), False)
        assert exc.value.status_code == 404

    asyncio.run(run())


def test_missing_oauth_app_has_actionable_error(monkeypatch, flow_store):
    monkeypatch.delenv("UNSLOTH_GITHUB_COPILOT_CLIENT_ID")
    with pytest.raises(HTTPException, match = "UNSLOTH_GITHUB_COPILOT_CLIENT_ID"):
        asyncio.run(auth.start("connection", ("user", None), False))


def test_unsupported_python_fails_before_authorization(monkeypatch, flow_store):
    monkeypatch.setattr(auth.sys, "version_info", (3, 10))
    with pytest.raises(HTTPException, match = "Python 3.11"):
        asyncio.run(auth.start("connection", ("user", None), False))


class Session:
    def __init__(self):
        self.closed = False
        self.aborted = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        self.closed = True

    def on(self, callback):
        self.callback = callback

    async def send_and_wait(self, prompt, timeout):
        self.prompt = prompt
        self.callback(Obj(type = "assistant.message_delta", data = Obj(delta_content = "Hello")))
        self.callback(Obj(type = "assistant.message", data = Obj(content = "Hello")))
        self.callback(Obj(type = "assistant.usage", data = Obj(input_tokens = 10, output_tokens = 2)))

    async def abort(self):
        self.aborted = True


@pytest.fixture
def fake_sdk(monkeypatch):
    session = Session()
    state = {}

    class Client:
        async def create_session(self, **kwargs):
            state.update(kwargs)
            return session

        async def list_models(self):
            return [
                Obj(id = name, name = name, policy = policy)
                for name, policy in (
                    ("allowed", Obj(state = Obj(value = "enabled"))),
                    ("no-policy", None),
                    ("blocked", Obj(state = Obj(value = "disabled"))),
                )
            ]

    @asynccontextmanager
    async def client(token):
        assert token == "gho_user-token"
        try:
            yield Client()
        finally:
            state["client_closed"] = True

    monkeypatch.setattr(copilot, "sdk_client", client)
    return session, state


def test_transport_streams_once_and_preserves_editable_history(fake_sdk):
    session, state = fake_sdk

    async def run():
        client = ExternalProviderClient("github_copilot", copilot.BASE_URL, "gho_user-token")
        lines = [
            line
            async for line in client.stream_chat_completion(
                [
                    {"role": "system", "content": "Be concise"},
                    {"role": "user", "content": "Remember 7"},
                    {"role": "assistant", "content": "OK"},
                    {"role": "user", "content": "What number?"},
                ],
                "allowed",
            )
        ]
        chunks = [json.loads(line[6:]) for line in lines if "[DONE]" not in line]
        assert "".join(c["choices"][0]["delta"].get("content", "") for c in chunks) == "Hello"
        assert chunks[-1]["usage"]["total_tokens"] == 12
        assert lines[-1] == "data: [DONE]\n\n"
        assert state["system_message"]["content"] == "Be concise"
        assert state["tools"] == {"available_tools": []}
        assert state["permissions"]["on_request"]().kind == "reject"
        assert "Remember 7" in session.prompt and "What number?" in session.prompt
        assert session.closed and state["client_closed"]

    asyncio.run(run())


def test_models_use_sdk_and_filter_disabled_models(fake_sdk):
    assert [m["id"] for m in asyncio.run(copilot.list_models("gho_user-token"))] == [
        "allowed",
        "no-policy",
    ]


def test_cancel_aborts_sdk_generation(fake_sdk, monkeypatch):
    session, state = fake_sdk

    async def waiting(*args, **kwargs):
        await asyncio.Event().wait()

    monkeypatch.setattr(session, "send_and_wait", waiting)

    async def run():
        stream = copilot.stream_chat(
            "gho_user-token", [{"role": "user", "content": "Hi"}], "allowed"
        )
        await anext(stream)
        await stream.aclose()
        assert session.aborted and session.closed and state["client_closed"]

    asyncio.run(run())


def test_sdk_errors_never_return_success(fake_sdk, monkeypatch):
    session, _ = fake_sdk

    async def error(*args, **kwargs):
        session.callback(Obj(type = "session.error", data = Obj(message = "private-token")))

    monkeypatch.setattr(session, "send_and_wait", error)

    async def run():
        with pytest.raises(RuntimeError, match = "subscription") as exc:
            async for _ in copilot.stream_chat(
                "gho_user-token", [{"role": "user", "content": "Hi"}], "allowed"
            ):
                pass
        assert "private-token" not in str(exc.value)

    asyncio.run(run())


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "secret"}}]}],
        [{"role": "tool", "content": "result"}],
    ],
)
def test_unsupported_content_is_rejected(messages):
    with pytest.raises(ValueError):
        copilot._prompts(messages)


def test_fixed_routing_and_registry_contract():
    with pytest.raises(ValueError, match = "routing is fixed"):
        ExternalProviderClient("github_copilot", "https://elsewhere.example", "gho_secret")
    info = get_provider_info("github_copilot")
    assert info["auth_kind"] == "github_oauth" and not info["supports_tool_calling"]
    assert "github_copilot" not in {p["provider_type"] for p in list_available_providers()}
    assert "github_copilot" in {
        p["provider_type"] for p in list_available_providers(include_oauth = True)
    }


def test_subscription_rejects_key_edits_sharing_and_output_overrides():
    from routes import providers

    info = get_provider_info("github_copilot")
    for changes in (
        {"encrypted_api_key": "replacement"},
        {"clear_api_key": True},
        {"base_url": "https://elsewhere.example"},
    ):
        kwargs = {"encrypted_api_key": None, "base_url": None, "models": [], "updating": True}
        kwargs.update(changes)
        with pytest.raises(HTTPException) as error:
            providers._validate_provider_auth_contract(info, **kwargs)
        assert error.value.status_code == 400
    with pytest.raises(HTTPException):
        providers._validate_share_request(True, "github_copilot", copilot.BASE_URL)
    with pytest.raises(HTTPException):
        providers._validate_max_output_tokens_contract("github_copilot", True, 1024)
