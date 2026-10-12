# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import os

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from auth.authentication import (
    authenticated_via_api_key,
    get_current_subject,
    request_admitted_without_credential,
)
from core.inference import mcp_client
from core.inference.tool_loop_controller import coerce_tool_arguments
from core.inference.tools import execute_tool, get_enabled_mcp_tools
from routes import mcp_servers as routes_mcp
from storage import mcp_servers_db
from utils import host_policy

TOOL = '''
import os

def shout(text: str) -> str:
    """Shout the text back."""
    return text.upper() + os.environ.get("SUFFIX", "")

def add(a: int, b: int = 2) -> int:
    """Add two numbers."""
    return a + b

def _helper():
    return "private"
'''


@pytest.fixture
def client(tmp_path, monkeypatch):
    saved = os.environ.get("UNSLOTH_STUDIO_ALLOW_STDIO_MCP")
    host_policy._reset_loopback_default_state()
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", "1")
    monkeypatch.setattr(mcp_servers_db, "_schema_ready", set())
    mcp_client.invalidate_tool_cache()
    app = FastAPI()
    app.dependency_overrides[get_current_subject] = lambda: "u"
    app.dependency_overrides[authenticated_via_api_key] = _api_key_dependency
    app.dependency_overrides[request_admitted_without_credential] = lambda: False
    app.include_router(routes_mcp.router, prefix = "/mcp")
    with TestClient(app) as test_client:
        yield test_client
    mcp_client.close_mcp_sessions()
    host_policy._reset_loopback_default_state()
    if saved is None:
        os.environ.pop("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", None)
    else:
        os.environ["UNSLOTH_STUDIO_ALLOW_STDIO_MCP"] = saved


def _api_key_dependency(request: Request) -> bool:
    return "x-api-key" in request.headers


def _save(client, code, server_id = None, **extra):
    body = {"display_name": "My tools", "code": code, **extra}
    if server_id is None:
        return client.post("/mcp/custom-tools", json = body)
    return client.put(f"/mcp/custom-tools/{server_id}", json = body)


def _catalog():
    return {spec["function"]["name"]: spec for spec in asyncio.run(get_enabled_mcp_tools())}


@pytest.mark.timeout(120)
def test_a_tool_written_in_studio_reaches_the_model_and_runs(client):
    response = _save(client, TOOL, headers = {"SUFFIX": "!"})
    assert response.status_code == 201, response.text
    saved = response.json()
    assert saved["probe"] == {
        "ok": True,
        "tool_count": 2,
        "error": None,
        "blender_ready": None,
        "blender_error": None,
    }
    assert saved["code"] == TOOL and saved["headers"] == {"SUFFIX": "!"}

    catalog = _catalog()
    shout = f"mcp__{saved['id']}__shout"
    assert set(catalog) == {shout, f"mcp__{saved['id']}__add"}
    assert "Shout the text back." in catalog[shout]["function"]["description"]

    healed = coerce_tool_arguments(
        "hello", heal = True, tool_name = shout, tool_schemas = list(catalog.values())
    )
    assert healed.arguments == {"text": "hello"}
    assert "HELLO!" in execute_tool(shout, healed.arguments)
    assert "5" in execute_tool(f"mcp__{saved['id']}__add", {"a": 3})


@pytest.mark.timeout(120)
def test_editing_disabling_and_deleting_a_tool(client):
    server_id = _save(client, TOOL).json()["id"]
    assert "HELLO" in execute_tool(f"mcp__{server_id}__shout", {"text": "hello"})

    edited = _save(client, TOOL.replace(".upper()", ".lower()"), server_id).json()
    assert edited["probe"]["ok"] and edited["code"].count(".lower()") == 1
    assert "quiet" in execute_tool(f"mcp__{server_id}__shout", {"text": "QUIET"})
    assert client.get(f"/mcp/custom-tools/{server_id}").json()["code"] == edited["code"]

    assert client.put(f"/mcp/{server_id}", json = {"is_enabled": False}).status_code == 200
    assert _catalog() == {}
    assert client.put(f"/mcp/{server_id}", json = {"is_enabled": True}).status_code == 200
    assert len(_catalog()) == 2

    assert client.delete(f"/mcp/{server_id}").status_code == 204
    assert client.get("/mcp/").json() == []
    assert _catalog() == {}
    assert client.get(f"/mcp/custom-tools/{server_id}").status_code == 404


@pytest.mark.timeout(120)
def test_errors_in_the_code_are_shown_without_running_it(client, tmp_path):
    marker = tmp_path / "ran"
    side_effect = f"open({str(marker)!r}, 'w').close()\n"
    parsed = client.post("/mcp/custom-tools/parse", json = {"code": side_effect + TOOL}).json()
    assert not marker.exists()
    assert parsed["error"] is None and parsed["unavailable_reason"] is None
    assert parsed["tools"] == [
        {
            "name": "shout",
            "description": "Shout the text back.",
            "params": [{"name": "text", "type": "str", "required": True}],
        },
        {
            "name": "add",
            "description": "Add two numbers.",
            "params": [
                {"name": "a", "type": "int", "required": True},
                {"name": "b", "type": "int", "required": False},
            ],
        },
    ]
    broken = client.post("/mcp/custom-tools/parse", json = {"code": "def f(:\n"}).json()
    assert broken["tools"] == [] and broken["error"].startswith("Line 1:")

    saved = _save(client, "import not_a_real_module_xyz\n" + TOOL).json()
    assert saved["probe"]["ok"] is False
    assert saved["probe"]["error"] == "ModuleNotFoundError: No module named 'not_a_real_module_xyz'"


def test_custom_tools_need_the_ui_and_local_commands(client, monkeypatch):
    api_key = {"x-api-key": "1"}
    assert _save(client, TOOL).status_code == 201
    server_id = client.get("/mcp/").json()[0]["id"]
    for response in (
        client.post("/mcp/custom-tools", json = {"display_name": "x", "code": TOOL}, headers = api_key),
        client.get(f"/mcp/custom-tools/{server_id}", headers = api_key),
        client.post("/mcp/custom-tools/parse", json = {"code": TOOL}, headers = api_key),
        client.delete(f"/mcp/{server_id}", headers = api_key),
    ):
        assert response.status_code == 403
    assert client.get("/mcp/", headers = api_key).json() == []

    monkeypatch.setenv("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", "0")
    refused = _save(client, TOOL)
    assert refused.status_code == 400
    assert "Local commands aren't enabled" in refused.json()["detail"]
    parsed = client.post("/mcp/custom-tools/parse", json = {"code": TOOL}).json()
    assert "Local commands aren't enabled" in parsed["unavailable_reason"]
    assert len(client.get("/mcp/").json()) == 1
    assert _catalog() == {}
