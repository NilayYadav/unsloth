# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
from routes.chat_history import router
from storage import studio_db


def _agent(**overrides) -> dict:
    agent = {
        "id": "agent-1",
        "name": "Researcher",
        "description": "Finds and cites sources.",
        "modelId": "unsloth/Qwen3-0.6B-GGUF",
        "modelGgufVariant": "Q4_K_M",
        "systemPrompt": "Cite every claim.",
        "tools": {"webSearch": True, "code": False, "images": False},
        "mcpServerIds": ["mcp-docs"],
        "skills": ["pdf"],
        "createdAt": 1_700_000_000_000,
        "updatedAt": 1_700_000_000_000,
    }
    agent.update(overrides)
    return agent


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", set())
    app = FastAPI()
    app.include_router(router, prefix = "/api/chat")
    app.dependency_overrides[get_current_subject] = lambda: "unsloth"
    with TestClient(app) as test_client:
        yield test_client


def test_a_saved_agent_is_listed_with_everything_it_was_saved_with(client):
    assert client.get("/api/chat/agents").json() == {"agents": []}

    saved = client.post("/api/chat/agents", json = _agent())

    assert saved.status_code == 200
    assert saved.json() == _agent()
    assert client.get("/api/chat/agents").json() == {"agents": [_agent()]}


def test_saved_agents_survive_reopening_the_database(client, monkeypatch):
    client.post("/api/chat/agents", json = _agent())

    monkeypatch.setattr(studio_db, "_schema_ready", set())

    assert client.get("/api/chat/agents").json() == {"agents": [_agent()]}


def test_saving_an_existing_agent_edits_it_and_keeps_when_it_was_created(client):
    client.post("/api/chat/agents", json = _agent())

    edited = _agent(
        name = "Analyst",
        modelId = None,
        modelGgufVariant = None,
        tools = {"webSearch": False, "code": True, "images": False},
        mcpServerIds = [],
        skills = [],
        createdAt = 5,
        updatedAt = 1_700_000_000_500,
    )
    assert client.post("/api/chat/agents", json = edited).status_code == 200

    assert client.get("/api/chat/agents").json() == {
        "agents": [{**edited, "createdAt": 1_700_000_000_000}]
    }


def test_agents_are_listed_most_recently_edited_first(client):
    client.post("/api/chat/agents", json = _agent(id = "old", name = "Old", updatedAt = 1))
    client.post("/api/chat/agents", json = _agent(id = "new", name = "New", updatedAt = 2))

    names = [agent["name"] for agent in client.get("/api/chat/agents").json()["agents"]]

    assert names == ["New", "Old"]


def test_an_agent_needs_only_a_name(client):
    minimal = {"id": "agent-2", "name": "  Helper  ", "createdAt": 1, "updatedAt": 1}

    saved = client.post("/api/chat/agents", json = minimal).json()

    assert saved == {
        "id": "agent-2",
        "name": "Helper",
        "description": "",
        "modelId": None,
        "modelGgufVariant": None,
        "systemPrompt": "",
        "tools": {"webSearch": False, "code": False, "images": False},
        "mcpServerIds": [],
        "skills": [],
        "createdAt": 1,
        "updatedAt": 1,
    }


@pytest.mark.parametrize(
    "overrides",
    [{"name": "   "}, {"tools": {"terminal": True}}, {"mcpServerIds": "mcp-docs"}],
)
def test_an_agent_that_is_not_well_formed_is_rejected_and_not_saved(client, overrides):
    assert client.post("/api/chat/agents", json = _agent(**overrides)).status_code == 422
    assert client.get("/api/chat/agents").json() == {"agents": []}


def test_a_deleted_agent_is_gone_and_deleting_it_again_is_not_found(client):
    client.post("/api/chat/agents", json = _agent())
    client.post("/api/chat/agents", json = _agent(id = "agent-2", name = "Coder"))

    assert client.delete("/api/chat/agents/agent-1").status_code == 204

    assert [a["id"] for a in client.get("/api/chat/agents").json()["agents"]] == ["agent-2"]
    assert client.delete("/api/chat/agents/agent-1").status_code == 404
