# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
from core.inference.ask_agent_tool import askable_agents
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
        "canAsk": ["agent-7"],
        "filesForAskedOnly": True,
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
        canAsk = [],
        filesForAskedOnly = False,
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
        "canAsk": [],
        "filesForAskedOnly": False,
        "createdAt": 1,
        "updatedAt": 1,
    }


@pytest.mark.parametrize(
    "overrides",
    [
        {"name": "   "},
        {"tools": {"terminal": True}},
        {"mcpServerIds": "mcp-docs"},
        {"canAsk": ["agent-7", "agent-1"]},
    ],
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


def test_agents_an_agent_can_ask_are_kept_as_saved_and_only_the_ones_here_can_be_asked(client):
    client.post("/api/chat/agents", json = _agent(id = "agent-2", name = "Reader", canAsk = ["agent-1"]))
    client.post("/api/chat/agents", json = _agent(canAsk = ["agent-2", "gone", "agent-2"]))

    def saved():
        agents = client.get("/api/chat/agents").json()["agents"]
        return next(agent["canAsk"] for agent in agents if agent["id"] == "agent-1")

    assert saved() == ["agent-2", "gone"]
    assert [agent["name"] for agent in askable_agents("agent-1")] == ["Reader"]

    client.delete("/api/chat/agents/agent-2")

    assert saved() == ["agent-2", "gone"]
    assert askable_agents("agent-1") == []


def test_agents_saved_before_they_could_ask_each_other_still_load(client, tmp_path):
    client.post("/api/chat/agents", json = _agent())
    conn = studio_db.get_connection()
    conn.execute("ALTER TABLE chat_agents DROP COLUMN can_ask_json")
    conn.execute("ALTER TABLE chat_agents DROP COLUMN files_for_asked_only")
    conn.commit()
    conn.close()
    studio_db._schema_ready.clear()

    assert client.get("/api/chat/agents").json() == {
        "agents": [_agent(canAsk = [], filesForAskedOnly = False)]
    }


def test_a_chat_remembers_the_agent_it_was_started_from(client):
    thread = {"id": "t1", "modelType": "base", "createdAt": 1, "agentId": "agent-1"}

    assert client.post("/api/chat/threads", json = thread).json()["agentId"] == "agent-1"

    client.post("/api/chat/threads", json = {**thread, "title": "Renamed", "agentId": None})

    assert client.get("/api/chat/threads/t1").json()["agentId"] == "agent-1"
