# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth import policy
from auth.authentication import get_current_subject
from routes import memories
from storage import studio_db
from utils.account_context import OWNER, AccountContext, run_as

ALICE = AccountContext("a" * 32, "alice")


@pytest.fixture
def client():
    app = FastAPI()
    app.dependency_overrides[get_current_subject] = lambda: "unsloth"
    app.include_router(memories.router, prefix = "/memories")
    with TestClient(app) as test_client:
        yield test_client


def _memory(
    memory_id = "m1",
    text = "My dog is named Biscuit",
    **overrides,
):
    return {
        "id": memory_id,
        "text": text,
        "source": "manual",
        "enabled": True,
        "createdAt": 1,
        "updatedAt": 1,
        **overrides,
    }


def test_memories_round_trip_through_the_routes(client):
    assert client.get("/memories/entries").json() == {"entries": []}
    assert client.put("/memories/entries/m1", json = _memory()).status_code == 200
    assert (
        client.put(
            "/memories/entries/m2", json = _memory("m2", "Prefers metric units", createdAt = 2)
        ).status_code
        == 200
    )

    entries = client.get("/memories/entries").json()["entries"]
    assert [e["id"] for e in entries] == ["m2", "m1"]
    assert entries[1] == _memory()

    edited = _memory(text = "My dog is named Biscuit Jr.", enabled = False, updatedAt = 5)
    assert client.put("/memories/entries/m1", json = edited).status_code == 200
    assert client.get("/memories/entries").json()["entries"][1] == edited

    assert client.delete("/memories/entries/m1").status_code == 204
    assert [e["id"] for e in client.get("/memories/entries").json()["entries"]] == ["m2"]


def test_bulk_import_keeps_the_source_and_updates_existing_ids(client):
    client.put("/memories/entries/a", json = _memory("a", "old text"))
    imported = [
        _memory("a", "Lives in Lisbon", source = "import", updatedAt = 3),
        _memory("b", "Works as a nurse", source = "import", createdAt = 2),
    ]
    response = client.post("/memories/entries/bulk", json = {"entries": imported})
    assert response.json() == {"count": 2}
    entries = {e["id"]: e for e in client.get("/memories/entries").json()["entries"]}
    assert entries == {"a": imported[0], "b": imported[1]}


@pytest.mark.parametrize(
    "path,body",
    [
        ("/memories/entries/other", _memory()),
        ("/memories/entries/m1", _memory(text = "x" * 2001)),
        ("/memories/entries/m1", _memory(text = "")),
        ("/memories/entries/m1", _memory(source = "chatgpt")),
    ],
)
def test_bad_memories_are_refused(client, path, body):
    assert client.put(path, json = body).status_code in (400, 422)
    assert client.get("/memories/entries").json() == {"entries": []}


def test_each_account_keeps_its_own_memories(monkeypatch):
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    monkeypatch.setattr(studio_db, "_schema_ready", set())
    run_as(ALICE, studio_db.upsert_user_memory, _memory(text = "Alice's cat is Miso"))
    run_as(OWNER, studio_db.upsert_user_memory, _memory("o1", "Owner likes tea"))
    assert [e["text"] for e in run_as(ALICE, studio_db.list_user_memories)] == [
        "Alice's cat is Miso"
    ]
    assert [e["text"] for e in run_as(OWNER, studio_db.list_user_memories)] == ["Owner likes tea"]
