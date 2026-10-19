# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from auth.authentication import get_current_subject
from storage.studio_db import (
    bulk_upsert_user_memories,
    delete_user_memory,
    list_user_memories,
    upsert_user_memory,
)

router = APIRouter()


class MemoryEntry(BaseModel):
    id: str = Field(max_length = 128)
    text: str = Field(min_length = 1, max_length = 2000)
    source: Literal["manual", "import"] = "manual"
    enabled: bool = True
    createdAt: int
    updatedAt: int


class BulkMemoriesRequest(BaseModel):
    entries: list[MemoryEntry] = Field(max_length = 1000)


@router.get("/entries")
def get_entries(current_subject: str = Depends(get_current_subject)):
    return {"entries": list_user_memories()}


@router.put("/entries/{entry_id}")
def put_entry(
    entry_id: str,
    entry: MemoryEntry,
    current_subject: str = Depends(get_current_subject),
):
    if entry.id != entry_id:
        raise HTTPException(status_code = 400, detail = "ID mismatch")
    return upsert_user_memory(entry.model_dump())


@router.delete("/entries/{entry_id}", status_code = 204)
def remove_entry(entry_id: str, current_subject: str = Depends(get_current_subject)):
    delete_user_memory(entry_id)


@router.post("/entries/bulk")
def bulk_entries(req: BulkMemoriesRequest, current_subject: str = Depends(get_current_subject)):
    count = bulk_upsert_user_memories([e.model_dump() for e in req.entries])
    return {"count": count}
