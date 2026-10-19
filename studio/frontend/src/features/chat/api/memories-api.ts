// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";

export type MemorySource = "manual" | "import";

export interface MemoryEntry {
  id: string;
  text: string;
  source: MemorySource;
  enabled: boolean;
  createdAt: number;
  updatedAt: number;
}

async function parseJsonOrThrow<T>(res: Response): Promise<T> {
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    const detail = (body as { detail?: string } | null)?.detail;
    throw new Error(
      typeof detail === "string" ? detail : `Request failed (${res.status})`,
    );
  }
  return body as T;
}

export async function listMemoryEntries(): Promise<MemoryEntry[]> {
  const res = await authFetch("/api/memories/entries");
  const data = await parseJsonOrThrow<{ entries: MemoryEntry[] }>(res);
  return data.entries;
}

export async function saveMemoryEntry(
  entry: MemoryEntry,
): Promise<MemoryEntry> {
  const res = await authFetch(`/api/memories/entries/${entry.id}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(entry),
  });
  return parseJsonOrThrow<MemoryEntry>(res);
}

export async function deleteMemoryEntry(id: string): Promise<void> {
  const res = await authFetch(`/api/memories/entries/${id}`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error(`Delete failed (${res.status})`);
}

export async function bulkSaveMemoryEntries(
  entries: MemoryEntry[],
): Promise<number> {
  if (!entries.length) return 0;
  const res = await authFetch("/api/memories/entries/bulk", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ entries }),
  });
  const data = await parseJsonOrThrow<{ count: number }>(res);
  return data.count;
}
