// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { AUTH_SESSION_CLEARED_EVENT, authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";
import type { AgentRecord } from "./agent-start";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await authFetch(`/api/chat/agents${path}`, {
    ...init,
    headers: init?.body ? { "Content-Type": "application/json" } : undefined,
  });
  if (!response.ok) throw new Error(await readFastApiError(response));
  return (response.status === 204 ? undefined : await response.json()) as T;
}

interface ChatAgentsState {
  agents: AgentRecord[];
  status: "idle" | "loading" | "ready" | "error";
  error: string | null;
  pendingModelLoad: {
    threadId: string;
    modelId: string;
    ggufVariant: string | null;
  } | null;
  refresh: () => Promise<void>;
  save: (
    agent: Omit<AgentRecord, "createdAt" | "updatedAt"> & { createdAt?: number },
  ) => Promise<AgentRecord>;
  remove: (id: string) => Promise<void>;
}

let refreshGeneration = 0;

export const useChatAgentsStore = create<ChatAgentsState>()((set, get) => ({
  agents: [],
  status: "idle",
  error: null,
  pendingModelLoad: null,
  refresh: async () => {
    const generation = ++refreshGeneration;
    if (get().status === "idle") set({ status: "loading" });
    try {
      const { agents } = await request<{ agents: AgentRecord[] }>("");
      if (generation !== refreshGeneration) return;
      set({ agents, status: "ready", error: null });
    } catch (error) {
      if (generation !== refreshGeneration) return;
      set({
        status: get().status === "ready" ? "ready" : "error",
        error: error instanceof Error ? error.message : String(error),
      });
    }
  },
  save: async (agent) => {
    const now = Date.now();
    const saved = await request<AgentRecord>("", {
      method: "POST",
      body: JSON.stringify({ ...agent, createdAt: agent.createdAt ?? now, updatedAt: now }),
    });
    refreshGeneration += 1;
    set((state) => ({
      agents: [saved, ...state.agents.filter((entry) => entry.id !== saved.id)],
      status: "ready",
    }));
    return saved;
  },
  remove: async (id) => {
    await request<void>(`/${encodeURIComponent(id)}`, { method: "DELETE" });
    refreshGeneration += 1;
    set((state) => ({
      agents: state.agents.filter((entry) => entry.id !== id),
    }));
  },
}));

if (typeof window !== "undefined") {
  window.addEventListener(AUTH_SESSION_CLEARED_EVENT, () => {
    refreshGeneration += 1;
    useChatAgentsStore.setState({
      agents: [],
      status: "idle",
      error: null,
      pendingModelLoad: null,
    });
  });
}
