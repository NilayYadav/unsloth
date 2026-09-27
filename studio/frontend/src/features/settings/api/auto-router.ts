// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { formatApiErrorBody } from "@/lib/format-fastapi-error";

export const AUTO_TASKS = ["code", "reasoning", "writing", "general", "vision"] as const;
export type AutoTask = (typeof AUTO_TASKS)[number];

export interface AutoRouterModel {
  id: string;
  tasks: AutoTask[];
  vision: boolean;
  tools: boolean;
  context_length: number | null;
}

export interface AutoRouterSettings {
  models: AutoRouterModel[];
  default_model: string | null;
  rules: { contains: string; model: string }[];
}

async function readSettings(response: Response): Promise<AutoRouterSettings> {
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(formatApiErrorBody(body) || `HTTP ${response.status}`);
  }
  return body as AutoRouterSettings;
}

export async function loadAutoRouterSettings(): Promise<AutoRouterSettings> {
  return readSettings(await authFetch("/api/settings/auto-router"));
}

export async function saveAutoRouterSettings(
  settings: AutoRouterSettings,
): Promise<AutoRouterSettings> {
  return readSettings(
    await authFetch("/api/settings/auto-router", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(settings),
    }),
  );
}

export async function listAutoRouterCandidates(): Promise<{ id: string; name: string }[]> {
  const response = await authFetch("/v1/models");
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(formatApiErrorBody(body) || `HTTP ${response.status}`);
  }
  const models = Array.isArray(body?.data) ? body.data : [];
  return models
    .filter((model: { id?: unknown; task?: unknown }) =>
      typeof model.id === "string" && model.id !== "auto" && !model.task,
    )
    .map((model: { id: string; display_name?: string }) => ({
      id: model.id,
      name: model.display_name || model.id,
    }));
}
