// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { formatApiErrorBody } from "@/lib/format-fastapi-error";

export const AUTO_TASKS = ["code", "reasoning", "writing", "general", "vision"] as const;
export type AutoTask = (typeof AUTO_TASKS)[number];

const AUTO_ROUTER_EVENT = "unsloth-auto-router-change";

export interface AutoRouterModel {
  id: string;
  tasks: AutoTask[];
  vision: boolean;
  tools: boolean;
  context_length: number | null;
}

export type AutoRouterLayaStatus = "ready" | "loading" | "unavailable" | "not_needed";

export interface AutoRouterSettings {
  models: AutoRouterModel[];
  default_model: string | null;
  rules: { contains: string; model: string }[];
  laya?: AutoRouterLayaStatus;
}

export interface AutoRouterCandidate {
  id: string;
  name: string;
  vision: boolean;
  tools: boolean;
  context_length: number | null;
  tasks: string[];
}

let cachedSettings: AutoRouterSettings | null = null;
let inFlightSettings: Promise<AutoRouterSettings> | null = null;

async function readJson<T>(response: Response): Promise<T> {
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(formatApiErrorBody(body) || `HTTP ${response.status}`);
  }
  return body as T;
}

function cacheSettings(settings: AutoRouterSettings): AutoRouterSettings {
  cachedSettings = settings;
  window.dispatchEvent(new CustomEvent(AUTO_ROUTER_EVENT, { detail: settings }));
  return settings;
}

export function subscribeAutoRouterSettings(
  listener: (settings: AutoRouterSettings) => void,
): () => void {
  const handleChange = (event: Event) => {
    listener((event as CustomEvent<AutoRouterSettings>).detail);
  };
  window.addEventListener(AUTO_ROUTER_EVENT, handleChange);
  return () => window.removeEventListener(AUTO_ROUTER_EVENT, handleChange);
}

export async function loadAutoRouterSettings({ force = false } = {}): Promise<AutoRouterSettings> {
  if (cachedSettings && !force) return cachedSettings;
  inFlightSettings ??= authFetch("/api/settings/auto-router")
    .then((response) => readJson<AutoRouterSettings>(response))
    .then(cacheSettings)
    .finally(() => {
      inFlightSettings = null;
    });
  return inFlightSettings;
}

export async function saveAutoRouterSettings(
  settings: AutoRouterSettings,
): Promise<AutoRouterSettings> {
  const { models, default_model, rules } = settings;
  const response = await authFetch("/api/settings/auto-router", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ models, default_model, rules }),
  });
  return cacheSettings(await readJson<AutoRouterSettings>(response));
}

export async function listAutoRouterCandidates(): Promise<AutoRouterCandidate[]> {
  const body = await readJson<{ models?: unknown }>(
    await authFetch("/api/settings/auto-router/candidates"),
  );
  return (Array.isArray(body?.models) ? body.models : [])
    .filter((model): model is AutoRouterCandidate =>
      typeof model?.id === "string" && model.id !== "auto",
    )
    .map((model) => ({
      id: model.id,
      name: typeof model.name === "string" && model.name ? model.name : model.id,
      vision: model.vision === true,
      tools: model.tools === true,
      context_length: typeof model.context_length === "number" ? model.context_length : null,
      tasks: Array.isArray(model.tasks) ? model.tasks.filter((task) => typeof task === "string") : [],
    }));
}

export function autoRouterModelFromCandidate(candidate: AutoRouterCandidate): AutoRouterModel {
  const tasks = candidate.tasks.filter((task): task is AutoTask =>
    (AUTO_TASKS as readonly string[]).includes(task),
  );
  return {
    id: candidate.id,
    tasks: tasks.length > 0 ? tasks : ["general"],
    vision: candidate.vision || tasks.includes("vision"),
    tools: candidate.tools,
    context_length: candidate.context_length,
  };
}
