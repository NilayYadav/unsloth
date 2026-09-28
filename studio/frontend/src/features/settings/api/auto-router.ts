// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { formatApiErrorBody } from "@/lib/format-fastapi-error";

export const AUTO_TASKS = ["code", "reasoning", "writing", "general", "vision"] as const;
export type AutoTask = (typeof AUTO_TASKS)[number];

const AUTO_ROUTER_EVENT = "unsloth-auto-router-change";
const NAMED_ROUTERS_EVENT = "unsloth-named-routers-change";

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
  automatic?: boolean;
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

export function resetAutoRouterSettings(): Promise<AutoRouterSettings> {
  return saveAutoRouterSettings({ models: [], default_model: null, rules: [] });
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

export const ROUTER_SLOTS = ["code", "reasoning", "writing", "general", "vision"] as const;
export type RouterSlot = (typeof ROUTER_SLOTS)[number];

export interface NamedRouter {
  id: string;
  name: string;
  slots: Partial<Record<RouterSlot, string>>;
  model?: string;
  ready?: boolean;
  tools?: boolean;
  vision?: boolean;
}

export interface RouterPreview {
  model: string;
  reason: string;
  task: string | null;
  loaded: boolean;
}

let cachedRouters: NamedRouter[] | null = null;

function cacheRouters(routers: NamedRouter[]): NamedRouter[] {
  cachedRouters = routers;
  window.dispatchEvent(new CustomEvent(NAMED_ROUTERS_EVENT, { detail: routers }));
  return routers;
}

export function subscribeNamedRouters(listener: (routers: NamedRouter[]) => void): () => void {
  const handleChange = (event: Event) => {
    listener((event as CustomEvent<NamedRouter[]>).detail);
  };
  window.addEventListener(NAMED_ROUTERS_EVENT, handleChange);
  return () => window.removeEventListener(NAMED_ROUTERS_EVENT, handleChange);
}

export async function loadNamedRouters({ force = false } = {}): Promise<NamedRouter[]> {
  if (cachedRouters && !force) return cachedRouters;
  const body = await readJson<{ routers?: NamedRouter[] }>(
    await authFetch("/api/settings/auto-router/routers"),
  );
  return cacheRouters(Array.isArray(body?.routers) ? body.routers : []);
}

export async function saveNamedRouters(routers: NamedRouter[]): Promise<NamedRouter[]> {
  const response = await authFetch("/api/settings/auto-router/routers", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      routers: routers.map(({ id, name, slots }) => ({ id, name, slots })),
    }),
  });
  const body = await readJson<{ routers?: NamedRouter[] }>(response);
  return cacheRouters(Array.isArray(body?.routers) ? body.routers : []);
}

export async function previewRouter(model: string, prompt: string): Promise<RouterPreview> {
  const response = await authFetch("/api/settings/auto-router/preview", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model, prompt }),
  });
  return readJson<RouterPreview>(response);
}

export function routerIdFromName(name: string, taken: readonly string[]): string {
  const base =
    name
      .toLowerCase()
      .normalize("NFKD")
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .slice(0, 40) || "router";
  let id = base;
  for (let n = 2; taken.includes(id); n += 1) id = `${base}-${n}`;
  return id;
}
