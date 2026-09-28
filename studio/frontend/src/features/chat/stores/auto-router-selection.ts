// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useSyncExternalStore } from "react";

const ENABLED_KEY = "unsloth_router_auto_enabled";
const MODEL_KEY = "unsloth_router_model";
const PINS_KEY = "unsloth_router_pins";

interface ThreadChoice {
  pin: string | null;
  lastModel: string | null;
}

export interface RouterRow {
  model: string;
  name: string;
  tools: boolean;
}

interface Selection {
  enabled: boolean;
  routerModel: string;
  configured: boolean;
  autoToolsCapable: boolean;
  toolsCapable: boolean;
  routers: RouterRow[];
  threads: Record<string, ThreadChoice>;
}

const EMPTY_CHOICE: ThreadChoice = { pin: null, lastModel: null };

function saved(key: string): string | null {
  try { return window.localStorage.getItem(key); } catch { return null; }
}

function savedPins(): Record<string, ThreadChoice> {
  if (typeof window === "undefined") return {};
  try {
    const value = JSON.parse(saved(PINS_KEY) || "{}");
    if (!value || typeof value !== "object" || Array.isArray(value)) return {};
    return Object.fromEntries(
      Object.entries(value)
        .filter(([threadId, pin]) => threadId !== "__default" && typeof pin === "string")
        .map(([threadId, pin]) => [threadId, { ...EMPTY_CHOICE, pin: pin as string }]),
    );
  } catch { return {}; }
}

function savedRouterModel(): string {
  const value = typeof window === "undefined" ? null : saved(MODEL_KEY);
  return value && (value === "auto" || value.startsWith("router/")) ? value : "auto";
}

let selection: Selection = {
  enabled: typeof window !== "undefined" && saved(ENABLED_KEY) === "1",
  routerModel: savedRouterModel(),
  configured: false,
  autoToolsCapable: false,
  toolsCapable: false,
  routers: [],
  threads: savedPins(),
};
const serverSelection: Selection = {
  enabled: false,
  routerModel: "auto",
  configured: false,
  autoToolsCapable: false,
  toolsCapable: false,
  routers: [],
  threads: {},
};
const listeners = new Set<() => void>();

function withToolsCapable(next: Selection): Selection {
  const router = next.routers.find((row) => row.model === next.routerModel);
  const toolsCapable = next.routerModel === "auto" ? next.autoToolsCapable : router?.tools === true;
  return toolsCapable === next.toolsCapable ? next : { ...next, toolsCapable };
}

function publish(next: Selection) {
  selection = withToolsCapable(next);
  listeners.forEach((listener) => listener());
}

export function isRouterModelId(model: unknown): model is string {
  return model === "auto" || (typeof model === "string" && model.startsWith("router/"));
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => { listeners.delete(listener); };
}

export function useAutoRouterSelection(): Selection {
  return useSyncExternalStore(subscribe, () => selection, () => serverSelection);
}

export function autoRouterSelection(): Selection {
  return selection;
}

export function autoRouterThread(threadId: string | null | undefined): ThreadChoice {
  return (threadId && selection.threads[threadId]) || EMPTY_CHOICE;
}

export function setAutoRouterEnabled(enabled: boolean) {
  if (selection.enabled === enabled) return;
  try {
    window.localStorage.setItem(ENABLED_KEY, enabled ? "1" : "0");
  } catch {
    // Ignore unavailable storage.
  }
  publish({ ...selection, enabled });
}

export function selectRouter(model: string) {
  try {
    window.localStorage.setItem(MODEL_KEY, model);
    window.localStorage.setItem(ENABLED_KEY, "1");
  } catch {
    // Ignore unavailable storage.
  }
  if (selection.enabled && selection.routerModel === model) return;
  publish({ ...selection, enabled: true, routerModel: model });
}

export function setAutoRouterSettings(models: { tools: boolean }[]) {
  const configured = models.length > 0;
  const autoToolsCapable = models.some((model) => model.tools);
  if (!configured && selection.enabled && selection.routerModel === "auto") setAutoRouterEnabled(false);
  if (selection.configured === configured && selection.autoToolsCapable === autoToolsCapable) return;
  publish({ ...selection, configured, autoToolsCapable });
}

export function setNamedRouters(routers: { model?: string; id: string; name: string; tools?: boolean }[]) {
  const rows = routers.map((router) => ({
    model: router.model ?? `router/${router.id}`,
    name: router.name,
    tools: router.tools === true,
  }));
  const next = { ...selection, routers: rows };
  if (next.routerModel !== "auto" && !rows.some((row) => row.model === next.routerModel)) {
    next.routerModel = "auto";
    try {
      window.localStorage.setItem(MODEL_KEY, "auto");
    } catch {
      // Ignore unavailable storage.
    }
  }
  publish(next);
}

export function setAutoRouterPin(threadId: string | null | undefined, pin: string | null) {
  if (!threadId) return;
  const threads = { ...selection.threads, [threadId]: { ...autoRouterThread(threadId), pin } };
  try {
    const pins = Object.fromEntries(Object.entries(threads).filter(([, choice]) => choice.pin).map(([id, choice]) => [id, choice.pin]));
    window.localStorage.setItem(PINS_KEY, JSON.stringify(pins));
  } catch {
    // Ignore unavailable storage.
  }
  publish({ ...selection, threads });
}

export function recordAutoRouterChoice(threadId: string | null | undefined, model: string) {
  if (!threadId) return;
  const current = autoRouterThread(threadId);
  if (current.lastModel === model) return;
  publish({
    ...selection,
    threads: { ...selection.threads, [threadId]: { ...current, lastModel: model } },
  });
}
