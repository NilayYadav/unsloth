// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useSyncExternalStore } from "react";

const ENABLED_KEY = "unsloth_router_auto_enabled";
const PINS_KEY = "unsloth_router_pins";

interface ThreadChoice {
  pin: string | null;
  lastModel: string | null;
}

interface Selection {
  enabled: boolean;
  configured: boolean;
  toolsCapable: boolean;
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

let selection: Selection = {
  enabled: typeof window !== "undefined" && saved(ENABLED_KEY) === "1",
  configured: false,
  toolsCapable: false,
  threads: savedPins(),
};
const serverSelection: Selection = { enabled: false, configured: false, toolsCapable: false, threads: {} };
const listeners = new Set<() => void>();

function publish(next: Selection) {
  selection = next;
  listeners.forEach((listener) => listener());
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

export function setAutoRouterSettings(models: { tools: boolean }[]) {
  const configured = models.length > 0;
  const toolsCapable = models.some((model) => model.tools);
  if (!configured && selection.enabled) setAutoRouterEnabled(false);
  if (selection.configured === configured && selection.toolsCapable === toolsCapable) return;
  publish({ ...selection, configured, toolsCapable });
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
