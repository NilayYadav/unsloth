// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useSyncExternalStore } from "react";

const ENABLED_KEY = "unsloth_router_auto_enabled";
const PINS_KEY = "unsloth_router_pins";

interface ThreadChoice {
  pin: string | null;
  lastModel: string | null;
  lastReason: string | null;
}

interface Selection {
  enabled: boolean;
  toolsCapable: boolean;
  threads: Record<string, ThreadChoice>;
}

function saved(key: string): string | null {
  try { return window.localStorage.getItem(key); } catch { return null; }
}

function savedPins(): Record<string, ThreadChoice> {
  if (typeof window === "undefined") return {};
  try {
    const value = JSON.parse(saved(PINS_KEY) || "{}");
    if (!value || typeof value !== "object" || Array.isArray(value)) return {};
    return Object.fromEntries(
      Object.entries(value).filter(([, pin]) => typeof pin === "string").map(([threadId, pin]) => [threadId, {
        pin: pin as string,
        lastModel: null,
        lastReason: null,
      }]),
    );
  } catch { return {}; }
}

let selection: Selection = {
  enabled: typeof window !== "undefined" && saved(ENABLED_KEY) === "1",
  toolsCapable: false,
  threads: savedPins(),
};
const serverSelection: Selection = { enabled: false, toolsCapable: false, threads: {} };
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
  return selection.threads[threadId || "__default"] || { pin: null, lastModel: null, lastReason: null };
}

export function setAutoRouterEnabled(enabled: boolean) {
  try { window.localStorage.setItem(ENABLED_KEY, enabled ? "1" : "0"); } catch {}
  publish({ ...selection, enabled, toolsCapable: enabled && selection.toolsCapable });
}

export function setAutoRouterToolsCapable(toolsCapable: boolean) {
  publish({ ...selection, toolsCapable });
}

export function setAutoRouterPin(threadId: string | null | undefined, pin: string | null) {
  const key = threadId || "__default";
  const threads = { ...selection.threads, [key]: { ...autoRouterThread(threadId), pin } };
  try {
    const pins = Object.fromEntries(Object.entries(threads).filter(([, choice]) => choice.pin).map(([id, choice]) => [id, choice.pin]));
    window.localStorage.setItem(PINS_KEY, JSON.stringify(pins));
  } catch {}
  publish({ ...selection, threads });
}

export function recordAutoRouterChoice(threadId: string | null | undefined, model: string, reason: string | null) {
  const key = threadId || "__default";
  publish({
    ...selection,
    threads: { ...selection.threads, [key]: { ...autoRouterThread(threadId), lastModel: model, lastReason: reason } },
  });
}
