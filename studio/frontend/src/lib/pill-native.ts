// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isTauri } from "@/lib/api-base";

export type PillSelectionPayload = {
  sessionId: number;
  text: string;
  appName: string;
  bundleId: string;
  editable: boolean;
  error: string | null;
};

export type PillNativeStatus = {
  supported: boolean;
  enabled: boolean;
  axTrusted: boolean;
  hotkey: string;
  excludedApps: string[];
};

export type PillNativeConfig = {
  enabled: boolean;
  hotkey: string;
  excludedApps: string[];
};

async function invokeNative<T>(command: string, args?: Record<string, unknown>): Promise<T> {
  if (!isTauri) {
    throw new Error("Native desktop features are only available in the Tauri app.");
  }
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<T>(command, args);
}

export const isMacPlatform = (): boolean =>
  typeof navigator !== "undefined" && /Mac/.test(navigator.userAgent);

export async function pillStatus(): Promise<PillNativeStatus> {
  return invokeNative<PillNativeStatus>("pill_status");
}

export async function pillSetConfig(config: PillNativeConfig): Promise<PillNativeStatus> {
  return invokeNative<PillNativeStatus>("pill_set_config", { config });
}

export async function pillRequestPermission(): Promise<boolean> {
  return invokeNative<boolean>("pill_request_permission");
}

export async function pillOpenPrivacySettings(): Promise<void> {
  return invokeNative<void>("pill_open_privacy_settings");
}

export async function pillGetCapture(): Promise<PillSelectionPayload | null> {
  return invokeNative<PillSelectionPayload | null>("pill_get_capture");
}

export async function pillReplaceSelection(sessionId: number, text: string): Promise<void> {
  return invokeNative<void>("pill_replace_selection", { sessionId, text });
}

export async function pillInsertBelow(sessionId: number, text: string): Promise<void> {
  return invokeNative<void>("pill_insert_below", { sessionId, text });
}

export async function pillDismiss(): Promise<void> {
  return invokeNative<void>("pill_dismiss");
}

export async function pillResize(width: number, height: number): Promise<void> {
  return invokeNative<void>("pill_resize", { width, height });
}

export async function pillServerPort(): Promise<number | null> {
  return invokeNative<number | null>("pill_server_port");
}

export async function pillOpenAsk(text: string): Promise<void> {
  return invokeNative<void>("pill_open_ask", { text });
}

export async function askHide(): Promise<void> {
  return invokeNative<void>("ask_hide");
}

export async function askResize(width: number, height: number): Promise<void> {
  return invokeNative<void>("ask_resize", { width, height });
}
