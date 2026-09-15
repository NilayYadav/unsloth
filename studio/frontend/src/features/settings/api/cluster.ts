// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";
import {
  type ClusterMode,
  type ClusterState,
  normalizeClusterState,
} from "./cluster-state";

export type { ClusterState } from "./cluster-state";

export type ClusterShareSettingsUpdate = {
  direct?: boolean;
  cacheCapGib?: number;
  resetCacheCap?: boolean;
};

async function requestCluster(
  path = "",
  init?: RequestInit,
): Promise<ClusterState> {
  const response = await authFetch(`/api/cluster${path}`, init);
  if (!response.ok) {
    throw new Error(await readFastApiError(response, "Cluster request failed"));
  }
  return normalizeClusterState(await response.json());
}

function sendCluster(path: string, method: string, body: unknown) {
  return requestCluster(path, {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

const nodePath = (nodeId: string) => `/nodes/${encodeURIComponent(nodeId)}`;

export const loadCluster = () => requestCluster();
export const updateClusterMode = (mode: ClusterMode) =>
  sendCluster("/mode", "PUT", { mode });

export const startClusterShare = () =>
  requestCluster("/share/start", { method: "POST" });
export const stopClusterShare = () =>
  requestCluster("/share/stop", { method: "POST" });
export const updateClusterShareSettings = (
  settings: ClusterShareSettingsUpdate,
) =>
  sendCluster("/share/settings", "PUT", {
    direct: settings.direct,
    // biome-ignore lint/style/useNamingConvention: API schema
    cache_cap_gib: settings.cacheCapGib,
    // biome-ignore lint/style/useNamingConvention: API schema
    reset_cache_cap: settings.resetCacheCap,
  });
export const newClusterPairingCode = () =>
  requestCluster("/share/pairing-code", { method: "POST" });
export const clearClusterShareCache = () =>
  requestCluster("/share/cache/clear", { method: "POST" });
export const revokeClusterHead = (headId: string) =>
  requestCluster(`/share/heads/${encodeURIComponent(headId)}`, {
    method: "DELETE",
  });

export const addClusterNode = (address: string, code: string) =>
  sendCluster("/nodes", "POST", { address, code });
export const updateClusterNode = (nodeId: string, enabled: boolean) =>
  sendCluster(nodePath(nodeId), "PATCH", { enabled });
export const removeClusterNode = (nodeId: string) =>
  requestCluster(nodePath(nodeId), { method: "DELETE" });
export const testClusterNode = (nodeId: string) =>
  requestCluster(`${nodePath(nodeId)}/test`, { method: "POST" });
