// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useMemo } from "react";
// eslint-disable-next-line no-restricted-imports -- The hub barrel imports chat; its inventory module does not.
import { useDeviceInventorySources } from "@/features/hub/inventory";
import {
  type ExternalProviderConfig,
  buildExternalModelId,
} from "../external-providers";
import { chatLocalModelOptions } from "../local-model-options";
import { useExternalProvidersStore } from "../stores/external-providers-store";

export type ConnectionChoice = { id: string; modelId: string; provider: ExternalProviderConfig };

/** Models a task can run on: local and connected providers, never one needing a download. */
export function useModelChoices(current: string | null) {
  const inventory = useDeviceInventorySources(["localModels", "cachedGguf"]);
  const providers = useExternalProvidersStore((state) => state.providers);
  const connectionsEnabled = useExternalProvidersStore((state) => state.connectionsEnabled);
  return useMemo(() => {
    const local = chatLocalModelOptions(inventory.localModels.rows).map(({ id, name }) => ({ id, name }));
    // GGUF only: a run switches back afterwards only to a GGUF.
    for (const repo of inventory.cachedGguf.rows) {
      if (!local.some((option) => option.id === repo.repo_id)) {
        local.push({ id: repo.repo_id, name: repo.repo_id });
      }
    }
    const connections = (connectionsEnabled ? providers : []).map((provider) => ({
      provider,
      models: provider.models.map(
        (modelId): ConnectionChoice => ({ id: buildExternalModelId(provider.id, modelId), modelId, provider }),
      ),
    }));
    const known =
      current === null ||
      local.some((option) => option.id === current) ||
      connections.some((group) => group.models.some((option) => option.id === current));
    // A saved choice that is gone (deleted, disconnected) stays visible rather than silently changing.
    const missing = known ? null : current;
    const connection = connections.flatMap((group) => group.models).find((option) => option.id === current);
    return { local, connections, missing, connection };
  }, [inventory.localModels.rows, inventory.cachedGguf.rows, providers, connectionsEnabled, current]);
}
