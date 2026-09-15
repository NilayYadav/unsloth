// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useT } from "@/i18n";
import { ClusterNodesSection } from "../components/cluster-nodes-section";
import { ClusterShareSection } from "../components/cluster-share-section";
import { useClusterState } from "../hooks/use-cluster-state";

export function ClusterTab() {
  const t = useT();
  const cluster = useClusterState();

  return (
    <div className="flex min-w-0 max-w-full flex-col gap-6">
      <header className="flex min-w-0 flex-col gap-1">
        <h1
          data-settings-label={t("settings.cluster.title")}
          className="text-xl font-semibold font-heading"
        >
          {t("settings.cluster.title")}
        </h1>
        <p
          data-settings-label={t("settings.cluster.description")}
          className="text-xs text-muted-foreground"
        >
          {t("settings.cluster.description")}
        </p>
      </header>

      {cluster.loadFailed && cluster.state === null ? (
        <p className="text-xs text-destructive">
          Couldn't load cluster settings. Retrying…
        </p>
      ) : null}

      <ClusterNodesSection cluster={cluster} />

      <ClusterShareSection cluster={cluster} />
    </div>
  );
}
