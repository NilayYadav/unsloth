// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useT } from "@/i18n";
import { ClusterNodesSection } from "../components/cluster-nodes-section";
import { ClusterShareSection } from "../components/cluster-share-section";
import { useClusterState } from "../hooks/use-cluster-state";

const SPEED_TIPS = [
  "Use Ethernet (2.5 or 10 GbE) or a Thunderbolt cable between computers.",
  "The first load copies model layers over the network. Later loads reuse each computer's cache.",
  "Every extra computer adds a network hop per token, so fewer, faster computers beat many slow ones.",
  "All computers need the same kind of GPU (Apple Silicon with Apple Silicon, NVIDIA with NVIDIA) and the same Unsloth version.",
];

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

      <section
        data-settings-label="Tips for speed"
        className="rounded-lg border border-border/70 p-4"
      >
        <h2 className="text-sm font-semibold font-heading text-foreground">
          Tips for speed
        </h2>
        <ul className="mt-2 flex list-disc flex-col gap-1 pl-4 text-xs leading-snug text-muted-foreground">
          {SPEED_TIPS.map((tip) => (
            <li key={tip}>{tip}</li>
          ))}
        </ul>
      </section>
    </div>
  );
}
