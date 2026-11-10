// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useNavigate } from "@tanstack/react-router";
import { useEffect } from "react";
import {
  INVENTORY_FRESHNESS_WINDOW_MS,
  useDeviceInventorySources,
  useDeviceInventoryStore,
} from "@/features/hub";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { modelDisplayName } from "../../model-picker/model-config/model-identity";
import { listGgufVariants } from "../api/chat-api";
import { listMcpServers, updateMcpServer } from "../api/mcp-servers-api";
import { type SkillRecord, listSkills, setSkillEnabled } from "../api/skills-api";
import { chatModelIsResident } from "../components/chat-model-notice-switch";
import { useModelChoices } from "../hooks/use-model-choices";
import { externalModelLabel } from "../lib/external-model-label";
import { chatLocalModelOptions } from "../local-model-options";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import { useSkillsSwitchStore } from "../stores/skills-switch-store";
import { saveStoredChatThread } from "../utils/chat-history-storage";
import { type AgentRecord, planAgentStart } from "./agent-start";
import { useChatAgentsStore } from "./agents-store";

export function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export function modelLabel(id: string): string {
  return externalModelLabel(id) ?? modelDisplayName(id);
}

export function usableSkills(skills: readonly SkillRecord[]): SkillRecord[] {
  return skills.filter((skill) => skill.valid && !skill.shadowed);
}

async function downloadedQuants(modelId: string): Promise<string[]> {
  const listing = await listGgufVariants(modelId, undefined, { localOnly: true }).catch(() => null);
  if (!listing) return [];
  return listing.variants
    .filter((variant) => variant.downloaded)
    .map((variant) => variant.quant)
    .sort((a, b) => Number(b === listing.default_variant) - Number(a === listing.default_variant));
}

export function useModelsListed(): boolean | null {
  const { localModels, cachedGguf } = useDeviceInventorySources(["localModels", "cachedGguf"]);
  if (localModels.ready && cachedGguf.ready) return true;
  return localModels.loading || cachedGguf.loading ? null : false;
}


export function useStartAgentChat(): {
  modelMissing: (modelId: string | null) => boolean;
  start: (agent: AgentRecord) => Promise<boolean>;
} {
  const t = useT();
  const navigate = useNavigate();
  const models = useModelChoices(null);
  const listed = useModelsListed();
  const { refreshIfOlderThan } = useDeviceInventorySources(["localModels", "cachedGguf"]);

  useEffect(() => {
    void refreshIfOlderThan(INVENTORY_FRESHNESS_WINDOW_MS);
  }, [refreshIfOlderThan]);

  const isConnectionModel = (modelId: string) =>
    models.connections.some((group) => group.models.some((option) => option.id === modelId));
  const modelMissing = (modelId: string | null) =>
    listed === true &&
    modelId !== null &&
    !models.local.some((option) => option.id === modelId) &&
    !isConnectionModel(modelId);

  // Never from the list alone: it can be a scan behind, and loading a model that is gone downloads it.
  async function modelToLoad(agent: AgentRecord) {
    const model = agent.modelId;
    if (!model) return null;
    if (isConnectionModel(model)) return { model, ggufVariant: null };
    const quants = await downloadedQuants(model);
    if (quants.length > 0) {
      const loaded = useChatRuntimeStore.getState();
      const wanted = [
        agent.modelGgufVariant,
        loaded.params.checkpoint === model ? loaded.activeGgufVariant : null,
      ].find((quant) => quant && quants.includes(quant));
      return { model, ggufVariant: wanted ?? quants[0] };
    }
    await refreshIfOlderThan(INVENTORY_FRESHNESS_WINDOW_MS);
    const device = useDeviceInventoryStore.getState();
    const onDevice =
      !device.cachedGguf.rows.some((repo) => repo.repo_id === model) &&
      chatLocalModelOptions(device.localModels.rows).some((option) => option.id === model);
    return onDevice ? { model, ggufVariant: agent.modelGgufVariant } : null;
  }

  async function start(agent: AgentRecord): Promise<boolean> {
    const modelGone = agent.modelId
      ? t("agentsPage.modelUnavailable", { model: modelLabel(agent.modelId) })
      : null;
    try {
      const [liveServers, liveSkills] = await Promise.all([listMcpServers(), listSkills(true)]);
      const plan = planAgentStart(agent, liveServers, usableSkills(liveSkills));
      await Promise.all([
        ...plan.mcpUpdates.map((update) => updateMcpServer(update.id, { isEnabled: update.isEnabled })),
        ...plan.skillUpdates.map((update) => setSkillEnabled(update.name, update.enabled)),
      ]);
      if (agent.skills.length > plan.missingSkills.length) {
        useSkillsSwitchStore.getState().setOn(true);
      }
      const { model, ggufVariant } = (await modelToLoad(agent)) ?? { model: null, ggufVariant: null };
      const threadId = crypto.randomUUID();
      await saveStoredChatThread({
        id: threadId,
        title: agent.name,
        modelType: "base",
        modelId: model ?? "",
        modelGgufVariant: ggufVariant,
        archived: false,
        createdAt: Date.now(),
        settings: plan.settings,
      });
      const loaded = useChatRuntimeStore.getState();
      useChatAgentsStore.setState({
        pendingModelLoad:
          model &&
          !chatModelIsResident(
            { modelId: model, ggufVariant },
            loaded.params.checkpoint,
            loaded.activeGgufVariant,
          )
            ? { threadId, modelId: model, ggufVariant }
            : null,
      });
      const gone = [
        agent.modelId && !model ? modelGone : null,
        plan.missingMcpServerIds.length > 0
          ? t("agentsPage.mcpUnavailable", { count: plan.missingMcpServerIds.length })
          : null,
        plan.missingSkills.length > 0
          ? t("agentsPage.skillUnavailable", { count: plan.missingSkills.length })
          : null,
      ].filter(Boolean);
      if (gone.length > 0) {
        toast.warning(t("agentsPage.startedWithout", { name: agent.name }), {
          description: gone.join(" "),
        });
      }
      void navigate({ to: "/chat", search: { thread: threadId } });
      return true;
    } catch (error) {
      toast.error(t("agentsPage.startFailed"), { description: errorText(error) });
      return false;
    }
  }

  return { modelMissing, start };
}
