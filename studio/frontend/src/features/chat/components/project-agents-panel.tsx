// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Switch } from "@/components/ui/switch";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { useState, type ReactElement } from "react";
import { updateChatProjectSubagents } from "../hooks/use-chat-projects";
import type { ProjectRecord, SubagentEffort } from "../types";

const EFFORT_OPTIONS: { value: SubagentEffort | null; label: string; hint: string }[] = [
  { value: null, label: "Auto", hint: "Planner picks per executor" },
  { value: "low", label: "Low", hint: "Fast, 6 tool calls" },
  { value: "medium", label: "Medium", hint: "Balanced, 12 tool calls" },
  { value: "high", label: "High", hint: "Deep thinking, 25 tool calls" },
];

export function ProjectAgentsPanel({
  project,
}: {
  project: ProjectRecord | null | undefined;
}): ReactElement {
  const [saving, setSaving] = useState(false);
  const enabled = Boolean(project?.subagentsEnabled);
  const effort = project?.subagentEffort ?? null;

  async function save(patch: {
    subagentsEnabled?: boolean;
    subagentEffort?: SubagentEffort | null;
  }): Promise<void> {
    if (!project) return;
    setSaving(true);
    try {
      await updateChatProjectSubagents(project.id, patch);
    } catch (error) {
      toast.error("Failed to update sub-agents", {
        description: error instanceof Error ? error.message : undefined,
      });
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="mt-8 flex flex-col gap-6 rounded-[16px] border border-border/70 px-6 py-6">
      <div className="flex items-start justify-between gap-6">
        <div className="space-y-1">
          <p className="text-ui-14 font-semibold text-foreground">Sub-agents</p>
          <p className="max-w-md text-sm text-muted-foreground">
            The model acts as a planner and can spawn executor agents for parts
            of a task. Each executor gets its own tool scope (a subset of the
            tools enabled in the chat) and a reasoning effort, and works in this
            project's workspace.
          </p>
        </div>
        <Switch
          checked={enabled}
          disabled={!project || saving}
          onCheckedChange={(checked) => void save({ subagentsEnabled: checked })}
          aria-label="Enable sub-agents"
        />
      </div>

      <div className={cn("space-y-3", !enabled && "opacity-50")}>
        <p className="text-sm font-medium text-foreground">Executor effort</p>
        <div role="radiogroup" className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          {EFFORT_OPTIONS.map((option) => {
            const active = option.value === effort;
            return (
              <button
                key={option.label}
                type="button"
                role="radio"
                aria-checked={active}
                disabled={!enabled || saving}
                onClick={() => void save({ subagentEffort: option.value })}
                data-active={active}
                className="flex flex-col items-start gap-0.5 rounded-[12px] border px-3 py-2 text-left transition-colors disabled:cursor-not-allowed data-[active=true]:border-foreground/40 data-[active=true]:bg-muted data-[active=false]:border-border/70 data-[active=false]:hover:bg-nav-surface-hover"
              >
                <span className="text-ui-14 font-semibold text-foreground">
                  {option.label}
                </span>
                <span className="text-[12px] text-muted-foreground">{option.hint}</span>
              </button>
            );
          })}
        </div>
        <p className="text-[12px] text-muted-foreground">
          Executors can only use tools that are turned on in the chat (Search,
          Code, Docs, MCP). Works with local models that support tool calling.
        </p>
      </div>
    </div>
  );
}
