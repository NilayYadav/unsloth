// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import type { ToolCallMessagePartComponent } from "@assistant-ui/react";
import { BotIcon } from "lucide-react";
import { Spinner } from "@/components/ui/spinner";
import { memo } from "react";
import { Streamdown } from "streamdown";
import {
  ToolFallbackContent,
  ToolFallbackRoot,
  ToolFallbackTrigger,
} from "./tool-fallback";
import { isToolCallRunning, toolArgText } from "./tool-arg-text";
import { toolResultText } from "@/features/chat";

const REPORT_MARKER = "\n\nReport:\n";

function splitResult(result: string): { meta: string[]; report: string } {
  const idx = result.indexOf(REPORT_MARKER);
  if (idx < 0) return { meta: [], report: result };
  return {
    meta: result.slice(0, idx).split("\n").filter(Boolean),
    report: result.slice(idx + REPORT_MARKER.length),
  };
}

const SubagentToolUIImpl: ToolCallMessagePartComponent = ({
  args,
  result,
  status,
}) => {
  const a = (args ?? {}) as Record<string, unknown>;
  const label = toolArgText(a.name).trim() || "executor";
  const effort = toolArgText(a.reasoning_effort);
  const task = toolArgText(a.task);
  const tools = Array.isArray(a.tools) ? a.tools.map(toolArgText) : [];
  const isRunning = isToolCallRunning(status);
  const text = result == null ? "" : toolResultText(result);
  const { meta, report } = splitResult(text);

  return (
    <ToolFallbackRoot defaultOpen={isRunning}>
      <ToolFallbackTrigger
        toolName={`Sub-agent · ${label}${effort ? ` (${effort})` : ""}`}
        status={status}
        icon={BotIcon}
      />
      <ToolFallbackContent>
        <div className="space-y-2 border-l-2 border-muted-foreground/20 pl-2 text-sm">
          {task ? (
            <div>
              <span className="text-xs font-medium text-muted-foreground">task</span>
              <p className="mt-0.5 whitespace-pre-wrap break-words">{task}</p>
            </div>
          ) : null}
          {meta.length > 0 ? (
            <div className="space-y-0.5 text-xs text-muted-foreground">
              {meta.map((line) => (
                <p key={line}>{line}</p>
              ))}
            </div>
          ) : tools.length > 0 ? (
            <p className="text-xs text-muted-foreground">tools: {tools.join(", ")}</p>
          ) : null}
          {isRunning ? (
            <div className="flex items-center gap-2 text-muted-foreground">
              <Spinner className="size-3.5" />
              <span>Working&hellip;</span>
            </div>
          ) : report ? (
            <div>
              <span className="text-xs font-medium text-muted-foreground">report</span>
              <div className="mt-0.5 max-h-80 overflow-auto">
                <Streamdown mode="static">{report}</Streamdown>
              </div>
            </div>
          ) : null}
        </div>
      </ToolFallbackContent>
    </ToolFallbackRoot>
  );
};

export const SubagentToolUI = memo(
  SubagentToolUIImpl,
) as unknown as ToolCallMessagePartComponent;
SubagentToolUI.displayName = "SubagentToolUI";
