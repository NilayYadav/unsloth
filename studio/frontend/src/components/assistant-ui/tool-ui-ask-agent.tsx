// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Spinner } from "@/components/ui/spinner";
import { useToolAwaitingApproval } from "@/features/chat";
import { stringifyToolResult } from "@/lib/strip-ansi";
import {
  type ToolCallMessagePartComponent,
  useAuiState,
} from "@assistant-ui/react";
import { partsHaveNonEmptyText } from "@/components/assistant-ui/message-derived";
import { Robot01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ComponentProps, memo } from "react";
import { toolArgText } from "./tool-arg-text";
import {
  ToolFallbackContent,
  ToolFallbackRoot,
  ToolFallbackTrigger,
} from "./tool-fallback";
import { useToolActivityOpen } from "./use-tool-activity-open";
import { ScrollPane } from "./scroll-pane";

// Mirrors the result ask_agent returns: "<agent> (<model>) replied:\n\n<reply>".
const REPLY_RE = /^.+? \((.*)\) replied:\n\n([\s\S]*)$/;

const AgentIcon = (props: Omit<ComponentProps<typeof HugeiconsIcon>, "icon">) => (
  <HugeiconsIcon icon={Robot01Icon} strokeWidth={1.75} {...props} />
);

const AskAgentToolUIImpl: ToolCallMessagePartComponent = ({
  args,
  result,
  status,
  toolCallId,
}) => {
  const asked = (args ?? {}) as { agent?: unknown; task?: unknown; files?: unknown };
  const name = toolArgText(asked.agent) || "agent";
  const task = toolArgText(asked.task);
  const files = toolArgText(
    Array.isArray(asked.files) ? asked.files.map(toolArgText).join(", ") : asked.files,
  );
  const isRunning = status?.type === "running";
  const resultText = result == null ? "" : stringifyToolResult(result);
  const [, model = "", reply = resultText] = REPLY_RE.exec(resultText) ?? [];
  const hasText = useAuiState(({ message }) =>
    partsHaveNonEmptyText(message.content),
  );
  const awaitingApproval = useToolAwaitingApproval(toolCallId);
  const [open, setOpen] = useToolActivityOpen(isRunning, hasText);

  return (
    <ToolFallbackRoot
      open={open}
      onOpenChange={setOpen}
      awaitingApproval={awaitingApproval}
    >
      <ToolFallbackTrigger
        toolName={
          isRunning ? `Asking ${name}…` : model ? `Asked ${name} · ${model}` : `Asked ${name}`
        }
        status={status}
        icon={AgentIcon}
      />
      <ToolFallbackContent>
        <div className="flex flex-col gap-2">
          <ScrollPane
            className="rounded-lg bg-muted/50 p-2"
            scrollerClassName="max-h-40 overflow-auto whitespace-pre-wrap break-words text-xs"
          >
            {task}
            {files ? `\n\nFiles: ${files}` : ""}
          </ScrollPane>
          {isRunning ? (
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Spinner className="size-3.5" />
              <span>Waiting for {name}&hellip;</span>
            </div>
          ) : (
            reply && (
              <ScrollPane
                className="rounded-lg bg-muted/50 p-2"
                scrollerClassName="max-h-64 overflow-auto whitespace-pre-wrap break-words text-xs"
              >
                {reply}
              </ScrollPane>
            )
          )}
        </div>
      </ToolFallbackContent>
    </ToolFallbackRoot>
  );
};

export const AskAgentToolUI = memo(
  AskAgentToolUIImpl,
) as unknown as ToolCallMessagePartComponent;
AskAgentToolUI.displayName = "AskAgentToolUI";
