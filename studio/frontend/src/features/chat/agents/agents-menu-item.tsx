// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Robot01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate } from "@tanstack/react-router";
import { useEffect } from "react";
import {
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
} from "@/components/ui/dropdown-menu";
import { useT } from "@/i18n";
import { useChatAgentsStore } from "./agents-store";
import { useStartAgentChat } from "./use-start-agent-chat";

const MENU_AGENT_LIMIT = 6;

function AgentsMenuList() {
  const t = useT();
  const navigate = useNavigate();
  const agents = useChatAgentsStore((state) => state.agents);
  const refresh = useChatAgentsStore((state) => state.refresh);
  const { start } = useStartAgentChat();

  useEffect(() => {
    void refresh();
  }, [refresh]);

  return (
    <>
      {agents.length > 0 ? (
        agents.slice(0, MENU_AGENT_LIMIT).map((agent) => (
          <DropdownMenuItem key={agent.id} onSelect={() => void start(agent)}>
            <HugeiconsIcon icon={Robot01Icon} strokeWidth={2} />
            <span className="truncate">{agent.name}</span>
          </DropdownMenuItem>
        ))
      ) : (
        <DropdownMenuItem disabled={true}>{t("agentsPage.emptyTitle")}</DropdownMenuItem>
      )}
      <DropdownMenuSeparator />
      <DropdownMenuItem onSelect={() => void navigate({ to: "/agents" })}>
        {t("agentsPage.allAgents")}
      </DropdownMenuItem>
    </>
  );
}

export function AgentsMenuItem() {
  const t = useT();
  return (
    <DropdownMenuSub>
      <DropdownMenuSubTrigger>
        <HugeiconsIcon icon={Robot01Icon} strokeWidth={2} />
        {t("shell.navigation.agents")}
      </DropdownMenuSubTrigger>
      <DropdownMenuSubContent className="unsloth-plus-menu w-[calc(232px*var(--ui-space-scale,1))]">
        <AgentsMenuList />
      </DropdownMenuSubContent>
    </DropdownMenuSub>
  );
}
