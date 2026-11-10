// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export interface AgentTools {
  webSearch: boolean;
  code: boolean;
  images: boolean;
}

export interface AgentRecord {
  id: string;
  name: string;
  description: string;
  modelId: string | null;
  modelGgufVariant: string | null;
  systemPrompt: string;
  tools: AgentTools;
  mcpServerIds: string[];
  skills: string[];
  createdAt: number;
  updatedAt: number;
}

export interface AgentStartPlan {
  settings: {
    systemPrompt: string;
    toolsEnabled: boolean;
    codeToolsEnabled: boolean;
    imageToolsEnabled: boolean;
    mcpEnabledForChat: boolean;
  };
  mcpUpdates: { id: string; isEnabled: boolean }[];
  skillUpdates: { name: string; enabled: boolean }[];
  missingMcpServerIds: string[];
  missingSkills: string[];
}

// MCP servers and skills are on or off for every chat, so an agent that picks none leaves them alone.
export function planAgentStart(
  agent: AgentRecord,
  servers: readonly { id: string; is_enabled: boolean }[],
  skills: readonly { name: string; enabled: boolean }[],
): AgentStartPlan {
  const serverIds = new Set(servers.map((server) => server.id));
  const wantedServers = new Set(
    agent.mcpServerIds.filter((id) => serverIds.has(id)),
  );
  const skillNames = new Set(skills.map((skill) => skill.name));
  const wantedSkills = new Set(
    agent.skills.filter((name) => skillNames.has(name)),
  );
  return {
    settings: {
      systemPrompt: agent.systemPrompt,
      toolsEnabled: agent.tools.webSearch,
      codeToolsEnabled: agent.tools.code,
      imageToolsEnabled: agent.tools.images,
      mcpEnabledForChat: wantedServers.size > 0,
    },
    mcpUpdates:
      wantedServers.size === 0
        ? []
        : servers
            .filter((server) => server.is_enabled !== wantedServers.has(server.id))
            .map((server) => ({
              id: server.id,
              isEnabled: wantedServers.has(server.id),
            })),
    skillUpdates:
      wantedSkills.size === 0
        ? []
        : skills
            .filter((skill) => skill.enabled !== wantedSkills.has(skill.name))
            .map((skill) => ({
              name: skill.name,
              enabled: wantedSkills.has(skill.name),
            })),
    missingMcpServerIds: agent.mcpServerIds.filter((id) => !serverIds.has(id)),
    missingSkills: agent.skills.filter((name) => !skillNames.has(name)),
  };
}
