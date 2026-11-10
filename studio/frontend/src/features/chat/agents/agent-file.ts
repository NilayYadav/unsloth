// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { AgentRecord } from "./agent-start.ts";

export const AGENT_FILE_KIND = "unsloth-studio-agent";
export const AGENT_FILE_MAX_BYTES = 1_000_000;
const NAME_MAX = 200;

export type AgentSetup = Omit<AgentRecord, "id" | "createdAt" | "updatedAt">;

function text(value: unknown, max: number): string {
  return typeof value === "string" ? value.slice(0, max) : "";
}

function idList(value: unknown, max: number): string[] {
  if (!Array.isArray(value)) return [];
  const ids = value.filter(
    (entry): entry is string => typeof entry === "string" && entry.length > 0 && entry.length <= 128,
  );
  return [...new Set(ids)].slice(0, max);
}

export function agentFileText(agent: AgentRecord): string {
  const setup: AgentSetup = {
    name: agent.name,
    description: agent.description,
    modelId: agent.modelId,
    modelGgufVariant: agent.modelGgufVariant,
    systemPrompt: agent.systemPrompt,
    tools: agent.tools,
    mcpServerIds: agent.mcpServerIds,
    skills: agent.skills,
  };
  return `${JSON.stringify({ kind: AGENT_FILE_KIND, version: 1, agent: setup }, null, 2)}\n`;
}

export function agentFileName(agent: AgentRecord): string {
  const base = agent.name.replace(/[\\/:*?"<>|]/g, "_").trim().slice(0, 80) || "agent";
  return `${base}.agent.json`;
}

export function parseAgentFile(content: string): AgentSetup | null {
  let file: unknown;
  try {
    file = JSON.parse(content);
  } catch {
    return null;
  }
  if (typeof file !== "object" || file === null) return null;
  const { kind, version, agent } = file as Record<string, unknown>;
  if (kind !== AGENT_FILE_KIND || version !== 1 || typeof agent !== "object" || agent === null) {
    return null;
  }
  const raw = agent as Record<string, unknown>;
  const name = text(raw.name, 10_000).trim().slice(0, NAME_MAX);
  if (!name) return null;
  const tools = (typeof raw.tools === "object" && raw.tools !== null ? raw.tools : {}) as Record<
    string,
    unknown
  >;
  const modelId = text(raw.modelId, 2000) || null;
  return {
    name,
    description: text(raw.description, 2000),
    modelId,
    modelGgufVariant: modelId ? text(raw.modelGgufVariant, 200) || null : null,
    systemPrompt: text(raw.systemPrompt, Number.MAX_SAFE_INTEGER),
    tools: {
      webSearch: tools.webSearch === true,
      code: tools.code === true,
      images: tools.images === true,
    },
    mcpServerIds: idList(raw.mcpServerIds, 100),
    skills: idList(raw.skills, 200),
  };
}

export function uniqueAgentName(name: string, taken: readonly string[]): string {
  const used = new Set(taken.map((entry) => entry.trim().toLowerCase()));
  if (!used.has(name.toLowerCase())) return name;
  for (let index = 2; ; index += 1) {
    const suffix = ` ${index}`;
    const candidate = `${name.slice(0, NAME_MAX - suffix.length)}${suffix}`;
    if (!used.has(candidate.toLowerCase())) return candidate;
  }
}

export function agentCopyName(name: string, taken: readonly string[]): string {
  const suffix = " copy";
  return uniqueAgentName(`${name.slice(0, NAME_MAX - suffix.length)}${suffix}`, taken);
}
