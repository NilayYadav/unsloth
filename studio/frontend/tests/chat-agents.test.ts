// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  agentCopyName,
  agentFileName,
  agentFileText,
  parseAgentFile,
  uniqueAgentName,
} from "../src/features/chat/agents/agent-file.ts";
import {
  type AgentRecord,
  planAgentStart,
} from "../src/features/chat/agents/agent-start.ts";
import { normalizePlusMenuOrder } from "../src/features/chat/stores/plus-menu-prefs-store.ts";
import { sanitizeThreadScopedSettings } from "../src/features/chat/utils/thread-scoped-settings.ts";

function agent(overrides: Partial<AgentRecord> = {}): AgentRecord {
  return {
    id: "agent-1",
    name: "Researcher",
    description: "",
    modelId: "unsloth/Qwen3-0.6B-GGUF",
    modelGgufVariant: "Q4_K_M",
    systemPrompt: "Cite every claim.",
    tools: { webSearch: true, code: false, images: false },
    mcpServerIds: ["docs"],
    skills: ["pdf"],
    createdAt: 1,
    updatedAt: 1,
    ...overrides,
  };
}

const SERVERS = [
  { id: "docs", is_enabled: false },
  { id: "search", is_enabled: true },
  { id: "files", is_enabled: false },
];
const SKILLS = [
  { name: "pdf", enabled: false },
  { name: "slides", enabled: true },
];

test("a chat started from an agent runs with the agent's prompt and tools", () => {
  const { settings } = planAgentStart(agent(), SERVERS, SKILLS);

  assert.deepEqual(settings, {
    systemPrompt: "Cite every claim.",
    toolsEnabled: true,
    codeToolsEnabled: false,
    imageToolsEnabled: false,
    mcpEnabledForChat: true,
  });
  assert.deepEqual(sanitizeThreadScopedSettings(settings), settings);
});

test("an agent turns on exactly the MCP servers and skills it picked", () => {
  const plan = planAgentStart(agent(), SERVERS, SKILLS);

  assert.deepEqual(plan.mcpUpdates, [
    { id: "docs", isEnabled: true },
    { id: "search", isEnabled: false },
  ]);
  assert.deepEqual(plan.skillUpdates, [
    { name: "pdf", enabled: true },
    { name: "slides", enabled: false },
  ]);
});

test("an agent that picks no MCP servers or skills leaves them as they are", () => {
  const plan = planAgentStart(
    agent({ mcpServerIds: [], skills: [] }),
    SERVERS,
    SKILLS,
  );

  assert.deepEqual(plan.mcpUpdates, []);
  assert.deepEqual(plan.skillUpdates, []);
  assert.equal(plan.settings.mcpEnabledForChat, false);
});

test("an MCP server or skill removed since the agent was saved is reported and skipped", () => {
  const plan = planAgentStart(
    agent({ mcpServerIds: ["docs", "gone"], skills: ["pdf", "retired"] }),
    SERVERS,
    SKILLS,
  );

  assert.deepEqual(plan.missingMcpServerIds, ["gone"]);
  assert.deepEqual(plan.missingSkills, ["retired"]);
  assert.deepEqual(plan.mcpUpdates, [
    { id: "docs", isEnabled: true },
    { id: "search", isEnabled: false },
  ]);
});

test("an agent whose every MCP server is gone starts the chat with MCP off and changes nothing", () => {
  const plan = planAgentStart(agent({ mcpServerIds: ["gone"] }), SERVERS, SKILLS);

  assert.equal(plan.settings.mcpEnabledForChat, false);
  assert.deepEqual(plan.mcpUpdates, []);
  assert.deepEqual(plan.missingMcpServerIds, ["gone"]);
});

test("an exported agent imports back with the same setup and none of its ids or dates", () => {
  const source = agent({ description: "Reads papers", tools: { webSearch: true, code: true, images: false } });
  const { id: _id, createdAt: _createdAt, updatedAt: _updatedAt, ...setup } = source;

  assert.deepEqual(parseAgentFile(agentFileText(source)), setup);
  assert.equal(agentFileName(agent({ name: "Notes: a/b?" })), "Notes_ a_b_.agent.json");
});

test("a file that is not an exported agent is refused", () => {
  const valid = JSON.parse(agentFileText(agent()));

  assert.equal(parseAgentFile("not json"), null);
  assert.equal(parseAgentFile("[]"), null);
  assert.equal(parseAgentFile(JSON.stringify({ name: "Researcher" })), null);
  assert.equal(parseAgentFile(JSON.stringify({ ...valid, kind: "something-else" })), null);
  assert.equal(parseAgentFile(JSON.stringify({ ...valid, version: 2 })), null);
  assert.equal(parseAgentFile(JSON.stringify({ ...valid, agent: { ...valid.agent, name: "  " } })), null);
});

test("an imported file with wrong or oversized fields is cut down to a setup the server accepts", () => {
  const setup = parseAgentFile(
    JSON.stringify({
      kind: "unsloth-studio-agent",
      version: 1,
      agent: {
        id: "from-another-install",
        name: `  ${"n".repeat(300)}  `,
        description: 7,
        modelId: "",
        modelGgufVariant: "Q4_K_M",
        systemPrompt: null,
        tools: { webSearch: "yes", code: true },
        mcpServerIds: ["docs", "docs", 3, "", "x".repeat(129)],
        skills: "pdf",
      },
    }),
  );

  assert.deepEqual(setup, {
    name: "n".repeat(200),
    description: "",
    modelId: null,
    modelGgufVariant: null,
    systemPrompt: "",
    tools: { webSearch: false, code: true, images: false },
    mcpServerIds: ["docs"],
    skills: [],
  });
});

test("an imported or copied agent never takes a name that is in use", () => {
  assert.equal(uniqueAgentName("Researcher", ["Coder"]), "Researcher");
  assert.equal(uniqueAgentName("Researcher", ["researcher ", "Researcher 2"]), "Researcher 3");
  assert.equal(agentCopyName("Researcher", ["Researcher"]), "Researcher copy");
  assert.equal(
    agentCopyName("Researcher", ["Researcher", "Researcher copy"]),
    "Researcher copy 2",
  );
  assert.equal(agentCopyName("n".repeat(200), []).length, 200);
  assert.equal(uniqueAgentName("n".repeat(200), ["n".repeat(200)]), `${"n".repeat(198)} 2`);
});

test("a + menu order saved before Agents existed gains it once", () => {
  const order = normalizePlusMenuOrder(["projects", "skills"]);

  assert.equal(order.filter((id) => id === "agents").length, 1);
  assert.equal(order.indexOf("agents"), order.indexOf("projects") + 1);
});
