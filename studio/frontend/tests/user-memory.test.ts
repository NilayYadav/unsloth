// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { composeChatInstructions } from "../src/features/chat/utils/agents-md.ts";
import {
  MEMORY_CHAR_LIMIT,
  buildUserMemoryBlock,
  parseImportedMemories,
} from "../src/features/chat/utils/user-memory.ts";
import { readSrc } from "./helpers/kit.ts";

const ADAPTER = readSrc("features/chat/api/chat-adapter.ts");
const DIALOG = readSrc(
  "features/chat/prompt-storage/prompt-storage-dialog.tsx",
);

test("a pasted ChatGPT reply becomes one clean memory per line", () => {
  const reply = [
    "Here is everything I have saved about you:",
    "```",
    "- [2025-03-01] Has a dog named Biscuit",
    "* 2025-04-12: Works as a nurse in Lisbon",
    "• Prefers metric units",
    "1. Is learning Portuguese",
    "2) Likes short answers",
    "  ",
    "- has a dog   named biscuit",
    "All code examples should use TypeScript",
    "```",
    "This list is complete.",
  ].join("\n");
  assert.deepEqual(parseImportedMemories(reply), [
    "Has a dog named Biscuit",
    "Works as a nurse in Lisbon",
    "Prefers metric units",
    "Is learning Portuguese",
    "Likes short answers",
    "All code examples should use TypeScript",
  ]);
});

test("without a code block every line counts", () => {
  assert.deepEqual(
    parseImportedMemories(
      "- Likes tea\n\n- Lives in Lisbon\nThat's everything.",
    ),
    ["Likes tea", "Lives in Lisbon"],
  );
});

test("completeness trailers are dropped", () => {
  for (const trailer of [
    "That's everything.",
    "That’s everything I remember about you.",
    "That is all.",
    "This is the complete list.",
    "The list above is complete.",
    "Yes, this list is complete.",
    "Note: the list is not complete",
    "Complete: yes",
  ]) {
    assert.deepEqual(parseImportedMemories(`Likes tea\n${trailer}`), [
      "Likes tea",
    ]);
  }
});

test("the memory block lists each memory once and keeps the newest within the cap", () => {
  assert.equal(buildUserMemoryBlock([]), "");
  assert.equal(buildUserMemoryBlock(["  "]), "");
  assert.equal(
    buildUserMemoryBlock(["My dog is named Biscuit", "Lives in Lisbon"]),
    "<user_memory>\nNotes the user saved about themselves. Use them only when they matter for the request, and do not bring them up otherwise:\n- My dog is named Biscuit\n- Lives in Lisbon\n</user_memory>",
  );
  const newest = Array.from(
    { length: 10 },
    (_, i) => `${i} ${"x".repeat(990)}`,
  );
  const block = buildUserMemoryBlock(newest);
  assert.ok(block.length <= MEMORY_CHAR_LIMIT, String(block.length));
  assert.match(block, /^- 0 x/m);
  assert.doesNotMatch(block, /^- 9 x/m);
});

test("memory sits in the system prompt before the typed prompt, which keeps the last word", () => {
  const userMemory = buildUserMemoryBlock(["My dog is named Biscuit"]);
  assert.equal(
    composeChatInstructions({
      agentsMd: { text: "Be brief." },
      projectInstructions: "Cite sources.",
      userMemory,
      systemPrompt: "Always answer in French.",
    }),
    [
      "<agents_md>\nBe brief.\n</agents_md>",
      "<project_instructions>\nCite sources.\n</project_instructions>",
      userMemory,
      "Always answer in French.",
    ].join("\n\n"),
  );
  assert.equal(
    composeChatInstructions({
      projectInstructions: "",
      userMemory: "",
      systemPrompt: "",
    }),
    "",
  );
});

test("memory is read at send time, only for requests the model sees, and only while switched on", () => {
  const calls =
    ADAPTER.match(/await resolveChatInstructions\([\s\S]*?\);/g) ?? [];
  assert.equal(calls.length, 2, "chat send and deep research");
  for (const call of calls) {
    assert.match(call, /userMemory: true/);
  }
  assert.doesNotMatch(DIALOG, /userMemory/);
  const recount = ADAPTER.slice(
    ADAPTER.indexOf("export async function buildLocalTokenCountHistory"),
    ADAPTER.indexOf("export function buildLocalTokenCountReasoning"),
  );
  assert.match(recount, /resolveUserMemory\(\)/);
  const resolver = ADAPTER.slice(
    ADAPTER.indexOf("function resolveUserMemory"),
    ADAPTER.indexOf("export async function resolveChatInstructions"),
  );
  assert.match(resolver, /memoryEnabled/);
  assert.match(resolver, /entry\.enabled/);
  assert.doesNotMatch(ADAPTER, /systemPrompt:\s*[^,\n]*userMemory/);
});
