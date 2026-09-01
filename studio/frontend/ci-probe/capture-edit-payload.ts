// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Runs the shipped updateThreadMessage against a reply the durable generation path produced and
// writes the exact record it would PUT to disk, so the Python half of this probe can hand that
// record to the real studio_db instead of a hand-written guess at it.

import { writeFileSync } from "node:fs";

import { MessageRepository } from "@assistant-ui/core/internal";
import * as researchSync from "../src/features/chat/utils/research-message-sync.ts";
import { loadWithStubs } from "../tests/helpers/module-stubs.ts";

const RECORD_MODULE = loadWithStubs<Record<string, unknown>>(
  new URL(
    "../src/features/chat/utils/delete-thread-message.ts",
    import.meta.url,
  ),
  {
    "@assistant-ui/core/internal": { MessageRepository },
    "../api/chat-api": { listChatMessages: async () => [] },
    "./chat-history-storage": {
      ensureStoredChatThread: async () => {},
      syncStoredChatMessages: async (_t: string, r: unknown) => r,
    },
    "./research-message-sync": researchSync,
  },
);

const saved: Record<string, unknown>[] = [];
const module = loadWithStubs<{
  updateThreadMessage: (args: Record<string, unknown>) => Promise<unknown>;
}>(
  new URL(
    "../src/features/chat/utils/update-thread-message.ts",
    import.meta.url,
  ),
  {
    "../api/chat-api": {
      saveChatMessage: async (record: Record<string, unknown>) => {
        saved.push(record);
        return record;
      },
    },
    "./delete-thread-message": RECORD_MODULE,
  },
);

// Exactly what toThreadMessage hands back for a settled durable reply that hit the token
// limit: the run's ownership fields alongside the details the reply is displayed with.
const custom = {
  serverManaged: true,
  generationRunId: "run-1",
  generationSeq: 3,
  generationStatus: "completed",
  generationSettled: true,
  incomplete: { reason: "length" },
  timing: { tokensPerSecond: 42.5, durationMs: 1200 },
  contextUsage: { promptTokens: 900, contextLength: 4096 },
};

const exported = {
  headId: "assistant-1",
  messages: [
    {
      parentId: null,
      message: {
        id: "user-1",
        role: "user",
        content: [{ type: "text", text: "hello" }],
        createdAt: new Date(2),
      },
    },
    {
      parentId: "user-1",
      message: {
        id: "assistant-1",
        role: "assistant",
        content: [{ type: "text", text: "the original reply" }],
        createdAt: new Date(Number(process.env.PROBE_CREATED_AT)),
        metadata: { custom },
      },
    },
  ],
};

await module.updateThreadMessage({
  thread: { export: () => exported, import: () => {} },
  messageId: "assistant-1",
  remoteId: "thread-1",
  newText: "an edited reply",
  isIncognito: false,
});

if (saved.length !== 1) {
  console.error(`FAIL: the edit produced ${saved.length} saves, expected 1`);
  process.exit(1);
}
writeFileSync(process.argv[2], JSON.stringify(saved[0], null, 2));
console.log("captured PUT body:", JSON.stringify(saved[0]));
