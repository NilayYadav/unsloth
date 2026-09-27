// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

// The settings API module reaches authFetch through the auth barrel. See helpers/auth-stub.mjs.
register("./helpers/settings-api-resolver.mjs", import.meta.url);

const { readRouterDecision, routerDecisionChunk } = await import(
  "../src/features/chat/api/router-decision.ts"
);
const { normalizeChatGenerationChunkPayload } = await import(
  "../src/features/chat/api/chat-generation-api.ts"
);
const { generationChunkCountsTowardTiming, recoveredGenerationFinalMetadata } =
  await import("../src/features/chat/utils/chat-generation-recovery.ts");
const store = await import(
  "../src/features/chat/stores/auto-router-selection.ts"
);
const { autoRouterModelFromCandidate } = await import(
  "../src/features/settings/api/auto-router.ts"
);

const FRAME = {
  type: "router_decision",
  model: "unsloth/Qwen3-4B-GGUF",
  reason: "Looks like code",
  task: "code",
};

test("a replayed router_decision frame is recognised, not treated as text", () => {
  const chunk = normalizeChatGenerationChunkPayload(FRAME);
  assert.deepEqual(readRouterDecision(chunk), {
    model: "unsloth/Qwen3-4B-GGUF",
    reason: "Looks like code",
    task: "code",
  });
  assert.equal("choices" in chunk, false);
  assert.equal(generationChunkCountsTowardTiming(chunk), false);
});

test("the live and replayed paths build the same decision", () => {
  assert.deepEqual(
    routerDecisionChunk(FRAME),
    normalizeChatGenerationChunkPayload(FRAME),
  );
  assert.equal(readRouterDecision({ choices: [] }), null);
  assert.equal(readRouterDecision(routerDecisionChunk({ reason: "x" })), null);
});

test("a recovered Auto reply records the model that answered it", () => {
  const metadata = recoveredGenerationFinalMetadata({
    current: {},
    run: {
      id: "run",
      requestPayload: { model: "auto" },
      createdAt: 1,
      startedAt: 1,
      completedAt: 2,
    },
    totalChunks: 1,
    router: { model: "unsloth/Qwen3-4B-GGUF", reason: "Looks like code" },
  });
  const details = metadata.responseDetails as Record<string, unknown>;
  assert.equal(details.modelId, "auto");
  assert.equal(details.routerModel, "unsloth/Qwen3-4B-GGUF");
  assert.equal(details.responseModelId, "unsloth/Qwen3-4B-GGUF");
  assert.equal(details.routerReason, "Looks like code");
});

test("pins and choices are only kept for real thread ids", () => {
  store.setAutoRouterPin(null, "model-a");
  store.setAutoRouterPin(undefined, "model-a");
  store.recordAutoRouterChoice(null, "model-a");
  assert.deepEqual(store.autoRouterSelection().threads, {});
  assert.equal(store.autoRouterThread(null).pin, null);

  store.setAutoRouterPin("thread-1", "model-a");
  assert.equal(store.autoRouterThread("thread-1").pin, "model-a");
  assert.equal(store.autoRouterThread("thread-2").pin, null);
});

test("saved settings decide whether Auto is offered and can run tools", () => {
  store.setAutoRouterEnabled(true);
  store.setAutoRouterSettings([{ tools: false }, { tools: true }]);
  assert.equal(store.autoRouterSelection().configured, true);
  assert.equal(store.autoRouterSelection().toolsCapable, true);

  store.setAutoRouterEnabled(false);
  store.setAutoRouterEnabled(true);
  assert.equal(store.autoRouterSelection().toolsCapable, true);

  store.setAutoRouterSettings([]);
  assert.equal(store.autoRouterSelection().configured, false);
  assert.equal(store.autoRouterSelection().enabled, false);
});

test("a candidate's detected capabilities prefill the router entry", () => {
  assert.deepEqual(
    autoRouterModelFromCandidate({
      id: "m",
      name: "M",
      vision: false,
      tools: true,
      context_length: 32768,
      tasks: ["vision", "code", "unknown-task"],
    }),
    {
      id: "m",
      tasks: ["vision", "code"],
      vision: true,
      tools: true,
      context_length: 32768,
    },
  );
  assert.deepEqual(
    autoRouterModelFromCandidate({
      id: "n",
      name: "N",
      vision: false,
      tools: false,
      context_length: null,
      tasks: [],
    }).tasks,
    ["general"],
  );
});
