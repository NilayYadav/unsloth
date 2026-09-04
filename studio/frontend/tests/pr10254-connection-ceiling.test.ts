// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// PR 10254 A/B probe: the research request must carry the connection's resolved output ceiling.
import assert from "node:assert/strict";
import test from "node:test";

import { buildResearchInferenceRequest } from "../src/features/chat/research-inference-request.ts";

test("the research request carries the saved connection's output ceiling", () => {
  const request = buildResearchInferenceRequest({
    checkpoint: "local-model",
    external: {
      providerId: "conn-1",
      providerType: "gemini",
      modelId: "gemini-3.6-flash",
      maxOutputTokens: 32768,
    },
    temperature: 0.2,
    topP: 0.9,
    maxTokens: 4096,
  } as never);
  console.log(
    `PR10254_FRONTEND_FACT maxOutputTokens=${JSON.stringify(
      (request as Record<string, unknown>).maxOutputTokens,
    )}`,
  );
  assert.equal((request as Record<string, unknown>).maxOutputTokens, 32768);
});
