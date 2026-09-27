// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { OpenAIChatChunk } from "../types/api";

export interface RouterDecision {
  model: string;
  reason: string | null;
  task: string | null;
}

export function routerDecisionChunk(frame: object): OpenAIChatChunk {
  const { model, reason, task } = frame as Record<string, unknown>;
  return {
    _routerDecision: {
      model: typeof model === "string" ? model : "",
      reason: typeof reason === "string" && reason ? reason : null,
      task: typeof task === "string" && task ? task : null,
    },
  } as unknown as OpenAIChatChunk;
}

export function readRouterDecision(chunk: unknown): RouterDecision | null {
  if (!chunk || typeof chunk !== "object" || !("_routerDecision" in chunk)) return null;
  const decision = (chunk as { _routerDecision?: Partial<RouterDecision> })._routerDecision;
  if (!decision || typeof decision.model !== "string" || !decision.model) return null;
  return {
    model: decision.model,
    reason: typeof decision.reason === "string" ? decision.reason : null,
    task: typeof decision.task === "string" ? decision.task : null,
  };
}
