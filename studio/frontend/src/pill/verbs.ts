// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { useT } from "@/i18n";

export type PillVerb = {
  id: "proofread" | "rewrite" | "summarize";
  label: string;
  prompt: string;
};

export function pillVerbs(t: ReturnType<typeof useT>): PillVerb[] {
  return [
    {
      id: "proofread",
      label: t("systemPill.actions.proofread"),
      prompt: t("systemPill.actions.proofreadPrompt"),
    },
    {
      id: "rewrite",
      label: t("systemPill.actions.rewrite"),
      prompt: t("systemPill.actions.rewritePrompt"),
    },
    {
      id: "summarize",
      label: t("systemPill.actions.summarize"),
      prompt: t("systemPill.actions.summarizePrompt"),
    },
  ];
}
