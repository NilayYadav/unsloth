// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ReactElement } from "react";
import { useT } from "@/i18n";
import { pillOpenAsk } from "../native";
import { pillVerbs, type PillVerb } from "../verbs";

export function VerbRow({
  selectionText,
  onRun,
}: {
  selectionText: string;
  onRun: (verb: PillVerb) => void;
}): ReactElement {
  const t = useT();

  return (
    <div className="flex items-center gap-1 px-1.5 py-1.5">
      {pillVerbs(t).map((verb) => (
        <button
          key={verb.id}
          type="button"
          onClick={() => onRun(verb)}
          className="rounded-lg px-2.5 py-1 text-xs font-medium text-foreground/90 transition-colors hover:bg-accent hover:text-accent-foreground"
        >
          {verb.label}
        </button>
      ))}
      <div className="mx-0.5 h-4 w-px bg-border/70" />
      <button
        type="button"
        onClick={() => void pillOpenAsk(selectionText)}
        className="rounded-lg px-2.5 py-1 text-xs font-medium text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground"
      >
        {t("systemPill.actions.ask")}
      </button>
    </div>
  );
}
