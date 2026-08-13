// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useState, type ReactElement } from "react";
import { useT } from "@/i18n";
import { usePillStore } from "../pill-state";
import { cancelActiveRun } from "../run-action";
import { pillDismiss, pillInsertBelow, pillReplaceSelection } from "../native";

async function copyToClipboard(text: string): Promise<void> {
  const { writeText } = await import("@tauri-apps/plugin-clipboard-manager");
  await writeText(text);
}

export function ResultView(): ReactElement {
  const t = useT();
  const phase = usePillStore((s) => s.phase);
  const selection = usePillStore((s) => s.selection);
  const resultText = usePillStore((s) => s.resultText);
  const loadingLabel = usePillStore((s) => s.loadingLabel);
  const loadingModel = usePillStore((s) => s.loadingModel);
  const applied = usePillStore((s) => s.applied);
  const [copied, setCopied] = useState(false);
  const [applyFailed, setApplyFailed] = useState(false);

  const streaming = phase === "streaming";
  const sessionId = selection?.sessionId ?? 0;
  const editable = selection?.editable ?? false;

  const handleReplace = async () => {
    try {
      await pillReplaceSelection(sessionId, resultText);
      usePillStore.getState().markApplied();
    } catch {
      setApplyFailed(true);
      await copyToClipboard(resultText).catch(() => undefined);
    }
  };

  const handleInsertBelow = async () => {
    try {
      await pillInsertBelow(sessionId, resultText);
      usePillStore.getState().markApplied();
    } catch {
      setApplyFailed(true);
      await copyToClipboard(resultText).catch(() => undefined);
    }
  };

  const handleCopy = async () => {
    await copyToClipboard(resultText).catch(() => undefined);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  return (
    <div className="flex max-h-72 flex-col">
      {streaming && resultText === "" && (
        <div className="flex items-center gap-2 px-3 py-2 text-xs text-muted-foreground">
          <span className="inline-block size-3 animate-spin rounded-full border border-current border-t-transparent" />
          {loadingLabel === "loading-model"
            ? t("systemPill.pill.loadingModel", { model: loadingModel ?? "" })
            : t("systemPill.pill.thinking")}
        </div>
      )}
      {resultText !== "" && (
        <div className="min-h-0 flex-1 overflow-y-auto whitespace-pre-wrap px-3 py-2 text-xs leading-relaxed text-foreground">
          {resultText}
        </div>
      )}
      <div className="flex items-center gap-1 border-t border-border/60 px-1.5 py-1">
        {streaming ? (
          <FooterButton
            label={t("systemPill.pill.cancel")}
            onClick={() => cancelActiveRun()}
          />
        ) : (
          <>
            {applied ? (
              <span className="px-2 py-1 text-xs text-muted-foreground">
                {t("systemPill.pill.replaced")}
              </span>
            ) : (
              <>
                {editable && !applyFailed && (
                  <FooterButton
                    label={t("systemPill.pill.replace")}
                    primary={true}
                    onClick={handleReplace}
                  />
                )}
                {editable && !applyFailed && (
                  <FooterButton
                    label={t("systemPill.pill.insertBelow")}
                    onClick={handleInsertBelow}
                  />
                )}
              </>
            )}
            {applyFailed && (
              <span className="px-2 py-1 text-xs text-destructive">
                {t("systemPill.pill.applyFailed")}
              </span>
            )}
            <FooterButton
              label={copied ? t("systemPill.pill.copied") : t("systemPill.pill.copy")}
              onClick={handleCopy}
            />
            <div className="flex-1" />
            <FooterButton
              label={t("systemPill.pill.dismiss")}
              onClick={() => void pillDismiss()}
            />
          </>
        )}
      </div>
    </div>
  );
}

function FooterButton({
  label,
  onClick,
  primary = false,
}: {
  label: string;
  onClick: () => void | Promise<void>;
  primary?: boolean;
}): ReactElement {
  return (
    <button
      type="button"
      onClick={() => void onClick()}
      className={
        primary
          ? "rounded-md bg-primary px-2.5 py-1 text-xs font-medium text-primary-foreground hover:bg-primary/90"
          : "rounded-md px-2 py-1 text-xs text-muted-foreground hover:bg-accent hover:text-accent-foreground"
      }
    >
      {label}
    </button>
  );
}
