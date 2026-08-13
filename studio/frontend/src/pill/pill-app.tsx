// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useLayoutEffect, useRef, type ReactElement } from "react";
import { useT } from "@/i18n";
import { fetchPillSettings } from "./api";
import { ResultView } from "./components/result-view";
import { VerbRow } from "./components/verb-row";
import {
  pillDismiss,
  pillGetCapture,
  pillResize,
  pillServerPort,
  type PillSelectionPayload,
} from "./native";
import { usePillStore } from "./pill-state";
import { cancelActiveRun, runVerb } from "./run-action";

function handleSelection(payload: PillSelectionPayload): void {
  cancelActiveRun();
  usePillStore.getState().showSelection(payload);
  // Warm the settings cache for the first verb run.
  void fetchPillSettings().catch(() => undefined);
}

export function PillApp(): ReactElement | null {
  const t = useT();
  const phase = usePillStore((s) => s.phase);
  const selection = usePillStore((s) => s.selection);
  const errorKey = usePillStore((s) => s.errorKey);
  const errorModel = usePillStore((s) => s.errorModel);
  const resultText = usePillStore((s) => s.resultText);
  const containerRef = useRef<HTMLDivElement>(null);
  const lastSizeRef = useRef({ width: 0, height: 0 });

  useEffect(() => {
    let disposed = false;
    const cleanups: Array<() => void> = [];

    void (async () => {
      const { isTauri } = await import("@/lib/api-base");
      if (!isTauri) return;
      const { listen } = await import("@tauri-apps/api/event");
      const unlistenSelection = await listen<PillSelectionPayload>(
        "pill://selection",
        (event) => handleSelection(event.payload),
      );
      const unlistenHide = await listen("pill://hide", () => {
        cancelActiveRun();
        usePillStore.getState().reset();
      });
      const unlistenPort = await listen<number>("server-port", (event) => {
        void import("@/lib/api-base").then(({ setApiBase }) =>
          setApiBase(event.payload),
        );
      });
      if (disposed) {
        unlistenSelection();
        unlistenHide();
        unlistenPort();
        return;
      }
      cleanups.push(unlistenSelection, unlistenHide, unlistenPort);

      // The server-port broadcast may predate this listener; pull the current
      // port, falling back to the value the main window persisted (externally
      // attached backends never reach the Rust-side state).
      let port = await pillServerPort().catch(() => null);
      if (port == null) {
        const stored = window.localStorage.getItem("unsloth_backend_port");
        port = stored ? Number(stored) || null : null;
      }
      if (port != null) {
        const { setApiBase } = await import("@/lib/api-base");
        setApiBase(port);
      }

      // The window may have been shown before this listener attached (first
      // trigger racing webview startup) — pull the pending capture if any.
      const pending = await pillGetCapture().catch(() => null);
      if (!disposed && pending && usePillStore.getState().phase === "idle") {
        handleSelection(pending);
      }
    })();

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        cancelActiveRun();
        void pillDismiss();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    cleanups.push(() => window.removeEventListener("keydown", onKeyDown));

    return () => {
      disposed = true;
      for (const cleanup of cleanups) cleanup();
    };
  }, []);

  useLayoutEffect(() => {
    const node = containerRef.current;
    if (!node || phase === "idle") return;
    const rect = node.getBoundingClientRect();
    const width = Math.ceil(rect.width);
    const height = Math.ceil(rect.height);
    // Streaming updates land per token; a native resize per token saturates
    // the main thread and freezes the app. Only resize on real change.
    if (
      width === lastSizeRef.current.width &&
      height === lastSizeRef.current.height
    ) {
      return;
    }
    lastSizeRef.current = { width, height };
    void pillResize(width, height).catch(() => undefined);
  }, [phase, resultText, errorKey]);

  if (phase === "idle") return null;

  return (
    <div
      ref={containerRef}
      className="w-fit min-w-44 max-w-96 overflow-hidden rounded-xl border border-border/60 bg-popover/70 text-popover-foreground shadow-lg"
    >
      {phase === "actions" && selection && (
        <VerbRow
          selectionText={selection.text}
          onRun={(verb) => void runVerb(verb, selection.text)}
        />
      )}
      {(phase === "streaming" || phase === "result") && <ResultView />}
      {phase === "error" && (
        <div className="flex items-center gap-2 px-3 py-2">
          <span className="text-xs text-muted-foreground">
            {errorMessage(t, errorKey, errorModel)}
          </span>
          <button
            type="button"
            onClick={() => void pillDismiss()}
            className="rounded-md px-2 py-1 text-xs text-muted-foreground hover:bg-accent hover:text-accent-foreground"
          >
            {t("systemPill.pill.dismiss")}
          </button>
        </div>
      )}
    </div>
  );
}

function errorMessage(
  t: ReturnType<typeof useT>,
  errorKey: string | null,
  errorModel: string | null,
): string {
  switch (errorKey) {
    case "no-selection":
      return t("systemPill.pill.noSelection");
    case "secure-input":
      return t("systemPill.pill.secureInput");
    case "no-focused-element":
      return t("systemPill.pill.captureFailed");
    case "backendDown":
      return t("systemPill.pill.backendDown");
    case "signedOut":
      return t("systemPill.pill.signedOut");
    case "modelMissing":
      return t("systemPill.pill.modelMissing");
    case "loadFailed":
      return t("systemPill.pill.loadFailed", { model: errorModel ?? "" });
    case "stalled":
      return t("systemPill.pill.stalled");
    default:
      return t("systemPill.pill.captureFailed");
  }
}
