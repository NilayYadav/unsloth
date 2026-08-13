// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import type { PillSelectionPayload } from "./native";

export type PillPhase = "idle" | "actions" | "streaming" | "result" | "error";

export type LoadingLabel = "thinking" | "loading-model";

type PillStore = {
  phase: PillPhase;
  selection: PillSelectionPayload | null;
  loadingLabel: LoadingLabel;
  loadingModel: string | null;
  resultText: string;
  errorKey: string | null;
  errorModel: string | null;
  applied: boolean;
  showSelection: (selection: PillSelectionPayload) => void;
  startStreaming: () => void;
  setLoadingLabel: (label: LoadingLabel, model?: string | null) => void;
  appendResult: (delta: string) => void;
  finishStreaming: () => void;
  markApplied: () => void;
  fail: (errorKey: string, model?: string | null) => void;
  reset: () => void;
};

export const usePillStore = create<PillStore>((set) => ({
  phase: "idle",
  selection: null,
  loadingLabel: "thinking",
  loadingModel: null,
  resultText: "",
  errorKey: null,
  errorModel: null,
  applied: false,
  showSelection: (selection) =>
    set({
      phase: selection.error ? "error" : "actions",
      selection,
      resultText: "",
      errorKey: selection.error,
      errorModel: null,
      applied: false,
    }),
  startStreaming: () =>
    set({
      phase: "streaming",
      loadingLabel: "thinking",
      loadingModel: null,
      resultText: "",
      errorKey: null,
      applied: false,
    }),
  setLoadingLabel: (label, model = null) =>
    set({ loadingLabel: label, loadingModel: model }),
  appendResult: (delta) =>
    set((state) => ({ resultText: state.resultText + delta })),
  finishStreaming: () => set({ phase: "result" }),
  markApplied: () => set({ applied: true }),
  fail: (errorKey, model = null) =>
    set({ phase: "error", errorKey, errorModel: model }),
  reset: () =>
    set({
      phase: "idle",
      selection: null,
      resultText: "",
      errorKey: null,
      errorModel: null,
      applied: false,
    }),
}));
