// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export function missingDecisionColumns(columns: readonly string[]): string[] {
  const present = new Set(columns);
  const missing = ["state", "questions"].filter(
    (column) => !present.has(column),
  );
  if (!(present.has("gold") || present.has("answers"))) {
    missing.push("gold");
  }
  return missing;
}
