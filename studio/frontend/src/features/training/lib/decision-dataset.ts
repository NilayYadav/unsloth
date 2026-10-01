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

// The decision trainer reads JSON rows itself, so a format check that cannot parse them must not block the start.
export async function checkDecisionDatasetColumns(
  check: () => Promise<{ columns: readonly string[] } | null>,
): Promise<string[] | null> {
  let result: { columns: readonly string[] } | null;
  try {
    result = await check();
  } catch {
    return [];
  }
  return result ? missingDecisionColumns(result.columns) : null;
}
