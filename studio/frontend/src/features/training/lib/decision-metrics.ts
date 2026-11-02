// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type DecisionConfusion = Record<string, Record<string, number>>;

export interface DecisionMetrics {
  decisions: number;
  accuracy_base: number | null;
  accuracy: number | null;
  ece_base: number | null;
  ece: number | null;
  questions_total: number;
  questions: { name: string; confusion: DecisionConfusion }[];
}

export interface ConfusionRow {
  option: string;
  counts: number[];
  precision: number | null;
  recall: number | null;
}

export function confusionTable(confusion: DecisionConfusion): {
  options: string[];
  rows: ConfusionRow[];
} {
  const options = [
    ...new Set(
      Object.entries(confusion).flatMap(([gold, predicted]) => [
        gold,
        ...Object.keys(predicted),
      ]),
    ),
  ].sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
  const counts = options.map((gold) =>
    options.map((predicted) => confusion[gold]?.[predicted] ?? 0),
  );
  const rows = options.map((option, i) => {
    const hits = counts[i][i];
    const expected = counts[i].reduce((sum, count) => sum + count, 0);
    const answered = counts.reduce((sum, row) => sum + row[i], 0);
    return {
      option,
      counts: counts[i],
      precision: answered ? hits / answered : null,
      recall: expected ? hits / expected : null,
    };
  });
  return { options, rows };
}
