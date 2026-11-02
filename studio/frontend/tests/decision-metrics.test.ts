// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { confusionTable } from "../src/features/training/lib/decision-metrics.ts";

test("per-class scores come from the confusion counts", () => {
  const { options, rows } = confusionTable({
    spam: { spam: 8, ham: 2 },
    ham: { spam: 4, ham: 6 },
  });
  assert.deepEqual(options, ["ham", "spam"]);
  assert.deepEqual(rows[0].counts, [6, 4]);
  assert.deepEqual(rows[1].counts, [2, 8]);
  assert.equal(rows[1].recall, 0.8);
  assert.equal(rows[1].precision, 8 / 12);
});

test("a class the model answered but no gold named has no recall", () => {
  const { options, rows } = confusionTable({ a: { b: 3 } });
  assert.deepEqual(options, ["a", "b"]);
  assert.equal(rows[0].precision, null);
  assert.equal(rows[0].recall, 0);
  assert.equal(rows[1].precision, 0);
  assert.equal(rows[1].recall, null);
});

test("score levels sort as numbers", () => {
  assert.deepEqual(
    confusionTable({ "10": { "2": 1 }, "1": { "1": 1 } }).options,
    ["1", "2", "10"],
  );
});
