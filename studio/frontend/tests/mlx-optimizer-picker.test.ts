// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { DEFAULT_HYPERPARAMS, MLX_OPTIMIZER_OPTIONS, mlxEffectiveOptimizer } =
  await import("../src/config/training.ts");

test("the Mac picker shows the 8-bit AdamW that MLX trains by default", () => {
  assert.equal(
    mlxEffectiveOptimizer(DEFAULT_HYPERPARAMS.optimizerType),
    "adamw_8bit",
  );
  assert.ok(
    MLX_OPTIMIZER_OPTIONS.some((option) => option.value === "adamw_8bit"),
  );
});

test("CUDA-only optimizer names still read as AdamW on Mac", () => {
  assert.equal(mlxEffectiveOptimizer("paged_adamw_8bit"), "adamw");
  assert.equal(mlxEffectiveOptimizer("adamw_torch_fused"), "adamw");
  assert.equal(mlxEffectiveOptimizer("lion"), "lion");
});
