// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test, { after } from "node:test";

import {
  installLocalStorageFake,
  registerStoreStubResolver,
} from "./helpers/kit.ts";

registerStoreStubResolver();
installLocalStorageFake();

const { setAuthFetchHandler } = await import("./helpers/store-stubs/auth.ts");
const { useTrainingConfigStore } = await import(
  "../src/features/training/stores/training-config-store.ts"
);
const { buildTrainingStartPayload } = await import(
  "../src/features/training/api/mappers.ts"
);
const { validateTrainingConfig } = await import(
  "../src/features/training/lib/validation.ts"
);
const { missingDecisionColumns } = await import(
  "../src/features/training/lib/decision-dataset.ts"
);

const LAYA = "convaiinnovations/laya";

const LAYA_CONFIG = {
  id: LAYA,
  config: {
    training: {
      learning_rate: 8e-4,
      batch_size: 8,
      gradient_accumulation_steps: 8,
      num_epochs: 4,
      weight_decay: 0.01,
      lr_scheduler_type: "cosine",
      gradient_checkpointing: "unsloth",
    },
    lora: { lora_r: 64, lora_alpha: 64, lora_dropout: 0 },
  },
  is_vision: false,
  is_embedding: false,
  is_decision: true,
  decision_checkpoints: [
    { name: "laya-multilingual", subfolder: "multilingual", description: "" },
    { name: "laya-english", subfolder: null, description: "" },
    {
      name: "laya-typed-decisions",
      subfolder: "typed-decisions",
      description: "",
    },
  ],
  is_audio: false,
  audio_type_known: true,
  is_lora: false,
  model_type: "decision",
  model_size_bytes: 600_000_000,
  max_position_embeddings: null,
};

async function waitForModelDefaults(model: string): Promise<void> {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    const state = useTrainingConfigStore.getState();
    if (
      !state.isLoadingModelDefaults &&
      state.modelDefaultsAppliedFor === model
    ) {
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
  throw new Error("model defaults did not settle");
}

async function selectLaya(): Promise<string[]> {
  const requested: string[] = [];
  setAuthFetchHandler((input) => {
    requested.push(input);
    return Promise.resolve(Response.json(LAYA_CONFIG));
  });
  useTrainingConfigStore.getState().selectTrainingModel(LAYA, "text");
  await waitForModelDefaults(LAYA);
  return requested;
}

after(() => setAuthFetchHandler(null));

test("picking a decision model starts a LoRA run with the model's defaults", async () => {
  useTrainingConfigStore.getState().reset();
  useTrainingConfigStore.setState({
    trainingMethod: "qlora",
    packing: true,
    datasetStreaming: true,
  });

  const requested = await selectLaya();

  assert.deepEqual(
    requested.filter((url) => !url.startsWith("/api/models/config/")),
    [],
    "a decision model must not ask the hardware for a LoRA/QLoRA pick",
  );
  const state = useTrainingConfigStore.getState();
  assert.equal(state.modelType, "decision");
  assert.equal(state.trainingMethod, "lora");
  assert.equal(state.learningRate, 8e-4);
  assert.equal(state.loraRank, 64);
  assert.equal(state.loraAlpha, 64);
  assert.equal(state.batchSize, 8);
  assert.equal(state.gradientAccumulation, 8);
  assert.equal(state.datasetStreaming, false);
  assert.equal(state.decisionCheckpoints?.length, 3);
  assert.equal(state.modelSubfolder, "multilingual");
});

test("leaving CPT for a decision model does not keep CPT's learning rate", async () => {
  useTrainingConfigStore.getState().reset();
  useTrainingConfigStore.getState().setTrainingMethod("cpt");

  await selectLaya();

  const state = useTrainingConfigStore.getState();
  assert.equal(state.trainingMethod, "lora");
  assert.equal(state.learningRate, 8e-4);
});

test("a decision model can switch to full fine-tuning at the full learning rate", async () => {
  useTrainingConfigStore.getState().reset();
  await selectLaya();
  useTrainingConfigStore.getState().setTrainingMethod("full");

  const payload = buildTrainingStartPayload(
    useTrainingConfigStore.getState(),
    null,
  );

  assert.equal(payload.training_type, "Full Finetuning");
  assert.equal(payload.use_lora, false);
  assert.equal(payload.learning_rate, "0.00002");
});

test("a decision run sends the decision fields and nothing the recipe ignores", async () => {
  useTrainingConfigStore.getState().reset();
  await selectLaya();
  useTrainingConfigStore.getState().setModelSubfolder(null);
  useTrainingConfigStore.setState({
    datasetSource: "huggingface",
    dataset: "LocalLLaMA/typed-decisions",
    datasetStreaming: true,
    packing: true,
    trainOnCompletions: true,
    loraVariant: "dora",
    datasetManualMapping: { state: "user", gold: "assistant" },
  });

  const payload = buildTrainingStartPayload(
    useTrainingConfigStore.getState(),
    null,
  );

  assert.equal(payload.is_decision, true);
  assert.equal(payload.model_subfolder, null);
  assert.equal(payload.is_embedding, false);
  assert.equal(payload.training_type, "LoRA/QLoRA");
  assert.equal(payload.load_in_4bit, false);
  assert.equal(payload.use_lora, true);
  assert.equal(payload.lora_r, 64);
  assert.equal(payload.use_dora, false);
  assert.equal(payload.packing, false);
  assert.equal(payload.train_on_completions, false);
  assert.equal(payload.dataset_streaming, false);
  assert.equal(payload.custom_format_mapping, undefined);
  assert.equal(payload.learning_rate, "0.0008");
});

test("an LLM run keeps its method and sends no decision fields", () => {
  useTrainingConfigStore.getState().reset();
  const payload = buildTrainingStartPayload(
    {
      ...useTrainingConfigStore.getState(),
      selectedModel: "unsloth/Qwen3-0.6B",
      modelType: "text",
      trainingMethod: "qlora",
      modelSubfolder: "multilingual",
    },
    null,
  );

  assert.equal(payload.is_decision, false);
  assert.equal(payload.model_subfolder, null);
  assert.equal(payload.training_type, "LoRA/QLoRA");
  assert.equal(payload.load_in_4bit, true);
});

test("decision training is refused on Apple Silicon only", () => {
  const config = {
    ...useTrainingConfigStore.getState(),
    selectedModel: LAYA,
    modelType: "decision" as const,
    trainingMethod: "full" as const,
    datasetSource: "huggingface" as const,
    dataset: "LocalLLaMA/typed-decisions",
    datasetKnownCached: false,
    manualDatasetOptionsValid: true,
  };

  assert.deepEqual(validateTrainingConfig(config, "mac"), {
    ok: false,
    errorKey: "studio.params.notSupportedAppleSilicon",
  });
  assert.deepEqual(validateTrainingConfig(config, "cuda"), {
    ok: true,
    errorKey: null,
  });
});

test("the start check names the decision columns a dataset lacks", () => {
  assert.deepEqual(missingDecisionColumns(["state", "questions", "gold"]), []);
  assert.deepEqual(
    missingDecisionColumns(["state", "questions", "answers", "id"]),
    [],
  );
  assert.deepEqual(missingDecisionColumns(["messages"]), [
    "state",
    "questions",
    "gold",
  ]);
  assert.deepEqual(missingDecisionColumns(["state", "gold"]), ["questions"]);
});
