// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { preferFullToolOutput } = await import(
  "../src/features/chat/tool-output-scope.ts"
);

// A timed-out python/terminal call returns the output it had already printed and then
// says it timed out. Past the model's cap that reads
// `<head>\n\n... (truncated ...)\nExecution timed out after N seconds.`, and the live
// stdout the card kept has no such sentence in it: only the backend adds it. Taking the
// stream alone therefore shows a command that looks like it finished normally.
const SENTENCE = "Execution timed out after 300 seconds.";
const STDOUT = `${"x".repeat(200_000)}\n`;
const NOTICE =
  "\n\n... (truncated to 6000 chars for the model; 200001 chars total. The full " +
  "output is not retained here; any files the code wrote persist in the working " +
  "directory.)";

test("a truncated timed-out card still says the call timed out", () => {
  const result = `${"x".repeat(6000)}${NOTICE}\n${SENTENCE}`;

  const card = preferFullToolOutput(STDOUT, result);

  // The full stream is what the user should read -- it is longer than the model's copy --
  // but the status the stream never carried has to survive with it.
  assert.ok(card.startsWith("x".repeat(6000)), "the stream was not preserved");
  assert.ok(card.length > result.length, "the fuller stream was not used");
  assert.ok(card.includes(SENTENCE), "the card no longer says the call timed out");
});

test("an untruncated timed-out card is left exactly as the backend wrote it", () => {
  const result = `progress\n\n${SENTENCE}`;

  assert.equal(preferFullToolOutput("progress\n", result), result);
});

test("a completed truncated call still shows the stream alone", () => {
  // The control: with nothing after the footer there is no status to re-attach, and the
  // card must not grow a second copy of the output.
  const result = `${"x".repeat(6000)}${NOTICE}`;

  assert.equal(preferFullToolOutput(STDOUT, result), STDOUT);
});
