// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// PR #9641 repro probe.
//
// Takes the argument derivation straight out of each shipped tool card, the way
// tests/search-images.test.ts already does, transpiles it and runs it on the
// argument shapes local models actually send. A card that reads a model
// argument without coercing it throws here exactly as it throws inside React's
// render, where the only catcher left is the router's: all of Studio is
// replaced with "Something went wrong!".
//
// Exit 0 = every card survived. Exit 1 = at least one card crashed.

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import ts from "typescript";

const CARDS_DIR = new URL(
  "../src/components/assistant-ui/",
  import.meta.url,
);

// Reproduces src/components/assistant-ui/tool-arg-text.ts for the pre-fix tree,
// which does not have it yet. The post-fix tree calls its own.
const toolArgText = (value) => (value == null ? "" : String(value));

const CASES = [
  {
    file: "tool-ui-python.tsx",
    from: "const code =",
    to: "const isRunning =",
    run: "code.split('\\n')[0]?.slice(0, 60) ?? ''",
    args: { code: 42 },
    expect: "42",
    what: "python: code.split on a number",
  },
  {
    file: "tool-ui-terminal.tsx",
    from: "const command =",
    to: "const isRunning =",
    run: "command.slice(0, 60)",
    args: { command: 42 },
    expect: "42",
    what: "terminal: command.slice on a number",
  },
  {
    file: "tool-ui-knowledge-base.tsx",
    from: "const query =",
    to: "const isRunning =",
    // The one card that never threw: it only interpolates the query into JSX.
    // Kept so the header it renders is still checked after the change.
    run: "`Searched documents for \"${query}\"`",
    args: { query: 42 },
    expect: 'Searched documents for "42"',
    what: "knowledge base: query interpolated into the header",
  },
  {
    file: "tool-ui-web-search.tsx",
    from: "const query =",
    to: "const isUrlFetch =",
    run: "[query.trim(), url].join('|')",
    args: { query: 42, url: 7 },
    expect: "42|7",
    what: "web search: query.trim and url on numbers",
  },
  {
    file: "tool-ui-code-execution.tsx",
    from: "const parsedArgs =",
    to: "const isRunning =",
    run: "command.replace(/\\s+/g, ' ').trim()",
    args: { command: 42 },
    expect: "42",
    what: "code execution: truncateCommandLabel on a number",
  },
  {
    file: "tool-ui-image-generation.tsx",
    from: "const parsedArgs =",
    to: "const isRunning =",
    run: "prompt.trim()",
    args: { prompt: 42 },
    expect: "42",
    what: "image generation: prompt.trim on a number",
  },
];

const source = (file) =>
  readFileSync(fileURLToPath(new URL(file, CARDS_DIR)), "utf8");

let failed = 0;
for (const card of CASES) {
  const text = source(card.file);
  const start = text.indexOf(card.from);
  const end = text.indexOf(card.to);
  if (start < 0 || end < 0 || end <= start) {
    console.log(`FAIL  ${card.what}\n      derivation markers not found in ${card.file}`);
    failed += 1;
    continue;
  }
  const derivation = text.slice(start, end);
  const body = ts.transpileModule(`${derivation}\nreturn ${card.run};`, {
    compilerOptions: { target: ts.ScriptTarget.ES2022 },
  }).outputText;

  let actual;
  try {
    actual = new Function("args", "toolArgText", body)(card.args, toolArgText);
  } catch (error) {
    console.log(
      `FAIL  ${card.what}\n` +
        `      args ${JSON.stringify(card.args)} -> ${error.constructor.name}: ${error.message}\n` +
        `      this throw reaches the router boundary: Studio is replaced with "Something went wrong!"`,
    );
    failed += 1;
    continue;
  }
  if (actual !== card.expect) {
    console.log(
      `FAIL  ${card.what}\n      args ${JSON.stringify(card.args)} -> ${JSON.stringify(actual)}, expected ${JSON.stringify(card.expect)}`,
    );
    failed += 1;
    continue;
  }
  console.log(
    `PASS  ${card.what}\n      args ${JSON.stringify(card.args)} -> ${JSON.stringify(actual)}`,
  );
}

console.log(`\n${CASES.length - failed}/${CASES.length} tool cards survived a non-string argument`);
if (failed > 0) {
  console.log(`CRASHED: ${failed}`);
  process.exit(1);
}
