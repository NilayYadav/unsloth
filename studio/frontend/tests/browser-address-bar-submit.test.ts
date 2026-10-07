// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import ts from "typescript";
import { readSrc } from "./helpers/kit.ts";
import { isWebUrl, resolveAddress } from "../src/features/browser/address.ts";

const FILE = "features/browser/browser-panel.tsx";

function submitHandler(deps: Record<string, unknown>) {
  const source = ts.createSourceFile(FILE, readSrc(FILE), ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
  const matches: ts.ArrowFunction[] = [];
  const visit = (node: ts.Node) => {
    if (
      ts.isJsxAttribute(node) &&
      node.name.getText(source) === "onSubmit" &&
      node.initializer &&
      ts.isJsxExpression(node.initializer) &&
      node.initializer.expression &&
      ts.isArrowFunction(node.initializer.expression) &&
      node.initializer.expression.getText(source).includes("resolveAddress(")
    )
      matches.push(node.initializer.expression);
    ts.forEachChild(node, visit);
  };
  visit(source);
  assert.equal(matches.length, 1, `${FILE}: one address bar submit handler`);
  const code = ts.transpileModule(`return (${matches[0].getText(source)});`, {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None },
  }).outputText;
  return new Function(...Object.keys(deps), code)(...Object.values(deps)) as (event: object) => void;
}

function submit(value: string) {
  const navigated: string[] = [];
  const toasts: string[] = [];
  let editing = true;
  const toast = Object.assign((message: string) => toasts.push(message), {
    error: (message: string) => toasts.push(message),
  });
  const handler = submitHandler({
    value,
    engine: "duckduckgo",
    tab: { id: "tab-1" },
    resolveAddress,
    isWebUrl,
    toast,
    t: (key: string) => key,
    useBrowserStore: { getState: () => ({ navigate: (_id: string, request: { url: string }) => navigated.push(request.url) }) },
    setEditing: (next: boolean) => {
      editing = next;
    },
    inputRef: { current: { blur() {} } },
  });
  handler({ preventDefault() {} });
  return { navigated, toasts, editing };
}

test("typing an address the panel can't open says so and keeps it in the bar", () => {
  for (const address of ["file:///etc/hosts", "chrome://settings", "ftp://example.com/a.txt"]) {
    const result = submit(address);
    assert.deepEqual(result.navigated, [], address);
    assert.deepEqual(result.toasts, ["browser.native.blocked"], address);
    assert.equal(result.editing, true, address);
  }
});

test("web addresses and searches still open without a message", () => {
  for (const [typed, url] of [
    ["https://unsloth.ai/docs", "https://unsloth.ai/docs"],
    ["docs.unsloth.ai", "https://docs.unsloth.ai"],
    ["what is lora", "https://html.duckduckgo.com/html/?q=what%20is%20lora"],
  ]) {
    const result = submit(typed);
    assert.deepEqual(result.navigated, [url]);
    assert.deepEqual(result.toasts, []);
    assert.equal(result.editing, false);
  }
});
