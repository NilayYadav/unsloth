// SPDX-License-Identifier: AGPL-3.0-only
// PR 10455 repro probe: what the composer serialises for an image with no caption.
// Identical on both A/B branches; only chat-adapter.ts differs.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import ts from "typescript";

const adapterPath = path.resolve(
  process.argv[2] ?? "studio/frontend/src/features/chat/api/chat-adapter.ts",
);
const adapterSource = readFileSync(adapterPath, "utf8");

function lift(opener) {
  const start = adapterSource.indexOf(opener);
  assert.ok(start >= 0, `${opener} is no longer defined in chat-adapter.ts`);
  const end = adapterSource.indexOf("\n}", start);
  assert.ok(end > start, `could not find the end of ${opener}`);
  return adapterSource.slice(start, end + 2);
}

const js = ts.transpileModule(
  [
    "function serializeAssistantReplayMessages() { throw new Error('not under test'); }",
    lift("function collectTextParts("),
    lift("function collectImageParts("),
    lift("function buildReplayContent("),
    lift("function toOpenAIMessages("),
    "return toOpenAIMessages;",
  ].join("\n\n"),
  { compilerOptions: { target: ts.ScriptTarget.ES2022 } },
).outputText;

const toOpenAIMessages = new Function(js)();
const IMAGE = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg==";

const failures = [];
function report(label, content) {
  const [serialized] = toOpenAIMessages({ role: "user", content });
  console.log(`--- ${label}`);
  console.log(`    serialised: ${JSON.stringify(serialized.content).slice(0, 220)}`);
  return serialized.content;
}

// The PR's case: an image attached with nothing typed.
const uncaptioned = report("uncaptioned image (the reported bug)", [
  { type: "image", image: IMAGE },
]);
if (
  Array.isArray(uncaptioned) &&
  uncaptioned.some((p) => p.type === "text" && !p.text)
) {
  failures.push(
    "REPRO: the composer sent an empty text block in front of the image",
  );
}

// Controls: identical on either side of the fix.
const captioned = report("captioned image (control)", [
  { type: "text", text: "what is this?" },
  { type: "image", image: IMAGE },
]);
try {
  assert.deepEqual(captioned, [
    { type: "text", text: "what is this?" },
    { type: "image_url", image_url: { url: IMAGE } },
  ]);
} catch {
  failures.push("CONTROL: a captioned image no longer leads with its caption");
}

const textOnly = report("text only (control)", [
  { type: "text", text: "no attachments here" },
]);
if (textOnly !== "no attachments here") {
  failures.push("CONTROL: a text-only turn no longer serialises to a plain string");
}

console.log();
if (failures.length) {
  for (const line of failures) console.log(`FAIL ${line}`);
  process.exit(1);
}
console.log("PASS an uncaptioned image serialises to the image alone; controls unchanged");
