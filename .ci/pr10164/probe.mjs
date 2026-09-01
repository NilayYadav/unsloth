// Evaluate the rag_scope `context_length` expression exactly as it stands in
// chat-adapter.ts at some revision, for a set of scenarios. No reimplementation:
// the expression text is lifted out of the file and run.
import { readFileSync, existsSync } from "node:fs";
import { pathToFileURL } from "node:url";

const [, , adapterPath, helperPath, scenariosPath] = process.argv;

function extractExpressions(source) {
  const out = [];
  const marker = "context_length:";
  let from = 0;
  for (;;) {
    const at = source.indexOf(marker, from);
    if (at < 0) break;
    from = at + marker.length;
    let depth = 0;
    let i = from;
    for (; i < source.length; i++) {
      const ch = source[i];
      if ("([{".includes(ch)) depth++;
      else if (")]}".includes(ch)) {
        if (depth === 0) break;
        depth--;
      } else if (ch === "," && depth === 0) break;
    }
    const text = source.slice(from, i).trim();
    // Only the two rag_scope windows: the passthrough ceiling next door is a
    // different call and is not what this PR touches.
    if (/^(runtime\.|ragScopeContextLength\()/.test(text)) out.push(text);
  }
  return out;
}

const expressions = extractExpressions(readFileSync(adapterPath, "utf8"));
let ragScopeContextLength;
if (helperPath && existsSync(helperPath)) {
  ({ ragScopeContextLength } = await import(pathToFileURL(helperPath).href));
}

const scenarios = JSON.parse(readFileSync(scenariosPath, "utf8"));
const results = [];
for (const scenario of scenarios) {
  const { isExternalRequest, runtime, params } = scenario;
  const values = expressions.map((expression) => {
    const run = new Function(
      "runtime",
      "params",
      "isExternalRequest",
      "ragScopeContextLength",
      `return (${expression});`,
    );
    const value = run(runtime, params, isExternalRequest, ragScopeContextLength);
    return value === undefined ? null : value;
  });
  const unique = [...new Set(values.map((v) => JSON.stringify(v)))];
  if (unique.length !== 1) {
    throw new Error(`the two rag_scope windows disagree: ${unique.join(" vs ")}`);
  }
  results.push({ name: scenario.name, context_length: values[0] });
}
console.log(
  JSON.stringify({ expressions_found: expressions.length, results }, null, 2),
);
