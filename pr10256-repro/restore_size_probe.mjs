// Ask THIS tree what size "Restore settings" puts into the Create form.
// Identical on both A/B arms: nothing here names the fix. The region of restoreSettings that
// derives the size is executed as-is, with every set* stubbed and every free identifier resolved
// from the sibling modules images-page.tsx imports.
import { readFileSync, readdirSync } from "node:fs";
import { pathToFileURL } from "node:url";
import path from "node:path";

const root = process.argv[2];
const W = Number(process.argv[3]);
const H = Number(process.argv[4]);
const WF = process.argv[5] === "null" ? null : process.argv[5];
const dir = path.join(root, "studio/frontend/src/features/images");
const source = readFileSync(path.join(dir, "images-page.tsx"), "utf8");

const start = source.indexOf("const restoreSettings = useCallback(");
if (start < 0) throw new Error("restoreSettings not found");
const body = source.slice(start, source.indexOf("\n  }, [", start));
const stop = body.indexOf("setBatchSize(");
if (stop < 0) throw new Error("size region not found");
const region = body.slice(body.indexOf("=> {") + 4, stop);

// Every export of every sibling .ts module, so a helper the tree factored out is in scope.
const symbols = new Map();
for (const f of readdirSync(dir).filter((f) => f.endsWith(".ts"))) {
  let mod;
  try {
    mod = await import(pathToFileURL(path.join(dir, f)).href);
  } catch {
    continue; // a module needing the bundler is not one this region can call
  }
  for (const [k, v] of Object.entries(mod)) if (!symbols.has(k)) symbols.set(k, v);
}

const captured = {};
const noop = () => {};
const scope = new Proxy(
  {},
  {
    has: () => true,
    get(_t, k) {
      if (k === "image") return { width: W, height: H, workflow: WF, guidance: 0, negative_prompt: "", prompt: "p", steps: 9, seed: 1, batch_seed: 1 };
      if (k === "setWidth") return (v) => { captured.width = v; };
      if (k === "setHeight") return (v) => { captured.height = v; };
      if (symbols.has(k)) return symbols.get(k);
      if (typeof k === "string" && k.startsWith("set")) return noop;
      if (k === Symbol.unscopables) return undefined;
      return noop;
    },
  },
);

new Function("scope", `with (scope) { ${region} }`)(scope);
if (typeof captured.width !== "number" || typeof captured.height !== "number")
  throw new Error("the region set no size");
console.log(JSON.stringify(captured));
