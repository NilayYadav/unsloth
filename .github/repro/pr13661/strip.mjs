import { readFileSync, writeFileSync } from "node:fs";
import { stripTypeScriptTypes } from "node:module";

const [src, out] = process.argv.slice(2);
const js = stripTypeScriptTypes(readFileSync(src, "utf8")).replace(/^export /gm, "");
writeFileSync(out, `${js}\nglobalThis.__resolveAddress = resolveAddress;\n`);
