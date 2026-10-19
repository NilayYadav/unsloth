// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const MEMORY_EXTRACTION_PROMPT =
  "List everything you remember about me: saved memories, custom instructions, preferences, projects, and facts you've learned from our chats. Keep my own wording where you can. Write one item per line, in a single code block, with nothing summarised or left out. End with a line saying whether the list is complete.";

export const MEMORY_TOKEN_LIMIT = 1500;
export const MEMORY_CHAR_LIMIT = MEMORY_TOKEN_LIMIT * 4;

const MEMORY_HEADER =
  "Notes the user saved about themselves. Use them only when they matter for the request, and do not bring them up otherwise:";
const MEMORY_BLOCK_OVERHEAD =
  "<user_memory>\n\n</user_memory>".length + MEMORY_HEADER.length + 1;

const FENCE = /^\s*(```|~~~)/;
const BULLET = /^\s*(?:[-*•+]|\d+[.)])\s+/;
const LEADING_DATE =
  /^\s*\[?\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?\]?\s*[:\-–—]?\s*/;
const TRAILERS = [
  /^(?:that[’']?s|that is|this is|those are|these are)\s+(?:everything|all|the (?:complete|full|whole) list)\b/i,
  /\blist(?:\s+above)?\s+(?:is|was)\s+(?:now\s+)?(?:complete|incomplete|not complete|exhaustive|partial)\b/i,
  /^(?:is the list )?complete(?:ness)?\s*[:?]\s*(?:yes|no)\b/i,
];

export function memoryKey(text: string): string {
  return text.trim().replace(/\s+/g, " ").toLowerCase();
}

export function parseImportedMemories(text: string): string[] {
  const lines = text.split(/\r?\n/);
  const fenced = lines.filter((line) => FENCE.test(line)).length >= 2;
  const seen = new Set<string>();
  const out: string[] = [];
  let inFence = false;
  for (const raw of lines) {
    if (FENCE.test(raw)) {
      inFence = !inFence;
      continue;
    }
    if (fenced && !inFence) continue;
    const line = raw
      .replace(BULLET, "")
      .replace(LEADING_DATE, "")
      .replace(BULLET, "")
      .trim();
    if (!line || TRAILERS.some((re) => re.test(line))) continue;
    const key = memoryKey(line);
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(line);
  }
  return out;
}

export function selectMemoriesForPrompt(
  texts: readonly string[],
  charLimit = MEMORY_CHAR_LIMIT - MEMORY_BLOCK_OVERHEAD,
): string[] {
  const kept: string[] = [];
  let used = 0;
  for (const text of texts) {
    const line = text.trim();
    if (!line) continue;
    const cost = line.length + 3;
    if (used + cost > charLimit) break;
    kept.push(line);
    used += cost;
  }
  return kept;
}

export function buildUserMemoryBlock(texts: readonly string[]): string {
  const kept = selectMemoriesForPrompt(texts);
  if (!kept.length) return "";
  return [
    "<user_memory>",
    MEMORY_HEADER,
    ...kept.map((text) => `- ${text.replace(/\s*\n\s*/g, " ")}`),
    "</user_memory>",
  ].join("\n");
}

export function estimateMemoryTokens(block: string): number {
  return Math.ceil(block.length / 4);
}
