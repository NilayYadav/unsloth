// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isPastedTextContent } from "../utils/pasted-text.ts";

// Backend cap: past it the whole request is refused.
const MAX_ASK_AGENT_FILES = 64;
const WRAPPED_RE = /^<attachment name=[^\n]*>\n([\s\S]*)\n<\/attachment>$/;
const encoder = new TextEncoder();

type Part = { type?: string; text?: string };
type Attachment = { name?: string; content?: readonly Part[] };

function sizeLabel(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

// A one-line text is a stand-in already ("only the python tool can read this file").
function isFileText(part: Part): part is { type: "text"; text: string } {
  return (
    part.type === "text" &&
    typeof part.text === "string" &&
    part.text.includes("\n") &&
    !isPastedTextContent(part.text)
  );
}

/** Attached documents become a name and a size in the messages; their text travels beside them for ask_agent. */
export function withFilesKeptForAskedAgents<M extends { attachments?: readonly unknown[] }>(
  messages: readonly M[],
) {
  const files = new Map<string, string>();
  const kept = messages.map((message) => {
    let changed = false;
    const attachments = (message.attachments ?? []).map((attachment) => {
      const { name = "", content = [] } = attachment as Attachment;
      const texts = content.filter(isFileText);
      if (!name || texts.length === 0) return attachment;
      changed = true;
      const text = texts.map((part) => WRAPPED_RE.exec(part.text)?.[1] ?? part.text).join("\n");
      files.delete(name);
      files.set(name, text);
      const note = `[${name} (${sizeLabel(encoder.encode(text).length)}) is attached. Its content is not shown to you. To have it read, call ask_agent with ${JSON.stringify(name)} in files.]`;
      return {
        ...(attachment as object),
        content: [{ type: "text", text: note }, ...content.filter((part) => !isFileText(part))],
      };
    });
    return changed ? ({ ...message, attachments } as M) : message;
  });
  const askAgentFiles = [...files]
    .slice(-MAX_ASK_AGENT_FILES)
    .map(([name, text]) => ({ name, text }));
  return { messages: kept, askAgentFiles };
}
