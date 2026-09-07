# SPDX-License-Identifier: AGPL-3.0-only
"""PR 10455 repro probe: an uncaptioned image must not reach Anthropic as an empty text block.

Runs the real `ExternalProviderClient._stream_anthropic` against a stand-in
Messages API that enforces Anthropic's documented rule ("text content blocks
must be non-empty", HTTP 400). Identical on both A/B branches; only the
implementation under test differs.
"""

import asyncio
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from core.inference.external_provider import ExternalProviderClient

RECEIVED: list = []

SSE_OK = (
    b'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_probe",'
    b'"type":"message","role":"assistant","model":"claude-opus-4-7","content":[],'
    b'"usage":{"input_tokens":1,"output_tokens":1}}}\n\n'
    b'event: content_block_start\ndata: {"type":"content_block_start","index":0,'
    b'"content_block":{"type":"text","text":""}}\n\n'
    b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
    b'"delta":{"type":"text_delta","text":"a photo of a cat"}}\n\n'
    b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
)


class StandInAnthropic(BaseHTTPRequestHandler):
    """Rejects empty text blocks exactly the way api.anthropic.com does."""

    def log_message(self, *_args):
        pass

    def do_POST(self):
        length = int(self.headers.get("content-length") or 0)
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        RECEIVED.append(body)

        for m_i, message in enumerate(body.get("messages") or []):
            content = message.get("content")
            if not isinstance(content, list):
                continue
            if not content:
                return self._reject(f"messages.{m_i}.content: at least one block is required")
            for b_i, block in enumerate(content):
                if block.get("type") == "text" and not block.get("text"):
                    return self._reject(
                        f"messages.{m_i}.content.{b_i}.text: text content blocks must be non-empty"
                    )

        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(SSE_OK)))
        self.end_headers()
        self.wfile.write(SSE_OK)

    def _reject(self, message: str):
        payload = json.dumps(
            {"type": "error", "error": {"type": "invalid_request_error", "message": message}}
        ).encode("utf-8")
        self.send_response(400)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


IMAGE = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="


async def _run(port: int, content):
    client = ExternalProviderClient(
        provider_type = "anthropic",
        base_url = f"http://127.0.0.1:{port}/v1",
        api_key = "sk-ant-probe-placeholder",
    )
    chunks = []
    async for chunk in client._stream_anthropic(
        messages = [{"role": "user", "content": content}],
        model = "claude-opus-4-7",
        temperature = 0.7,
        top_p = 0.95,
        max_tokens = 64,
        enable_prompt_caching = False,
    ):
        chunks.append(chunk)
    await client.close()
    return chunks


def case(port, label, content):
    RECEIVED.clear()
    chunks = asyncio.new_event_loop().run_until_complete(_run(port, content))
    sent = RECEIVED[-1] if RECEIVED else {}
    wire = (sent.get("messages") or [{}])[0].get("content")
    error = None
    text = ""
    for chunk in chunks:
        for line in chunk.splitlines():
            if not line.startswith("data: "):
                continue
            try:
                parsed = json.loads(line[6:])
            except Exception:
                continue
            if isinstance(parsed, dict) and parsed.get("error"):
                error = parsed["error"].get("message")
            for choice in (parsed.get("choices") or []) if isinstance(parsed, dict) else []:
                text += (choice.get("delta") or {}).get("content") or ""
    print(f"--- {label}")
    print(f"    wire content : {json.dumps(wire)[:220]}")
    print(f"    assistant    : {text!r}")
    print(f"    provider err : {error}")
    return wire, text, error


def main() -> int:
    server = ThreadingHTTPServer(("127.0.0.1", 0), StandInAnthropic)
    port = server.server_address[1]
    threading.Thread(target = server.serve_forever, daemon = True).start()
    print(f"stand-in Anthropic Messages API on 127.0.0.1:{port}")

    failures = []

    # The PR's case: an image attached with nothing typed.
    wire, text, error = case(
        port,
        "uncaptioned image (the reported bug)",
        [{"type": "text", "text": ""}, {"type": "image_url", "image_url": {"url": IMAGE}}],
    )
    empty_blocks = [p for p in (wire or []) if p.get("type") == "text" and not p.get("text")]
    if empty_blocks:
        failures.append(
            "REPRO: an empty text block reached the provider and it answered 400 "
            f"({error!r}); the user sees Generation failed"
        )
    if not text:
        failures.append("REPRO: the turn produced no assistant text")

    # Controls: both must behave identically on either side of the fix.
    wire, text, error = case(
        port,
        "captioned image (control)",
        [
            {"type": "text", "text": "what is this?"},
            {"type": "image_url", "image_url": {"url": IMAGE}},
        ],
    )
    if wire != [
        {"type": "text", "text": "what is this?"},
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "iVBORw0KGgoAAAANSUhEUg==",
            },
        },
    ]:
        failures.append("CONTROL: a captioned image no longer sends its caption then the image")
    if not text or error:
        failures.append("CONTROL: the captioned image turn did not complete")

    wire, text, error = case(port, "text only (control)", "just text, no attachment")
    if not text or error:
        failures.append("CONTROL: a plain text turn did not complete")

    server.shutdown()
    print()
    if failures:
        for line in failures:
            print(f"FAIL {line}")
        return 1
    print("PASS an uncaptioned image reaches Anthropic with no empty text block; controls unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
