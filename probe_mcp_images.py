"""PR 10088 A/B probe: does an image an MCP tool returned actually reach the model?

Runs against the shipped Studio backend, no model and no network. The
__MCP_IMAGES__ envelope already exists on main, so both branches build the same
conversation; only whether the image survives into the model's messages differs.
"""
import base64, io, json, os, sys

BACKEND = os.path.join(os.path.dirname(os.path.abspath(__file__)), "studio", "backend")
sys.path.insert(0, BACKEND)

from PIL import Image

import routes.inference as R
from models.inference import ChatCompletionRequest

buf = io.BytesIO()
Image.new("RGB", (8, 8), (10, 120, 200)).save(buf, format="PNG")
PNG_B64 = base64.b64encode(buf.getvalue()).decode()
ENVELOPE = "\n__MCP_IMAGES__:" + json.dumps([{"data": PNG_B64, "mimeType": "image/png"}])

HISTORY = [
    {"role": "user", "content": "read cat.png and tell me what colour the square is"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "call_0", "type": "function",
         "function": {"name": "mcp__fs__read_media_file", "arguments": "{}"}}]},
    {"role": "tool", "tool_call_id": "call_0", "content": "[1 image returned]" + ENVELOPE},
]

def image_parts(messages):
    out = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            out += [p for p in c if isinstance(p, dict)
                    and p.get("type") in ("image", "image_url", "input_image")]
    return out

results = []
def check(name, ok, detail):
    results.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'} :: {name} :: {detail}", flush=True)

req = ChatCompletionRequest(model="default", messages=HISTORY)

# A1 - local GGUF vision path
msgs, _ = R._openai_messages_for_gguf_chat(req, is_vision=True)
parts = image_parts(msgs)
check("gguf-vision: the returned image reaches the model",
      len(parts) == 1, f"{len(parts)} image part(s) in the model's messages")

# A2 - external provider vision path
ext = R._build_external_messages(req.messages, supports_vision=True, provider_type="openai")
eparts = image_parts(ext)
check("external-vision: the returned image reaches the provider",
      len(eparts) == 1, f"{len(eparts)} image part(s) in the provider payload")

# A3 - guard that must hold on BOTH branches: no base64 ever shown to a text-only model
tmsgs, _ = R._openai_messages_for_gguf_chat(req, is_vision=False)
leaked = any(PNG_B64 in str(m.get("content")) for m in tmsgs)
check("text-only: no base64 is sent to a model that cannot see it",
      not leaked, "no payload in the text-only conversation" if not leaked else "BASE64 LEAKED")

failed = [n for n, ok, _ in results if not ok]
print(f"\n{len(results) - len(failed)}/{len(results)} assertions passed", flush=True)
sys.exit(1 if failed else 0)
