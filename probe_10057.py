"""#10057: an MCP tool returns an image; does the model actually get the pixels?

Route-level probe. No model, no server, no network: it builds the model-facing
message list the external-provider path sends, from a conversation whose tool
result carries the backend's __MCP_IMAGES__ envelope, and counts what the model
would see. Runs identically on main and on the PR head.
"""
import base64
import io
import json
import sys

sys.path.insert(0, "studio/backend")

from PIL import Image

from routes.inference import _build_external_messages
from models.inference import ChatMessage

buffer = io.BytesIO()
Image.new("RGB", (24, 24), (0, 0, 255)).save(buffer, format="PNG")
PNG_B64 = base64.b64encode(buffer.getvalue()).decode()
ENVELOPE = "\n__MCP_IMAGES__:" + json.dumps([{"data": PNG_B64, "mimeType": "image/png"}])

CONVERSATION = [
    ChatMessage(role="user", content="take a screenshot and tell me its colour"),
    ChatMessage(
        role="assistant",
        content="",
        tool_calls=[
            {
                "id": "call_0",
                "type": "function",
                "function": {"name": "mcp__shot__capture", "arguments": "{}"},
            }
        ],
    ),
    ChatMessage(
        role="tool",
        tool_call_id="call_0",
        name="mcp__shot__capture",
        content="[1 image returned]" + ENVELOPE,
    ),
    ChatMessage(role="user", content="what colour was it"),
]

built = _build_external_messages(list(CONVERSATION), True, provider_type="openai")

image_parts = sum(
    1
    for message in built
    if isinstance(message.get("content"), list)
    for part in message["content"]
    if isinstance(part, dict) and part.get("type") == "image_url"
)
def _text_of(message):
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


prompt_text = "".join(_text_of(message) for message in built)
base64_as_text = PNG_B64 in prompt_text
sentinel_left = "__MCP_IMAGES__" in prompt_text

print("PROBE #10057 -- what the model is sent when an MCP tool returns an image")
print(f"  image parts reaching the model : {image_parts}")
print(f"  __MCP_IMAGES__ left in the text: {sentinel_left}")
print(f"  raw base64 left as prompt text : {base64_as_text}")

if image_parts == 0:
    print("REPRO: FAIL -- the model was told an image was attached and shown none")
    sys.exit(1)
print("REPRO: PASS -- the pixels reach the model as image input")
sys.exit(0)
