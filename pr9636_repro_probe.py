"""PR 9636 reproduction: do MCP images returned as EmbeddedResource render?

Builds the blocks with the REAL fastmcp helpers and the REAL mcp types, then runs
the studio backend's own _flatten_result. Exits non-zero on any mismatch.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parent / "studio" / "backend"
sys.path.insert(0, str(BACKEND))

from core.inference.mcp_client import MCP_IMAGES_SENTINEL, _flatten_result  # noqa: E402

import json  # noqa: E402

from fastmcp.client.client import CallToolResult  # noqa: E402
from fastmcp.utilities.types import File, Image  # noqa: E402
from mcp.types import TextContent  # noqa: E402

PNG = bytes.fromhex("89504e470d0a1a0a0000000d49484452")
tmp = Path(tempfile.mkdtemp())
png_path = tmp / "gen.png"
png_path.write_bytes(PNG)
pdf_path = tmp / "doc.pdf"
pdf_path.write_bytes(b"%PDF-1.4")


def result(*blocks):
    return CallToolResult(
        content=list(blocks), structured_content=None, meta=None, data=None, is_error=False
    )


def outcome(flat: str) -> str:
    if MCP_IMAGES_SENTINEL not in flat:
        return "empty" if flat == "" else f"text:{flat}"
    payload = flat.split(MCP_IMAGES_SENTINEL, 1)[1]
    return "render:" + json.loads(payload)[0]["mimeType"]


CASES = [
    ("File(path='gen.png')", File(path=png_path).to_resource_content(), "render:image/png"),
    ("File(data=..., format='png')", File(data=PNG, format="png").to_resource_content(), "render:image/png"),
    ("File(data=..., format='webp')", File(data=PNG, format="webp").to_resource_content(), "render:image/webp"),
    ("File(data=...) no format", File(data=PNG).to_resource_content(), "empty"),
    ("Image(data=..., format='png')", Image(data=PNG, format="png").to_image_content(), "render:image/png"),
    ("File(path='doc.pdf')", File(path=pdf_path).to_resource_content(), "empty"),
    ("TextContent", TextContent(type="text", text="hello"), "text:hello"),
]

# Mixed case and parameters: media type type/subtype are case-insensitive
# (RFC 9110 8.3.1), so these are well-formed and must still resolve to an image.
from mcp.types import BlobResourceContents, EmbeddedResource  # noqa: E402

for _mime, _want in (
    ("IMAGE/PNG", "render:image/png"),
    ("Image/Jpeg", "render:image/jpeg"),
    ("APPLICATION/PNG", "render:image/png"),
    ("image/png; charset=binary", "render:image/png"),
):
    CASES.append((
        f"EmbeddedResource {_mime!r}",
        EmbeddedResource(
            type="resource",
            resource=BlobResourceContents(
                uri="file:///out/gen.png",
                mimeType=_mime,
                blob=__import__("base64").b64encode(PNG).decode(),
            ),
        ),
        _want,
    ))

print(f"{'REAL FastMCP CALL':34} {'DECLARED MIME':26} {'EXPECTED':20} {'ACTUAL':20} RESULT")
print("=" * 124)
failures = 0
for name, block, expected in CASES:
    mime = getattr(block, "mimeType", None) or getattr(
        getattr(block, "resource", None), "mimeType", "-"
    )
    actual = outcome(_flatten_result(result(block)))
    ok = actual == expected
    failures += not ok
    print(f"{name:34} {str(mime):26} {expected:20} {actual:20} {'PASS' if ok else 'FAIL'}")

print("=" * 124)
if failures:
    print(f"REPRO: {failures} case(s) FAILED -- MCP embedded-resource images are not rendered")
    sys.exit(1)
print("REPRO: all cases PASSED -- MCP embedded-resource images render correctly")
