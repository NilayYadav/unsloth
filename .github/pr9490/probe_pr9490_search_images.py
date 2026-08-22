"""PR 9490 A/B probe: web_search returns inline images, the model never sees an
image URL, and the thumbnail proxy serves resized JPEG bytes that a clear-all reaps.

Identical on both branches. Run from the repository root. No network: ddgs and the
outbound fetch are stubbed, so this measures the changed code path and nothing else.
"""

from __future__ import annotations

import io
import json
import re
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

TOKEN_RE = re.compile(r"\[\[img:([0-9a-f]{12})\]\]")
THUMB_HOST = "tse1.mm.bing.net"

FAILURES: list[str] = []


def check(condition: object, label: str) -> None:
    if condition:
        print(f"PROBE PASS: {label}", flush = True)
    else:
        print(f"PROBE FAIL: {label}", flush = True)
        FAILURES.append(label)


def die(label: str) -> None:
    print(f"PROBE FAIL: {label}", flush = True)
    print("PROBE RESULT: FAIL", flush = True)
    raise SystemExit(1)


def png_bytes(size = (900, 600)) -> bytes:
    from PIL import Image

    out = io.BytesIO()
    Image.new("RGB", size, (12, 140, 220)).save(out, format = "PNG")
    return out.getvalue()


RAW_IMAGES = [
    {
        "title": "Golden Retriever portrait",
        "thumbnail": f"https://{THUMB_HOST}/th?id=golden",
        "image": "https://img.example.com/golden.jpg",
        "url": "https://www.akc.org/dog-breeds/golden-retriever/",
    },
    {
        "title": "German Shepherd portrait",
        "thumbnail": f"https://{THUMB_HOST}/th?id=shepherd",
        "image": "https://img.example.com/shepherd.jpg",
        "url": "https://www.akc.org/dog-breeds/german-shepherd-dog/",
    },
]


class StubDDGS:
    """Stands in for the ddgs client: fixed text and image sweeps, no network."""

    def __init__(self, **_kwargs) -> None:
        pass

    def text(self, query, max_results = 5, **_kwargs):
        return [
            {
                "title": "Popular dog breeds",
                "href": "https://www.akc.org/dog-breeds/",
                "body": "A list of breeds.",
            }
        ]

    def images(self, query, max_results = 5, **_kwargs):
        # Per subject, so "one picture for the thing you named" is what is measured.
        wanted = str(query).lower()
        matched = [r for r in RAW_IMAGES if r["title"].split(" portrait")[0].lower() in wanted]
        return matched or RAW_IMAGES


def main() -> int:
    print("SETUP: importing third-party dependencies", flush = True)
    try:
        from PIL import Image  # noqa: F401
        import structlog  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"SETUP FAIL: dependency import failed ({exc})", flush = True)
        return 2
    backend = Path("studio/backend").resolve()
    if not backend.is_dir():
        print(f"SETUP FAIL: {backend} missing", flush = True)
        return 2
    sys.path.insert(0, str(backend))
    print("SETUP: ok", flush = True)

    try:
        from core.inference import search_images, tools
        from core.inference.tool_loop_controller import strip_result_for_model
    except ImportError as exc:
        die(f"web_search image results are absent from this tree ({exc})")

    if "search_images" not in getattr(tools.execute_tool, "__code__").co_varnames:
        die("execute_tool does not accept search_images, so no tool call can return pictures")

    cache = Path(tempfile.mkdtemp(prefix = "pr9490-"))
    search_images._cache_dir = lambda: cache
    sys.modules["ddgs"] = SimpleNamespace(DDGS = StubDDGS)

    result = tools.execute_tool(
        "web_search",
        {"query": "popular dog breeds", "image_queries": ["Golden Retriever", "German Shepherd"]},
        search_images = True,
    )

    ids = TOKEN_RE.findall(result)
    check(len(ids) >= 2, f"the tool result offers one [[img:...]] token per subject (got {len(ids)})")
    check(
        search_images.SEARCH_IMAGES_SENTINEL in result,
        "the frontend envelope rides along with the result",
    )
    check(THUMB_HOST not in result, "no image host URL reaches the model or the envelope")

    _text, entries = search_images.split_images_envelope(result)
    check(
        bool(entries) and all(search_images.is_image_entry(e) for e in entries),
        f"the envelope parses into well-formed entries ({len(entries)} entries)",
    )
    subjects = {e.get("subject") for e in entries}
    check(
        {"Golden Retriever", "German Shepherd"} <= subjects,
        f"each requested subject is represented ({sorted(s for s in subjects if s)})",
    )

    for_model = strip_result_for_model(result, "web_search")
    check(
        search_images.SEARCH_IMAGES_SENTINEL not in for_model,
        "strip_result_for_model removes the envelope before the model sees the result",
    )
    check(
        "[[img:" in for_model,
        "the tokens the model is asked to place survive that strip",
    )

    tools._fetch_url_raw = lambda url, **kw: (None, png_bytes(), "image/png")
    image_id = entries[0]["id"]
    data = search_images.thumbnail_bytes(image_id)
    check(
        isinstance(data, bytes) and data[:3] == b"\xff\xd8\xff",
        "the proxy re-encodes the fetched picture as JPEG",
    )
    if isinstance(data, bytes) and data[:3] == b"\xff\xd8\xff":
        from PIL import Image as _Image

        with _Image.open(io.BytesIO(data)) as thumb:
            check(
                max(thumb.size) <= search_images.THUMBNAIL_EDGE_PX,
                f"the 900x600 source is resized to {thumb.size}, within "
                f"{search_images.THUMBNAIL_EDGE_PX}px",
            )
    check(
        (cache / f"{image_id}.jpg").is_file(),
        "the JPEG is cached under Studio's own directory, not fetched by the browser",
    )

    search_images.clear_cache()
    check(
        search_images.thumbnail_bytes(image_id) is None
        and not list(cache.glob("*.jpg"))
        and not list(cache.glob("*.json")),
        "clear all chats reaps the thumbnails and their metadata",
    )

    print(f"PROBE ENTRIES: {json.dumps(entries)[:400]}", flush = True)
    print(f"PROBE RESULT: {'FAIL' if FAILURES else 'PASS'}", flush = True)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
