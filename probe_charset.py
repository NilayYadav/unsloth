"""PR 10221 probe: a page whose charset label we cannot decode with must still be read.

Drives the real core.inference.tools._fetch_page_text with the network stubbed the
same way studio/backend/tests/test_web_fetch_binary_guard.py does. Identical on the
base and head branches; only the implementation under it changes.
"""

import codecs
import sys
import urllib.request
from email.message import Message
from pathlib import Path

BACKEND = Path(__file__).resolve().parent / "studio" / "backend"
sys.path.insert(0, str(BACKEND))

from core.inference import tools

MARKER = "MARKERWORD"
HTML = f"<html><body><p>{MARKER} in a page a browser renders fine.</p></body></html>".encode()


class _Resp:
    def __init__(self, body, ctype):
        self._b, self._p = body, 0
        self.headers = Message()
        self.headers["Content-Type"] = ctype

    def read(self, n=None):
        c = self._b[self._p:] if n is None else self._b[self._p:self._p + n]
        self._p += len(c)
        return c


class _Opener:
    def __init__(self, r): self._r = r
    def open(self, req, timeout=None): return self._r


def fetch(body, ctype):
    tools._validate_and_resolve_host = lambda host, port: (True, "", "93.184.216.34")
    urllib.request.build_opener = lambda *a, **k: _Opener(_Resp(body, ctype))
    try:
        return tools._fetch_page_text("https://example.com/thing", timeout=5)
    except Exception as e:
        return f"<probe caught {type(e).__name__}: {e}>"


CASES = [
    ("A unknown label 'unicode'",      HTML, "text/html; charset=unicode"),
    ("B unknown label 'utf8mb4'",      HTML, "text/html; charset=utf8mb4"),
    ("C unknown label 'x-user-defined'", HTML, "text/html; charset=x-user-defined"),
    ("D unknown label + UTF-16 BOM",
     codecs.BOM_UTF16_LE + f"{MARKER} and more text".encode("utf-16-le"),
     "text/plain; charset=utf16le"),
    ("E control: charset=utf-8",       HTML, "text/html; charset=utf-8"),
    ("F control: charset=iso-8859-1",  HTML, "text/html; charset=iso-8859-1"),
    ("G non-text codec 'base64'",      HTML, "text/html; charset=base64"),
]

print(f"python {sys.version.split()[0]}")
print(f"tools.py sha-context: {BACKEND}")
failed = []
for name, body, ctype in CASES:
    out = fetch(body, ctype)
    ok = MARKER in out
    snippet = out.replace("\n", " ")[:110]
    print(f"{'PASS' if ok else 'FAIL'} | {name:34} | Content-Type: {ctype}")
    print(f"       -> {snippet!r}")
    if not ok:
        failed.append(name)

print()
if failed:
    print(f"REPRO: {len(failed)}/{len(CASES)} case(s) could not read the page: {failed}")
    sys.exit(1)
print(f"REPRO: all {len(CASES)} cases returned the page text")
