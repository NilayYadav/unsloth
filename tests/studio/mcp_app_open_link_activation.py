# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Does a link clicked inside an MCP App widget still open on the web build?

The bridge answers ui/open-link on a MessagePort task rather than in the click
handler, so the question is whether the host window still holds transient user
activation by the time it calls window.open. Reads the shim out of
mcp-app-frame.tsx so this cannot drift from what ships.

Firefox is run twice: once as configured, once with dom.disable_open_during_load
so a real popup blocker is in play. That second run carries a negative control
(a popup with no activation at all); unless the control is blocked the run
proves nothing, and it is asserted.
"""

import functools
import http.server
import json
import re
import socketserver
import sys
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parents[2]
FRAME = REPO / "studio/frontend/src/features/chat/mcp-apps/mcp-app-frame.tsx"
TOKEN = "tok-ci-open-link"


def bridge_shim(host_origin: str, token: str) -> str:
    """The shipped shim body, with only its two interpolations filled in."""
    text = FRAME.read_text()
    start = text.index("export function bridgeShim(")
    body = text[start : text.index("\n}\n", start)]
    literal = re.search(r"return `(.*?)`;", body, re.S)
    if not literal:
        raise SystemExit("FAIL: bridgeShim no longer returns a template literal")
    shim = literal.group(1)
    shim = shim.replace("${JSON.stringify(hostOrigin)}", json.dumps(host_origin))
    shim = shim.replace("${JSON.stringify(token)}", json.dumps(token))
    if "${" in shim:
        raise SystemExit(f"FAIL: unhandled interpolation in bridgeShim: {shim}")
    return shim


def open_link_source() -> str:
    """The shipped openLink, so the noopener argument is the real one."""
    text = (REPO / "studio/frontend/src/lib/open-link.ts").read_text()
    start = text.index("export function openLink(")
    # +3 keeps the closing brace the "\n}\n" match sits on.
    body = text[start : text.index("\n}\n", start) + 3]
    return body.replace("export function", "function").replace(
        "if (isTauri) {", "if (false) {"
    ).replace(": string", "").replace(": boolean", "")


def host_html(origin: str) -> str:
    widget = (
        "<!doctype html><meta charset=\"utf-8\">"
        "<a id=\"lnk\" href=\"https://example.com/widget\">open</a>"
        f"<script>{bridge_shim(origin, TOKEN)}</script>"
        "<script>document.getElementById('lnk').addEventListener('click', (e) => {"
        "e.preventDefault();"
        "window.parent.postMessage({jsonrpc:'2.0',id:1,method:'ui/open-link',"
        "params:{url:'https://example.com/widget'}});});</script>"
    )
    # "<\\/" so the widget's own </script> cannot close the host's script block.
    return """<!doctype html><meta charset="utf-8"><title>open-link activation</title><body>
<div id="mount"></div>
<script>
window.RESULTS = {};
%s
function probe(url) {
  const w = window.open(url, "_blank");
  if (w) { try { w.close(); } catch (e) {} }
  return w === null ? "blocked" : "opened";
}
const widget = %s;
const iframe = document.createElement("iframe");
iframe.src = URL.createObjectURL(new Blob([widget], {type: "text/html"}));
iframe.setAttribute("sandbox", "allow-scripts");
iframe.setAttribute("referrerpolicy", "no-referrer");
document.getElementById("mount").appendChild(iframe);
function handler(event) {
  const msg = event.data;
  if (!msg || msg.method !== "ui/open-link") return;
  window.RESULTS.activation = navigator.userActivation
    ? navigator.userActivation.isActive : "n/a";
  window.RESULTS.openLinkReturn = openLink(msg.params.url);
  window.RESULTS.probe = probe(msg.params.url);
  window.RESULTS.handled = true;
}
window.addEventListener("message", (event) => {
  if (event.source !== iframe.contentWindow) return;
  if (event.origin !== "null") return;
  const env = event.data;
  if (!env || env.__unslothMcpApp !== %s || env.__unslothMcpAppPort !== true) return;
  const port = event.ports[0];
  if (!port) return;
  port.onmessage = handler;
  window.RESULTS.handshake = true;
});
</script>
""" % (
        open_link_source(),
        json.dumps(widget).replace("</", "<\\/"),
        json.dumps(TOKEN),
    )


NO_ACTIVATION = """<!doctype html><meta charset="utf-8"><body><div id="r">pending</div>
<script>
(function(){
  var w = window.open('https://example.com/none','_blank');
  if (w) { try { w.close(); } catch(e){} }
  document.getElementById('r').textContent = w === null ? "blocked" : "opened";
})();
</script>
"""


def main() -> int:
    root = Path(__file__).parent / "_served"
    root.mkdir(exist_ok=True)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))

    class Quiet(socketserver.TCPServer):
        allow_reuse_address = True

        def handle_error(self, *_a):
            pass

    httpd = Quiet(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    origin = f"http://127.0.0.1:{port}"
    (root / "host.html").write_text(host_html(origin))
    (root / "none.html").write_text(NO_ACTIVATION)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    failures = []
    rows = []
    with sync_playwright() as p:
        cases = [
            ("chromium", {}, False),
            ("firefox", {}, False),
            ("webkit", {}, False),
            ("firefox", {"firefox_user_prefs": {"dom.disable_open_during_load": True}}, True),
        ]
        for engine, kwargs, blocker in cases:
            label = f"{engine}+blocker" if blocker else engine
            browser = getattr(p, engine).launch(**kwargs)
            ctx = browser.new_context()

            control = "n/a"
            if blocker:
                page = ctx.new_page()
                page.goto(f"{origin}/none.html")
                page.wait_for_function(
                    "document.getElementById('r').textContent !== 'pending'", timeout=15000
                )
                control = page.text_content("#r")
                if control != "blocked":
                    failures.append(
                        f"{label}: negative control was '{control}', so the blocker is "
                        "inert and this run cannot prove anything"
                    )

            page = ctx.new_page()
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(f"{origin}/host.html")
            try:
                page.wait_for_function(
                    "window.RESULTS && window.RESULTS.handshake === true", timeout=30000
                )
            except Exception:
                raise SystemExit(
                    f"FAIL {label}: the widget never handed the host a port"
                    + (f"; page errors: {errors}" if errors else "")
                )
            page.frames[1].click("#lnk")
            page.wait_for_function("window.RESULTS.handled === true", timeout=15000)
            res = page.evaluate("window.RESULTS")
            browser.close()

            rows.append((label, control, res.get("activation"), res.get("probe")))
            if res.get("activation") is not True:
                failures.append(
                    f"{label}: host had no transient activation when the handler ran "
                    f"({res.get('activation')!r}); a popup blocker would refuse this open"
                )
            if blocker and res.get("probe") != "opened":
                failures.append(
                    f"{label}: the widget's link did not open ({res.get('probe')!r}) "
                    "with a live popup blocker"
                )
            if res.get("openLinkReturn") is not True:
                failures.append(f"{label}: openLink returned {res.get('openLinkReturn')!r}")

    httpd.shutdown()

    print("\nengine              negative-control  host-activation  widget-link")
    for label, control, activation, probe in rows:
        print(f"{label:<20}{str(control):<18}{str(activation):<17}{probe}")
    print()
    for line in failures:
        print(f"FAIL {line}")
    if failures:
        return 1
    print("PASS ui/open-link keeps the host's transient user activation, and the")
    print("PASS widget's link opens with a popup blocker that blocks the control.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
