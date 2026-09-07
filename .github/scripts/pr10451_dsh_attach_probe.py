# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Capture what `unsloth start dsh --context-length 0` asks a running Unsloth to load.

Stands a fake Unsloth Studio on loopback, points the real CLI at it with an explicit
--api-key (so no identity handshake is needed), runs the real Typer command with
--no-launch, and prints every request the CLI made plus the /api/inference/load body.
"""

import json
import os
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.getcwd())

RESIDENT = "unsloth/Qwen3-8B"
RESIDENT_CTX = 4096

REQUESTS = []
LOADS = []


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        REQUESTS.append(("GET", self.path))
        if self.path.startswith("/api/health"):
            return self._send({"status": "healthy"})
        if self.path.startswith("/v1/models"):
            return self._send({"data": [{"id": RESIDENT, "loaded": True}]})
        if self.path.startswith("/api/inference/status"):
            return self._send(
                {
                    "is_gguf": True,
                    "active_model": RESIDENT,
                    "model_identifier": RESIDENT,
                    "gguf_variant": "UD-Q4_K_XL",
                    "requested_context_length": RESIDENT_CTX,
                    "requested_load_mode": None,
                }
            )
        return self._send({})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(raw.decode() or "{}")
        except ValueError:
            payload = {"__raw__": raw.decode(errors="replace")}
        REQUESTS.append(("POST", self.path))
        if self.path.startswith("/api/inference/load"):
            LOADS.append(payload)
            return self._send({"status": "already_loaded", "model": RESIDENT})
        return self._send({})


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    base = f"http://127.0.0.1:{port}"
    threading.Thread(target=server.serve_forever, daemon=True).start()

    home = tempfile.mkdtemp(prefix="dsh-probe-")
    os.environ["UNSLOTH_STUDIO_URL"] = base
    os.environ["UNSLOTH_STUDIO_HOME"] = home
    os.environ["HOME"] = home

    argv = ["dsh", "--no-launch", "--api-key", "k", "--context-length", "0"]
    print(f"[probe] resident {RESIDENT} at context {RESIDENT_CTX}")
    print(f"[probe] running: unsloth start {' '.join(argv)}")

    import unsloth_cli.commands.start as start_cli
    from typer.testing import CliRunner

    result = CliRunner().invoke(start_cli.start_app, argv)
    if result.output:
        for line in result.output.rstrip().splitlines():
            print(f"[cli] {line}")
    if result.exception is not None and not isinstance(result.exception, SystemExit):
        import traceback

        traceback.print_exception(
            type(result.exception), result.exception, result.exception.__traceback__
        )

    print(f"[probe] exit code {result.exit_code}")
    print("[probe] requests the CLI made:")
    for method, path in REQUESTS:
        print(f"[probe]   {method} {path}")

    print(f"[probe] POST /api/inference/load count = {len(LOADS)}")
    for payload in LOADS:
        print(f"[probe] load body = {json.dumps(payload, sort_keys=True)}")

    if not LOADS:
        print("[probe] RESULT: FAIL -- --context-length 0 was dropped; no load was requested,")
        print(f"[probe]         so the model stays at context {RESIDENT_CTX}.")
        return 1
    body = LOADS[0]
    if body.get("max_seq_length") != 0:
        print(f"[probe] RESULT: FAIL -- load sent but max_seq_length={body.get('max_seq_length')!r}")
        return 1
    print("[probe] RESULT: PASS -- the load carries max_seq_length=0, so the reset is applied.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
