"""PR 9893 repro probe: how long does Cancel take to land while a load is in flight?

Mirrors the real contention. load_model() holds self._lock across _wait_for_health();
unload_model() sets self._cancel_event and then blocks acquiring that same lock.
llama-server here is alive but never healthy, which is what a big GGUF on a slow disk
looks like from the health probe's side.

Prints real measured seconds and exits non-zero when Cancel does not land promptly.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import types
from pathlib import Path
from unittest import mock

BACKEND = Path(__file__).resolve().parent / "studio" / "backend"
sys.path.insert(0, str(BACKEND))

_loggers = types.ModuleType("loggers")
_loggers.get_logger = lambda name: logging.getLogger(name)
sys.modules.setdefault("loggers", _loggers)
sys.modules.setdefault("structlog", types.ModuleType("structlog"))

import httpx  # noqa: E402

from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402

HEALTH_TIMEOUT = float(os.environ.get("PROBE_HEALTH_TIMEOUT", "180"))
CLICK_AFTER = float(os.environ.get("PROBE_CLICK_AFTER", "2"))
PASS_UNDER = float(os.environ.get("PROBE_PASS_UNDER", "10"))

backend = LlamaCppBackend.__new__(LlamaCppBackend)
backend._port = 12345
backend._stdout_thread = None
backend._stdout_lines = []
backend._process = mock.Mock()
backend._process.poll.return_value = None
backend._lock = threading.Lock()
backend._cancel_event = threading.Event()
backend._health_probe_event = threading.Event()

httpx.get = lambda *a, **kw: mock.Mock(status_code = 503)

in_wait = threading.Event()
out = {}


def _load() -> None:
    with backend._lock:
        in_wait.set()
        started = time.monotonic()
        out["healthy"] = backend._wait_for_health(timeout = HEALTH_TIMEOUT, interval = 0.5)
        out["wait_secs"] = round(time.monotonic() - started, 2)


loader = threading.Thread(target = _load, name = "load", daemon = True)
loader.start()
if not in_wait.wait(30):
    print("PROBE HARNESS ERROR: load thread never entered the health wait")
    raise SystemExit(2)

time.sleep(CLICK_AFTER)

# Exactly what unload_model() does: raise the flag, then take the lock the load holds.
clicked = time.monotonic()
backend._cancel_event.set()
acquired = backend._lock.acquire(timeout = HEALTH_TIMEOUT + 120)
cancel_secs = round(time.monotonic() - clicked, 2)
if acquired:
    backend._lock.release()
loader.join(timeout = 30)

has_fix = "cancel_event" in (Path(BACKEND) / "core" / "inference" / "llama_cpp.py").read_text(
    encoding = "utf-8"
).split("def _wait_for_health")[1][:2000]

facts = {
    "health_timeout_s": HEALTH_TIMEOUT,
    "clicked_cancel_after_s": CLICK_AFTER,
    "cancel_landed_after_s": cancel_secs,
    "health_wait_ran_for_s": out.get("wait_secs"),
    "wait_returned_healthy": out.get("healthy"),
    "cancel_lock_acquired": bool(acquired),
    "timeout_marker_logged": any("health check timed out" in ln for ln in backend._stdout_lines),
    "source_has_cancel_check": has_fix,
    "pass_threshold_s": PASS_UNDER,
}
print("PROBE FACTS " + json.dumps(facts, indent = 2))

ok = bool(acquired) and cancel_secs < PASS_UNDER
print(
    f"PROBE {'PASS' if ok else 'FAIL'}: Cancel landed after {cancel_secs}s "
    f"(threshold {PASS_UNDER}s, health timeout {HEALTH_TIMEOUT}s)"
)
raise SystemExit(0 if ok else 1)
