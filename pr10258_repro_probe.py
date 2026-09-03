#!/usr/bin/env python3
"""PR 10258 repro probe: is the export response wait an inactivity timeout or an absolute one?

Drives the REAL ExportOrchestrator._wait_response against a REAL spawned worker process over a
real multiprocessing queue on a real wall clock. Nothing about time or the queue is faked; only
the worker payload stands in for llama.cpp (hosted runners have no GPU).

Case BUSY : worker reports every GAP seconds for N_LINES lines, then finishes.
            Every gap is under the limit; the whole export is far over it.
            A busy export must NOT be killed  -> probe fails on the buggy implementation.
Case QUIET: worker says nothing at all. It must still be given up on near the limit.
            Identical on both sides; this is the control.
"""

import multiprocessing as mp
import sys
import time
import types
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent / "studio" / "backend"
sys.path.insert(0, str(_BACKEND))

_loggers = types.ModuleType("loggers")
_loggers.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers)
sys.modules.setdefault("structlog", types.ModuleType("structlog"))
_utils = types.ModuleType("utils"); _utils.__path__ = []
_paths = types.ModuleType("utils.paths"); _paths.outputs_root = lambda: Path("/tmp")
sys.modules.setdefault("utils", _utils)
sys.modules.setdefault("utils.paths", _paths)

TIMEOUT = 8.0     # the inactivity limit under test (stands in for the real 3600s)
GAP = 3.0         # worker reports this often: comfortably under TIMEOUT
N_LINES = 6       # ~18s of reported work: far over TIMEOUT
QUIET_HOLD = 40.0


def chatty_worker(q, n_lines, gap, done_type):
    for i in range(n_lines):
        time.sleep(gap)
        q.put({
            "type": "log", "stream": "stdout",
            "line": f"[{i + 1}/{n_lines}] llama-quantize: writing tensor block",
            "ts": time.time(),
        })
    q.put({"type": done_type, "success": True, "message": "export complete",
           "output_path": "/tmp/model.gguf"})


def quiet_worker(q, hold):
    time.sleep(hold)


def run_case(worker, args, expected_type):
    from core.export import orchestrator as orch_mod
    ctx = orch_mod._CTX
    q = ctx.Queue()
    proc = ctx.Process(target = worker, args = (q, *args), daemon = True)
    proc.start()

    orch = orch_mod.ExportOrchestrator()
    orch._resp_queue = q
    orch._proc = proc

    started = time.monotonic()
    try:
        resp = orch._wait_response(expected_type, timeout = TIMEOUT)
        outcome, detail = "completed", resp.get("type", "")
    except RuntimeError as exc:
        outcome, detail = "timed_out", str(exc)
    elapsed = time.monotonic() - started

    logs, _ = orch.get_logs_since(0)
    proc.terminate(); proc.join(timeout = 5)
    return {"outcome": outcome, "detail": detail, "elapsed": elapsed, "log_lines": len(logs)}


def main() -> int:
    import importlib
    src = _BACKEND / "core" / "export" / "orchestrator.py"
    print(f"orchestrator.py sha:  {__import__('hashlib').sha256(src.read_bytes()).hexdigest()[:16]}")
    print(f"limit={TIMEOUT}s  gap between reports={GAP}s  reports={N_LINES}"
          f"  (work spans ~{GAP * N_LINES:.0f}s)\n")

    failures = []

    busy = run_case(chatty_worker, (N_LINES, GAP, "export_gguf_done"), "export_gguf_done")
    print(f"BUSY  worker reported {busy['log_lines']} lines, none more than {GAP}s apart")
    print(f"BUSY  wait {busy['outcome']} after {busy['elapsed']:.1f}s :: {busy['detail']}")
    if busy["outcome"] != "completed":
        failures.append(
            f"REPRO: a busy export was stopped after {busy['elapsed']:.1f}s with the limit at "
            f"{TIMEOUT}s, even though it reported {busy['log_lines']} times and never went quiet "
            f"for more than {GAP}s. Downstream this fails the export and the worker is torn down "
            f"mid-write."
        )
    elif busy["elapsed"] <= TIMEOUT:
        failures.append(f"INVALID: busy case ended in {busy['elapsed']:.1f}s, never reaching the {TIMEOUT}s limit")

    quiet = run_case(quiet_worker, (QUIET_HOLD,), "export_gguf_done")
    print(f"\nQUIET worker reported {quiet['log_lines']} lines")
    print(f"QUIET wait {quiet['outcome']} after {quiet['elapsed']:.1f}s :: {quiet['detail']}")
    if quiet["outcome"] != "timed_out":
        failures.append("CONTROL BROKEN: a silent worker was waited on forever")
    elif quiet["elapsed"] > TIMEOUT * 2:
        failures.append(f"CONTROL BROKEN: silent worker took {quiet['elapsed']:.1f}s to give up (limit {TIMEOUT}s)")

    print()
    if failures:
        for line in failures:
            print(f"FAIL {line}")
        return 1
    print("PASS a busy export survives past the limit; a silent one is still given up on")
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
