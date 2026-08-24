"""PR 9634 probe: print real observable values for a Stop-loading vs a real worker crash.

Never asserts and always exits 0 -- the pytest step is the A/B discriminator. This only
reports what load_model() actually does on each branch so the two runs can be compared.
"""
import sys
import threading
import types

sys.path.insert(0, ".")

from core.inference import orchestrator as orch_mod
from core.inference.orchestrator import InferenceOrchestrator
from utils import transformers_version as tv

CRASH_MSG = (
    "The inference worker stopped unexpectedly while loading the model. "
    "Details: process missing."
)


def _bare():
    o = InferenceOrchestrator.__new__(InferenceOrchestrator)
    o._gen_lock = threading.Lock()
    o._send_order_lock = threading.Lock()
    o._active_cancel_lock = threading.Lock()
    o._active_cancel_events = []
    o._executing_cancel_events = []
    o._cancel_event = threading.Event()
    o._drain_event = threading.Event()
    o._proc = object()
    o._cmd_queue = object()
    o._resp_queue = object()
    o._dispatcher_thread = None
    o._dispatcher_stop = threading.Event()
    o._dispatcher_lifecycle_lock = threading.Lock()
    o._unload_pending = False
    o._exclusive_tts_pending = False
    o.active_model_name = None
    o.models = {}
    o.loading_models = set()
    return o


def run(cancelled):
    o = _bare()
    o._proc = None
    o._ensure_subprocess_alive = lambda: False
    o._shutdown_subprocess = lambda *a, **k: None
    o._spawn_subprocess = lambda cfg: None
    tv.needs_transformers_5 = lambda name: False
    orch_mod.prepare_gpu_selection = lambda gpu_ids, **k: ([0], "sel")

    def _dead(expected, timeout=300.0):
        if cancelled:
            # cancel_load discards the loading marker, THEN kills the worker.
            o.loading_models.discard("m")
        raise RuntimeError(CRASH_MSG)

    o._wait_response = _dead
    cfg = types.SimpleNamespace(identifier="m", gguf_variant=None)

    label = "STOP-LOADING (user pressed Stop)" if cancelled else "REAL WORKER CRASH"
    print(f"--- scenario: {label} ---")
    try:
        result = o.load_model(cfg)
        print(f"  outcome           = returned {result!r}")
    except Exception as exc:
        print(f"  outcome           = raised {type(exc).__name__}: {exc}")
    print(f"  active_model_name = {o.active_model_name!r}")
    print(f"  models            = {o.models!r}")
    print(f"  loading_models    = {sorted(o.loading_models)!r}")
    print()


run(cancelled=True)
run(cancelled=False)
