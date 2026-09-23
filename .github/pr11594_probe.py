import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.getcwd(), "studio", "backend"))
os.chdir(os.path.join(os.getcwd(), "studio", "backend"))

logging.basicConfig(level = logging.INFO, format = "%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("probe")

import utils.hf_xet_fallback as shim
from hub.services import download_lifecycle as dl
from hub.utils import download_registry

SCENARIO = sys.argv[1]
REPO = os.environ.get("PROBE_REPO", "unsloth/Qwen3-0.6B")

assert shim._load_shared(), f"real unsloth_zoo watchdog not loaded: {shim._shared_import_error!r}"
zoo = shim._shared
log.info("zoo watchdog module: %s", zoo.__file__)

if SCENARIO == "frozen-sensors":
    real_sizes = zoo._active_incomplete_blob_sizes
    first_seen = {}

    def frozen_sizes(*a, **k):
        sizes = real_sizes(*a, **k)
        for name, n in sizes.items():
            if n and name not in first_seen:
                first_seen[name] = n
        return {name: first_seen.get(name, n) for name, n in sizes.items()}

    zoo._active_incomplete_blob_sizes = frozen_sizes
    REAL_SIZES = real_sizes
    zoo._child_rss = lambda pid: 1
    os.environ.pop("UNSLOTH_HF_XET_FORCE_STALL", None)
else:
    os.environ["UNSLOTH_HF_XET_FORCE_STALL"] = "1"

registry = download_registry.DownloadRegistry()
key = download_registry.normalize_job_key(REPO)
stalled = []
lifecycle_log = logging.getLogger("lifecycle")
rearms = []


class _Count(logging.Handler):
    def emit(self, record):
        if "still receiving data" in record.getMessage():
            rearms.append(time.monotonic())


lifecycle_log.addHandler(_Count())

t0 = time.monotonic()
proc = dl.spawn_worker(["--repo-id", REPO], None, use_xet = True, allow_ambient_token = False)
has_heartbeat = "--heartbeat" in proc.args
timeline = []


def _sample():
    import threading

    from hub.utils import download_heartbeat as hb

    beat = proc.args[proc.args.index("--heartbeat") + 1] if has_heartbeat else None
    while proc.poll() is None:
        real = None
        if SCENARIO == "frozen-sensors":
            try:
                real = sum(REAL_SIZES("model", REPO, None).values())
            except Exception as exc:  # noqa: BLE001
                real = f"err {exc}"
        timeline.append((round(time.monotonic() - t0, 1), hb.read(beat), None if hb.age(beat) is None else round(hb.age(beat), 1), real))
        time.sleep(0.5)


import threading

threading.Thread(target = _sample, daemon = True).start()
stop = dl._start_stall_watchdog(
    registry,
    key,
    proc,
    repo_type = "model",
    repo_id = REPO,
    label = REPO,
    log_prefix = "Download",
    logger = lifecycle_log,
    on_stall = stalled.append,
)
assert stop is not None, "watchdog did not start"
try:
    rc = proc.wait(timeout = float(os.environ.get("PROBE_TIMEOUT", "600")))
except Exception:
    proc.kill()
    rc = proc.wait()
stop.set()
elapsed = time.monotonic() - t0
stderr = (proc.stderr.read() or b"").decode("utf-8", "replace")[-1500:]

from huggingface_hub import scan_cache_dir

complete_bytes = 0
for repo in scan_cache_dir().repos:
    if repo.repo_id == REPO:
        complete_bytes = repo.size_on_disk

result = {
    "scenario": SCENARIO,
    "repo": REPO,
    "stall_timeout_s": float(os.environ.get("UNSLOTH_XET_STALL_TIMEOUT", "30")),
    "worker_has_heartbeat_arg": has_heartbeat,
    "worker_rc": rc,
    "stall_verdicts_acted_on": len(stalled),
    "verdicts_overruled_by_heartbeat": len(rearms),
    "elapsed_s": round(elapsed, 1),
    "bytes_in_cache": complete_bytes,
}
print("PROBE_RESULT " + json.dumps(result), flush = True)
print("TIMELINE t_s heartbeat_bytes heartbeat_age_s real_partial_bytes", flush = True)
for row in timeline:
    print("TL", *row, flush = True)
if rc != 0:
    print("worker stderr tail:\n" + stderr, flush = True)

expect = os.environ["PROBE_EXPECT"]
if expect == "survives":
    ok = rc == 0 and not stalled and complete_bytes > 0
elif expect == "killed":
    ok = rc != 0 and len(stalled) == 1
else:
    raise SystemExit(f"bad PROBE_EXPECT {expect}")
print(("PASS" if ok else "FAIL") + f" expect={expect} " + json.dumps(result), flush = True)
sys.exit(0 if ok else 1)
