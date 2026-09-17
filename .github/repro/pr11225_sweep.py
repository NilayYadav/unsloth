import importlib.util, sys, pathlib, itertools
root = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root / "studio/backend"))

def run(waits, stops):
    spec = importlib.util.spec_from_file_location("ct_s", root / "studio/backend/cloudflare_tunnel.py")
    ct = importlib.util.module_from_spec(spec); sys.modules["ct_s"] = ct; spec.loader.exec_module(ct)
    attempts, clock = [], [0.0]
    class _Stub:
        def __init__(self, port, binary, protocol=None, origin_host="localhost"):
            self.url = None; attempts.append(protocol)
        def start(self): pass
        def wait_for_ready(self, timeout):
            clock[0] += min(waits[min(len(attempts)-1, len(waits)-1)], timeout); return None
        def stop(self): clock[0] += stops[min(len(attempts)-1, len(stops)-1)]
    ct.ensure_cloudflared = lambda: "/bin/cloudflared"
    ct.CloudflareTunnel = _Stub
    ct.verify_public_url = lambda url, **kw: True
    ct.time.monotonic = lambda: clock[0]
    ct.time.sleep = lambda s: clock.__setitem__(0, clock[0] + s)
    if hasattr(ct, "_wait_before_retry"):
        ct._wait_before_retry = lambda d: clock.__setitem__(0, clock[0] + d) or False
    ct.start_studio_tunnel(8080)
    return clock[0], len(attempts), getattr(ct, "_NO_URL_RETRY_BUDGET", None)

grid = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0, 12.0, 13.0, 14.0, 15.0]
stopg = [0.0, 1.0, 5.0, 10.0]
worst = (0.0, None)
budget = None
for w in itertools.product(grid, repeat=3):
    for st in itertools.product(stopg, repeat=3):
        total, n, budget = run(list(w), list(st))
        if total > worst[0]:
            worst = (total, (w, st, n))
print(f"budget={budget}")
print(f"worst stall found = {worst[0]}s  waits={worst[1][0]} stops={worst[1][1]} attempts={worst[1][2]}")
