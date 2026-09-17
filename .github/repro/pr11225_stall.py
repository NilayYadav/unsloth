"""Startup stall on the synchronous pre-banner start_studio_tunnel() call, for one failure profile."""
import importlib.util, sys, pathlib
root = pathlib.Path(sys.argv[1]).resolve()
profile = [float(x) for x in sys.argv[2].split(",")]
sys.path.insert(0, str(root / "studio/backend"))
spec = importlib.util.spec_from_file_location("ct", root / "studio/backend/cloudflare_tunnel.py")
ct = importlib.util.module_from_spec(spec); sys.modules["ct"] = ct; spec.loader.exec_module(ct)

attempts, clock = [], [0.0]
class _Stub:
    def __init__(self, port, binary, protocol=None, origin_host="localhost"):
        self.url = None; attempts.append(protocol)
    def start(self): pass
    def wait_for_ready(self, timeout):
        i = min(len(attempts) - 1, len(profile) - 1)
        clock[0] += min(profile[i], timeout); return None
    def stop(self): pass
ct.ensure_cloudflared = lambda: "/bin/cloudflared"
ct.CloudflareTunnel = _Stub
ct.verify_public_url = lambda url, **kw: True
ct.time.monotonic = lambda: clock[0]
ct.time.sleep = lambda s: clock.__setitem__(0, clock[0] + s)
if hasattr(ct, "_wait_before_retry"):  # the retry delay moved off time.sleep onto a cancellable wait
    ct._wait_before_retry = lambda d: clock.__setitem__(0, clock[0] + d) or False
assert ct.start_studio_tunnel(8080) is None
print(f"  ready_timeout={ct._READY_TIMEOUT} delays={getattr(ct,'_NO_URL_RETRY_DELAYS',None)} "
      f"budget={getattr(ct,'_NO_URL_RETRY_BUDGET','n/a')} -> attempts={len(attempts)} stall={clock[0]}s")
