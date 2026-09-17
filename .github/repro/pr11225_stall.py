import importlib.util, sys, pathlib
root = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root / "studio/backend"))
spec = importlib.util.spec_from_file_location("ct", root / "studio/backend/cloudflare_tunnel.py")
ct = importlib.util.module_from_spec(spec); sys.modules["ct"] = ct; spec.loader.exec_module(ct)

attempts, waits, clock = [], [], [0.0]
class _Stub:  # a network that swallows the request: every attempt waits out the full timeout
    def __init__(self, port, binary, protocol=None, origin_host="localhost"):
        self.url = None; attempts.append(protocol)
    def start(self): pass
    def wait_for_ready(self, timeout):
        waits.append(timeout); clock[0] += timeout; return None
    def stop(self): pass
ct.ensure_cloudflared = lambda: "/bin/cloudflared"
ct.CloudflareTunnel = _Stub
ct.verify_public_url = lambda url, **kw: True
ct.time.monotonic = lambda: clock[0]
ct.time.sleep = lambda s: clock.__setitem__(0, clock[0] + s)

assert ct.start_studio_tunnel(8080) is None
print(f"_READY_TIMEOUT={ct._READY_TIMEOUT} delays={getattr(ct,'_NO_URL_RETRY_DELAYS',None)} budget={getattr(ct,'_NO_URL_RETRY_BUDGET','n/a')}")
print(f"attempts={len(attempts)} waits={waits}")
print(f"pre-banner stall on a network that swallows the request = {clock[0]}s")
