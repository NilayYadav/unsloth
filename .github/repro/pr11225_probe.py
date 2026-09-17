"""Real-subprocess repro for PR 11225: cloudflared mints no URL on the first run.

Stands a fake cloudflared in front of the unmodified CloudflareTunnel/reader/start_studio_tunnel
path. Run 1 prints a failure with no URL and exits; run 2 prints a real URL and registers.
"""
import importlib.util, json, os, pathlib, stat, sys, tempfile

root = pathlib.Path(sys.argv[1]).resolve()
expect = sys.argv[2]

sys.path.insert(0, str(root / "studio/backend"))
spec = importlib.util.spec_from_file_location("ct_probe", root / "studio/backend/cloudflare_tunnel.py")
ct = importlib.util.module_from_spec(spec)
sys.modules["ct_probe"] = ct
spec.loader.exec_module(ct)

work = pathlib.Path(tempfile.mkdtemp(prefix = "pr11225-"))
counter = work / "runs"
counter.write_text("")
fake = work / "cloudflared"
fake.write_text(f"""#!/bin/sh
echo "$@" >> "{counter}"
n=$(wc -l < "{counter}" | tr -d ' ')
if [ "$n" = "1" ]; then
  echo "failed to request quick Tunnel: Post \\"https://api.trycloudflare.com/tunnel\\": context deadline exceeded"
  echo "ERR Couldn't start tunnel error=\\"failed to request quick tunnel\\""
  exit 1
fi
echo "INF +--------------------------------------------+"
echo "INF |  https://mimir-pr11225-repro.trycloudflare.com  |"
echo "INF Registered tunnel connection connIndex=0"
sleep 120
""")
fake.chmod(fake.stat().st_mode | stat.S_IEXEC)

ct.ensure_cloudflared = lambda: str(fake)
ct.verify_public_url = lambda url, **kw: True
# Keep the probe quick; the retry *count* is what is under test, not the wall clock.
if hasattr(ct, "_NO_URL_RETRY_DELAYS"):
    ct._NO_URL_RETRY_DELAYS = tuple(0.1 for _ in ct._NO_URL_RETRY_DELAYS)

url = ct.start_studio_tunnel(8080)
status = ct.get_studio_tunnel_status()
runs = len([l for l in counter.read_text().splitlines() if l.strip()])
try:
    ct.stop_studio_tunnel()
except Exception:
    pass

facts = {"expect": expect, "url": url, "state": status.get("state"), "error": status.get("error"), "cloudflared_runs": runs}
print("PROBE_FACTS " + json.dumps(facts))

if expect == "before":
    ok = url is None and status.get("state") == "error" and status.get("error") == "cloudflared did not produce a URL" and runs == 1
    why = "one cloudflared run, no retry, hard error"
else:
    ok = url == "https://mimir-pr11225-repro.trycloudflare.com" and status.get("state") == "online" and runs == 2
    why = "retried once, second run minted a URL, tunnel online"

print(("PASS " if ok else "FAIL ") + expect + ": expected " + why)
sys.exit(0 if ok else 1)
