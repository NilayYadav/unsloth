# Real uvicorn server over TCP, real anyio threadpool. Uploads a 179 MB file with the
# REAL (unstubbed) route body while a separate process-level client polls GET /ping.
import os, sys, threading, time, tempfile, pathlib, statistics

os.environ.setdefault("UNSLOTH_ALLOW_CPU", "1")
os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
os.environ.setdefault("UNSLOTH_STUDIO_DISABLE_DEVICE_PROBE", "1")

BACKEND = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(BACKEND))
home = tempfile.mkdtemp(prefix = "ragreal-")
os.environ["UNSLOTH_STUDIO_HOME"] = home
os.environ["UNSLOTH_HOME"] = home

import uvicorn, httpx
from fastapi import FastAPI
from auth.authentication import get_current_subject
from routes import rag as rag_routes
from core.rag import store
from storage import rag_db

app = FastAPI()
app.include_router(rag_routes.router, prefix = "/api/rag")
app.dependency_overrides[get_current_subject] = lambda: "tester"


@app.get("/ping")
async def ping():
    return {"ok": True}


conn = rag_db.get_connection()
try:
    KB = store.create_kb(conn, name = "RealServer")
finally:
    conn.close()

PORT = 8931
config = uvicorn.Config(app, host = "127.0.0.1", port = PORT, log_level = "error")
server = uvicorn.Server(config)
t = threading.Thread(target = server.run, daemon = True)
t.start()
for _ in range(200):
    try:
        httpx.get(f"http://127.0.0.1:{PORT}/ping", timeout = 1.0)
        break
    except Exception:
        time.sleep(0.05)

MB = 179
payload = b"alpha bravo charlie delta echo foxtrot golf hotel\n" * ((MB * 1024 * 1024) // 49)

lat, served, stop = [], 0, False


def poll():
    global served
    with httpx.Client(timeout = 120.0) as c:
        while not stop:
            t0 = time.perf_counter()
            try:
                c.get(f"http://127.0.0.1:{PORT}/ping")
            except Exception:
                break
            lat.append(time.perf_counter() - t0)
            served += 1
            time.sleep(0.005)


p = threading.Thread(target = poll, daemon = True)
p.start()
time.sleep(0.3)
n0 = served
t0 = time.perf_counter()
with httpx.Client(timeout = 900.0) as c:
    r = c.post(f"http://127.0.0.1:{PORT}/api/rag/knowledge-bases/{KB}/documents",
               files = {"file": ("big.txt", payload, "text/plain")})
total = time.perf_counter() - t0
stop = True
p.join(timeout = 5)
during = served - n0
window = [x for x in lat[n0:]]
print(f"REAL-SERVER {MB} MB: status={r.status_code} upload={total:.2f}s "
      f"pings_answered_during_upload={during} worst_ping={max(window):.3f}s "
      f"median_ping={statistics.median(window):.4f}s", flush = True)
server.should_exit = True
