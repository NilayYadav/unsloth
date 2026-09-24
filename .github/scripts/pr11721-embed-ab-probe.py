#!/usr/bin/env python3
"""PR 11721 A/B probe: long inputs to /v1/embeddings on a MEAN-pooling embedding GGUF.

Identical on both branches. Gating assertion: a ~1.7k-token input to nomic-embed-text-v1.5
(native context 2048) returns HTTP 200 with a 768-d vector. The fix-reverted branch is
expected to fail exactly there. bge-m3 (CLS, 8192) is measured but never gates.
Never prints passwords, tokens or keys.
"""

from __future__ import annotations

import asyncio
import html
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import httpx
from studio_test_kit.auth import login

NOMIC = ("nomic-ai/nomic-embed-text-v1.5-GGUF", "Q4_K_M", 768)
BGE = ("gpustack/bge-m3-GGUF", "Q4_K_M", 1024)
SENTENCE = (
    "Studio loads an embedding model and every document chunk is turned into one vector "
    "that the retrieval index compares against the query. "
)


def text_of(sentences: int) -> str:
    return "".join(f"{i}. {SENTENCE}" for i in range(sentences))


INPUTS = {"short": text_of(1), "medium": text_of(30), "long": text_of(55)}
BGE_INPUT = text_of(220)


def log(msg: str) -> None:
    print(msg, flush=True)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def find_bin(home: Path) -> Path:
    for c in (home / "bin" / "unsloth", home / "unsloth_studio" / "bin" / "unsloth"):
        if c.is_file():
            return c
    raise SystemExit(f"FAIL no unsloth CLI under {home}")


def wait_health(base: str, timeout: int = 300) -> None:
    end = time.time() + timeout
    while time.time() < end:
        try:
            with urllib.request.urlopen(f"{base}/api/health", timeout=3) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(2)
    raise SystemExit("FAIL Studio never became healthy")


def bootstrap_password(home: Path) -> str:
    p = home / "auth" / ".bootstrap_password"
    for _ in range(60):
        if p.is_file() and p.read_text().strip():
            return p.read_text().strip()
        time.sleep(1)
    raise SystemExit("FAIL no bootstrap password")


def llama_servers() -> list[dict]:
    out = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            argv = (p / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        argv = [a.decode(errors="replace") for a in argv if a]
        if argv and Path(argv[0]).name == "llama-server":
            out.append({"pid": int(p.name), "argv": argv})
    return out


def flag(argv: list[str], *names: str):
    val = None
    for i, a in enumerate(argv):
        if a in names and i + 1 < len(argv):
            val = argv[i + 1]
    return val


def rss_mib(pid: int) -> float | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 1024, 1)
    except OSError:
        return None
    return None


class Client:
    def __init__(self, base: str, token: str):
        self.base, self.h = base, {"Authorization": f"Bearer {token}"}

    def call(self, method: str, path: str, body: dict | None = None, timeout: float = 900):
        r = httpx.request(method, f"{self.base}{path}", json=body, headers=self.h, timeout=timeout)
        try:
            data = r.json()
        except ValueError:
            data = {"text": r.text[:2000]}
        return r.status_code, data


def load(c: Client, repo: str, variant: str) -> dict:
    code, est = c.call("POST", "/api/inference/estimate-memory",
                       {"model_path": repo, "gguf_variant": variant})
    estimate = {k: est.get(k) for k in ("available", "n_ctx", "compute_bytes", "kv_bytes")} if code == 200 else {
        "status": code, "body": est}
    t0 = time.time()
    code, body = c.call("POST", "/api/inference/load",
                        {"model_path": repo, "gguf_variant": variant, "max_seq_length": 0}, timeout=2400)
    if code != 200:
        raise SystemExit(f"FAIL load {repo}: HTTP {code} {json.dumps(body)[:800]}")
    end = time.time() + 1500
    while time.time() < end:
        code, st = c.call("GET", "/api/inference/status")
        if code == 200 and not st.get("loading") and st.get("is_gguf") and repo in json.dumps(st):
            break
        time.sleep(3)
    servers = llama_servers()
    if not servers:
        raise SystemExit(f"FAIL no llama-server running after loading {repo}")
    argv = servers[-1]["argv"]
    return {
        "repo": repo, "variant": variant, "load_seconds": round(time.time() - t0),
        "estimate": estimate, "pid": servers[-1]["pid"],
        "launch": {"ctx": flag(argv, "-c", "--ctx-size"), "batch": flag(argv, "-b", "--batch-size"),
                   "ubatch": flag(argv, "-ub", "--ubatch-size"), "embedding": "--embedding" in argv,
                   "parallel": flag(argv, "-np", "--parallel")},
        "rss_mib_idle": rss_mib(servers[-1]["pid"]),
    }


def embed(c: Client, model: str, text: str, pid: int | None = None) -> dict:
    peak = {"v": 0.0}
    stop = threading.Event()

    def watch():
        while not stop.is_set() and pid:
            v = rss_mib(pid) or 0.0
            peak["v"] = max(peak["v"], v)
            time.sleep(0.2)

    t = threading.Thread(target=watch, daemon=True)
    t.start()
    t0 = time.time()
    code, body = c.call("POST", "/v1/embeddings", {"model": model, "input": text}, timeout=900)
    stop.set()
    t.join()
    vec = ((body.get("data") or [{}])[0].get("embedding") or []) if code == 200 else []
    err = None if code == 200 else (body.get("error") or body.get("detail") or body)
    if isinstance(err, dict):
        err = err.get("message") or json.dumps(err)
    return {"status": code, "dims": len(vec), "prompt_tokens": (body.get("usage") or {}).get("prompt_tokens"),
            "seconds": round(time.time() - t0, 2), "error": (str(err)[:400] if err else None),
            "chars": len(text), "rss_peak_mib": peak["v"] or None}


def card(facts: dict) -> str:
    rows = "".join(
        f"<tr class='{'ok' if r['status'] == 200 else 'bad'}'><td>{html.escape(n)}</td><td>{r['prompt_tokens'] or '-'}</td>"
        f"<td>HTTP {r['status']}</td><td>{r['dims'] or '-'}</td><td>{html.escape(r['error'] or '')}</td></tr>"
        for n, r in facts["nomic"]["embeddings"].items())
    L = facts["nomic"]["launch"]
    e = facts["nomic"]["estimate"]
    return (
        "<html><body style='font:14px -apple-system,Segoe UI,sans-serif;margin:20px;width:900px'>"
        "<style>table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:6px 8px;text-align:left}"
        "th{background:#f3f3f3}.ok td:nth-child(3){color:#11702b;font-weight:600}.bad td:nth-child(3){color:#b3261e;font-weight:600}"
        "code{background:#f3f3f3;padding:1px 4px}</style>"
        f"<h2>{html.escape(facts['label'])}: POST /v1/embeddings (real responses)</h2>"
        f"<p>Branch <code>{html.escape(facts['branch'])}</code> @ <code>{facts['sha'][:10]}</code>, "
        f"model <b>{NOMIC[0]}</b> {NOMIC[1]} (MEAN pooling, native context 2048), loaded via "
        "<code>/api/inference/load</code> with Auto context.</p>"
        f"<p>llama-server launched with <code>-c {L['ctx']}</code> <code>--batch-size {L['batch'] or '(default)'}</code> "
        f"<code>--ubatch-size {L['ubatch'] or '(default 512)'}</code>; pre-load estimate compute "
        f"{(e.get('compute_bytes') or 0) / 2**20:.1f} MiB.</p>"
        "<table><tr><th>input</th><th>tokens</th><th>result</th><th>dims</th><th>error</th></tr>"
        f"{rows}</table></body></html>")


async def main() -> None:
    home = Path(os.environ["UNSLOTH_STUDIO_HOME"]).resolve()
    art = Path(os.environ.get("STUDIO_ARTIFACT_DIR", "artifacts")).resolve()
    art.mkdir(parents=True, exist_ok=True)
    branch = os.environ.get("GITHUB_REF_NAME", "local")
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    impl = {f: subprocess.check_output(["git", "hash-object", f], text=True).strip() for f in (
        "studio/backend/core/inference/llama_cpp.py", "studio/backend/routes/inference.py")}
    label = "FIX REVERTED (expected to fail)" if "reverted" in branch else "PR HEAD"
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    log_path = art / "studio.log"
    env = dict(os.environ, UNSLOTH_STUDIO_HOME=str(home))
    with log_path.open("w") as lf:
        proc = subprocess.Popen([str(find_bin(home)), "studio", "-H", "127.0.0.1", "-p", str(port)],
                                stdout=lf, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    facts: dict = {"label": label, "branch": branch, "sha": sha, "impl_blobs": impl}
    try:
        wait_health(base)
        pw = bootstrap_password(home)
        auth = await login(base, "unsloth", pw)
        token = auth.access_token
        if auth.must_change_password:
            r = httpx.post(f"{base}/api/auth/change-password", headers={"Authorization": f"Bearer {token}"},
                           json={"current_password": pw, "new_password": "UnslothStudioCI2026!"}, timeout=30)
            r.raise_for_status()
            token = r.json()["access_token"]
        c = Client(base, token)

        nomic = load(c, NOMIC[0], NOMIC[1])
        nomic["embeddings"] = {k: embed(c, NOMIC[0], v, nomic["pid"]) for k, v in INPUTS.items()}
        nomic["rss_mib_after"] = rss_mib(nomic["pid"])
        facts["nomic"] = nomic
        log("NOMIC " + json.dumps(nomic))

        try:
            bge = load(c, BGE[0], BGE[1])
            bge["embeddings"] = {"short": embed(c, BGE[0], INPUTS["short"], bge["pid"]),
                                 "very_long": embed(c, BGE[0], BGE_INPUT, bge["pid"])}
            bge["rss_mib_after"] = rss_mib(bge["pid"])
            facts["bge_m3"] = bge
            log("BGE " + json.dumps(bge))
        except BaseException as exc:  # noqa: BLE001 -- measurement only, never gates
            facts["bge_m3"] = {"error": f"{type(exc).__name__}: {exc}"}
            log(f"WARN bge-m3 measurement failed: {exc}")

        from playwright.async_api import async_playwright

        async with async_playwright() as pw_:
            b = await pw_.chromium.launch()
            page = await b.new_page(viewport={"width": 960, "height": 400})
            await page.set_content(card(facts))
            await page.screenshot(path=str(art / "embeddings-card.png"), full_page=True)
            await b.close()
    finally:
        (art / "facts.json").write_text(json.dumps(facts, indent=2))
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except OSError:
            pass

    e = facts["nomic"]["embeddings"]
    if e["short"]["status"] != 200 or e["short"]["dims"] != NOMIC[2]:
        raise SystemExit(f"FAIL harness: short input did not embed: {e['short']}")
    log(f"PASS short input embeds ({e['short']['prompt_tokens']} tokens, {e['short']['dims']} dims)")
    for k in ("medium", "long"):
        if e[k]["status"] != 200 or e[k]["dims"] != NOMIC[2]:
            print(f"FAIL {k} input ({e[k]['prompt_tokens']} tokens): HTTP {e[k]['status']} {e[k]['error']}",
                  file=sys.stderr, flush=True)
            raise SystemExit(1)
        log(f"PASS {k} input embeds ({e[k]['prompt_tokens']} tokens, {e[k]['dims']} dims)")


if __name__ == "__main__":
    asyncio.run(main())
