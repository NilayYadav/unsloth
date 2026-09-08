"""PR 10551 probe: `unsloth start` must track a download for a child that prints no
early API key marker, without minting a key on a normal launch that is about to."""

import pytest
import typer

import unsloth_cli.commands.start as start_cli


BASE = "http://127.0.0.1:8888"
MODEL = "unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF"
KEY_LINE = f"{start_cli._START_API_KEY_PREFIX}sk-unsloth-test\n"
EXPECTED_BYTES = 500 * 1024**3
STEP_S = 120.0
MAX_ITERATIONS = 5000


class FakeClock:
    def __init__(self, step):
        self.now = 1000.0
        self.start = 1000.0
        self.step = step

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += self.step

    @property
    def elapsed(self):
        return self.now - self.start


class FakePopen:
    def poll(self):
        return None


class Harness:
    def __init__(self, monkeypatch, *, tail, healthy, startup_key = None,
                 chunk_bytes = 0, marker_at = None, ready_at = None,
                 startup_key_at = None, flap_every = 0, healthy_until = None,
                 step = STEP_S):
        self.clock = FakeClock(step)
        self.tail = tail
        self.healthy = healthy
        self.startup_key = startup_key
        self.startup_key_at = startup_key_at
        self.flap_every = flap_every
        self.healthy_until = healthy_until
        self.chunk_bytes = chunk_bytes
        self.marker_at = marker_at
        self.ready_at = ready_at
        self.downloaded_bytes = 0
        self.mints = 0
        self.calls = []
        self.polls = 0
        self.iterations = 0
        self.shutdowns = []
        self.server = FakePopen()
        monkeypatch.setattr(start_cli, "time", self.clock)
        monkeypatch.setattr(start_cli, "_http_json", self.http_json)
        monkeypatch.setattr(start_cli, "_log_tail", lambda path, lines = 20: self.tail)
        monkeypatch.setattr(start_cli, "_studio_healthy", self.studio_healthy)
        # Absent on main, where the whole minting path does not exist yet.
        monkeypatch.setattr(start_cli, "_startup_api_key", self.startup_api_key, raising = False)
        monkeypatch.setattr(start_cli, "_shutdown_server", self.shutdowns.append)
        monkeypatch.setattr(start_cli, "_auto_served_server", None)
        monkeypatch.setattr(start_cli.atexit, "register", lambda *a, **k: None)
        monkeypatch.setattr(start_cli.subprocess, "Popen", lambda *a, **k: self.server)

    def http_json(self, method, url, token, payload = None, timeout = 30, error = None):
        if "gguf-variants" in url:
            return {
                "default_variant": "Q4_K_M",
                "variants": [{"quant": "Q4_K_M", "download_size_bytes": EXPECTED_BYTES}],
            }
        if "download-progress" in url:
            self.polls += 1
            self.calls.append("progress")
            self.downloaded_bytes += self.chunk_bytes
            return {
                "downloaded_bytes": self.downloaded_bytes,
                "expected_bytes": EXPECTED_BYTES,
                "progress": self.downloaded_bytes / EXPECTED_BYTES,
                "cache_measured": True,
            }
        raise AssertionError(f"unexpected request: {method} {url}")

    def studio_healthy(self, base, timeout = 3.0):
        self.calls.append("health")
        self.iterations += 1
        assert self.iterations <= MAX_ITERATIONS, "readiness loop never terminates"
        if self.flap_every and self.iterations % self.flap_every == 0:
            return False
        if self.healthy_until is not None and self.iterations > self.healthy_until:
            if self.ready_at is not None and self.iterations >= self.ready_at:
                self.tail = f"{KEY_LINE}Model loaded: {MODEL}\n"
                return True
            return False
        if self.marker_at is not None and self.iterations >= self.marker_at:
            self.tail = KEY_LINE
        if self.ready_at is not None and self.iterations >= self.ready_at:
            self.tail = f"{KEY_LINE}Model loaded: {MODEL}\n"
            return True
        return self.healthy

    def startup_api_key(self, base):
        self.mints += 1
        if self.startup_key_at is not None and self.iterations < self.startup_key_at:
            return None
        return self.startup_key

    def start(self):
        return start_cli._start_studio_server(BASE, MODEL, start_cli.LoadOptions())


def test_a_download_behind_an_unmarked_child_is_tracked_to_ready(monkeypatch):
    """A: the defect. Healthy child, no early key line, 17-minute download."""
    harness = Harness(
        monkeypatch,
        tail = "starting\n",
        healthy = True,
        startup_key = "sk-unsloth-minted",
        chunk_bytes = 1024**3,
        ready_at = 40,
        marker_at = None,
    )

    server = harness.start()

    assert server is harness.server, "the loop returned no server"
    assert harness.shutdowns == [], "the server was killed mid-download"
    assert harness.polls >= 30, f"download progress was never polled ({harness.polls})"
    assert harness.clock.elapsed > start_cli._SERVER_START_TIMEOUT_S


def test_a_normal_launch_mints_no_key_of_its_own(monkeypatch):
    """B: the regression guard. The child echoes its marker one poll after health."""
    harness = Harness(
        monkeypatch,
        tail = "starting\n",
        healthy = True,
        startup_key = "sk-unsloth-minted",
        chunk_bytes = 1024**3,
        marker_at = 1,
        ready_at = 40,
    )

    server = harness.start()

    assert server is harness.server
    assert harness.mints == 0, f"a normal launch minted {harness.mints} extra key(s)"


def test_minting_is_retried_at_a_slow_cadence_not_every_poll(monkeypatch):
    """C: not once per sleep, and not retired after a handful of early failures."""
    harness = Harness(monkeypatch, tail = "starting\n", healthy = True, step = 2.0)

    with pytest.raises(typer.Exit):
        harness.start()

    passes = harness.iterations
    assert passes > 400, f"the loop only ran {passes} passes, so nothing was throttled"
    ceiling = harness.clock.elapsed / 30.0 + 2
    assert 3 < harness.mints <= ceiling, f"{harness.mints} mint attempts over {passes} passes"


def test_auth_that_settles_after_the_health_gate_still_starts_progress(monkeypatch):
    """D: /api/health can open before /api/auth is usable (Codex P2 on this PR)."""
    harness = Harness(
        monkeypatch,
        tail = "starting\n",
        healthy = True,
        startup_key = "sk-unsloth-minted",
        startup_key_at = 100,
        chunk_bytes = 1024**3,
        ready_at = 500,
        step = 2.0,
    )

    server = harness.start()

    assert server is harness.server
    assert harness.shutdowns == [], "the server was killed after auth recovered"
    assert harness.polls > 0
    assert harness.clock.elapsed > start_cli._SERVER_START_TIMEOUT_S


def test_a_health_probe_that_times_out_under_load_does_not_retire_minting(monkeypatch):
    """E: the 3s health probe can miss a pass while the disk is saturated."""
    harness = Harness(
        monkeypatch,
        tail = "starting\n",
        healthy = True,
        startup_key = "sk-unsloth-minted",
        chunk_bytes = 1024**3,
        flap_every = 2,
        ready_at = 41,
    )

    server = harness.start()

    assert server is harness.server
    assert harness.shutdowns == [], "a missed health poll retired minting"
    assert harness.mints == 1
    assert harness.polls > 0


def test_one_healthy_answer_is_enough_to_start_minting(monkeypatch):
    """F: health answers once, then times out under download pressure for 15 minutes."""
    harness = Harness(
        monkeypatch,
        tail = "starting\n",
        healthy = True,
        healthy_until = 1,
        startup_key = "sk-unsloth-minted",
        chunk_bytes = 1024**3,
        ready_at = 500,
        step = 2.0,
    )

    server = harness.start()

    assert server is harness.server
    assert harness.shutdowns == [], "one missed probe run killed a live download"
    assert harness.mints == 1
    assert harness.polls > 0
    assert harness.clock.elapsed > start_cli._SERVER_START_TIMEOUT_S


def test_readiness_is_declared_on_a_health_answer_taken_after_the_progress_request(monkeypatch):
    """G: the health answer readiness is declared on must not predate the progress request."""
    harness = Harness(
        monkeypatch,
        tail = "starting\n",
        healthy = True,
        startup_key = "sk-unsloth-minted",
        chunk_bytes = 1024**3,
        ready_at = 10,
    )

    server = harness.start()

    assert server is harness.server
    assert harness.polls > 0
    assert harness.calls[-1] == "health", f"readiness read a stale probe: {harness.calls[-3:]}"


def test_a_stalled_catalog_check_still_mints_a_startup_key(monkeypatch, tmp_path):
    """H: /v1/models scans the disk the download is saturating (Codex P2 on this PR)."""
    mint = getattr(start_cli, "_startup_api_key", None)
    assert mint is not None, "this revision has no startup mint path at all"
    calls = []

    def http_json(method, url, token, payload = None, timeout = 30, error = None):
        calls.append((url, timeout))
        if url.endswith("/v1/models"):
            raise TimeoutError("the catalog scan did not finish")
        if url.endswith("/api/auth/api-keys"):
            return {"key": "sk-unsloth-minted"}
        raise AssertionError(f"unexpected request: {method} {url}")

    cache = tmp_path / "agent_api_key.json"
    monkeypatch.setattr(start_cli, "verify_studio_identity", lambda base: True)
    monkeypatch.setattr(start_cli, "_studio_token", lambda: "jwt-token")
    monkeypatch.setattr(start_cli, "_key_cache_path", lambda: cache)
    monkeypatch.setattr(start_cli, "_http_json", http_json)
    start_cli._remember_key(cache, BASE, "sk-unsloth-cached", "minted")

    assert mint(BASE) == "sk-unsloth-minted", f"a stalled check retired the mint: {calls}"
    assert calls[0] == (f"{BASE}/v1/models", 10), f"catalog check budget: {calls[0]}"
