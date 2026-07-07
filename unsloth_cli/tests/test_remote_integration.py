# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Integration test: AgentClient against a real in-process Studio backend.

Gated behind UNSLOTH_REMOTE_IT=1 because it boots the actual FastAPI app
(requires the Studio backend dependencies; no GPU or model download needed).
Validates the live API contract the remote client codes against: route paths,
auth behavior, response shapes, and the SSE stream framing.
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from unsloth_cli.remote import RemoteError
from unsloth_cli.remote.agent import AgentClient

pytestmark = pytest.mark.skipif(
    os.environ.get("UNSLOTH_REMOTE_IT") != "1",
    reason = "integration test; set UNSLOTH_REMOTE_IT=1 to run",
)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope = "module")
def live_agent(tmp_path_factory):
    """Boot the Studio backend headless in-process; yield (base_url, api_key)."""
    home = tmp_path_factory.mktemp("studio-home")
    os.environ["UNSLOTH_STUDIO_HOME"] = str(home)

    backend_dir = _REPO_ROOT / "studio" / "backend"
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))

    uvicorn = pytest.importorskip("uvicorn")
    try:
        from main import app  # the Studio FastAPI app
        from auth import storage
    except Exception as e:  # missing backend deps
        pytest.skip(f"studio backend not importable: {e}")

    storage.ensure_default_admin()
    api_key, _row = storage.create_api_key(
        username = storage.DEFAULT_ADMIN_USERNAME, name = "integration-test",
    )

    port = _free_port()
    config = uvicorn.Config(app, host = "127.0.0.1", port = port, log_level = "error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target = server.run, daemon = True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/api/health", timeout = 2) as r:
                if r.status == 200:
                    break
        except Exception:
            time.sleep(0.3)
    else:
        pytest.fail("backend did not become healthy in 30s")

    yield base_url, api_key

    server.should_exit = True
    thread.join(timeout = 10)


class TestLiveAgentContract:
    def test_health(self, live_agent):
        url, key = live_agent
        assert AgentClient(url, api_key = key).health()

    def test_auth_required(self, live_agent):
        url, _ = live_agent
        with pytest.raises(RemoteError):
            AgentClient(url, api_key = None).train_status()

    def test_bad_key_rejected(self, live_agent):
        url, _ = live_agent
        with pytest.raises(RemoteError, match = "API key"):
            AgentClient(url, api_key = "sk-unsloth-wrong").train_status()

    def test_train_status_idle(self, live_agent):
        url, key = live_agent
        status = AgentClient(url, api_key = key).train_status()
        assert status["phase"] == "idle"
        assert status["is_training_running"] is False

    def test_runs_route_path_and_shape(self, live_agent):
        # Validates the /api/train/runs mount the client depends on.
        url, key = live_agent
        runs = AgentClient(url, api_key = key).list_runs()
        assert runs["runs"] == []
        assert runs["total"] == 0

    def test_hardware_route(self, live_agent):
        url, key = live_agent
        hardware = AgentClient(url, api_key = key).hardware()
        assert isinstance(hardware, dict)

    def test_invalid_start_payload_rejected(self, live_agent):
        url, key = live_agent
        with pytest.raises(RemoteError, match = "HTTP 4"):
            AgentClient(url, api_key = key).start_training({"format_type": "auto"})

    def test_sse_stream_connects_and_frames(self, live_agent):
        url, key = live_agent
        stream = AgentClient(url, api_key = key).stream_progress(max_reconnects = 0)
        event = next(stream)
        # Idle server sends the initial progress snapshot or a heartbeat.
        assert event.event in ("progress", "heartbeat")
        assert isinstance(event.data, dict)
        stream.close()
