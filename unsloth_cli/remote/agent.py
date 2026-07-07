# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""HTTP client for the remote agent (a headless Studio backend behind an SSH tunnel).

Stdlib-only (urllib): the CLI must not need requests/httpx just to talk to a
localhost-forwarded port.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Iterator, Optional

from unsloth_cli.remote import RemoteError

_DEFAULT_TIMEOUT = 20
# SSE heartbeats arrive every few seconds; anything past this means the stream died.
_SSE_READ_TIMEOUT = 90
_SSE_MAX_RECONNECTS = 30
_SSE_RECONNECT_DELAY = 3.0


class RemoteBusyError(RemoteError):
    """The remote is already running a training job."""

    def __init__(self, message: str, active_job_id: str = ""):
        super().__init__(
            message,
            hint = (
                f"Watch it with: unsloth logs {active_job_id} -f" if active_job_id
                else "Check it with: unsloth remote status <name>"
            ),
        )
        self.active_job_id = active_job_id


@dataclass
class SSEEvent:
    event: str
    data: dict = field(default_factory = dict)
    event_id: Optional[str] = None


class AgentClient:
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def _headers(self, streaming: bool = False) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if streaming:
            headers["Accept"] = "text/event-stream"
        return headers

    def _request(
        self,
        method: str,
        path: str,
        body: Optional[dict] = None,
        timeout: int = _DEFAULT_TIMEOUT,
    ) -> dict:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(
            f"{self.base_url}{path}", data = data, headers = self._headers(), method = method,
        )
        try:
            with urllib.request.urlopen(request, timeout = timeout) as response:
                text = response.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors = "replace")
            try:
                detail = json.loads(detail).get("detail", detail)
            except (json.JSONDecodeError, AttributeError):
                pass
            if e.code in (401, 403):
                raise RemoteError(
                    "The agent rejected our API key.",
                    hint = "Re-run `unsloth remote add` to refresh it, or check remotes.toml.",
                ) from None
            raise RemoteError(f"Agent returned HTTP {e.code} for {path}: {detail}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            raise RemoteError(
                f"Could not reach the agent at {self.base_url}{path}: {e}",
                hint = "Is the remote up? Check with: unsloth remote doctor <name>",
            ) from None
        return json.loads(text) if text else {}

    # -- health / hardware ------------------------------------------------------

    def health(self) -> dict:
        return self._request("GET", "/api/health")

    def hardware(self) -> dict:
        return self._request("GET", "/api/train/hardware")

    # -- training ---------------------------------------------------------------

    def train_status(self) -> dict:
        return self._request("GET", "/api/train/status")

    def start_training(self, payload: dict) -> dict:
        try:
            response = self._request("POST", "/api/train/start", body = payload, timeout = 60)
        except RemoteError as e:
            # The inference-in-flight guard rejects with HTTP 409.
            if "HTTP 409" in str(e):
                raise RemoteBusyError(str(e)) from None
            raise
        # A busy trainer is reported as a 200 with a body-level error.
        if response.get("status") == "error":
            if "already" in (response.get("error") or "").lower() + (response.get("message") or "").lower():
                raise RemoteBusyError(
                    response.get("message") or "Training is already in progress.",
                    active_job_id = response.get("job_id") or "",
                )
            raise RemoteError(response.get("message") or response.get("error") or "Training failed to start.")
        return response

    def stop_training(self, save: bool = True) -> dict:
        return self._request("POST", "/api/train/stop", body = {"save": save}, timeout = 120)

    def list_runs(self, limit: int = 50, skip: int = 0) -> dict:
        return self._request("GET", f"/api/train/runs?skip={skip}&limit={limit}")

    def run_detail(self, run_id: str) -> dict:
        return self._request("GET", f"/api/train/runs/{run_id}")

    # -- export -----------------------------------------------------------------

    def export_load_checkpoint(self, checkpoint_path: str) -> dict:
        return self._request(
            "POST", "/api/export/load-checkpoint",
            body = {"checkpoint_path": checkpoint_path}, timeout = 600,
        )

    def export_gguf(self, save_directory: str, quantization: str) -> dict:
        return self._request(
            "POST", "/api/export/gguf",
            body = {"save_directory": save_directory, "quantization": quantization},
            timeout = 60,
        )

    def export_status(self) -> dict:
        return self._request("GET", "/api/export/status")

    # -- SSE progress stream ------------------------------------------------------

    def stream_progress(
        self,
        last_event_id: Optional[str] = None,
        max_reconnects: int = _SSE_MAX_RECONNECTS,
    ) -> Iterator[SSEEvent]:
        """Yield progress events, transparently reconnecting with Last-Event-ID.

        Terminates after a `complete` or `error` event, or when reconnection
        attempts are exhausted (raises RemoteError in that case).
        """
        reconnects = 0
        while True:
            headers = self._headers(streaming = True)
            if last_event_id is not None:
                headers["Last-Event-ID"] = str(last_event_id)
            request = urllib.request.Request(
                f"{self.base_url}/api/train/progress", headers = headers,
            )
            try:
                with urllib.request.urlopen(request, timeout = _SSE_READ_TIMEOUT) as response:
                    reconnects = 0
                    for event in _parse_sse(response):
                        if event.event_id is not None:
                            last_event_id = event.event_id
                        yield event
                        if event.event in ("complete", "error"):
                            return
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
                pass
            reconnects += 1
            if reconnects > max_reconnects:
                raise RemoteError(
                    "Lost the progress stream and could not reconnect.",
                    hint = "The job keeps running on the remote. Reattach with: unsloth logs <job> -f",
                )
            time.sleep(_SSE_RECONNECT_DELAY)


def _parse_sse(response) -> Iterator[SSEEvent]:
    """Parse an SSE byte stream into events (id/event/data fields)."""
    event_type = "message"
    event_id: Optional[str] = None
    data_lines: list = []
    for raw in response:
        line = raw.decode("utf-8", errors = "replace").rstrip("\r\n")
        if line == "":
            if data_lines:
                text = "\n".join(data_lines)
                try:
                    data = json.loads(text)
                except json.JSONDecodeError:
                    data = {"raw": text}
                yield SSEEvent(event = event_type, data = data, event_id = event_id)
            event_type = "message"
            data_lines = []
            continue
        if line.startswith(":"):
            continue
        if ":" in line:
            fieldname, _, value = line.partition(":")
            value = value.lstrip(" ")
        else:
            fieldname, value = line, ""
        if fieldname == "event":
            event_type = value
        elif fieldname == "data":
            data_lines.append(value)
        elif fieldname == "id":
            event_id = value
