# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import os
import socket
import struct
import time
from typing import Optional

import httpx

from core.cluster.relay import PROBE_MAGIC, tune_socket

_BLOCK = 4 * 1024 * 1024
_MIN_PROBE_BYTES = 8 * 1024 * 1024
_MAX_PROBE_BYTES = 1024 * 1024 * 1024
_random_block: Optional[bytes] = None


class WorkerError(RuntimeError):
    def __init__(self, code: str, status: Optional[int] = None):
        super().__init__(code)
        self.code = code
        self.status = status


def format_host(host: str) -> str:
    return f"[{host}]" if ":" in host and not host.startswith("[") else host


def endpoint(host: str, port: int) -> str:
    return f"{format_host(host)}:{port}"


class WorkerClient:
    def __init__(self, host: str, control_port: int, token: Optional[str] = None, timeout: float = 3.0):
        self.host = host
        self.control_port = int(control_port)
        self.token = token
        self.timeout = timeout

    def _request(self, method: str, path: str, payload: Optional[dict] = None, timeout: Optional[float] = None):
        url = f"http://{endpoint(self.host, self.control_port)}/cluster/v1{path}"
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        try:
            # trust_env off: a configured HTTP proxy must never carry LAN cluster traffic
            with httpx.Client(timeout = timeout or self.timeout, trust_env = False) as client:
                response = client.request(method, url, json = payload, headers = headers)
        except httpx.TimeoutException as exc:
            raise WorkerError("timeout") from exc
        except httpx.HTTPError as exc:
            raise WorkerError("unreachable") from exc
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail")
            except ValueError:
                detail = None
            raise WorkerError(str(detail or f"http_{response.status_code}"), response.status_code)
        try:
            return response.json()
        except ValueError as exc:
            raise WorkerError("bad_response", response.status_code) from exc

    def hello(self) -> dict:
        data = self._request("GET", "/hello")
        if not isinstance(data, dict) or data.get("service") != "unsloth-cluster":
            raise WorkerError("not_unsloth")
        return data

    def pair(self, code: str, head_id: str, head_name: str) -> dict:
        return self._request("POST", "/pair", {"code": code, "head_id": head_id, "head_name": head_name})

    def info(self) -> dict:
        return self._request("GET", "/info")

    def lease(self, ttl: float) -> dict:
        return self._request("POST", "/lease", {"ttl": ttl})

    def release(self) -> dict:
        return self._request("POST", "/release")

    def unpair(self) -> dict:
        return self._request("POST", "/unpair")


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ConnectionError("closed")
        data += chunk
    return bytes(data)


def _probe_once(host: str, port: int, size: int, timeout: float) -> tuple[int, int, float]:
    global _random_block
    if _random_block is None:
        _random_block = os.urandom(_BLOCK)
    view = memoryview(_random_block)
    with socket.create_connection((host, port), timeout = timeout) as sock:
        tune_socket(sock)
        started = time.perf_counter()
        sock.sendall(PROBE_MAGIC + struct.pack("!Q", size))
        sent = 0
        while sent < size:
            chunk = view[: min(_BLOCK, size - sent)]
            sock.sendall(chunk)
            sent += len(chunk)
        elapsed_ns, received = struct.unpack("!QQ", _recv_exact(sock, 16))
        wall = time.perf_counter() - started
    return int(elapsed_ns), int(received), wall


def probe_data_path(host: str, port: int, timeout: float = 2.0) -> Optional[float]:
    try:
        _elapsed, _received, wall = _probe_once(host, port, 0, timeout)
    except (OSError, ConnectionError, struct.error):
        return None
    return round(wall * 1000.0, 3)


def probe_throughput(host: str, port: int, seconds: float = 1.5, timeout: float = 15.0) -> Optional[float]:
    try:
        elapsed, received, _wall = _probe_once(host, port, _MIN_PROBE_BYTES, timeout)
        if received <= 0 or elapsed <= 0:
            return None
        rate = received / (elapsed / 1e9)
        size = int(min(_MAX_PROBE_BYTES, max(_MIN_PROBE_BYTES, rate * seconds)))
        if size > _MIN_PROBE_BYTES * 2:
            elapsed, received, _wall = _probe_once(host, port, size, timeout)
            if received <= 0 or elapsed <= 0:
                return None
            rate = received / (elapsed / 1e9)
    except (OSError, ConnectionError, struct.error):
        return None
    return round(rate * 8 / 1e6, 1)
