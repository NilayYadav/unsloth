# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from core.cluster import relay as relay_module
from core.cluster.client import probe_data_path, probe_throughput

RELAY = Path(relay_module.__file__)


class EchoServer:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.port = self.sock.getsockname()[1]
        self.connections = 0
        threading.Thread(target = self._accept, daemon = True).start()

    def _accept(self):
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            self.connections += 1
            threading.Thread(target = self._echo, args = (conn,), daemon = True).start()

    @staticmethod
    def _echo(conn):
        with conn:
            while True:
                data = conn.recv(1 << 20)
                if not data:
                    return
                conn.sendall(data)

    def close(self):
        self.sock.close()


def _write_allow(path: Path, entries: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"allow": entries}))
    os.replace(tmp, path)


@pytest.fixture
def echo():
    server = EchoServer()
    yield server
    server.close()


def _spawn(allow: Path, target_port: int):
    proc = subprocess.Popen(
        [
            sys.executable,
            "-u",
            str(RELAY),
            "--listen-host",
            "127.0.0.1",
            "--listen-port",
            "0",
            "--target-port",
            str(target_port),
            "--allow-file",
            str(allow),
        ],
        stdin = subprocess.PIPE,
        stdout = subprocess.PIPE,
        text = True,
    )
    event = json.loads(proc.stdout.readline())
    assert event["event"] == "listening"
    return proc, event["port"]


@pytest.fixture
def relay(tmp_path, echo):
    allow = tmp_path / "allow.json"
    _write_allow(allow, {})
    proc, port = _spawn(allow, echo.port)
    events = []

    def read():
        for line in proc.stdout:
            events.append(json.loads(line))

    threading.Thread(target = read, daemon = True).start()
    yield {"port": port, "allow": allow, "events": events}
    proc.stdin.close()
    proc.terminate()
    proc.wait(timeout = 5)


def _round_trip(port: int, payload: bytes, source: str = None) -> bytes:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if source:
        sock.bind((source, 0))
    sock.settimeout(10)
    sock.connect(("127.0.0.1", port))
    received = bytearray()

    def reader():
        while True:
            try:
                chunk = sock.recv(1 << 20)
            except OSError:
                return
            if not chunk:
                return
            received.extend(chunk)

    thread = threading.Thread(target = reader)
    thread.start()
    try:
        sock.sendall(payload)
        sock.shutdown(socket.SHUT_WR)
    except OSError:
        pass
    thread.join(10)
    sock.close()
    return bytes(received)


def test_a_peer_without_a_lease_reaches_nothing(relay, echo):
    assert _round_trip(relay["port"], b"hello") == b""
    assert echo.connections == 0


def test_a_leased_peer_round_trips_bytes_exactly(relay, echo):
    _write_allow(relay["allow"], {"127.0.0.1": time.time() + 60})
    payload = os.urandom(32 * 1024 * 1024)
    assert _round_trip(relay["port"], payload) == payload
    assert echo.connections == 1


def test_an_expired_lease_is_refused(relay, echo):
    _write_allow(relay["allow"], {"127.0.0.1": time.time() - 1})
    assert _round_trip(relay["port"], b"hello") == b""
    assert echo.connections == 0


def test_lease_changes_apply_without_restarting_the_relay(relay, echo):
    assert _round_trip(relay["port"], b"one") == b""
    _write_allow(relay["allow"], {"127.0.0.1": time.time() + 60})
    assert _round_trip(relay["port"], b"two") == b"two"
    _write_allow(relay["allow"], {})
    assert _round_trip(relay["port"], b"three") == b""


def test_link_probes_never_touch_ggml_rpc_server(relay, echo):
    assert probe_data_path("127.0.0.1", relay["port"]) is None
    _write_allow(relay["allow"], {"127.0.0.1": time.time() + 60})
    assert probe_data_path("127.0.0.1", relay["port"]) is not None
    mbps = probe_throughput("127.0.0.1", relay["port"], seconds = 0.2)
    assert mbps is not None and mbps > 10
    assert echo.connections == 0


def test_a_second_computer_is_refused_while_one_is_connected(relay, echo):
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind(("127.0.0.2", 0))
    except OSError:
        pytest.skip("127.0.0.2 is not routable on this host")
    finally:
        probe.close()
    _write_allow(relay["allow"], {"127.0.0.1": time.time() + 60, "127.0.0.2": time.time() + 60})
    first = socket.create_connection(("127.0.0.1", relay["port"]), timeout = 5)
    try:
        first.sendall(b"x")
        assert first.recv(1) == b"x"
        assert _round_trip(relay["port"], b"y", source = "127.0.0.2") == b""
    finally:
        first.close()
    time.sleep(0.2)
    assert _round_trip(relay["port"], b"z", source = "127.0.0.2") == b"z"


def test_the_relay_exits_with_its_parent(tmp_path, echo):
    allow = tmp_path / "allow.json"
    _write_allow(allow, {})
    proc, _port = _spawn(allow, echo.port)
    proc.stdin.close()
    assert proc.wait(timeout = 5) == 0


def test_ipv4_mapped_and_scoped_addresses_normalize():
    assert relay_module.normalize_ip("::ffff:192.168.1.5") == "192.168.1.5"
    assert relay_module.normalize_ip("fe80::1%en0") == "fe80::1"
    assert relay_module.normalize_ip("10.0.0.2") == "10.0.0.2"
