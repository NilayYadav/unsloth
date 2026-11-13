# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import threading

import pytest

from core.inference import os_sandbox, sandbox_egress, tools
from core.inference.os_sandbox import ToolLaunchPlan

_SESSION = "__LOCALID_sandbox_egress"


@pytest.fixture
def proxy_enabled(monkeypatch):
    monkeypatch.setenv(sandbox_egress.FLAG_ENV, "1")
    if not sandbox_egress.enabled():
        pytest.skip("the egress proxy is Linux and macOS only")


def _proxy() -> socket.socket:
    return socket.create_connection(("127.0.0.1", sandbox_egress.tcp_port()), timeout = 10)


def _connect(target: str) -> str:
    with _proxy() as sock:
        sock.sendall(f"CONNECT {target} HTTP/1.1\r\nHost: {target}\r\n\r\n".encode())
        return sock.recv(4096).decode("latin-1").split("\r\n", 1)[0]


@pytest.mark.parametrize(
    "target,named",
    [
        ("example.com:443", "example.com is not in the sandbox network allowlist"),
        ("1.1.1.1:443", "1.1.1.1 is an IP address"),
        ("169.254.169.254:80", "169.254.169.254 is an IP address"),
        ("[::1]:443", "::1 is an IP address"),
        ("pypi.org:22", "pypi.org:22 is blocked"),
    ],
)
def test_the_proxy_refuses_and_names_what_it_refused(target, named):
    status = _connect(target)
    assert status.startswith("HTTP/1.1 403 ")
    assert named in status


def test_an_allowlisted_name_that_resolves_to_a_private_address_is_refused(monkeypatch):
    monkeypatch.setattr(sandbox_egress, "ALLOWED_HOSTS", ("localhost",))
    status = _connect("localhost:443")
    assert status.startswith("HTTP/1.1 403 ")
    assert "localhost resolves to a non-public address" in status


def test_a_plain_http_request_to_an_unlisted_host_gets_a_readable_refusal():
    with _proxy() as sock:
        sock.sendall(b"GET http://example.com/ HTTP/1.1\r\nHost: example.com\r\n\r\n")
        reply = b""
        while chunk := sock.recv(4096):
            reply += chunk
    assert reply.startswith(b"HTTP/1.1 403 ")
    assert reply.endswith(b"Unsloth sandbox: example.com is not in the sandbox network allowlist\n")


def test_socks5_refuses_an_unlisted_host():
    with _proxy() as sock:
        sock.sendall(b"\x05\x01\x00")
        assert sock.recv(2) == b"\x05\x00"
        host = b"example.com"
        sock.sendall(b"\x05\x01\x00\x03" + bytes([len(host)]) + host + (443).to_bytes(2, "big"))
        assert sock.recv(10)[:2] == b"\x05\x02"


def test_an_allowed_tunnel_relays_bytes_to_the_validated_address(monkeypatch):
    upstream = socket.create_server(("127.0.0.1", 0))
    port = upstream.getsockname()[1]

    def echo():
        conn, _ = upstream.accept()
        with conn:
            conn.sendall(conn.recv(1024).upper())

    threading.Thread(target = echo, daemon = True).start()
    monkeypatch.setattr(sandbox_egress, "ALLOWED_HOSTS", ("localhost",))
    monkeypatch.setattr(sandbox_egress, "ALLOWED_PORTS", (port,))
    monkeypatch.setattr(sandbox_egress, "_public", lambda address: True)
    with upstream, _proxy() as sock:
        sock.sendall(f"CONNECT localhost:{port} HTTP/1.1\r\n\r\n".encode())
        assert sock.recv(4096).startswith(b"HTTP/1.1 200 ")
        sock.sendall(b"hello")
        assert sock.recv(1024) == b"HELLO"


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.1.2.3",
        "172.16.0.1",
        "192.168.1.1",
        "169.254.169.254",
        "100.64.0.1",
        "100.100.100.200",
        "0.0.0.0",
        "224.0.0.251",
        "::1",
        "fe80::1",
        "fd00:ec2::254",
        "::ffff:127.0.0.1",
        "::ffff:10.0.0.1",
    ],
)
def test_non_public_addresses_are_never_dialled(address):
    assert not sandbox_egress._public(address)


@pytest.mark.parametrize("address", ["1.1.1.1", "151.101.0.223", "2606:4700:4700::1111"])
def test_public_addresses_are_dialled(address):
    assert sandbox_egress._public(address)


@pytest.mark.parametrize(
    "host,allowed",
    [
        ("pypi.org", True),
        ("PyPI.org.", True),
        ("objects.githubusercontent.com", True),
        ("cas-bridge.xethub.hf.co", True),
        ("githubusercontent.com.evil.com", False),
        ("evilpypi.org", False),
        ("pypi.org.evil.com", False),
        ("example.com", False),
    ],
)
def test_the_allowlist_matches_whole_labels(host, allowed):
    assert sandbox_egress.host_allowed(host) is allowed


def test_the_flag_is_off_by_default(monkeypatch):
    monkeypatch.delenv(sandbox_egress.FLAG_ENV, raising = False)
    assert not sandbox_egress.enabled()


def test_curl_stays_blocked_when_the_call_would_run_without_os_isolation(proxy_enabled):
    out = tools._bash_exec("curl --version", None, 60, _SESSION, tool_execution_mode = "software")
    assert out == "Blocked command(s) for safety: curl"


def test_curl_stays_blocked_when_the_flag_is_off(monkeypatch):
    monkeypatch.delenv(sandbox_egress.FLAG_ENV, raising = False)
    assert tools._bash_exec("curl --version", None, 60, _SESSION).startswith(
        "Blocked command(s) for safety: curl"
    )


def _isolated() -> bool:
    if sys.platform not in ("linux", "darwin") or not os_sandbox.capability_snapshot().available:
        return False
    if sys.platform == "darwin":
        return True
    probe = subprocess.run(
        ["bwrap", "--unshare-user", "--unshare-net", "--ro-bind", "/", "/", "true"],
        capture_output = True,
    )
    return probe.returncode == 0


_needs_isolation = pytest.mark.skipif(
    not _isolated(), reason = "needs a working OS sandbox that can own its network"
)


def _run(tmp_path, code: str) -> subprocess.CompletedProcess:
    prepared = os_sandbox.prepare_tool_launch(
        ToolLaunchPlan(
            argv = (sys.executable, "-I", "-S", "-c", code),
            workdir = str(tmp_path),
            env = {"PATH": "/usr/bin:/bin"},
            requested_mode = "required",
        )
    )
    try:
        return subprocess.run(
            prepared.argv,
            env = prepared.env,
            cwd = prepared.workdir,
            pass_fds = prepared.pass_fds,
            preexec_fn = prepared.preexec_fn,
            capture_output = True,
            text = True,
            timeout = 60,
        )
    finally:
        prepared.cleanup()


@_needs_isolation
def test_a_raw_socket_cannot_leave_the_sandbox(tmp_path, proxy_enabled):
    result = _run(
        tmp_path,
        "import socket\n"
        "for target in (('1.1.1.1', 443), ('169.254.169.254', 80)):\n"
        "    try:\n"
        "        socket.create_connection(target, timeout = 5)\n"
        "        print('ESCAPED', target)\n"
        "    except OSError as exc:\n"
        "        print('refused', type(exc).__name__)\n",
    )
    assert "ESCAPED" not in result.stdout
    assert result.stdout.count("refused") == 2, result.stderr


@_needs_isolation
def test_the_proxy_is_the_way_out_and_names_a_refused_host(tmp_path, proxy_enabled):
    result = _run(
        tmp_path,
        "import os, socket\n"
        "host, port = os.environ['HTTPS_PROXY'].rsplit('/', 1)[1].split(':')\n"
        "with socket.create_connection((host, int(port)), timeout = 10) as sock:\n"
        "    sock.sendall(b'CONNECT example.com:443 HTTP/1.1\\r\\n\\r\\n')\n"
        "    print(sock.recv(4096).decode().splitlines()[0])\n",
    )
    assert "403 Unsloth sandbox: example.com is not in the sandbox network allowlist" in (
        result.stdout
    ), result.stderr


@_needs_isolation
def test_curl_runs_in_the_terminal_behind_the_proxy(proxy_enabled):
    if shutil.which("curl") is None:
        pytest.skip("curl is not installed")
    out = tools._bash_exec("curl --version", None, 60, _SESSION)
    assert out.startswith("curl "), out
    out = tools._bash_exec("curl -sS https://example.com/ || true", None, 60, _SESSION)
    assert "403" in out


@pytest.mark.skipif(sys.platform != "linux", reason = "the bubblewrap backend is Linux only")
def test_bwrap_keeps_the_host_network_when_the_flag_is_off(tmp_path, monkeypatch):
    if shutil.which("bwrap") is None:
        pytest.skip("bubblewrap is not installed")
    from core.inference import sandbox_linux

    monkeypatch.delenv(sandbox_egress.FLAG_ENV, raising = False)
    prepared = sandbox_linux.prepare(
        ToolLaunchPlan(argv = ("/bin/true",), workdir = str(tmp_path), env = {})
    )
    prepared.cleanup()
    assert "--unshare-net" not in prepared.argv
    assert prepared.argv[-1] == "/bin/true"
    assert not any("PROXY" in arg for arg in prepared.argv)


@pytest.mark.skipif(sys.platform != "darwin", reason = "the Seatbelt backend is macOS only")
def test_seatbelt_allows_outbound_only_to_the_proxy_port(tmp_path):
    from core.inference import sandbox_macos

    kwargs = dict(workdir = str(tmp_path), private_tmp = str(tmp_path), runtime_paths = ())
    assert '(allow network-outbound (remote ip "*:*"))' in sandbox_macos.build_profile(**kwargs)
    proxied = sandbox_macos.build_profile(**kwargs, proxy_port = 4321)
    assert '(allow network-outbound (remote ip "localhost:4321"))' in proxied
    assert '(remote ip "*:*")' not in proxied
