# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""HTTP CONNECT and SOCKS5 proxy that is a sandboxed tool call's only way out, filtered by host name."""

from __future__ import annotations

import ipaddress
import os
import socket
import socketserver
import sys
import tempfile
import threading
from urllib.parse import urlsplit

from loggers import get_logger

logger = get_logger(__name__)

FLAG_ENV = "UNSLOTH_STUDIO_SANDBOX_NETWORK_PROXY"
ALLOWED_HOSTS = (
    "pypi.org",
    "files.pythonhosted.org",
    "registry.npmjs.org",
    "proxy.golang.org",
    "sum.golang.org",
    "index.crates.io",
    "static.crates.io",
    "github.com",
    "codeload.github.com",
    "*.githubusercontent.com",
    "huggingface.co",
    "*.huggingface.co",
    "hf.co",
    "*.hf.co",
)
ALLOWED_PORTS = (80, 443)
BRIDGE_PORT = 3128
_HEADER_LIMIT = 64 * 1024
_IDLE_SECONDS = 300

# Run inside the jail with `python -I -S`: forwards loopback BRIDGE_PORT to the proxy's unix socket.
BRIDGE_CODE = """
import os, socket, sys, threading
path, port, argv = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
server = socket.create_server(("127.0.0.1", port))
if os.fork() == 0:
    null = os.open(os.devnull, os.O_RDWR)
    for fd in (0, 1, 2):
        os.dup2(null, fd)
    def pump(src, dst):
        try:
            while data := src.recv(65536):
                dst.sendall(data)
        except OSError:
            pass
        finally:
            try:
                dst.shutdown(socket.SHUT_WR)
            except OSError:
                pass
    def serve(client):
        with client, socket.socket(socket.AF_UNIX) as upstream:
            try:
                upstream.connect(path)
            except OSError:
                return
            other = threading.Thread(target=pump, args=(upstream, client), daemon=True)
            other.start()
            pump(client, upstream)
            other.join()
    while True:
        client, _ = server.accept()
        threading.Thread(target=serve, args=(client,), daemon=True).start()
server.close()
os.execvp(argv[0], argv)
"""


def enabled() -> bool:
    return sys.platform in ("linux", "darwin") and os.environ.get(FLAG_ENV, "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def proxy_env(port: int) -> dict[str, str]:
    http, socks = f"http://127.0.0.1:{port}", f"socks5h://127.0.0.1:{port}"
    env = {}
    for name, value in (
        ("http_proxy", http),
        ("https_proxy", http),
        ("all_proxy", socks),
        ("no_proxy", "localhost,127.0.0.1,::1"),
    ):
        env[name] = env[name.upper()] = value
    return env


def host_allowed(host: str) -> bool:
    host = host.lower().rstrip(".")
    return any(
        host.endswith(pattern[1:]) if pattern.startswith("*.") else host == pattern
        for pattern in ALLOWED_HOSTS
    )


def _public(address: str) -> bool:
    ip = ipaddress.ip_address(address.split("%", 1)[0])
    if ip.version == 6 and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


class _Refused(Exception):
    pass


def _open_upstream(host: str, port: int) -> socket.socket:
    host = host.lower().rstrip(".")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        raise _Refused(f"{host} is an IP address; only allowlisted host names are reachable")
    if port not in ALLOWED_PORTS:
        raise _Refused(f"{host}:{port} is blocked: only ports 80 and 443 are reachable")
    if not host_allowed(host):
        raise _Refused(f"{host} is not in the sandbox network allowlist")
    try:
        infos = socket.getaddrinfo(host, port, type = socket.SOCK_STREAM)
    except OSError:
        raise _Refused(f"{host} does not resolve") from None
    private = next((info[4][0] for info in infos if not _public(info[4][0])), None)
    if private is not None:
        raise _Refused(f"{host} resolves to a non-public address ({private})")
    error: OSError | None = None
    for info in infos:
        try:
            # The validated address, never the name again: a second lookup could rebind it.
            return socket.create_connection(info[4][:2], timeout = 10)
        except OSError as exc:
            error = exc
    raise _Refused(f"{host} is unreachable: {error}")


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    data = b""
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ConnectionError("closed")
        data += chunk
    return data


def _pump(src: socket.socket, dst: socket.socket) -> None:
    try:
        while data := src.recv(65536):
            dst.sendall(data)
    except OSError:
        pass
    finally:
        try:
            dst.shutdown(socket.SHUT_WR)
        except OSError:
            pass


def _relay(client: socket.socket, upstream: socket.socket) -> None:
    client.settimeout(_IDLE_SECONDS)
    upstream.settimeout(_IDLE_SECONDS)
    other = threading.Thread(target = _pump, args = (upstream, client), daemon = True)
    other.start()
    _pump(client, upstream)
    other.join()


def _refuse_http(client: socket.socket, message: str) -> None:
    logger.info("Sandbox network proxy refused: %s", message)
    body = f"Unsloth sandbox: {message}\n".encode()
    client.sendall(
        f"HTTP/1.1 403 Unsloth sandbox: {message}\r\n".encode()
        + b"Content-Type: text/plain\r\nConnection: close\r\n"
        + f"Content-Length: {len(body)}\r\n\r\n".encode()
        + body
    )


def _serve_socks(client: socket.socket) -> None:
    _, count = _recv_exact(client, 2)
    _recv_exact(client, count)
    client.sendall(b"\x05\x00")
    _, command, _, kind = _recv_exact(client, 4)
    if kind == 3:
        host = _recv_exact(client, _recv_exact(client, 1)[0]).decode("idna")
    elif kind in (1, 4):
        host = str(ipaddress.ip_address(_recv_exact(client, 4 if kind == 1 else 16)))
    else:
        client.sendall(b"\x05\x08\x00\x01" + bytes(6))
        return
    port = int.from_bytes(_recv_exact(client, 2), "big")
    if command != 1:
        client.sendall(b"\x05\x07\x00\x01" + bytes(6))
        return
    try:
        upstream = _open_upstream(host, port)
    except _Refused as exc:
        logger.info("Sandbox network proxy refused: %s", exc)
        client.sendall(b"\x05\x02\x00\x01" + bytes(6))
        return
    with upstream:
        client.sendall(b"\x05\x00\x00\x01" + bytes(6))
        _relay(client, upstream)


def _serve_http(client: socket.socket) -> None:
    head = b""
    while b"\r\n\r\n" not in head:
        chunk = client.recv(65536)
        if not chunk or len(head) > _HEADER_LIMIT:
            return
        head += chunk
    head, rest = head.split(b"\r\n\r\n", 1)
    request_line, *headers = head.decode("latin-1").split("\r\n")
    try:
        method, target, version = request_line.split(" ")
    except ValueError:
        return
    if method == "CONNECT":
        url = urlsplit(f"//{target}")
    else:
        url = urlsplit(target)
        if url.scheme != "http":
            _refuse_http(client, f"{target} is not an absolute http:// URL")
            return
    try:
        host, port = url.hostname or "", url.port or 80
    except ValueError:
        _refuse_http(client, f"{target} has an invalid port")
        return
    try:
        upstream = _open_upstream(host, port)
    except _Refused as exc:
        _refuse_http(client, str(exc))
        return
    with upstream:
        if method == "CONNECT":
            client.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
            if rest:
                upstream.sendall(rest)
        else:
            path = (url.path or "/") + (f"?{url.query}" if url.query else "")
            kept = [h for h in headers if not h.lower().startswith("proxy-")]
            upstream.sendall(
                "\r\n".join([f"{method} {path} {version}", *kept, "", ""]).encode("latin-1") + rest
            )
        _relay(client, upstream)


class _Handler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        client = self.request
        client.settimeout(30)
        try:
            first = client.recv(1, socket.MSG_PEEK)
            if first == b"\x05":
                _serve_socks(client)
            elif first:
                _serve_http(client)
        except (OSError, ValueError, UnicodeError):
            pass


class _UnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


class _TCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True


_lock = threading.Lock()
_servers: dict[str, socketserver.BaseServer] = {}


def _start(kind: str) -> socketserver.BaseServer:
    with _lock:
        server = _servers.get(kind)
        if server is None:
            if kind == "unix":
                directory = tempfile.mkdtemp(prefix = "unsloth-egress-")
                server = _UnixServer(os.path.join(directory, "proxy.sock"), _Handler)
            else:
                server = _TCPServer(("127.0.0.1", 0), _Handler)
            threading.Thread(
                target = server.serve_forever, name = f"unsloth-egress-{kind}", daemon = True
            ).start()
            _servers[kind] = server
        return server


def unix_socket_path() -> str:
    return _start("unix").server_address


def tcp_port() -> int:
    return _start("tcp").server_address[1]
