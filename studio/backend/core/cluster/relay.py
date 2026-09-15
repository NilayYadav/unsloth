# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

# Allowlisting TCP relay in front of ggml-rpc-server, which has no authentication of its own.
# Runs as its own process so Studio's GIL never sits on the token path. Stdlib only.

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import socket
import struct
import sys
import threading
import time

PROBE_MAGIC = b"UNSLOTH-PROBE1"
_COPY_BUFFER = 4 * 1024 * 1024
_PIPE_SIZE = 1 << 20
_MAX_PROBE_BYTES = 1 << 30
_F_SETPIPE_SZ = 1031

_emit_lock = threading.Lock()
_session_lock = threading.Lock()
_active_sessions: dict[int, str] = {}


def _emit(event: str, **fields) -> None:
    line = json.dumps({"event": event, "ts": time.time(), **fields}, separators = (",", ":"))
    with _emit_lock:
        try:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()
        except (OSError, ValueError):
            pass


def normalize_ip(address: str) -> str:
    address = str(address).split("%", 1)[0]
    try:
        parsed = ipaddress.ip_address(address)
    except ValueError:
        return address
    if isinstance(parsed, ipaddress.IPv6Address) and parsed.ipv4_mapped is not None:
        parsed = parsed.ipv4_mapped
    return str(parsed)


class Allowlist:
    def __init__(self, path: str):
        self.path = path
        self._stamp = None
        self._entries: dict[str, float] = {}

    def allows(self, address: str) -> bool:
        try:
            st = os.stat(self.path)
        except OSError:
            return False
        stamp = (st.st_mtime_ns, st.st_size, st.st_ino)
        if stamp != self._stamp:
            try:
                with open(self.path, encoding = "utf-8") as handle:
                    data = json.load(handle)
                entries = {normalize_ip(k): float(v) for k, v in (data.get("allow") or {}).items()}
            except (OSError, ValueError, TypeError, AttributeError):
                return False
            self._entries, self._stamp = entries, stamp
        expiry = self._entries.get(normalize_ip(address))
        return expiry is not None and expiry > time.time()


def _raise_thread_qos() -> None:
    # macOS parks default-QoS threads on efficiency cores, which costs wake-up latency per message
    if sys.platform != "darwin":
        return
    try:
        import ctypes

        ctypes.CDLL("/usr/lib/libSystem.B.dylib").pthread_set_qos_class_self_np(0x21, 0)
    except Exception:
        pass


def tune_socket(sock: socket.socket) -> None:
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    # a head that vanished must not pin ggml-rpc-server, which serves one client at a time
    options = [("TCP_KEEPINTVL", 10), ("TCP_KEEPCNT", 6)]
    options.append(("TCP_KEEPIDLE", 30) if hasattr(socket, "TCP_KEEPIDLE") else ("TCP_KEEPALIVE", 30))
    for name, value in options:
        opt = getattr(socket, name, None)
        if opt is None:
            continue
        try:
            sock.setsockopt(socket.IPPROTO_TCP, opt, value)
        except OSError:
            pass


def _pump_copy(src: socket.socket, dst: socket.socket, counters: list, index: int) -> None:
    buf = bytearray(_COPY_BUFFER)
    view = memoryview(buf)
    recv_into, sendall = src.recv_into, dst.sendall
    while True:
        n = recv_into(buf)
        if not n:
            return
        sendall(view[:n])
        counters[index] += n


def _pump_splice(src: socket.socket, dst: socket.socket, counters: list, index: int) -> None:
    import fcntl

    r, w = os.pipe()
    try:
        try:
            fcntl.fcntl(w, _F_SETPIPE_SZ, _PIPE_SIZE)
        except OSError:
            pass
        sfd, dfd = src.fileno(), dst.fileno()
        flags = getattr(os, "SPLICE_F_MOVE", 0)
        while True:
            n = os.splice(sfd, w, _PIPE_SIZE, flags = flags)
            if n == 0:
                return
            left = n
            while left:
                left -= os.splice(r, dfd, left, flags = flags)
            counters[index] += n
    finally:
        os.close(r)
        os.close(w)


def _pump(src: socket.socket, dst: socket.socket, counters: list, index: int) -> None:
    _raise_thread_qos()
    try:
        if sys.platform == "linux" and hasattr(os, "splice"):
            try:
                _pump_splice(src, dst, counters, index)
            except OSError:
                if counters[index]:
                    raise
                _pump_copy(src, dst, counters, index)
        else:
            _pump_copy(src, dst, counters, index)
        try:
            dst.shutdown(socket.SHUT_WR)
        except OSError:
            pass
    except OSError:
        for sock in (src, dst):
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ConnectionError("peer closed")
        data += chunk
    return bytes(data)


def _is_probe(client: socket.socket) -> bool:
    client.settimeout(5.0)
    try:
        seen = b""
        while len(seen) < len(PROBE_MAGIC):
            seen = client.recv(len(PROBE_MAGIC), socket.MSG_PEEK)
            if not seen or not PROBE_MAGIC.startswith(seen):
                return False
            if len(seen) < len(PROBE_MAGIC):
                time.sleep(0.001)
        return True
    except (OSError, ValueError):
        return False
    finally:
        client.settimeout(None)


def _serve_probe(client: socket.socket, peer: str) -> None:
    client.settimeout(30.0)
    _recv_exact(client, len(PROBE_MAGIC))
    (size,) = struct.unpack("!Q", _recv_exact(client, 8))
    size = min(size, _MAX_PROBE_BYTES)
    buf = bytearray(_COPY_BUFFER)
    received = 0
    started = time.perf_counter_ns()
    while received < size:
        n = client.recv_into(buf, min(len(buf), size - received))
        if not n:
            break
        received += n
    elapsed = time.perf_counter_ns() - started
    client.sendall(struct.pack("!QQ", elapsed, received))
    _emit("probe", peer = peer, bytes = received, elapsed_ns = elapsed)


def _serve(client: socket.socket, address, target: tuple[str, int], allow: Allowlist) -> None:
    peer = normalize_ip(address[0])
    upstream = None
    session = None
    try:
        if not allow.allows(peer):
            _emit("rejected", peer = peer, reason = "not_leased")
            return
        tune_socket(client)
        if _is_probe(client):
            _serve_probe(client, peer)
            return
        with _session_lock:
            if any(owner != peer for owner in _active_sessions.values()):
                _emit("rejected", peer = peer, reason = "busy")
                return
            session = id(client)
            _active_sessions[session] = peer
        try:
            upstream = socket.create_connection(target, timeout = 5.0)
        except OSError as exc:
            _emit("upstream_failed", peer = peer, error = str(exc))
            return
        upstream.settimeout(None)
        tune_socket(upstream)
        counters = [0, 0]
        started = time.monotonic()
        _emit("opened", peer = peer)
        inbound = threading.Thread(target = _pump, args = (client, upstream, counters, 0), daemon = True)
        inbound.start()
        _pump(upstream, client, counters, 1)
        inbound.join()
        _emit(
            "closed",
            peer = peer,
            bytes_in = counters[0],
            bytes_out = counters[1],
            seconds = round(time.monotonic() - started, 3),
        )
    except (OSError, ConnectionError, struct.error) as exc:
        _emit("error", peer = peer, error = str(exc))
    finally:
        if session is not None:
            with _session_lock:
                _active_sessions.pop(session, None)
        for sock in (upstream, client):
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass


def _exit_with_parent() -> None:
    try:
        while sys.stdin.buffer.read(1024):
            pass
    except (OSError, ValueError):
        pass
    os._exit(0)


def main(argv = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-host", default = "0.0.0.0")
    parser.add_argument("--listen-port", type = int, required = True)
    parser.add_argument("--target-host", default = "127.0.0.1")
    parser.add_argument("--target-port", type = int, required = True)
    parser.add_argument("--allow-file", required = True)
    parser.add_argument("--no-parent-watch", action = "store_true")
    args = parser.parse_args(argv)

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if sys.platform != "win32":
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        listener.bind((args.listen_host, args.listen_port))
    except OSError as exc:
        _emit("bind_failed", error = str(exc), port = args.listen_port)
        return 2
    listener.listen(64)
    _raise_thread_qos()
    if not args.no_parent_watch:
        threading.Thread(target = _exit_with_parent, daemon = True).start()
    allow = Allowlist(args.allow_file)
    target = (args.target_host, args.target_port)
    _emit("listening", host = args.listen_host, port = listener.getsockname()[1])
    while True:
        try:
            client, address = listener.accept()
        except OSError as exc:
            _emit("accept_failed", error = str(exc))
            time.sleep(0.05)
            continue
        threading.Thread(target = _serve, args = (client, address, target, allow), daemon = True).start()


if __name__ == "__main__":
    sys.exit(main())
