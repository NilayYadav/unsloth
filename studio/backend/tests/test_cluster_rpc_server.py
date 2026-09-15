# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import os
import sys
import time
from pathlib import Path

import pytest

from core.cluster import rpc_server
from core.cluster.rpc_server import RpcBanner, TensorCache, parse_banner_line

CUDA_BANNER = """Starting RPC server v6.0.0
  endpoint       : 127.0.0.1:50052
  local cache    : /tmp/cache/rpc/
Devices:
  CUDA0: NVIDIA GeForce RTX 4090 (24080 MiB, 23657 MiB free)
  CUDA1: NVIDIA GeForce RTX 3090 (24126 MiB, 23900 MiB free)
  transport      : TCP (RDMA auto-negotiate enabled)
"""


def _banner(text: str) -> RpcBanner:
    banner = RpcBanner()
    for line in text.splitlines():
        parse_banner_line(banner, line)
    return banner


def test_banner_yields_version_devices_and_transport():
    banner = _banner(CUDA_BANNER)
    assert banner.version == (6, 0, 0)
    assert [d["name"] for d in banner.devices] == ["CUDA0", "CUDA1"]
    assert banner.devices[0] == {
        "name": "CUDA0",
        "description": "NVIDIA GeForce RTX 4090",
        "total_mib": 24080,
        "free_mib": 23657,
    }
    assert banner.transport == "TCP (RDMA auto-negotiate enabled)"


def test_banner_reads_metal_devices_and_ignores_repeats():
    banner = _banner("  MTL0: Apple M2 Ultra (147456 MiB, 147000 MiB free)\n" * 2)
    assert banner.devices == [
        {"name": "MTL0", "description": "Apple M2 Ultra", "total_mib": 147456, "free_mib": 147000}
    ]


def _cache(tmp_path) -> TensorCache:
    cache = TensorCache(tmp_path)
    cache.directory.mkdir(parents = True)
    return cache


def _put(cache: TensorCache, name: str, size: int, age: float = 100.0) -> Path:
    path = cache.directory / name
    path.write_bytes(os.urandom(size))
    stamp = time.time() - age
    os.utime(path, (stamp, stamp))
    return path


def test_files_not_sealed_after_a_finished_session_are_dropped(tmp_path):
    cache = _cache(tmp_path)
    kept = _put(cache, "0123456789abcdef", 1024)
    cache.seal()
    partial = _put(cache, "fedcba9876543210", 512)
    assert cache.sanitize() == 1
    assert kept.exists() and not partial.exists()


def test_a_sealed_file_that_changed_is_dropped(tmp_path):
    cache = _cache(tmp_path)
    path = _put(cache, "0123456789abcdef", 1024)
    cache.seal()
    with path.open("ab") as handle:
        handle.write(b"x")
    assert cache.sanitize() == 1
    assert not path.exists()


def test_a_quiet_seal_leaves_out_files_still_being_written(tmp_path):
    cache = _cache(tmp_path)
    old = _put(cache, "0123456789abcdef", 1024, age = 100)
    fresh = _put(cache, "fedcba9876543210", 1024, age = 0)
    cache.seal(quiet_seconds = 2.0)
    cache.sanitize()
    assert old.exists() and not fresh.exists()


def test_prune_evicts_the_least_recently_used_first(tmp_path):
    cache = _cache(tmp_path)
    oldest = _put(cache, "000000000000000a", 1000, age = 300)
    middle = _put(cache, "000000000000000b", 1000, age = 200)
    newest = _put(cache, "000000000000000c", 1000, age = 100)
    assert cache.prune(2000, min_free_bytes = 0) == 1000
    assert not oldest.exists() and middle.exists() and newest.exists()


def test_cache_bookkeeping_ignores_foreign_files(tmp_path):
    cache = _cache(tmp_path)
    _put(cache, "000000000000000a", 1000)
    stray = cache.directory / "notes.txt"
    stray.write_text("keep me")
    assert cache.usage() == {"bytes": 1000, "files": 1}
    cache.clear()
    assert stray.exists() and cache.usage()["files"] == 0


@pytest.mark.skipif(sys.platform == "win32", reason = "posix executable bits")
def test_the_rpc_server_is_found_beside_llama_server(tmp_path, monkeypatch):
    monkeypatch.delenv("UNSLOTH_RPC_SERVER_PATH", raising = False)
    root = tmp_path / "llama.cpp"
    bin_dir = root / "build" / "bin"
    bin_dir.mkdir(parents = True)
    for path in (bin_dir / "llama-server", bin_dir / "ggml-rpc-server", root / "llama-server"):
        path.write_text("")
        path.chmod(0o755)
    assert rpc_server.find_rpc_server_binary(str(root / "llama-server")) == str(bin_dir / "ggml-rpc-server")
    assert rpc_server.find_rpc_server_binary(str(bin_dir / "llama-server")) == str(bin_dir / "ggml-rpc-server")
    assert rpc_server.find_rpc_server_binary(str(tmp_path / "missing" / "llama-server")) is None


FAKE_RPC_SERVER = """#!{python}
import socket, sys
port = int(sys.argv[sys.argv.index("-p") + 1])
print("Starting RPC server v6.0.0")
print("Devices:")
print("  CUDA0: NVIDIA A10 (22502 MiB, 22000 MiB free)")
print("  transport      : TCP")
listener = socket.socket()
listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listener.bind(("127.0.0.1", port))
listener.listen(4)
while True:
    conn, _ = listener.accept()
    print("Accepted client connection", flush=True)
    conn.recv(1)
    conn.close()
    print("Client connection closed", flush=True)
"""


@pytest.mark.skipif(sys.platform == "win32", reason = "posix shebang script")
def test_startup_does_not_wait_on_a_banner_stuck_in_the_pipe_buffer(tmp_path):
    fake = tmp_path / "ggml-rpc-server"
    fake.write_text(FAKE_RPC_SERVER.format(python = sys.executable))
    fake.chmod(0o755)
    process = rpc_server.RpcServerProcess(str(fake), "127.0.0.1", rpc_server.free_local_port(), tmp_path / "cache")
    started = time.monotonic()
    try:
        banner = process.start(timeout = 20.0)
        assert time.monotonic() - started < 10.0
        assert banner.version == (6, 0, 0)
        assert [d["name"] for d in banner.devices] == ["CUDA0"]
        assert process.alive
    finally:
        process.stop()
    assert not process.alive


@pytest.mark.skipif(sys.platform == "win32", reason = "posix shebang script")
def test_the_rpc_server_outlives_the_short_lived_thread_that_started_it(tmp_path):
    import threading

    fake = tmp_path / "ggml-rpc-server"
    fake.write_text(FAKE_RPC_SERVER.format(python = sys.executable))
    fake.chmod(0o755)
    process = rpc_server.RpcServerProcess(str(fake), "127.0.0.1", rpc_server.free_local_port(), tmp_path / "cache")
    errors = []

    def start():
        try:
            process.start(timeout = 20.0)
        except Exception as exc:
            errors.append(exc)

    starter = threading.Thread(target = start)
    starter.start()
    starter.join(30)
    try:
        assert not errors
        time.sleep(1.5)
        assert process.alive, "the Share button starts sharing on a worker thread; its exit must not kill the server"
    finally:
        process.stop()


@pytest.mark.skipif(sys.platform == "win32", reason = "posix executable bits")
def test_an_rpc_server_extracted_without_its_exec_bit_is_repaired(tmp_path, monkeypatch):
    monkeypatch.delenv("UNSLOTH_RPC_SERVER_PATH", raising = False)
    server = tmp_path / "llama-server"
    server.write_text("")
    server.chmod(0o755)
    rpc = tmp_path / "ggml-rpc-server"
    rpc.write_text("")
    rpc.chmod(0o644)
    assert rpc_server.find_rpc_server_binary(str(server)) == str(rpc)
    assert os.access(rpc, os.X_OK)
