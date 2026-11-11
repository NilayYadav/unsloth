# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import os
import signal
import subprocess
import sys
import time

import pytest

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

from core.inference import llama_cpp  # noqa: E402
from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402
from utils.process_lifetime import mark_process_shutting_down, mark_quit_signal_received  # noqa: E402

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason = "POSIX process groups")

_LLAMA_SERVER_LIKE = r"""
import os, signal, socket, sys, time
listener = socket.create_server(("127.0.0.1", 0))
seen = []
def handler(signum, frame):
    if seen:
        os._exit(1)
    seen.append(signum)
    listener.close()
signal.signal(signal.SIGINT, handler)
signal.signal(signal.SIGTERM, handler)
print(listener.getsockname()[1], flush = True)
while not seen:
    time.sleep(0.01)
time.sleep(float(sys.argv[1]))
sys.exit(0)
"""


def _backend_holding(proc) -> LlamaCppBackend:
    backend = LlamaCppBackend.__new__(LlamaCppBackend)
    backend._process = proc
    backend._port = proc.port
    backend._healthy = True
    backend._owns_pidfile = False
    return backend


def _spawn(
    shutdown_s = 0.5,
    through = lambda spawn: spawn(),
    **popen_kwargs,
):
    proc = through(
        lambda: subprocess.Popen(
            [sys.executable, "-c", _LLAMA_SERVER_LIKE, str(shutdown_s)],
            stdout = subprocess.PIPE,
            text = True,
            **popen_kwargs,
        )
    )
    proc.port = int(proc.stdout.readline())
    assert os.getpgid(proc.pid) == os.getpgrp()
    return proc


def test_ctrl_c_shutdown_lets_llama_server_exit_on_its_own():
    proc = _spawn()
    try:
        os.kill(proc.pid, signal.SIGINT)
        mark_quit_signal_received()
        _backend_holding(proc)._kill_process(teardown = True)
        assert proc.returncode == 0, "llama-server got a second signal mid-shutdown"
    finally:
        proc.kill()
        proc.wait()
        proc.stdout.close()


def test_llama_server_stuck_after_ctrl_c_is_killed_not_signalled_twice():
    proc = _spawn(shutdown_s = 30)
    try:
        os.kill(proc.pid, signal.SIGINT)
        mark_quit_signal_received()
        _backend_holding(proc)._kill_process(teardown = True)
        assert proc.returncode == -signal.SIGKILL
    finally:
        proc.kill()
        proc.wait()
        proc.stdout.close()


@pytest.mark.skipif(sys.platform != "linux", reason = "llama-server sits out Ctrl-C on Linux only")
def test_llama_server_that_sat_out_ctrl_c_is_stopped_at_once(monkeypatch):
    from utils import process_lifetime
    proc = _spawn(preexec_fn = llama_cpp._block_sigint)
    try:
        os.kill(proc.pid, signal.SIGINT)
        monkeypatch.setattr(process_lifetime, "_quit_signal_at", time.monotonic())
        started = time.monotonic()
        _backend_holding(proc)._kill_process(teardown = True)
        assert proc.returncode == 0
        assert time.monotonic() - started < 1.5
    finally:
        proc.kill()
        proc.wait()
        proc.stdout.close()


@pytest.mark.skipif(sys.platform != "linux", reason = "llama-server sits out Ctrl-C on Linux only")
def test_llama_server_stuck_after_sitting_out_ctrl_c_is_killed_within_the_grace(monkeypatch):
    from utils import process_lifetime
    proc = _spawn(shutdown_s = 30, preexec_fn = llama_cpp._block_sigint)
    try:
        os.kill(proc.pid, signal.SIGINT)
        monkeypatch.setattr(process_lifetime, "_quit_signal_at", time.monotonic())
        started = time.monotonic()
        _backend_holding(proc)._kill_process(teardown = True)
        assert proc.returncode == -signal.SIGKILL
        assert time.monotonic() - started < 4
    finally:
        proc.kill()
        proc.wait()
        proc.stdout.close()


def test_chat_llama_server_sits_out_ctrl_c_until_studio_stops_it(monkeypatch):
    from utils import process_lifetime
    proc = _spawn(
        through = llama_cpp._spawn_chat_server,
        **llama_cpp._child_popen_kwargs(llama_cpp._block_sigint),
    )
    try:
        os.kill(proc.pid, signal.SIGINT)
        time.sleep(1.5)
        assert proc.poll() is None
        monkeypatch.setattr(process_lifetime, "_quit_signal_at", time.monotonic())
        _backend_holding(proc)._kill_process(teardown = True)
        assert proc.returncode == 0
    finally:
        proc.kill()
        proc.wait()
        proc.stdout.close()


_STUDIO_LIKE = r"""
import subprocess, sys, time
sys.path.insert(0, sys.argv[1])
from core.inference import llama_cpp
server = llama_cpp._spawn_chat_server(
    lambda: subprocess.Popen(
        [sys.executable, "-c", sys.argv[2], "0.1"],
        stdout = subprocess.PIPE,
        text = True,
        **llama_cpp._child_popen_kwargs(llama_cpp._block_sigint),
    )
)
server.stdout.readline()
print(server.pid, flush = True)
time.sleep(120)
"""


def test_chat_llama_server_does_not_outlive_a_killed_studio():
    from utils.process_lifetime import pid_is_running

    studio = subprocess.Popen(
        [sys.executable, "-c", _STUDIO_LIKE, _backend, _LLAMA_SERVER_LIKE],
        stdout = subprocess.PIPE,
        text = True,
    )
    server = None
    try:
        server = int(studio.stdout.readline())
        os.kill(server, signal.SIGINT)
        time.sleep(1.5)
        assert pid_is_running(server)
        studio.kill()
        studio.wait()
        deadline = time.monotonic() + 10
        while pid_is_running(server) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not pid_is_running(server)
    finally:
        studio.kill()
        studio.wait()
        studio.stdout.close()
        if server and pid_is_running(server):
            os.kill(server, signal.SIGKILL)


@pytest.mark.skipif(sys.platform != "darwin", reason = "the watcher is macOS only")
def test_macos_watcher_leaves_with_llama_server():
    from utils.process_lifetime import pid_is_running
    proc = _spawn(
        through = llama_cpp._spawn_chat_server,
        **llama_cpp._child_popen_kwargs(llama_cpp._block_sigint),
    )
    try:
        found = subprocess.run(
            ["pgrep", "-P", str(os.getpid()), "-f", f" {os.getpid()} {proc.pid}$"],
            capture_output = True,
            text = True,
        )
        (watcher,) = map(int, found.stdout.split())
        proc.terminate()
        assert proc.wait(timeout = 5) == 0
        deadline = time.monotonic() + 5
        while pid_is_running(watcher) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not pid_is_running(watcher)
    finally:
        proc.kill()
        proc.wait()
        proc.stdout.close()


def test_a_stand_in_child_gets_no_macos_watcher(monkeypatch):
    spawned = []
    monkeypatch.setattr(llama_cpp.sys, "platform", "darwin")
    monkeypatch.setattr(
        llama_cpp.subprocess, "Popen", lambda *args, **kwargs: spawned.append(args) or object()
    )
    llama_cpp._spawn_chat_server(lambda: llama_cpp.subprocess.Popen(["llama-server"]))
    assert spawned == [(["llama-server"],)]
    assert signal.SIGINT not in signal.pthread_sigmask(signal.SIG_BLOCK, ())


def test_unloading_a_model_still_stops_llama_server_at_once():
    proc = _spawn()
    try:
        started = time.monotonic()
        _backend_holding(proc)._kill_process()
        assert proc.returncode == 0
        assert time.monotonic() - started < 1.5
    finally:
        proc.kill()
        proc.wait()
        proc.stdout.close()


def test_quit_without_a_signal_still_stops_llama_server_at_once():
    proc = _spawn()
    try:
        mark_process_shutting_down()
        started = time.monotonic()
        _backend_holding(proc)._kill_process(teardown = True)
        assert proc.returncode == 0
        assert time.monotonic() - started < 1.5
    finally:
        proc.kill()
        proc.wait()
        proc.stdout.close()


def test_servers_share_one_grace_period():
    procs = [_spawn(), _spawn()]
    try:
        mark_quit_signal_received()
        started = time.monotonic()
        for proc in procs:
            _backend_holding(proc)._kill_process(teardown = True)
            assert proc.returncode == 0
        assert time.monotonic() - started < 4
    finally:
        for proc in procs:
            proc.kill()
            proc.wait()
            proc.stdout.close()
