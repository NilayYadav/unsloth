# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A worker that pytest-timeout ends (thread method: ``os._exit``) under ``--dist loadgroup`` / ``loadfile`` fails
its test and the run finishes; with pytest-xdist's own scheduler the run waits on the replacement worker forever."""

import ast
import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("xdist")
pytest.importorskip("pytest_timeout")

_SHARED = Path(__file__).resolve().parent / "_shared"

_TESTS = """
import time
import pytest

@pytest.mark.parametrize("i", range(4))
def test_fast(i):
    time.sleep(0.3)

def test_hangs():
    time.sleep(300)

@pytest.mark.xdist_group("pair")
def test_pair_one():
    pass

@pytest.mark.xdist_group("pair")
def test_pair_two():
    pass

def test_last():
    pass
"""

_CONFTEST = f"""
import sys
import pytest

sys.path.insert(0, {str(_SHARED)!r})


@pytest.hookimpl(optionalhook = True)
def pytest_xdist_make_scheduler(config, log):
    from xdist_crash_requeue import make_scheduler

    return make_scheduler(config, log)
"""


@pytest.mark.parametrize("dist", ("loadgroup", "loadfile"))
def test_timed_out_worker_fails_its_test_and_the_run_finishes(tmp_path, dist):
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding = "utf-8")
    (tmp_path / "conftest.py").write_text(textwrap.dedent(_CONFTEST), encoding = "utf-8")
    (tmp_path / "test_crash.py").write_text(textwrap.dedent(_TESTS), encoding = "utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST_")}
    cmd = [sys.executable, "-m", "pytest", "-c", str(tmp_path / "pytest.ini"), "--rootdir", str(tmp_path),
           "-p", "no:cacheprovider", "-q", "-rf", "-n", "2", "--dist", dist,
           "--timeout=3", "--timeout-method=thread", "test_crash.py"]  # fmt: skip
    proc = subprocess.Popen(
        cmd,
        cwd = tmp_path,
        env = env,
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT,
        text = True,
        start_new_session = True,
    )
    try:
        out, _ = proc.communicate(timeout = 120)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        out, _ = proc.communicate()
        pytest.fail(f"the run hung after the worker crash:\n{out[-3000:]}")
    assert proc.returncode == 1, out[-3000:]
    assert out.count("crashed while running 'test_crash.py::test_hangs'") == 1, out[-3000:]
    assert out.count("node down") == 1, out[-3000:]  # the crashed test is not run again
    assert "1 failed, 7 passed" in out, out[-3000:]


def test_conftest_installs_the_crash_safe_scheduler():
    tree = ast.parse((Path(__file__).resolve().parent / "conftest.py").read_text(encoding = "utf-8"))
    hooks = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "pytest_xdist_make_scheduler"
    ]
    assert len(hooks) == 1
    assert "make_scheduler" in {
        getattr(node.func, "id", None) for node in ast.walk(hooks[0]) if isinstance(node, ast.Call)
    }
