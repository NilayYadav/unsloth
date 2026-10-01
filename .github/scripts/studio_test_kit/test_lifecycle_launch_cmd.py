"""The detached-launch argv must not name a binary the host does not have.

`setsid` is util-linux. On macOS the Popen raised FileNotFoundError before Studio
was started, which surfaced only as a traceback halfway through a two-install
evidence run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from studio_test_kit.lifecycle import detached_launch_cmd


@pytest.fixture
def no_setsid(monkeypatch):
    monkeypatch.setattr("studio_test_kit.lifecycle.shutil.which", lambda _name: None)


@pytest.fixture
def has_setsid(monkeypatch):
    monkeypatch.setattr("studio_test_kit.lifecycle.shutil.which", lambda _name: "/usr/bin/setsid")


def test_a_host_without_setsid_launches_without_it(no_setsid):
    cmd = detached_launch_cmd("/opt/x/unsloth", 8991, Path("/tmp/studio.log"))
    assert "setsid" not in cmd
    assert cmd[0] == "bash"
    assert "unsloth studio -p 8991" in cmd[-1]


def test_a_host_without_setsid_redirects_instead_of_piping(no_setsid):
    """No `tee`: nothing is left to read the pipe, and a server writing into a
    closed pipeline dies with BrokenPipeError mid-run."""
    cmd = detached_launch_cmd("/opt/x/unsloth", 8991, Path("/tmp/studio.log"))
    assert "| tee" not in cmd[-1]
    assert ">> /tmp/studio.log 2>&1" in cmd[-1]


def test_setsid_is_still_used_where_it_exists(has_setsid):
    cmd = detached_launch_cmd("/opt/x/unsloth", 8991, Path("/tmp/studio.log"))
    assert cmd[:3] == ["setsid", "-f", "bash"]
    assert "tee -a /tmp/studio.log" in cmd[-1]


def test_a_path_with_spaces_survives_either_shape(has_setsid, monkeypatch):
    for which in (lambda _n: None, lambda _n: "/usr/bin/setsid"):
        monkeypatch.setattr("studio_test_kit.lifecycle.shutil.which", which)
        cmd = detached_launch_cmd("/opt/my studio/unsloth", 8991, Path("/tmp/my logs/s.log"))
        assert "'/opt/my studio/unsloth'" in cmd[-1]
        assert "'/tmp/my logs/s.log'" in cmd[-1]
