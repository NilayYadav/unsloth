# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Bootstrap a bare VM into a training-ready Unsloth remote.

Every step is an idempotent shell command over SSH: install uv, build an
isolated env at ~/.unsloth/env, install unsloth + the Studio backend deps,
mint an API key, and supervise the headless agent (systemd --user preferred,
tmux fallback).
"""

from __future__ import annotations

import socket
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from unsloth_cli.remote import RemoteError
from unsloth_cli.remote.ssh import SSHRunner
from unsloth_cli.remote.state import RemoteCapabilities

ENV_DIR = "~/.unsloth/env"
ENV_PYTHON = f"{ENV_DIR}/bin/python"
ENV_UNSLOTH = f"{ENV_DIR}/bin/unsloth"
AGENT_SERVICE = "unsloth-agent"
UV_INSTALL_URL = "https://astral.sh/uv/install.sh"
# uv lands in ~/.local/bin (older installers used ~/.cargo/bin).
_UV = 'env PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH" uv'

_SYSTEMD_UNIT = """\
[Unit]
Description=Unsloth Remote Agent (headless Studio backend)
After=network.target

[Service]
ExecStart=%h/.unsloth/env/bin/python -m studio.backend.run --api-only --silent --no-cloudflare --host 127.0.0.1 --port {port}
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
"""

_TMUX_LOOP = (
    "while true; do "
    "~/.unsloth/env/bin/python -m studio.backend.run "
    "--api-only --silent --no-cloudflare --host 127.0.0.1 --port {port}; "
    "sleep 5; done"
)

_WAIT_FOR_AGENT_PY = (
    "import time, sys, urllib.request\n"
    "deadline = time.time() + {timeout}\n"
    "while time.time() < deadline:\n"
    "    try:\n"
    "        r = urllib.request.urlopen('http://127.0.0.1:{port}/api/health', timeout = 2)\n"
    "        if r.status == 200:\n"
    "            sys.exit(0)\n"
    "    except Exception:\n"
    "        pass\n"
    "    time.sleep(1)\n"
    "sys.exit(1)\n"
)


@dataclass
class BootstrapResult:
    api_key: str
    supervision: str
    remote_unsloth_version: Optional[str] = None


def _local_unsloth_pin() -> Optional[str]:
    """Pin the remote to the laptop's unsloth version when it is a clean release."""
    try:
        from importlib.metadata import version
        v = version("unsloth")
    except Exception:
        return None
    if v and all(c.isdigit() or c == "." for c in v):
        return v
    return None


def bootstrap_steps(
    caps: RemoteCapabilities,
    agent_port: int,
    skip_install: bool = False,
) -> List[Tuple[str, Callable[[SSHRunner], Optional[str]]]]:
    """Ordered (label, action) bootstrap steps. Actions may return a value."""
    steps: List[Tuple[str, Callable[[SSHRunner], Optional[str]]]] = []

    if not skip_install:
        steps.append((
            "Install uv",
            lambda r: r.run(
                f"command -v uv >/dev/null 2>&1 || test -x ~/.local/bin/uv "
                f"|| curl -LsSf {UV_INSTALL_URL} | sh",
                timeout = 300,
            ) and None,
        ))
        steps.append((
            "Create isolated Python env",
            lambda r: r.run(
                f"test -x {ENV_PYTHON} || {_UV} venv {ENV_DIR} --python 3.11",
                timeout = 600,
            ) and None,
        ))
        pin = _local_unsloth_pin()
        spec = f"unsloth[huggingface]=={pin}" if pin else "unsloth[huggingface]"
        packages = f'"{spec}"'
        if caps.driver:
            # The huggingface extra has no bitsandbytes; 4-bit training on
            # NVIDIA needs it (mirrors what the cuXXX extras add).
            packages += ' "bitsandbytes>=0.45.5,!=0.46.0,!=0.48.0"'
        steps.append((
            "Install unsloth (downloads torch — takes a few minutes)",
            lambda r: r.run(
                f"{_UV} pip install --python {ENV_PYTHON} {packages}",
                timeout = 2400,
            ) and None,
        ))
        # Note the quoting: the -c payload is single-quoted for the remote
        # shell, so everything inside it must use double quotes only.
        locate_requirements = (
            f"{ENV_PYTHON} -c 'import studio, os; "
            'print(os.path.join(os.path.dirname(studio.__file__), '
            '"backend", "requirements", "studio.txt"))\''
        )
        steps.append((
            "Install agent server dependencies",
            lambda r: r.run(
                f'{_UV} pip install --python {ENV_PYTHON} -r "$({locate_requirements})"',
                timeout = 900,
            ) and None,
        ))

    steps.append(("Mint agent API key", _mint_api_key))
    if caps.has_systemd_user:
        steps.append(("Start agent (systemd --user)", lambda r: _supervise_systemd(r, agent_port)))
    else:
        steps.append(("Start agent (tmux)", lambda r: _supervise_tmux(r, agent_port)))
    steps.append((
        "Wait for agent health",
        lambda r: _wait_for_agent(r, agent_port) and None,
    ))
    steps.append(("Read agent unsloth version", _remote_version))
    return steps


def _mint_api_key(runner: SSHRunner) -> str:
    label = f"remote-{socket.gethostname()}"
    proc = runner.run(
        f"{ENV_UNSLOTH} studio create-api-key --name '{label}'", timeout = 120,
    )
    lines = [l.strip() for l in proc.stdout.splitlines() if l.strip()]
    if not lines:
        raise RemoteError(
            "create-api-key printed nothing.",
            hint = "The remote unsloth install may be broken; re-run `unsloth remote add`.",
        )
    return lines[-1]


def _supervise_systemd(runner: SSHRunner, port: int) -> str:
    unit = _SYSTEMD_UNIT.format(port = port)
    runner.run(
        "mkdir -p ~/.config/systemd/user && "
        f"cat > ~/.config/systemd/user/{AGENT_SERVICE}.service",
        input_text = unit,
    )
    # Linger keeps the user manager (and the agent) alive across logout/reboot.
    runner.run(f'loginctl enable-linger "$USER" 2>/dev/null || true', check = False)
    runner.run(
        f"systemctl --user daemon-reload && systemctl --user enable {AGENT_SERVICE} "
        f"&& systemctl --user restart {AGENT_SERVICE}",
        timeout = 120,
    )
    return "systemd"


def _supervise_tmux(runner: SSHRunner, port: int) -> str:
    loop = _TMUX_LOOP.format(port = port)
    runner.run(
        f"tmux kill-session -t {AGENT_SERVICE} 2>/dev/null; "
        f"tmux new-session -d -s {AGENT_SERVICE} '{loop}'",
        timeout = 60,
    )
    return "tmux"


def _wait_for_agent(runner: SSHRunner, port: int, timeout: int = 120) -> bool:
    script = _WAIT_FOR_AGENT_PY.format(port = port, timeout = timeout)
    proc = runner.run(
        f"{ENV_PYTHON} - <<'UNSLOTH_EOF'\n{script}UNSLOTH_EOF",
        timeout = timeout + 30,
        check = False,
    )
    if proc.returncode != 0:
        raise RemoteError(
            "The agent did not become healthy in time.",
            hint = (
                "Inspect it on the VM: systemctl --user status unsloth-agent, or "
                "tail ~/.unsloth/studio/logs/server/server-*.log"
            ),
        )
    return True


def _remote_version(runner: SSHRunner) -> Optional[str]:
    proc = runner.run(
        f"{ENV_PYTHON} -c \"from importlib.metadata import version; print(version('unsloth'))\"",
        check = False,
    )
    return proc.stdout.strip() or None


def run_bootstrap(
    runner: SSHRunner,
    caps: RemoteCapabilities,
    agent_port: int,
    skip_install: bool = False,
    on_step: Optional[Callable[[str], None]] = None,
) -> BootstrapResult:
    api_key = ""
    supervision = "none"
    remote_version: Optional[str] = None
    for label, action in bootstrap_steps(caps, agent_port, skip_install = skip_install):
        if on_step is not None:
            on_step(label)
        try:
            value = action(runner)
        except RemoteError as e:
            raise RemoteError(f"[{label}] {e}", hint = e.hint) from None
        if label == "Mint agent API key":
            api_key = value or ""
        elif label.startswith("Start agent"):
            supervision = value or "none"
        elif label == "Read agent unsloth version":
            remote_version = value
    return BootstrapResult(
        api_key = api_key,
        supervision = supervision,
        remote_unsloth_version = remote_version,
    )
