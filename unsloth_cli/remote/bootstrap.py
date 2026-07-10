# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

# Probe a bare VM and bootstrap it into a training-ready Unsloth remote.

from __future__ import annotations

import re
import secrets
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from unsloth_cli.remote import RemoteError
from unsloth_cli.remote.ssh import SSHRunner
from unsloth_cli.remote.state import RemoteCapabilities

ENV_DIR = "~/.unsloth/env"
ENV_PYTHON = f"{ENV_DIR}/bin/python"
ENV_UNSLOTH = f"{ENV_DIR}/bin/unsloth"
AGENT_SERVICE = "unsloth-agent"
UV_INSTALL_URL = "https://astral.sh/uv/install.sh"
_UV = 'env PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH" uv'

# Shared pin set: keeps the two pip installs from picking conflicting versions.
CONSTRAINTS_REMOTE = "~/.unsloth/constraints.txt"

DEFAULT_ADMIN_USERNAME = "unsloth"
BOOTSTRAP_PASSWORD_PATH = "~/.unsloth/studio/auth/.bootstrap_password"

# Torch cu12x wheels refuse to initialize below this driver major version.
MIN_DRIVER_MAJOR = 525
RECOMMENDED_DRIVER = "550"
MIN_DISK_FREE_GB = 40.0


# ---------------------------------------------------------------------------
# Probe: one SSH round-trip emitting flat key=value lines.

PROBE_SCRIPT = """\
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | sed 's/^/gpu=/'
echo "cuda=$(nvidia-smi 2>/dev/null | grep -o 'CUDA Version: [0-9.]*' | cut -d' ' -f3)"
echo "python=$(python3 --version 2>/dev/null | cut -d' ' -f2)"
echo "disk_kb=$(df -Pk "$HOME" 2>/dev/null | tail -1 | awk '{print $4}')"
echo "arch=$(uname -m)"
for t in rsync tmux uv curl wget; do
  command -v "$t" >/dev/null 2>&1 && echo "$t=yes" || echo "$t=no"
done
echo "uid=$(id -u)"
sudo -n true >/dev/null 2>&1 && echo sudo=yes || echo sudo=no
s=$(systemctl --user is-system-running 2>/dev/null); echo "systemd=${s:-none}"
echo "os=$( (. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME") || uname -s)"
"""


@dataclass
class ProbeCheck:
    name: str
    ok: bool
    value: str
    hint: str = ""
    fatal: bool = False


@dataclass
class ProbeResult:
    checks: List[ProbeCheck] = field(default_factory = list)
    capabilities: RemoteCapabilities = field(default_factory = RemoteCapabilities)

    @property
    def fatal_failures(self) -> List[ProbeCheck]:
        return [c for c in self.checks if not c.ok and c.fatal]

    @property
    def warnings(self) -> List[ProbeCheck]:
        return [c for c in self.checks if not c.ok and not c.fatal]


def run_probe(runner: SSHRunner) -> ProbeResult:
    return parse_probe_output(runner.run(PROBE_SCRIPT, timeout = 60).stdout)


def missing_system_packages(caps: RemoteCapabilities) -> List[str]:
    missing = []
    if not caps.has_systemd_user and not caps.has_tmux:
        missing.append("tmux")
    if not caps.has_rsync:
        missing.append("rsync")
    if not (caps.has_curl or caps.has_wget or caps.has_uv):
        missing.append("curl")
    return missing


def parse_probe_output(output: str) -> ProbeResult:
    values: dict = {}
    for line in output.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            values.setdefault(key.strip(), []).append(value.strip())

    def one(key: str) -> str:
        return (values.get(key) or [""])[0]

    result = ProbeResult()
    checks = result.checks
    caps = result.capabilities

    gpus = [v for v in values.get("gpu", []) if v]
    for line in gpus:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            caps.gpu_names.append(parts[0])
            mem = parts[1].split()
            if mem and mem[0].replace(".", "", 1).isdigit():
                caps.vram_gb.append(round(float(mem[0]) / 1024, 1))
            caps.driver = parts[2]
    if not gpus:
        checks.append(ProbeCheck(
            name = "GPU", ok = False, value = "no NVIDIA GPU visible", fatal = True,
            hint = (
                "nvidia-smi is missing or failed. If this box has an NVIDIA GPU, install the "
                "driver first (Ubuntu: sudo apt install nvidia-driver-550, then reboot). "
                "Cloud GPU images usually ship with it preinstalled."
            ),
        ))
    else:
        names = ", ".join(
            f"{n} ({v:g} GB)" for n, v in zip(caps.gpu_names, caps.vram_gb)
        ) or gpus[0]
        checks.append(ProbeCheck(name = "GPU", ok = True, value = names))
        head = (caps.driver or "").split(".")[0]
        if head.isdigit() and int(head) < MIN_DRIVER_MAJOR:
            checks.append(ProbeCheck(
                name = "NVIDIA driver", ok = False, value = caps.driver, fatal = True,
                hint = (
                    f"torch CUDA 12 wheels need driver >= {MIN_DRIVER_MAJOR} "
                    f"(recommended {RECOMMENDED_DRIVER}+). "
                    f"Upgrade: sudo apt install nvidia-driver-{RECOMMENDED_DRIVER}, then reboot."
                ),
            ))
        else:
            checks.append(ProbeCheck(
                name = "NVIDIA driver", ok = True, value = caps.driver or "unknown",
            ))

    caps.cuda = one("cuda") or None
    caps.arch = one("arch") or None
    caps.os_release = one("os") or None

    caps.python = one("python") or None
    checks.append(
        ProbeCheck(name = "python3", ok = True, value = caps.python) if caps.python
        else ProbeCheck(
            name = "python3", ok = False, value = "not found",
            hint = "Not required: uv will install a managed Python 3.11.",
        )
    )

    if one("disk_kb").isdigit():
        caps.disk_free_gb = round(int(one("disk_kb")) / 1024 / 1024, 1)
        enough = caps.disk_free_gb >= MIN_DISK_FREE_GB
        checks.append(ProbeCheck(
            name = "Disk free ($HOME)", ok = enough, value = f"{caps.disk_free_gb:g} GB",
            hint = "" if enough else (
                f"Less than {MIN_DISK_FREE_GB:g} GB free. Models, datasets and checkpoints are "
                "large; free up space or expand the volume."
            ),
        ))
    else:
        checks.append(ProbeCheck(
            name = "Disk free ($HOME)", ok = False, value = "could not determine",
            hint = "df output was unparseable; continuing anyway.",
        ))

    caps.has_rsync = one("rsync") == "yes"
    caps.has_tmux = one("tmux") == "yes"
    caps.has_uv = one("uv") == "yes"
    caps.has_curl = one("curl") == "yes"
    caps.has_wget = one("wget") == "yes"
    caps.is_root = one("uid") == "0"
    caps.has_passwordless_sudo = one("sudo") == "yes"
    caps.has_systemd_user = one("systemd") in {"running", "degraded"}

    missing = missing_system_packages(caps)
    if missing and caps.can_install_packages:
        checks.append(ProbeCheck(
            name = "System packages", ok = True,
            value = f"will auto-install: {', '.join(missing)}",
        ))
    else:
        if not caps.has_rsync:
            checks.append(ProbeCheck(
                name = "rsync", ok = False, value = "not found",
                hint = "File transfers fall back to tar-over-ssh (slower, not resumable). "
                       "Install: sudo apt install rsync",
            ))
        if "curl" in missing:
            checks.append(ProbeCheck(
                name = "curl", ok = False, value = "not found", fatal = True,
                hint = "Bootstrap downloads uv with curl (or wget). Install: sudo apt install curl",
            ))

    if caps.has_systemd_user:
        supervision = ProbeCheck(name = "Agent supervision", ok = True, value = "systemd (user)")
    elif caps.has_tmux:
        supervision = ProbeCheck(name = "Agent supervision", ok = True, value = "tmux")
    elif caps.can_install_packages:
        supervision = ProbeCheck(name = "Agent supervision", ok = True, value = "tmux (auto-install)")
    else:
        supervision = ProbeCheck(
            name = "Agent supervision", ok = False, fatal = True,
            value = "neither user systemd nor tmux available",
            hint = "The agent needs one to stay alive. Install tmux: sudo apt install tmux",
        )
    checks.append(supervision)
    return result


# ---------------------------------------------------------------------------
# Bootstrap: idempotent install steps, then agent supervision and health.

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

_WAIT_FOR_AGENT_PY = """\
import time, sys, urllib.request
deadline = time.time() + {timeout}
while time.time() < deadline:
    try:
        if urllib.request.urlopen("http://127.0.0.1:{port}/api/health", timeout = 2).status == 200:
            sys.exit(0)
    except Exception:
        pass
    time.sleep(1)
sys.exit(1)
"""


@dataclass
class BootstrapResult:
    supervision: str
    remote_unsloth_version: Optional[str] = None


def _local_unsloth_pin() -> Optional[str]:
    # Pin the remote to the laptop's version when it is a clean release.
    try:
        from importlib.metadata import version
        v = version("unsloth")
    except Exception:
        return None
    return v if v and all(c.isdigit() or c == "." for c in v) else None


def _constraints_text() -> Optional[str]:
    # unsloth_cli and studio are siblings in both the repo and the wheel.
    path = (
        Path(__file__).resolve().parents[2]
        / "studio" / "backend" / "requirements" / "single-env" / "constraints.txt"
    )
    try:
        return path.read_text(encoding = "utf-8")
    except OSError:
        return None


def _install_packages_command(packages: List[str], is_root: bool) -> str:
    pkgs = " ".join(packages)
    sudo = "" if is_root else "sudo -n "
    apt = f"{sudo}env DEBIAN_FRONTEND=noninteractive apt-get"
    return (
        f"if command -v apt-get >/dev/null 2>&1; then {apt} update -qq && {apt} install -y -qq {pkgs}; "
        f"elif command -v dnf >/dev/null 2>&1; then {sudo}dnf install -y -q {pkgs}; "
        f"elif command -v yum >/dev/null 2>&1; then {sudo}yum install -y -q {pkgs}; "
        f"else echo 'no supported package manager (apt-get/dnf/yum)' >&2; exit 1; fi"
    )


def _sh(command: str, timeout: int = 120, input_text: Optional[str] = None):
    return lambda runner: runner.run(command, timeout = timeout, input_text = input_text)


def bootstrap_steps(
    caps: RemoteCapabilities,
    agent_port: int,
) -> List[Tuple[str, Callable[[SSHRunner], object]]]:
    steps = []

    missing = missing_system_packages(caps)
    if missing and caps.can_install_packages:
        steps.append((
            f"Install system packages ({', '.join(missing)})",
            _sh(_install_packages_command(missing, caps.is_root), timeout = 600),
        ))

    steps.append(("Install uv", _sh(
        f"command -v uv >/dev/null 2>&1 || test -x ~/.local/bin/uv "
        f"|| (command -v curl >/dev/null 2>&1 && curl -LsSf {UV_INSTALL_URL} | sh) "
        f"|| (command -v wget >/dev/null 2>&1 && wget -qO- {UV_INSTALL_URL} | sh)",
        timeout = 300,
    )))
    steps.append(("Create isolated Python env", _sh(
        f"test -x {ENV_PYTHON} || {_UV} venv {ENV_DIR} --python 3.11", timeout = 600,
    )))

    constraint_flag = ""
    constraints = _constraints_text()
    if constraints:
        constraint_flag = f" -c {CONSTRAINTS_REMOTE}"
        steps.append(("Write dependency constraints", _sh(
            f"mkdir -p ~/.unsloth && cat > {CONSTRAINTS_REMOTE}", input_text = constraints,
        )))

    pin = _local_unsloth_pin()
    packages = f'"unsloth[huggingface]=={pin}"' if pin else '"unsloth[huggingface]"'
    if caps.driver:
        # huggingface extra lacks bitsandbytes; NVIDIA 4-bit needs it.
        packages += ' "bitsandbytes>=0.45.5,!=0.46.0,!=0.48.0"'
    steps.append(("Install unsloth (downloads torch — takes a few minutes)", _sh(
        f"{_UV} pip install --python {ENV_PYTHON}{constraint_flag} {packages}", timeout = 2400,
    )))

    # Single-quoted -c payload: double quotes only inside.
    locate_requirements = (
        f"{ENV_PYTHON} -c 'import studio, os; "
        'print(os.path.join(os.path.dirname(studio.__file__), '
        '"backend", "requirements", "studio.txt"))\''
    )
    steps.append(("Install agent server dependencies", _sh(
        f'{_UV} pip install --python {ENV_PYTHON}{constraint_flag} -r "$({locate_requirements})"',
        timeout = 900,
    )))

    if caps.has_systemd_user:
        steps.append(("Start agent (systemd --user)", lambda r: _supervise_systemd(r, agent_port)))
    else:
        steps.append(("Start agent (tmux)", lambda r: _supervise_tmux(r, agent_port)))
    steps.append(("Wait for agent health", lambda r: _wait_for_agent(r, agent_port)))
    return steps


def _supervise_systemd(runner: SSHRunner, port: int) -> None:
    runner.run(
        "mkdir -p ~/.config/systemd/user && "
        f"cat > ~/.config/systemd/user/{AGENT_SERVICE}.service",
        input_text = _SYSTEMD_UNIT.format(port = port),
    )
    runner.run('loginctl enable-linger "$USER" 2>/dev/null || true', check = False)
    runner.run(
        f"systemctl --user daemon-reload && systemctl --user enable {AGENT_SERVICE} "
        f"&& systemctl --user restart {AGENT_SERVICE}",
        timeout = 120,
    )


def _supervise_tmux(runner: SSHRunner, port: int) -> None:
    runner.run(
        f"tmux kill-session -t {AGENT_SERVICE} 2>/dev/null; "
        f"tmux new-session -d -s {AGENT_SERVICE} '{_TMUX_LOOP.format(port = port)}'",
        timeout = 60,
    )


def _wait_for_agent(runner: SSHRunner, port: int, timeout: int = 120) -> None:
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
    on_step: Optional[Callable[[str], None]] = None,
) -> BootstrapResult:
    for label, action in bootstrap_steps(caps, agent_port):
        if on_step is not None:
            on_step(label)
        try:
            action(runner)
        except RemoteError as e:
            raise RemoteError(f"[{label}] {e}", hint = e.hint) from None
    return BootstrapResult(
        supervision = "systemd" if caps.has_systemd_user else "tmux",
        remote_unsloth_version = _remote_version(runner),
    )


def agent_key_label() -> str:
    hostname = re.sub(r"[^A-Za-z0-9._-]", "-", socket.gethostname() or "client")
    return f"remote-{hostname}"


def initialize_agent_auth(runner: SSHRunner, client) -> str:
    # The human first-run flow, automated over public endpoints: read the
    # bootstrap password via SSH, log in, rotate it, mint an API key.
    proc = runner.run(f"cat {BOOTSTRAP_PASSWORD_PATH}", check = False)
    bootstrap_password = proc.stdout.strip()
    if proc.returncode != 0 or not bootstrap_password:
        raise RemoteError(
            "The agent's admin account is already set up (no bootstrap password on the machine).",
            hint = (
                "Reset it on the VM and re-run `unsloth remote add`:\n"
                f"  {ENV_UNSLOTH} studio reset-password   # then restart the agent:\n"
                f"  systemctl --user restart {AGENT_SERVICE}  (or tmux kill-session -t {AGENT_SERVICE})"
            ),
        )

    token = client.login(DEFAULT_ADMIN_USERNAME, bootstrap_password)
    if token.get("must_change_password", True):
        replacement = secrets.token_urlsafe(24)
        token = client.change_password(
            bootstrap_password, replacement, bearer = token["access_token"],
        )
    return client.mint_api_key(agent_key_label(), bearer = token["access_token"])
