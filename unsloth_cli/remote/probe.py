# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pre-flight probe of a remote machine: GPU, driver, disk, python, tooling.

Everything is gathered in a single SSH round-trip and parsed client-side so
failures surface as actionable messages before any installation starts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from unsloth_cli.remote.ssh import SSHRunner
from unsloth_cli.remote.state import RemoteCapabilities

# Torch cu12x wheels refuse to initialize below this driver major version.
MIN_DRIVER_MAJOR = 525
RECOMMENDED_DRIVER = "550"
MIN_DISK_FREE_GB = 40.0

_SECTION = "===UNSLOTH:"

PROBE_SCRIPT = f"""
echo '{_SECTION}GPU==='
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo NO_GPU
echo '{_SECTION}CUDA==='
nvidia-smi 2>/dev/null | grep -o 'CUDA Version: [0-9.]*' | head -1 || true
echo '{_SECTION}PYTHON==='
python3 --version 2>&1 || echo NO_PYTHON
echo '{_SECTION}DISK==='
df -Pk "$HOME" | tail -1
echo '{_SECTION}ARCH==='
uname -m
echo '{_SECTION}TOOLS==='
for t in rsync tmux systemctl uv curl; do
  if command -v "$t" >/dev/null 2>&1; then echo "$t=yes"; else echo "$t=no"; fi
done
echo '{_SECTION}SYSTEMD==='
systemctl --user is-system-running 2>/dev/null || echo none
echo '{_SECTION}OS==='
( . /etc/os-release 2>/dev/null && echo "$PRETTY_NAME" ) || uname -s
""".strip()


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
    proc = runner.run(PROBE_SCRIPT, timeout = 60)
    return parse_probe_output(proc.stdout)


def _sections(output: str) -> dict:
    sections: dict = {}
    current: Optional[str] = None
    for line in output.splitlines():
        if line.startswith(_SECTION):
            current = line[len(_SECTION):].rstrip("=")
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items()}


def parse_probe_output(output: str) -> ProbeResult:
    sections = _sections(output)
    result = ProbeResult()
    caps = result.capabilities

    # -- GPU + driver ---------------------------------------------------------
    gpu_text = sections.get("GPU", "")
    if not gpu_text or "NO_GPU" in gpu_text:
        result.checks.append(ProbeCheck(
            name = "GPU",
            ok = False,
            value = "no NVIDIA GPU visible",
            hint = (
                "nvidia-smi is missing or failed. If this box has an NVIDIA GPU, install the "
                "driver first (Ubuntu: sudo apt install nvidia-driver-550, then reboot). "
                "Cloud GPU images usually ship with it preinstalled."
            ),
            fatal = True,
        ))
    else:
        driver = None
        for line in gpu_text.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                caps.gpu_names.append(parts[0])
                mem = parts[1].split()
                if mem and mem[0].replace(".", "", 1).isdigit():
                    caps.vram_gb.append(round(float(mem[0]) / 1024, 1))
                driver = parts[2]
        caps.driver = driver
        names = ", ".join(
            f"{n} ({v:g} GB)" for n, v in zip(caps.gpu_names, caps.vram_gb)
        ) or gpu_text
        result.checks.append(ProbeCheck(name = "GPU", ok = True, value = names))

        driver_major = None
        if driver:
            head = driver.split(".")[0]
            driver_major = int(head) if head.isdigit() else None
        if driver_major is not None and driver_major < MIN_DRIVER_MAJOR:
            result.checks.append(ProbeCheck(
                name = "NVIDIA driver",
                ok = False,
                value = driver,
                hint = (
                    f"torch CUDA 12 wheels need driver >= {MIN_DRIVER_MAJOR} "
                    f"(recommended {RECOMMENDED_DRIVER}+). "
                    f"Upgrade: sudo apt install nvidia-driver-{RECOMMENDED_DRIVER}, then reboot."
                ),
                fatal = True,
            ))
        else:
            result.checks.append(ProbeCheck(
                name = "NVIDIA driver", ok = True, value = driver or "unknown",
            ))

    cuda = sections.get("CUDA", "")
    if "CUDA Version:" in cuda:
        caps.cuda = cuda.split("CUDA Version:")[1].strip()

    # -- Python (informational: uv installs its own CPython) -------------------
    python_text = sections.get("PYTHON", "")
    if python_text.startswith("Python "):
        caps.python = python_text.split()[1]
        result.checks.append(ProbeCheck(name = "python3", ok = True, value = caps.python))
    else:
        result.checks.append(ProbeCheck(
            name = "python3",
            ok = False,
            value = "not found",
            hint = "Not required: uv will install a managed Python 3.11.",
        ))

    # -- Disk -------------------------------------------------------------------
    disk_text = sections.get("DISK", "")
    fields = disk_text.split()
    if len(fields) >= 4 and fields[3].isdigit():
        caps.disk_free_gb = round(int(fields[3]) / 1024 / 1024, 1)
        enough = caps.disk_free_gb >= MIN_DISK_FREE_GB
        result.checks.append(ProbeCheck(
            name = "Disk free ($HOME)",
            ok = enough,
            value = f"{caps.disk_free_gb:g} GB",
            hint = "" if enough else (
                f"Less than {MIN_DISK_FREE_GB:g} GB free. Models, datasets and checkpoints are "
                "large; free up space or expand the volume."
            ),
        ))
    else:
        result.checks.append(ProbeCheck(
            name = "Disk free ($HOME)", ok = False, value = "could not determine",
            hint = "df output was unparseable; continuing anyway.",
        ))

    caps.arch = sections.get("ARCH", "") or None
    caps.os_release = sections.get("OS", "") or None

    # -- Tools ------------------------------------------------------------------
    tools = dict(
        line.split("=", 1) for line in sections.get("TOOLS", "").splitlines() if "=" in line
    )
    caps.has_rsync = tools.get("rsync") == "yes"
    caps.has_tmux = tools.get("tmux") == "yes"
    if not caps.has_rsync:
        result.checks.append(ProbeCheck(
            name = "rsync",
            ok = False,
            value = "not found",
            hint = "File transfers fall back to tar-over-ssh (slower, not resumable). "
                   "Install: sudo apt install rsync",
        ))

    systemd_state = sections.get("SYSTEMD", "none")
    caps.has_systemd_user = systemd_state in {"running", "degraded"}
    if not caps.has_systemd_user and not caps.has_tmux:
        result.checks.append(ProbeCheck(
            name = "Agent supervision",
            ok = False,
            value = "neither user systemd nor tmux available",
            hint = "The agent needs one to stay alive. Install tmux: sudo apt install tmux",
            fatal = True,
        ))
    else:
        result.checks.append(ProbeCheck(
            name = "Agent supervision",
            ok = True,
            value = "systemd (user)" if caps.has_systemd_user else "tmux",
        ))

    return result
