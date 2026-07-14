# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from typing import Optional

import typer

CORE_PACKAGES = ["transformers", "trl", "peft"]
GPU_PACKAGES = ["triton", "bitsandbytes", "xformers"]
EXTRA_PACKAGES = ["vllm", "mlx", "mlx-lm"]
IMPORT_TIMEOUT = 90
INSTALL_DOCS = "https://unsloth.ai/docs/get-started/install"
UPGRADE = "pip install --upgrade unsloth unsloth_zoo"

_MARKS = {
    "ok": ("✓", typer.colors.GREEN),
    "warn": ("!", typer.colors.YELLOW),
    "fail": ("✗", typer.colors.RED),
    "info": ("·", None),
}


@dataclass
class Row:
    status: str  # ok | warn | fail | info
    text: str
    fix: Optional[str] = None


def _pkg(name: str) -> Optional[str]:
    try:
        return pkg_version(name)
    except PackageNotFoundError:
        return None


def _latest_pypi(name: str) -> Optional[str]:
    import urllib.request

    try:
        with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/json", timeout = 2) as resp:
            return json.load(resp)["info"]["version"]
    except Exception:
        return None


def _env_kind() -> str:
    if "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ:
        return "Colab"
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ or "KAGGLE_URL_BASE" in os.environ:
        return "Kaggle"
    if os.path.exists("/.dockerenv"):
        return "Docker"
    if sys.platform == "linux" and "microsoft" in platform.uname().release.lower():
        return "WSL2"
    return "local"


def _run(cmd: list[str], timeout: int = 10) -> Optional[str]:
    try:
        out = subprocess.run(cmd, capture_output = True, text = True, timeout = timeout)
        return out.stdout if out.returncode == 0 else None
    except Exception:
        return None


def _gpu_info() -> dict:
    info: dict = {"kind": None, "gpus": [], "driver": None, "driver_cuda": None}

    if sys.platform == "darwin" and platform.machine() == "arm64":
        info["kind"] = "apple"
        chip = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        mem = _run(["sysctl", "-n", "hw.memsize"])
        gb = f"{int(mem) / 1024**3:.0f} GB unified" if mem else "unknown memory"
        info["gpus"] = [{"name": (chip or "Apple Silicon").strip(), "memory": gb}]
        return info

    if shutil.which("nvidia-smi"):
        rows = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
        if rows:
            info["kind"] = "nvidia"
            for line in rows.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    info["gpus"].append({"name": parts[0], "memory": parts[1]})
                    info["driver"] = parts[2]
            head = _run(["nvidia-smi"]) or ""
            match = re.search(r"CUDA Version:\s*([0-9.]+)", head)
            if match:
                info["driver_cuda"] = match.group(1)
            return info

    if shutil.which("rocm-smi"):
        info["kind"] = "amd"
        return info

    return info


def _torch_probe() -> Optional[dict]:
    code = (
        "import json, torch\n"
        "d = {'version': torch.__version__, 'cuda': torch.version.cuda,"
        " 'hip': getattr(torch.version, 'hip', None),"
        " 'available': torch.cuda.is_available()}\n"
        "xpu = getattr(torch, 'xpu', None)\n"
        "d['xpu'] = bool(xpu and xpu.is_available())\n"
        "d['mps'] = bool(getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available())\n"
        "print(json.dumps(d))"
    )
    out = _run([sys.executable, "-c", code], timeout = 60)
    if not out:
        return None
    try:
        return json.loads(out.strip().splitlines()[-1])
    except Exception:
        return None


def _mlx_probe() -> bool:
    code = "from unsloth_zoo.mlx import is_mlx_available; raise SystemExit(not is_mlx_available())"
    return _run([sys.executable, "-c", code]) is not None


def _import_probe() -> tuple[bool, str]:
    try:
        out = subprocess.run(
            [sys.executable, "-c", "import unsloth"],
            capture_output = True,
            text = True,
            timeout = IMPORT_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return False, f"timed out after {IMPORT_TIMEOUT}s"
    except Exception as e:
        return False, str(e)
    if out.returncode == 0:
        return True, ""
    lines = [l for l in out.stderr.strip().splitlines() if l.strip()]
    return False, lines[-1][:200] if lines else "unknown error"


def _cuda_newer_than_driver(torch_cuda: str, driver_cuda: str) -> bool:
    try:
        tc = tuple(int(x) for x in torch_cuda.split("."))
        dc = tuple(int(x) for x in driver_cuda.split("."))
        return tc > dc
    except Exception:
        return False


def _collect() -> list[Row]:
    rows: list[Row] = []

    unsloth_v = _pkg("unsloth")
    zoo_v = _pkg("unsloth_zoo")
    head = f"unsloth {unsloth_v or 'not installed'}"
    if zoo_v:
        head += f" · unsloth_zoo {zoo_v}"
    if not unsloth_v:
        rows.append(Row("fail", head, "pip install unsloth"))
    else:
        latest = _latest_pypi("unsloth")
        if latest and latest != unsloth_v:
            rows.append(Row("warn", f"{head} — {latest} is available", UPGRADE))
        else:
            rows.append(Row("ok", head + (" (latest)" if latest else "")))

    os_name = f"macOS {platform.mac_ver()[0]}" if sys.platform == "darwin" else f"{platform.system()} {platform.release()}"
    venv = " (venv)" if sys.prefix != sys.base_prefix else ""
    rows.append(Row("info", f"{os_name} {platform.machine()} · Python {platform.python_version()}{venv} · {_env_kind()}"))

    gpu = _gpu_info()
    torch_info = _torch_probe()
    mlx_ready = _mlx_probe() if gpu["kind"] == "apple" else False

    if gpu["kind"] == "apple":
        g = gpu["gpus"][0]
        if mlx_ready:
            rows.append(Row("ok", f"GPU: {g['name']} · {g['memory']} · MLX available"))
        else:
            rows.append(Row("fail", f"GPU: {g['name']} · {g['memory']} · Unsloth's MLX backend is unavailable", "pip install --upgrade mlx unsloth_zoo"))
    elif gpu["kind"] == "nvidia" and gpu["gpus"]:
        for g in gpu["gpus"]:
            desc = f"GPU: {g['name']} · {g['memory']}"
            if gpu["driver"]:
                desc += f" · driver {gpu['driver']}"
            if gpu["driver_cuda"]:
                desc += f" (CUDA ≤ {gpu['driver_cuda']})"
            if torch_info and torch_info.get("available"):
                rows.append(Row("ok", desc))
            else:
                rows.append(Row("fail", desc + " · PyTorch cannot see it", f"Install a CUDA-enabled torch build: {INSTALL_DOCS}"))
    elif gpu["kind"] == "amd":
        if torch_info and torch_info.get("available") and torch_info.get("hip"):
            rows.append(Row("ok", "GPU: AMD (ROCm available in PyTorch)"))
        else:
            rows.append(Row("fail", "GPU: AMD detected but PyTorch ROCm unavailable", f"Install a ROCm-enabled torch build: {INSTALL_DOCS}"))
    elif torch_info and torch_info.get("available"):
        rows.append(Row("ok", "GPU: detected via torch"))
    elif torch_info and torch_info.get("xpu"):
        rows.append(Row("ok", "GPU: Intel XPU available in PyTorch"))
    else:
        rows.append(Row("fail", "GPU: none detected — Unsloth needs an NVIDIA/AMD/Intel GPU or Apple Silicon"))

    torch_v = torch_info["version"] if torch_info else _pkg("torch")
    torch_cuda = torch_info.get("cuda") if torch_info else None
    if torch_cuda == "None":
        torch_cuda = None

    if torch_v:
        if torch_cuda and gpu["driver_cuda"] and _cuda_newer_than_driver(torch_cuda, gpu["driver_cuda"]):
            rows.append(Row(
                "fail",
                f"torch {torch_v} — built for CUDA {torch_cuda} but driver supports ≤ {gpu['driver_cuda']}",
                f"Reinstall torch built for CUDA ≤ {gpu['driver_cuda']}: {INSTALL_DOCS}",
            ))
        else:
            if torch_cuda:
                backend = f"cu{torch_cuda}"
            elif torch_info and torch_info.get("hip"):
                backend = f"rocm {torch_info['hip']}"
            elif torch_info and torch_info.get("mps"):
                backend = "mps"
            elif torch_info and torch_info.get("xpu"):
                backend = "xpu"
            else:
                backend = "cpu"
            rows.append(Row("ok", f"torch {torch_v} ({backend})"))
    elif gpu["kind"] == "apple":
        rows.append(Row("info", "torch not installed (not required with MLX)"))
    else:
        rows.append(Row("fail", "torch not installed", "pip install torch"))

    missing: list[str] = []

    def versions(names: list[str], required: bool) -> str:
        parts = []
        for name in names:
            v = _pkg(name)
            if v:
                parts.append(f"{name} {v}")
            elif required:
                missing.append(name)
        return " · ".join(parts)

    for line in (
        versions(CORE_PACKAGES, required = True),
        versions(GPU_PACKAGES, required = gpu["kind"] == "nvidia"),
        versions(EXTRA_PACKAGES, required = False),
    ):
        if line:
            rows.append(Row("info", line))
    for name in missing:
        rows.append(Row("warn", f"{name} not installed"))

    for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        value = os.environ.get(name)
        if value is not None:
            rows.append(Row("info", f"{name}={value[:200]}"))

    if unsloth_v:
        typer.secho("… testing `import unsloth` (can take a minute)", dim = True, err = True)
        ok, err = _import_probe()
        if ok:
            rows.append(Row("ok", "`import unsloth` OK"))
        else:
            rows.append(Row("fail", f"`import unsloth` failed: {err}", UPGRADE))

    return rows


def _render_terminal(rows: list[Row]) -> None:
    typer.secho("Unsloth Doctor 🦥", bold = True)
    typer.echo()
    for row in rows:
        mark, color = _MARKS[row.status]
        typer.echo("  ", nl = False)
        typer.secho(mark, fg = color, nl = False)
        typer.echo(f" {row.text}")

    fixes = list(dict.fromkeys(r.fix for r in rows if r.fix))
    fails = sum(1 for r in rows if r.status == "fail")
    warns = sum(1 for r in rows if r.status == "warn")
    typer.echo()
    if fails or warns:
        typer.secho(
            f"Doctor found {fails} error(s), {warns} warning(s).",
            fg = typer.colors.RED if fails else typer.colors.YELLOW,
        )
        if fixes:
            typer.echo("\nFix:")
            for fix in fixes:
                typer.echo(f"  {fix}")
    else:
        typer.secho("No issues found.", fg = typer.colors.GREEN)


def _render_report(rows: list[Row]) -> None:
    plain = {"ok": "[ok]", "warn": "[!]", "fail": "[x]", "info": "[-]"}
    typer.echo("### Unsloth `doctor` report\n")
    typer.echo("```text")
    for row in rows:
        typer.echo(f"{plain[row.status]} {row.text}")
    typer.echo("```")


def doctor(
    report: bool = typer.Option(False, "--report", help = "Print a markdown block to paste into GitHub issues."),
):
    """Check your Unsloth install and print the environment info needed for bug reports."""
    rows = _collect()
    if report:
        _render_report(rows)
    else:
        _render_terminal(rows)
    if any(r.status == "fail" for r in rows):
        raise typer.Exit(code = 1)
