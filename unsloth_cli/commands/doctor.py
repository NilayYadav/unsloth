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
from importlib.metadata import PackageNotFoundError, requires
from importlib.metadata import version as pkg_version
from typing import Optional

import typer

CORE_PACKAGES = ["torch", "triton", "xformers", "bitsandbytes", "transformers", "trl", "peft", "unsloth_zoo"]
OPTIONAL_PACKAGES = ["vllm", "mlx", "mlx-lm"]
APPLE_OPTIONAL = {"torch", "triton", "xformers", "bitsandbytes"}
IMPORT_TIMEOUT = 180

_MARKS = {
    "ok": ("✓", typer.colors.GREEN),
    "warn": ("!", typer.colors.YELLOW),
    "fail": ("✗", typer.colors.RED),
    "info": ("·", None),
}


@dataclass
class Row:
    status: str  # ok | warn | fail | info | header
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
    return False, out.stderr.strip()


_KNOWN_ERRORS = [
    ("No module named 'unsloth'", "unsloth is not installed in this environment", "pip install unsloth"),
    ("LLVM ERROR", "Triton kernel compilation failed — torch/triton/GPU mismatch", None),
    ("PassManager::run failed", "Triton kernel compilation failed — torch/triton/GPU mismatch", None),
    ("CUDA Setup failed", "bitsandbytes could not find CUDA libraries", "pip install --force-reinstall bitsandbytes"),
    ("undefined symbol", "a package was built against a different torch — reinstall matching wheels", None),
    ("libcuda", "CUDA driver library not found — check the NVIDIA driver install", None),
    ("only works on NVIDIA, AMD and Intel", "no supported GPU detected by torch", None),
]


def _classify_import_error(stderr: str) -> tuple[str, Optional[str]]:
    for needle, message, fix in _KNOWN_ERRORS:
        if needle in stderr:
            return message, fix
    tail = stderr.strip().splitlines()[-1] if stderr.strip() else "unknown error"
    return tail[:200], None


# Mirrors the torch/CUDA table in unsloth/_auto_install.py
def _install_extra(torch_version: str, cuda: Optional[str]) -> Optional[str]:
    from packaging.version import Version as V

    if not cuda or cuda == "None":
        return None
    cuda = str(cuda)
    if cuda not in ("11.8", "12.1", "12.4", "12.6", "12.8", "13.0"):
        return None
    match = re.match(r"[0-9.]{3,}", torch_version)
    if not match:
        return None
    v = V(match.group(0))
    if v <= V("2.1.0"):
        return None
    table = [
        ("2.1.1", "torch211", "le"), ("2.1.2", "torch212", "le"), ("2.3.0", "torch220", "lt"),
        ("2.4.0", "torch230", "lt"), ("2.5.0", "torch240", "lt"), ("2.5.1", "torch250", "lt"),
        ("2.5.1", "torch251", "le"), ("2.7.0", "torch260", "lt"), ("2.7.9", "torch270", "lt"),
        ("2.8.0", "torch271", "lt"), ("2.8.9", "torch280", "lt"), ("2.9.1", "torch290", "lt"),
        ("2.9.2", "torch291", "lt"), ("2.10.1", "torch2100", "lt"),
    ]
    tag = None
    for bound, name, op in table:
        if (op == "le" and v <= V(bound)) or (op == "lt" and v < V(bound)):
            tag = name
            break
    if tag is None:
        return None
    if v > V("2.6.9") and cuda not in ("11.8", "12.6", "12.8", "13.0"):
        return None
    if v >= V("2.10.0") and cuda not in ("12.6", "12.8", "13.0"):
        return None
    return f"cu{cuda.replace('.', '')}-{tag}"


def _best_cuda_for_driver(driver_cuda: str) -> Optional[str]:
    try:
        dc = tuple(int(x) for x in driver_cuda.split("."))
    except Exception:
        return None
    supported = ["11.8", "12.1", "12.4", "12.6", "12.8", "13.0"]
    best = None
    for cuda in supported:
        if tuple(int(x) for x in cuda.split(".")) <= dc:
            best = cuda
    return best


def _torch_pin_ok(package: str, torch_version: str) -> Optional[bool]:
    from packaging.requirements import Requirement
    from packaging.version import Version as V

    try:
        base = re.match(r"[0-9.]{3,}", torch_version)
        if not base:
            return None
        installed = V(base.group(0))
        for line in requires(package) or []:
            try:
                req = Requirement(line)
            except Exception:
                continue
            if req.name.lower() != "torch" or not req.specifier:
                continue
            if req.marker is not None and not req.marker.evaluate():
                continue
            return req.specifier.contains(str(installed), prereleases = True)
    except Exception:
        return None
    return None


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
    latest = _latest_pypi("unsloth") if unsloth_v else None
    head = f"unsloth {unsloth_v or 'not installed'}"
    if zoo_v:
        head += f" · unsloth_zoo {zoo_v}"
    if latest and unsloth_v and latest != unsloth_v:
        rows.append(Row("warn", f"{head} ({latest} available)", "pip install --upgrade unsloth unsloth_zoo"))
    elif unsloth_v:
        rows.append(Row("ok", head + (" (latest)" if latest else "")))
    else:
        rows.append(Row("fail", head, "pip install unsloth"))

    os_name = f"macOS {platform.mac_ver()[0]}" if sys.platform == "darwin" else f"{platform.system()} {platform.release()}"
    venv = " (venv)" if sys.prefix != sys.base_prefix else ""
    rows.append(Row("info", f"System: {os_name} {platform.machine()} · Python {platform.python_version()}{venv} · {_env_kind()}"))

    gpu = _gpu_info()
    torch_info = _torch_probe()
    if gpu["gpus"]:
        for g in gpu["gpus"]:
            desc = f"GPU: {g['name']} · {g['memory']}"
            if gpu["driver"]:
                desc += f" · driver {gpu['driver']}"
                if gpu["driver_cuda"]:
                    desc += f" (CUDA ≤ {gpu['driver_cuda']})"
            rows.append(Row("ok", desc))
    elif gpu["kind"] == "amd":
        rows.append(Row("ok", "GPU: AMD (rocm-smi found)"))
    elif torch_info and torch_info.get("available"):
        rows.append(Row("ok", "GPU: detected via torch"))
    else:
        rows.append(Row("fail", "GPU: none detected — Unsloth needs an NVIDIA/AMD/Intel GPU or Apple Silicon"))

    rows.append(Row("header", "Packages"))
    torch_v = torch_info["version"] if torch_info else _pkg("torch")
    torch_cuda = torch_info.get("cuda") if torch_info else None
    if torch_cuda == "None":
        torch_cuda = None
    fix_extra = _install_extra(torch_v, torch_cuda) if torch_v else None
    if fix_extra is None and torch_v and gpu["driver_cuda"]:
        best = _best_cuda_for_driver(gpu["driver_cuda"])
        fix_extra = _install_extra(torch_v, best) if best else None
    reinstall = f'pip install --force-reinstall "unsloth[{fix_extra}]"' if fix_extra else None

    if torch_v:
        if torch_cuda:
            backend = f"cu{torch_cuda}"
        elif torch_info and torch_info.get("hip"):
            backend = "rocm"
        elif torch_info and torch_info.get("mps"):
            backend = "mps"
        else:
            backend = "cpu"
        if torch_cuda and gpu["driver_cuda"] and _cuda_newer_than_driver(torch_cuda, gpu["driver_cuda"]):
            best = _best_cuda_for_driver(gpu["driver_cuda"])
            driver_extra = _install_extra(torch_v, best) if best else None
            driver_fix = f'pip install --force-reinstall "unsloth[{driver_extra}]"' if driver_extra else reinstall
            rows.append(Row("fail", f"torch {torch_v} — built for CUDA {torch_cuda} but driver supports ≤ {gpu['driver_cuda']}", driver_fix))
        else:
            rows.append(Row("ok", f"torch {torch_v} ({backend})"))
    elif gpu["kind"] == "apple":
        rows.append(Row("info", "torch not installed (optional with MLX)"))
    else:
        rows.append(Row("fail", "torch not installed", "pip install torch"))

    for name in CORE_PACKAGES[1:] + OPTIONAL_PACKAGES:
        v = _pkg(name)
        if v is None:
            if name in OPTIONAL_PACKAGES:
                continue
            optional = gpu["kind"] == "apple" and name in APPLE_OPTIONAL or name == "triton"
            rows.append(Row("info" if optional else "warn", f"{name} not installed"))
            continue
        pin_ok = _torch_pin_ok(name, torch_v) if torch_v else None
        if pin_ok is False:
            rows.append(Row("fail", f"{name} {v} — built for a different torch (yours is {torch_v})", reinstall))
        else:
            rows.append(Row("ok", f"{name} {v}"))

    rows.append(Row("header", "Import"))
    if unsloth_v:
        ok, stderr = _import_probe()
        if ok:
            rows.append(Row("ok", "`import unsloth` succeeded"))
        else:
            message, fix = _classify_import_error(stderr)
            rows.append(Row("fail", f"`import unsloth` failed: {message}", fix or reinstall))
    else:
        rows.append(Row("info", "import check skipped (unsloth not installed)"))

    return rows


def _render_terminal(rows: list[Row]) -> None:
    typer.secho("Unsloth Doctor 🦥", bold = True)
    typer.echo()
    for row in rows:
        if row.status == "header":
            typer.secho(row.text, bold = True)
            continue
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
        if row.status == "header":
            typer.echo(row.text)
        else:
            typer.echo(f"  {plain[row.status]} {row.text}")
    typer.echo("```")


def doctor(
    report: bool = typer.Option(False, "--report", help = "Print a markdown block to paste into GitHub issues."),
):
    """Diagnose your Unsloth installation and environment."""
    rows = _collect()
    if report:
        _render_report(rows)
    else:
        _render_terminal(rows)
    if any(r.status == "fail" for r in rows):
        raise typer.Exit(code = 1)
