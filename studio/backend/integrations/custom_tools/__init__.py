# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import ast
import subprocess
import sys
from pathlib import Path

from core.inference.mcp_client import join_stdio_command, parse_server_headers
from utils.paths import account_path

PREFIX = "custom_tool:"
_LAUNCHER = Path(__file__).resolve().with_name("launcher.py")


def is_custom_tool(row: dict) -> bool:
    return (row.get("builtin_id") or "").startswith(PREFIX)


def source_path(server_id: str) -> Path:
    return account_path("custom_tools") / f"{server_id}.py"


def write_source(server_id: str, code: str) -> None:
    path = source_path(server_id)
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(code, encoding = "utf-8")


def resolve_server(row: dict) -> dict:
    command = [sys.executable, str(_LAUNCHER), str(source_path(row["id"]))]
    return {**row, "url": join_stdio_command(command), "use_oauth": False}


def parse_tools(code: str) -> tuple[list[dict], str | None]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [], f"Line {exc.lineno}: {exc.msg}"
    tools = {}
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name.startswith("_"):
            continue
        args = node.args
        positional = args.posonlyargs + args.args
        first_default = len(positional) - len(args.defaults)
        params = [(arg, index < first_default) for index, arg in enumerate(positional)]
        params += [(arg, default is None) for arg, default in zip(args.kwonlyargs, args.kw_defaults)]
        tools[node.name] = {
            "name": node.name,
            "description": ast.get_docstring(node) or "",
            "params": [
                {
                    "name": arg.arg,
                    "type": ast.unparse(arg.annotation) if arg.annotation else None,
                    "required": required,
                }
                for arg, required in params
            ],
        }
    return list(tools.values()), None


def load_error(row: dict) -> str | None:
    from mcp.client.stdio import get_default_environment

    try:
        result = subprocess.run(
            [sys.executable, str(_LAUNCHER), "--check", str(source_path(row["id"]))],
            capture_output = True,
            text = True,
            timeout = 30,
            env = {**get_default_environment(), **(parse_server_headers(row) or {})},
        )
    except subprocess.TimeoutExpired:
        return "Loading the tool took longer than 30 seconds."
    lines = [line for line in result.stderr.splitlines() if line.strip()]
    return lines[-1] if result.returncode and lines else None
