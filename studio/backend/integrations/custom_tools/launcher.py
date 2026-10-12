# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import importlib.util
import inspect
import sys
from pathlib import Path

from fastmcp import FastMCP


def main() -> int:
    path = Path(sys.argv[-1])
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    server = FastMCP(path.stem)
    for name, function in vars(module).items():
        if (
            not name.startswith("_")
            and inspect.isfunction(function)
            and function.__module__ == module.__name__
        ):
            server.tool(function)
    if sys.argv[1] != "--check":
        server.run(show_banner = False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
