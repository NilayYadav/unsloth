# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import sys
from pathlib import Path

_BACKEND = str(Path(__file__).resolve().parent)
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from core.inference.tools import _bash_exec, _python_exec

PY = "import sys, time\nprint('progress')\nsys.stdout.flush()\ntime.sleep(30)\n"
SH = "echo progress; sleep 30"

r_py = _python_exec(PY, timeout = 1)
r_sh = _bash_exec(SH, timeout = 1)

print("PROBE python.repr      = " + repr(r_py))
print("PROBE bash.repr        = " + repr(r_sh))
print("PROBE python.kept      = %s" % ("progress" in r_py))
print("PROBE bash.kept        = %s" % ("progress" in r_sh))
print("PROBE python.endswith  = %s" % r_py.endswith("Execution timed out after 1 seconds."))
print("PROBE bash.endswith    = %s" % r_sh.endswith("Execution timed out after 1 seconds."))
print("PROBE python.len       = %d" % len(r_py))
print("PROBE bash.len         = %d" % len(r_sh))
