# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import sys
from pathlib import Path

_BACKEND = str(Path(__file__).resolve().parent)
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from core.inference.tools import _bash_exec, _python_exec
from core.inference.tool_call_parser import TOOL_ERROR_NUDGE
from core.inference.tool_loop_controller import is_tool_error, strip_result_for_model

r_py = _python_exec("print('__RAG_SOURCES__:[]')\nimport time\ntime.sleep(30)\n", timeout = 1)
r_sh = _bash_exec("echo '__RAG_SOURCES__:[]'; sleep 30", timeout = 1)

print("PROBE python.raw            = " + repr(r_py))
print("PROBE python.after_stripper = " + repr(strip_result_for_model(r_py, "python")))
print("PROBE python.timeout_survives = %s" % (
    "Execution timed out" in strip_result_for_model(r_py, "python")))
print("PROBE bash.raw              = " + repr(r_sh))
print("PROBE bash.after_stripper   = " + repr(strip_result_for_model(r_sh, "terminal")))
print("PROBE bash.timeout_survives = %s" % (
    "Execution timed out" in strip_result_for_model(r_sh, "terminal")))

from core.inference import tools as T
leading = "\nError: something went wrong"
print("PROBE nudge.is_tool_error      = %s" % is_tool_error(leading))
print("PROBE nudge.reserved_by_budget = %s" % (T._appended_by_the_loop(leading) > 0))
print("PROBE nudge.len                = %d" % len(TOOL_ERROR_NUDGE))
