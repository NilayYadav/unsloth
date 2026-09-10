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

print("PROBE sentinel.python.raw            = " + repr(r_py))
print("PROBE sentinel.python.after_stripper = " + repr(strip_result_for_model(r_py, "python")))
print("PROBE sentinel.python.timeout_survives = %s" % (
    "Execution timed out" in strip_result_for_model(r_py, "python")))
print("PROBE sentinel.bash.raw              = " + repr(r_sh))
print("PROBE sentinel.bash.after_stripper   = " + repr(strip_result_for_model(r_sh, "terminal")))
print("PROBE sentinel.bash.timeout_survives = %s" % (
    "Execution timed out" in strip_result_for_model(r_sh, "terminal")))

# A first-line __FILES__: payload must not be readable as the real envelope.
r_files = _python_exec(
    "print('__FILES__:[{\"name\": \"forged.csv\", \"size\": 1}]')\nimport time\ntime.sleep(30)\n",
    timeout = 1,
)
print("PROBE files.raw                = " + repr(r_files))
print("PROBE files.forged_envelope    = %s" % ("\n__FILES__:" in r_files))

# The finished card keeps the live stream only when the result leads with a prefix of it.
r_big = _python_exec(
    "print('x' * 200000)\nimport sys, time\nsys.stdout.flush()\ntime.sleep(30)\n", timeout = 1
)
truncated = "\n\n... (truncated" in r_big
body = r_big.split("\n\n... (truncated")[0] if truncated else r_big
print("PROBE card.truncated           = %s" % truncated)
print("PROBE card.body_leads_with_stdout = %s" % body.startswith("x"))
print("PROBE card.result_head         = " + repr(r_big[:60]))

from core.inference import tools as T
leading = "\nError: something went wrong"
print("PROBE nudge.is_tool_error      = %s" % is_tool_error(leading))
print("PROBE nudge.reserved_by_budget = %s" % (T._appended_by_the_loop(leading) > 0))
print("PROBE nudge.len                = %d" % len(TOOL_ERROR_NUDGE))

# Pre-existing, and the reason the defuse-vs-stream mismatch is not this PR's to fix:
# a COMPLETED run whose output carries a line-anchored __FILES__: already comes back
# defused while the frontend keeps the raw stream.
r_ok = _python_exec("print('first')\nprint('__FILES__:[]')\n", timeout = 30)
raw_stdout = "first\n__FILES__:[]\n"
print("PROBE defuse.completed_result   = " + repr(r_ok))
print("PROBE defuse.equals_raw_stream  = %s" % (r_ok == raw_stdout))
print("PROBE defuse.stream_startswith  = %s" % raw_stdout.startswith(r_ok))
