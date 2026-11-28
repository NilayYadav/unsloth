# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A crashed pytest-xdist worker under ``--dist loadgroup`` / ``loadscope`` / ``loadfile`` must not hang the run.

pytest-xdist 3.8.0's ``LoadScopeScheduling.remove_node`` puts the crashed worker's whole workload back on the
queue: its finished units, and the test it crashed in. The replacement worker is then handed a finished unit, an
empty run request that never reports back, so the controller waits on it until the job is killed (gloo-ep-2 sat
from its first "node down" to the 2100 s cap). A worker crashes this way whenever pytest-timeout's thread method
fires (it ends the worker with ``os._exit``). ``--dist load`` reports the crashed test and requeues only what never
ran; this scheduler does the same per unit. Loaded from tests/conftest.py by ``pytest_xdist_make_scheduler``."""

from __future__ import annotations

from typing import Any, Optional

_DISTS = ("loadgroup", "loadscope", "loadfile")


def _crash_safe(base: type) -> type:
    class CrashSafeScheduling(base):
        def remove_node(self, node: Any) -> Optional[str]:
            workload = self.assigned_work.pop(node)
            crashitem = next(
                (nodeid for unit in workload.values() for nodeid, done in unit.items() if not done),
                None,
            )
            if crashitem is None:  # a clean shutdown
                return None
            for unit in workload.values():
                if crashitem in unit:
                    unit[crashitem] = True  # reported as crashed by the caller, never run again
            self.workqueue.update(
                (scope, unit) for scope, unit in workload.items() if not all(unit.values())
            )
            for other in self.assigned_work:
                self._reschedule(other)
            return crashitem

    CrashSafeScheduling.__name__ = CrashSafeScheduling.__qualname__ = f"CrashSafe{base.__name__}"
    return CrashSafeScheduling


def make_scheduler(config: Any, log: Any) -> Any:
    """The crash-safe scheduler for the grouped dist modes, else None (xdist's own choice)."""
    dist = config.getvalue("dist")
    if dist not in _DISTS:
        return None
    from xdist import scheduler

    base = {
        "loadgroup": scheduler.LoadGroupScheduling,
        "loadscope": scheduler.LoadScopeScheduling,
        "loadfile": scheduler.LoadFileScheduling,
    }[dist]
    return _crash_safe(base)(config, log)
