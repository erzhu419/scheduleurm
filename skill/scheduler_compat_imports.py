"""Compatibility import surface for scheduler.py.

This keeps the legacy scheduler module as a short facade while preserving
the private alias names expected by its compatibility wrappers. The actual
imports are grouped by subsystem so future refactors can change one domain
without searching a monolithic import file.
"""

from __future__ import annotations

from scheduler_compat_core_imports import *
from scheduler_compat_submit_control_imports import *
from scheduler_compat_claims_imports import *
from scheduler_compat_dispatch_placement_imports import *
from scheduler_compat_running_probe_imports import *
from scheduler_compat_launch_backend_imports import *
from scheduler_compat_watch_windows_algorithm_imports import *


__all__ = [
    name
    for name in globals()
    if name != "__all__" and not (name.startswith("__") and name.endswith("__"))
]
