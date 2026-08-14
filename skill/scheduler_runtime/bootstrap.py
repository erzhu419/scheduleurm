"""Ordered runtime assembly for the scheduler facade."""

from __future__ import annotations

from typing import Any, Callable

from scheduler_algorithm.runtime import build_algorithm_runtime_exports
from scheduler_claim.wrappers import build_claims_runtime_exports
from scheduler_cpu.runtime import build_cpu_runtime_exports
from scheduler_cpu.submit_runtime import build_cpu_submit_runtime_exports
from scheduler_runtime.data_motion import build_data_motion_runtime_exports
from scheduler_dispatch.runtime import build_dispatch_runtime_exports
from scheduler_eta.runtime import build_eta_runtime_exports
from scheduler_failure.runtime import build_failure_runtime_exports
from scheduler_launch.runtime import build_launch_runtime_exports
from scheduler_placement_engine.runtime import build_placement_runtime_exports
from scheduler_probe.runtime import build_probe_runtime_exports
from scheduler_recovery.runtime import build_recovery_runtime_exports
from scheduler_result.history_runtime import build_results_history_runtime_exports
from scheduler_running.runtime import build_running_runtime_exports
from .history_runtime import build_runtime_history_runtime_exports
from scheduler_state.runtime import build_state_runtime_exports
from scheduler_stdlib_runtime import build_stdlib_runtime_exports
from scheduler_submit.runtime import build_submit_runtime_exports
from scheduler_watch.runtime import build_watch_runtime_exports


RuntimeBuilder = Callable[[dict[str, Any]], dict[str, Any]]


RUNTIME_BUILDERS: tuple[RuntimeBuilder, ...] = (
    build_stdlib_runtime_exports,
    build_algorithm_runtime_exports,
    build_state_runtime_exports,
    build_data_motion_runtime_exports,
    build_cpu_runtime_exports,
    build_running_runtime_exports,
    build_eta_runtime_exports,
    build_failure_runtime_exports,
    build_placement_runtime_exports,
    build_claims_runtime_exports,
    build_results_history_runtime_exports,
    build_probe_runtime_exports,
    build_launch_runtime_exports,
    build_submit_runtime_exports,
    build_runtime_history_runtime_exports,
    build_cpu_submit_runtime_exports,
    build_watch_runtime_exports,
    build_dispatch_runtime_exports,
    build_recovery_runtime_exports,
)


def build_scheduler_runtime_exports(namespace: dict[str, Any]) -> dict[str, Any]:
    """Install all runtime compatibility exports into ``namespace`` once."""
    exports: dict[str, Any] = {}
    for builder in RUNTIME_BUILDERS:
        new_exports = builder(namespace)
        exports.update(new_exports)
        namespace.update(new_exports)
    return exports
