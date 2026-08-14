"""Running-task probe, terminal-diagnostic, and ETA-tail snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

try:
    from scheduler_eta.tail_snapshot import (
        EtaTailOutputDeps,
        EtaTailSnapshotDeps,
        eta_tail_outputs_by_node,
        eta_tail_snapshot_outside_lock,
    )
    from scheduler_failure.terminal_snapshot import (
        TerminalDiagnosticsSnapshotDeps,
        terminal_diagnostic_fingerprint,
        terminal_diagnostic_matches,
        terminal_diagnostics_snapshot_outside_lock,
    )
except ModuleNotFoundError:
    from ..scheduler_eta.tail_snapshot import (
        EtaTailOutputDeps,
        EtaTailSnapshotDeps,
        eta_tail_outputs_by_node,
        eta_tail_snapshot_outside_lock,
    )
    from ..scheduler_failure.terminal_snapshot import (
        TerminalDiagnosticsSnapshotDeps,
        terminal_diagnostic_fingerprint,
        terminal_diagnostic_matches,
        terminal_diagnostics_snapshot_outside_lock,
    )


@dataclass(frozen=True)
class RunningProbeSnapshotDeps:
    state_lock: Callable[..., Any]
    load_state: Callable[[], dict]
    running_probe_due: Callable[[dict], bool]
    backend_batch_probe: Callable[[dict], dict]
    notify: Callable[..., Any]


@dataclass(frozen=True)
class BatchCheckRunningDeps:
    backend_batch_probe: Callable[[dict], dict]
    running_lifecycle_deps: Callable[[], Any]
    handle_unknown_probe_result: Callable[..., Any]
    clear_probe_unknown: Callable[[dict], Optional[dict]]
    handle_dead_probe_result: Callable[..., Any]
    apply_alive_probe_result: Callable[..., Any]


@dataclass(frozen=True)
class UpdateRunningTasksDeps:
    now: Callable[[], float]
    running_probe_due: Callable[..., bool]
    batch_check_running: Callable[..., Any]
    eta_refresh_due: Callable[..., bool]
    refresh_eta_from_logs: Callable[..., Any]


def running_probe_snapshot_outside_lock(
    purpose: str = "running-probe",
    *,
    deps: RunningProbeSnapshotDeps,
) -> tuple[dict, set]:
    """Collect backend liveness data without holding the global writer lock."""
    try:
        with deps.state_lock(shared=True, purpose=f"{purpose}:snapshot"):
            snapshot = deps.load_state()
            if not deps.running_probe_due(snapshot):
                pending = {
                    str(task.get("id") or ""): {
                        "state": "dead",
                        "alive_pids": [],
                        "vram_mb": 0,
                        "ram_mb": 0,
                        "pcpu": 0.0,
                        "backend_state": "TERMINAL_DIAG_PENDING",
                        "terminal_reason": (
                            "retrying terminal diagnosis after a previously confirmed process exit"
                        ),
                        "terminal_diagnosis_deferred": True,
                        "terminal_retry_only": True,
                    }
                    for task in snapshot.get("tasks", [])
                    if task.get("status") == "running"
                    and task.get("id")
                    and task.get("terminal_transition_deferred")
                }
                return pending, set(pending)
            running_ids = {
                str(t.get("id") or "")
                for t in snapshot.get("tasks", [])
                if t.get("status") == "running" and t.get("id")
            }
            if not running_ids:
                return {}, running_ids
    except Exception as e:
        deps.notify(
            "running_probe_snapshot_error",
            {"purpose": purpose, "error": str(e)[:200]},
            feishu_enabled=False,
        )
        return {}, set()
    try:
        return deps.backend_batch_probe(snapshot), running_ids
    except Exception as e:
        deps.notify(
            "running_probe_error_outer",
            {"purpose": purpose, "error": str(e)[:200]},
            feishu_enabled=False,
        )
        return {}, running_ids


def batch_check_running(
    state: dict,
    probe_results: Optional[dict] = None,
    probed_task_ids: Optional[set] = None,
    terminal_diagnostics: Optional[dict] = None,
    *,
    deps: BatchCheckRunningDeps,
) -> None:
    """Apply backend probe results to all running tasks in a state snapshot."""
    lifecycle_deps = deps.running_lifecycle_deps()

    if probe_results is None:
        probe_results = deps.backend_batch_probe(state)
        probed_task_ids = {
            str(t.get("id") or "")
            for t in state.get("tasks", [])
            if t.get("status") == "running" and t.get("id")
        }
    elif probed_task_ids is not None:
        probed_task_ids = {str(x) for x in probed_task_ids if str(x)}
    if not probe_results:
        return

    for task in state["tasks"]:
        if task["status"] != "running":
            continue
        if probed_task_ids is not None and str(task.get("id") or "") not in probed_task_ids:
            continue
        res = probe_results.get(task["id"])
        if res is None or res["state"] == "unknown":
            deps.handle_unknown_probe_result(
                task,
                state,
                res,
                terminal_diagnostics,
                deps=lifecycle_deps,
            )
            continue
        reconnect_sync = deps.clear_probe_unknown(task)
        if res["state"] == "dead":
            deps.handle_dead_probe_result(
                task,
                state,
                res,
                reconnect_sync,
                terminal_diagnostics,
                deps=lifecycle_deps,
            )
            continue
        deps.apply_alive_probe_result(task, res, reconnect_sync, deps=lifecycle_deps)


def update_running_tasks(
    state: dict,
    probe_results: Optional[dict] = None,
    probed_task_ids: Optional[set] = None,
    eta_tail_outputs: Optional[dict] = None,
    eta_probed_task_ids: Optional[set] = None,
    terminal_diagnostics: Optional[dict] = None,
    defer_eta_refresh: bool = False,
    *,
    deps: UpdateRunningTasksDeps,
) -> None:
    """Run the liveness pass and ETA-refresh pass for running tasks."""
    now = deps.now()
    # A snapshot that was due when collection started remains authoritative at
    # commit time. Another status/watch process may refresh the global probe
    # timestamp while remote terminal diagnostics are still running; dropping
    # this completed snapshot would leave dead tasks stuck until another full
    # probe cycle. Task status and terminal fingerprints still reject stale
    # per-task results in batch_check_running.
    precomputed_probe_supplied = (
        probe_results is not None or probed_task_ids is not None
    )
    precomputed_probe_ready = bool(probe_results) or bool(probed_task_ids)
    # An explicitly supplied empty snapshot means the outside-lock preflight
    # decided that no probe was due. Do not re-evaluate the timer after slower
    # preflight work and fall back to backend_batch_probe while the caller holds
    # the state writer lock.
    if precomputed_probe_ready or (
        not precomputed_probe_supplied
        and deps.running_probe_due(state, now)
    ):
        deps.batch_check_running(
            state,
            probe_results=probe_results,
            probed_task_ids=probed_task_ids,
            terminal_diagnostics=terminal_diagnostics,
        )
        retry_only = bool(probe_results) and all(
            isinstance(result, dict) and result.get("terminal_retry_only")
            for result in probe_results.values()
        )
        if not retry_only:
            state["_last_running_probe_at"] = deps.now()
    if not defer_eta_refresh and deps.eta_refresh_due(state, now):
        deps.refresh_eta_from_logs(
            state,
            tail_outputs=eta_tail_outputs,
            probed_task_ids=eta_probed_task_ids,
        )
        state["_last_eta_refresh_at"] = deps.now()


def effective_elapsed_s(task: dict, *, now: Callable[[], float]) -> float:
    """Return compute elapsed seconds, preserving legacy external pending semantics."""
    started_at = task.get("started_at")
    if not started_at:
        return 0.0
    if task.get("slurm_job_id"):
        actual = task.get("actual_started_at")
        if actual:
            return max(0.0, now() - actual)
        return 0.0
    return max(0.0, now() - started_at)
