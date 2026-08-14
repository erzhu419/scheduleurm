"""Runtime-bound running task snapshot and update wrappers."""

from __future__ import annotations

from typing import Any, Mapping, Optional


def _ns(namespace: Mapping[str, Any], name: str) -> Any:
    return namespace[name]


def build_running_update_runtime_exports(namespace: dict[str, Any]) -> dict[str, Any]:
    def _running_lifecycle_deps():
        return _ns(namespace, "_build_running_lifecycle_deps")(namespace)

    def _running_probe_snapshot_deps():
        return _ns(namespace, "_build_running_probe_snapshot_deps")(namespace)

    def _running_probe_snapshot_outside_lock(purpose: str = "running-probe"):
        return _ns(namespace, "_running_probe_snapshot_outside_lock_impl")(
            purpose,
            deps=_ns(namespace, "_running_probe_snapshot_deps")(),
        )

    def _terminal_diagnostic_fingerprint(task: dict) -> tuple:
        return _ns(namespace, "_terminal_diagnostic_fingerprint_impl")(task)

    def _terminal_diagnostic_matches(task: dict, payload: Optional[dict]) -> bool:
        return _ns(namespace, "_terminal_diagnostic_matches_impl")(task, payload)

    def _terminal_diagnostics_snapshot_deps():
        return _ns(namespace, "_build_terminal_diagnostics_snapshot_deps")(namespace)

    def _terminal_diagnostics_snapshot_outside_lock(
        probe_results: Optional[dict],
        probed_task_ids: Optional[set],
        purpose: str = "terminal-diagnostics",
    ) -> dict:
        return _ns(namespace, "_terminal_diagnostics_snapshot_outside_lock_impl")(
            probe_results,
            probed_task_ids,
            purpose,
            deps=_ns(namespace, "_terminal_diagnostics_snapshot_deps")(),
        )

    def _batch_check_running_deps():
        return _ns(namespace, "_build_batch_check_running_deps")(namespace)

    def _batch_check_running(
        state,
        probe_results: Optional[dict] = None,
        probed_task_ids: Optional[set] = None,
        terminal_diagnostics: Optional[dict] = None,
    ):
        return _ns(namespace, "_batch_check_running_impl")(
            state,
            probe_results=probe_results,
            probed_task_ids=probed_task_ids,
            terminal_diagnostics=terminal_diagnostics,
            deps=_ns(namespace, "_batch_check_running_deps")(),
        )

    def _update_running_tasks_deps():
        return _ns(namespace, "_build_update_running_tasks_deps")(namespace)

    def update_running_tasks(
        state,
        probe_results: Optional[dict] = None,
        probed_task_ids: Optional[set] = None,
        eta_tail_outputs: Optional[dict] = None,
        eta_probed_task_ids: Optional[set] = None,
        terminal_diagnostics: Optional[dict] = None,
        defer_eta_refresh: bool = False,
    ):
        return _ns(namespace, "_update_running_tasks_impl")(
            state,
            probe_results=probe_results,
            probed_task_ids=probed_task_ids,
            eta_tail_outputs=eta_tail_outputs,
            eta_probed_task_ids=eta_probed_task_ids,
            terminal_diagnostics=terminal_diagnostics,
            defer_eta_refresh=defer_eta_refresh,
            deps=_ns(namespace, "_update_running_tasks_deps")(),
        )

    return {
        "_running_lifecycle_deps": _running_lifecycle_deps,
        "_running_probe_snapshot_deps": _running_probe_snapshot_deps,
        "_running_probe_snapshot_outside_lock": _running_probe_snapshot_outside_lock,
        "_terminal_diagnostic_fingerprint": _terminal_diagnostic_fingerprint,
        "_terminal_diagnostic_matches": _terminal_diagnostic_matches,
        "_terminal_diagnostics_snapshot_deps": _terminal_diagnostics_snapshot_deps,
        "_terminal_diagnostics_snapshot_outside_lock": _terminal_diagnostics_snapshot_outside_lock,
        "_batch_check_running_deps": _batch_check_running_deps,
        "_batch_check_running": _batch_check_running,
        "_update_running_tasks_deps": _update_running_tasks_deps,
        "update_running_tasks": update_running_tasks,
    }
