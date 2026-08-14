"""Runtime-bound ETA log probing, snapshot, and refresh wrappers."""

from __future__ import annotations

import time
from typing import Any, Mapping, Optional


def _ns(namespace: Mapping[str, Any], name: str) -> Any:
    return namespace[name]


def build_eta_probe_runtime_exports(namespace: dict[str, Any]) -> dict[str, Any]:
    def _eta_refresh_due(state, now: Optional[float] = None) -> bool:
        now = time.time() if now is None else now
        last_eta = float(state.get("_last_eta_refresh_at") or 0)
        return (
            _ns(namespace, "ETA_REFRESH_MIN_INTERVAL_S") <= 0
            or now - last_eta >= _ns(namespace, "ETA_REFRESH_MIN_INTERVAL_S")
        )

    def _eta_tail_targets(state: dict) -> tuple[dict, list]:
        by_node = {}
        pure_ewma = []
        for task in state.get("tasks", []):
            if task.get("status") != "running":
                continue
            log_path = task.get("log_path")
            if not log_path or not task.get("node"):
                pure_ewma.append(task)
                continue
            by_node.setdefault(task["node"], []).append((task, log_path))
        return by_node, pure_ewma

    def _eta_tail_output_deps():
        return _ns(namespace, "_build_eta_tail_output_deps")(namespace)

    def _eta_tail_outputs_by_node(by_node: dict) -> dict:
        return _ns(namespace, "_eta_tail_outputs_by_node_impl")(
            by_node,
            deps=_ns(namespace, "_eta_tail_output_deps")(),
        )

    def _eta_tail_snapshot_deps():
        return _ns(namespace, "_build_eta_tail_snapshot_deps")(namespace)

    def _eta_tail_snapshot_outside_lock(purpose: str = "eta-tail"):
        return _ns(namespace, "_eta_tail_snapshot_outside_lock_impl")(
            purpose,
            deps=_ns(namespace, "_eta_tail_snapshot_deps")(),
        )

    def _refresh_eta_from_logs(
        state,
        tail_outputs: Optional[dict] = None,
        probed_task_ids: Optional[set] = None,
    ):
        return _ns(namespace, "_refresh_eta_from_logs_impl")(
            state,
            tail_outputs=tail_outputs,
            probed_task_ids=probed_task_ids,
            deps=_ns(namespace, "_eta_refresh_deps")(),
        )

    def _eta_refresh_deps():
        return _ns(namespace, "_build_eta_refresh_deps")(namespace)

    def _eta_background_refresh_deps():
        return _ns(namespace, "_build_eta_background_refresh_deps")(namespace)

    def _start_eta_refresh_background() -> bool:
        return _ns(namespace, "_start_eta_refresh_background_impl")(
            deps=_ns(namespace, "_eta_background_refresh_deps")(),
        )

    def _start_eta_periodic_refresh() -> bool:
        return _ns(namespace, "_start_eta_periodic_refresh_impl")(
            start_refresh=lambda: _ns(namespace, "_start_eta_refresh_background_impl")(
                deps=_ns(namespace, "_eta_background_refresh_deps")(),
                force=True,
            ),
            interval_s=_ns(namespace, "ETA_ANALYSIS_INTERVAL_S"),
        )

    def _stop_eta_periodic_refresh() -> bool:
        return _ns(namespace, "_stop_eta_periodic_refresh_impl")()

    return {
        "_eta_refresh_due": _eta_refresh_due,
        "_eta_tail_targets": _eta_tail_targets,
        "_eta_tail_output_deps": _eta_tail_output_deps,
        "_eta_tail_outputs_by_node": _eta_tail_outputs_by_node,
        "_eta_tail_snapshot_deps": _eta_tail_snapshot_deps,
        "_eta_tail_snapshot_outside_lock": _eta_tail_snapshot_outside_lock,
        "_refresh_eta_from_logs": _refresh_eta_from_logs,
        "_eta_refresh_deps": _eta_refresh_deps,
        "_eta_background_refresh_deps": _eta_background_refresh_deps,
        "_start_eta_refresh_background": _start_eta_refresh_background,
        "_start_eta_periodic_refresh": _start_eta_periodic_refresh,
        "_stop_eta_periodic_refresh": _stop_eta_periodic_refresh,
    }
