"""Runtime binding for ETA analysis logging."""

from __future__ import annotations

from typing import Any, Mapping

from .analysis_log import record_eta_analysis_tasks


def _ns(namespace: Mapping[str, Any], name: str) -> Any:
    return namespace[name]


def build_eta_analysis_runtime_exports(namespace: dict[str, Any]) -> dict[str, Any]:
    def _record_eta_analysis_tasks(
        tasks,
        *,
        include_periodic: bool = False,
        force_periodic: bool = False,
    ):
        return record_eta_analysis_tasks(
            tasks,
            log_dir=_ns(namespace, "ETA_ANALYSIS_DIR"),
            interval_s=_ns(namespace, "ETA_ANALYSIS_INTERVAL_S"),
            include_periodic=include_periodic,
            force_periodic=force_periodic,
            now=_ns(namespace, "time").time(),
        )

    return {"_record_eta_analysis_tasks": _record_eta_analysis_tasks}
