"""Runtime-bound ETA field, formatting, and elapsed-time wrappers."""

from __future__ import annotations

import time
from typing import Any, Mapping


def _ns(namespace: Mapping[str, Any], name: str) -> Any:
    return namespace[name]


def build_eta_fields_runtime_exports(namespace: dict[str, Any]) -> dict[str, Any]:
    def _eta_source_base(source: object) -> str:
        return _ns(namespace, "_eta_source_base_impl")(source)

    def _is_live_remaining_eta_source(source: object) -> bool:
        return _ns(namespace, "_is_live_remaining_eta_source_impl")(source)

    def _is_live_runtime_projection_source(source: object) -> bool:
        return _ns(namespace, "_is_live_runtime_projection_source_impl")(source)

    def _clear_live_eta_fields(task: dict, clear_runtime_projection: bool = False) -> bool:
        return _ns(namespace, "_clear_live_eta_fields_impl")(
            task,
            clear_runtime_projection=clear_runtime_projection,
        )

    def _queued_has_stale_live_eta(task: dict) -> bool:
        return _ns(namespace, "_queued_has_stale_live_eta_impl")(task)

    def _eta_confidence_for_source(source: str) -> str:
        return _ns(namespace, "_eta_confidence_for_source_impl")(source)

    def _fmt_eta_seconds(seconds: int) -> str:
        return _ns(namespace, "_fmt_eta_seconds_impl")(seconds)

    def _eta_source_tag(source: object) -> str:
        return _ns(namespace, "_eta_source_tag_impl")(source)

    def _format_task_eta(task: dict) -> str:
        return _ns(namespace, "_format_task_eta_impl")(task)

    def _history_fallback_eta_seconds(total_s: int, elapsed_s: float) -> tuple[int, bool]:
        return _ns(namespace, "_history_fallback_eta_seconds_impl")(
            total_s,
            elapsed_s,
            overrun_floor_s=_ns(namespace, "ETA_HISTORY_OVERRUN_FLOOR_S"),
            overrun_fraction=_ns(namespace, "ETA_HISTORY_OVERRUN_FRACTION"),
            overrun_max_s=_ns(namespace, "ETA_HISTORY_OVERRUN_MAX_S"),
        )

    def _effective_elapsed_s(task: dict) -> float:
        return _ns(namespace, "_effective_elapsed_s_impl")(task, now=time.time)

    return {
        "_eta_source_base": _eta_source_base,
        "_is_live_remaining_eta_source": _is_live_remaining_eta_source,
        "_is_live_runtime_projection_source": _is_live_runtime_projection_source,
        "_clear_live_eta_fields": _clear_live_eta_fields,
        "_queued_has_stale_live_eta": _queued_has_stale_live_eta,
        "_eta_confidence_for_source": _eta_confidence_for_source,
        "_fmt_eta_seconds": _fmt_eta_seconds,
        "_eta_source_tag": _eta_source_tag,
        "_format_task_eta": _format_task_eta,
        "_history_fallback_eta_seconds": _history_fallback_eta_seconds,
        "_effective_elapsed_s": _effective_elapsed_s,
    }
