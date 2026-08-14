"""Runtime-bound running-probe unknown/offline state wrappers."""

from __future__ import annotations

import time
from typing import Any, Mapping, Optional


def _ns(namespace: Mapping[str, Any], name: str) -> Any:
    return namespace[name]


def build_running_probe_state_runtime_exports(namespace: dict[str, Any]) -> dict[str, Any]:
    def _mark_probe_unknown(task: dict, res: Optional[dict] = None) -> None:
        now = time.time()
        if not task.get("probe_unknown_since"):
            task["probe_unknown_since"] = now
        task["last_probe_unknown_at"] = now
        task["probe_unknown_count"] = int(task.get("probe_unknown_count") or 0) + 1
        err = (res or {}).get("error") or (res or {}).get("terminal_reason")
        task["last_probe_unknown_reason"] = str(err or "backend probe returned unknown")[:300]
        task["node_probe_state"] = "unknown"
        task["last_status_sync_at"] = now
        task["last_status_sync_status"] = "running"
        task["last_status_sync_reason"] = task["last_probe_unknown_reason"]

    def _clear_probe_unknown(task: dict, now: Optional[float] = None) -> Optional[dict]:
        since = task.get("probe_unknown_since")
        if not since:
            return None
        now = now or time.time()
        try:
            duration_s = max(0, int(now - float(since)))
        except Exception:
            duration_s = 0
        rec = {
            "synced_at": now,
            "duration_s": duration_s,
            "count": int(task.get("probe_unknown_count") or 0),
            "reason": task.get("last_probe_unknown_reason") or "backend probe returned unknown",
        }
        task["last_reconnected_at"] = now
        task["last_probe_unknown_duration_s"] = duration_s
        task["last_probe_unknown_count"] = rec["count"]
        task["last_probe_unknown_reason"] = rec["reason"]
        task["node_probe_state"] = "ok"
        task["last_status_sync_at"] = now
        task["last_status_sync_reason"] = f"reconnected after {duration_s}s unknown probe window"
        task.pop("probe_unknown_since", None)
        task.pop("last_probe_unknown_at", None)
        task.pop("probe_unknown_count", None)
        return rec

    def _annotate_diag_after_unknown(diag: dict, sync_rec: Optional[dict]) -> dict:
        if not sync_rec:
            return diag
        diag = dict(diag or {})
        duration_s = int(sync_rec.get("duration_s") or 0)
        count = int(sync_rec.get("count") or 0)
        note = (
            f"terminal state synced after remote probe was unknown for {duration_s}s "
            f"({count} probe cycle(s))"
        )
        reason = diag.get("reason") or ""
        diag["reason"] = f"{reason}; {note}" if reason else note
        diag["offline_sync_s"] = duration_s
        diag["offline_probe_unknown_count"] = count
        diag["offline_probe_unknown_reason"] = sync_rec.get("reason")
        return diag

    def _running_probe_due(state, now: Optional[float] = None) -> bool:
        now = time.time() if now is None else now
        last_probe = float(state.get("_last_running_probe_at") or 0)
        return (
            _ns(namespace, "RUNNING_PROBE_MIN_INTERVAL_S") <= 0
            or now - last_probe >= _ns(namespace, "RUNNING_PROBE_MIN_INTERVAL_S")
        )

    return {
        "_mark_probe_unknown": _mark_probe_unknown,
        "_clear_probe_unknown": _clear_probe_unknown,
        "_annotate_diag_after_unknown": _annotate_diag_after_unknown,
        "_running_probe_due": _running_probe_due,
    }
