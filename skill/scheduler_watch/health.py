"""Health decisions for the systemd watcher keepalive."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class WatcherHealthDecision:
    action: str
    reason: str
    age_s: float = 0.0
    phase: str = "unknown"


def watcher_last_update(state: Mapping[str, Any]) -> float:
    values = []
    for key in (
        "last_progress_ts",
        "last_resource_log_ts",
        "last_heartbeat_ts",
        "started_at",
    ):
        try:
            values.append(float(state.get(key) or 0.0))
        except (TypeError, ValueError):
            continue
    return max(values, default=0.0)


def watcher_phase_stale_after(phase: str, base_stale_s: float) -> float:
    base = max(60.0, float(base_stale_s))
    phase = str(phase or "unknown")
    if phase in {"launching", "staging"}:
        return max(base, 3600.0)
    if phase in {"notifications_and_sync", "claim_maintenance"}:
        return max(base, 1800.0)
    return base


def decide_watcher_health(
    *,
    service_active: bool,
    main_pid: int,
    process_alive: bool,
    process_age_s: float,
    state: Any,
    now: float,
    stale_after_s: float = 600.0,
    startup_grace_s: float = 180.0,
) -> WatcherHealthDecision:
    if not service_active or main_pid <= 0 or not process_alive:
        return WatcherHealthDecision("start", "watcher service has no live main process")
    if process_age_s < startup_grace_s:
        return WatcherHealthDecision("none", "watcher is inside startup grace")
    if not isinstance(state, Mapping):
        return WatcherHealthDecision(
            "restart",
            "watcher state is missing or invalid after startup grace",
            process_age_s,
        )
    try:
        state_pid = int(state.get("pid") or 0)
    except (TypeError, ValueError):
        state_pid = 0
    if state_pid != main_pid:
        return WatcherHealthDecision(
            "restart",
            f"watcher state pid {state_pid or '?'} does not match service pid {main_pid}",
            process_age_s,
            str(state.get("phase") or "unknown"),
        )
    phase = str(state.get("phase") or "unknown")
    last_update = watcher_last_update(state)
    age_s = max(0.0, float(now) - last_update) if last_update else process_age_s
    limit_s = watcher_phase_stale_after(phase, stale_after_s)
    if age_s > limit_s:
        return WatcherHealthDecision(
            "restart",
            f"no watcher progress for {age_s:.0f}s (limit {limit_s:.0f}s)",
            age_s,
            phase,
        )
    return WatcherHealthDecision("none", "watcher progress is current", age_s, phase)
