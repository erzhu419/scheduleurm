"""Watcher state timing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class WatcherTickPlan:
    state: dict
    emit_resource_log: bool
    emit_heartbeat: bool
    dirty: bool


def initial_watcher_state(
    *,
    pid: int,
    started_at: float,
    interval: int,
    heartbeat: int,
    resource_log_interval: int,
) -> dict:
    return {
        "pid": pid,
        "started_at": started_at,
        "last_progress_ts": started_at,
        "phase": "starting",
        "phase_started_at": started_at,
        "last_heartbeat_ts": 0,
        "last_resource_log_ts": 0,
        "interval": interval,
        "heartbeat": heartbeat,
        "resource_log_interval": resource_log_interval,
    }


def live_watcher_pid(existing_state: Any, *, pid_alive: Callable[[Any], bool]) -> Any | None:
    """Return the pid from watcher state when that process is still alive."""
    if not isinstance(existing_state, dict):
        return None
    pid = existing_state.get("pid")
    if not pid:
        return None
    return pid if pid_alive(pid) else None


def plan_watcher_tick(
    existing_state: Any,
    *,
    now: float,
    pid: int,
    interval: int,
    heartbeat: int,
    resource_log_interval: int,
) -> WatcherTickPlan:
    """Apply watcher-state defaults and decide which periodic events are due."""
    state = existing_state if isinstance(existing_state, dict) else {}
    state.setdefault("pid", pid)
    state.setdefault("started_at", now)
    state.setdefault("last_progress_ts", state.get("started_at", now))
    state.setdefault("phase", "starting")
    state.setdefault("phase_started_at", state.get("started_at", now))
    state.setdefault("interval", interval)
    state.setdefault("heartbeat", heartbeat)
    state.setdefault("resource_log_interval", resource_log_interval)

    resource_interval = max(0, int(resource_log_interval or 0))
    emit_resource = bool(
        resource_interval
        and now - state.get("last_resource_log_ts", 0) >= resource_interval
    )
    emit_heartbeat = bool(now - state.get("last_heartbeat_ts", 0) >= heartbeat)
    if emit_resource:
        state["last_resource_log_ts"] = now
    if emit_heartbeat:
        state["last_heartbeat_ts"] = now
    return WatcherTickPlan(
        state=state,
        emit_resource_log=emit_resource,
        emit_heartbeat=emit_heartbeat,
        dirty=emit_resource or emit_heartbeat,
    )
