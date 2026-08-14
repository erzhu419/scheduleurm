"""Single-flight background ETA refresh for the watcher.

ETA log collection is observational and may spend tens of seconds traversing
SSH jump hosts.  It must not sit in front of dispatch or parse a large runtime
history while holding the scheduler writer lock.
"""

from __future__ import annotations

import copy
import threading
import time as _time
from dataclasses import dataclass
from typing import Any, Callable

from .state import ETA_AUDIT_FIELDS, RUNTIME_PROJECTION_FIELDS

try:
    from ..scheduler_watch.shutdown import WatcherShutdownRequested
except ImportError:
    from scheduler_watch.shutdown import WatcherShutdownRequested


@dataclass(frozen=True)
class EtaBackgroundRefreshDeps:
    state_lock: Callable[..., Any]
    load_state: Callable[[], dict]
    save_state: Callable[[dict], Any]
    eta_refresh_due: Callable[[dict], bool]
    eta_tail_targets: Callable[[dict], tuple[dict, list]]
    eta_tail_outputs_by_node: Callable[[dict], dict]
    refresh_eta_from_logs: Callable[..., Any]
    notify: Callable[..., Any]
    now: Callable[[], float] = _time.time
    record_eta_analysis_tasks: Callable[..., Any] = lambda *_args, **_kwargs: {}


_ETA_FIELDS = tuple(dict.fromkeys((*ETA_AUDIT_FIELDS, *RUNTIME_PROJECTION_FIELDS)))
_BACKGROUND_LOCK = threading.Lock()
_BACKGROUND_THREAD: threading.Thread | None = None
_PERIODIC_LOCK = threading.Lock()
_PERIODIC_THREAD: threading.Thread | None = None
_PERIODIC_STOP = threading.Event()


def _notify(deps: EtaBackgroundRefreshDeps, event: str, payload: dict) -> None:
    try:
        deps.notify(event, payload, feishu_enabled=False)
    except Exception:
        pass


def _same_running_attempt(source: dict, target: dict) -> bool:
    if source.get("status") != "running" or target.get("status") != "running":
        return False
    return (
        source.get("started_at") == target.get("started_at")
        and source.get("node") == target.get("node")
    )


def _merge_eta_fields(latest: dict, parsed: dict, task_ids: set[str]) -> int:
    parsed_by_id = {
        str(task.get("id") or ""): task
        for task in parsed.get("tasks", [])
        if task.get("id")
    }
    changed_tasks = 0
    for target in latest.get("tasks", []):
        task_id = str(target.get("id") or "")
        if task_id not in task_ids:
            continue
        source = parsed_by_id.get(task_id)
        if source is None or not _same_running_attempt(source, target):
            continue
        changed = False
        for key in _ETA_FIELDS:
            if key in source:
                value = copy.deepcopy(source[key])
                if target.get(key) != value or key not in target:
                    target[key] = value
                    changed = True
            elif key in target:
                target.pop(key, None)
                changed = True
        if changed:
            changed_tasks += 1
    return changed_tasks


def refresh_eta_snapshot(*, deps: EtaBackgroundRefreshDeps, force: bool = False) -> dict:
    """Collect, parse, and merge one ETA snapshot without blocking dispatch."""
    started_at = deps.now()
    try:
        with deps.state_lock(shared=True, purpose="watch:eta-background-snapshot"):
            snapshot = deps.load_state()
            if not force and not deps.eta_refresh_due(snapshot):
                return {"status": "not_due", "updated": 0}
            by_node, _pure_ewma = deps.eta_tail_targets(snapshot)
            task_ids = {
                str(task.get("id") or "")
                for task in snapshot.get("tasks", [])
                if task.get("status") == "running" and task.get("id")
            }
        if not task_ids:
            return {"status": "no_running_tasks", "updated": 0}

        probe_started_at = deps.now()
        tail_outputs = deps.eta_tail_outputs_by_node(by_node)
        probe_s = max(0.0, deps.now() - probe_started_at)

        parse_started_at = deps.now()
        deps.refresh_eta_from_logs(
            snapshot,
            tail_outputs=tail_outputs,
            probed_task_ids=task_ids,
        )
        parse_s = max(0.0, deps.now() - parse_started_at)

        commit_started_at = deps.now()
        with deps.state_lock(purpose="watch:eta-background-commit"):
            latest = deps.load_state()
            updated = _merge_eta_fields(latest, snapshot, task_ids)
            latest["_last_eta_refresh_at"] = deps.now()
            deps.save_state(latest)
            analysis_tasks = [
                dict(task)
                for task in latest.get("tasks", [])
                if task.get("status") == "running"
                and str(task.get("id") or "") in task_ids
            ]
        commit_s = max(0.0, deps.now() - commit_started_at)
        try:
            analysis = deps.record_eta_analysis_tasks(
                analysis_tasks,
                include_periodic=True,
                force_periodic=force,
            ) or {}
        except Exception as exc:
            analysis = {"records": 0, "errors": [{"error": str(exc)[:200]}]}
        payload = {
            "status": "ok",
            "forced": bool(force),
            "refresh_started_at": round(started_at, 3),
            "updated": updated,
            "probed": len(task_ids),
            "nodes": len(by_node),
            "probe_s": round(probe_s, 3),
            "parse_s": round(parse_s, 3),
            "commit_s": round(commit_s, 3),
            "total_s": round(max(0.0, deps.now() - started_at), 3),
            "eta_analysis_records": int(analysis.get("records") or 0),
            "eta_analysis_errors": len(analysis.get("errors") or []),
        }
        _notify(deps, "eta_refresh_background_done", payload)
        return payload
    except Exception as exc:
        payload = {
            "status": "error",
            "updated": 0,
            "error": str(exc)[:200],
            "total_s": round(max(0.0, deps.now() - started_at), 3),
        }
        _notify(deps, "eta_refresh_background_error", payload)
        return payload


def _background_target(deps: EtaBackgroundRefreshDeps, force: bool) -> None:
    global _BACKGROUND_THREAD
    try:
        refresh_eta_snapshot(deps=deps, force=force)
    except WatcherShutdownRequested:
        # SIGTERM interrupts active SSH probes with this BaseException. Consume
        # it here so interpreter shutdown never races a traceback on stderr.
        pass
    finally:
        with _BACKGROUND_LOCK:
            if _BACKGROUND_THREAD is threading.current_thread():
                _BACKGROUND_THREAD = None


def start_eta_refresh_background(
    *,
    deps: EtaBackgroundRefreshDeps,
    force: bool = False,
) -> bool:
    """Start one daemon ETA worker; return False while another is active."""
    global _BACKGROUND_THREAD
    with _BACKGROUND_LOCK:
        if _BACKGROUND_THREAD is not None and _BACKGROUND_THREAD.is_alive():
            return False
        thread = threading.Thread(
            target=_background_target,
            args=(deps, force),
            name="scheduleurm-eta-refresh",
            daemon=True,
        )
        _BACKGROUND_THREAD = thread
        thread.start()
        return True


def _periodic_target(start_refresh: Callable[[], Any], interval_s: float) -> None:
    global _PERIODIC_THREAD
    deadline = _time.monotonic()
    try:
        while not _PERIODIC_STOP.is_set():
            delay = max(0.0, deadline - _time.monotonic())
            if _PERIODIC_STOP.wait(delay):
                break
            try:
                start_refresh()
            except Exception:
                pass
            deadline += interval_s
            now = _time.monotonic()
            if deadline < now:
                missed = int((now - deadline) // interval_s) + 1
                deadline += missed * interval_s
    finally:
        with _PERIODIC_LOCK:
            if _PERIODIC_THREAD is threading.current_thread():
                _PERIODIC_THREAD = None


def start_eta_periodic_refresh(
    *,
    start_refresh: Callable[[], Any],
    interval_s: float = 60.0,
) -> bool:
    """Start one fixed-rate ETA trigger independent of watcher cycle duration."""
    global _PERIODIC_THREAD
    interval_s = max(0.01, float(interval_s))
    with _PERIODIC_LOCK:
        if _PERIODIC_THREAD is not None and _PERIODIC_THREAD.is_alive():
            return False
        _PERIODIC_STOP.clear()
        thread = threading.Thread(
            target=_periodic_target,
            args=(start_refresh, interval_s),
            name="scheduleurm-eta-periodic",
            daemon=True,
        )
        _PERIODIC_THREAD = thread
        thread.start()
        return True


def stop_eta_periodic_refresh(*, timeout_s: float = 5.0) -> bool:
    """Stop the ETA trigger and wait briefly for both ETA worker threads."""
    with _PERIODIC_LOCK:
        periodic_thread = _PERIODIC_THREAD
        _PERIODIC_STOP.set()
    with _BACKGROUND_LOCK:
        background_thread = _BACKGROUND_THREAD
    threads = [
        thread
        for thread in (periodic_thread, background_thread)
        if thread is not None and thread is not threading.current_thread()
    ]
    if periodic_thread is None and background_thread is None:
        return False
    deadline = _time.monotonic() + max(0.0, float(timeout_s))
    for thread in threads:
        thread.join(timeout=max(0.0, deadline - _time.monotonic()))
    return all(not thread.is_alive() for thread in threads)
