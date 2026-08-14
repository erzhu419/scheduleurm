"""Commit deferred launch results back into scheduler state."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class LaunchCommitDeps:
    state_lock: Callable[..., Any]
    load_state: Callable[[], dict]
    save_state: Callable[[dict], None]
    apply_launch_result_to_task: Callable[[dict, bool, str, list], bool]
    kill_task: Callable[[dict], Any]
    release_task_claims_and_intents: Callable[[dict], Any]


@dataclass(frozen=True)
class DeferredLaunchDeps:
    max_workers: int
    launch: Callable[..., tuple[bool, str]]
    launch_exec_slot_lock: Callable[[str], Any]
    execute_launch_intents: Callable[..., list]
    notify: Callable[..., Any]
    commit_deps: LaunchCommitDeps


def _launch_commit_identity(intent: dict, launched_task: dict) -> tuple[str, str]:
    tid = str(intent.get("task_id") or launched_task.get("id") or "")
    token = str((intent.get("task") or {}).get("launch_token") or "")
    return tid, token


def _append_missing_token_event(commit_events: list, tid: str) -> None:
    commit_events.append({
        "type": "launch_commit_skipped",
        "task_id": tid or "?",
        "reason": "missing launch token",
    })


def _lease_current(task: dict | None, token: str) -> bool:
    return bool(
        task
        and task.get("status") == "launching"
        and str(task.get("launch_token") or "") == token
    )


def _lease_skip_reason(task: dict | None) -> str:
    return "launch lease no longer current" if task else "task no longer exists in state"


def _record_post_launch_snapshot(task: dict, tid: str, intent: dict, commit_events: list) -> None:
    post_launch_snapshot = intent.get("post_launch_snapshot")
    if not post_launch_snapshot:
        return
    task["last_launch_post_snapshot"] = post_launch_snapshot
    task["last_resource_snapshot"] = post_launch_snapshot
    commit_events.append({
        "type": "post_launch_snapshot",
        "task_id": tid,
        "snapshot": post_launch_snapshot,
    })


def _abort_stale_launched_tasks(tasks: list[dict], deps: LaunchCommitDeps) -> None:
    for task in tasks:
        try:
            deps.kill_task(task)
        except Exception:
            pass
        try:
            deps.release_task_claims_and_intents(task)
        except Exception:
            pass


def commit_deferred_launch_result(
    intent: dict,
    launched_task: dict,
    ok: bool,
    msg: str,
    *,
    lock_timeout=None,
    mark_notified_launch: bool = False,
    deps: LaunchCommitDeps,
) -> list:
    tid, token = _launch_commit_identity(intent, launched_task)
    commit_events: list = []
    if not tid or not token:
        _append_missing_token_event(commit_events, tid)
        return commit_events

    abort_launched_task = None
    with deps.state_lock(timeout_s=lock_timeout, purpose="launch:commit"):
        state = deps.load_state()
        by_id = {str(task.get("id") or ""): task for task in state.get("tasks", [])}
        task = by_id.get(tid)
        lease_current = _lease_current(task, token)
        if not lease_current:
            if ok:
                abort_launched_task = copy.deepcopy(launched_task)
            commit_events.append({
                "type": "launch_commit_skipped",
                "task_id": tid,
                "reason": _lease_skip_reason(task),
            })
        elif ok:
            task.clear()
            task.update(launched_task)
            if mark_notified_launch:
                task["notified_launch"] = True
            deps.apply_launch_result_to_task(task, True, msg, commit_events)
            _record_post_launch_snapshot(task, tid, intent, commit_events)
        else:
            deps.apply_launch_result_to_task(task, False, msg, commit_events)
        if lease_current:
            deps.save_state(state)

    if abort_launched_task is not None:
        _abort_stale_launched_tasks([abort_launched_task], deps)
    return commit_events


def commit_deferred_launch_results_batch(
    records: list[dict],
    *,
    lock_timeout=None,
    mark_notified_launch: bool = False,
    deps: LaunchCommitDeps,
) -> dict[int, list]:
    results_by_idx: dict[int, list] = {}
    abort_launched_tasks: list[dict] = []
    if not records:
        return results_by_idx

    with deps.state_lock(timeout_s=lock_timeout, purpose="launch:commit-batch"):
        state = deps.load_state()
        by_id = {str(task.get("id") or ""): task for task in state.get("tasks", [])}
        dirty = False
        for record in sorted(records, key=lambda r: int(r.get("idx") or 0)):
            idx = int(record.get("idx") or 0)
            intent = record.get("intent") or {}
            launched_task = record.get("task") or {}
            ok = bool(record.get("ok"))
            msg = str(record.get("msg") or "")
            tid, token = _launch_commit_identity(intent, launched_task)
            commit_events: list = []
            results_by_idx[idx] = commit_events
            if not tid or not token:
                _append_missing_token_event(commit_events, tid)
                continue

            task = by_id.get(tid)
            if not _lease_current(task, token):
                if ok:
                    abort_launched_tasks.append(copy.deepcopy(launched_task))
                commit_events.append({
                    "type": "launch_commit_skipped",
                    "task_id": tid,
                    "reason": _lease_skip_reason(task),
                })
                continue

            if ok:
                task.clear()
                task.update(launched_task)
                if mark_notified_launch:
                    task["notified_launch"] = True
                deps.apply_launch_result_to_task(task, True, msg, commit_events)
                _record_post_launch_snapshot(task, tid, intent, commit_events)
            else:
                deps.apply_launch_result_to_task(task, False, msg, commit_events)
            dirty = True
        if dirty:
            deps.save_state(state)

    _abort_stale_launched_tasks(abort_launched_tasks, deps)
    return results_by_idx


def execute_deferred_launches(
    events: list,
    *,
    lock_timeout=None,
    mark_notified_launch: bool = False,
    deps: DeferredLaunchDeps,
) -> list:
    def _commit_intent(ev: dict, task: dict, ok: bool, msg: str) -> list:
        return commit_deferred_launch_result(
            ev,
            task,
            ok,
            msg,
            lock_timeout=lock_timeout,
            mark_notified_launch=mark_notified_launch,
            deps=deps.commit_deps,
        )

    def _commit_batch(records: list[dict]) -> dict[int, list]:
        return commit_deferred_launch_results_batch(
            records,
            lock_timeout=lock_timeout,
            mark_notified_launch=mark_notified_launch,
            deps=deps.commit_deps,
        )

    return deps.execute_launch_intents(
        events,
        max_workers=deps.max_workers,
        launch_fn=deps.launch,
        commit_fn=_commit_intent,
        batch_commit_fn=_commit_batch,
        slot_lock_factory=lambda node: deps.launch_exec_slot_lock(node),
        notify_fn=deps.notify,
    )
