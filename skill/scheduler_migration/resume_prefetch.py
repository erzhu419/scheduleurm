"""Claim, execute, and commit resume scans without holding scheduler state."""

from __future__ import annotations

import copy
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class ResumePrefetchDeps:
    state_lock: Callable[..., Any]
    load_state: Callable[[], dict]
    save_state: Callable[[dict], Any]
    task_requires_resume_scan: Callable[[dict], bool]
    resume_scan_key: Callable[[dict, Optional[list]], list]
    cached_resume_locations_for_task: Callable[[dict, list], tuple[bool, list, dict]]
    scan_resume_locations: Callable[..., tuple[list, dict]]
    apply_resume_scan_result: Callable[..., None]
    notify: Callable[..., Any]
    max_tasks_per_pass: int
    task_workers: int
    inflight_ttl_s: int
    now: Callable[[], float]
    token_factory: Callable[[dict], str]


def _owner_process_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        return True
    return True


def _inflight_owner_is_dead(inflight: dict) -> bool:
    token = str(inflight.get("token") or "")
    owner, separator, _ = token.partition(":")
    if not separator or not owner.isdigit():
        return False
    pid = int(owner)
    if pid <= 0 or pid == os.getpid():
        return False
    return not _owner_process_is_alive(pid)


def _ordered_queued_tasks(state: dict, task_ids: set[str]) -> list[dict]:
    tasks = [task for task in state.get("tasks", []) if task.get("status") == "queued"]
    if task_ids:
        tasks = [task for task in tasks if str(task.get("id") or "") in task_ids]
    buckets = {"high": [], "normal": [], "low": []}
    for task in tasks:
        priority = str(task.get("priority") or "normal")
        buckets[priority if priority in buckets else "normal"].append(task)
    for bucket in buckets.values():
        bucket.sort(key=lambda task: (
            float(task.get("submitted_at") or 0),
            str(task.get("id") or ""),
        ))

    offsets = {priority: 0 for priority in buckets}
    ordered = []
    while len(ordered) < len(tasks):
        advanced = False
        for priority, weight in (("high", 4), ("normal", 2), ("low", 1)):
            for _ in range(weight):
                idx = offsets[priority]
                if idx >= len(buckets[priority]):
                    break
                ordered.append(buckets[priority][idx])
                offsets[priority] = idx + 1
                advanced = True
        if not advanced:
            break
    return ordered


def _claim_scan_work(
    nodes: list,
    task_ids: set[str],
    deps: ResumePrefetchDeps,
) -> list[tuple[dict, str, list]]:
    now = deps.now()
    claimed: list[tuple[dict, str, list]] = []
    dirty = False
    with deps.state_lock(purpose="resume-prefetch:claim"):
        state = deps.load_state()
        for task in _ordered_queued_tasks(state, task_ids):
            if len(claimed) >= max(1, int(deps.max_tasks_per_pass or 1)):
                break
            if not deps.task_requires_resume_scan(task):
                continue
            fresh, _, _ = deps.cached_resume_locations_for_task(task, nodes)
            if fresh:
                if task.pop("resume_scan_inflight", None) is not None:
                    dirty = True
                continue
            scan_key = deps.resume_scan_key(task, nodes)
            inflight = task.get("resume_scan_inflight") or {}
            try:
                inflight_age = now - float(inflight.get("started_at") or 0)
            except Exception:
                inflight_age = deps.inflight_ttl_s + 1
            if (
                inflight.get("token")
                and inflight.get("key") == scan_key
                and 0 <= inflight_age <= deps.inflight_ttl_s
                and not _inflight_owner_is_dead(inflight)
            ):
                continue
            token = deps.token_factory(task)
            task["resume_scan_inflight"] = {
                "token": token,
                "started_at": now,
                "key": scan_key,
            }
            claimed.append((copy.deepcopy(task), token, scan_key))
            dirty = True
        if dirty:
            deps.save_state(state)
    return claimed


def _scan_claimed_tasks(
    claimed: list[tuple[dict, str, list]],
    nodes: list,
    deps: ResumePrefetchDeps,
) -> list[dict]:
    if not claimed:
        return []

    def scan_one(item: tuple[dict, str, list]) -> dict:
        task, token, scan_key = item
        try:
            locations, errors = deps.scan_resume_locations(
                task,
                nodes=nodes,
                cache={},
            )
        except Exception as exc:
            locations, errors = [], {"prefetch": str(exc)[:300]}
        return {
            "task_id": task.get("id"),
            "token": token,
            "scan_key": scan_key,
            "locations": locations,
            "errors": errors,
            "completed_at": deps.now(),
        }

    workers = max(1, min(int(deps.task_workers or 1), len(claimed)))
    if workers == 1:
        return [scan_one(item) for item in claimed]
    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(scan_one, item) for item in claimed]
        for future in as_completed(futures):
            results.append(future.result())
    return results


def _commit_scan_results(
    results: list[dict],
    nodes: list,
    deps: ResumePrefetchDeps,
) -> int:
    if not results:
        return 0
    committed = 0
    dirty = False
    with deps.state_lock(purpose="resume-prefetch:commit"):
        state = deps.load_state()
        by_id = {task.get("id"): task for task in state.get("tasks", [])}
        for result in results:
            task = by_id.get(result.get("task_id"))
            if not task:
                continue
            inflight = task.get("resume_scan_inflight") or {}
            if inflight.get("token") != result.get("token"):
                continue
            current_key = deps.resume_scan_key(task, nodes)
            task.pop("resume_scan_inflight", None)
            dirty = True
            if task.get("status") != "queued" or current_key != result.get("scan_key"):
                continue
            deps.apply_resume_scan_result(
                task,
                scan_key=result["scan_key"],
                locations=result.get("locations") or [],
                errors=result.get("errors") or {},
                completed_at=float(result.get("completed_at") or deps.now()),
                source="outside_state_lock",
            )
            committed += 1
        if dirty:
            deps.save_state(state)
    return committed


def prefetch_resume_scans_outside_lock(
    nodes: list,
    task_ids: Optional[set[str]] = None,
    *,
    deps: ResumePrefetchDeps,
) -> dict:
    """Run a bounded resume-scan work batch using claim/work/commit phases."""
    started_at = deps.now()
    selected_ids = {str(task_id) for task_id in (task_ids or set()) if str(task_id)}
    claimed = _claim_scan_work(nodes, selected_ids, deps)
    results = _scan_claimed_tasks(claimed, nodes, deps)
    committed = _commit_scan_results(results, nodes, deps)
    summary = {
        "claimed": len(claimed),
        "completed": len(results),
        "committed": committed,
        "seconds": round(max(0.0, deps.now() - started_at), 3),
    }
    if claimed:
        try:
            deps.notify("resume_scan_prefetch", summary, feishu_enabled=False)
        except Exception:
            pass
    return summary
