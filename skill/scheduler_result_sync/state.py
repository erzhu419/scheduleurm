from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class ResultSyncDeps:
    state_lock: Callable[..., Any]
    load_state: Callable[[], dict]
    save_state: Callable[[dict], Any]
    notify: Callable[..., Any]
    node_configs: dict
    infer_bapr_result_dirs_from_cmd: Callable[[str, str], list]
    sync_one_result: Callable[[dict], tuple[bool, str]]
    result_sync_timeout_s: int
    result_sync_stale_grace_s: int
    result_sync_max_per_cycle: int
    result_sync_dependency_max_per_cycle: int
    result_sync_max_attempts: int
    now: Callable[[], float]
    getpid: Callable[[], int] = lambda: 0
    pid_alive: Callable[[int], bool] = lambda _pid: False


def _notify(deps: ResultSyncDeps, event_type: str, payload: dict) -> None:
    try:
        deps.notify(event_type, payload, feishu_enabled=False)
    except Exception:
        pass


def _candidate_from_task(task: dict, result_dir, result_dirs: list, host) -> dict:
    candidate = {
        "id": task["id"],
        "node": task.get("node"),
        "host": host,
        "result_dir": result_dir,
        "local_result_dir": task.get("local_result_dir") or result_dir,
    }
    if result_dirs:
        candidate["result_dirs"] = result_dirs
        if task.get("local_result_dirs"):
            candidate["local_result_dirs"] = list(task.get("local_result_dirs") or [])
        elif task.get("local_result_dir"):
            candidate["local_result_dir_base"] = task.get("local_result_dir")
        candidate["result_dir"] = result_dirs[0]
        candidate["local_result_dir"] = task.get("local_result_dir") or result_dirs[0]
    return candidate


def _dependency_priority(task: dict, active_wait_files: tuple[str, ...]) -> int:
    local_dirs = list(task.get("local_result_dirs") or [])
    if not local_dirs:
        local_dir = task.get("local_result_dir") or task.get("result_dir")
        if local_dir:
            local_dirs.append(local_dir)
    for local_dir in local_dirs:
        prefix = str(local_dir).rstrip("/")
        if any(
            str(wait_file) == prefix
            or str(wait_file).startswith(prefix + "/")
            for wait_file in active_wait_files
        ):
            return 0
    return 1


def _snapshot_candidates(
    limit: int,
    dependency_limit: int,
    deps: ResultSyncDeps,
) -> list[dict]:
    stale_threshold = deps.result_sync_timeout_s + deps.result_sync_stale_grace_s
    candidates = []
    dependency_candidates = 0
    with deps.state_lock(purpose="watch:startup-recover-launching"):
        state = deps.load_state()
        now = deps.now()
        tasks = list(state.get("tasks", []))
        active_wait_files = tuple(
            str(wait_file)
            for waiter in tasks
            if waiter.get("status") in {"queued", "launching"}
            for wait_file in (waiter.get("wait_for_files") or [])
        )
        ordered_tasks = sorted(
            enumerate(tasks),
            key=lambda item: (
                _dependency_priority(item[1], active_wait_files),
                item[0],
            ),
        )
        for _, task in ordered_tasks:
            is_dependency = _dependency_priority(task, active_wait_files) == 0
            if is_dependency:
                if dependency_candidates >= dependency_limit:
                    continue
            elif len(candidates) >= limit:
                break
            if task.get("status") != "done":
                continue
            result_dir = task.get("result_dir")
            result_dirs = list(task.get("result_dirs") or [])
            if not result_dir and not result_dirs:
                result_dirs = deps.infer_bapr_result_dirs_from_cmd(
                    task.get("cmd", ""),
                    task.get("cwd", ""),
                )
            if not result_dir and not result_dirs:
                continue
            if task.get("result_synced_at"):
                continue
            if int(task.get("result_sync_attempts") or 0) >= deps.result_sync_max_attempts:
                continue
            node = task.get("node")
            if not node:
                continue
            node_info = deps.node_configs.get(node, {}) or {}
            if node_info.get("retired"):
                continue
            host = node_info.get("host")
            if not host:
                continue
            syncing_at = task.get("result_syncing_at")
            if syncing_at:
                age = now - float(syncing_at)
                owner_pid = int(task.get("result_syncing_pid") or 0)
                owner_dead = owner_pid > 0 and not deps.pid_alive(owner_pid)
                if age < stale_threshold and not owner_dead:
                    continue
                _notify(
                    deps,
                    "result_sync_claim_reclaimed",
                    {
                        "task_id": task["id"],
                        "stale_age_s": int(age),
                        "owner_pid": owner_pid or None,
                        "owner_dead": owner_dead,
                    },
                )
            task["result_syncing_at"] = now
            task["result_syncing_pid"] = deps.getpid()
            candidates.append(_candidate_from_task(task, result_dir, result_dirs, host))
            if is_dependency:
                dependency_candidates += 1
        if candidates:
            deps.save_state(state)
    return candidates


def _commit_results(results: list[tuple[str, bool, str]], deps: ResultSyncDeps) -> None:
    with deps.state_lock():
        state = deps.load_state()
        by_id = {task["id"]: task for task in state.get("tasks", [])}
        for task_id, ok, msg in results:
            task = by_id.get(task_id)
            if not task:
                continue
            task.pop("result_syncing_at", None)
            task.pop("result_syncing_pid", None)
            if task.get("status") != "done":
                continue
            if ok:
                task["result_synced_at"] = deps.now()
                task["result_sync_error"] = None
            else:
                task["result_sync_error"] = msg[:300]
                task["result_sync_attempts"] = int(task.get("result_sync_attempts") or 0) + 1
        deps.save_state(state)


def sync_completed_results_outside_lock(
    max_candidates: Optional[int] = None,
    *,
    deps: ResultSyncDeps,
) -> None:
    limit = (
        deps.result_sync_max_per_cycle
        if max_candidates is None
        else max(1, int(max_candidates))
    )
    dependency_limit = limit
    if max_candidates is None:
        dependency_limit = max(
            limit,
            int(deps.result_sync_dependency_max_per_cycle or 1),
        )
    try:
        candidates = _snapshot_candidates(limit, dependency_limit, deps)
    except Exception as exc:
        _notify(deps, "result_sync_snapshot_error", {"error": str(exc)[:200]})
        return

    if not candidates:
        return

    results = []
    for candidate in candidates:
        ok, msg = deps.sync_one_result(candidate)
        results.append((candidate["id"], ok, msg))
        _notify(
            deps,
            "result_sync_done" if ok else "result_sync_failed",
            {
                "task_id": candidate["id"],
                "result_dir": candidate["result_dir"],
                "local_result_dir": candidate["local_result_dir"],
                "msg": msg[:200],
            },
        )

    try:
        _commit_results(results, deps)
    except Exception as exc:
        _notify(deps, "result_sync_commit_error", {"error": str(exc)[:200]})
