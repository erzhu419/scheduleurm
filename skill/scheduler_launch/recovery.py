"""Recovery for stale launch WAL records."""

from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Optional

from scheduler_backend.local_launch import runtime_exit_status_path


@dataclass(frozen=True)
class LaunchRecoveryDeps:
    try_recover_orphan_local_task: Callable[[dict, str], bool]
    try_finalize_terminal_local_task: Callable[[dict, str, dict], bool]
    claim_enabled_for: Callable[[str], bool]
    release_task_claims_and_intents: Callable[..., Any]
    clear_live_eta_fields: Callable[..., Any]
    now: Callable[[], float]


@dataclass(frozen=True)
class StaleLaunchRecoveryDeps:
    state_lock: Callable[..., Any]
    load_state: Callable[[], dict]
    save_state: Callable[[dict], Any]
    scan_node_evidence: Callable[[str, list[dict]], dict]
    adopt_live_pid_rows: Callable[[dict, str, list, Optional[str]], bool]
    node_is_windows: Callable[[str], bool]
    claim_enabled_for: Callable[[str], bool]
    release_task_claims_and_intents: Callable[..., Any]
    clear_live_eta_fields: Callable[..., Any]
    set_current_usage: Callable[[dict, int, int, float], Any]
    notify: Callable[..., Any]
    now: Callable[[], float]
    max_workers: int = 8


def _launch_identity(task: dict) -> tuple[str, str, float, str]:
    return (
        str(task.get("id") or ""),
        str(task.get("launch_token") or ""),
        float(task.get("launching_started_at") or 0.0),
        str(task.get("node") or ""),
    )


def _revert_stale_launch(task: dict, *, age: float, deps: StaleLaunchRecoveryDeps) -> None:
    node = task.get("node")
    if node and deps.claim_enabled_for(node):
        try:
            deps.release_task_claims_and_intents(task, extra_nodes=[node])
        except Exception:
            pass
    task["status"] = "queued"
    task["last_block_reason"] = (
        f"WAL recovery: was 'launching' for {max(0, age):.0f}s with no live "
        "process or exit sentinel; reverted to queued"
    )
    deps.clear_live_eta_fields(task, clear_runtime_projection=True)
    for key in (
        "launching_started_at",
        "launch_token",
        "exit_status_path",
        "exit_status_token",
        "remote_pid_start_ticks",
        "remote_pids",
        "alive_pids",
        "process_group",
    ):
        task.pop(key, None)


def _promote_exit_sentinel(
    task: dict,
    exit_status: dict,
    *,
    deps: StaleLaunchRecoveryDeps,
) -> bool:
    task_id = str(task.get("id") or "")
    token = str(task.get("launch_token") or task.get("exit_status_token") or "")
    if not task_id or not token or str(exit_status.get("token") or "") != token:
        return False
    task["status"] = "running"
    task["started_at"] = task.get("launching_started_at") or task.get("started_at") or deps.now()
    task["finished_at"] = None
    task["remote_pids"] = []
    task["alive_pids"] = []
    task["process_group"] = None
    task["exit_status_token"] = token
    task["exit_status_path"] = str(
        task.get("exit_status_path") or runtime_exit_status_path(task_id, token)
    )
    task["exit_code"] = int(exit_status["exit_code"])
    task["backend_finished_at"] = float(exit_status["finished_at"])
    task["orphan_recovered_at"] = deps.now()
    task["orphan_recovered_from_status"] = "launching_exit_sentinel"
    task["last_block_reason"] = (
        "WAL recovery: launcher exit sentinel found; terminal diagnosis pending"
    )
    task.pop("launching_started_at", None)
    task.pop("launch_token", None)
    task.pop("remote_pid_start_ticks", None)
    deps.set_current_usage(task, 0, 0, 0.0)
    return True


def recover_stale_launching_tasks_outside_lock(
    *,
    reset_s: int,
    deps: StaleLaunchRecoveryDeps,
    purpose: str = "launch-recovery",
    lock_timeout=None,
) -> int:
    """Probe stale launch leases outside the state lock and merge by identity."""
    lock_kwargs = {"timeout_s": lock_timeout} if lock_timeout is not None else {}
    snapshot_now = deps.now()
    with deps.state_lock(shared=True, purpose=f"{purpose}:snapshot", **lock_kwargs):
        state = deps.load_state()
        candidates = [
            copy.deepcopy(task)
            for task in state.get("tasks", [])
            if task.get("status") == "launching"
            and snapshot_now - (task.get("launching_started_at") or snapshot_now) >= reset_s
        ]
    if not candidates:
        return 0

    tasks_by_node: dict[str, list[dict]] = {}
    for task in candidates:
        tasks_by_node.setdefault(str(task.get("node") or ""), []).append(task)

    def scan(item: tuple[str, list[dict]]) -> tuple[str, dict]:
        node, tasks = item
        if not node:
            return node, {
                "ok": True,
                "supported": True,
                "rows_by_task": {},
                "exit_statuses": {},
            }
        if deps.node_is_windows(node):
            return node, {
                "ok": True,
                "supported": False,
                "rows_by_task": {},
                "exit_statuses": {},
            }
        try:
            return node, deps.scan_node_evidence(node, tasks)
        except Exception as exc:
            return node, {
                "ok": False,
                "supported": True,
                "rows_by_task": {},
                "exit_statuses": {},
                "error": f"{type(exc).__name__}: {str(exc)[:220]}",
            }

    items = list(tasks_by_node.items())
    workers = max(1, min(len(items), int(deps.max_workers or 1)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        evidence_by_node = dict(pool.map(scan, items))

    snapshot_by_identity = {_launch_identity(task): task for task in candidates}
    candidate_ids = {str(task.get("id") or "") for task in candidates}
    reverted = adopted = terminal = deferred = changed = identity_changed = 0
    commit_now = deps.now()
    with deps.state_lock(purpose=f"{purpose}:commit", **lock_kwargs):
        state = deps.load_state()
        for task in state.get("tasks", []):
            identity = _launch_identity(task)
            snapshot = snapshot_by_identity.get(identity)
            if snapshot is None or task.get("status") != "launching":
                if (
                    task.get("status") == "launching"
                    and str(task.get("id") or "") in candidate_ids
                ):
                    identity_changed += 1
                continue
            node = str(task.get("node") or "")
            evidence = evidence_by_node.get(node) or {}
            if not evidence.get("ok"):
                deferred += 1
                continue
            if not evidence.get("supported", True):
                deferred += 1
                continue

            rows = (evidence.get("rows_by_task") or {}).get(task.get("id")) or []
            if rows:
                if deps.adopt_live_pid_rows(task, node, rows, "launching"):
                    adopted += 1
                    changed += 1
                else:
                    deferred += 1
                continue

            exit_status = (evidence.get("exit_statuses") or {}).get(task.get("id"))
            if exit_status:
                if _promote_exit_sentinel(task, exit_status, deps=deps):
                    terminal += 1
                    changed += 1
                else:
                    deferred += 1
                continue

            if not evidence.get("absence_proven", True):
                deferred += 1
                continue

            age = commit_now - (task.get("launching_started_at") or commit_now)
            _revert_stale_launch(task, age=age, deps=deps)
            reverted += 1
            changed += 1
        if changed:
            deps.save_state(state)

    errors = {
        node: evidence.get("error")
        for node, evidence in evidence_by_node.items()
        if evidence.get("error")
    }
    deps.notify(
        "launching_recovery_batch",
        {
            "candidates": len(candidates),
            "nodes": len(tasks_by_node),
            "adopted": adopted,
            "terminal_sentinel": terminal,
            "reverted": reverted,
            "deferred": deferred,
            "identity_changed": identity_changed,
            "errors": errors,
        },
        feishu_enabled=False,
    )
    return reverted


def recover_stale_launching_tasks(
    state: dict,
    *,
    reset_s: int,
    now: Optional[float] = None,
    deps: LaunchRecoveryDeps,
) -> int:
    """Revert or adopt stale LocalBackend launch WAL records."""
    now = deps.now() if now is None else now
    reverted = 0
    for task in state.get("tasks", []):
        if task.get("status") != "launching":
            continue
        age = now - (task.get("launching_started_at") or now)
        if age < reset_s:
            continue

        node = task.get("node")
        if node:
            if deps.try_recover_orphan_local_task(task, node):
                continue
            if deps.try_finalize_terminal_local_task(task, node, state):
                continue

        if node and deps.claim_enabled_for(node):
            try:
                deps.release_task_claims_and_intents(task, extra_nodes=[node])
            except Exception:
                pass
        task["status"] = "queued"
        task["last_block_reason"] = (
            f"WAL recovery: was 'launching' for {max(0, age):.0f}s, reverted to queued"
        )
        deps.clear_live_eta_fields(task, clear_runtime_projection=True)
        task.pop("launching_started_at", None)
        task.pop("launch_token", None)
        reverted += 1
    return reverted
