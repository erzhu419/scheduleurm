from __future__ import annotations

import copy
import fnmatch
import re
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable


ACTIVE_CANCEL_STATUSES = frozenset({"queued", "launching", "running"})
TASK_ID_RANGE_RE = re.compile(r"^t(\d+)-t?(\d+)$", re.IGNORECASE)
MAX_CANCEL_RANGE_SIZE = 10000


@dataclass(frozen=True)
class TaskControlDeps:
    read_dispatch_intent: Callable[[], dict | None]
    cancel_defers_to_dispatch_intent: bool
    dispatch_intent_message: Callable[[dict], str]
    state_lock: Callable[..., object]
    load_state: Callable[[], dict]
    save_state: Callable[[dict], object]
    recover_stale_launching_tasks: Callable[[dict], object]
    release_task_claims_and_intents: Callable[[dict], object]
    mark_user_cancelled: Callable[[dict, str], object]
    cancel_related_queued_retries: Callable[[dict, dict, str], int]
    notify: Callable[..., object]
    task_pids: Callable[[dict], list]
    record_task_kill_actor: Callable[[dict, str, str], dict]
    kill_task_processes: Callable[..., tuple]
    actor_info: Callable[[str, str], dict]
    now: Callable[[], float]
    write_dispatch_intent: Callable[..., object] = lambda **_kwargs: None
    clear_dispatch_intent: Callable[[], object] = lambda: None
    kill_task_processes_batch: Callable[..., dict] | None = None
    cancel_kill_max_workers: int = 4


def parse_cancel_task_ids(raw_values) -> list[str]:
    """Expand comma groups and inclusive tNNN-tMMM ranges, preserving order."""
    if isinstance(raw_values, str):
        raw_values = [raw_values]
    out = []
    seen = set()
    for raw in raw_values or []:
        for token in str(raw).split(","):
            token = token.strip()
            if not token:
                continue
            match = TASK_ID_RANGE_RE.fullmatch(token)
            if match:
                start_text, end_text = match.groups()
                start, end = int(start_text), int(end_text)
                count = abs(end - start) + 1
                if count > MAX_CANCEL_RANGE_SIZE:
                    raise ValueError(
                        f"task id range {token!r} expands to {count} tasks; "
                        f"maximum is {MAX_CANCEL_RANGE_SIZE}"
                    )
                width = max(len(start_text), len(end_text))
                step = 1 if end >= start else -1
                expanded = [f"t{value:0{width}d}" for value in range(start, end + step, step)]
            else:
                expanded = [token]
            for task_id in expanded:
                if task_id not in seen:
                    seen.add(task_id)
                    out.append(task_id)
    return out


def _cancel_ids_from_args(args) -> list[str]:
    raw_ids = getattr(args, "ids", None)
    if not raw_ids:
        legacy_id = getattr(args, "id", None)
        raw_ids = [legacy_id] if legacy_id else []
    try:
        return parse_cancel_task_ids(raw_ids)
    except ValueError as exc:
        sys.exit(str(exc))


def _batch_cancel_requested(args, task_ids: list[str]) -> bool:
    return bool(
        getattr(args, "cmd", "") == "cancel-batch"
        or len(task_ids) != 1
        or getattr(args, "project", None)
        or getattr(args, "signature", None)
        or getattr(args, "statuses", None)
        or getattr(args, "dry_run", False)
        or getattr(args, "confirm", False)
        or getattr(args, "ignore_missing", False)
    )


def _check_cancel_dispatch_intent(args, deps: TaskControlDeps) -> dict | None:
    intent = deps.read_dispatch_intent()
    if (
        intent
        and deps.cancel_defers_to_dispatch_intent
        and not getattr(args, "ignore_dispatch_intent", False)
        and not getattr(args, "force_lock_wait", False)
    ):
        sys.exit(
            "cancel deferred: " + deps.dispatch_intent_message(intent)
            + "; rerun with --force-lock-wait or --ignore-dispatch-intent for urgent cancellation"
        )
    return intent


def _cmd_cancel_one(args, *, deps: TaskControlDeps) -> None:
    _check_cancel_dispatch_intent(args, deps)
    lock_timeout = getattr(args, "lock_timeout", None)
    with deps.state_lock(timeout_s=lock_timeout, purpose="cancel"):
        state = deps.load_state()
        deps.recover_stale_launching_tasks(state)
        for task in state["tasks"]:
            if task["id"] != args.id:
                continue
            if task["status"] in ("queued", "launching"):
                prev = task["status"]
                deps.release_task_claims_and_intents(task)
                deps.mark_user_cancelled(task, "user cancel")
                related = deps.cancel_related_queued_retries(state, task, "user cancel")
                task.pop("launching_started_at", None)
                task.pop("launch_token", None)
                deps.save_state(state)
                deps.notify("task_cancelled", {
                    "task_id": task.get("id"),
                    "status_before": prev,
                    "actor": task.get("cancel_actor"),
                    "reason": task.get("cancel_reason"),
                    "related_cancelled": related,
                }, feishu_enabled=False)
                suffix = f" (+{related} duplicate queued retry)" if related else ""
                print(f"cancelled {prev} task {args.id}{suffix} by {task.get('cancelled_by')}")
                return
            if task["status"] == "running":
                if not args.force:
                    sys.exit(f"task {args.id} is RUNNING — pass --force to kill it (will not affect other tasks)")
                pids = deps.task_pids(task)
                actor = deps.record_task_kill_actor(
                    task,
                    "user force-cancel",
                    "scheduler.py cancel --force",
                )
                request_token = uuid.uuid4().hex
                task["cancel_request_token"] = request_token
                task["cancel_requested_at"] = deps.now()
                task["cancel_request_actor"] = actor
                task["cancel_request_reason"] = "user force-cancel"
                deps.save_state(state)
                ok, kill_msg = deps.kill_task_processes(task, timeout=15)
                task["last_kill_actor"] = actor
                task["last_killed_by"] = actor["label"]
                task["last_kill_ok"] = bool(ok)
                task["last_kill_message"] = str(kill_msg or "")[:500]
                related = 0
                if ok:
                    try:
                        deps.release_task_claims_and_intents(task)
                    except Exception:
                        pass
                    deps.mark_user_cancelled(task, "user force-cancel")
                    task["last_cancel_request_token"] = request_token
                    for key in (
                        "cancel_request_token",
                        "cancel_requested_at",
                        "cancel_request_actor",
                        "cancel_request_reason",
                    ):
                        task.pop(key, None)
                    related = deps.cancel_related_queued_retries(
                        state,
                        task,
                        "user force-cancel",
                    )
                else:
                    task["last_block_reason"] = (
                        "cancel requested but remote process-tree termination was not confirmed"
                    )
                deps.save_state(state)
                deps.notify("task_killed", {
                    "task_id": task.get("id"),
                    "node": task.get("node"),
                    "pids": pids,
                    "actor": actor,
                    "action": "user force-cancel",
                    "reason": "scheduler.py cancel --force",
                    "kill_ok": ok,
                    "kill_msg": kill_msg,
                    "related_cancelled": related,
                }, feishu_enabled=False)
                if not ok:
                    sys.exit(
                        f"kill failed for task {args.id}; task remains RUNNING and tracked: "
                        f"{kill_msg}"
                    )
                suffix = kill_msg
                dup_suffix = f"; also cancelled {related} duplicate queued retry" if related else ""
                print(
                    f"killed pids={pids} on {task['node']} by {actor['label']} "
                    f"and cancelled {args.id} ({suffix}{dup_suffix})"
                )
                return
            sys.exit(f"task {args.id} is in state {task['status']!r} — nothing to do")
        sys.exit(f"task {args.id} not found")


def _select_batch_cancel_tasks(
    state: dict,
    args,
    requested_ids: list[str],
) -> tuple[list[dict], list[str], list[str]]:
    by_id = {
        str(task.get("id") or ""): task
        for task in state.get("tasks", [])
        if task.get("id")
    }
    missing = [task_id for task_id in requested_ids if task_id not in by_id]
    candidates = [by_id[task_id] for task_id in requested_ids if task_id in by_id]
    if not requested_ids:
        candidates = list(state.get("tasks", []))

    project_glob = str(getattr(args, "project", None) or "")
    signature_glob = str(getattr(args, "signature", None) or "")
    statuses = set(getattr(args, "statuses", None) or ACTIVE_CANCEL_STATUSES)
    selected = []
    skipped = []
    for task in candidates:
        task_id = str(task.get("id") or "")
        if project_glob and not fnmatch.fnmatchcase(str(task.get("project") or ""), project_glob):
            if requested_ids:
                skipped.append(task_id)
            continue
        if signature_glob and not fnmatch.fnmatchcase(
            str(task.get("signature") or ""), signature_glob
        ):
            if requested_ids:
                skipped.append(task_id)
            continue
        if str(task.get("status") or "") not in statuses:
            if requested_ids:
                skipped.append(task_id)
            continue
        selected.append(task)
    return selected, missing, skipped


def _task_id_summary(task_ids: list[str], *, limit: int = 30) -> str:
    shown = task_ids[:limit]
    suffix = f", ... (+{len(task_ids) - limit})" if len(task_ids) > limit else ""
    return "[" + ", ".join(shown) + suffix + "]"


def _batch_status_counts(records: list[dict]) -> str:
    counts = {
        status: sum(1 for rec in records if rec.get("status_before") == status)
        for status in ("queued", "launching", "running")
    }
    return " ".join(f"{status}={counts[status]}" for status in counts)


def _kill_batch_outside_lock(records: list[dict], *, deps: TaskControlDeps) -> None:
    by_node: dict[str, list[dict]] = {}
    for rec in records:
        by_node.setdefault(str(rec["task"].get("node") or ""), []).append(rec)

    def _kill_node(group: list[dict]) -> list[dict]:
        if deps.kill_task_processes_batch is not None:
            try:
                results = deps.kill_task_processes_batch(
                    [rec["task"] for rec in group],
                    timeout=15,
                ) or {}
            except Exception as exc:
                results = {
                    rec["id"]: (False, str(exc)[:200])
                    for rec in group
                }
            for rec in group:
                ok, message = results.get(rec["id"], (False, "missing batch kill result"))
                rec["kill_ok"] = bool(ok)
                rec["kill_msg"] = str(message or "")[:500]
            return group
        for rec in group:
            try:
                ok, message = deps.kill_task_processes(rec["task"], timeout=15)
            except Exception as exc:
                ok, message = False, str(exc)[:200]
            rec["kill_ok"] = bool(ok)
            rec["kill_msg"] = str(message or "")[:500]
        return group

    workers = max(1, min(len(by_node), int(deps.cancel_kill_max_workers or 1)))
    if by_node:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(_kill_node, by_node.values()))


def cmd_cancel_batch(args, *, deps: TaskControlDeps) -> None:
    requested_ids = _cancel_ids_from_args(args)
    project_glob = str(getattr(args, "project", None) or "")
    signature_glob = str(getattr(args, "signature", None) or "")
    if not requested_ids and not project_glob and not signature_glob:
        sys.exit("cancel-batch requires task IDs, --project, or --signature")

    selector_only = not requested_ids and bool(project_glob or signature_glob)
    preview = bool(getattr(args, "dry_run", False) or (
        selector_only and not getattr(args, "confirm", False)
    ))
    intent = None
    owns_intent = False
    if not preview:
        intent = _check_cancel_dispatch_intent(args, deps)
        if not intent:
            written = deps.write_dispatch_intent(label="batch-cancel", ttl_s=900)
            owns_intent = written is not None

    records: list[dict] = []
    ignored_missing: list[str] = []
    skipped: list[str] = []
    request_token = uuid.uuid4().hex
    try:
        lock_kwargs = {
            "timeout_s": getattr(args, "lock_timeout", None),
            "purpose": "cancel-batch:preview" if preview else "cancel-batch:prepare",
        }
        if preview:
            lock_kwargs["shared"] = True
        with deps.state_lock(**lock_kwargs):
            state = deps.load_state()
            if not preview:
                deps.recover_stale_launching_tasks(state)
            selected, missing, skipped = _select_batch_cancel_tasks(state, args, requested_ids)
            if missing and not getattr(args, "ignore_missing", False):
                sys.exit(
                    f"task(s) not found: {_task_id_summary(missing)}; "
                    "no tasks were cancelled (pass --ignore-missing for idempotent cleanup)"
                )
            ignored_missing = missing
            if not selected:
                detail = f"; skipped={_task_id_summary(skipped)}" if skipped else ""
                print(f"no active tasks matched batch cancel{detail}")
                return
            running_ids = [str(task.get("id")) for task in selected if task.get("status") == "running"]
            if running_ids and not getattr(args, "force", False) and not preview:
                sys.exit(
                    f"batch contains {len(running_ids)} RUNNING task(s) "
                    f"{_task_id_summary(running_ids)}; no tasks were cancelled. "
                    "Pass --force to kill the whole selected running set"
                )

            records = [
                {
                    "id": str(task.get("id") or ""),
                    "status_before": str(task.get("status") or ""),
                    "task_ref": task,
                    "pids": deps.task_pids(task),
                    "related_cancelled": 0,
                }
                for task in selected
            ]
            if preview:
                print(
                    f"would cancel {len(records)} task(s): {_batch_status_counts(records)} "
                    f"ids={_task_id_summary([rec['id'] for rec in records])}"
                )
                if selector_only and not getattr(args, "dry_run", False):
                    print("selector-based batch cancel is preview-only without --confirm")
                return

            reason = "user batch force-cancel" if running_ids else "user batch cancel"
            for rec in records:
                task = rec["task_ref"]
                if rec["status_before"] == "running":
                    actor = deps.record_task_kill_actor(
                        task,
                        "user batch force-cancel",
                        "scheduler.py cancel-batch --force",
                    )
                    task["cancel_request_token"] = request_token
                    task["cancel_requested_at"] = deps.now()
                    task["cancel_request_actor"] = actor
                    task["cancel_request_reason"] = reason
                    rec["actor"] = actor
                else:
                    deps.mark_user_cancelled(task, reason)
                    rec["actor"] = copy.deepcopy(task.get("cancel_actor") or {})

            for rec in records:
                if rec["status_before"] != "running":
                    rec["related_cancelled"] += deps.cancel_related_queued_retries(
                        state,
                        rec["task_ref"],
                        reason,
                    )
                rec["task"] = copy.deepcopy(rec["task_ref"])
                rec.pop("task_ref", None)
            deps.save_state(state)

        running_records = [rec for rec in records if rec["status_before"] == "running"]
        _kill_batch_outside_lock(running_records, deps=deps)

        release_warnings = []
        for rec in records:
            if rec["status_before"] == "running" and not rec.get("kill_ok"):
                continue
            try:
                deps.release_task_claims_and_intents(rec["task"])
            except Exception as exc:
                release_warnings.append(f"{rec['id']}: {str(exc)[:120]}")

        if running_records:
            with deps.state_lock(
                timeout_s=getattr(args, "lock_timeout", None),
                purpose="cancel-batch:finalize",
            ):
                state = deps.load_state()
                by_id = {
                    str(task.get("id") or ""): task
                    for task in state.get("tasks", [])
                    if task.get("id")
                }
                changed = False
                for rec in running_records:
                    task = by_id.get(rec["id"])
                    if task is None:
                        continue
                    request_pending = task.get("cancel_request_token") == request_token
                    request_already_finalized = (
                        task.get("status") == "cancelled"
                        and task.get("last_cancel_request_token") == request_token
                    )
                    if not request_pending and not request_already_finalized:
                        continue
                    kill_ok = bool(rec.get("kill_ok"))
                    if request_pending and kill_ok:
                        deps.mark_user_cancelled(task, "user batch force-cancel")
                    actor = rec.get("actor") or {}
                    task["last_kill_actor"] = actor
                    task["last_killed_by"] = actor.get("label") or task.get("cancelled_by")
                    task["last_kill_ok"] = kill_ok
                    task["last_kill_message"] = rec.get("kill_msg") or ""
                    if kill_ok or request_already_finalized:
                        task["last_cancel_request_token"] = request_token
                        for key in (
                            "cancel_request_token",
                            "cancel_requested_at",
                            "cancel_request_actor",
                            "cancel_request_reason",
                        ):
                            task.pop(key, None)
                        if kill_ok:
                            rec["related_cancelled"] += deps.cancel_related_queued_retries(
                                state,
                                task,
                                "user batch force-cancel",
                            )
                    else:
                        task["last_block_reason"] = (
                            "cancel requested but remote process-tree termination was not confirmed"
                        )
                    changed = True
                if changed:
                    deps.save_state(state)

        failed_running = [
            rec for rec in running_records
            if not rec.get("kill_ok")
        ]
        cancelled_records = [
            rec for rec in records
            if rec["status_before"] != "running" or rec.get("kill_ok")
        ]
        payload = {
            "count": len(cancelled_records),
            "selected_count": len(records),
            "task_ids": [rec["id"] for rec in cancelled_records],
            "selected_task_ids": [rec["id"] for rec in records],
            "kill_failed_task_ids": [rec["id"] for rec in failed_running],
            "status_counts": {
                status: sum(1 for rec in records if rec["status_before"] == status)
                for status in ("queued", "launching", "running")
            },
            "force": bool(getattr(args, "force", False)),
            "related_cancelled": sum(rec["related_cancelled"] for rec in records),
            "kill_warnings": [
                {"task_id": rec["id"], "message": rec.get("kill_msg") or ""}
                for rec in running_records
                if not rec.get("kill_ok")
            ],
            "release_warnings": release_warnings,
            "ignored_missing": ignored_missing,
            "skipped": skipped,
            "actor": deps.actor_info("cancel-batch", "user batch cancel"),
        }
        deps.notify("tasks_cancelled_batch", payload, feishu_enabled=False)
        print(
            f"cancelled {len(cancelled_records)} task(s) in batch "
            f"({len(records)} selected): "
            f"{_batch_status_counts(records)}; related={payload['related_cancelled']} "
            f"kill_failures={len(failed_running)}"
        )
        print(f"cancelled_task_ids={_task_id_summary(payload['task_ids'])}")
        if failed_running:
            print(
                "still_running_task_ids="
                f"{_task_id_summary(payload['kill_failed_task_ids'])}"
            )
        if ignored_missing:
            print(f"ignored missing task_ids={_task_id_summary(ignored_missing)}")
        if failed_running:
            sys.exit(
                f"failed to confirm process-tree termination for {len(failed_running)} "
                "task(s); they remain RUNNING and tracked"
            )
    finally:
        if owns_intent:
            deps.clear_dispatch_intent()


def cmd_cancel(args, *, deps: TaskControlDeps) -> None:
    task_ids = _cancel_ids_from_args(args)
    if not _batch_cancel_requested(args, task_ids):
        args.id = task_ids[0]
        return _cmd_cancel_one(args, deps=deps)
    return cmd_cancel_batch(args, deps=deps)


def cmd_forget(args, *, deps: TaskControlDeps) -> None:
    with deps.state_lock():
        state = deps.load_state()
        for task in state["tasks"]:
            if task["id"] != args.id:
                continue
            prev = task["status"]
            deps.release_task_claims_and_intents(task)
            task["status"] = "forgotten"
            task["finished_at"] = deps.now()
            deps.save_state(state)
            print(f"forgot {args.id} (was {prev}). No processes were touched.")
            return
        sys.exit(f"task {args.id} not found")


def cmd_clear_queue(args, *, deps: TaskControlDeps) -> None:
    with deps.state_lock():
        state = deps.load_state()
        ids = [task["id"] for task in state["tasks"] if task["status"] == "queued"]
        if not args.confirm:
            print(f"would cancel {len(ids)} queued tasks: {ids}")
            print("running tasks would NOT be touched. Re-run with --confirm to apply.")
            return
        for task in state["tasks"]:
            if task["status"] == "queued":
                deps.release_task_claims_and_intents(task)
                deps.mark_user_cancelled(task, "user clear-queue")
        deps.save_state(state)
        deps.notify("clear_queue_cancelled", {
            "count": len(ids),
            "task_ids": ids,
            "actor": deps.actor_info("clear-queue", "user clear-queue"),
        }, feishu_enabled=False)
        print(f"cancelled {len(ids)} queued tasks (running tasks untouched)")
