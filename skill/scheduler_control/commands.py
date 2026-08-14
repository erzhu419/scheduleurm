from __future__ import annotations

import json
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class ControlCommandDeps:
    state_lock: Callable[..., object]
    load_state: Callable[[], dict]
    save_state: Callable[[dict], object]
    lock_timeout_error: type[BaseException]
    queue_file: Path
    archive_file: Path
    archive_max_hot_terminal: int
    compact_age_days_from_args: Callable[[object], float]
    terminal_archive_stats: Callable[..., dict]
    archive_terminal_tasks: Callable[..., int]
    node_configs: dict
    claim_enabled_for: Callable[[str], bool]
    claim_snapshot: Callable[[str], dict]
    format_claim_record: Callable[[dict], str]
    find_task_record: Callable[..., tuple[dict | None, str]]
    tail_task_log: Callable[..., tuple[bool, str, str]]
    node_is_windows: Callable[[str], bool]
    ssh_base_args: Callable[[str], list[str]]
    print_result_artifacts: Callable[..., object]


def cmd_compact(args, *, deps: ControlCommandDeps) -> None:
    """Move terminal tasks out of hot queue.json into queue_archive.jsonl."""
    age_days = deps.compact_age_days_from_args(args)
    max_hot_terminal = getattr(args, "keep_terminal", deps.archive_max_hot_terminal)
    if getattr(args, "all_terminal", False):
        max_hot_terminal = 0
    lock_timeout = getattr(args, "lock_timeout", None)
    try:
        with deps.state_lock(timeout_s=lock_timeout, purpose="compact"):
            state = deps.load_state()
            before_tasks = len(state.get("tasks", []))
            before_size = deps.queue_file.stat().st_size if deps.queue_file.exists() else 0
            stats = deps.terminal_archive_stats(
                state,
                age_days,
                max_hot_terminal=max_hot_terminal,
            )
            if getattr(args, "dry_run", False) or stats["count"] <= 0:
                action = "would archive" if getattr(args, "dry_run", False) else "archived"
                print(
                    f"{action} {stats['count']} terminal tasks "
                    f"(age>{age_days * 24.0:.2f}h: {stats['age_count']}, "
                    f"extra: {stats['extra_count']}, "
                    f"hot-terminal-excess: {stats['excess_count']}); "
                    f"keep_terminal={max_hot_terminal}; "
                    f"approx_hot_bytes={stats['approx_bytes']}"
                )
                print(f"  by_status={stats['by_status']}")
                print(f"  hot_queue_tasks={before_tasks} hot_queue_size={before_size}B")
                return
            archived = deps.archive_terminal_tasks(
                state,
                age_days=age_days,
                max_hot_terminal=max_hot_terminal,
            )
            deps.save_state(state)
            after_tasks = len(state.get("tasks", []))
        after_size = deps.queue_file.stat().st_size if deps.queue_file.exists() else 0
    except deps.lock_timeout_error as exc:
        sys.exit(str(exc))
    print(
        f"archived {archived} terminal tasks "
        f"(age>{age_days * 24.0:.2f}h: {stats['age_count']}, "
        f"extra: {stats['extra_count']}, "
        f"hot-terminal-excess: {stats['excess_count']}; "
        f"keep_terminal={max_hot_terminal}) "
        f"to {deps.archive_file}"
    )
    print(
        f"  hot_queue_tasks: {before_tasks} -> {after_tasks}; "
        f"hot_queue_size: {before_size}B -> {after_size}B"
    )
    print(f"  by_status={stats['by_status']}")


def cmd_claims(args, *, deps: ControlCommandDeps) -> None:
    nodes = [args.node] if args.node else [n for n in deps.node_configs if deps.claim_enabled_for(n)]
    snapshots = {}
    for node in nodes:
        snapshots[node] = deps.claim_snapshot(node)
    if getattr(args, "json", False):
        print(json.dumps(snapshots, indent=2))
        return
    if not snapshots:
        print("(no claims-enabled nodes)")
        return
    for node, snap in snapshots.items():
        if not snap.get("ok"):
            print(f"=== {node} claims ERROR ===")
            print(f"  {snap.get('error') or 'unknown error'}")
            continue
        claims = list(snap.get("claims") or [])
        intents = list(snap.get("intents") or [])
        print(f"=== {node} claims ===")
        if claims:
            for claim in sorted(claims, key=lambda item: (str(item.get("gpu_idx")), str(item.get("task_id")))):
                pid = claim.get("pid")
                pid_s = f" pid={pid}" if pid else " pending"
                print(f"  {deps.format_claim_record(claim)}{pid_s}")
        else:
            print("  (none)")
        print(f"=== {node} intents ===")
        if intents:
            intents.sort(
                key=lambda item: (
                    float(item.get("intent_at", 0) or 0),
                    str(item.get("scheduler_id")),
                    str(item.get("task_id")),
                )
            )
            for index, intent in enumerate(intents, 1):
                print(f"  {index:02d}. {deps.format_claim_record(intent)}")
        else:
            print("  (none)")


def cmd_task_log(args, *, deps: ControlCommandDeps) -> None:
    with deps.state_lock(shared=True, purpose="task-log:snapshot"):
        task, source = deps.find_task_record(
            args.id,
            include_archive=not getattr(args, "no_archive", False),
        )
    if not task:
        sys.exit(f"task {args.id} not found in queue" + (" or archive" if not args.no_archive else ""))
    ok, log_path, text = deps.tail_task_log(task, lines=args.lines)
    payload = {
        "ok": bool(ok),
        "id": task.get("id"),
        "status": task.get("status"),
        "node": task.get("node"),
        "log_path": log_path or task.get("log_path"),
        "source": source,
        "tail": text if ok else "",
        "error": "" if ok else text,
    }
    if getattr(args, "json", False):
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    if not ok:
        sys.exit(text)
    if not getattr(args, "no_header", False):
        print(
            f"==> {task.get('id')} {task.get('status')} "
            f"{task.get('node') or '?'}:{log_path} <=="
        )
    print(text, end="" if text.endswith("\n") else "\n")


def cmd_show(args, *, deps: ControlCommandDeps) -> None:
    with deps.state_lock():
        task, source = deps.find_task_record(args.id, include_archive=True)
    if not task:
        sys.exit(f"task {args.id} not found in queue or archive")
    print(json.dumps(task, indent=2))
    if source == "archive":
        print("\n# source: archive (terminal task moved out of hot queue.json)")
    if task.get("log_path") and task.get("node"):
        node = task["node"]
        host = deps.node_configs.get(node, {}).get("host") or "local"
        if deps.node_is_windows(node):
            cmd = deps.ssh_base_args(node) + [
                "powershell",
                "Get-Content",
                "-Tail",
                "80",
                "-Wait",
                "-LiteralPath",
                task["log_path"],
            ]
            print(f"\n# tail log: {' '.join(shlex.quote(item) for item in cmd)}")
        elif host == "local":
            print(f"\n# tail log: tail -f {shlex.quote(task['log_path'])}")
        else:
            cmd = deps.ssh_base_args(node) + [f"tail -f {shlex.quote(task['log_path'])}"]
            print(f"\n# tail log: {' '.join(shlex.quote(item) for item in cmd)}")
    deps.print_result_artifacts(task, include_log=True)


def cmd_priority(args, *, deps: ControlCommandDeps) -> None:
    """Change priority on a queued task without cancel+resubmit."""
    new = args.level
    with deps.state_lock():
        state = deps.load_state()
        for task in state["tasks"]:
            if task["id"] != args.id:
                continue
            if task.get("status") != "queued":
                sys.exit(
                    f"task {args.id} is {task.get('status')!r}, not queued; "
                    f"priority only affects queue ordering"
                )
            old = task.get("priority", "normal")
            if old == new:
                print(f"{args.id} priority already {new!r}")
                return
            task["priority"] = new
            deps.save_state(state)
            print(
                f"{args.id}: priority {old!r} -> {new!r}  "
                f"({(task.get('description') or '')[:60]})"
            )
            return
        sys.exit(f"task {args.id} not found")
