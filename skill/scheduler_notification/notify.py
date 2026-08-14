from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


TASK_EVENT_PAYLOAD_KEYS = (
    "id", "status", "description", "project", "signature", "cmd", "cwd",
    "origin", "submitted_by", "submitted_host", "scheduler_id",
    "process_owner", "shared_account_suspect",
    "cpu_parallel_items", "cpu_parallel_total_items", "cpu_parallel_start",
    "cpu_parallel_logical_items", "cpu_parallel_item_multiplier",
    "cpu_parallel_end", "cpu_auto_workers", "cpu_parallel_waves",
    "cpu_parallel_physical_cores", "cpu_parallel_shard_index",
    "cpu_parallel_num_shards", "cpu_parallel_total_physical_cores",
    "cpu_parallel_last_wave_items", "cpu_batch_plan",
    "est_vram_mb", "ram_mb", "cpu_cores", "priority", "preferred_node",
    "require_node", "require_gpu_idx", "allow_gpu_over_one_third",
    "allowed_nodes", "stage_excludes", "reroute_on_node_down",
    "node_down_requeue_s", "node", "gpu_idx", "remote_pids", "slurm_job_id",
    "slurm_state", "log_path", "submitted_at", "started_at", "finished_at",
    "peak_vram_mb", "peak_ram_mb", "current_vram_mb", "current_ram_mb",
    "current_pcpu", "resume_from", "eta_seconds", "eta_source",
    "eta_confidence", "runtime_current_unit", "runtime_total_units",
    "placement_algorithm", "placement_algorithm_config", "placement_algorithm_audit",
    "last_progress_line", "result_artifacts", "result_artifacts_discovered_at",
    "parent_id", "retry_count", "requeued_as", "launch_fail_count",
    "failure_category", "last_block_reason", "notified_done", "notified_launch",
    "cancelled_by", "cancel_reason", "last_killed_by", "last_kill_action",
    "last_kill_reason", "node_probe_state", "probe_unknown_since",
    "last_probe_unknown_at", "last_probe_unknown_duration_s",
    "last_probe_unknown_count", "last_probe_unknown_reason",
    "last_reconnected_at", "last_status_sync_at", "last_status_sync_status",
    "last_status_sync_reason",
)


@dataclass(frozen=True)
class NotifyDeps:
    watcher_log: Path
    watcher_log_max_mb: int
    watcher_log_generations: int
    feishu_config: Path
    maybe_rotate_log: Callable[[Path, int, int], object]
    format_task_location: Callable[[dict], str]
    format_mem_gb: Callable[[int], str]
    max_launch_retry: int
    now: Callable[[], float] = time.time
    request_factory: Callable = urllib.request.Request
    urlopen: Callable = urllib.request.urlopen


def load_feishu_cfg(config_path: Path) -> dict | None:
    if not config_path.exists():
        return None
    try:
        cfg = json.loads(config_path.read_text())
    except Exception:
        return None
    if cfg.get("mode") != "push":
        return None
    if not cfg.get("webhook_url"):
        return None
    return cfg


def send_feishu(webhook_url: str, text: str, *, deps: NotifyDeps) -> None:
    payload = json.dumps({"msg_type": "text", "content": {"text": text}}).encode()
    req = deps.request_factory(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with deps.urlopen(req, timeout=5) as response:
            response.read()
    except Exception as exc:
        try:
            with open(deps.watcher_log, "a") as handle:
                handle.write(json.dumps({
                    "ts": deps.now(),
                    "type": "feishu_send_failed",
                    "error": str(exc)[:200],
                }) + "\n")
        except Exception:
            pass


def format_feishu(event_type: str, payload: dict, *, deps: NotifyDeps) -> str:
    if event_type == "task_done":
        task = payload
        runtime_min = (task.get("finished_at", deps.now()) - task.get("started_at", deps.now())) / 60
        return (f"[scheduler] ✅ {task['id']} {task.get('project','?')} done — "
                f"{runtime_min:.1f}m, peak {task.get('peak_vram_mb', 0)}MB on {deps.format_task_location(task)} | "
                f"{task.get('description','')[:60]}")
    if event_type == "task_launched":
        task = payload
        handle = (
            f"slurm_job_id={task['slurm_job_id']}"
            if task.get("slurm_job_id")
            else f"pid={(task.get('remote_pids') or ['?'])[0]}"
        )
        return (f"[scheduler] 🚀 {task['id']} {task.get('project','?')} launched on "
                f"{deps.format_task_location(task)} {handle} | {task.get('description','')[:60]}")
    if event_type == "task_auto_adopted":
        task = payload
        return (f"[scheduler] 👁 auto-adopted {task['id']} {task.get('project','?')} on "
                f"{deps.format_task_location(task)} ({len(task.get('remote_pids', []))} procs, {task.get('peak_vram_mb', 0)}MB) — "
                f"launched outside scheduler, now tracked")
    if event_type == "heartbeat":
        return (f"[scheduler] ⏱ heartbeat — running:{payload['running']} "
                f"launching:{payload.get('launching', 0)} queued:{payload['queued']} | "
                + " ; ".join(payload["nodes"]))
    if event_type == "watcher_started":
        return f"[scheduler] watcher started (pid={payload['pid']}, interval={payload['interval']}s, heartbeat={payload['heartbeat']}s)"
    if event_type == "watcher_stopped":
        return f"[scheduler] watcher stopped (pid={payload['pid']})"
    if event_type == "task_blocked":
        return f"[scheduler] ⚠ {payload['task_id']} blocked: {payload['reason']}"
    if event_type == "task_migrated":
        return (f"[scheduler] 🔀 {payload['task_id']} re-pinned "
                f"{payload.get('from_node') or '?'} → {payload.get('to_node') or '?'} "
                f"(eta={payload.get('eta_seconds', 0)}s, load-balance)")
    if event_type == "task_preempted":
        return (f"[scheduler] ⚡ {payload['task_id']} preempted on "
                f"{payload.get('freed_node') or '?'} "
                f"(freed {payload.get('cpu_freed', 0)}c/{deps.format_mem_gb(payload.get('ram_freed', 0))})")
    if event_type == "task_killed":
        actor = payload.get("actor") or {}
        return (f"[scheduler] kill {payload.get('task_id','?')} by "
                f"{actor.get('label') or '?'} action={payload.get('action','?')} "
                f"node={payload.get('node','?')} reason={str(payload.get('reason',''))[:120]}")
    if event_type == "task_launch_retry":
        return (f"[scheduler] ⚠ {payload['task_id']} launch failed "
                f"({payload.get('attempt','?')}/{deps.max_launch_retry}); will retry/fallback: "
                f"{payload['error'][:120]}")
    if event_type == "task_failed":
        return f"[scheduler] ❌ {payload['task_id']} launch failed: {payload['error'][:120]}"
    if event_type == "task_crashed":
        crash = payload
        tail_short = crash.get("tail", "").replace("\n", " | ")[-200:]
        return (f"[scheduler] 💥 {crash['id']} {crash.get('project','?')} CRASHED on {crash.get('node')}:GPU{crash.get('gpu_idx')} — "
                f"died after {crash.get('lifetime_s', 0)}s, log {crash.get('log_size', 0)}B. "
                f"reason: {crash.get('reason','?')[:120]}. tail: {tail_short}")
    if event_type == "heal_awaiting_claude":
        payload = payload
        return (f"[scheduler] 🤖 heal needs Claude — {payload.get('prefix','?')} ({payload.get('category','?')}) on {payload.get('node','?')}: "
                f"{payload.get('question','need decision')[:200]} | inbox: ~/.claude/scheduler/HEAL_NEEDS_CLAUDE.md")
    if event_type == "heal_awaiting_user":
        payload = payload
        return (f"[scheduler] 🆘 heal AWAITING_USER (Claude couldn't decide) — {payload.get('prefix','?')} ({payload.get('category','?')}) on {payload.get('node','?')}: "
                f"{payload.get('question','need decision')[:200]} | see ~/.claude/scheduler/HEAL_NEEDS_USER.md")
    if event_type == "batch_complete":
        batch = payload
        elapsed_min = (batch.get("latest_finish", 0) - batch.get("earliest_submit", 0)) / 60
        bits = []
        if batch.get("done"):
            bits.append(f"✅{batch['done']}")
        if batch.get("failed"):
            bits.append(f"💥{batch['failed']}")
        if batch.get("cancelled"):
            bits.append(f"🚫{batch['cancelled']}")
        return (f"[scheduler] 🏁 batch '{batch['prefix']}' complete — {batch['total']} tasks ({' '.join(bits)}) "
                f"in {elapsed_min:.1f}m | ids: {','.join(batch.get('task_ids', [])[:5])}"
                + ("..." if len(batch.get('task_ids', [])) > 5 else ""))
    return f"[scheduler] {event_type}: {json.dumps(payload)[:200]}"


def compact_task_event_payload(task: dict) -> dict:
    out = {key: task.get(key) for key in TASK_EVENT_PAYLOAD_KEYS if key in task}
    diag = task.get("_diagnosis")
    if isinstance(diag, dict):
        out["_diagnosis"] = {
            key: diag.get(key)
            for key in ("is_crash", "reason", "lifetime_s", "log_size", "log_path", "success_marker")
            if key in diag
        }
    return out


def notify(event_type: str, payload: dict, *, feishu_enabled: bool = True,
           deps: NotifyDeps) -> None:
    line = json.dumps({"ts": deps.now(), "type": event_type, "payload": payload}, default=str)
    try:
        deps.watcher_log.parent.mkdir(parents=True, exist_ok=True)
        deps.maybe_rotate_log(deps.watcher_log, deps.watcher_log_max_mb, deps.watcher_log_generations)
        with open(deps.watcher_log, "a") as handle:
            handle.write(line + "\n")
    except Exception:
        pass
    if not feishu_enabled:
        return
    cfg = load_feishu_cfg(deps.feishu_config)
    if not cfg:
        return
    send_feishu(cfg["webhook_url"], format_feishu(event_type, payload, deps=deps), deps=deps)
