"""Task eviction and relaunch-cooldown helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class EvictionDeps:
    release_task_claims_and_intents: Callable[[dict], Any]
    task_pids: Callable[[dict], list[int]]
    record_task_kill_actor: Callable[[dict, str, str], dict]
    kill_task_processes: Callable[..., tuple[bool, str]]
    notify: Callable[..., Any]
    set_current_usage: Callable[[dict, int, int, float], Any]
    clear_live_eta_fields: Callable[..., Any]
    local_cpu_pressure_high: Callable[[Optional[dict], int], tuple[bool, str]]
    evict_relaunch_cooldown_s: int
    local_cpu_evict_node_cooldown_s: int
    local_gpu_host_cpu_block_pct: int
    now: Callable[[], float]


def evict_to_queue(
    victim: dict,
    state: dict,
    reason: str,
    eviction_kind: str = "preempt",
    *,
    deps: EvictionDeps,
) -> None:
    """Send a running task back to queued without bumping crash retry counters."""
    try:
        deps.release_task_claims_and_intents(victim)
    except Exception:
        pass
    old_node = victim.get("node")
    old_pids = list(deps.task_pids(victim))
    actor = deps.record_task_kill_actor(victim, eviction_kind, reason)
    ok, kill_msg = deps.kill_task_processes(victim, timeout=15)
    deps.notify(
        "task_killed",
        {
            "task_id": victim.get("id"),
            "node": old_node,
            "pids": old_pids,
            "actor": actor,
            "action": eviction_kind,
            "reason": reason,
            "kill_ok": ok,
            "kill_msg": kill_msg,
        },
        feishu_enabled=False,
    )
    victim["status"] = "queued"
    for key in ("node", "gpu_idx", "process_group", "log_path", "started_at", "finished_at", "_diagnosis"):
        victim[key] = None
    victim["remote_pids"] = []
    victim["alive_pids"] = []
    victim["peak_vram_mb"] = 0
    victim["peak_ram_mb"] = 0
    deps.set_current_usage(victim, 0, 0, 0.0)
    deps.clear_live_eta_fields(victim, clear_runtime_projection=True)
    victim["notified_done"] = False
    victim["last_block_reason"] = reason
    now = deps.now()
    victim["last_evicted_at"] = now
    victim["last_eviction_kind"] = eviction_kind
    victim.pop("evict_cooldown_until", None)
    if eviction_kind == "local_cpu_budget":
        cooldown = max(0, int(deps.local_cpu_evict_node_cooldown_s))
    else:
        cooldown = max(0, int(deps.evict_relaunch_cooldown_s))
    if old_node and cooldown > 0:
        raw = victim.get("evict_node_cooldowns")
        if not isinstance(raw, dict):
            raw = {}
        raw[str(old_node)] = now + cooldown
        victim["evict_node_cooldowns"] = raw


def eviction_cooldown_block_reason(task: dict, *, now: Optional[float] = None, deps: EvictionDeps) -> str:
    until = float(task.get("evict_cooldown_until") or 0)
    if until <= 0:
        return ""
    now = deps.now() if now is None else now
    if until <= now:
        task.pop("evict_cooldown_until", None)
        return ""
    ev = task.get("last_resource_eviction")
    evicted_node = ev.get("node") if isinstance(ev, dict) else None
    if evicted_node:
        raw = task.get("evict_node_cooldowns")
        if not isinstance(raw, dict):
            raw = {}
        raw.setdefault(str(evicted_node), until)
        task["evict_node_cooldowns"] = raw
        task.pop("evict_cooldown_until", None)
        return ""
    remain = int(max(0, until - now))
    return (
        f"recently {task.get('last_eviction_kind') or 'evicted'}; "
        f"cooling down {remain}s before relaunch to avoid kill/relaunch thrash"
    )


def evict_node_cooldown_block_reason(
    task: dict,
    node: str,
    *,
    now: Optional[float] = None,
    node_state: Optional[dict] = None,
    deps: EvictionDeps,
) -> str:
    """Block only the node that just evicted this task."""
    if not node:
        return ""
    now = deps.now() if now is None else now
    raw = task.get("evict_node_cooldowns")
    if not isinstance(raw, dict):
        raw = {}
    try:
        until = float(raw.get(node) or 0)
    except Exception:
        until = 0.0
    if until <= 0 and node == "local" and task.get("last_eviction_kind") == "local_cpu_budget":
        try:
            last = float(task.get("last_evicted_at") or 0)
        except Exception:
            last = 0.0
        if last > 0:
            until = last + max(0, int(deps.local_cpu_evict_node_cooldown_s))
            if until > now:
                raw[node] = until
                task["evict_node_cooldowns"] = raw
    if (
        until > now
        and node == "local"
        and task.get("last_eviction_kind") == "local_cpu_budget"
        and node_state is not None
    ):
        high, _ = deps.local_cpu_pressure_high(
            node_state,
            deps.local_gpu_host_cpu_block_pct,
        )
        if not high:
            raw.pop(node, None)
            if raw:
                task["evict_node_cooldowns"] = raw
            else:
                task.pop("evict_node_cooldowns", None)
            return ""
    if until <= 0:
        return ""
    if until <= now:
        if raw:
            raw.pop(node, None)
            if raw:
                task["evict_node_cooldowns"] = raw
            else:
                task.pop("evict_node_cooldowns", None)
        return ""
    remain = int(max(0, until - now))
    return (
        f"recently {task.get('last_eviction_kind') or 'evicted'} on {node}; "
        f"cooling down {remain}s before relaunching there"
    )
