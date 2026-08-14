"""Post-dispatch resource-pressure rollback policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class PostDispatchThresholdDeps:
    node_configs: dict
    default_cpu_cores: int
    gpu_evict_stable_progress_min_age_s: int
    gpu_evict_rollback_max_age_s: int
    local_gpu_host_cpu_evict_pct: int
    effective_elapsed_s: Callable[[dict], float]
    task_has_progress_evidence: Callable[[dict], bool]
    task_progress_ratio: Callable[[dict], float | None]
    task_ram_pressure_mb: Callable[[dict], int]
    is_slurm_managed: Callable[[dict], bool]
    task_evict_loss_protected: Callable[[dict], bool]
    ckpt_task_evict_protected: Callable[[dict], bool]
    task_ignores_one_third_pack_rule: Callable[[dict, dict], bool]
    local_cpu_pressure_high: Callable[[dict, int], tuple[bool, str]]
    gpu_freeze_line_mb: Callable[[int], int]
    gpu_threshold_snapshot: Callable[[dict], dict]
    summarize_task_for_resource_log: Callable[[dict], dict]
    node_ram_snapshot: Callable[[dict], dict | None]
    format_mem_gb: Callable[[int], str]
    evict_to_queue: Callable[[dict, dict, str, str], Any]
    notify: Callable[..., Any]
    now: Callable[[], float]


def _gpu_evict_candidate(task: dict, deps: PostDispatchThresholdDeps) -> bool:
    elapsed = deps.effective_elapsed_s(task)
    if (
        elapsed >= max(0, int(deps.gpu_evict_stable_progress_min_age_s))
        and deps.task_has_progress_evidence(task)
    ):
        return False
    if elapsed <= max(0, int(deps.gpu_evict_rollback_max_age_s)):
        return True
    return True


def _gpu_victim_key(task: dict, deps: PostDispatchThresholdDeps) -> tuple:
    eta = int(task.get("eta_seconds") or 0)
    eta_rank = eta if eta > 0 else 10 ** 12
    progress = deps.task_progress_ratio(task)
    progress_rank = -(progress if progress is not None else 0.0)
    no_progress_rank = 0 if deps.task_has_progress_evidence(task) else 1
    return (float(task.get("started_at") or 0), no_progress_rank, progress_rank, eta_rank)


def _ram_victim_key(task: dict, deps: PostDispatchThresholdDeps) -> tuple:
    progress = deps.task_progress_ratio(task)
    progress_rank = -(progress if progress is not None else 0.0)
    eta = int(task.get("eta_seconds") or 0)
    eta_rank = eta if eta > 0 else 0
    return (deps.task_ram_pressure_mb(task), progress_rank, float(task.get("started_at") or 0), eta_rank)


def _cpu_victim_key(task: dict, deps: PostDispatchThresholdDeps) -> tuple:
    progress = deps.task_progress_ratio(task)
    progress_rank = -(progress if progress is not None else 0.0)
    no_progress_rank = 0 if deps.task_has_progress_evidence(task) else 1
    return (
        no_progress_rank,
        progress_rank,
        float(task.get("started_at") or 0),
        max(0, int(task.get("cpu_cores") or deps.default_cpu_cores)),
    )


def _eligible_running(
    state: dict,
    evicted_set: set,
    where: Callable[[dict], bool],
    deps: PostDispatchThresholdDeps,
) -> list:
    return [
        task for task in state["tasks"]
        if task.get("status") == "running"
        and task.get("started_at")
        and not task.get("auto_adopted")
        and not deps.is_slurm_managed(task)
        and task.get("id") not in evicted_set
        and where(task)
    ]


def _evict(
    victim: dict,
    state: dict,
    reason: str,
    payload: dict,
    evicted: list,
    evicted_set: set,
    deps: PostDispatchThresholdDeps,
) -> None:
    if payload.get("kind") == "gpu_one_third":
        observed_vram = max(
            int(victim.get("current_vram_mb") or 0),
            int(victim.get("peak_vram_mb") or 0),
        )
        for rec in payload.get("same_gpu_tasks") or []:
            if rec.get("id") == victim.get("id"):
                observed_vram = max(
                    observed_vram,
                    int(rec.get("current_vram_mb") or 0),
                    int(rec.get("peak_vram_mb") or 0),
                )
        cur_est = int(victim.get("est_vram_mb") or 0)
        if observed_vram >= 100 and observed_vram > cur_est:
            new_est = max(observed_vram + 128, int(observed_vram * 1.15))
            victim["est_vram_mb"] = new_est
            victim["est_vram_mb_adjusted_by_eviction"] = {
                "kind": payload.get("kind"),
                "old_est_vram_mb": cur_est,
                "observed_vram_mb": observed_vram,
                "new_est_vram_mb": new_est,
                "ts": deps.now(),
                "reason": "gpu freeze-line rollback observed higher live VRAM than estimate",
            }
    deps.evict_to_queue(victim, state, reason, payload.get("kind") or "resource_pressure")
    victim["last_resource_eviction"] = payload
    evicted.append(victim["id"])
    evicted_set.add(victim["id"])


def _enforce_local_cpu_budget(
    state: dict,
    node: dict,
    evicted: list,
    evicted_set: set,
    deps: PostDispatchThresholdDeps,
) -> None:
    budget = int(deps.node_configs.get("local", {}).get("cpu_cores") or node.get("total_cpu") or 0)
    if budget <= 0:
        return
    tasks_here = _eligible_running(
        state, evicted_set, lambda task: task.get("node") == "local", deps)
    reserved = sum(
        max(0, int(task.get("cpu_cores") or deps.default_cpu_cores))
        for task in tasks_here
    )
    if reserved > budget:
        cpu_high, cpu_why = deps.local_cpu_pressure_high(
            node, deps.local_gpu_host_cpu_evict_pct)
        if not cpu_high:
            deps.notify("local_cpu_over_reserved_no_evict", {
                "reserved": reserved,
                "budget": budget,
                "reason": cpu_why,
                "task_ids": [task.get("id") for task in tasks_here],
            }, feishu_enabled=False)
            return
    while reserved > budget and len(tasks_here) > 1:
        protected = [task for task in tasks_here if deps.task_evict_loss_protected(task)]
        protected_ids = {id(task) for task in protected}
        candidates = [task for task in tasks_here if id(task) not in protected_ids] or tasks_here
        if not candidates:
            break
        victim = max(candidates, key=lambda task: _cpu_victim_key(task, deps))
        payload = {
            "kind": "local_cpu_budget",
            "task_id": victim["id"],
            "node": "local",
            "cpu_reserved": reserved,
            "cpu_budget": budget,
            "same_node_tasks": [deps.summarize_task_for_resource_log(task) for task in tasks_here],
            "protected_evict_loss_task_ids": [task.get("id") for task in protected],
            "selection": "least_progress_then_newest_then_highest_cpu",
        }
        reason = (
            f"evicted from local after CPU budget breach "
            f"(reserved={reserved}/{budget}); "
            f"victim selected by least progress / newest / highest CPU"
        )
        freed = max(0, int(victim.get("cpu_cores") or deps.default_cpu_cores))
        _evict(victim, state, reason, payload, evicted, evicted_set, deps)
        reserved = max(0, reserved - freed)
        tasks_here = [task for task in tasks_here if task.get("id") != victim.get("id")]
        if not tasks_here:
            break


def _enforce_gpu_freeze_line(
    state: dict,
    node: dict,
    gpu: dict,
    evicted: list,
    evicted_set: set,
    deps: PostDispatchThresholdDeps,
) -> None:
    freeze = deps.gpu_freeze_line_mb(int(gpu.get("total_mb") or 0))
    occupied = gpu["used_mb"] > 100
    mem_over = occupied and freeze > 0 and gpu["used_mb"] >= freeze
    if not mem_over:
        return
    tasks_here = _eligible_running(
        state,
        evicted_set,
        lambda task, node_name=node["name"], gpu_idx=gpu["idx"]: (
            task.get("node") == node_name and task.get("gpu_idx") == gpu_idx
        ),
        deps,
    )
    if len(tasks_here) < 2:
        return
    candidates = [
        task for task in tasks_here
        if _gpu_evict_candidate(task, deps)
        and not deps.task_ignores_one_third_pack_rule(
            task, deps.node_configs.get(task.get("node") or "", {}))
    ]
    if not candidates:
        return
    protected = [task for task in tasks_here if task not in candidates]
    victim = max(candidates, key=lambda task: _gpu_victim_key(task, deps))
    payload = {
        "kind": "gpu_one_third",
        "task_id": victim["id"],
        "node": node["name"],
        "gpu_idx": gpu["idx"],
        "gpu": deps.gpu_threshold_snapshot(gpu),
        "same_gpu_tasks": [deps.summarize_task_for_resource_log(task) for task in tasks_here],
        "protected_stable_progress_task_ids": [task.get("id") for task in protected],
        "selection": "newest_launch_rollback_then_no_progress_then_least_progress",
    }
    reason = (
        f"evicted from {node['name']}:GPU{gpu['idx']} after GPU memory freeze-line breach "
        f"(mem={deps.format_mem_gb(gpu['used_mb'])}/{deps.format_mem_gb(gpu['total_mb'])}, "
        f"freeze={deps.format_mem_gb(freeze)}, util={gpu.get('util_pct','?')}%); "
        f"victim selected by newest launch rollback / no progress / least progress"
    )
    _evict(victim, state, reason, payload, evicted, evicted_set, deps)


def _enforce_ram_headroom(
    state: dict,
    node: dict,
    evicted: list,
    evicted_set: set,
    deps: PostDispatchThresholdDeps,
) -> None:
    ram = deps.node_ram_snapshot(node)
    if not ram or not ram.get("below_eviction_headroom"):
        return
    tasks_here = _eligible_running(
        state, evicted_set, lambda task, node_name=node["name"]: task.get("node") == node_name, deps)
    if len(tasks_here) < 2:
        return
    protected = [task for task in tasks_here if deps.task_evict_loss_protected(task)]
    protected_ids = {id(task) for task in protected}
    candidates = [task for task in tasks_here if id(task) not in protected_ids] or tasks_here
    victim = max(candidates, key=lambda task: _ram_victim_key(task, deps))
    payload = {
        "kind": "node_ram_headroom",
        "task_id": victim["id"],
        "node": node["name"],
        "node_ram": ram,
        "same_node_tasks": [deps.summarize_task_for_resource_log(task) for task in tasks_here],
        "protected_evict_loss_task_ids": [task.get("id") for task in protected],
        "protected_ckpt_task_ids": [
            task.get("id") for task in protected if deps.ckpt_task_evict_protected(task)
        ],
        "selection": "highest_ram_pressure_then_least_progress_then_newest_loss_aware",
    }
    reason = (
        f"evicted from {node['name']} after RAM headroom breach "
        f"(free={deps.format_mem_gb(ram['free_mb'])} < "
        f"eviction_headroom={deps.format_mem_gb(ram['eviction_headroom_mb'])}, "
        f"headroom={deps.format_mem_gb(ram['headroom_mb'])}, "
        f"grace={deps.format_mem_gb(ram['headroom_grace_mb'])}); "
        f"victim selected by RAM pressure / least progress / newest with loss protection"
    )
    _evict(victim, state, reason, payload, evicted, evicted_set, deps)


def enforce_post_dispatch_thresholds(
    state: dict,
    nodes: list,
    *,
    deps: PostDispatchThresholdDeps,
) -> list:
    """Requeue scheduler-owned work when post-dispatch CPU/GPU/RAM thresholds breach."""
    evicted = []
    evicted_set = set()
    for node in nodes:
        if not node.get("alive"):
            continue
        if node.get("measurement_reservation_active"):
            continue
        if node.get("name") == "local":
            _enforce_local_cpu_budget(state, node, evicted, evicted_set, deps)
        for gpu in node["gpus"]:
            _enforce_gpu_freeze_line(state, node, gpu, evicted, evicted_set, deps)
        _enforce_ram_headroom(state, node, evicted, evicted_set, deps)
    return evicted
