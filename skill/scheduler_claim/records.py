from __future__ import annotations

from typing import Callable, Optional


def claim_intent_nodes(task: dict) -> list:
    nodes = []
    for key in ("claim_intent_node",):
        node = task.get(key)
        if node and node not in nodes:
            nodes.append(node)
    raw = task.get("claim_intent_nodes") or []
    if isinstance(raw, str):
        raw = [raw]
    if isinstance(raw, list):
        for node in raw:
            if node and node not in nodes:
                nodes.append(node)
    return nodes


def remember_claim_intent(
    task: dict,
    node: Optional[str],
    *,
    claim_enabled_for: Callable[[str], bool],
    now: Callable[[], float],
) -> None:
    if not node or not claim_enabled_for(node):
        return
    nodes = claim_intent_nodes(task)
    if node not in nodes:
        nodes.append(node)
    task["claim_intent_nodes"] = nodes
    task["claim_intent_at"] = now()


def clear_claim_intent_markers(task: dict) -> None:
    task.pop("claim_intent_node", None)
    task.pop("claim_intent_nodes", None)
    task.pop("claim_intent_at", None)


def claim_resource_record_for_task(
    task: dict,
    *,
    node_configs: dict,
    ignore_cpu_for_server_gpu_task: Callable[..., bool],
    task_ignores_one_third_pack_rule: Callable[[dict, Optional[dict]], bool],
    default_ram_mb: int,
    default_cpu_cores: int,
    startup_floor_mb: int,
) -> dict:
    """Current claim budget for a running task."""
    pids = task.get("remote_pids") or []
    current_vram = int(task.get("current_vram_mb") or 0)
    peak_vram = int(task.get("peak_vram_mb") or 0)
    est_vram = int(task.get("est_vram_mb") or 0)
    current_ram = int(task.get("current_ram_mb") or 0)
    est_ram = int(task.get("ram_mb") or default_ram_mb)
    node = task.get("node")
    node_info = node_configs.get(node or "", {})
    ignore_cpu = ignore_cpu_for_server_gpu_task(
        task,
        node_info=node_info,
        node_name=node,
        gpu_idx=task.get("gpu_idx"),
    )
    ignore_one_third = task_ignores_one_third_pack_rule(task, node_info)
    observed_vram = max(0, current_vram, peak_vram)
    if observed_vram >= 100:
        vram_budget = observed_vram
    elif task.get("last_progress_line") or int(task.get("runtime_current_unit") or 0) > 0:
        vram_budget = observed_vram
    elif est_vram > 0:
        vram_budget = min(est_vram, startup_floor_mb)
    else:
        vram_budget = observed_vram
    return {
        "gpu_idx": task.get("gpu_idx"),
        "vram_mb": vram_budget,
        "cpu_cores": 0 if ignore_cpu else max(0, int(task.get("cpu_cores") or default_cpu_cores)),
        "ram_mb": max(0, est_ram, current_ram),
        "ignore_cpu_capacity": bool(ignore_cpu),
        "ignore_one_third_pack_rule": bool(ignore_one_third),
        "pid": int(pids[0]) if pids else None,
    }


def release_task_claims_and_intents(
    task: dict,
    *,
    claim_enabled_for: Callable[[str], bool],
    claim_release: Callable[[str, str], bool],
    extra_nodes=None,
    exclude_nodes=None,
    clear_markers: bool = True,
) -> int:
    """Best-effort release of this scheduler's claim and pending FIFO intents."""
    exclude = set(exclude_nodes or [])
    nodes = []
    for key in ("node", "last_node"):
        if task.get(key):
            nodes.append(task.get(key))
    nodes.extend(claim_intent_nodes(task))
    if extra_nodes:
        nodes.extend(extra_nodes if isinstance(extra_nodes, (list, tuple, set)) else [extra_nodes])

    seen = set()
    released = 0
    released_nodes = set()
    for node in nodes:
        if not node or node in seen or node in exclude:
            continue
        seen.add(node)
        if not claim_enabled_for(node):
            continue
        try:
            if claim_release(node, task["id"]):
                released += 1
                released_nodes.add(node)
        except Exception:
            pass

    if clear_markers:
        clear_claim_intent_markers(task)
    elif released_nodes:
        kept = [node for node in claim_intent_nodes(task) if node not in released_nodes]
        if kept:
            task["claim_intent_nodes"] = kept
        else:
            clear_claim_intent_markers(task)
    return released
