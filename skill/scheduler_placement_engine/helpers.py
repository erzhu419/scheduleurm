from __future__ import annotations

from typing import Callable, Optional


def node_ram_headroom_mb(node_state: dict, node_info: dict, *, default_frac: float) -> int:
    fixed_headroom = node_info.get("ram_headroom_mb")
    if fixed_headroom is not None:
        return max(0, int(fixed_headroom))
    frac = node_info.get("ram_headroom_frac", default_frac)
    total_for_headroom = node_state.get("total_ram_mb") or node_info.get("ram_mb", 0)
    return int(total_for_headroom * frac)


def gpu_freeze_line_mb(total_mb: int, *, one_third_pack_grace_mb: int) -> int:
    if total_mb <= 0:
        return 0
    return min(total_mb, total_mb // 3 + max(0, int(one_third_pack_grace_mb)))


def gpu_threshold_snapshot(
    gpu: Optional[dict],
    *,
    gpu_freeze_line_mb: Callable[[int], int],
) -> Optional[dict]:
    if not gpu:
        return None
    total = int(gpu.get("total_mb") or 0)
    used = int(gpu.get("used_mb") or 0)
    free = int(gpu.get("free_mb") or max(0, total - used))
    third = total // 3 if total > 0 else 0
    freeze = gpu_freeze_line_mb(total)
    return {
        "idx": int(gpu.get("idx") or 0),
        "used_mb": used,
        "free_mb": free,
        "total_mb": total,
        "util_pct": int(gpu.get("util_pct") or 0),
        "one_third_mb": third,
        "freeze_line_mb": freeze,
        "over_one_third": bool(total > 0 and used >= third and used > 100),
        "over_freeze_line": bool(total > 0 and used >= freeze and used > 100),
        "over_full": bool(total > 0 and used >= total),
        "used_pct": int(round(100 * used / max(total, 1))),
    }


def node_ram_snapshot(
    node_state: Optional[dict],
    *,
    node_configs: dict,
    node_ram_headroom_mb: Callable[[dict, dict], int],
    ram_headroom_eviction_grace_mb: int,
) -> Optional[dict]:
    if not node_state:
        return None
    node_info = node_configs.get(node_state.get("name"), {})
    total = int(node_state.get("total_ram_mb") or node_info.get("ram_mb") or 0)
    free = int(node_state.get("free_ram_mb") or 0)
    headroom = node_ram_headroom_mb(node_state, node_info)
    grace = max(0, int(ram_headroom_eviction_grace_mb))
    eviction_headroom = max(0, headroom - grace)
    return {
        "free_mb": free,
        "total_mb": total,
        "headroom_mb": headroom,
        "eviction_headroom_mb": eviction_headroom,
        "headroom_grace_mb": grace,
        "below_headroom": bool(free < headroom),
        "below_eviction_headroom": bool(free < eviction_headroom),
        "available_after_headroom_mb": free - headroom,
        "available_after_eviction_headroom_mb": free - eviction_headroom,
        "headroom_gap_mb": max(0, headroom - free),
        "eviction_gap_mb": max(0, eviction_headroom - free),
    }


def assign_windows_pin_plan(
    task: dict,
    state: dict,
    node_state: Optional[dict],
    *,
    node_is_windows: Callable[[str], bool],
    node_physical_cores: Callable[[Optional[str], Optional[dict]], int],
    node_configs: dict,
    default_cpu_cores: int,
) -> None:
    node = task.get("node")
    if not node or not node_is_windows(node):
        task.pop("windows_pin_base", None)
        task.pop("windows_pin_cores", None)
        return
    total = max(1, int(node_physical_cores(node, node_state) or node_configs.get(node, {}).get("cpu_cores") or 1))
    width = max(1, min(total, int(task.get("cpu_cores") or default_cpu_cores or 1)))
    occupied = [False] * total
    legacy_cursor = 0
    running = [
        other for other in state.get("tasks", [])
        if other is not task
        and other.get("status") in ("running", "launching")
        and (other.get("node") or other.get("assigned_node")) == node
    ]
    running.sort(key=lambda other: other.get("started_at") or other.get("submitted_at") or 0)
    for other in running:
        other_width = max(1, min(total, int(other.get("windows_pin_cores")
                                            or other.get("cpu_cores")
                                            or default_cpu_cores
                                            or 1)))
        base = other.get("windows_pin_base")
        if base is None:
            base = legacy_cursor
            legacy_cursor += other_width
        try:
            base = int(base)
        except Exception:
            continue
        for idx in range(other_width):
            occupied[(base + idx) % total] = True

    def fits(start: int) -> bool:
        return all(not occupied[(start + idx) % total] for idx in range(width))

    base = None
    for start in range(total):
        if fits(start):
            base = start
            break
    if base is None:
        base = min(total - 1, legacy_cursor % total)
    task["windows_pin_base"] = int(base)
    task["windows_pin_cores"] = int(width)


def task_is_gpu_capacity_task(task: dict, *, default_vram_mb: int) -> bool:
    try:
        need_vram = int(task.get("est_vram_mb", default_vram_mb) or 0)
    except Exception:
        need_vram = int(default_vram_mb or 0)
    return need_vram > 0 and not bool(task.get("cpu_fallback_selected"))


def flag_enabled(value, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() not in ("", "0", "false", "no", "off", "none")
    return bool(value)


def ignore_cpu_for_server_gpu_task(
    task: dict,
    node_state: Optional[dict] = None,
    node_info: Optional[dict] = None,
    node_name: Optional[str] = None,
    gpu_idx: Optional[int] = None,
    *,
    task_is_gpu_capacity_task: Callable[[dict], bool],
    node_is_windows: Callable[[str], bool],
    node_configs: dict,
) -> bool:
    """Remote Linux GPU jobs are GPU/RAM placed; CPU pressure only gates local and CPU-only work."""
    if not task_is_gpu_capacity_task(task):
        return False
    node_name = node_name or (node_state or {}).get("name") or task.get("node")
    if not node_name or node_name == "local" or node_is_windows(node_name):
        return False
    info = node_info if node_info is not None else node_configs.get(node_name, {})
    if not (info or {}).get("host"):
        return False
    if (info or {}).get("max_vram_per_task") == 0:
        return False
    if node_state is not None:
        if not node_state.get("gpus"):
            return False
    elif gpu_idx is None and task.get("gpu_idx") is None:
        return False
    return flag_enabled((info or {}).get("ignore_cpu_for_gpu_tasks"), True)


def task_ignores_one_third_pack_rule(
    task: dict,
    node_info: Optional[dict] = None,
    *,
    hard_rule_bypassed: Callable[..., bool],
) -> bool:
    if hard_rule_bypassed("one_third_pack", task, node_info=node_info):
        return True
    if flag_enabled((task or {}).get("allow_gpu_over_one_third"), False):
        return True
    if str((task or {}).get("signature") or "").startswith("BAPR/bus_v2/"):
        return True
    return flag_enabled((node_info or {}).get("allow_gpu_over_one_third"), False)


def node_gpu_util_limit(node_info, *, default_limit: int):
    util_limit = (node_info or {}).get("gpu_util_saturation_pct", default_limit)
    if isinstance(util_limit, str) and util_limit.lower() in ("", "none", "off", "false", "ignore"):
        return None
    if util_limit is None:
        return None
    return int(util_limit)


def clear_disallowed_cpu_fallback_selection(task: dict) -> bool:
    """Drop stale CPU fallback placement left by older watcher versions."""
    if int(task.get("est_vram_mb") or 0) <= 0:
        return False
    if not task.get("cpu_fallback_selected"):
        return False
    if bool(task.get("allow_cpu_training", False)):
        return False
    task.pop("cpu_fallback_selected", None)
    task.pop("cpu_fallback_original_vram_mb", None)
    task.pop("cpu_fallback_capability", None)
    return True


def task_launch_cpu_mode(task: dict) -> bool:
    if int(task.get("est_vram_mb") or 0) <= 0:
        return True
    return bool(task.get("cpu_fallback_selected")) and bool(task.get("allow_cpu_training", False))
