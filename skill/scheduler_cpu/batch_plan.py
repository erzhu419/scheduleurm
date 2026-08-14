"""CPU batch shard planning and audit payload helpers."""

from __future__ import annotations

from typing import Optional

from .capacity import (
    ceil_div,
    cpu_max_workers_per_process,
    cpu_planning_cores_for_node,
)


def cpu_worker_plan_for_items(total_items: int, physical_cores: int) -> dict:
    """Return the preferred worker/wave plan for M independent CPU items."""
    items = max(0, int(total_items or 0))
    physical = max(1, int(physical_cores or 1))
    if items <= 0:
        return {
            "items": 0,
            "physical_cores": physical,
            "waves": 0,
            "workers": 0,
            "last_wave_items": 0,
        }
    waves = ceil_div(items, physical)
    workers = min(physical, ceil_div(items, waves))
    last = items - workers * max(0, waves - 1)
    if last <= 0:
        last = workers
    return {
        "items": items,
        "physical_cores": physical,
        "waves": waves,
        "workers": workers,
        "last_wave_items": last,
    }


def cpu_batch_plan(
    total_items: int,
    node_names: list[str],
    *,
    node_states: Optional[dict] = None,
    node_info_by_name: Optional[dict] = None,
    default_cpu_cores: int = 1,
    windows_cpu_max_workers_per_process: int = 0,
) -> list[dict]:
    """Split M independent CPU items across CPU-labor nodes by free physical cores."""
    items_total = max(0, int(total_items or 0))
    if items_total <= 0:
        return []
    states = node_states or {}
    node_infos = node_info_by_name or {}
    caps = []
    for name in list(node_names or []):
        avail, total = cpu_planning_cores_for_node(
            name,
            node_state=states.get(name),
            node_info=node_infos.get(name, {}),
            default_cpu_cores=default_cpu_cores,
        )
        if avail > 0:
            caps.append((name, avail, total))
    if not caps:
        return []
    total_cap = sum(cap for _, cap, _total in caps)
    global_waves = ceil_div(items_total, total_cap)
    raw = []
    assigned = 0
    for idx, (name, cap, total) in enumerate(caps):
        share = items_total * cap / float(total_cap)
        base = int(share)
        raw.append([name, cap, total, base, share - base, idx])
        assigned += base
    remaining_items = items_total - assigned
    raw.sort(key=lambda row: (-row[4], row[5]))
    for idx in range(remaining_items):
        raw[idx % len(raw)][3] += 1
    raw.sort(key=lambda row: row[5])

    out: list[dict] = []
    cursor = 0
    node_shard_idx = 0
    for name, cap, total, items, _frac, _idx in [row for row in raw if row[3] > 0]:
        start = cursor
        end = start + items
        cursor = end
        per_node = cpu_worker_plan_for_items(items, cap)
        workers = min(cap, ceil_div(items, global_waves))
        waves = ceil_div(items, workers) if workers else 0
        last = items - workers * max(0, waves - 1)
        if last <= 0 and items > 0:
            last = workers
        base = {
            "node": name,
            "physical_cores": cap,
            "available_cores": cap,
            "total_physical_cores": total,
            "items": items,
            "start": start,
            "end": end,
            "workers": workers,
            "waves": waves,
            "single_node_workers": per_node["workers"],
            "single_node_waves": per_node["waves"],
            "last_wave_items": last,
            "global_waves": global_waves,
            "node_shard_index": node_shard_idx,
        }
        node_shard_idx += 1
        max_workers = cpu_max_workers_per_process(
            name,
            node_info=node_infos.get(name, {}),
            windows_cpu_max_workers_per_process=windows_cpu_max_workers_per_process,
        )
        if max_workers and workers > max_workers:
            lanes = ceil_div(workers, max_workers)
            lane_cursor = start
            remaining = items
            for lane_idx in range(lanes):
                lane_count = lanes - lane_idx
                lane_items = ceil_div(remaining, lane_count)
                lane_start = lane_cursor
                lane_end = lane_start + lane_items
                lane_cursor = lane_end
                remaining -= lane_items
                lane_workers = min(max_workers, ceil_div(lane_items, max(1, waves)))
                if lane_items > 0 and lane_workers <= 0:
                    lane_workers = 1
                lane_waves = ceil_div(lane_items, lane_workers) if lane_workers else 0
                lane_last = lane_items - lane_workers * max(0, lane_waves - 1)
                if lane_last <= 0 and lane_items > 0:
                    lane_last = lane_workers
                lane = dict(base)
                lane.update({
                    "items": lane_items,
                    "start": lane_start,
                    "end": lane_end,
                    "workers": lane_workers,
                    "waves": lane_waves,
                    "last_wave_items": lane_last,
                    "physical_cores": lane_workers,
                    "available_cores": lane_workers,
                    "worker_process_limit": max_workers,
                    "worker_lane_index": lane_idx,
                    "worker_lane_count": lanes,
                    "unsplit_workers": workers,
                    "unsplit_items": items,
                    "unsplit_start": start,
                    "unsplit_end": end,
                })
                out.append(lane)
        else:
            out.append(base)
    for shard_idx, plan in enumerate(out):
        plan["shard_index"] = shard_idx
        plan["num_shards"] = len(out)
    return out


def cpu_wave_summary(items: int, workers: int) -> dict:
    """Compact per-wave item counts for CPU batch audit logs."""
    items = max(0, int(items or 0))
    workers = max(0, int(workers or 0))
    if items <= 0 or workers <= 0:
        return {"waves": 0, "workers": workers, "last_wave_items": 0, "wave_items": []}
    waves = ceil_div(items, workers)
    last = items - workers * max(0, waves - 1)
    if last <= 0:
        last = workers
    if waves <= 32:
        wave_items = [min(workers, max(0, items - idx * workers)) for idx in range(waves)]
        return {
            "waves": waves,
            "workers": workers,
            "last_wave_items": last,
            "wave_items": wave_items,
        }
    head = [min(workers, max(0, items - idx * workers)) for idx in range(8)]
    tail_start = max(8, waves - 4)
    tail = [min(workers, max(0, items - idx * workers)) for idx in range(tail_start, waves)]
    return {
        "waves": waves,
        "workers": workers,
        "last_wave_items": last,
        "wave_items_head": head,
        "wave_items_tail": tail,
        "wave_items_omitted": max(0, waves - len(head) - len(tail)),
    }


def cpu_batch_log_payload(
    total_items: int,
    plan: list,
    *,
    node_states: Optional[dict] = None,
    use_total_cores: bool = False,
    templates: Optional[dict] = None,
    logical_items: Optional[int] = None,
    item_multiplier: int = 1,
) -> dict:
    """Detailed JSON-serializable payload for CPU batch planning and submission."""
    states = node_states or {}
    nodes = []
    for shard in plan or []:
        node = shard.get("node")
        state = states.get(node) or {}
        nodes.append({
            "node": node,
            "range": [int(shard.get("start") or 0), int(shard.get("end") or 0)],
            "assigned_items": int(shard.get("items") or 0),
            "free_physical_cores_used_for_plan": int(shard.get("physical_cores") or 0),
            "available_physical_cores": int(shard.get("available_cores") or shard.get("physical_cores") or 0),
            "total_physical_cores": int(shard.get("total_physical_cores") or shard.get("physical_cores") or 0),
            "workers": int(shard.get("workers") or 0),
            "waves": int(shard.get("waves") or 0),
            "global_waves": int(shard.get("global_waves") or 0),
            "last_wave_items": int(shard.get("last_wave_items") or 0),
            "shard_index": int(shard.get("shard_index") or 0),
            "num_shards": int(shard.get("num_shards") or 1),
            "wave_plan": cpu_wave_summary(shard.get("items") or 0, shard.get("workers") or 0),
            "live_node": {
                "alive": state.get("alive"),
                "free_cpu": state.get("free_cpu"),
                "total_cpu": state.get("total_cpu"),
                "logical_cpu": state.get("logical_cpu") or state.get("logical_cores"),
                "free_ram_mb": state.get("free_ram_mb"),
                "total_ram_mb": state.get("total_ram_mb"),
                "running_count": state.get("running_count"),
                "os": state.get("os"),
            },
        })
    total_free = sum(int(shard.get("physical_cores") or 0) for shard in plan or [])
    total_capacity = sum(
        int(shard.get("total_physical_cores") or shard.get("physical_cores") or 0)
        for shard in plan or []
    )
    return {
        "total_items": int(total_items or 0),
        "logical_items": int(logical_items if logical_items is not None else total_items or 0),
        "item_multiplier": int(item_multiplier or 1),
        "node_count": len(plan or []),
        "free_physical_cores_used_for_plan": total_free,
        "total_physical_cores": total_capacity,
        "global_waves": max([
            int(shard.get("global_waves") or shard.get("waves") or 0)
            for shard in plan or []
        ] or [0]),
        "use_total_cores": bool(use_total_cores),
        "nodes": nodes,
        "templates": templates or {},
    }
