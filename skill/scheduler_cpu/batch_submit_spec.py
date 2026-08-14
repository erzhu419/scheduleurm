from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class CpuBatchSubmitDeps:
    cpu_parallel_template_values: Callable[..., dict]
    format_cpu_parallel_template: Callable[..., Any]
    rewrite_cpu_parallel_cmd: Callable[..., str]


def build_cpu_batch_submit_spec(
    args: Any,
    plan_row: dict,
    *,
    total_items: int,
    logical_items: int,
    item_multiplier: int,
    node_names: list[str],
    deps: CpuBatchSubmitDeps,
) -> tuple[dict, dict]:
    vals = deps.cpu_parallel_template_values(
        plan_row, total_items, logical_items, item_multiplier)
    env = list(getattr(args, "env", None) or [])
    env.extend([
        f"SCHEDULEURM_CPU_TOTAL_ITEMS={total_items}",
        f"SCHEDULEURM_CPU_TOTAL_WORK_ITEMS={total_items}",
        f"SCHEDULEURM_CPU_LOGICAL_ITEMS={logical_items}",
        f"SCHEDULEURM_CPU_ITEM_MULTIPLIER={item_multiplier}",
        f"SCHEDULEURM_CPU_ITEMS_PER_UNIT={item_multiplier}",
        f"SCHEDULEURM_CPU_EPISODES_PER_ITEM={item_multiplier}",
        f"SCHEDULEURM_CPU_SHARD_START={plan_row['start']}",
        f"SCHEDULEURM_CPU_SHARD_END={plan_row['end']}",
        f"SCHEDULEURM_CPU_SHARD_ITEMS={plan_row['items']}",
        f"SCHEDULEURM_CPU_WORKERS={plan_row['workers']}",
        f"SCHEDULEURM_CPU_WAVES={plan_row['waves']}",
        f"SCHEDULEURM_CPU_PHYSICAL_CORES={plan_row['physical_cores']}",
        f"SCHEDULEURM_CPU_TOTAL_PHYSICAL_CORES={plan_row.get('total_physical_cores', plan_row['physical_cores'])}",
        f"SCHEDULEURM_CPU_LAST_WAVE_ITEMS={plan_row['last_wave_items']}",
        f"SCHEDULEURM_CPU_SHARD_INDEX={plan_row['shard_index']}",
        f"SCHEDULEURM_CPU_NUM_SHARDS={plan_row['num_shards']}",
    ])
    spec = {
        "description": deps.format_cpu_parallel_template(
            args.description, plan_row, total_items, logical_items, item_multiplier),
        "cmd": deps.rewrite_cpu_parallel_cmd(
            args.cmd_template, plan_row, total_items, logical_items, item_multiplier),
        "cwd": deps.format_cpu_parallel_template(
            args.cwd, plan_row, total_items, logical_items, item_multiplier),
        "signature": deps.format_cpu_parallel_template(
            args.signature, plan_row, total_items, logical_items, item_multiplier),
        "vram": 0,
        "ram_mb": args.ram_mb,
        "cpu": int(plan_row["workers"]),
        "priority": args.priority,
        "project": args.project,
        "preferred_node": plan_row["node"],
        "require_node": None,
        "allowed_nodes": list(node_names),
        "stage_excludes": list(getattr(args, "stage_exclude", None) or []),
        "reroute_on_node_down": True,
        "node_down_requeue_s": int(getattr(args, "node_down_requeue_s", 0) or 0),
        "git_repo": None,
        "ckpt_dir": None,
        "result_dir": deps.format_cpu_parallel_template(
            getattr(args, "result_dir_template", None),
            plan_row, total_items, logical_items, item_multiplier),
        "local_result_dir": deps.format_cpu_parallel_template(
            getattr(args, "local_result_dir_template", None),
            plan_row, total_items, logical_items, item_multiplier),
        "wait_for_files": [
            deps.format_cpu_parallel_template(
                w, plan_row, total_items, logical_items, item_multiplier)
            for w in (getattr(args, "wait_for_file_template", None) or [])
        ],
        "test_log": None,
        "ckpt_glob": "*",
        "resume_flag": "",
        "env": env,
        "allow_cpu_training": bool(getattr(args, "allow_cpu_training", False)),
        "cpu_training_justification": getattr(args, "cpu_training_justification", "") or "",
        "allow_remote_large_data": bool(getattr(args, "allow_remote_large_data", False)),
        "allow_duplicate": bool(getattr(args, "allow_duplicate", False)),
        "skip_launch_staging": bool(getattr(args, "skip_launch_staging", False)),
        "env_spec": getattr(args, "env_spec", None) or "none",
        "image": getattr(args, "image", None) or "",
        "cpu_parallel_items": int(plan_row["items"]),
        "cpu_parallel_total_items": int(total_items),
        "cpu_parallel_logical_items": int(logical_items),
        "cpu_parallel_item_multiplier": int(item_multiplier),
        "cpu_parallel_start": int(plan_row["start"]),
        "cpu_parallel_end": int(plan_row["end"]),
        "cpu_parallel_shard_index": int(plan_row["shard_index"]),
        "cpu_parallel_num_shards": int(plan_row["num_shards"]),
        "cpu_batch_plan": dict(plan_row),
    }
    return spec, vals
