"""submit and submit-jsonl command entrypoints."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class SubmitCommandDeps:
    split_bapr_seed_batch_submit_args: Callable[[Any], list]
    submit_one: Callable[[Any], Any]
    run_submit_preflight: Callable[..., Any]
    submit_preflight_deps: Callable[[], Any]
    submit_preflight_refusal_type: type[BaseException]
    history_record: Callable[..., Any]
    state_lock: Callable[..., Any]
    load_state: Callable[[], dict]
    build_submit_task_for_state: Callable[..., Any]
    submit_task_deps: Callable[[], Any]
    save_state: Callable[[dict], Any]
    submit_task_refusal_type: type[BaseException]
    default_vram_mb: int
    default_ram_mb: int
    default_cpu_cores: int
    print_fn: Callable[..., Any] = print
    exit_fn: Callable[[Any], Any] = sys.exit


@dataclass(frozen=True)
class SubmitJsonlCommandDeps:
    load_submit_jsonl_specs: Callable[[Any], list]
    write_dispatch_intent: Callable[..., Any]
    clear_dispatch_intent: Callable[[], Any]
    load_history: Callable[[], dict]
    load_runtime_history: Callable[[], dict]
    runtime_history_closest_index: Callable[[dict], list]
    state_lock: Callable[..., Any]
    load_state: Callable[[], dict]
    allocate_task_ids: Callable[[dict, int], list]
    build_bulk_submit_tasks: Callable[..., list]
    save_state: Callable[[dict], Any]
    dispatch_intent_ttl_s: float
    json_dumps: Callable[..., str] = json.dumps
    print_fn: Callable[..., Any] = print
    exit_fn: Callable[[Any], Any] = sys.exit


def cmd_submit(args, *, deps: SubmitCommandDeps):
    split_args = deps.split_bapr_seed_batch_submit_args(args)
    if split_args:
        deps.print_fn(
            f"NOTE: auto-splitting BAPR seed batch into {len(split_args)} per-seed tasks",
            file=sys.stderr,
        )
        for child_args in split_args:
            deps.submit_one(child_args)
        return

    try:
        preflight = deps.run_submit_preflight(
            args,
            deps=deps.submit_preflight_deps(),
            default_vram_mb=deps.default_vram_mb,
        )
    except deps.submit_preflight_refusal_type as refusal:
        for line in refusal.lines:
            deps.print_fn(line, file=sys.stderr)
        deps.exit_fn(refusal.code)
        return
    for note in preflight.notes:
        deps.print_fn(note, file=sys.stderr)

    test_peak_vram = int(getattr(args, "test_peak_vram_mb", 0) or 0)
    test_peak_ram = int(getattr(args, "test_peak_ram_mb", 0) or 0)
    test_cpu = int(getattr(args, "test_cpu", 0) or 0)
    if test_peak_vram > 0 or test_peak_ram > 0 or test_cpu > 0:
        deps.history_record(
            args.signature,
            peak_vram_mb=test_peak_vram,
            peak_ram_mb=test_peak_ram,
            cpu_cores=test_cpu,
        )

    try:
        with deps.state_lock():
            state = deps.load_state()
            submit_result = deps.build_submit_task_for_state(
                state,
                args,
                preflight,
                deps=deps.submit_task_deps(),
                default_ram_mb=deps.default_ram_mb,
                default_cpu_cores=deps.default_cpu_cores,
            )
            task = submit_result.task
            state["tasks"].append(task)
            deps.save_state(state)
    except deps.submit_task_refusal_type as refusal:
        stream = sys.stdout if refusal.stream == "stdout" else sys.stderr
        for line in refusal.lines:
            deps.print_fn(line, file=stream)
        deps.exit_fn(refusal.code)
        return

    deps.print_fn(
        f"submitted {task['id']}  cpu={submit_result.cpu_cores} "
        f"ram={submit_result.ram_mb}MB vram={submit_result.est_vram}MB  "
        f"prio={args.priority}  ({submit_result.source_label})  "
        f"{args.description[:50]}"
    )
    if task.get("cpu_parallel_items"):
        deps.print_fn(
            f"  cpu-workers: items={task.get('cpu_parallel_items')} "
            f"physical={task.get('cpu_parallel_physical_cores')} "
            f"waves={task.get('cpu_parallel_waves')} "
            f"workers={task.get('cpu_auto_workers')} "
            f"last_wave={task.get('cpu_parallel_last_wave_items')}"
        )
    deps.print_fn(
        "  run `dispatch` to launch (resource-aware: respects 1/3 VRAM rule, CPU/RAM headroom)."
    )


def _submit_jsonl_payload(submitted: list[dict]) -> dict:
    return {
        "count": len(submitted),
        "submitted": [
            {
                "id": task["id"],
                "signature": task.get("signature"),
                "description": task.get("description"),
                "require_node": task.get("require_node"),
                "require_gpu_idx": task.get("require_gpu_idx"),
                "allowed_nodes": task.get("allowed_nodes"),
                "cpu": task.get("cpu_cores"),
                "ram_mb": task.get("ram_mb"),
                "vram": task.get("est_vram_mb"),
            }
            for task in submitted
        ],
    }


def cmd_submit_jsonl(args, *, deps: SubmitJsonlCommandDeps):
    if not getattr(args, "trusted", False):
        deps.exit_fn(
            "submit-jsonl is a trusted fast path for project submit scripts; "
            "pass --trusted after constructing scheduler-compatible task specs"
        )
        return

    specs = deps.load_submit_jsonl_specs(args)
    if not specs:
        deps.print_fn(
            deps.json_dumps({"submitted": [], "count": 0}) if args.json else "submitted 0 tasks"
        )
        return

    label = getattr(args, "intent_label", "") or f"bulk-submit-{len(specs)}"
    ttl = getattr(args, "intent_ttl", None) or deps.dispatch_intent_ttl_s
    deps.write_dispatch_intent(label=label, ttl_s=ttl)
    submitted = []
    resource_history = deps.load_history()
    runtime_history = deps.load_runtime_history()
    runtime_closest_index = deps.runtime_history_closest_index(runtime_history)
    try:
        with deps.state_lock(timeout_s=getattr(args, "lock_timeout", None), purpose="submit-jsonl"):
            state = deps.load_state()
            task_ids = deps.allocate_task_ids(state, len(specs))
            submitted = deps.build_bulk_submit_tasks(
                state,
                specs,
                task_ids,
                resource_history=resource_history,
                runtime_history=runtime_history,
                runtime_closest_index=runtime_closest_index,
            )
            deps.save_state(state)
    finally:
        deps.clear_dispatch_intent()

    payload = _submit_jsonl_payload(submitted)
    if args.json:
        deps.print_fn(deps.json_dumps(payload, ensure_ascii=False))
        return
    for task in submitted:
        deps.print_fn(
            f"submitted {task['id']}  cpu={task.get('cpu_cores')} "
            f"ram={task.get('ram_mb')}MB vram={task.get('est_vram_mb')}MB  "
            f"prio={task.get('priority')}  {(task.get('description') or '')[:50]}"
        )
    deps.print_fn(f"submitted {len(submitted)} tasks via submit-jsonl")
