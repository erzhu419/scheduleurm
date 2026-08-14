"""CLI orchestration for CPU batch submission."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class SubmitCpuBatchCommandDeps:
    cpu_batch_item_counts: Callable[[Any], tuple[int, int, int]]
    parse_cpu_node_list: Callable[[Any], list[str]]
    cpu_plan_live_node_states: Callable[[list[str]], dict]
    cpu_batch_plan: Callable[..., list[dict]]
    cpu_parallel_template_values: Callable[..., dict]
    rewrite_cpu_parallel_cmd: Callable[..., str]
    format_cpu_parallel_template: Callable[..., Any]
    print_cpu_plan: Callable[[list[dict], int], None]
    cpu_batch_log_payload: Callable[..., dict]
    notify: Callable[..., Any]
    cpu_batch_submit_spec: Callable[..., tuple[dict, dict]]
    validate_cpu_batch_submit_spec: Callable[[dict, Any], None]
    load_history: Callable[[], dict]
    load_runtime_history: Callable[[], dict]
    runtime_history_closest_index: Callable[[dict], list[dict]]
    write_dispatch_intent: Callable[..., Any]
    state_lock: Callable[..., Any]
    load_state: Callable[[], dict]
    validate_bulk_submit_result_conflicts: Callable[..., None]
    allocate_task_ids: Callable[[dict, int], list[str]]
    build_bulk_submit_tasks: Callable[..., list[dict]]
    save_state: Callable[[dict], None]
    clear_dispatch_intent: Callable[[], Any]
    dispatch_intent_ttl_s: int


@dataclass(frozen=True)
class CpuBatchSubmitValidationDeps:
    submit_policy_text: Callable[[str, str], str]
    simple_sac_large_data_reason: Callable[[str, str], str]
    cpu_training_policy_reason: Callable[[str, str | None, bool, int], str]
    task_looks_like_training: Callable[[str, str | None], bool]
    cmd_looks_like_training: Callable[[str], bool]
    resume_capability_reason: Callable[[str, str | None, str, bool], str]
    checkpoint_contract_reason: Callable[[str, str, str | None, str, bool], str]
    same_declared_path: Callable[[str | None, str | None], bool]


def validate_cpu_batch_submit_spec(
    spec: dict,
    args: Any,
    *,
    deps: CpuBatchSubmitValidationDeps,
) -> None:
    raw_submit_cmd = spec.get("cmd") or ""
    cwd = spec.get("cwd") or ""
    policy_cmd = deps.submit_policy_text(raw_submit_cmd, cwd)
    large_data_reason = deps.simple_sac_large_data_reason(raw_submit_cmd, cwd)
    if large_data_reason and not bool(getattr(args, "allow_remote_large_data", False)):
        preferred = spec.get("preferred_node")
        if preferred and preferred != "local":
            sys.exit(
                "REFUSED: CPU batch shard prefers a remote node but appears to "
                f"require large local-only SimpleSAC data: {large_data_reason}"
            )

    allow_cpu_training = bool(getattr(args, "allow_cpu_training", False))
    cpu_training_reason = deps.cpu_training_policy_reason(
        policy_cmd,
        spec.get("description"),
        allow_cpu_training,
        int(spec.get("vram") or 0),
    )
    if cpu_training_reason:
        sys.exit(
            "REFUSED: CPU batch command looks like training but scheduler CPU/GPU "
            f"policy is inconsistent: {cpu_training_reason}"
        )
    if allow_cpu_training and deps.task_looks_like_training(policy_cmd, spec.get("description")):
        just = (getattr(args, "cpu_training_justification", "") or "").strip()
        if len(just) < 30:
            sys.exit(
                "REFUSED: --allow-cpu-training requires "
                "--cpu-training-justification with >=30 chars of explanation"
            )
    allow_no_ckpt = bool(getattr(args, "allow_no_ckpt", False))
    allow_no_resume = bool(getattr(args, "allow_no_resume", False))
    if deps.cmd_looks_like_training(policy_cmd) and not spec.get("ckpt_dir") and not allow_no_ckpt:
        sys.exit("REFUSED: CPU batch command looks like training but --ckpt-dir is not set")
    resume_reason = deps.resume_capability_reason(
        policy_cmd,
        spec.get("ckpt_dir"),
        spec.get("resume_flag") or "",
        allow_no_resume,
    )
    if resume_reason:
        sys.exit(f"REFUSED: CPU batch resume is not wired up: {resume_reason}")
    contract_reason = deps.checkpoint_contract_reason(
        raw_submit_cmd,
        cwd,
        spec.get("ckpt_dir"),
        spec.get("resume_flag") or "",
        allow_no_resume,
    )
    if contract_reason:
        sys.exit(f"REFUSED: CPU batch checkpoint contract is not verifiable: {contract_reason}")
    if spec.get("ckpt_dir") and deps.same_declared_path(spec.get("ckpt_dir"), cwd):
        sys.exit("REFUSED: CPU batch --ckpt-dir must not equal --cwd")
    if spec.get("result_dir") and deps.same_declared_path(spec.get("result_dir"), cwd):
        sys.exit("REFUSED: CPU batch --result-dir must not equal --cwd")


def _cpu_batch_dry_rows(
    args: Any,
    plan: list[dict],
    *,
    total_items: int,
    logical_items: int,
    item_multiplier: int,
    deps: SubmitCpuBatchCommandDeps,
) -> list[dict]:
    rows = []
    for row in plan:
        vals = deps.cpu_parallel_template_values(
            row, total_items, logical_items, item_multiplier)
        rows.append({
            **row,
            "cmd": deps.rewrite_cpu_parallel_cmd(
                args.cmd_template, row, total_items, logical_items, item_multiplier),
            "cwd": deps.format_cpu_parallel_template(
                args.cwd, row, total_items, logical_items, item_multiplier),
            "signature": deps.format_cpu_parallel_template(
                args.signature, row, total_items, logical_items, item_multiplier),
            "description": deps.format_cpu_parallel_template(
                args.description, row, total_items, logical_items, item_multiplier),
            "env": vals,
        })
    return rows


def _validate_multi_node_templates(args: Any, plan: list[dict]) -> None:
    split_tokens = ("{start}", "{end}", "{node}", "{shard_index}", "{shard_id}", "{num_shards}")
    if len(plan) <= 1 or getattr(args, "allow_env_only_shard", False):
        return
    templated = " ".join(str(x or "") for x in (
        args.cmd_template,
        getattr(args, "result_dir_template", None),
        getattr(args, "local_result_dir_template", None),
    ))
    if any(token in templated for token in split_tokens):
        return
    sys.exit(
        "REFUSED: submit-cpu-batch spans multiple nodes but the command/result templates "
        "do not contain shard placeholders like {start}/{end}/{node}. "
        "Pass --allow-env-only-shard only if the script reads SCHEDULEURM_CPU_* env vars."
    )


def _emit_cpu_batch_dry_or_json(
    args: Any,
    plan: list[dict],
    *,
    total_items: int,
    logical_items: int,
    item_multiplier: int,
    deps: SubmitCpuBatchCommandDeps,
) -> None:
    rows = _cpu_batch_dry_rows(
        args,
        plan,
        total_items=total_items,
        logical_items=logical_items,
        item_multiplier=item_multiplier,
        deps=deps,
    )
    if args.json:
        print(json.dumps({
            "logical_items": logical_items,
            "item_multiplier": item_multiplier,
            "total_items": total_items,
            "plan": rows,
        }, indent=2))
        return
    if item_multiplier != 1:
        print(
            f"cpu-plan input: logical_items={logical_items} "
            f"item_multiplier={item_multiplier} total_items={total_items}"
        )
    deps.print_cpu_plan(plan, total_items)
    for row in rows:
        print(f"  cmd[{row['node']}]: {row['cmd']}")


def _build_and_notify_cpu_batch_specs(
    args: Any,
    plan: list[dict],
    *,
    names: list[str],
    total_items: int,
    logical_items: int,
    item_multiplier: int,
    deps: SubmitCpuBatchCommandDeps,
) -> list[dict]:
    specs = []
    for row in plan:
        spec, vals = deps.cpu_batch_submit_spec(
            args, row, total_items, logical_items, item_multiplier, names)
        deps.validate_cpu_batch_submit_spec(spec, args)
        specs.append(spec)
        deps.notify("cpu_batch_shard_submit", {
            "node": row["node"],
            "range": [int(row["start"]), int(row["end"])],
            "items": int(row["items"]),
            "workers": int(row["workers"]),
            "waves": int(row["waves"]),
            "last_wave_items": int(row["last_wave_items"]),
            "signature": spec.get("signature"),
            "description": spec.get("description"),
            "cmd": spec.get("cmd"),
            "env": vals,
            "logical_items": logical_items,
            "item_multiplier": item_multiplier,
            "total_items": total_items,
        }, feishu_enabled=False)
    return specs


def _submit_cpu_batch_specs(
    args: Any,
    specs: list[dict],
    *,
    deps: SubmitCpuBatchCommandDeps,
) -> list[dict]:
    submitted = []
    resource_history = deps.load_history()
    runtime_history = deps.load_runtime_history()
    runtime_closest_index = deps.runtime_history_closest_index(runtime_history)
    deps.write_dispatch_intent(
        label=f"submit-cpu-batch-{len(specs)}",
        ttl_s=deps.dispatch_intent_ttl_s,
    )
    try:
        with deps.state_lock(purpose="submit-cpu-batch"):
            state = deps.load_state()
            deps.validate_bulk_submit_result_conflicts(
                state,
                specs,
                allow_shared_result_dir=bool(getattr(args, "allow_shared_result_dir", False)),
            )
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
    return submitted


def _print_submitted_cpu_batch(submitted: list[dict]) -> None:
    for task in submitted:
        print(
            f"submitted {task['id']}  cpu={task.get('cpu_cores')} "
            f"ram={task.get('ram_mb')}MB vram={task.get('est_vram_mb')}MB  "
            f"prio={task.get('priority')}  {(task.get('description') or '')[:50]}"
        )
        if task.get("cpu_parallel_items"):
            print(
                f"  cpu-workers: items={task.get('cpu_parallel_items')} "
                f"physical={task.get('cpu_parallel_physical_cores')} "
                f"waves={task.get('cpu_parallel_waves')} "
                f"workers={task.get('cpu_auto_workers')} "
                f"last_wave={task.get('cpu_parallel_last_wave_items')}"
            )
    print(f"submitted {len(submitted)} CPU batch shard(s) via one state-lock transaction")


def cmd_submit_cpu_batch(args: Any, *, deps: SubmitCpuBatchCommandDeps) -> None:
    logical_items, item_multiplier, total_items = deps.cpu_batch_item_counts(args)
    names = deps.parse_cpu_node_list(getattr(args, "nodes", None))
    node_states = None if getattr(args, "use_total_cores", False) else deps.cpu_plan_live_node_states(names)
    plan = deps.cpu_batch_plan(total_items, names, node_states=node_states)
    if not plan:
        sys.exit("no CPU batch plan could be built")
    _validate_multi_node_templates(args, plan)

    if args.json or args.dry_run:
        _emit_cpu_batch_dry_or_json(
            args,
            plan,
            total_items=total_items,
            logical_items=logical_items,
            item_multiplier=item_multiplier,
            deps=deps,
        )
        if args.dry_run:
            return

    plan_payload = deps.cpu_batch_log_payload(
        total_items,
        plan,
        node_states=node_states,
        use_total_cores=bool(getattr(args, "use_total_cores", False)),
        logical_items=logical_items,
        item_multiplier=item_multiplier,
        templates={
            "cmd": args.cmd_template,
            "cwd": args.cwd,
            "signature": args.signature,
            "description": args.description,
            "result_dir": getattr(args, "result_dir_template", None),
            "local_result_dir": getattr(args, "local_result_dir_template", None),
        },
    )
    deps.notify("cpu_batch_plan", plan_payload, feishu_enabled=False)

    specs = _build_and_notify_cpu_batch_specs(
        args,
        plan,
        names=names,
        total_items=total_items,
        logical_items=logical_items,
        item_multiplier=item_multiplier,
        deps=deps,
    )
    submitted = _submit_cpu_batch_specs(args, specs, deps=deps)
    _print_submitted_cpu_batch(submitted)
