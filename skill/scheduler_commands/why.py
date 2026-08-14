from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class WhyCommandDeps:
    node_configs: dict
    vram_margin_mb: int
    hpc_cpu_pool_soft_require_nodes: Callable[[dict], list]
    blocked_nodes_for_task: Callable[[dict], set]
    launch_failed_nodes_for_task: Callable[[dict], set]
    node_resources_ok: Callable[[dict, dict, dict], tuple[bool, str]]
    node_is_windows: Callable[[str], bool]
    node_cpu_fallback_block_reason: Callable[[dict, str, dict], str | None]
    gpu_fits: Callable[[dict, dict, dict], bool]
    algorithm_gpu_fit_block_reason: Callable[[dict, dict, dict], str | None]
    hard_rule_bypassed: Callable[..., bool]
    gpu_freeze_line_mb: Callable[[int], int]
    task_ignores_one_third_pack_rule: Callable[[dict, dict], bool]
    node_gpu_util_limit: Callable[[dict], int | None]
    state_lock: Callable[..., Any]
    lock_timeout_error: type[Exception]
    load_state: Callable[[], dict]
    load_history: Callable[[], dict]
    probe_all: Callable[[], list]
    apply_cpu_slot_accounting_to_nodes: Callable[[dict, list], Any]
    claim_enabled_for: Callable[[str], bool]
    format_claim_intent_hint_for_task: Callable[[dict, str, dict], str]


def explain_node_fit(task: dict, node_state: dict, *, deps: WhyCommandDeps) -> str:
    """Return a one-line explanation of whether a queued task would fit on a node."""
    name = node_state["name"]
    if not node_state.get("alive"):
        return f"DOWN ({node_state.get('error', '?')})"
    node_info = deps.node_configs.get(name, {})
    if node_info.get("retired"):
        return "retired: node is no longer a scheduler candidate"
    if node_info.get("monitor_only"):
        return "login-node-disabled: monitor/control node, scheduler will never launch tasks here"

    soft_require_pool = deps.hpc_cpu_pool_soft_require_nodes(task)
    if soft_require_pool and name not in soft_require_pool:
        return "outside-cpu-pool: scheduler routes this CPU shard to node001-node006"

    allowed_nodes = {str(node) for node in (task.get("allowed_nodes") or []) if str(node)}
    if not soft_require_pool and allowed_nodes and name not in allowed_nodes:
        return "outside-allowed-nodes: task is restricted by allowed_nodes"

    require = task.get("require_node")
    if not soft_require_pool and require and require != name:
        return f"require-node-mismatch: task requires {require}"

    if name in deps.blocked_nodes_for_task(task):
        return "BLOCKED: pending env_missing/python_import escalation against this signature/cwd/project"

    soft_blocked = name in deps.launch_failed_nodes_for_task(task)
    soft_note = "  (soft-blocked: prior launch_failed; may retry as last resort)" if soft_blocked else ""

    node_info = deps.node_configs[name]
    ok, why = deps.node_resources_ok(task, node_state, node_info)
    if not ok:
        return f"node-reject: {why}{soft_note}"

    cpu_only = (task.get("est_vram_mb") or 0) <= 0
    if cpu_only:
        return (f"FITS (CPU-only): free_cpu={node_state.get('free_cpu')} "
                f"free_ram={node_state.get('free_ram_mb')}MB{soft_note}")

    cpu_fallback_node = (
        deps.node_is_windows(name)
        or not node_state.get("gpus")
        or node_info.get("max_vram_per_task") == 0
    )
    if cpu_fallback_node:
        block = deps.node_cpu_fallback_block_reason(task, name, node_info)
        if block:
            return f"BLOCKED: {block}{soft_note}"

    fit_gpus = []
    reject_gpus = []
    est = int(task.get("est_vram_mb") or 0)
    cap_per_task = node_info.get("max_vram_per_task")
    for gpu in node_state.get("gpus") or []:
        if deps.gpu_fits(task, gpu, node_info):
            fit_gpus.append(f"GPU{gpu['idx']}(free={gpu['free_mb']}MB)")
            continue

        reasons = []
        algo_reason = deps.algorithm_gpu_fit_block_reason(task, gpu, node_info)
        if algo_reason:
            reasons.append(algo_reason)
        if (
            cap_per_task is not None
            and est > cap_per_task
            and not deps.hard_rule_bypassed(
                "max_vram_per_task",
                task,
                node_info=node_info,
                node_state=node_state,
                gpu=gpu,
            )
        ):
            reasons.append(f"est={est}>per-task cap={cap_per_task}")
        freeze = deps.gpu_freeze_line_mb(int(gpu.get("total_mb") or 0))
        if (
            not deps.task_ignores_one_third_pack_rule(task, node_info)
            and freeze > 0
            and gpu["used_mb"] > 100
            and (gpu["used_mb"] >= freeze or gpu["used_mb"] + est >= freeze)
        ):
            reasons.append(
                f"used+est {gpu['used_mb']}+{est}/{gpu['total_mb']}MB ≥ "
                f"1/3+grace {freeze}MB (packing rule)"
            )
        util_limit = deps.node_gpu_util_limit(node_info)
        if (
            util_limit is not None
            and gpu["used_mb"] > 100
            and gpu.get("util_pct", 0) >= util_limit
            and not deps.hard_rule_bypassed(
                "gpu_util_saturation",
                task,
                node_info=node_info,
                node_state=node_state,
                gpu=gpu,
            )
        ):
            reasons.append(f"util={gpu.get('util_pct')}% ≥ {util_limit}% (compute saturation)")
        if (
            gpu["free_mb"] < est + deps.vram_margin_mb
            and not deps.hard_rule_bypassed(
                "vram_margin",
                task,
                node_info=node_info,
                node_state=node_state,
                gpu=gpu,
            )
        ):
            reasons.append(
                f"free={gpu['free_mb']}MB < est+margin ({est}+{deps.vram_margin_mb})"
            )
        reject_gpus.append(f"GPU{gpu['idx']}({'; '.join(reasons) or 'unknown'})")

    if fit_gpus:
        return f"FITS: {', '.join(fit_gpus)}{soft_note}"
    if not reject_gpus:
        return f"no GPUs probed (CPU-only node?){soft_note}"
    return f"no-GPU-fit: {' | '.join(reject_gpus)}{soft_note}"


def cmd_why(args, *, deps: WhyCommandDeps) -> None:
    """Print why a queued task is not being placed."""
    try:
        with deps.state_lock(
            timeout_s=getattr(args, "lock_timeout", None),
            shared=True,
            purpose="why",
        ):
            state = deps.load_state()
    except deps.lock_timeout_error as exc:
        sys.exit(str(exc))

    task = next((item for item in state["tasks"] if item["id"] == args.id), None)
    if not task:
        sys.exit(f"task {args.id} not found")

    print(f"=== why {args.id} ===")
    print(f"  status:       {task.get('status')}")
    print(f"  description:  {(task.get('description') or '')[:100]}")
    print(f"  signature:    {task.get('signature')}")
    print(f"  priority:     {task.get('priority')}")
    gpu_pin = task.get("require_gpu_idx")
    gpu_pin_text = "" if gpu_pin is None else f"  require_gpu: {gpu_pin!r}"
    print(
        f"  preferred:    {task.get('preferred_node')!r}  "
        f"require:    {task.get('require_node')!r}{gpu_pin_text}"
    )
    print(
        f"  est:          vram={task.get('est_vram_mb')}MB "
        f"ram={task.get('ram_mb')}MB cpu={task.get('cpu_cores')}"
    )
    if task.get("wait_for_files"):
        print(f"  wait_files:   {task.get('wait_for_files')}")
    if task.get("status") != "queued":
        print()
        print(f"  (task is {task.get('status')!r} — `why` is for diagnosing "
              f"queued tasks that aren't getting placed)")
        return

    print()
    print("  last_block_reason:")
    last_block_reason = (
        task.get("last_block_reason")
        or "(none yet — task hasn't been considered for placement)"
    )
    print(f"    {last_block_reason}")

    sig = task.get("signature") or ""
    if sig:
        history = deps.load_history()
        own = history.get(sig)
        if own is not None:
            if isinstance(own, int):
                own = {"vram_mb": own}
            print()
            print(f"  history[{sig!r}]:")
            print(f"    vram_mb={own.get('vram_mb', 0)}  ram_mb={own.get('ram_mb', 0)}  "
                  f"cpu_cores={own.get('cpu_cores', 0)}  runs={own.get('dur_s_runs', 0)}")
            samples = own.get("vram_samples") or []
            if samples:
                print(f"    vram_samples={samples}  (single-sample outliers may "
                      f"poison estimate — `history --drop {sig}` to clear)")
        prefix = "/".join(sig.split("/")[:2])
        siblings = []
        for other_sig, value in history.items():
            if other_sig == sig:
                continue
            if not other_sig.startswith(prefix + "/"):
                continue
            vram_mb = value.get("vram_mb", 0) if isinstance(value, dict) else value
            siblings.append((other_sig, vram_mb))
        if siblings:
            print()
            print(
                f"  sibling signatures under {prefix!r}/ "
                f"(compare against own est={task.get('est_vram_mb')}MB):"
            )
            for other_sig, vram_mb in sorted(siblings, key=lambda item: -item[1])[:8]:
                marker = "  ← much smaller, est may be over" if (
                    task.get("est_vram_mb") and vram_mb > 0
                    and task["est_vram_mb"] > 2 * vram_mb
                ) else ""
                print(f"    {other_sig:<55s} vram={vram_mb}MB{marker}")

    print()
    print("  per-node fit analysis (probe_all snapshot):")
    nodes = deps.probe_all()
    deps.apply_cpu_slot_accounting_to_nodes(state, nodes)
    for node in nodes:
        print(f"    {node['name']:11s}: {explain_node_fit(task, node, deps=deps)}")
        if deps.claim_enabled_for(node["name"]):
            snap = {
                "claims": node.get("pending_claims") or [],
                "intents": node.get("claim_intents") or [],
                "error": node.get("claim_snapshot_error"),
                "ok": not node.get("claim_snapshot_error"),
            }
            hint = deps.format_claim_intent_hint_for_task(task, node["name"], snap)
            if hint:
                print(f"      {hint}")
            elif snap.get("error"):
                print(f"      claim-snapshot-error: {snap['error']}")
