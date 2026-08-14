"""Placement and GPU fit policy for scheduleurm dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class PlacementRuntime:
    nodes_by_name: dict
    default_vram_mb: int
    vram_margin_mb: int
    gpu_empty_used_mb: int
    one_third_pack_rule: bool
    hpc_cpu_pool_soft_require_nodes: Callable[[dict], list]
    task_required_gpu_idx: Callable[[dict], Any]
    blocked_nodes_for_task: Callable[[dict], set]
    launch_failed_nodes_for_task: Callable[[dict], set]
    algorithm_trace_enabled: Callable[[], bool] | None
    algorithm_trace_slot_id: Callable[[dict], str] | None
    algorithm_trace_candidate_row: Callable[..., dict] | None
    algorithm_trace_decision_slot: Callable[..., dict] | None
    algorithm_log_decision_slot: Callable[[dict], Any] | None
    algorithm_name: Callable[[], str]
    algorithm_selected_gpu_audit: Callable[[dict, dict, dict], dict]
    evict_node_cooldown_block_reason: Callable[..., Any]
    node_is_windows: Callable[[str], bool]
    node_cpu_fallback_block_reason: Callable[[dict, str, dict], str | None]
    node_gpu_task_block_reason: Callable[[dict, str, dict], str | None]
    node_resources_ok: Callable[[dict, dict, dict], tuple[bool, str]]
    node_schedulable_cpu_free: Callable[[dict, dict, dict], tuple]
    candidate_runtime_seconds: Callable[[dict, str, Any], int]
    gpu_task_cap_for_task: Callable[[dict, dict], Any]
    hard_rule_bypassed: Callable[..., bool]
    algorithm_gpu_score: Callable[[dict, dict, dict, tuple], tuple]
    task_ignores_one_third_pack_rule: Callable[[dict, dict], bool]
    gpu_freeze_line_mb: Callable[[int], int]
    node_gpu_util_limit: Callable[[dict], Any]
    algorithm_gpu_fit_block_reason: Callable[[dict, dict, dict], str | None]


def gpu_fits(task, gpu, node_info, runtime: PlacementRuntime):
    """VRAM + compute-saturation check on a specific GPU.

    Policy: on an empty/noise-only GPU, one large task may exceed the freeze line;
    on an occupied GPU, do not place a task if the GPU is already at the 1/3
    memory line plus a small grace window, or this placement would cross it.
    No small-task exemption. This freezes warm cards until old work completes,
    because dispatching into a
    partially occupied JAX/PyTorch card is exactly the scenario that makes OOM
    forensics ambiguous and can kill useful in-flight progress.
    """
    cap = node_info.get("max_vram_per_task")
    if (cap is not None and task["est_vram_mb"] > cap
            and not runtime.hard_rule_bypassed("max_vram_per_task", task, node_info=node_info, gpu=gpu)):
        return False
    if runtime.one_third_pack_rule and not runtime.task_ignores_one_third_pack_rule(task, node_info):
        freeze = runtime.gpu_freeze_line_mb(int(gpu.get("total_mb") or 0))
        used = int(gpu.get("used_mb") or 0)
        need = int(task.get("est_vram_mb") or 0)
        tracked_occupied = int(gpu.get("running_task_count") or 0) > 0
        memory_occupied = used >= max(1, int(runtime.gpu_empty_used_mb or 0))
        if (
            freeze > 0
            and (tracked_occupied or memory_occupied)
            and (used >= freeze or used + need >= freeze)
        ):
            return False
    # Compute saturation: if there's already a task on this GPU and it's pinning the chip,
    # don't pack more — the new task would just steal cycles and slow everyone down.
    # The "occupied" guard (>100MB) avoids blocking on a transient util spike on an empty GPU.
    util_limit = runtime.node_gpu_util_limit(node_info)
    if (util_limit is not None and gpu["used_mb"] > 100 and gpu.get("util_pct", 0) >= util_limit
            and not runtime.hard_rule_bypassed("gpu_util_saturation", task, node_info=node_info, gpu=gpu)):
        return False
    if gpu["free_mb"] < task["est_vram_mb"] + runtime.vram_margin_mb:
        if not runtime.hard_rule_bypassed("vram_margin", task, node_info=node_info, gpu=gpu):
            return False
        if gpu["free_mb"] < task["est_vram_mb"]:
            return False
    if runtime.algorithm_gpu_fit_block_reason(task, gpu, node_info):
        return False
    return True

def pick_placement(task, nodes, runtime: PlacementRuntime, extra_allowed_nodes=None):
    """Pick (node, gpu_idx) given current per-node free resources. gpu_idx=None for CPU-only tasks.
    preferred_node is a SOFT preference: try it first; if it can't fit, fall back to any other node
    that satisfies all constraints. This prevents tasks from getting stuck when their preferred node
    is full but other nodes have headroom.

    Also consults pending escalations: if THIS signature has a pending ENV_MISSING/PYTHON_IMPORT
    escalation on a node, that node is excluded from candidates here so we don't keep redispatching
    the task to a node we already know can't run it. /scheduler-heal resolves the escalation when
        the env is fixed, freeing that node again."""
    cpu_only = task.get("est_vram_mb", runtime.default_vram_mb) <= 0
    allowed_nodes = task.get("allowed_nodes") or []
    extra_allowed = {str(n) for n in (extra_allowed_nodes or []) if str(n)}
    if allowed_nodes or extra_allowed:
        allowed = {str(n) for n in allowed_nodes} | extra_allowed
        nodes = [n for n in nodes if n.get("name") in allowed]
    preferred = task.get("preferred_node")
    require = task.get("require_node")  # HARD pin — never falls back
    soft_require_pool = runtime.hpc_cpu_pool_soft_require_nodes(task)
    if soft_require_pool:
        # Generated FreqDuet/freq_hrl per-node pins describe the shared CPU
        # pool, not a real locality preference. Let the pool score decide so
        # empty siblings such as node006 receive work.
        preferred = None
        require = None
        allowed = set(soft_require_pool)
        nodes = [n for n in nodes if n.get("name") in allowed]
    require_gpu_idx = runtime.task_required_gpu_idx(task)
    resume_preferred = []
    for n in (task.get("resume_preferred_nodes") or []):
        if n and n not in resume_preferred:
            resume_preferred.append(n)
    # Phase 3.0.15 P1 fix: a task that was migrated has its cwd/ckpt staged ONLY
    # on staged_node. Without promoting that to a hard pin, pick_placement's
    # fallback path could land the task on a third node where staging never ran
    # → resume task silently restarts from step 0. User-explicit require_node
    # still wins over the migration pin (operator override beats auto-balance).
    staged = task.get("staged_node")
    if staged and not require:
        require = staged
    blocked = runtime.blocked_nodes_for_task(task)
    launch_failed = runtime.launch_failed_nodes_for_task(task)
    if blocked:
        nodes = [n for n in nodes if n["name"] not in blocked]

    trace_this_decision = bool(
        runtime.algorithm_trace_enabled is not None and runtime.algorithm_trace_enabled()
    )
    trace_slot_id = (
        runtime.algorithm_trace_slot_id(task) if trace_this_decision and runtime.algorithm_trace_slot_id else ""
    )

    def _node_state_by_name(name, search_nodes):
        return next((n for n in search_nodes if n.get("name") == name), None)

    def _gpu_state_by_idx(node_state, gpu_idx):
        if node_state is None or gpu_idx is None:
            return None
        for gpu in node_state.get("gpus") or []:
            try:
                if int(gpu.get("idx")) == int(gpu_idx):
                    return gpu
            except Exception:
                continue
        return None

    def _trace_candidate_slot(phase, search_nodes, cands, selected):
        if not trace_this_decision:
            return
        if (
            runtime.algorithm_trace_candidate_row is None
            or runtime.algorithm_trace_decision_slot is None
            or runtime.algorithm_log_decision_slot is None
        ):
            return
        selected_node, selected_gpu_idx = selected
        rows = []
        audits = []
        for score, node_name, gpu_idx in cands:
            node_state = _node_state_by_name(node_name, search_nodes)
            gpu_state = _gpu_state_by_idx(node_state, gpu_idx)
            audit = {}
            if gpu_state is not None:
                audit = runtime.algorithm_selected_gpu_audit(task, node_state, gpu_state)
            audits.append(audit)
            rows.append(runtime.algorithm_trace_candidate_row(
                node=node_name,
                gpu_idx=gpu_idx,
                score=score,
                selected=(
                    str(node_name) == str(selected_node)
                    and (gpu_idx is selected_gpu_idx or str(gpu_idx) == str(selected_gpu_idx))
                ),
                algorithm_audit=audit,
                lower_service=audit.get("lower_service") if isinstance(audit, dict) else None,
                penalty_units=audit.get("penalty_units") if isinstance(audit, dict) else None,
            ))
        theorem_ready = bool(audits) and all(
            isinstance(a, dict)
            and a.get("score_semantics") == "robust_maxweight_lower_service"
            and a.get("lower_service")
            for a in audits
        )
        selected_audit = {}
        for row, audit in zip(rows, audits):
            if row.get("selected"):
                selected_audit = audit if isinstance(audit, dict) else {}
                break
        slot = runtime.algorithm_trace_decision_slot(
            slot_id=trace_slot_id,
            task=task,
            algorithm=runtime.algorithm_name(),
            phase=phase,
            candidates=rows,
            score_semantics=(
                "robust_maxweight_lower_service"
                if theorem_ready else
                "scheduler_sort_key_minimization"
            ),
            queue_vector=(
                selected_audit.get("queue_vector")
                if isinstance(selected_audit, dict) and theorem_ready else
                None
            ),
            hard_constraints={
                "require_node": require,
                "preferred_node": preferred,
                "require_gpu_idx": require_gpu_idx,
                "allowed_nodes": allowed_nodes,
                "extra_allowed_nodes": sorted(extra_allowed),
                "resume_preferred_nodes": resume_preferred,
            },
        )
        if theorem_ready:
            slot["best_score_exact_over_candidate_set"] = all(
                bool(a.get("best_score_exact_over_candidate_set", True))
                for a in audits if isinstance(a, dict)
            )
        try:
            runtime.algorithm_log_decision_slot(slot)
        except Exception:
            pass

    def _candidates_for_node(n):
        """Return list of (score, name, gpu_idx) candidates this node can offer (may be empty)."""
        if not n["alive"]: return []
        if n.get("measurement_reservation_active"):
            return []
        if runtime.evict_node_cooldown_block_reason(task, n["name"], node_state=n):
            return []
        node_info = runtime.nodes_by_name[n["name"]]
        node_info_with_state = dict(node_info)
        node_info_with_state["_scheduler_node_state"] = n
        if node_info.get("retired"):
            return []
        if node_info.get("monitor_only"):
            return []
        if node_info.get("only_when_targeted"):
            targeted = (
                require == n["name"]
                or preferred == n["name"]
                or n["name"] in (task.get("allowed_nodes") or [])
                or n["name"] in soft_require_pool
            )
            if not targeted:
                return []
        cpu_fallback = False
        if not cpu_only:
            cpu_fallback = (
                runtime.node_is_windows(n["name"])
                or not n.get("gpus")
                or node_info.get("max_vram_per_task") == 0
            )
            if cpu_fallback and runtime.node_cpu_fallback_block_reason(task, n["name"], node_info):
                return []
        if runtime.node_is_windows(n["name"]) and not (cpu_only or cpu_fallback):
            return []
        if not cpu_only and runtime.node_gpu_task_block_reason(task, n["name"], node_info):
            if not cpu_fallback:
                return []
        ok, _why = runtime.node_resources_ok(task, n, node_info)
        if not ok: return []
        out = []
        if cpu_only or cpu_fallback:
            node_info = runtime.nodes_by_name.get(n["name"], {})
            cpu_labor_bonus = -1 if node_info.get("cpu_labor_node") else 0
            fallback_penalty = 2 if cpu_fallback and not cpu_only else 0
            rt = runtime.candidate_runtime_seconds(task, n["name"], None)
            rt_unknown = 1 if rt <= 0 else 0
            score_free_cpu = runtime.node_schedulable_cpu_free(task, n, node_info)[0]
            if soft_require_pool and cpu_only:
                score = (fallback_penalty, cpu_labor_bonus, -score_free_cpu, rt_unknown, rt, -n["free_ram_mb"])
            else:
                score = (fallback_penalty, rt_unknown, rt, cpu_labor_bonus, -score_free_cpu, -n["free_ram_mb"])
            out.append((score, n["name"], None))
        else:
            for g in n["gpus"]:
                try:
                    g_idx = int(g.get("idx"))
                except Exception:
                    continue
                if require_gpu_idx is not None and g_idx != require_gpu_idx:
                    continue
                max_tasks_per_gpu = runtime.gpu_task_cap_for_task(task, node_info)
                if max_tasks_per_gpu is not None:
                    try:
                        gpu_task_count = int(g.get("running_task_count") or 0)
                        gpu_task_cap = int(max_tasks_per_gpu)
                    except Exception:
                        gpu_task_count = 0
                        gpu_task_cap = 0
                    if (gpu_task_cap > 0 and gpu_task_count >= gpu_task_cap
                            and not runtime.hard_rule_bypassed(
                                "max_tasks_per_gpu", task, node_info=node_info,
                                node_state=n, gpu=g)):
                        continue
                if not gpu_fits(task, g, node_info_with_state, runtime):
                    continue
                # Empty-first placement: if a node has a genuinely idle GPU, use it before
                # stacking onto a warm card. Among warm fitting cards, prefer the lowest
                # post-placement memory pressure. With the 1/3+grace freeze rule, packing
                # the fullest warm card first just pushes it to the freeze line while a
                # cooler sibling stays underused.
                fits_remaining = g["free_mb"] - (task.get("est_vram_mb") or 0)
                used = int(g.get("used_mb") or 0)
                total = max(1, int(g.get("total_mb") or 1))
                need = int(task.get("est_vram_mb") or 0)
                occupied = 1 if used >= runtime.gpu_empty_used_mb else 0
                rt = runtime.candidate_runtime_seconds(task, n["name"], g["idx"])
                rt_unknown = 1 if rt <= 0 else 0
                if occupied:
                    score = (occupied, rt_unknown, rt, float(used + need) / float(total), used + need)
                else:
                    score = (occupied, rt_unknown, rt, fits_remaining)
                score = runtime.algorithm_gpu_score(task, n, g, score)
                out.append((score, n["name"], g["idx"]))
        return out

    def _search(search_nodes):
        # 0) Hard pin — refuse to place anywhere except `require` node.
        if require:
            for n in search_nodes:
                if n["name"] == require:
                    cands = _candidates_for_node(n)
                    if cands:
                        cands.sort()
                        chosen = (cands[0][1], cands[0][2])
                        _trace_candidate_slot("require_node", search_nodes, cands, chosen)
                        return chosen
                    return None  # required node not ready — wait, do not fall back
            return None

        # 1) Resume locality wins over ordinary soft preference. If a
        # checkpoint already exists on a server, launching there avoids
        # silent step-0 restarts and large checkpoint rsyncs.
        tried = set()
        for resume_node in resume_preferred:
            for n in search_nodes:
                if n["name"] != resume_node:
                    continue
                tried.add(n["name"])
                cands = _candidates_for_node(n)
                if cands:
                    cands.sort()
                    chosen = (cands[0][1], cands[0][2])
                    _trace_candidate_slot("resume_preferred", search_nodes, cands, chosen)
                    return chosen

        # 2) Try preferred node next (if specified and alive).
        if preferred:
            for n in search_nodes:
                if n["name"] == preferred:
                    tried.add(n["name"])
                    cands = _candidates_for_node(n)
                    if cands:
                        cands.sort()
                        chosen = (cands[0][1], cands[0][2])
                        _trace_candidate_slot("preferred_node", search_nodes, cands, chosen)
                        return chosen
                    # preferred is alive but full — fall through to fallback search

        # 3) Fallback: scan all nodes (excluding nodes already tried above).
        cands = []
        for n in search_nodes:
            if n["name"] in tried:
                continue
            cands.extend(_candidates_for_node(n))
        if not cands: return None
        cands.sort()
        chosen = (cands[0][1], cands[0][2])
        _trace_candidate_slot("fallback", search_nodes, cands, chosen)
        return chosen

    # Launch-failed nodes are a soft block for unpinned tasks: prefer fresh nodes, but fall
    # back to retrying failed nodes if every viable node has already failed.
    if launch_failed and not require:
        fresh_nodes = [n for n in nodes if n["name"] not in launch_failed]
        placement = _search(fresh_nodes)
        if placement:
            return placement
    return _search(nodes)


def build_no_fit_reason(
    task,
    nodes,
    runtime: PlacementRuntime,
    extra_allowed_nodes=None,
    highlight_nodes=("node007",),
):
    """Explain why pick_placement could not find a candidate."""
    blocked = runtime.blocked_nodes_for_task(task)
    require = task.get("require_node")
    prefer = task.get("preferred_node")
    soft_require_pool = runtime.hpc_cpu_pool_soft_require_nodes(task)
    allowed_for_reason = {str(n) for n in (task.get("allowed_nodes") or []) if str(n)}
    allowed_for_reason.update(str(n) for n in (extra_allowed_nodes or []) if str(n))
    reasons = []
    for n in nodes:
        name = n["name"]
        node_info = runtime.nodes_by_name.get(name, {}) or {}
        if not n.get("alive"):
            reasons.append(f"{name}=DOWN")
            continue
        if n.get("measurement_reservation_active"):
            reservation = n.get("measurement_reservation") or {}
            purpose = str(reservation.get("purpose") or "calibration")
            reasons.append(f"{name}=measurement-reserved({purpose})")
            continue
        if node_info.get("monitor_only"):
            reasons.append(f"{name}=login-node-disabled")
            continue
        if node_info.get("retired"):
            reasons.append(f"{name}=retired")
            continue
        if soft_require_pool and name not in soft_require_pool:
            reasons.append(f"{name}=outside-cpu-pool(node001-node006)")
            continue
        if not soft_require_pool and allowed_for_reason and name not in allowed_for_reason:
            reasons.append(f"{name}=outside-allowed-nodes")
            continue
        if not soft_require_pool and require and require != name:
            reasons.append(f"{name}=require!={require}")
            continue
        node_evict_cooldown = runtime.evict_node_cooldown_block_reason(task, name, node_state=n)
        if node_evict_cooldown:
            reasons.append(f"{name}={node_evict_cooldown}")
            continue
        if name in (blocked or set()):
            reasons.append(f"{name}=blocklisted")
            continue
        cpu_fallback_node = (
            (task.get("est_vram_mb") or 0) > 0
            and (
                runtime.node_is_windows(name)
                or not n.get("gpus")
                or node_info.get("max_vram_per_task") == 0
            )
        )
        if cpu_fallback_node:
            cpu_fb_block = runtime.node_cpu_fallback_block_reason(task, name, node_info)
            if cpu_fb_block:
                reasons.append(f"{name}={cpu_fb_block}")
                continue
            ok_node, why_node = runtime.node_resources_ok(task, n, node_info)
            if not ok_node:
                reasons.append(f"{name}={why_node}")
            else:
                reasons.append(f"{name}=cpu-fallback fits?(unexpected)")
            continue
        gpu_policy_block = runtime.node_gpu_task_block_reason(task, name, node_info)
        if gpu_policy_block:
            reasons.append(f"{name}={gpu_policy_block}")
            continue
        ok_node, why_node = runtime.node_resources_ok(task, n, node_info)
        if not ok_node:
            reasons.append(f"{name}={why_node}")
            continue
        if (task.get("est_vram_mb") or 0) > 0:
            node_info_with_state = dict(node_info)
            node_info_with_state["_scheduler_node_state"] = n
            gpu_reasons = []
            for g in n["gpus"]:
                sub = []
                algo_reason = runtime.algorithm_gpu_fit_block_reason(
                    task, g, node_info_with_state)
                if algo_reason:
                    sub.append(algo_reason)
                max_tasks_per_gpu = runtime.gpu_task_cap_for_task(task, node_info)
                if max_tasks_per_gpu is not None:
                    try:
                        gpu_task_count = int(g.get("running_task_count") or 0)
                        gpu_task_cap = int(max_tasks_per_gpu)
                    except Exception:
                        gpu_task_count = 0
                        gpu_task_cap = 0
                    if (gpu_task_cap > 0 and gpu_task_count >= gpu_task_cap
                            and not runtime.hard_rule_bypassed(
                                "max_tasks_per_gpu", task, node_info=node_info,
                                node_state=n, gpu=g)):
                        sub.append(f"tasks {gpu_task_count}/{gpu_task_cap}")
                if not gpu_fits(task, g, node_info_with_state, runtime):
                    freeze = runtime.gpu_freeze_line_mb(int(g.get("total_mb") or 0))
                    tracked_occupied = int(g.get("running_task_count") or 0) > 0
                    memory_occupied = int(g.get("used_mb") or 0) >= max(
                        1, int(runtime.gpu_empty_used_mb or 0)
                    )
                    if (
                        not runtime.task_ignores_one_third_pack_rule(task, node_info)
                        and (tracked_occupied or memory_occupied)
                        and (
                            g["used_mb"] >= freeze
                            or g["used_mb"] + (task.get("est_vram_mb") or 0) >= freeze
                        )
                    ):
                        sub.append(
                            f"1/3+grace mem ({g['used_mb']}+{task.get('est_vram_mb') or 0}>={freeze}MB)")
                    util_limit = runtime.node_gpu_util_limit(node_info)
                    if util_limit is not None and g["used_mb"] > 100 and g.get("util_pct", 0) >= util_limit:
                        sub.append(f"util {g['util_pct']}%")
                    if (g["free_mb"] < (task.get("est_vram_mb") or 0) + runtime.vram_margin_mb
                            and not runtime.hard_rule_bypassed(
                                "vram_margin", task, node_info=node_info,
                                node_state=n, gpu=g)):
                        sub.append("free<est+margin")
                    cap = node_info.get("max_vram_per_task")
                    if cap is not None and task.get("est_vram_mb", 0) > cap:
                        sub.append(f"per-task cap {cap}")
                if sub:
                    gpu_reasons.append(f"GPU{g['idx']}=" + "&".join(sub))
            if gpu_reasons:
                reasons.append(f"{name}=node-ok-but-" + "/".join(gpu_reasons))
            else:
                reasons.append(f"{name}=fits?(unexpected)")
    if soft_require_pool:
        # The generic first-four truncation used to spend the entire diagnostic
        # budget on local/GPU nodes that this task can never use. Report every
        # actual CPU-pool candidate so the persisted reason explains the real
        # admission gate (CPU, RAM, threads, cooldown, or connectivity).
        pool_names = set(soft_require_pool)
        reported_reasons = [
            reason
            for reason in reasons
            if reason.split("=", 1)[0] in pool_names
        ]
        if not reported_reasons:
            reported_reasons = list(reasons[:4])
        diagnostic_highlights = (require, prefer)
    else:
        reported_reasons = list(reasons[:4])
        diagnostic_highlights = (require, prefer, *tuple(highlight_nodes or ()))
    for extra_node in diagnostic_highlights:
        if not extra_node:
            continue
        for reason in reasons:
            if reason.startswith(f"{extra_node}=") and reason not in reported_reasons:
                reported_reasons.append(reason)
                break
    pin = f"require={require} " if require else (f"prefer={prefer} " if prefer else "")
    return f"no fit ({pin}prio={task.get('priority','normal')}): " + " | ".join(reported_reasons)
