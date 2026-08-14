"""Node-level placement gates and task capability helpers."""

from __future__ import annotations

import json
import shlex
import time as _time
from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class BlockedNodesDeps:
    escalations_file: Any
    project_wide_env_block_ttl_s: int
    now: Callable[[], float] = _time.time
    latest_escalations: Callable[[], dict] | None = None


@dataclass(frozen=True)
class NodeResourceGateDeps:
    default_cpu_cores: int
    default_ram_mb: int
    hard_rule_bypassed: Callable[..., bool]
    ignore_cpu_for_server_gpu_task: Callable[[dict, Optional[dict], Optional[dict]], bool]
    task_is_gpu_capacity_task: Callable[[dict], bool]
    node_ram_headroom_mb: Callable[[Optional[dict], Optional[dict]], int]
    local_gpu_host_cpu_block_pct: int


def task_required_gpu_idx(task: dict) -> Optional[int]:
    raw = task.get("require_gpu_idx")
    if raw is None or raw == "":
        return None
    try:
        idx = int(raw)
    except Exception:
        return None
    return idx if idx >= 0 else None


def python_runtime_identity(command: str) -> str:
    """Return the explicit Python executable token used by a command."""
    try:
        tokens = shlex.split(command or "")
    except ValueError:
        tokens = str(command or "").split()
    for token in tokens:
        executable = token.rsplit("/", 1)[-1].lower()
        if executable == "python" or executable.startswith("python3"):
            return token
    return ""


def python_entrypoint_identity(command: str) -> str:
    """Return the Python script/module launched by a command."""
    try:
        tokens = shlex.split(command or "")
    except ValueError:
        tokens = str(command or "").split()
    python_idx = None
    for idx, token in enumerate(tokens):
        executable = token.rsplit("/", 1)[-1].lower()
        if executable == "python" or executable.startswith("python3"):
            python_idx = idx
            break
    if python_idx is None:
        return ""
    idx = python_idx + 1
    while idx < len(tokens):
        token = tokens[idx]
        if token == "-m":
            return f"module:{tokens[idx + 1]}" if idx + 1 < len(tokens) else ""
        if token == "-c":
            return f"inline:{tokens[idx + 1]}" if idx + 1 < len(tokens) else ""
        if token in ("-W", "-X"):
            idx += 2
            continue
        if token.startswith("-"):
            idx += 1
            continue
        return token
    return ""


def node_numeric(node_state: Optional[dict], *keys, default=None):
    for key in keys:
        if not node_state or key not in node_state:
            continue
        val = node_state.get(key)
        if val is None or val == "":
            continue
        try:
            return float(val)
        except Exception:
            continue
    return default


def local_cpu_pressure_high(node_state: Optional[dict], threshold_pct: int) -> tuple[bool, str]:
    """True when local is actually CPU pressured, not merely over-reserved."""
    host_cpu = node_numeric(node_state, "host_cpu_load_pct")
    if host_cpu is not None:
        high = host_cpu >= threshold_pct
        return high, f"host_cpu={int(host_cpu)}% threshold={threshold_pct}%"
    total = int(node_numeric(node_state, "total_cpu", default=0) or 0)
    free = node_numeric(node_state, "observed_free_cpu", "wsl_free_cpu", "free_cpu")
    load = node_numeric(node_state, "observed_loadavg", "wsl_loadavg", "loadavg")
    if total <= 0:
        return False, "cpu pressure unknown"
    if free is not None:
        high = int(free) <= 0
        return high, f"free_cpu={int(free)}/{total}"
    if load is not None:
        high = float(load) >= max(1, total)
        return high, f"load={float(load):.1f}/{total}"
    return False, "cpu pressure unknown"


def local_gpu_cpu_block_reason(
    task: dict,
    node_state: Optional[dict],
    *,
    task_is_gpu_capacity_task: Callable[[dict], bool],
    threshold_pct: int,
) -> str:
    if (node_state or {}).get("name") != "local":
        return ""
    if not task_is_gpu_capacity_task(task):
        return ""
    high, why = local_cpu_pressure_high(node_state, threshold_pct)
    if high:
        return f"host cpu pressure: {why}"
    return ""


def blocked_nodes_for_task(task: dict, *, deps: BlockedNodesDeps) -> set:
    """Return nodes with pending env/runtime escalations for this task family."""
    sig = task.get("signature") or ""
    cwd = task.get("cwd") or ""
    project = task.get("project") or ""
    if deps.latest_escalations is not None:
        latest = deps.latest_escalations()
    else:
        if not deps.escalations_file.exists():
            return set()
        latest = {}
        try:
            for line in deps.escalations_file.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                tid = rec.get("task_id")
                if tid:
                    latest[tid] = rec
        except Exception:
            return set()
    blocked = set()
    for rec in latest.values():
        if rec.get("status") != "pending":
            continue
        if rec.get("category") not in ("ENV_MISSING", "PYTHON_IMPORT", "CUDA_RUNTIME"):
            continue
        if not rec.get("node"):
            continue
        same_sig = sig and rec.get("signature") == sig
        same_cwd = cwd and rec.get("cwd") == cwd
        same_proj = project and rec.get("project") == project
        category = rec.get("category")
        if category == "PYTHON_IMPORT":
            task_runtime = python_runtime_identity(task.get("cmd") or "")
            failed_runtime = python_runtime_identity(rec.get("cmd") or "")
            task_entrypoint = python_entrypoint_identity(task.get("cmd") or "")
            failed_entrypoint = python_entrypoint_identity(rec.get("cmd") or "")
            same_python_invocation = bool(
                task_runtime
                and failed_runtime
                and task_runtime == failed_runtime
                and task_entrypoint
                and failed_entrypoint
                and task_entrypoint == failed_entrypoint
            )
            same_cwd = same_cwd and same_python_invocation
            same_proj = same_proj and same_python_invocation
        age_s = deps.now() - float(rec.get("ts") or 0)
        # Only the exact failing signature remains blocked until an explicit
        # heal resolves it.  CWD/project matches are a propagation heuristic:
        # treating them as permanent lets one stale task poison every future
        # task submitted from a shared monorepo checkout.
        related_scope_block_active = (
            not same_sig
            and (same_cwd or same_proj)
            and deps.project_wide_env_block_ttl_s > 0
            and age_s <= deps.project_wide_env_block_ttl_s
        )
        if same_sig or related_scope_block_active:
            blocked.add(rec["node"])
    return blocked


def blocked_nodes_for_signature(sig: str, *, deps: BlockedNodesDeps) -> set:
    return blocked_nodes_for_task({"signature": sig}, deps=deps)


def launch_failed_nodes_for_task(
    task: dict,
    *,
    known_nodes: dict,
    soft_block_ttl_s: int = 0,
    now: Callable[[], float] = _time.time,
) -> set:
    raw = task.get("launch_failed_nodes") or {}
    if isinstance(raw, dict):
        blocked = set()
        current = now()
        for node, rec in raw.items():
            if node not in known_nodes:
                continue
            if soft_block_ttl_s > 0 and isinstance(rec, dict):
                try:
                    failed_at = float(rec.get("ts") or 0)
                except Exception:
                    failed_at = 0
                if failed_at > 0 and current - failed_at > soft_block_ttl_s:
                    continue
            blocked.add(node)
        return blocked
    if isinstance(raw, list):
        return {n for n in raw if n in known_nodes}
    return set()


def node_thread_pressure_block_reason(task: dict, node_state: Optional[dict],
                                      node_info: Optional[dict]) -> str:
    """Block new GPU launches when a shared node is near its process/thread cap."""
    node_info = node_info or {}
    node_state = node_state or {}
    is_gpu = int(task.get("est_vram_mb") or 0) > 0
    is_jax_cpu = task_is_thread_heavy_jax_cpu(task)
    if is_gpu:
        min_free = int(
            node_info.get("min_free_user_threads_for_gpu_task") or 0)
        max_frac = float(
            node_info.get("max_user_thread_fraction_for_gpu_task") or 0.0)
    elif is_jax_cpu:
        min_free = int(
            node_info.get("min_free_user_threads_for_jax_cpu_task") or 0)
        max_frac = float(
            node_info.get("max_user_thread_fraction_for_jax_cpu_task") or 0.0)
    else:
        return ""
    if min_free <= 0 and max_frac <= 0:
        return ""
    used = node_numeric(node_state, "user_threads")
    limit = node_numeric(node_state, "user_thread_limit")
    if used is None or limit is None:
        return "user thread pressure: probe missing user_threads/user_thread_limit"
    used = int(used)
    limit = int(limit)
    if limit <= 0:
        return ""
    free = max(0, limit - used)
    if min_free > 0 and free < min_free:
        return (
            f"user thread pressure: free {free}/{limit} "
            f"(used {used}, reserve {min_free})"
        )
    if max_frac > 0 and used / max(1, limit) >= max_frac:
        return (
            f"user thread pressure: used {used}/{limit} "
            f"({used / max(1, limit):.0%}, limit {max_frac:.0%})"
        )
    return ""


def task_is_thread_heavy_jax_cpu(task: dict) -> bool:
    if int((task or {}).get("est_vram_mb") or 0) > 0:
        return False
    text = task_capability_text(task)
    return "jax_platforms=cpu" in text or "jax_platform_name=cpu" in text


def task_requires_persistent_cpu_reservation(task: dict) -> bool:
    """Keep declared slots for CPU jobs whose utilization arrives in bursts."""
    if int((task or {}).get("est_vram_mb") or 0) > 0:
        return False
    text = task_capability_text(task)
    canonical_saas = (
        "botorch_saasbo" in text
        and (
            "--saas-refit-schedule every_iteration" in text
            or "--saas-refit-schedule=every_iteration" in text
        )
    )
    traffic_saas = "benchmark_traffic_final_contract.py" in text
    return canonical_saas or traffic_saas


def node_concurrency_cap_for_task(task: dict, node_info: dict):
    cap = node_info.get("max_concurrent_running")
    threshold = int(node_info.get("small_vram_task_threshold_mb") or 0)
    small_cap = node_info.get("small_vram_max_concurrent_running")
    est_vram = int((task or {}).get("est_vram_mb") or 0)
    if threshold > 0 and est_vram > 0 and est_vram <= threshold and small_cap is not None:
        return small_cap
    return cap


def gpu_task_cap_for_task(task: dict, node_info: dict):
    cap = node_info.get("max_tasks_per_gpu")
    threshold = int(node_info.get("small_vram_task_threshold_mb") or 0)
    small_cap = node_info.get("small_vram_max_tasks_per_gpu")
    est_vram = int((task or {}).get("est_vram_mb") or 0)
    if threshold > 0 and est_vram > 0 and est_vram <= threshold and small_cap is not None:
        return small_cap
    return cap


def node_schedulable_cpu_free(task: dict, node_state: dict, node_info: dict) -> tuple[int, str, int, int | None]:
    """CPU free used for placement, including optional live backfill."""
    raw_free = int((node_state or {}).get("free_cpu", 0) or 0)
    observed = (node_state or {}).get("observed_free_cpu")
    observed_int = None
    try:
        if observed is not None:
            observed_int = max(0, int(observed))
    except Exception:
        observed_int = None
    candidate_widths = []
    plan = (task or {}).get("cpu_batch_plan")
    for value in (
        (task or {}).get("cpu_declared_cores"),
        (task or {}).get("cpu_auto_workers"),
        plan.get("workers") if isinstance(plan, dict) else None,
        (task or {}).get("cpu_cores"),
    ):
        try:
            parsed = int(value or 0)
        except Exception:
            parsed = 0
        if parsed > 0:
            candidate_widths.append(parsed)
    candidate_width = max(candidate_widths or [1])
    live_max_task_cores = max(
        1, int((node_info or {}).get("live_cpu_backfill_max_task_cores") or 1)
    )
    live_enabled = (
        bool((node_info or {}).get("live_cpu_backfill"))
        and bool((node_state or {}).get("cpu_slot_accounting"))
        and int((task or {}).get("est_vram_mb") or 0) <= 0
        and candidate_width <= live_max_task_cores
        and observed_int is not None
    )
    persistent_reservation = task_requires_persistent_cpu_reservation(task)
    use_live = live_enabled and not persistent_reservation
    hard_free = (node_state or {}).get("cpu_hard_free")
    try:
        hard_free_int = max(0, int(hard_free)) if hard_free is not None else None
    except Exception:
        hard_free_int = None
    reserved_cpu = max(0, int((node_info or {}).get("reserved_cpu_cores") or 0))
    persistent_declared_cap = max(
        0,
        int(
            (node_info or {}).get(
                "persistent_cpu_live_backfill_max_declared_cores", 0
            )
            or 0
        ),
    )
    if live_enabled and persistent_reservation and persistent_declared_cap > 0:
        total_cpu = max(
            0,
            int(
                (node_state or {}).get("total_cpu")
                or (node_info or {}).get("cpu_cores")
                or 0
            ),
        )
        try:
            slot_reserved = max(
                0, int((node_state or {}).get("cpu_slot_reserved") or 0)
            )
        except Exception:
            slot_reserved = max(0, total_cpu - raw_free)
        declared_budget = max(
            0, persistent_declared_cap - slot_reserved
        )
        live_budget = max(0, int(observed_int) - reserved_cpu)
        if hard_free_int is not None:
            live_budget = min(
                live_budget, max(0, hard_free_int - reserved_cpu)
            )
        return (
            min(declared_budget, live_budget),
            "persistent_live_capped",
            raw_free,
            observed_int,
        )
    if use_live:
        base_free = observed_int
        source = "live"
        if hard_free_int is not None:
            base_free = min(base_free, hard_free_int)
            source = "live_guarded"
    else:
        base_free = raw_free
        source = "slot"
        if observed_int is not None:
            base_free = min(base_free, observed_int)
            if observed_int < raw_free:
                source = "slot_observed_guarded"
    return max(0, int(base_free) - reserved_cpu), source, raw_free, observed_int


def node_resources_ok(task: dict, node_state: dict, node_info: dict,
                      *, deps: NodeResourceGateDeps) -> tuple[bool, str]:
    """CPU + RAM + concurrency check at node level."""
    cap = node_concurrency_cap_for_task(task, node_info)
    cur_running = node_state.get("running_count", 0)
    if (cap is not None and cur_running >= cap
            and not deps.hard_rule_bypassed("node_concurrency", task, node_info=node_info, node_state=node_state)):
        return False, f"concurrency cap: {cur_running}/{cap} tasks already running on {node_state.get('name','?')}"
    thread_block = node_thread_pressure_block_reason(task, node_state, node_info)
    if thread_block and not deps.hard_rule_bypassed("thread_pressure", task, node_info=node_info, node_state=node_state):
        return False, thread_block
    if not deps.ignore_cpu_for_server_gpu_task(task, node_state, node_info):
        local_gpu_block = local_gpu_cpu_block_reason(
            task,
            node_state,
            task_is_gpu_capacity_task=deps.task_is_gpu_capacity_task,
            threshold_pct=deps.local_gpu_host_cpu_block_pct,
        )
        if local_gpu_block:
            return False, local_gpu_block
        if not ((node_state or {}).get("name") == "local" and deps.task_is_gpu_capacity_task(task)):
            needed_cpu = task.get("cpu_cores", deps.default_cpu_cores)
            sched_free_cpu, cpu_source, slot_free_cpu, observed_free_cpu = node_schedulable_cpu_free(
                task, node_state, node_info
            )
            if sched_free_cpu < needed_cpu:
                reserve_note = ""
                reserved_cpu = max(0, int((node_info or {}).get("reserved_cpu_cores") or 0))
                if reserved_cpu:
                    reserve_note = f" reserve={reserved_cpu}"
                source_note = ""
                if observed_free_cpu is not None and cpu_source != "slot":
                    source_note = (
                        f" observed_free={observed_free_cpu}"
                        f" slot_free={slot_free_cpu}"
                    )
                    startup_hold = max(
                        0,
                        int((node_state or {}).get("cpu_hard_reserved") or 0),
                    )
                    if startup_hold:
                        source_note += f" startup_hold={startup_hold}"
                        startup_remaining_s = max(
                            0,
                            int((node_state or {}).get(
                                "cpu_startup_hold_remaining_s"
                            ) or 0),
                        )
                        if startup_remaining_s:
                            source_note += f" hold_release<={startup_remaining_s}s"
                return False, (
                    f"cpu: need {needed_cpu}, free {sched_free_cpu}/"
                    f"{node_state.get('total_cpu', '?')}"
                    f"{source_note}{reserve_note}"
                )
    needed_ram = task.get("ram_mb", deps.default_ram_mb)
    headroom = deps.node_ram_headroom_mb(node_state, node_info)
    if node_state.get("free_ram_mb", 0) - needed_ram < headroom:
        return False, f"ram: need {needed_ram}MB, free {node_state.get('free_ram_mb', 0)}MB (headroom {headroom}MB)"
    return True, "ok"


def task_capability_text(task: dict) -> str:
    return "\n".join(str(task.get(k) or "") for k in (
        "cmd", "cwd", "description", "signature", "project"
    )).lower()


def hpc_cpu_pool_soft_require_nodes(task: dict) -> list:
    if int((task or {}).get("est_vram_mb") or 0) > 0:
        return []
    require = (task or {}).get("require_node")
    pool = [f"node{i:03d}" for i in range(1, 7)]
    text = task_capability_text(task)
    project = str((task or {}).get("project") or "").lower()
    is_kg_synth = project in {"kg-synth", "kg_synth"} or any(
        marker in text for marker in (
            "kg_op/",
            "/kg_op",
            "sc-olh-kg",
            "kg_op_scheduler_deploy",
            "benchmark_lodo_meta_prior.py",
        )
    )
    if require not in pool and require != "zhengliang-hpc" and not is_kg_synth:
        return []
    is_freq_hrl = "freq_hrl" in text and (
        "/scheduleurm_work/transitduet" in text
        or "transit_hrl/freq_hrl" in text
    )
    is_freqduet = "freqduet" in text and (
        "run_freqduet_ablation.py" in text
        or "run_freqduet_external_baselines.py" in text
        or "/mine_code/transitduet/freqduet/freqduet" in text
        or "/scheduleurm_work/transitduet/freqduet/freqduet" in text
    )
    transitduet_staged_pool = (
        "/mine_code/transitduet/transit_duet" in text
        or "/scheduleurm_work/transitduet/transit_duet" in text
        or "/transitduet/transit_duet" in text
    )
    is_transitduet_cpu_pool = project == "transitduet" and transitduet_staged_pool and any(
        script in text for script in (
            "runner_v3.py",
            "run_round3_node_worker.py",
            "run_baseline_rule.py",
            "run_upper_comparison.py",
        )
    )
    is_bamor = "bamor" in text and (
        "train_bamor_mujoco.py" in text
        or "/scheduleurm_work/bamor" in text
        or "bamor/mujoco" in text
    )
    is_bapr = project in {"bapr", "cs-bapr", "cs_bapr"} or (
        ("bapr" in text or "cs-bapr" in text or "cs_bapr" in text)
        and (
            "run_seed.sh" in text
            or "/mine_code/bapr" in text
            or "/scheduleurm_work/bapr" in text
            or "bapr/" in text
        )
    )
    is_scheduleurm = project in {"scheduleurm", "scheduleurmbench"} or (
        "scheduleurmbench/" in text
        or "/mine_code/scheduleurm" in text
        or "/scheduleurm_work/scheduleurm" in text
    )
    is_laplace = project in {"laplace-smdp", "laplace-semi-mdp"} and (
        "run_planner_baseline_comparison.py" in text
        or "/scheduleurm_work/laplace-semi-mdp" in text
    )
    if not (is_freq_hrl or is_freqduet or is_transitduet_cpu_pool
            or is_bamor or is_bapr or is_scheduleurm or is_kg_synth or is_laplace):
        return []
    return pool


def task_gpu_capability(task: dict) -> str:
    """Classify the GPU runtime a task needs for node compatibility checks."""
    text = task_capability_text(task)
    if any(p in text for p in ("resac-jax", "jax_experiments", "xla_python_client", "xla_flags")):
        return "jax_cuda"
    if any(p in text for p in ("torch", "pytorch", "sac_ensemble_original_logging.py")):
        return "torch_cuda"
    return "cuda"


def task_cpu_fallback_capability(task: dict) -> str:
    gpu_cap = task_gpu_capability(task)
    if gpu_cap == "jax_cuda":
        return "jax_cpu"
    if gpu_cap == "torch_cuda":
        return "torch_cpu"
    return "cpu"


def node_cpu_fallback_block_reason(
    task: dict,
    node_name: str,
    node_info: dict,
    *,
    task_cpu_fallback_capability: Callable[[dict], str] = task_cpu_fallback_capability,
) -> str | None:
    """Return why a GPU task cannot opportunistically run CPU-only on this node."""
    if (task.get("est_vram_mb") or 0) <= 0:
        return None
    if not bool(task.get("allow_cpu_training", False)):
        return (
            "cpu-fallback disabled: task has vram>0 and is treated as GPU-required; "
            "resubmit with --allow-cpu-training only if CPU training is intentional"
        )
    caps = set(str(c).lower() for c in ((node_info or {}).get("capabilities") or []))
    need = task_cpu_fallback_capability(task)
    if caps and need not in caps:
        return f"cpu-fallback capability block: need {need}"
    return None


def node_gpu_task_block_reason(task: dict, node_name: str, node_info: dict) -> str | None:
    """Return a node policy reason if this GPU task should not run on node_name."""
    if (task.get("est_vram_mb") or 0) <= 0:
        return None
    project = str(task.get("project") or "").strip()
    blocked_projects = {
        str(value).strip().casefold()
        for value in ((node_info or {}).get("blocked_gpu_projects") or [])
        if str(value).strip()
    }
    if project and project.casefold() in blocked_projects:
        return f"gpu-policy block: project {project} is reserved off this node"
    caps = set(str(c).lower() for c in ((node_info or {}).get("capabilities") or []))
    need = task_gpu_capability(task)
    if caps and need not in caps and "cuda" not in caps:
        return f"gpu capability block: need {need}"
    if caps and need not in caps and need != "cuda":
        return f"gpu capability block: need {need}"
    patterns = (node_info or {}).get("blocked_gpu_cmd_patterns") or []
    if not patterns:
        return None
    text = task_capability_text(task)
    for pattern in patterns:
        pat = str(pattern or "").strip()
        if pat and pat.lower() in text:
            return f"gpu-policy block: matches {pat}"
    return None
