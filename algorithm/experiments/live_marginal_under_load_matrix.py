"""Live add-one ETA probes under resident load states.

This runner measures the quantity the empty-profile matrix cannot certify:
the marginal stable service rate of an additional task when the target resource
already has resident load.  It runs only controlled benchmark commands and
cleans up resident processes before writing theorem-facing rows.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from algorithm.features import hardware_class_for_node
from simulation.service_cache import ProfileRecord, ServiceRateCache

from .remote_workload_selected_profile_probe import (
    REMOTE_PROGRESS_WRAPPER_DIR,
    _deploy_progress_wrapper,
    _last_rate,
    _last_stable_rate,
    _run_remote_capture,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = Path.home() / ".claude" / "scheduler" / "experiments" / "runs"
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_DESIGN_CSV = ARTIFACT_ROOT / "full_factorial_eta_design_rows_20260629.csv"
CPU_FULL_RESERVE_DIVISOR = 16
RESIDENT_BACKGROUND_TOTAL_UNITS = 1_000_000
RL_CUDA_MEMORY_BUDGET = 0.70


def _resident_worker_counts(cpu_cores: int) -> tuple[int, int]:
    """Return half/full resident workers while retaining host headroom."""

    cores = max(1, int(cpu_cores))
    half = max(1, cores // 2)
    reserve = max(1, (cores + CPU_FULL_RESERVE_DIVISOR - 1) // CPU_FULL_RESERVE_DIVISOR)
    full = max(1, cores - reserve)
    return half, full


def _rl_cuda_memory_fraction(kind: str, concurrent_cuda_processes: int) -> float:
    """Bound per-process JAX memory so controlled co-location stays below 70%."""

    if kind.startswith("bapr_ant_"):
        requested = 0.18 if kind.endswith("_light") else 0.40
    else:
        requested = 0.10 if kind.endswith("_light") else 0.18
    shared_limit = RL_CUDA_MEMORY_BUDGET / max(1, int(concurrent_cuda_processes))
    return min(requested, shared_limit)


@dataclass(frozen=True)
class MarginalScenario:
    scenario_id: str
    node: str
    gpu: int | None
    workload_key: str
    workload_env: str
    resource_kind: str
    resource_state: str
    profile: int
    background_kind: str
    add_kind: str
    background_kinds: tuple[str, ...] = ()
    warmup_s: int = 18
    timeout_s: int = 240
    total_units: float = 80.0
    note: str = ""
    resident_mix: str = ""


DEFAULT_SCENARIOS = (
    MarginalScenario(
        "jtl110_cnn_half_loaded_add_cnn",
        "jtl110gpu",
        0,
        "gpu_cnn_torch_resnet50",
        "cnn",
        "gpu_cnn",
        "half_loaded",
        2,
        "cnn_resident",
        "cnn_add",
        note="one resident CNN plus one add-one CNN on the same 3080Ti GPU",
    ),
    MarginalScenario(
        "jtl311_cnn_half_loaded_add_cnn",
        "jtl311linux",
        0,
        "gpu_cnn_torch_resnet50",
        "cnn",
        "gpu_cnn",
        "half_loaded",
        2,
        "cnn_resident",
        "cnn_add",
        note="one resident CNN plus one add-one CNN on the CPU-fast RTX2080 node",
    ),
    MarginalScenario(
        "jtl110_cnn_high_vram_resident_add_cnn",
        "jtl110gpu",
        1,
        "gpu_cnn_torch_resnet50",
        "cnn",
        "gpu_cnn",
        "high_vram_resident",
        1,
        "high_vram_resident",
        "cnn_add_light",
        warmup_s=12,
        timeout_s=260,
        note="high-VRAM resident matrix job plus one add-one CNN",
    ),
    MarginalScenario(
        "jtl110_cnn_full_loaded_add_cnn",
        "jtl110gpu",
        0,
        "gpu_cnn_torch_resnet50",
        "cnn",
        "gpu_cnn",
        "full_loaded",
        4,
        "cnn_resident",
        "cnn_add_light",
        warmup_s=22,
        timeout_s=300,
        note="three resident CNN jobs plus one add-one CNN on the same 3080Ti GPU",
    ),
    MarginalScenario(
        "jtl110_llm_resident_add_cnn",
        "jtl110gpu",
        1,
        "gpu_cnn_torch_resnet50",
        "cnn",
        "gpu_cnn",
        "mixed_colocation",
        2,
        "llm_resident",
        "cnn_add_light",
        warmup_s=20,
        timeout_s=320,
        note="one resident LLM decoder job plus one add-one CNN on the same 3080Ti GPU",
    ),
    MarginalScenario(
        "jtl110_llm_rl_resident_add_cnn",
        "jtl110gpu",
        0,
        "gpu_cnn_torch_resnet50",
        "cnn",
        "gpu_cnn",
        "mixed_colocation",
        3,
        "mixed_llm_rl_resident",
        "cnn_add_light",
        background_kinds=("llm_resident_light", "rl_resac_ant_resident_light"),
        warmup_s=180,
        timeout_s=420,
        total_units=120.0,
        note=(
            "resident LLM plus resident RE-SAC Ant pressure, then one add-one "
            "CNN on the same 3080Ti GPU"
        ),
    ),
    MarginalScenario(
        "jtl110gpu2_cnn_high_vram_resident_p2_add_cnn",
        "jtl110gpu2",
        0,
        "gpu_cnn_torch_resnet50",
        "cnn",
        "gpu_cnn",
        "high_vram_resident",
        2,
        "high_vram_resident",
        "cnn_add_light",
        warmup_s=14,
        timeout_s=300,
        total_units=120.0,
        note="same-class spillover: high-VRAM resident plus one add-one CNN on jtl110gpu2",
    ),
    MarginalScenario(
        "jtl110gpu2_cnn_high_vram_resident_p3_add_cnn",
        "jtl110gpu2",
        0,
        "gpu_cnn_torch_resnet50",
        "cnn",
        "gpu_cnn",
        "high_vram_resident",
        3,
        "high_vram_resident",
        "cnn_add_light",
        warmup_s=16,
        timeout_s=340,
        total_units=120.0,
        note="same-class spillover: two high-VRAM residents plus one add-one CNN on jtl110gpu2",
    ),
    MarginalScenario(
        "jtl110gpu2_llm_resident_add_cnn_p1",
        "jtl110gpu2",
        1,
        "gpu_cnn_torch_resnet50",
        "cnn",
        "gpu_cnn",
        "mixed_colocation",
        1,
        "llm_resident",
        "cnn_add_light",
        warmup_s=20,
        timeout_s=320,
        total_units=120.0,
        note="same-class spillover: resident LLM plus one add-one CNN on jtl110gpu2",
    ),
    MarginalScenario(
        "jtl110gpu2_rl_resident_add_cnn_p1",
        "jtl110gpu2",
        0,
        "gpu_cnn_torch_resnet50",
        "cnn",
        "gpu_cnn",
        "mixed_colocation",
        1,
        "rl_resac_ant_resident_light",
        "cnn_add_light",
        warmup_s=180,
        timeout_s=420,
        total_units=120.0,
        note="same-class spillover: resident light RE-SAC Ant plus one add-one CNN on jtl110gpu2",
    ),
    MarginalScenario(
        "jtl110gpu2_llm_rl_resident_add_cnn_p1",
        "jtl110gpu2",
        0,
        "gpu_cnn_torch_resnet50",
        "cnn",
        "gpu_cnn",
        "mixed_colocation",
        1,
        "mixed_llm_rl_resident",
        "cnn_add_light",
        background_kinds=("llm_resident_light", "rl_resac_ant_resident_light"),
        warmup_s=180,
        timeout_s=420,
        total_units=120.0,
        note="same-class spillover: resident LLM plus resident light RE-SAC Ant plus one add-one CNN on jtl110gpu2",
    ),
    MarginalScenario(
        "node001_cpu_half_resident_add_cpu",
        "node001",
        None,
        "cpu_heavy_local_bench",
        "cpu",
        "cpu_heavy",
        "cpu_resident",
        1,
        "cpu_half_resident",
        "cpu_add_single",
        warmup_s=8,
        timeout_s=220,
        total_units=40.0,
        note="half-capacity resident CPU job scaled to target physical cores plus one small CPU probe",
    ),
    MarginalScenario(
        "node001_cpu_full_resident_add_cpu",
        "node001",
        None,
        "cpu_heavy_local_bench",
        "cpu",
        "cpu_heavy",
        "cpu_resident",
        1,
        "cpu_full_resident",
        "cpu_add_single",
        warmup_s=10,
        timeout_s=260,
        total_units=40.0,
        note="full resident CPU job scaled to target physical cores plus one small CPU probe",
    ),
    MarginalScenario(
        "node003_cpu_full_resident_add_cpu",
        "node003",
        None,
        "cpu_heavy_local_bench",
        "cpu",
        "cpu_heavy",
        "cpu_resident",
        1,
        "cpu_full_resident",
        "cpu_add_single",
        warmup_s=10,
        timeout_s=260,
        total_units=40.0,
        note="full resident CPU job scaled to target physical cores on an idle 192-core representative",
    ),
    MarginalScenario(
        "node005_cpu_half_resident_add_cpu",
        "node005",
        None,
        "cpu_heavy_local_bench",
        "cpu",
        "cpu_heavy",
        "cpu_resident",
        1,
        "cpu_half_resident",
        "cpu_add_single",
        warmup_s=8,
        timeout_s=220,
        total_units=40.0,
        note="half-capacity resident CPU equivalence check on a second 192-core node",
    ),
    MarginalScenario(
        "node002_light_full_resident_add_light",
        "node002",
        None,
        "light_control_local",
        "light",
        "light_control",
        "cpu_resident",
        1,
        "cpu_full_resident",
        "cpu_add_light",
        warmup_s=10,
        timeout_s=220,
        total_units=40.0,
        note="full resident CPU job plus one light-control add-one probe on the 192-core class",
    ),
    MarginalScenario(
        "node004_freqduet_full_resident_add_cpu",
        "node004",
        None,
        "freqduet_cpu_ablation_c17_32",
        "freqduet",
        "freqduet_cpu",
        "cpu_resident",
        1,
        "cpu_full_resident",
        "cpu_add_single",
        warmup_s=10,
        timeout_s=260,
        total_units=40.0,
        note="full resident CPU job plus one FreqDuet-like add-one probe on the 192-core class",
    ),
    MarginalScenario(
        "node006_sumo_full_resident_add_cpu",
        "node006",
        None,
        "sumo_eval_cpu",
        "sumo",
        "sumo_cpu",
        "cpu_resident",
        1,
        "cpu_full_resident",
        "cpu_add_single",
        warmup_s=10,
        timeout_s=260,
        total_units=40.0,
        note="full resident CPU job plus one SUMO-like add-one probe on the 192-core class",
    ),
)


def build_live_marginal_under_load_matrix(
    *,
    scenarios: Iterable[str] | None = None,
    run_id: str = "live_marginal_under_load_matrix_20260629",
    allow_launch: bool = False,
) -> dict[str, Any]:
    selected_ids = {str(x) for x in scenarios or []}
    selected = [
        row for row in DEFAULT_SCENARIOS
        if not selected_ids or row.scenario_id in selected_ids
    ]
    rows: list[dict[str, Any]] = []
    cache = ServiceRateCache()
    for scenario in selected:
        row = _run_scenario(scenario, run_id=run_id) if allow_launch else _manifest_row(scenario, run_id)
        rows.append(row)
        record = _record_from_row(row)
        if record is not None:
            cache.add(record, force_replace=True)
    admitted = [row for row in rows if row.get("measurement_valid")]
    boundaries = [row for row in rows if row.get("launched") and not row.get("measurement_valid")]
    pending = [row for row in rows if not row.get("launched")]
    return {
        "gate": "live_marginal_under_load_matrix",
        "run_id": run_id,
        "allow_launch": bool(allow_launch),
        "pass": bool(rows) and not pending and bool(admitted),
        "status": (
            "LIVE_MARGINAL_UNDER_LOAD_READY"
            if allow_launch and bool(rows) and not pending and bool(admitted)
            else ("LIVE_MARGINAL_UNDER_LOAD_OPEN" if allow_launch else "DRY_RUN_READY")
        ),
        "admitted_count": len(admitted),
        "boundary_count": len(boundaries),
        "pending_count": len(pending),
        "rows": rows,
        "admitted_rows": admitted,
        "boundary_rows": boundaries,
        "pending_rows": pending,
        "service_cache_snapshot": cache.snapshot(),
        "scope": (
            "Controlled add-one probes under resident load. These rows certify "
            "marginal ETA/service for matching resource_state only; they do not "
            "replace empty-resource rows."
        ),
    }


def build_live_marginal_from_design(
    *,
    design_csv: Path = DEFAULT_DESIGN_CSV,
    run_id: str = "live_marginal_from_design_20260630",
    allow_launch: bool = False,
    assigned_node: str = "",
    logical_nodes: Iterable[str] | None = None,
    hardware_groups: Iterable[str] | None = None,
    workloads: Iterable[str] | None = None,
    states: Iterable[str] | None = None,
    resident_mixes: Iterable[str] | None = None,
    max_rows: int = 1,
) -> dict[str, Any]:
    scenarios = _design_scenarios(
        design_csv,
        assigned_node=str(assigned_node or ""),
        logical_nodes={str(x) for x in logical_nodes or [] if str(x)},
        hardware_groups={str(x) for x in hardware_groups or [] if str(x)},
        workloads={str(x) for x in workloads or [] if str(x)},
        states={str(x) for x in states or [] if str(x)},
        resident_mixes={str(x) for x in resident_mixes or [] if str(x)},
        max_rows=int(max_rows),
    )
    rows: list[dict[str, Any]] = []
    cache = ServiceRateCache()
    for scenario in scenarios:
        row = _run_scenario(scenario, run_id=run_id) if allow_launch else _manifest_row(scenario, run_id)
        rows.append(row)
        record = _record_from_row(row)
        if record is not None:
            cache.add(record, force_replace=True)
    admitted = [row for row in rows if row.get("measurement_valid")]
    boundaries = [row for row in rows if row.get("launched") and not row.get("measurement_valid")]
    pending = [row for row in rows if not row.get("launched")]
    return {
        "gate": "live_marginal_from_design",
        "run_id": run_id,
        "allow_launch": bool(allow_launch),
        "pass": bool(rows) and not pending and bool(admitted),
        "status": (
            "LIVE_MARGINAL_DESIGN_READY"
            if allow_launch and bool(rows) and not pending and bool(admitted)
            else ("LIVE_MARGINAL_DESIGN_OPEN" if allow_launch else "DRY_RUN_READY")
        ),
        "design_csv": str(design_csv),
        "assigned_node": str(assigned_node or ""),
        "selected_scenario_count": len(scenarios),
        "admitted_count": len(admitted),
        "boundary_count": len(boundaries),
        "pending_count": len(pending),
        "rows": rows,
        "admitted_rows": admitted,
        "boundary_rows": boundaries,
        "pending_rows": pending,
        "service_cache_snapshot": cache.snapshot(),
        "scope": (
            "Generated from full-factorial T2 design rows. Each selected row "
            "uses a resident/mixed-load harness and task-native progress ETA."
        ),
    }


def _manifest_row(scenario: MarginalScenario, run_id: str) -> dict[str, Any]:
    return {
        **_scenario_base(scenario, run_id=run_id),
        "launched": False,
        "measurement_valid": False,
        "reason": "dry_run_manifest",
    }


def _design_scenarios(
    design_csv: Path,
    *,
    assigned_node: str,
    logical_nodes: set[str],
    hardware_groups: set[str],
    workloads: set[str],
    states: set[str],
    resident_mixes: set[str],
    max_rows: int,
) -> list[MarginalScenario]:
    out: list[MarginalScenario] = []
    with design_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if "pending" not in str(row.get("status") or ""):
                continue
            if str(row.get("tier") or "") != "T2_load_state":
                continue
            if logical_nodes and str(row.get("node") or "") not in logical_nodes:
                continue
            if hardware_groups and str(row.get("hardware_group") or "") not in hardware_groups:
                continue
            if workloads and str(row.get("workload_key") or "") not in workloads:
                continue
            if states and str(row.get("resource_state") or "") not in states:
                continue
            if resident_mixes and str(row.get("resident_mix") or "") not in resident_mixes:
                continue
            scenario = _scenario_from_design_row(row, assigned_node=assigned_node)
            if scenario is None:
                continue
            out.append(scenario)
            if max_rows > 0 and len(out) >= max_rows:
                break
    return out


def _scenario_from_design_row(row: Mapping[str, str], *, assigned_node: str = "") -> MarginalScenario | None:
    resource_state = str(row.get("resource_state") or "")
    resident_mix = str(row.get("resident_mix") or "")
    workload_key = str(row.get("workload_key") or "")
    workload_env = str(row.get("workload_env") or "")
    missing = _parse_profile_list(str(row.get("missing_profiles") or ""))
    if resource_state not in {"high_vram_resident", "cpu_resident", "mixed_colocation"}:
        return None
    if not missing:
        return None
    add_kind = _add_kind_for_workload(workload_key, workload_env)
    if not add_kind:
        return None
    background_kinds = _background_kinds_for_mix(resident_mix, workload_key, workload_env, resource_state)
    if not background_kinds:
        return None
    node = str(assigned_node or row.get("node") or "")
    if not node:
        return None
    profile = int(missing[0])
    is_gpu = str(row.get("resource_kind") or "") == "gpu"
    long_rl = any(kind.startswith(("rl_resac_", "bapr_ant_")) for kind in background_kinds + (add_kind,))
    scenario_id = _safe_token(
        "design_"
        f"{node}_{workload_key}_{workload_env}_{resource_state}_{resident_mix}_p{profile}"
    )
    gpu_total_units = 360.0 if add_kind.startswith("llm_") else 120.0
    gpu_timeout_s = 620 if add_kind.startswith("llm_") else 340
    return MarginalScenario(
        scenario_id=scenario_id,
        node=node,
        gpu=0 if is_gpu else None,
        workload_key=workload_key,
        workload_env=workload_env,
        resource_kind=str(row.get("family") or row.get("resource_kind") or ""),
        resource_state=resource_state,
        profile=profile,
        background_kind=background_kinds[0],
        add_kind=add_kind,
        background_kinds=background_kinds,
        warmup_s=180 if long_rl else (16 if is_gpu else 10),
        timeout_s=480 if long_rl else (gpu_timeout_s if is_gpu else 260),
        total_units=gpu_total_units if is_gpu else 40.0,
        resident_mix=resident_mix,
        note=f"generated from design row {row.get('row_id')}",
    )


def _parse_profile_list(text: str) -> list[int]:
    out = []
    for part in str(text or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            continue
    return sorted(set(out))


def _add_kind_for_workload(workload_key: str, workload_env: str) -> str:
    key = str(workload_key or "")
    env = str(workload_env or "ant").lower()
    if key == "gpu_cnn_torch_resnet50":
        return "cnn_add_light"
    if key == "gpu_llm_distilgpt2":
        return "llm_add_light"
    if key == "gpu_heavy_jax_matmul":
        return "matmul_add_light"
    if key.startswith("hybrid_rl_resac_"):
        return f"rl_resac_{env}_add_light"
    if key == "hybrid_rl_bapr_ant":
        return "bapr_ant_add_light"
    if key == "light_control_local":
        return "cpu_add_light"
    if key in {"cpu_heavy_local_bench", "freqduet_cpu_ablation_c17_32", "sumo_eval_cpu"}:
        return "cpu_add_single"
    return ""


def _component_for_workload(workload_key: str, workload_env: str) -> str:
    key = str(workload_key or "")
    if key == "gpu_cnn_torch_resnet50":
        return "cnn"
    if key == "gpu_llm_distilgpt2":
        return "llm"
    if key.startswith("hybrid_rl_"):
        return "hybrid_rl"
    if key == "gpu_heavy_jax_matmul":
        return "gpu_matmul"
    return str(workload_env or key)


def _background_kinds_for_mix(
    resident_mix: str,
    workload_key: str,
    workload_env: str,
    resource_state: str,
) -> tuple[str, ...]:
    if resource_state == "high_vram_resident":
        return ("high_vram_resident",)
    if resource_state == "cpu_resident":
        return ("cpu_full_resident",)
    components: dict[str, str] = {
        "cnn": "cnn_resident",
        "llm": "llm_resident_light",
        "hybrid_rl": "rl_resac_ant_resident_light",
    }
    mix_components = {
        "cnn_plus_llm": ("cnn", "llm"),
        "cnn_plus_hybrid_rl": ("cnn", "hybrid_rl"),
        "llm_plus_hybrid_rl": ("llm", "hybrid_rl"),
        "cnn_plus_llm_plus_hybrid_rl": ("cnn", "llm", "hybrid_rl"),
        "cpu_plus_gpu_target": ("cpu",),
    }.get(str(resident_mix or ""), ())
    if mix_components == ("cpu",):
        return ("cpu_full_resident",)
    target = _component_for_workload(workload_key, workload_env)
    resident_components = [component for component in mix_components if component != target]
    if not resident_components and mix_components:
        resident_components = list(mix_components)
    return tuple(components[component] for component in resident_components if component in components)


def _run_scenario(scenario: MarginalScenario, *, run_id: str) -> dict[str, Any]:
    run_dir = RUN_ROOT / run_id / scenario.scenario_id
    raw_dir = run_dir / "raw"
    report_dir = run_dir / "reports"
    raw_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    output_root = f"/tmp/scheduleurm_marginal_under_load/{run_id}/{scenario.scenario_id}"
    _deploy_progress_wrapper(
        scenario.node,
        raw_dir,
        cwd=str(REPO_ROOT),
        output_root=output_root,
    )
    script = _remote_script(scenario, run_id=run_id, output_root=output_root)
    started = time.time()
    rc, out, err = _run_remote_capture(
        scenario.node,
        script,
        raw_dir / "marginal_under_load_run",
        timeout_s=int(scenario.timeout_s) + int(scenario.warmup_s) + 180,
    )
    add_log = _extract_marker(out, "__ADD_LOG_BEGIN__", "__ADD_LOG_END__")
    bg_log = _extract_marker(out, "__BACKGROUND_LOG_BEGIN__", "__BACKGROUND_LOG_END__")
    diagnostics_log = _extract_marker(
        out,
        "__DIAGNOSTICS_LOG_BEGIN__",
        "__DIAGNOSTICS_LOG_END__",
    )
    (raw_dir / "add.log").write_text(add_log, encoding="utf-8")
    (raw_dir / "background.log").write_text(bg_log, encoding="utf-8")
    (raw_dir / "diagnostics.log").write_text(diagnostics_log, encoding="utf-8")
    stable = _last_stable_rate(add_log)
    rate, parsed_unit = _last_rate(add_log)
    add_rc = _extract_int(out, "__ADD_RC__", default=rc)
    bg_rc = _extract_int(out, "__BACKGROUND_RC__", default=0)
    bg_ready = bool(_extract_int(out, "__BACKGROUND_READY__", default=0))
    bg_alive_before_add = bool(
        _extract_int(out, "__BACKGROUND_ALIVE_BEFORE_ADD__", default=0)
    )
    bg_alive_after_add = bool(
        _extract_int(out, "__BACKGROUND_ALIVE_AFTER_ADD__", default=0)
    )
    add_exit = _returncode_diagnostic(
        add_rc,
        shell_signal_number=_extract_int(out, "__ADD_SIGNAL_NUMBER__", default=0),
        shell_signal_name=_extract_token(out, "__ADD_SIGNAL_NAME__", default=""),
    )
    bg_exit = _returncode_diagnostic(
        bg_rc,
        shell_signal_number=_extract_int(out, "__BACKGROUND_SIGNAL_NUMBER__", default=0),
        shell_signal_name=_extract_token(out, "__BACKGROUND_SIGNAL_NAME__", default=""),
    )
    remote_exit = _returncode_diagnostic(rc)
    bg_startup_failed = _background_startup_failed(bg_log)
    stable_rate = float(stable.get("last_rate") or stable.get("mean_rate") or 0.0)
    measurement_valid = (
        int(add_rc) == 0
        and bool(stable.get("ready"))
        and stable_rate > 0.0
        and bg_ready
        and bg_alive_before_add
        and bg_alive_after_add
        and not bg_startup_failed
    )
    row = {
        **_scenario_base(scenario, run_id=run_id),
        "launched": True,
        "measurement_valid": bool(measurement_valid),
        "returncode": int(rc),
        "add_returncode": int(add_rc),
        "background_returncode": int(bg_rc),
        "background_ready": bool(bg_ready),
        "background_alive_before_add": bool(bg_alive_before_add),
        "background_alive_after_add": bool(bg_alive_after_add),
        "exit_diagnostics": {
            "remote": remote_exit,
            "add_probe": add_exit,
            "background": bg_exit,
        },
        "target_cpu_logical": _extract_int(out, "__CPU_LOGICAL__", default=0),
        "target_cpu_cores": _extract_int(out, "__CPU_CORES__", default=0),
        "cpu_half_resident_workers": _extract_int(
            out,
            "__CPU_HALF_WORKERS__",
            default=0,
        ),
        "cpu_full_resident_workers": _extract_int(
            out,
            "__CPU_FULL_WORKERS__",
            default=0,
        ),
        "controlled_cuda_processes": _extract_int(
            out,
            "__CONTROLLED_CUDA_PROCESSES__",
            default=0,
        ),
        "rl_cuda_memory_budget_fraction": RL_CUDA_MEMORY_BUDGET,
        "elapsed_wall_s": time.time() - started,
        "rate_unit_s": float(rate),
        "stable_rate_unit_s": stable_rate,
        "stable_mean_rate_unit_s": float(stable.get("mean_rate") or 0.0),
        "stable_cv": float(stable.get("cv") or 0.0),
        "stable_samples": int(stable.get("samples") or 0),
        "unit": str(stable.get("unit") or parsed_unit or "unit"),
        "eta_source": "tqdm/progress",
        "stable_rate_ready": bool(stable.get("ready")),
        "background_startup_failed": bool(bg_startup_failed),
        "reason": "" if measurement_valid else _failure_reason(
            rc=rc,
            add_rc=add_rc,
            stable=stable,
            stderr=err,
            add_log=add_log,
            bg_log=bg_log,
            bg_ready=bg_ready,
            bg_alive_before_add=bg_alive_before_add,
            bg_alive_after_add=bg_alive_after_add,
            bg_startup_failed=bg_startup_failed,
        ),
        "add_log_path": str(raw_dir / "add.log"),
        "background_log_path": str(raw_dir / "background.log"),
        "diagnostics_log_path": str(raw_dir / "diagnostics.log"),
        "resource_diagnostics_captured": bool(diagnostics_log.strip()),
    }
    (report_dir / "summary.json").write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return row


def _scenario_base(scenario: MarginalScenario, *, run_id: str) -> dict[str, Any]:
    hardware = hardware_class_for_node(scenario.node)
    node_bucket = f"{scenario.node}:{hardware}"
    return {
        "scenario_id": scenario.scenario_id,
        "run_id": run_id,
        "node": scenario.node,
        "gpu": scenario.gpu,
        "node_bucket": node_bucket,
        "hardware_class": hardware,
        "workload_key": scenario.workload_key,
        "workload_env": scenario.workload_env,
        "resource_kind": scenario.resource_kind,
        "resource_state": scenario.resource_state,
        "resident_mix": scenario.resident_mix or _resident_mix_for_scenario(scenario),
        "profile": int(scenario.profile),
        "resident_count": max(0, int(scenario.profile) - 1),
        "add_count": 1,
        "post_task_count": int(scenario.profile),
        "service_semantics": "marginal_add_one",
        "hard_rule_mode": "clean_bench_direct_remote",
        "legacy_scheduler_limits_bypassed": True,
        "background_kind": scenario.background_kind,
        "add_kind": scenario.add_kind,
        "total_units": float(scenario.total_units),
        "note": scenario.note,
    }


def _record_from_row(row: Mapping[str, Any]) -> ProfileRecord | None:
    if not bool(row.get("measurement_valid")):
        return None
    rate = float(row.get("stable_rate_unit_s") or 0.0)
    if rate <= 0.0:
        return None
    return ProfileRecord(
        workload_key=str(row.get("workload_key")),
        command_fingerprint=f"live_marginal_under_load:{row.get('scenario_id')}",
        resource_kind=str(row.get("resource_kind")),
        node_bucket=str(row.get("node_bucket")),
        profile=int(row.get("profile") or 1),
        unit=str(row.get("unit") or "unit"),
        total_units=float(row.get("total_units") or 0.0),
        aggregate_rate=rate,
        per_task_rates=(rate,),
        source=f"live_marginal_under_load:{row.get('scenario_id')}",
        workload_env=str(row.get("workload_env") or ""),
        resource_state=str(row.get("resource_state") or ""),
        resident_mix=str(row.get("resident_mix") or ""),
        eta_source="tqdm/progress",
        stable_rate_ready=True,
        hardware_class=str(row.get("hardware_class") or ""),
    )


def _resident_mix_for_scenario(scenario: MarginalScenario) -> str:
    state = str(scenario.resource_state or "")
    background = tuple(scenario.background_kinds) or (str(scenario.background_kind or ""),)
    joined = ",".join(background)
    if state == "high_vram_resident":
        return "llm_or_memory_resident"
    if state == "cpu_resident":
        return "cpu_worker_resident"
    if "llm" in joined and "rl_resac" in joined:
        return "cnn_plus_llm_plus_hybrid_rl"
    if "llm" in joined and "cnn" in joined:
        return "cnn_plus_llm"
    if "rl_resac" in joined and "cnn" in joined:
        return "cnn_plus_hybrid_rl"
    if "llm" in joined:
        return "cnn_plus_llm"
    if "rl_resac" in joined or "bapr" in joined:
        return "cnn_plus_hybrid_rl"
    if "cpu_" in joined:
        return "cpu_plus_gpu_target" if state == "mixed_colocation" else "cpu_worker_resident"
    return str(scenario.background_kind or "")


def _remote_script(scenario: MarginalScenario, *, run_id: str, output_root: str) -> str:
    remote_dir = f"/tmp/scheduleurm_marginal_probe/{_safe_token(run_id)}_{_safe_token(scenario.scenario_id)}"
    resident_timeout_s = int(scenario.warmup_s) + int(scenario.timeout_s) + 120
    gpu_prefix = f"CUDA_VISIBLE_DEVICES={int(scenario.gpu)} " if scenario.gpu is not None else ""
    resident_count = max(1, int(scenario.profile) - 1)
    if scenario.resource_kind.startswith("cpu"):
        resident_count = 1
    background_kinds = tuple(scenario.background_kinds) or tuple(
        scenario.background_kind for _ in range(resident_count)
    )
    concurrent_cuda_processes = sum(
        1 for kind in background_kinds + (scenario.add_kind,) if _kind_uses_cuda(kind)
    )
    bg_cmds = [
        _wrapped_command(
            kind,
            label=f"{scenario.scenario_id}_background_{idx}",
            total=_background_total(kind),
            terminate_on_stable=False,
            gpu_prefix=gpu_prefix,
            concurrent_cuda_processes=concurrent_cuda_processes,
        )
        for idx, kind in enumerate(background_kinds)
    ]
    add_cmd = _wrapped_command(
        scenario.add_kind,
        label=f"{scenario.scenario_id}_add",
        total=int(scenario.total_units),
        terminate_on_stable=True,
        gpu_prefix=gpu_prefix,
        concurrent_cuda_processes=concurrent_cuda_processes,
    )
    lines = [
        "set +e",
        _torch_env_shell(),
        f"REMOTE_DIR={shlex.quote(remote_dir)}",
        'rm -rf "$REMOTE_DIR"',
        'mkdir -p "$REMOTE_DIR"',
        f"cd {shlex.quote(str(REPO_ROOT))}",
        f"mkdir -p {shlex.quote(output_root)}",
        "CPU_LOGICAL=$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)",
        'case "$CPU_LOGICAL" in ""|*[!0-9]*) CPU_LOGICAL=1 ;; esac',
        "CPU_PHYSICAL=0",
        "if command -v lscpu >/dev/null 2>&1; then",
        "  CPU_PHYSICAL=$(lscpu -p=CORE,SOCKET 2>/dev/null | sed '/^#/d' | sort -u | wc -l)",
        "fi",
        'case "$CPU_PHYSICAL" in ""|*[!0-9]*) CPU_PHYSICAL=0 ;; esac',
        'if [ "$CPU_PHYSICAL" -le 0 ]; then CPU_PHYSICAL="$CPU_LOGICAL"; fi',
        'if [ "$CPU_PHYSICAL" -lt "$CPU_LOGICAL" ]; then CPU_CORES="$CPU_PHYSICAL"; else CPU_CORES="$CPU_LOGICAL"; fi',
        'if [ "$CPU_CORES" -le 0 ]; then CPU_CORES=1; fi',
        'CPU_HALF_WORKERS=$((CPU_CORES / 2))',
        'if [ "$CPU_HALF_WORKERS" -le 0 ]; then CPU_HALF_WORKERS=1; fi',
        f'CPU_FULL_RESERVE=$(( (CPU_CORES + {CPU_FULL_RESERVE_DIVISOR - 1}) / {CPU_FULL_RESERVE_DIVISOR} ))',
        'if [ "$CPU_FULL_RESERVE" -le 0 ]; then CPU_FULL_RESERVE=1; fi',
        'CPU_FULL_WORKERS=$((CPU_CORES - CPU_FULL_RESERVE))',
        'if [ "$CPU_FULL_WORKERS" -le 0 ]; then CPU_FULL_WORKERS=1; fi',
        "export CPU_LOGICAL CPU_PHYSICAL CPU_CORES CPU_HALF_WORKERS CPU_FULL_WORKERS",
        f"CONTROLLED_CUDA_PROCESSES={concurrent_cuda_processes}",
        f"RL_CUDA_MEMORY_BUDGET={RL_CUDA_MEMORY_BUDGET:.4f}",
        "export CONTROLLED_CUDA_PROCESSES RL_CUDA_MEMORY_BUDGET",
        'DIAG_LOG="$REMOTE_DIR/diagnostics.log"',
        ': > "$DIAG_LOG"',
        "BG_PIDS=()",
        "BG_GROUPED=()",
        "BG_LOGS=()",
        "resource_snapshot() {",
        '  local label="$1"',
        '  printf "RESOURCE_SNAPSHOT label=%s timestamp=%s\\n" "$label" "$(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date)"',
        '  printf "cpu_logical=%s cpu_physical=%s cpu_cores=%s half_workers=%s full_workers=%s\\n" "$CPU_LOGICAL" "$CPU_PHYSICAL" "$CPU_CORES" "$CPU_HALF_WORKERS" "$CPU_FULL_WORKERS"',
        '  printf "controlled_cuda_processes=%s rl_cuda_memory_budget=%s\\n" "$CONTROLLED_CUDA_PROCESSES" "$RL_CUDA_MEMORY_BUDGET"',
        '  printf "[ulimit]\\n"',
        "  ulimit -a 2>&1 || true",
        '  printf "[memory]\\n"',
        "  if command -v free >/dev/null 2>&1; then free -h 2>&1 || true; else sed -n '1,8p' /proc/meminfo 2>/dev/null || true; fi",
        '  printf "[gpu]\\n"',
        "  if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits 2>&1 || true; else echo nvidia-smi-unavailable; fi",
        "}",
        "decode_shell_exit() {",
        '  local rc="$1"',
        "  SHELL_SIGNAL_NUMBER=0",
        "  SHELL_SIGNAL_NAME=NONE",
        '  if [ "$rc" -gt 128 ] && [ "$rc" -le 192 ]; then',
        '    SHELL_SIGNAL_NUMBER=$((rc - 128))',
        '    SHELL_SIGNAL_NAME=$(kill -l "$SHELL_SIGNAL_NUMBER" 2>/dev/null || true)',
        '    case "$SHELL_SIGNAL_NAME" in SIG*) ;; ""|*[!A-Za-z0-9_]*) SHELL_SIGNAL_NAME=UNKNOWN ;; *) SHELL_SIGNAL_NAME="SIG${SHELL_SIGNAL_NAME}" ;; esac',
        "  fi",
        "}",
        "record_shell_exit() {",
        '  local role="$1" rc="$2" encoding=exit_code',
        '  decode_shell_exit "$rc"',
        '  if [ "$SHELL_SIGNAL_NUMBER" -gt 0 ]; then encoding=128_plus_signal; fi',
        '  printf "SHELL_EXIT role=%s returncode=%s encoding=%s signal_number=%s signal_name=%s\\n" "$role" "$rc" "$encoding" "$SHELL_SIGNAL_NUMBER" "$SHELL_SIGNAL_NAME"',
        "}",
        "backgrounds_alive() {",
        '  local pid',
        '  [ "${#BG_PIDS[@]}" -gt 0 ] || return 1',
        '  for pid in "${BG_PIDS[@]}"; do kill -0 "$pid" >/dev/null 2>&1 || return 1; done',
        "  return 0",
        "}",
        "cleanup_backgrounds() {",
        '  local idx pid',
        '  for idx in "${!BG_PIDS[@]}"; do',
        '    pid="${BG_PIDS[$idx]}"',
        '    if [ "${BG_GROUPED[$idx]:-0}" -eq 1 ]; then kill -- "-$pid" >/dev/null 2>&1 || true; fi',
        '    kill "$pid" >/dev/null 2>&1 || true',
        "  done",
        "}",
        "trap cleanup_backgrounds EXIT",
        "trap 'cleanup_backgrounds; exit 129' HUP",
        "trap 'cleanup_backgrounds; exit 130' INT",
        "trap 'cleanup_backgrounds; exit 143' TERM",
        'resource_snapshot before_background >> "$DIAG_LOG" 2>&1',
    ]
    for idx, bg_cmd in enumerate(bg_cmds):
        lines.extend([
            "if command -v setsid >/dev/null 2>&1; then",
            f"  setsid timeout --foreground {resident_timeout_s}s bash -lc {shlex.quote(bg_cmd)} > \"$REMOTE_DIR/background_{idx}.log\" 2>&1 &",
            "  BG_PID=$!",
            "  BG_GROUPED+=(1)",
            "else",
            f"  timeout --foreground {resident_timeout_s}s bash -lc {shlex.quote(bg_cmd)} > \"$REMOTE_DIR/background_{idx}.log\" 2>&1 &",
            "  BG_PID=$!",
            "  BG_GROUPED+=(0)",
            "fi",
            'BG_PIDS+=("$BG_PID")',
            f"BG_LOGS+=(\"$REMOTE_DIR/background_{idx}.log\")",
        ])
    lines.extend([
        f"READY_DEADLINE=$((SECONDS + {int(scenario.warmup_s)}))",
        "BACKGROUND_READY=0",
        'while [ "$SECONDS" -lt "$READY_DEADLINE" ]; do',
        "  READY_COUNT=0",
        '  for IDX in "${!BG_LOGS[@]}"; do',
        '    BG_LOG="${BG_LOGS[$IDX]}"; BG_PID="${BG_PIDS[$IDX]}"',
        '    if kill -0 "$BG_PID" >/dev/null 2>&1 && grep -Eq "BENCH_PROGRESS|CPU_PARALLEL_PROGRESS|ScheduleurmProgress" "$BG_LOG" 2>/dev/null; then READY_COUNT=$((READY_COUNT + 1)); fi',
        "  done",
        '  if [ "$READY_COUNT" -eq "${#BG_LOGS[@]}" ]; then BACKGROUND_READY=1; break; fi',
        "  sleep 2",
        "done",
        "BACKGROUND_ALIVE_BEFORE_ADD=0",
        "if backgrounds_alive; then BACKGROUND_ALIVE_BEFORE_ADD=1; fi",
        'resource_snapshot before_add >> "$DIAG_LOG" 2>&1',
        f"timeout --foreground {int(scenario.timeout_s)}s bash -lc {shlex.quote(add_cmd)} > \"$REMOTE_DIR/add.log\" 2>&1",
        "ADD_RC=$?",
        'record_shell_exit add_probe "$ADD_RC" >> "$DIAG_LOG" 2>&1',
        "ADD_SIGNAL_NUMBER=$SHELL_SIGNAL_NUMBER",
        "ADD_SIGNAL_NAME=$SHELL_SIGNAL_NAME",
        "BACKGROUND_ALIVE_AFTER_ADD=0",
        "if backgrounds_alive; then BACKGROUND_ALIVE_AFTER_ADD=1; fi",
        'resource_snapshot after_add >> "$DIAG_LOG" 2>&1',
        "cleanup_backgrounds",
        'BG_RC=0',
        'for BG_PID in "${BG_PIDS[@]}"; do wait "$BG_PID" >/dev/null 2>&1; WAIT_RC=$?; if [ "$WAIT_RC" -ne 0 ]; then BG_RC=$WAIT_RC; fi; done',
        'record_shell_exit background "$BG_RC" >> "$DIAG_LOG" 2>&1',
        "BACKGROUND_SIGNAL_NUMBER=$SHELL_SIGNAL_NUMBER",
        "BACKGROUND_SIGNAL_NAME=$SHELL_SIGNAL_NAME",
        'resource_snapshot after_cleanup >> "$DIAG_LOG" 2>&1',
        'echo "__ADD_RC__ $ADD_RC"',
        'echo "__ADD_SIGNAL_NUMBER__ $ADD_SIGNAL_NUMBER"',
        'echo "__ADD_SIGNAL_NAME__ $ADD_SIGNAL_NAME"',
        'echo "__BACKGROUND_RC__ $BG_RC"',
        'echo "__BACKGROUND_SIGNAL_NUMBER__ $BACKGROUND_SIGNAL_NUMBER"',
        'echo "__BACKGROUND_SIGNAL_NAME__ $BACKGROUND_SIGNAL_NAME"',
        'echo "__BACKGROUND_READY__ $BACKGROUND_READY"',
        'echo "__BACKGROUND_ALIVE_BEFORE_ADD__ $BACKGROUND_ALIVE_BEFORE_ADD"',
        'echo "__BACKGROUND_ALIVE_AFTER_ADD__ $BACKGROUND_ALIVE_AFTER_ADD"',
        'echo "__CPU_LOGICAL__ $CPU_LOGICAL"',
        'echo "__CPU_CORES__ $CPU_CORES"',
        'echo "__CPU_HALF_WORKERS__ $CPU_HALF_WORKERS"',
        'echo "__CPU_FULL_WORKERS__ $CPU_FULL_WORKERS"',
        'echo "__CONTROLLED_CUDA_PROCESSES__ $CONTROLLED_CUDA_PROCESSES"',
        'echo "__ADD_LOG_BEGIN__"',
        'cat "$REMOTE_DIR/add.log" 2>/dev/null || true',
        'echo "__ADD_LOG_END__"',
        'echo "__BACKGROUND_LOG_BEGIN__"',
        'for BG_LOG in "${BG_LOGS[@]}"; do echo "### $BG_LOG"; cat "$BG_LOG" 2>/dev/null || true; done',
        'echo "__BACKGROUND_LOG_END__"',
        'echo "__DIAGNOSTICS_LOG_BEGIN__"',
        'cat "$DIAG_LOG" 2>/dev/null || true',
        'echo "__DIAGNOSTICS_LOG_END__"',
        "trap - EXIT HUP INT TERM",
        "exit 0",
    ])
    return "\n".join(lines)


def _kind_uses_cuda(kind: str) -> bool:
    return not str(kind or "").startswith("cpu_")


def _wrapped_command(
    kind: str,
    *,
    label: str,
    total: int,
    terminate_on_stable: bool,
    gpu_prefix: str,
    concurrent_cuda_processes: int = 1,
) -> str:
    wrapper = f"{REMOTE_PROGRESS_WRAPPER_DIR}/progress_wrapper.py"
    cycle_extra = " --stable-cycle-units 4" if kind.startswith("llm_") else ""
    extra = (
        "--terminate-on-stable --stable-windows 4 --min-rate-samples 5 "
        f"--stable-cv 0.10 --stable-rel-delta 0.06 --stable-skip-samples 1{cycle_extra}"
        if terminate_on_stable
        else ""
    )
    py = "${PY}"
    if kind.startswith("rl_resac_"):
        return _wrapped_resac_command(
            kind,
            label=label,
            total=max(1, int(total)),
            terminate_on_stable=terminate_on_stable,
            gpu_prefix=gpu_prefix,
            concurrent_cuda_processes=concurrent_cuda_processes,
        )
    if kind.startswith("bapr_ant_"):
        return _wrapped_bapr_command(
            kind,
            label=label,
            total=max(1, int(total)),
            terminate_on_stable=terminate_on_stable,
            gpu_prefix=gpu_prefix,
            concurrent_cuda_processes=concurrent_cuda_processes,
        )
    command = _benchmark_command(kind, label=label, steps=max(1, int(total)))
    return (
        f"{gpu_prefix}\"$PY\" -u {shlex.quote(wrapper)} --unit step --total {int(total)} "
        f"{extra} -- {py} -u {command}"
    )


def _background_total(kind: str) -> int:
    del kind
    return RESIDENT_BACKGROUND_TOTAL_UNITS


def _wrapped_resac_command(
    kind: str,
    *,
    label: str,
    total: int,
    terminate_on_stable: bool,
    gpu_prefix: str,
    concurrent_cuda_processes: int = 1,
) -> str:
    wrapper = f"{REMOTE_PROGRESS_WRAPPER_DIR}/progress_wrapper.py"
    env_name = _resac_env_for_kind(kind)
    mem_fraction = _rl_cuda_memory_fraction(kind, concurrent_cuda_processes)
    ensemble_size = "2" if kind.endswith("_light") else "10"
    extra = (
        "--terminate-on-stable --stable-windows 4 --min-rate-samples 5 "
        "--stable-cv 0.10 --stable-rel-delta 0.06 --stable-skip-samples 1 "
        "--stable-cycle-units 5"
        if terminate_on_stable
        else "--stable-cycle-units 5"
    )
    exports = (
        "if [ -x /home/erzhu419/.venvs/resac-jax-gpu1-0438/bin/python ]; then "
        "PY=/home/erzhu419/.venvs/resac-jax-gpu1-0438/bin/python; fi; "
        "export PYTHONPATH=/home/erzhu419/.claude/scheduler/runtime_patches:"
        "/home/erzhu419/mine_code/RE-SAC:${PYTHONPATH:-}; "
        "export JAX_PLATFORMS=cuda; "
        "export XLA_PYTHON_CLIENT_PREALLOCATE=false; "
        f"export XLA_PYTHON_CLIENT_MEM_FRACTION={mem_fraction:.4f}; "
        "export XLA_FLAGS='--xla_gpu_enable_triton_gemm=false --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'; "
        "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1; "
        "export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MALLOC_ARENA_MAX=2; "
    )
    command = (
        f"cd /home/erzhu419/mine_code/RE-SAC && {exports}"
        f"{gpu_prefix}\"$PY\" -u {shlex.quote(wrapper)} --unit iter --total {int(total)} {extra} -- "
        "\"$PY\" -u -m jax_experiments.train "
        f"--algo resac --env {env_name} --seed 9001 --max_iters {int(total)} "
        "--checkpoint_interval 0 --checkpoint_keep 1 "
        f"--save_root /tmp/scheduleurm_marginal_rl_resident --backend spring "
        "--device gpu --stationary --beta -2.0 --beta_ood 0.001 "
        "--weight_reg 0.001 --beta_bc 0.0001 --beta_start -2.0 "
        "--beta_warmup 0.2 --adaptive_beta "
        f"--ensemble_size {ensemble_size} "
        "--beta_end -1.0 --ema_tau 0.0 --anchor_lambda 0.01 "
        f"--independent_ratio 0.5 --run_name {shlex.quote(label)}"
    )
    return f"bash -lc {shlex.quote(command)}"


def _resac_env_for_kind(kind: str) -> str:
    lower = str(kind or "").lower()
    if "halfcheetah" in lower:
        return "HalfCheetah-v2"
    if "hopper" in lower:
        return "Hopper-v2"
    if "walker2d" in lower:
        return "Walker2d-v2"
    return "Ant-v2"


def _wrapped_bapr_command(
    kind: str,
    *,
    label: str,
    total: int,
    terminate_on_stable: bool,
    gpu_prefix: str,
    concurrent_cuda_processes: int = 1,
) -> str:
    wrapper = f"{REMOTE_PROGRESS_WRAPPER_DIR}/progress_wrapper.py"
    mem_fraction = _rl_cuda_memory_fraction(kind, concurrent_cuda_processes)
    extra = (
        "--terminate-on-stable --stable-windows 4 --min-rate-samples 5 "
        "--stable-cv 0.10 --stable-rel-delta 0.06 --stable-skip-samples 1 "
        "--stable-cycle-units 5"
        if terminate_on_stable
        else "--stable-cycle-units 5"
    )
    exports = (
        "if [ -x /home/erzhu419/.venvs/resac-jax-gpu1-0438/bin/python ]; then "
        "PY=/home/erzhu419/.venvs/resac-jax-gpu1-0438/bin/python; fi; "
        "export PYTHONPATH=/home/erzhu419/.claude/scheduler/runtime_patches:"
        "/home/erzhu419/mine_code/BAPR:${PYTHONPATH:-}; "
        "export JAX_PLATFORMS=cuda; "
        "export XLA_PYTHON_CLIENT_PREALLOCATE=false; "
        f"export XLA_PYTHON_CLIENT_MEM_FRACTION={mem_fraction:.4f}; "
        "export XLA_FLAGS='--xla_gpu_enable_triton_gemm=false --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'; "
        "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1; "
        "export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MALLOC_ARENA_MAX=2; "
    )
    command = (
        f"cd /home/erzhu419/mine_code/BAPR && {exports}"
        f"{gpu_prefix}\"$PY\" -u {shlex.quote(wrapper)} --unit iter --total {int(total)} {extra} -- "
        "\"$PY\" -u -m jax_experiments.train "
        "--algo bapr --env Ant-v2 "
        f"--seed 9101 --max_iters {int(total)} --log_interval 5 --eval_episodes 5 "
        "--save_interval 100 "
        f"--save_root /tmp/scheduleurm_marginal_bapr_resident --backend spring "
        "--env_type discrete_mode --mean_dwell_iters 60 --use_regime_belief "
        "--penalty_scale 0.5 --critic_target_mode min --no_per_trans_belief "
        f"--run_name {shlex.quote(label)}"
    )
    return f"bash -lc {shlex.quote(command)}"


def _benchmark_command(kind: str, *, label: str, steps: int) -> str:
    root = REMOTE_PROGRESS_WRAPPER_DIR
    if kind == "cnn_resident":
        return (
            f"{root}/torch_cnn_progress_benchmark.py --steps {int(steps)} "
            f"--warmup 8 --log-interval 5 --batch-size 32 --label {shlex.quote(label)}"
        )
    if kind == "cnn_add":
        return (
            f"{root}/torch_cnn_progress_benchmark.py --steps {int(steps)} "
            f"--warmup 4 --log-interval 5 --batch-size 32 --label {shlex.quote(label)}"
        )
    if kind == "cnn_add_light":
        return (
            f"{root}/torch_cnn_progress_benchmark.py --steps {int(steps)} "
            f"--warmup 3 --log-interval 5 --batch-size 16 --label {shlex.quote(label)}"
        )
    if kind == "high_vram_resident":
        return (
            f"{root}/gpu_multigpu_memory_progress_benchmark.py --steps {int(steps)} "
            f"--warmup 4 --reserve-gb-per-gpu 6.5 --matrix-size 1536 "
            f"--log-interval 5 --label {shlex.quote(label)}"
        )
    if kind == "llm_resident":
        return (
            f"{root}/torch_llm_progress_benchmark.py --steps {int(steps)} "
            f"--warmup 4 --log-interval 4 --batch-size 1 --seq-len 128 "
            f"--label {shlex.quote(label)}"
        )
    if kind == "llm_resident_light":
        return (
            f"{root}/torch_llm_progress_benchmark.py --steps {int(steps)} "
            f"--warmup 3 --log-interval 4 --batch-size 1 --seq-len 96 "
            f"--label {shlex.quote(label)}"
        )
    if kind == "llm_add":
        return (
            f"{root}/torch_llm_progress_benchmark.py --steps {int(steps)} "
            f"--warmup 4 --log-interval 4 --batch-size 2 --seq-len 128 "
            f"--label {shlex.quote(label)}"
        )
    if kind == "llm_add_light":
        return (
            f"{root}/torch_llm_progress_benchmark.py --steps {int(steps)} "
            f"--warmup 3 --log-interval 4 --batch-size 1 --seq-len 96 "
            f"--label {shlex.quote(label)}"
        )
    if kind == "matmul_resident":
        return (
            f"{root}/gpu_progress_benchmark.py --steps {int(steps)} "
            f"--size 2048 --label {shlex.quote(label)}"
        )
    if kind == "matmul_resident_light":
        return (
            f"{root}/gpu_progress_benchmark.py --steps {int(steps)} "
            f"--size 1536 --label {shlex.quote(label)}"
        )
    if kind == "matmul_add_light":
        return (
            f"{root}/gpu_progress_benchmark.py --steps {int(steps)} "
            f"--size 1536 --label {shlex.quote(label)}"
        )
    if kind == "cpu_half_resident":
        return (
            f'{root}/cpu_parallel_progress_benchmark.py --workers "${{CPU_HALF_WORKERS}}" --steps {int(steps)} '
            f"--work-items 90000 --log-interval 2 --label {shlex.quote(label)}"
        )
    if kind == "cpu_full_resident":
        return (
            f'{root}/cpu_parallel_progress_benchmark.py --workers "${{CPU_FULL_WORKERS}}" --steps {int(steps)} '
            f"--work-items 90000 --log-interval 2 --label {shlex.quote(label)}"
        )
    if kind == "cpu_add_single":
        return (
            f"{root}/cpu_progress_benchmark.py --mode cpu --steps {int(steps)} "
            f"--work-items 200000 --label {shlex.quote(label)}"
        )
    if kind == "cpu_add_light":
        return (
            f"{root}/cpu_progress_benchmark.py --mode light --steps {int(steps)} "
            f"--work-items 8000 --label {shlex.quote(label)}"
        )
    raise ValueError(f"unknown benchmark kind: {kind}")


def _torch_env_shell() -> str:
    return (
        'PY="${TORCH_BENCH_PYTHON:-}"; '
        'if [ -z "$PY" ] && [ -x /home/erzhu419/.venvs/scheduleurm-torch-bench/bin/python ]; then '
        'PY=/home/erzhu419/.venvs/scheduleurm-torch-bench/bin/python; fi; '
        'if [ -z "$PY" ] && [ -x /home/zhengliang01/scheduleurm_work/conda_envs/scomp-py310/bin/python ]; then '
        'PY=/home/zhengliang01/scheduleurm_work/conda_envs/scomp-py310/bin/python; fi; '
        'if [ -z "$PY" ]; then PY=$(command -v python3 || command -v python); fi; '
        'SITE=$("$PY" -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || true); '
        'for d in "$SITE"/nvidia/*/lib; do [ -d "$d" ] && export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"; done; '
        'export PY; export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1'
    )


def _extract_marker(text: str, begin: str, end: str) -> str:
    pattern = re.escape(begin) + r"\n?(.*?)\n?" + re.escape(end)
    match = re.search(pattern, text or "", flags=re.DOTALL)
    return match.group(1) if match else ""


def _extract_int(text: str, marker: str, *, default: int) -> int:
    match = re.search(re.escape(marker) + r"\s+(-?\d+)", text or "")
    if not match:
        return int(default)
    return int(match.group(1))


def _extract_token(text: str, marker: str, *, default: str) -> str:
    match = re.search(re.escape(marker) + r"\s+(\S+)", text or "")
    return str(match.group(1)) if match else str(default)


def _returncode_diagnostic(
    returncode: int,
    *,
    shell_signal_number: int = 0,
    shell_signal_name: str = "",
) -> dict[str, Any]:
    """Decode subprocess and shell 128+signal return conventions."""

    rc = int(returncode)
    max_signal = max(1, int(getattr(signal, "NSIG", 65)) - 1)
    signal_number: int | None = None
    encoding = "exit_code"
    if -max_signal <= rc < 0:
        signal_number = -rc
        encoding = "subprocess_negative_signal"
    elif 128 < rc <= 128 + max_signal:
        signal_number = rc - 128
        encoding = "shell_128_plus_signal"

    signal_name = ""
    signal_description = ""
    if signal_number is not None:
        try:
            signal_name = signal.Signals(signal_number).name
        except ValueError:
            signal_name = f"SIG{signal_number}"
        try:
            signal_description = str(signal.strsignal(signal_number) or "")
        except (AttributeError, ValueError):
            signal_description = ""

    shell_number = int(shell_signal_number or 0)
    shell_name = str(shell_signal_name or "")
    if shell_name.upper() in {"NONE", "UNKNOWN"}:
        shell_name = "" if shell_name.upper() == "NONE" else shell_name.upper()
    return {
        "returncode": rc,
        "kind": "signal" if signal_number is not None else "exit",
        "encoding": encoding,
        "exit_code": None if signal_number is not None else rc,
        "signal_number": signal_number,
        "signal_name": signal_name,
        "signal_description": signal_description,
        "shell_reported_signal": {
            "signal_number": shell_number or None,
            "signal_name": shell_name,
        },
        "shell_report_matches": bool(
            signal_number is not None
            and shell_number == signal_number
            and (not shell_name or shell_name == signal_name)
        ),
    }


def _background_startup_failed(bg_log: str) -> bool:
    """Return true when a resident pressure process did not actually start.

    Resident jobs are killed after the add-one probe, so their shell return code
    is not reliable.  Fatal import/runtime errors in the resident log are
    reliable evidence that the measured load state was not created.
    """

    lower = str(bg_log or "").lower()
    fatal_tokens = (
        "modulenotfounderror",
        "importerror",
        "traceback (most recent call last)",
        "cuda is required",
        "no python with torch import available",
        "scheduleurmprobeerror",
        "out of memory",
        "cublas",
        "cuda error",
        "resource exhausted",
        "syntax error",
        "未预期的符号",
        "语法错误",
    )
    return any(token in lower for token in fatal_tokens)


def _failure_reason(
    *,
    rc: int,
    add_rc: int,
    stable: Mapping[str, Any],
    stderr: str,
    add_log: str,
    bg_log: str = "",
    bg_ready: bool = False,
    bg_alive_before_add: bool = True,
    bg_alive_after_add: bool = True,
    bg_startup_failed: bool = False,
) -> str:
    lower = f"{stderr}\n{add_log}\n{bg_log}".lower()
    if bg_startup_failed:
        return "background_startup_failed"
    if not bg_ready:
        return "background_progress_not_ready"
    if not bg_alive_before_add:
        return "background_ended_before_add_probe"
    add_exit = _returncode_diagnostic(add_rc)
    if add_exit["kind"] == "signal":
        return f"add_probe_signal_{add_exit['signal_name']}"
    remote_exit = _returncode_diagnostic(rc)
    if remote_exit["kind"] == "signal":
        return f"remote_signal_{remote_exit['signal_name']}"
    if not bg_alive_after_add:
        return "background_ended_during_add_probe"
    if any(token in lower for token in ("out of memory", "cublas", "cuda error", "resource exhausted")):
        return "capacity_or_runtime_boundary"
    if int(add_rc) != 0:
        return f"add_probe_returncode_{int(add_rc)}"
    if not bool(stable.get("ready")):
        return str(stable.get("reason") or "stable_rate_not_ready")
    if int(rc) != 0:
        return f"remote_returncode_{int(rc)}"
    return "unknown"


def _safe_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("_") or "run"


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Live Marginal Under-Load Matrix",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Allow launch: `{str(bool(report.get('allow_launch'))).lower()}`",
        f"- Pass: `{str(bool(report.get('pass'))).lower()}`",
        f"- Admitted rows: `{report.get('admitted_count')}`",
        f"- Boundary rows: `{report.get('boundary_count')}`",
        "",
        "| Scenario | Node | State | Workload | Stable rate | Valid | Reason |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for row in report.get("rows") or []:
        lines.append(
            f"| `{row.get('scenario_id')}` | `{row.get('node')}` | `{row.get('resource_state')}` | "
            f"`{row.get('workload_key')}` | {float(row.get('stable_rate_unit_s') or 0.0):.6g} | "
            f"{str(bool(row.get('measurement_valid'))).lower()} | `{row.get('reason') or ''}` |"
        )
    lines.extend(["", str(report.get("scope") or ""), ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-launch", action="store_true")
    parser.add_argument("--scenarios", default="")
    parser.add_argument("--from-design", action="store_true")
    parser.add_argument("--design-csv", type=Path, default=DEFAULT_DESIGN_CSV)
    parser.add_argument("--assigned-node", default="")
    parser.add_argument("--logical-nodes", default="")
    parser.add_argument("--hardware-groups", default="")
    parser.add_argument("--workloads", default="")
    parser.add_argument("--states", default="")
    parser.add_argument("--resident-mixes", default="")
    parser.add_argument("--max-rows", type=int, default=1)
    parser.add_argument("--run-id", default="live_marginal_under_load_matrix_20260629")
    parser.add_argument("--output", type=Path, default=ARTIFACT_ROOT / "live_marginal_under_load_matrix_20260629.json")
    parser.add_argument("--cache-output", type=Path, default=ARTIFACT_ROOT / "service_cache_v2_marginal_under_load_20260629.json")
    parser.add_argument("--markdown-output", type=Path, default=REPO_ROOT / "md" / "live_marginal_under_load_matrix_20260629.md")
    args = parser.parse_args()
    selected = [x.strip() for x in str(args.scenarios or "").split(",") if x.strip()]
    if args.from_design:
        report = build_live_marginal_from_design(
            design_csv=args.design_csv,
            run_id=str(args.run_id),
            allow_launch=bool(args.allow_launch),
            assigned_node=str(args.assigned_node or ""),
            logical_nodes=[x.strip() for x in str(args.logical_nodes or "").split(",") if x.strip()],
            hardware_groups=[x.strip() for x in str(args.hardware_groups or "").split(",") if x.strip()],
            workloads=[x.strip() for x in str(args.workloads or "").split(",") if x.strip()],
            states=[x.strip() for x in str(args.states or "").split(",") if x.strip()],
            resident_mixes=[x.strip() for x in str(args.resident_mixes or "").split(",") if x.strip()],
            max_rows=int(args.max_rows),
        )
    else:
        report = build_live_marginal_under_load_matrix(
            scenarios=selected or None,
            run_id=str(args.run_id),
            allow_launch=bool(args.allow_launch),
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.cache_output.parent.mkdir(parents=True, exist_ok=True)
    args.cache_output.write_text(json.dumps(report["service_cache_snapshot"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
