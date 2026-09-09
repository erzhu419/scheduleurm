"""Full-factorial ETA lookup-table design for Scheduleurm experiments.

This module does not launch probes.  It defines the finite declared domain that
the live ETA table should cover before replaying algorithm/SOTA comparisons.
The generated CSV is the work order for controlled probes; rows already present
in a service-cache v2 snapshot are marked measured.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from simulation.service_cache import ServiceRateCache


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_CACHE = ARTIFACT_ROOT / "service_cache_v2_live_merged_20260629.json"
PROFILE_AXIS_ALLOCATION_WORKERS = "allocation_workers"
PROFILE_AXIS_COLOCATION_COUNT = "colocation_count"
PROFILE_AXES = frozenset({
    PROFILE_AXIS_ALLOCATION_WORKERS,
    PROFILE_AXIS_COLOCATION_COUNT,
})


@dataclass(frozen=True)
class HardwareType:
    hardware_group: str
    primary_node: str
    node_bucket: str
    equivalent_nodes: tuple[str, ...]
    resource_kind: str
    gpu_count: int
    cpu_cores: int
    role: str


@dataclass(frozen=True)
class WorkloadType:
    workload_key: str
    workload_env: str
    family: str
    quadrant: str
    resource_kind: str
    profile_ladder: tuple[int, ...]
    checkpointable: bool
    runner: str
    notes: str
    profile_axis: str = PROFILE_AXIS_COLOCATION_COUNT

    def __post_init__(self) -> None:
        if self.profile_axis not in PROFILE_AXES:
            raise ValueError(f"unsupported profile axis: {self.profile_axis!r}")


@dataclass(frozen=True)
class ResidentScenario:
    resource_state: str
    resident_mix: str
    target_kinds: tuple[str, ...]
    hardware_kinds: tuple[str, ...]
    required_profile_mode: str
    priority: str
    notes: str
    workload_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class MigrationPair:
    source_bucket: str
    destination_bucket: str
    target_families: tuple[str, ...]
    progress_points: tuple[float, ...]
    priority: str
    notes: str


HARDWARE_TYPES = (
    HardwareType(
        "gpu_3080ti_12gb_dual",
        "jtl110gpu",
        "jtl110gpu:gpu_3080ti_12gb_dual",
        ("jtl110gpu2",),
        "gpu",
        2,
        0,
        "Full representative GPU matrix; jtl110gpu2 is homogeneous equivalence.",
    ),
    HardwareType(
        "gpu_3080ti_12gb_dual_equiv",
        "jtl110gpu2",
        "jtl110gpu2:gpu_3080ti_12gb_dual",
        (),
        "gpu",
        2,
        0,
        "Equivalence validation for jtl110gpu; not a separate full matrix unless tolerance fails.",
    ),
    HardwareType(
        "gpu_rtx2080_8gb_dual_cpu_fast",
        "jtl311linux",
        "jtl311linux:gpu_rtx2080_8gb_dual_cpu_fast",
        (),
        "gpu",
        2,
        8,
        "CPU-fast/GPU-weaker node; full GPU, RL, LLM-feasible, and host-contention matrix.",
    ),
    HardwareType(
        "gpu_node007_4x11gb",
        "node007",
        "node007:gpu_node007_4x12gb",
        (),
        "gpu",
        4,
        64,
        "Four homogeneous RTX 2080 Ti GPUs; bucket name is historical, live memory is about 11GB.",
    ),
    HardwareType(
        "cpu_hpc_192c",
        "node001",
        "node001:cpu_hpc_192c",
        ("node002", "node003", "node004", "node005", "node006"),
        "cpu",
        0,
        192,
        "Full CPU matrix on one 192-core node; equivalence rows on additional nodes.",
    ),
    HardwareType(
        "cpu_hpc_192c_equiv",
        "node003",
        "node003:cpu_hpc_192c",
        ("node005",),
        "cpu",
        0,
        192,
        "CPU equivalence/replication node for 192-core class.",
    ),
    HardwareType(
        "cpu_jtl110_128c",
        "jtl110cpu",
        "jtl110cpu:cpu_jtl110_128c",
        ("jtl110cpu2",),
        "cpu",
        0,
        128,
        "Independent CPU-server class; required for cross-CPU-family extrapolation.",
    ),
)


WORKLOAD_TYPES = (
    WorkloadType(
        "light_control_local",
        "light",
        "light_control",
        "q00_low_cpu_low_gpu",
        "cpu",
        (1, 2, 4, 8, 13, 14),
        False,
        "light_progress_control",
        "Low-resource control row for scheduler overhead and short-task fragmentation.",
        profile_axis=PROFILE_AXIS_COLOCATION_COUNT,
    ),
    WorkloadType(
        "gpu_cnn_torch_resnet50",
        "cnn",
        "cnn",
        "q01_low_cpu_high_gpu",
        "gpu",
        (1, 2, 3, 4, 5, 6, 8),
        True,
        "torch_cnn_resnet50.cmd.tpl",
        "Pure GPU CNN training surrogate with native progress.",
        profile_axis=PROFILE_AXIS_COLOCATION_COUNT,
    ),
    WorkloadType(
        "gpu_llm_distilgpt2",
        "llm",
        "llm_small",
        "q01_low_cpu_high_gpu",
        "gpu",
        (1, 2, 3, 4),
        True,
        "torch_llm_distilgpt2.cmd.tpl",
        "Small LLM forward-pass row; larger LLM rows are separate capacity/staging rows.",
        profile_axis=PROFILE_AXIS_COLOCATION_COUNT,
    ),
    WorkloadType(
        "gpu_heavy_jax_matmul",
        "gpu_matmul",
        "gpu_matmul",
        "q01_low_cpu_high_gpu",
        "gpu",
        (1, 2, 3, 4, 5, 6, 8),
        False,
        "jax_matmul_progress",
        "Pure GPU compute row without model/data-loader effects.",
        profile_axis=PROFILE_AXIS_COLOCATION_COUNT,
    ),
    WorkloadType(
        "cpu_heavy_local_bench",
        "cpu",
        "cpu_parallel",
        "q10_high_cpu_low_gpu",
        "cpu",
        (1, 2, 4, 8, 16, 32, 64, 96, 128, 180, 192),
        True,
        "cpu_parallel_progress_benchmark",
        "Generic CPU-heavy progress benchmark.",
        profile_axis=PROFILE_AXIS_ALLOCATION_WORKERS,
    ),
    WorkloadType(
        "freqduet_cpu_ablation_c17_32",
        "freqduet",
        "freqduet_cpu",
        "q10_high_cpu_low_gpu",
        "cpu",
        (1, 2, 4, 8, 16, 32, 64, 96, 128, 180, 192),
        True,
        "freqduet_or_checkpointable_surrogate",
        "FreqDuet-like CPU workload; use real command only if safe resume is verified.",
        profile_axis=PROFILE_AXIS_COLOCATION_COUNT,
    ),
    WorkloadType(
        "freqduet_cpu_native",
        "freqduet",
        "freqduet_cpu",
        "q10_high_cpu_low_gpu",
        "cpu",
        (1, 2, 4),
        False,
        "native_cpu_colocation_completion_matrix",
        (
            "Native FreqDuet runner_v3 completion rows. The measured axis is "
            "independent one-worker task co-location, not worker allocation."
        ),
        profile_axis=PROFILE_AXIS_COLOCATION_COUNT,
    ),
    WorkloadType(
        "sumo_eval_cpu",
        "sumo",
        "sumo_cpu",
        "q10_high_cpu_low_gpu",
        "cpu",
        (1, 2, 4, 8, 16, 32, 64, 96, 128),
        False,
        "sumo_progress_or_surrogate",
        "SUMO/data-loader-style CPU row.",
        profile_axis=PROFILE_AXIS_COLOCATION_COUNT,
    ),
    WorkloadType(
        "sumo_eval_cpu_native",
        "sumo",
        "sumo_cpu",
        "q10_high_cpu_low_gpu",
        "cpu",
        (1, 2, 4),
        False,
        "native_cpu_colocation_completion_matrix",
        (
            "Native SUMO 1.25 evaluation completion rows. The measured axis "
            "is independent one-worker task co-location."
        ),
        profile_axis=PROFILE_AXIS_COLOCATION_COUNT,
    ),
    WorkloadType(
        "hybrid_rl_resac_ant",
        "ant",
        "hybrid_rl",
        "q11_high_cpu_high_gpu",
        "gpu",
        (1, 2, 3, 4, 5, 6),
        True,
        "resac_env_real_venv.cmd.tpl",
        "RE-SAC Ant train/eval cycle-average ETA.",
        profile_axis=PROFILE_AXIS_COLOCATION_COUNT,
    ),
    WorkloadType(
        "hybrid_rl_resac_halfcheetah",
        "halfcheetah",
        "hybrid_rl",
        "q11_high_cpu_high_gpu",
        "gpu",
        (1, 2, 3, 4, 5, 6),
        True,
        "resac_env_real_venv.cmd.tpl",
        "RE-SAC HalfCheetah train/eval cycle-average ETA.",
        profile_axis=PROFILE_AXIS_COLOCATION_COUNT,
    ),
    WorkloadType(
        "hybrid_rl_resac_hopper",
        "hopper",
        "hybrid_rl",
        "q11_high_cpu_high_gpu",
        "gpu",
        (1, 2, 3, 4, 5, 6),
        True,
        "resac_env_real_venv.cmd.tpl",
        "RE-SAC Hopper train/eval cycle-average ETA.",
        profile_axis=PROFILE_AXIS_COLOCATION_COUNT,
    ),
    WorkloadType(
        "hybrid_rl_resac_walker2d",
        "walker2d",
        "hybrid_rl",
        "q11_high_cpu_high_gpu",
        "gpu",
        (1, 2, 3, 4, 5, 6),
        True,
        "resac_env_real_venv.cmd.tpl",
        "RE-SAC Walker2d train/eval cycle-average ETA.",
        profile_axis=PROFILE_AXIS_COLOCATION_COUNT,
    ),
    WorkloadType(
        "hybrid_rl_bapr_ant",
        "bapr_ant",
        "hybrid_rl",
        "q11_high_cpu_high_gpu",
        "gpu",
        (1, 2, 3, 4, 5, 6),
        True,
        "bapr_ant_real.cmd.tpl",
        "BAPR-style hybrid RL row if environment remains staged.",
        profile_axis=PROFILE_AXIS_COLOCATION_COUNT,
    ),
)


RESIDENT_SCENARIOS = (
    ResidentScenario("empty", "none", ("gpu", "cpu"), ("gpu", "cpu"), "all_profiles", "required", "No resident load."),
    ResidentScenario("half_loaded", "same_workload_half_capacity", ("gpu",), ("gpu",), "middle_profiles", "required", "Same workload resident load below saturation."),
    ResidentScenario("full_loaded", "same_workload_to_capacity_boundary", ("gpu",), ("gpu",), "to_boundary", "required", "Escalate until stable row or capacity boundary."),
    ResidentScenario("half_loaded", "controlled_resident_workers=96", ("cpu",), ("cpu",), "cpu_controlled_half", "required", "Natural-completion CPU target under an independently started 96-worker resident load."),
    ResidentScenario("full_loaded", "controlled_resident_workers=180", ("cpu",), ("cpu",), "cpu_controlled_full", "required", "Natural-completion CPU target under an independently started 180-worker resident load."),
    ResidentScenario(
        "controlled_colocation",
        "none",
        ("cpu",),
        ("cpu",),
        "native_colocation_nontrivial",
        "required",
        (
            "The effective run state created by p2/p4 independent native CPU "
            "tasks on an otherwise idle node."
        ),
        ("freqduet_cpu_native", "sumo_eval_cpu_native"),
    ),
    ResidentScenario("high_vram_resident", "llm_or_memory_resident", ("gpu",), ("gpu",), "add_one", "required", "Resident high-VRAM process, then add target."),
    ResidentScenario("cpu_resident", "cpu_worker_resident", ("gpu", "cpu"), ("gpu", "cpu"), "add_one", "required", "Resident CPU pressure while measuring target marginal ETA."),
    ResidentScenario("mixed_colocation", "cnn_plus_llm", ("gpu",), ("gpu",), "add_one", "required", "Pairwise mixed GPU co-location."),
    ResidentScenario("mixed_colocation", "cnn_plus_hybrid_rl", ("gpu",), ("gpu",), "add_one", "required", "Pairwise CNN/RL co-location."),
    ResidentScenario("mixed_colocation", "llm_plus_hybrid_rl", ("gpu",), ("gpu",), "add_one", "required", "Pairwise LLM/RL co-location."),
    ResidentScenario("mixed_colocation", "cnn_plus_llm_plus_hybrid_rl", ("gpu",), ("gpu",), "add_one", "required", "Triple mixed GPU co-location."),
    ResidentScenario("mixed_colocation", "cpu_plus_gpu_target", ("gpu",), ("gpu",), "add_one", "required", "CPU resident pressure plus GPU target."),
    ResidentScenario("unknown_env", "description_only_or_unknown", ("gpu", "cpu"), ("gpu", "cpu"), "probe_defer", "admission", "Unknown env must probe/defer and cannot enter theorem claim."),
)


MIGRATION_PAIRS = (
    MigrationPair(
        "jtl110gpu:gpu_3080ti_12gb_dual",
        "jtl311linux:gpu_rtx2080_8gb_dual_cpu_fast",
        ("cnn", "llm_small", "hybrid_rl"),
        (0.25, 0.50, 0.75),
        "required",
        "GPU/hybrid migration into CPU-fast but GPU-weaker node.",
    ),
    MigrationPair(
        "jtl311linux:gpu_rtx2080_8gb_dual_cpu_fast",
        "jtl110gpu:gpu_3080ti_12gb_dual",
        ("cnn", "llm_small", "hybrid_rl"),
        (0.25, 0.50, 0.75),
        "required",
        "Reverse GPU/hybrid migration back to 3080Ti node.",
    ),
    MigrationPair(
        "jtl110gpu:gpu_3080ti_12gb_dual",
        "node007:gpu_node007_4x12gb",
        ("cnn", "llm_small", "hybrid_rl"),
        (0.25, 0.50, 0.75),
        "required",
        "Dual-GPU to four-GPU node migration.",
    ),
    MigrationPair(
        "node007:gpu_node007_4x12gb",
        "jtl110gpu:gpu_3080ti_12gb_dual",
        ("cnn", "llm_small", "hybrid_rl"),
        (0.25, 0.50, 0.75),
        "required",
        "Four-GPU node to 3080Ti node migration.",
    ),
    MigrationPair(
        "node001:cpu_hpc_192c",
        "node005:cpu_hpc_192c",
        ("cpu_parallel", "freqduet_cpu", "sumo_cpu"),
        (0.25, 0.50, 0.75),
        "required",
        "Within-class CPU migration/equivalence.",
    ),
    MigrationPair(
        "node001:cpu_hpc_192c",
        "jtl110cpu:cpu_jtl110_128c",
        ("cpu_parallel", "freqduet_cpu", "sumo_cpu"),
        (0.25, 0.50, 0.75),
        "required",
        "Cross-CPU-class migration with staging cost.",
    ),
)


def build_full_factorial_eta_design(*, cache_path: Path = DEFAULT_CACHE) -> dict[str, Any]:
    cache = ServiceRateCache.load(cache_path) if cache_path.exists() else ServiceRateCache()
    eta_rows = _eta_rows(cache)
    migration_rows = _migration_rows()
    status_counts: dict[str, int] = {}
    eta_status_counts: dict[str, int] = {}
    for row in eta_rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
        eta_status_counts[row["eta_status"]] = (
            eta_status_counts.get(row["eta_status"], 0) + 1
        )
    migration_counts: dict[str, int] = {}
    for row in migration_rows:
        migration_counts[row["status"]] = migration_counts.get(row["status"], 0) + 1
    return {
        "gate": "full_factorial_eta_design",
        "status": "DESIGN_READY",
        "cache_path": str(cache_path),
        "eta_row_count": len(eta_rows),
        "migration_row_count": len(migration_rows),
        "status_counts": status_counts,
        "eta_status_counts": eta_status_counts,
        "migration_status_counts": migration_counts,
        "hardware_types": [row.__dict__ for row in HARDWARE_TYPES],
        "workload_types": [row.__dict__ for row in WORKLOAD_TYPES],
        "resident_scenarios": [row.__dict__ for row in RESIDENT_SCENARIOS],
        "migration_pairs": [row.__dict__ for row in MIGRATION_PAIRS],
        "eta_rows": eta_rows,
        "migration_rows": migration_rows,
        "claim_boundary": (
            "This is the declared finite ETA lookup-table domain.  Rows marked "
            "measured are already in the service-cache snapshot.  Rows marked "
            "pending_probe must be measured with task-native tqdm/progress before "
            "they can enter theorem-facing replay.  This design is exhaustive "
            "over the listed hardware, workload, load-state, profile, and "
            "migration factors; it is not a claim about arbitrary future workloads."
        ),
    }


def _eta_rows(cache: ServiceRateCache) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    boundary_map = _boundary_map(cache)
    for hw in HARDWARE_TYPES:
        for workload in WORKLOAD_TYPES:
            if not _hardware_supports_workload(hw, workload):
                continue
            for scenario in RESIDENT_SCENARIOS:
                if not _scenario_applies(hw, workload, scenario):
                    continue
                profiles = _profiles_for(workload, scenario, hw=hw)
                if not profiles:
                    continue
                profile_dimensions = tuple(
                    _profile_dimensions(workload, profile) for profile in profiles
                )
                status, measured, missing, closed = _measurement_status(
                    cache,
                    workload,
                    hw,
                    scenario,
                    profiles,
                    boundary_map=boundary_map,
                )
                (
                    eta_status,
                    eta_measured,
                    service_only,
                    eta_missing,
                    eta_closed,
                ) = _completion_eta_status(
                    cache,
                    workload,
                    hw,
                    scenario,
                    profiles,
                    boundary_map=boundary_map,
                )
                rows.append({
                    "row_id": _row_id(hw, workload, scenario),
                    "tier": _tier(hw, scenario),
                    "priority": scenario.priority,
                    "quadrant": workload.quadrant,
                    "hardware_group": hw.hardware_group,
                    "node": hw.primary_node,
                    "node_bucket": hw.node_bucket,
                    "equivalent_nodes": ";".join(hw.equivalent_nodes),
                    "workload_key": workload.workload_key,
                    "workload_env": workload.workload_env,
                    "family": workload.family,
                    "resource_kind": workload.resource_kind,
                    "resource_state": scenario.resource_state,
                    "resident_mix": scenario.resident_mix,
                    "profile_axis": workload.profile_axis,
                    "profile_ladder": ",".join(str(x) for x in profiles),
                    "allocation_workers_ladder": ",".join(
                        str(allocation_workers)
                        for allocation_workers, _ in profile_dimensions
                    ),
                    "colocation_count_ladder": ",".join(
                        str(colocation_count)
                        for _, colocation_count in profile_dimensions
                    ),
                    "measured_profiles": ",".join(str(x) for x in measured),
                    "boundary_closed_profiles": ",".join(str(x) for x in closed),
                    "missing_profiles": ",".join(str(x) for x in missing),
                    "status": status,
                    "eta_status": eta_status,
                    "eta_measured_profiles": ",".join(
                        str(x) for x in eta_measured
                    ),
                    "service_only_profiles": ",".join(
                        str(x) for x in service_only
                    ),
                    "eta_missing_profiles": ",".join(
                        str(x) for x in eta_missing
                    ),
                    "eta_boundary_closed_profiles": ",".join(
                        str(x) for x in eta_closed
                    ),
                    "eta_source_required": "tqdm/progress",
                    "stable_gate": _stable_gate(workload, scenario),
                    "runner": workload.runner,
                    "legacy_limits_bypassed": "true",
                    "notes": f"{workload.notes} {scenario.notes}",
                })
    return rows


def _migration_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    workload_by_family: dict[str, list[WorkloadType]] = {}
    for workload in WORKLOAD_TYPES:
        if workload.checkpointable:
            workload_by_family.setdefault(workload.family, []).append(workload)
    for pair in MIGRATION_PAIRS:
        for family in pair.target_families:
            for workload in workload_by_family.get(family, []):
                for point in pair.progress_points:
                    rows.append({
                        "row_id": f"migration|{pair.source_bucket}|{pair.destination_bucket}|{workload.workload_key}|p{int(point*100)}",
                        "priority": pair.priority,
                        "source_bucket": pair.source_bucket,
                        "destination_bucket": pair.destination_bucket,
                        "workload_key": workload.workload_key,
                        "workload_env": workload.workload_env,
                        "family": workload.family,
                        "progress_fraction": point,
                        "cost_terms": "checkpoint_flush_s,sync_s,environment_staging_s,resume_warmup_s,lost_work_s,risk_penalty_s",
                        "eligibility": "controlled_benchmark_only; checkpoint_verified; resume_verified; no_user_running_task",
                        "status": "pending_or_measured_live_gate",
                        "notes": pair.notes,
                    })
    return rows


def _hardware_supports_workload(hw: HardwareType, workload: WorkloadType) -> bool:
    if workload.workload_key in {
        "freqduet_cpu_native",
        "sumo_eval_cpu_native",
    }:
        return hw.hardware_group.startswith("cpu_hpc_192c")
    if workload.resource_kind == "gpu":
        return hw.resource_kind == "gpu"
    if workload.resource_kind == "cpu":
        return hw.resource_kind == "cpu" or hw.hardware_group == "gpu_rtx2080_8gb_dual_cpu_fast"
    return False


def _scenario_applies(hw: HardwareType, workload: WorkloadType, scenario: ResidentScenario) -> bool:
    if scenario.workload_keys and workload.workload_key not in scenario.workload_keys:
        return False
    if hw.resource_kind not in scenario.hardware_kinds:
        return False
    if workload.resource_kind not in scenario.target_kinds:
        return False
    if scenario.resource_state in {"high_vram_resident", "mixed_colocation"} and workload.resource_kind != "gpu":
        return False
    if scenario.resident_mix == "cpu_plus_gpu_target" and workload.resource_kind != "gpu":
        return False
    if scenario.resource_state == "cpu_resident" and hw.resource_kind == "cpu" and workload.resource_kind != "cpu":
        return False
    if (
        scenario.resident_mix.startswith("controlled_resident_workers=")
        and workload.workload_key != "cpu_heavy_local_bench"
    ):
        return False
    if (
        scenario.resident_mix.startswith("controlled_resident_workers=")
        and hw.cpu_cores != 192
    ):
        return False
    if workload.workload_key in {
        "freqduet_cpu_native",
        "sumo_eval_cpu_native",
    } and scenario.resource_state not in {
        "empty",
        "controlled_colocation",
        "unknown_env",
    }:
        return False
    if scenario.resource_state == "unknown_env":
        return workload.family in {"hybrid_rl", "cpu_parallel", "freqduet_cpu"}
    if hw.hardware_group.endswith("_equiv") and scenario.resource_state != "empty":
        return bool(
            workload.workload_key
            in {"freqduet_cpu_native", "sumo_eval_cpu_native"}
            and scenario.resource_state == "controlled_colocation"
        )
    return True


def _profiles_for(
    workload: WorkloadType,
    scenario: ResidentScenario,
    hw: HardwareType | None = None,
) -> tuple[int, ...]:
    ladder = workload.profile_ladder
    if workload.workload_key in {
        "freqduet_cpu_native",
        "sumo_eval_cpu_native",
    }:
        if scenario.resource_state == "empty":
            profiles = (1,)
        elif scenario.resource_state == "controlled_colocation":
            profiles = tuple(profile for profile in ladder if profile > 1)
        elif scenario.required_profile_mode == "probe_defer":
            profiles = (1,)
        else:
            profiles = ()
    elif scenario.required_profile_mode == "all_profiles":
        profiles = ladder
    elif scenario.required_profile_mode == "middle_profiles":
        profiles = tuple(x for x in ladder if x in {2, 3, 4, 8, 16, 32, 96})
    elif scenario.required_profile_mode == "to_boundary":
        profiles = ladder
    elif scenario.required_profile_mode == "add_one":
        if workload.resource_kind == "cpu":
            profiles = (1,)
        else:
            profiles = (
                (1, 2, 3)
                if workload.family in {"cnn", "hybrid_rl"}
                else (1,)
            )
    elif scenario.required_profile_mode == "probe_defer":
        profiles = (1,)
    elif scenario.required_profile_mode in {
        "cpu_controlled_half",
        "cpu_controlled_full",
    }:
        profiles = ladder
    else:
        profiles = ladder

    if hw is None or workload.resource_kind != "cpu" or hw.cpu_cores <= 0:
        return profiles
    return tuple(
        profile
        for profile in profiles
        if _profile_core_demand(workload, profile) <= hw.cpu_cores
    )


def _profile_dimensions(
    workload: WorkloadType,
    profile: int,
) -> tuple[int, int]:
    profile_value = int(profile)
    if profile_value <= 0:
        raise ValueError(f"profile must be positive, got {profile!r}")
    if workload.profile_axis == PROFILE_AXIS_ALLOCATION_WORKERS:
        return profile_value, 1
    if workload.profile_axis == PROFILE_AXIS_COLOCATION_COUNT:
        return 1, profile_value
    raise ValueError(f"unsupported profile axis: {workload.profile_axis!r}")


def _profile_core_demand(workload: WorkloadType, profile: int) -> int:
    allocation_workers, colocation_count = _profile_dimensions(workload, profile)
    return allocation_workers * colocation_count


def _measurement_status(
    cache: ServiceRateCache,
    workload: WorkloadType,
    hw: HardwareType,
    scenario: ResidentScenario,
    profiles: Iterable[int],
    *,
    boundary_map: Mapping[tuple[str, ...], int],
) -> tuple[str, list[int], list[int], list[int]]:
    measured: list[int] = []
    missing: list[int] = []
    closed: list[int] = []
    node_boundary_key = (
        workload.workload_key,
        workload.workload_env,
        hw.node_bucket,
        scenario.resource_state,
    )
    boundary_profile = _boundary_profile_for_axis(
        boundary_map,
        node_boundary_key,
        workload.profile_axis,
    )
    if boundary_profile is None:
        hardware_boundary_key = (
            workload.workload_key,
            workload.workload_env,
            hw.hardware_group,
            scenario.resource_state,
        )
        boundary_profile = _boundary_profile_for_axis(
            boundary_map,
            hardware_boundary_key,
            workload.profile_axis,
        )
    for profile in profiles:
        rec = _strict_profile_record(
            cache,
            workload,
            hw,
            scenario,
            int(profile),
        )
        if rec is not None and rec.aggregate_rate > 0.0 and rec.stable_rate_ready:
            measured.append(int(profile))
        elif boundary_profile is not None and int(profile) >= int(boundary_profile):
            closed.append(int(profile))
        else:
            missing.append(int(profile))
    if scenario.resource_state == "unknown_env":
        return "probe_defer_by_design", measured, missing, closed
    if not missing:
        if closed and measured:
            return "measured_to_capacity_boundary", measured, missing, closed
        if closed:
            return "capacity_boundary_closed", measured, missing, closed
        return "measured", measured, missing, closed
    if measured:
        return "partial_pending_probe", measured, missing, closed
    if closed:
        return "boundary_plus_pending_probe", measured, missing, closed
    return "pending_probe", measured, missing, closed


def _completion_eta_status(
    cache: ServiceRateCache,
    workload: WorkloadType,
    hw: HardwareType,
    scenario: ResidentScenario,
    profiles: Iterable[int],
    *,
    boundary_map: Mapping[tuple[str, ...], int],
) -> tuple[str, list[int], list[int], list[int], list[int]]:
    """Separate full completion ETA readiness from steady-service readiness."""

    eta_measured: list[int] = []
    service_only: list[int] = []
    missing: list[int] = []
    closed: list[int] = []
    node_boundary_key = (
        workload.workload_key,
        workload.workload_env,
        hw.node_bucket,
        scenario.resource_state,
    )
    boundary_profile = _boundary_profile_for_axis(
        boundary_map,
        node_boundary_key,
        workload.profile_axis,
    )
    if boundary_profile is None:
        hardware_boundary_key = (
            workload.workload_key,
            workload.workload_env,
            hw.hardware_group,
            scenario.resource_state,
        )
        boundary_profile = _boundary_profile_for_axis(
            boundary_map,
            hardware_boundary_key,
            workload.profile_axis,
        )
    for profile in profiles:
        rec = _strict_profile_record(
            cache,
            workload,
            hw,
            scenario,
            int(profile),
        )
        service_ready = bool(
            rec is not None
            and rec.aggregate_rate > 0.0
            and rec.stable_rate_ready
        )
        completion_ready = bool(
            service_ready
            and rec.completion_model_ready
            and rec.completion_total_wall_s > 0.0
            and rec.completion_unit_s > 0.0
            and "history" not in str(rec.eta_source or "").lower()
        )
        if completion_ready:
            eta_measured.append(int(profile))
        elif service_ready:
            service_only.append(int(profile))
        elif boundary_profile is not None and int(profile) >= int(
            boundary_profile
        ):
            closed.append(int(profile))
        else:
            missing.append(int(profile))
    if scenario.resource_state == "unknown_env":
        return (
            "probe_defer_by_design",
            eta_measured,
            service_only,
            missing,
            closed,
        )
    pending = bool(service_only or missing)
    if not pending:
        if closed and eta_measured:
            status = "eta_measured_to_capacity_boundary"
        elif closed:
            status = "eta_capacity_boundary_closed"
        else:
            status = "eta_measured"
    elif eta_measured:
        status = "eta_partial_pending_probe"
    elif service_only:
        status = "service_measured_eta_pending"
    elif closed:
        status = "eta_boundary_plus_pending_probe"
    else:
        status = "eta_pending_probe"
    return status, eta_measured, service_only, missing, closed


def _strict_profile_record(
    cache: ServiceRateCache,
    workload: WorkloadType,
    hw: HardwareType,
    scenario: ResidentScenario,
    profile: int,
):
    lookup_mix = (
        "" if scenario.resident_mix in {"", "none"} else scenario.resident_mix
    )
    allocation_workers, colocation_count = _profile_dimensions(
        workload, int(profile)
    )
    return cache.get_statewise(
        workload.workload_key,
        allocation_workers=allocation_workers,
        colocation_count=colocation_count,
        node_bucket=hw.node_bucket,
        workload_env=workload.workload_env,
        resource_state=scenario.resource_state,
        resident_mix=lookup_mix,
        strict=True,
    )


def _boundary_profile_for_axis(
    boundary_map: Mapping[tuple[str, ...], int],
    base_key: tuple[str, str, str, str],
    profile_axis: str,
) -> int | None:
    profile = boundary_map.get((*base_key, profile_axis))
    if profile is not None:
        return profile
    if any((*base_key, axis) in boundary_map for axis in PROFILE_AXES):
        return None
    return boundary_map.get(base_key)


def _boundary_map(cache: ServiceRateCache) -> dict[tuple[str, ...], int]:
    out: dict[tuple[str, ...], int] = {}
    for record in (cache.snapshot().get("capacity_boundaries") or []):
        base_key = (
            str(record.get("workload_key") or ""),
            str(record.get("workload_env") or ""),
            str(record.get("resource_state") or "unspecified"),
        )
        scopes = [str(record.get("node_bucket") or "")]
        hardware_class = str(record.get("hardware_class") or "")
        if hardware_class:
            scopes.append(hardware_class)
        legacy_profile = int(record.get("profile") or 0)
        allocation_workers = max(
            1, int(record.get("allocation_workers") or 1)
        )
        colocation_count = max(
            1, int(record.get("colocation_count") or legacy_profile or 1)
        )
        if legacy_profile <= 0:
            continue
        axis_profiles = [(PROFILE_AXIS_COLOCATION_COUNT, colocation_count)]
        if allocation_workers > 1 and colocation_count == 1:
            axis_profiles = [
                (PROFILE_AXIS_ALLOCATION_WORKERS, allocation_workers)
            ]
        elif allocation_workers == 1 and colocation_count == 1:
            axis_profiles.append((PROFILE_AXIS_ALLOCATION_WORKERS, 1))
        for scope in scopes:
            if not scope:
                continue
            key = (base_key[0], base_key[1], scope, base_key[2])
            _put_minimum_boundary(out, key, legacy_profile)
            for profile_axis, profile in axis_profiles:
                _put_minimum_boundary(out, (*key, profile_axis), profile)
    return out


def _put_minimum_boundary(
    boundary_map: dict[tuple[str, ...], int],
    key: tuple[str, ...],
    profile: int,
) -> None:
    old = boundary_map.get(key)
    if old is None or profile < old:
        boundary_map[key] = profile


def _tier(hw: HardwareType, scenario: ResidentScenario) -> str:
    if hw.hardware_group.endswith("_equiv"):
        return "T1_equivalence"
    if scenario.resource_state == "unknown_env":
        return "T3_admission"
    if scenario.resource_state in {
        "half_loaded",
        "full_loaded",
        "mixed_colocation",
        "controlled_colocation",
        "high_vram_resident",
        "cpu_resident",
    }:
        return "T2_load_state"
    return "T0_core_curve"


def _stable_gate(workload: WorkloadType, scenario: ResidentScenario) -> str:
    base = "stable_windows>=5; min_rate_samples>=8; warmup_skip>=5"
    if workload.workload_key in {
        "freqduet_cpu_native",
        "sumo_eval_cpu_native",
    }:
        return (
            "task_native_durable_progress; natural_completion; "
            "startup+outer_loop+terminal_artifact_write; "
            "wave_max_split_conformal_holdout"
        )
    if scenario.resident_mix.startswith("controlled_resident_workers="):
        return (
            "task_native_tqdm; natural_completion; startup+outer_loop+"
            "checkpoint+final_save; exact_five_window_dispatch_regime"
        )
    if workload.family == "hybrid_rl":
        return base + "; stable_cycle_units>=5; timeout_s>=2400"
    if workload.resource_kind == "cpu":
        return "stable_windows>=5; min_rate_samples>=8; warmup_skip>=3; timeout_s>=1800"
    if scenario.resource_state != "empty":
        return base + "; no_early_exit_for_residents"
    return base


def _row_id(hw: HardwareType, workload: WorkloadType, scenario: ResidentScenario) -> str:
    return "|".join([
        hw.hardware_group,
        workload.quadrant,
        workload.workload_key,
        workload.workload_env,
        scenario.resource_state,
        scenario.resident_mix,
    ])


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _markdown(report: Mapping[str, Any], *, eta_csv: Path, migration_csv: Path) -> str:
    status_counts = report.get("status_counts") or {}
    eta_status_counts = report.get("eta_status_counts") or {}
    migration_counts = report.get("migration_status_counts") or {}
    lines = [
        "# Full-Factorial ETA Lookup-Table Design",
        "",
        f"- Status: `{report.get('status')}`",
        f"- ETA design rows: `{report.get('eta_row_count')}`",
        f"- Migration design rows: `{report.get('migration_row_count')}`",
        f"- ETA CSV: `{eta_csv}`",
        f"- Migration CSV: `{migration_csv}`",
        "",
        "## ETA Row Status",
        "",
        "| Status | Rows |",
        "|---|---:|",
    ]
    for key in sorted(status_counts):
        lines.append(f"| `{key}` | {status_counts[key]} |")
    lines.extend([
        "",
        "## Natural-Completion ETA Status",
        "",
        "| Status | Rows |",
        "|---|---:|",
    ])
    for key in sorted(eta_status_counts):
        lines.append(f"| `{key}` | {eta_status_counts[key]} |")
    lines.extend([
        "",
        "## Migration Row Status",
        "",
        "| Status | Rows |",
        "|---|---:|",
    ])
    for key in sorted(migration_counts):
        lines.append(f"| `{key}` | {migration_counts[key]} |")
    lines.extend([
        "",
        "## Hardware Types",
        "",
        "| Group | Primary node | Equivalent nodes | Role |",
        "|---|---|---|---|",
    ])
    for row in report.get("hardware_types") or []:
        lines.append(
            f"| `{row['hardware_group']}` | `{row['primary_node']}` | "
            f"`{', '.join(row.get('equivalent_nodes') or [])}` | {row['role']} |"
        )
    lines.extend([
        "",
        "## Workload Families",
        "",
        "| Quadrant | Workload | Env | Profile axis | Profiles | Runner |",
        "|---|---|---|---|---:|---|",
    ])
    for row in report.get("workload_types") or []:
        lines.append(
            f"| `{row['quadrant']}` | `{row['workload_key']}` | `{row['workload_env']}` | "
            f"`{row['profile_axis']}` | `{list(row['profile_ladder'])}` | `{row['runner']}` |"
        )
    lines.extend([
        "",
        "## Resident Load States",
        "",
        "| State | Resident mix | Profile mode | Priority |",
        "|---|---|---|---|",
    ])
    for row in report.get("resident_scenarios") or []:
        lines.append(
            f"| `{row['resource_state']}` | `{row['resident_mix']}` | "
            f"`{row['required_profile_mode']}` | `{row['priority']}` |"
        )
    lines.extend([
        "",
        "## Probe Order",
        "",
        "1. T0 core curves: empty and same-workload profile ladders to stable row or capacity boundary.",
        "2. T1 equivalence rows: homogeneous-node validation, not a substitute for heterogeneous nodes.",
        "3. T2 load-state rows: high-VRAM, CPU-resident, pairwise mixed, and triple mixed co-location.",
        "4. Migration rows: controlled checkpoint/resume at 25%, 50%, and 75% progress.",
        "5. Rerun SOTA/ours replay only after all required rows for the declared claim are measured.",
        "",
        "## Claim Boundary",
        "",
        str(report.get("claim_boundary") or ""),
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=ARTIFACT_ROOT / "full_factorial_eta_design_20260629.json")
    parser.add_argument("--eta-csv-output", type=Path, default=ARTIFACT_ROOT / "full_factorial_eta_design_rows_20260629.csv")
    parser.add_argument("--migration-csv-output", type=Path, default=ARTIFACT_ROOT / "full_factorial_migration_design_rows_20260629.csv")
    parser.add_argument("--markdown-output", type=Path, default=REPO_ROOT / "md" / "full_factorial_eta_design_20260629.md")
    args = parser.parse_args()
    report = build_full_factorial_eta_design(cache_path=args.cache)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(args.eta_csv_output, list(report["eta_rows"]))
    _write_csv(args.migration_csv_output, list(report["migration_rows"]))
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(
        _markdown(report, eta_csv=args.eta_csv_output, migration_csv=args.migration_csv_output),
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
