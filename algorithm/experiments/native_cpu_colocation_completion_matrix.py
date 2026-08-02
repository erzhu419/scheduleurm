"""Controlled task-native CPU co-location completion measurements.

The profile axis in this runner is always ``colocation_count``.  A profile
``pN`` launches N independent native tasks on one node; it never changes an
individual task's allocation worker count.  Launch is manifest-only unless
``--allow-launch`` is supplied.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import fmean
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithm.experiments.file_progress_completion_wrapper import (
    DISPATCH_CPU_SAMPLE_COUNT,
    DISPATCH_CPU_SAMPLE_WINDOW_S,
    MEASUREMENT_PROTOCOL,
    _sample_dispatch_cpu_state as _sample_child_dispatch_cpu_state,
)


ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
RUN_ROOT = Path.home() / ".claude" / "scheduler" / "experiments" / "runs"

REMOTE_ROOT = "/tmp/scheduleurm_native_cpu_colocation_pkg"
REMOTE_EXP_DIR = f"{REMOTE_ROOT}/algorithm/experiments"
REMOTE_RUNNER = f"{REMOTE_EXP_DIR}/native_cpu_colocation_completion_matrix.py"
REMOTE_GROUP_ROOT = "/tmp/scheduleurm_native_cpu_colocation_groups"
CONTROLLED_CPU_BENCH_ROOT = "/tmp/scheduleurm_native_cpu_benchmarks"
WRAPPER_MODULE = "algorithm.experiments.file_progress_completion_wrapper"

DEFAULT_NODES = tuple(f"node{index:03d}" for index in range(1, 7))
DEFAULT_KINDS = ("freqduet", "sumo")
SUPPORTED_KINDS = (*DEFAULT_KINDS, "light", "cpu_bench")
DEFAULT_PROFILES = (1, 2, 4)
DEFAULT_TASKS_PER_NODE = 4
DEFAULT_MAX_PARALLEL = 6
DEFAULT_SEED_BASE = 727_000
SEED_STRIDE = 10_000
MAX_START_SKEW_S = 1.0
MIN_INTERVAL_SAMPLE_COUNT = 5
MEASUREMENT_SCOPE_SCHEMA_VERSION = 1
DISPATCH_CPU_REGIMES = (
    "cpu_external_idle",
    "cpu_external_light",
    "cpu_external_moderate",
    "cpu_external_heavy",
    "cpu_external_saturated",
)
PROTOCOL_SOURCE_FILES = (
    "progress_units.py",
    "file_progress_completion_wrapper.py",
    "file_progress_cpu_benchmark.py",
    "native_cpu_colocation_completion_matrix.py",
)

COMPLETION_MODEL_PREFIX = "ScheduleurmCompletionModel "
RESOURCE_STATE_PREFIX = "ScheduleurmResourceState "
GROUP_RESULT_PREFIX = "ScheduleurmColocationGroup "
DISPATCH_STATE_PREFIX = "ScheduleurmDispatchState "

FREQDUET_ROOT = "/home/zhengliang01/scheduleurm_work/TransitDuet/FreqDuet/freqduet"
FREQDUET_PYTHON = (
    "/home/zhengliang01/scheduleurm_work/conda_envs/"
    "freqduet-cpu-py310/bin/python"
)
SUMO_WORK_ROOT = "/home/zhengliang01/scheduleurm_work"
SUMO_ROOT = f"{SUMO_WORK_ROOT}/offline-sumo"
SUMO_PYTHON = f"{SUMO_WORK_ROOT}/conda_envs/sumo-lstm-py310/bin/python"
SUMO_RUNTIME = f"{SUMO_WORK_ROOT}/sumo-1.25.0/site-packages"
SUMO_SOURCE = (
    f"{SUMO_WORK_ROOT}/sumo-rl/_standalone_f543609/SUMO_ruiguang"
)
WORKLOAD_METADATA = {
    "freqduet": {
        "workload_key": "freqduet_cpu_native",
        "workload_env": "freqduet",
    },
    "sumo": {
        "workload_key": "sumo_eval_cpu_native",
        "workload_env": "sumo",
    },
    "light": {
        "workload_key": "light_control_local",
        "workload_env": "light_control",
    },
    "cpu_bench": {
        "workload_key": "cpu_heavy_local_bench",
        "workload_env": "cpu",
    },
}


@dataclass(frozen=True)
class NativeCpuColocationTask:
    task_id: str
    task_index: int
    seed: int
    output_dir: str
    progress_file_glob: str
    total_outer_units: int
    allocation_workers: int = 1
    durable_outer_progress: bool = True
    phase_aware_completion_model_required: bool = True


@dataclass(frozen=True)
class NativeCpuColocationCell:
    cell_id: str
    node: str
    kind: str
    profile: int
    profile_label: str
    colocation_count: int
    tasks_per_node_limit: int
    tasks: tuple[NativeCpuColocationTask, ...]
    allocation_workers: int = 1


def colocation_cells(
    *,
    run_id: str,
    nodes: Iterable[str] | None = None,
    profiles: Iterable[int] | None = None,
    kinds: Iterable[str] | None = None,
    tasks_per_node: int = DEFAULT_TASKS_PER_NODE,
    seed_base: int = DEFAULT_SEED_BASE,
) -> tuple[NativeCpuColocationCell, ...]:
    """Build deterministic controlled cells without launching any process."""

    selected_nodes = _ordered_selection(
        nodes, allowed=DEFAULT_NODES, label="node"
    )
    selected_kinds = _ordered_selection(
        kinds, allowed=SUPPORTED_KINDS, label="kind"
    )
    selected_profiles = _normalize_profiles(profiles)
    task_limit = int(tasks_per_node)
    if task_limit <= 0:
        raise ValueError("tasks_per_node must be positive")
    oversized = [value for value in selected_profiles if value > task_limit]
    if oversized:
        raise ValueError(
            "profile colocation_count exceeds tasks_per_node limit: "
            + ",".join(f"p{value}" for value in oversized)
        )

    safe_run_id = _safe_id(run_id)
    cells: list[NativeCpuColocationCell] = []
    for node in selected_nodes:
        for kind in selected_kinds:
            kind_seed_offset = SUPPORTED_KINDS.index(kind) * SEED_STRIDE
            for profile in selected_profiles:
                cell_id = _safe_id(f"{safe_run_id}_{node}_{kind}_p{profile}")
                tasks = []
                for task_index in range(profile):
                    # Pair exact workload seeds across homogeneous nodes.  The
                    # p2 task set is also a prefix of p4, isolating marginal
                    # co-location effects from task-instance variation.
                    seed = int(seed_base) + kind_seed_offset + task_index
                    task_id = _safe_id(
                        f"{node}_{kind}_p{profile}_task{task_index + 1:03d}"
                    )
                    output_dir, progress_glob = _task_output_paths(
                        run_id=safe_run_id,
                        node=node,
                        kind=kind,
                        profile=profile,
                        task_index=task_index,
                        seed=seed,
                    )
                    tasks.append(
                        NativeCpuColocationTask(
                            task_id=task_id,
                            task_index=task_index,
                            seed=seed,
                            output_dir=output_dir,
                            progress_file_glob=progress_glob,
                            total_outer_units=_total_outer_units(kind),
                        )
                    )
                cells.append(
                    NativeCpuColocationCell(
                        cell_id=cell_id,
                        node=node,
                        kind=kind,
                        profile=profile,
                        profile_label=f"p{profile}",
                        colocation_count=profile,
                        tasks_per_node_limit=task_limit,
                        tasks=tuple(tasks),
                    )
                )
    return tuple(cells)


def build_native_cpu_colocation_completion_matrix(
    *,
    allow_launch: bool = False,
    run_id: str = "native_cpu_colocation_completion_20260727",
    nodes: Iterable[str] | None = None,
    profiles: Iterable[int] | None = None,
    profile_counts: Iterable[int] | None = None,
    kinds: Iterable[str] | None = None,
    max_parallel: int = DEFAULT_MAX_PARALLEL,
    tasks_per_node: int = DEFAULT_TASKS_PER_NODE,
    seed_base: int = DEFAULT_SEED_BASE,
    init_cache_state: str = "unspecified",
    required_dispatch_regimes: Mapping[str, str] | None = None,
    deploy_bundle: bool = True,
) -> dict[str, Any]:
    """Build a manifest or run controlled node/profile groups.

    ``max_parallel`` counts nodes.  Profiles on the same node are deliberately
    serialized so separate measured groups cannot interfere with each other.
    """

    normalized_profiles = (
        _normalize_profiles(profiles) if profiles is not None else None
    )
    normalized_profile_counts = (
        _normalize_profiles(profile_counts)
        if profile_counts is not None
        else None
    )
    if (
        normalized_profiles is not None
        and normalized_profile_counts is not None
    ):
        if normalized_profiles != normalized_profile_counts:
            raise ValueError("profiles and profile_counts disagree")
    selected_profiles = (
        normalized_profiles
        if normalized_profiles is not None
        else normalized_profile_counts
    )
    cells = colocation_cells(
        run_id=run_id,
        nodes=nodes,
        profiles=selected_profiles,
        kinds=kinds,
        tasks_per_node=tasks_per_node,
        seed_base=seed_base,
    )
    normalized_required_regimes = _normalize_required_dispatch_regimes(
        cells,
        required_dispatch_regimes,
    )
    measurement_scope = build_native_cpu_colocation_measurement_scope(
        run_id=run_id,
        cells=cells,
        tasks_per_node=tasks_per_node,
        seed_base=seed_base,
        init_cache_state=init_cache_state,
        required_dispatch_regimes=normalized_required_regimes,
    )
    measurement_scope_sha256 = _canonical_json_sha256(measurement_scope)
    parallelism = int(max_parallel)
    if parallelism <= 0:
        raise ValueError("max_parallel must be positive")

    started = time.time()
    if not allow_launch:
        rows = [_manifest_row(cell, init_cache_state=init_cache_state) for cell in cells]
    else:
        rows = _launch_cells(
            cells,
            run_id=run_id,
            max_parallel=parallelism,
            init_cache_state=init_cache_state,
            required_dispatch_regimes=normalized_required_regimes,
            deploy_bundle=bool(deploy_bundle),
        )

    valid_rows = [row for row in rows if bool(row.get("measurement_valid"))]
    ready_task_count = sum(
        int(row.get("completed_task_count") or 0) for row in rows
    )
    failed_task_count = (
        sum(int(row.get("failure_count") or 0) for row in rows)
        if allow_launch
        else 0
    )
    selected_task_count = sum(cell.colocation_count for cell in cells)
    all_ready = bool(rows) and len(valid_rows) == len(rows)
    return {
        "gate": "native_cpu_colocation_completion_matrix",
        "schema_version": 1,
        "measurement_protocol": MEASUREMENT_PROTOCOL,
        "measurement_scope_schema_version": MEASUREMENT_SCOPE_SCHEMA_VERSION,
        "measurement_scope": measurement_scope,
        "measurement_scope_sha256": measurement_scope_sha256,
        "run_id": _safe_id(run_id),
        "allow_launch": bool(allow_launch),
        "profile_axis": "colocation_count",
        "profile_semantics": (
            "pN means N parallel independent controlled tasks on one node; "
            "allocation_workers remains 1 for every task"
        ),
        "allocation_workers": 1,
        "min_interval_sample_count": MIN_INTERVAL_SAMPLE_COUNT,
        "paired_seed_across_nodes": True,
        "nested_task_set_across_profiles": True,
        "nodes": sorted({cell.node for cell in cells}),
        "kinds": sorted({cell.kind for cell in cells}),
        "profile_counts": sorted({cell.colocation_count for cell in cells}),
        "tasks_per_node": int(tasks_per_node),
        "max_parallel": parallelism,
        "parallelism_scope": "nodes",
        "same_node_profiles_serialized": True,
        "required_dispatch_regimes": normalized_required_regimes,
        "dispatch_regime_preflight_enabled": bool(
            normalized_required_regimes
        ),
        "deploy_bundle": bool(deploy_bundle),
        "selected_count": len(cells),
        "selected_task_count": selected_task_count,
        "launched_count": sum(bool(row.get("launched")) for row in rows),
        "admitted_count": len(valid_rows),
        "completed_task_count": ready_task_count,
        "failed_task_count": failed_task_count,
        "all_completion_models_ready": all_ready if allow_launch else False,
        "status": (
            "COMPLETE"
            if allow_launch and all_ready
            else ("RUN_FINISHED_WITH_GAPS" if allow_launch else "MANIFEST")
        ),
        "elapsed_wall_s": time.time() - started,
        "legacy_scheduler_limits_bypassed": True,
        "launch_route": "direct_controlled_remote_benchmark",
        "user_task_control_operations": [],
        "controlled_benchmarks_only": sorted({cell.kind for cell in cells}),
        "rows": rows,
        "claim_boundary": (
            "This runner measures only the explicitly selected controlled CPU "
            "co-location groups. It bypasses legacy scheduler admission limits without "
            "submitting, stopping, requeueing, or otherwise modifying user tasks."
        ),
    }


def build_native_cpu_colocation_measurement_scope(
    *,
    run_id: str,
    cells: Sequence[NativeCpuColocationCell],
    tasks_per_node: int,
    seed_base: int,
    init_cache_state: str,
    required_dispatch_regimes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return the exact reusable-artifact contract for one measurement wave."""

    selected_cells = tuple(cells)
    return {
        "schema_version": MEASUREMENT_SCOPE_SCHEMA_VERSION,
        "measurement_protocol": MEASUREMENT_PROTOCOL,
        "run_id": _safe_id(run_id),
        "nodes": sorted({cell.node for cell in selected_cells}),
        "kinds": sorted({cell.kind for cell in selected_cells}),
        "profile_counts": sorted(
            {int(cell.colocation_count) for cell in selected_cells}
        ),
        "cell_ids": sorted(cell.cell_id for cell in selected_cells),
        "calibration_cell_ids": sorted(
            _calibration_cell_id_for_cell(cell) for cell in selected_cells
        ),
        "expected_cell_count": len(selected_cells),
        "expected_task_count": sum(
            int(cell.colocation_count) for cell in selected_cells
        ),
        "tasks_per_node": int(tasks_per_node),
        "allocation_workers": 1,
        "seed_base": int(seed_base),
        "init_cache_state": str(init_cache_state),
        "required_dispatch_regimes": dict(
            sorted((required_dispatch_regimes or {}).items())
        ),
        "protocol_source_sha256": _protocol_source_sha256(),
    }


def aggregate_task_results(
    task_rows: Sequence[Mapping[str, Any]],
    *,
    expected_count: int | None = None,
    external_load_telemetry: Mapping[str, Any] | None = None,
    max_start_skew_s: float = MAX_START_SKEW_S,
) -> dict[str, Any]:
    """Aggregate completed tasks while excluding every failed task."""

    rows = [dict(row) for row in task_rows]
    expected = len(rows) if expected_count is None else max(0, int(expected_count))
    valid_rows = [row for row in rows if _task_measurement_valid(row)]
    failed_count = max(0, expected - len(valid_rows))

    flow_rows: list[tuple[str, float]] = []
    eta_rows: list[dict[str, Any]] = []
    service_units = 0.0
    model_service_rate = 0.0
    service_units_seen: set[str] = set()
    for row in valid_rows:
        task_id = str(row.get("task_id") or "")
        model = row.get("completion_model")
        model = model if isinstance(model, Mapping) else {}
        launch_offset = _finite_nonnegative(row.get("launch_offset_s"), default=0.0)
        flow_s = _finite_positive(row.get("flow_time_s"))
        if flow_s is None:
            model_wall = _finite_positive(model.get("total_wall_s"))
            if model_wall is not None:
                flow_s = launch_offset + model_wall
        if flow_s is not None:
            flow_rows.append((task_id, flow_s))

        eta_s = _finite_positive(row.get("eta_s"))
        if eta_s is None:
            eta_s = _model_eta_s(model)
        if eta_s is not None:
            eta_rows.append(
                {
                    "task_id": task_id,
                    "eta_s": eta_s,
                    "completion_offset_s": launch_offset + eta_s,
                }
            )

        total_units = _finite_positive(model.get("total_units"))
        if total_units is not None:
            service_units += total_units
        unit_s = _finite_positive(
            model.get("completion_unit_s", model.get("amortized_unit_s"))
        )
        if unit_s is not None:
            model_service_rate += 1.0 / unit_s
        unit = str(model.get("unit") or "")
        if unit:
            service_units_seen.add(unit)

    makespan_s = max((value for _, value in flow_rows), default=None)
    mean_flow_s = (
        fmean(value for _, value in flow_rows) if flow_rows else None
    )
    aggregate_service = (
        service_units / makespan_s
        if makespan_s is not None and makespan_s > 0.0 and service_units > 0.0
        else None
    )

    launch_offsets = [
        value
        for value in (
            _finite_nonnegative(row.get("launch_offset_s")) for row in rows
        )
        if value is not None
    ]
    launch_skew_s = (
        max(launch_offsets) - min(launch_offsets)
        if len(launch_offsets) >= 2
        else (0.0 if len(launch_offsets) == expected and expected > 0 else None)
    )
    near_sync = bool(
        expected > 0
        and len(launch_offsets) == expected
        and launch_skew_s is not None
        and launch_skew_s <= max(0.0, float(max_start_skew_s))
    )
    telemetry = (
        dict(external_load_telemetry)
        if isinstance(external_load_telemetry, Mapping)
        else None
    )
    return {
        "expected_task_count": expected,
        "attempted_task_count": len(rows),
        "completed_task_count": len(valid_rows),
        "failure_count": failed_count,
        "all_tasks_ready": expected > 0 and len(valid_rows) == expected,
        "valid_task_ids": [str(row.get("task_id") or "") for row in valid_rows],
        "failed_task_ids": [
            str(row.get("task_id") or "")
            for row in rows
            if not _task_measurement_valid(row)
        ],
        "per_task_eta_s": {
            str(row["task_id"]): float(row["eta_s"]) for row in eta_rows
        },
        "per_task_eta": eta_rows,
        "makespan_s": makespan_s,
        "mean_flow_s": mean_flow_s,
        "aggregate_service_units": service_units,
        "aggregate_service_unit_per_s": aggregate_service,
        "aggregate_model_service_unit_per_s": (
            model_service_rate if model_service_rate > 0.0 else None
        ),
        "service_unit": (
            next(iter(service_units_seen))
            if len(service_units_seen) == 1
            else ("mixed" if service_units_seen else None)
        ),
        "start_barrier_used": True,
        "launch_skew_s": launch_skew_s,
        "max_start_skew_s": max(0.0, float(max_start_skew_s)),
        "near_synchronous_start": near_sync,
        "resource_telemetry_ready": bool(
            telemetry and telemetry.get("resource_telemetry_ready")
        ),
        "measurement_protocol": (
            telemetry.get("measurement_protocol") if telemetry else None
        ),
        "external_cpu_core_equiv": (
            telemetry.get("external_cpu_core_equiv") if telemetry else None
        ),
        "external_cpu_bucket": (
            telemetry.get("external_cpu_bucket") if telemetry else None
        ),
        "dispatch_state_ready": bool(
            telemetry and telemetry.get("dispatch_state_ready")
        ),
        "dispatch_sample_count": (
            telemetry.get("dispatch_sample_count") if telemetry else None
        ),
        "dispatch_logical_cpu_count": (
            telemetry.get("dispatch_logical_cpu_count") if telemetry else None
        ),
        "dispatch_external_cpu_regime": (
            telemetry.get("dispatch_external_cpu_regime")
            if telemetry
            else None
        ),
        "dispatch_external_cpu_fraction": (
            telemetry.get("dispatch_external_cpu_fraction")
            if telemetry
            else None
        ),
        "dispatch_external_cpu_fraction_p95": (
            telemetry.get("dispatch_external_cpu_fraction_p95")
            if telemetry
            else None
        ),
        "dispatch_external_cpu_core_equiv": (
            telemetry.get("dispatch_external_cpu_core_equiv")
            if telemetry
            else None
        ),
        "dispatch_external_cpu_core_equiv_p95": (
            telemetry.get("dispatch_external_cpu_core_equiv_p95")
            if telemetry
            else None
        ),
        "dispatch_external_cpu_bucket": (
            telemetry.get("dispatch_external_cpu_bucket")
            if telemetry
            else None
        ),
        "run_window_external_cpu_core_equiv": (
            telemetry.get("run_window_external_cpu_core_equiv")
            if telemetry
            else None
        ),
        "run_window_external_cpu_bucket": (
            telemetry.get("run_window_external_cpu_bucket")
            if telemetry
            else None
        ),
        "external_load_telemetry": telemetry,
        "failed_tasks_excluded_from_aggregates": True,
    }


def _aggregate_task_results(
    task_rows: Sequence[Mapping[str, Any]],
    *,
    expected_count: int | None = None,
    external_load_telemetry: Mapping[str, Any] | None = None,
    max_start_skew_s: float = MAX_START_SKEW_S,
) -> dict[str, Any]:
    """Backward-friendly private spelling for focused experiment tests."""

    return aggregate_task_results(
        task_rows,
        expected_count=expected_count,
        external_load_telemetry=external_load_telemetry,
        max_start_skew_s=max_start_skew_s,
    )


def _manifest_row(
    cell: NativeCpuColocationCell, *, init_cache_state: str
) -> dict[str, Any]:
    cell_payload = asdict(cell)
    cell_payload["tasks"] = [asdict(task) for task in cell.tasks]
    workload = WORKLOAD_METADATA[cell.kind]
    node_bucket = f"{cell.node}:cpu_hpc_192c"
    return {
        **cell_payload,
        **workload,
        "calibration_cell_id": _calibration_cell_id_for_cell(cell),
        "node_bucket": node_bucket,
        "hardware_class": "cpu_hpc_192c",
        "resource_kind": "cpu",
        "resource_state": (
            "empty" if cell.colocation_count == 1 else "controlled_colocation"
        ),
        "init_cache_state": str(init_cache_state),
        "profile_axis": "colocation_count",
        "profile_semantics": (
            f"{cell.profile_label} is {cell.colocation_count} parallel "
            "independent tasks"
        ),
        "task_count": cell.colocation_count,
        "launched": False,
        "status": "MANIFEST",
        "measurement_valid": False,
        "completion_model_ready": False,
        "measurement_protocol": MEASUREMENT_PROTOCOL,
        "min_interval_sample_count": MIN_INTERVAL_SAMPLE_COUNT,
        "start_barrier_required": True,
        "legacy_scheduler_limits_bypassed": True,
        "user_task_control_operations": [],
    }


def _calibration_cell_id_for_cell(
    cell: NativeCpuColocationCell,
) -> str:
    workload = WORKLOAD_METADATA[cell.kind]
    return _safe_id(
        "|".join(
            (
                str(workload["workload_key"]),
                cell.node,
                f"aw{cell.allocation_workers}",
                f"cc{cell.colocation_count}",
                f"u{cell.tasks[0].total_outer_units if cell.tasks else 0}",
            )
        )
    )


def _normalize_required_dispatch_regimes(
    cells: Sequence[NativeCpuColocationCell],
    raw: Mapping[str, str] | None,
) -> dict[str, str]:
    if not raw:
        return {}
    expected = {
        _calibration_cell_id_for_cell(cell)
        for cell in cells
    }
    normalized = {}
    for raw_cell_id, raw_regime in raw.items():
        cell_id = str(raw_cell_id).strip()
        regime = str(raw_regime).strip()
        if cell_id not in expected:
            raise ValueError(
                f"required dispatch regime references unknown cell {cell_id!r}"
            )
        if regime not in DISPATCH_CPU_REGIMES:
            raise ValueError(
                f"unsupported required dispatch regime {regime!r}"
            )
        normalized[cell_id] = regime
    return dict(sorted(normalized.items()))


def _launch_cells(
    cells: Sequence[NativeCpuColocationCell],
    *,
    run_id: str,
    max_parallel: int,
    init_cache_state: str,
    required_dispatch_regimes: Mapping[str, str],
    deploy_bundle: bool,
) -> list[dict[str, Any]]:
    by_node: dict[str, list[NativeCpuColocationCell]] = {}
    for cell in cells:
        by_node.setdefault(cell.node, []).append(cell)
    if not by_node:
        return []

    rows: list[dict[str, Any]] = []
    workers = max(1, min(int(max_parallel), len(by_node)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _run_node_cells,
                node,
                node_cells,
                run_id=run_id,
                init_cache_state=init_cache_state,
                required_dispatch_regimes=required_dispatch_regimes,
                deploy_bundle=deploy_bundle,
            ): node
            for node, node_cells in by_node.items()
        }
        for future in as_completed(futures):
            node = futures[future]
            try:
                rows.extend(future.result())
            except Exception as exc:
                rows.extend(
                    _exception_row(
                        cell,
                        init_cache_state=init_cache_state,
                        exc=exc,
                    )
                    for cell in by_node[node]
                )
    rows.sort(key=lambda row: str(row.get("cell_id") or ""))
    return rows


def _run_node_cells(
    node: str,
    cells: Sequence[NativeCpuColocationCell],
    *,
    run_id: str,
    init_cache_state: str,
    required_dispatch_regimes: Mapping[str, str],
    deploy_bundle: bool,
) -> list[dict[str, Any]]:
    deployment_dir = (
        RUN_ROOT / _safe_id(f"{run_id}_{node}_colocation_deploy") / "raw"
    )
    deployment_dir.mkdir(parents=True, exist_ok=True)
    if deploy_bundle:
        try:
            _deploy_remote_bundle(node, deployment_dir)
        except Exception as exc:
            return [
                _exception_row(
                    cell, init_cache_state=init_cache_state, exc=exc
                )
                for cell in cells
            ]

    rows = []
    for cell in cells:
        try:
            rows.append(
                _run_profile_cell(
                    cell,
                    run_id=run_id,
                    init_cache_state=init_cache_state,
                    required_dispatch_regime=required_dispatch_regimes.get(
                        _calibration_cell_id_for_cell(cell)
                    ),
                )
            )
        except Exception as exc:
            rows.append(
                _exception_row(
                    cell, init_cache_state=init_cache_state, exc=exc
                )
            )
    return rows


def _run_profile_cell(
    cell: NativeCpuColocationCell,
    *,
    run_id: str,
    init_cache_state: str,
    required_dispatch_regime: str | None = None,
) -> dict[str, Any]:
    from .remote_workload_selected_profile_probe import (
        _run_remote_capture,
        _write_remote_text,
    )

    group_run_id = _safe_id(f"{run_id}_{cell.node}_{cell.kind}_p{cell.profile}")
    local_raw_dir = RUN_ROOT / group_run_id / "raw"
    local_raw_dir.mkdir(parents=True, exist_ok=True)
    remote_group_dir = f"{REMOTE_GROUP_ROOT}/{group_run_id}"
    runtime_tasks = [
        _runtime_task_spec(cell=cell, task=task, group_run_id=group_run_id)
        for task in cell.tasks
    ]
    group_spec = {
        "schema_version": 1,
        "group_id": group_run_id,
        "node": cell.node,
        "kind": cell.kind,
        "profile": cell.profile,
        "colocation_count": cell.colocation_count,
        "allocation_workers": 1,
        "group_dir": remote_group_dir,
        "required_dispatch_regime": required_dispatch_regime,
        "tasks": runtime_tasks,
    }
    remote_spec = f"{remote_group_dir}/group_spec.json"
    _write_remote_text(
        cell.node,
        remote_spec,
        json.dumps(group_spec, sort_keys=True),
        local_raw_dir / "group_spec",
    )
    command = (
        f"python3 -u {REMOTE_RUNNER} "
        f"--remote-group-spec-file {_shell_quote(remote_spec)}"
    )
    rc, out, err = _run_remote_capture(
        cell.node,
        command,
        local_raw_dir / "group_run",
        timeout_s=7200,
    )
    combined_log = local_raw_dir / "group_completion.log"
    combined_log.write_text(
        (out or "") + ("\nSTDERR:\n" + err if err else ""),
        encoding="utf-8",
    )
    payload = _parse_prefixed_json(out, GROUP_RESULT_PREFIX)
    if payload is None:
        return {
            **_manifest_row(cell, init_cache_state=init_cache_state),
            "run_id": group_run_id,
            "launched": True,
            "status": "NOT_READY",
            "returncode": int(rc),
            "task_results": [],
            "expected_task_count": cell.colocation_count,
            "completed_task_count": 0,
            "failure_count": cell.colocation_count,
            "all_tasks_ready": False,
            "near_synchronous_start": False,
            "log_path": str(combined_log),
            "stderr_tail": (err or "")[-1000:],
            "error": "remote_group_result_missing",
        }

    payload_status = str(payload.get("status") or "")
    telemetry = payload.get("external_load_telemetry")
    if payload_status == "DEFERRED_REGIME_MISMATCH":
        aggregate = aggregate_task_results(
            [],
            expected_count=0,
            external_load_telemetry=(
                telemetry if isinstance(telemetry, Mapping) else None
            ),
        )
        return {
            **_manifest_row(cell, init_cache_state=init_cache_state),
            "run_id": group_run_id,
            "launched": False,
            "controlled_tasks_launched": False,
            "preflight_sampled": True,
            "status": payload_status,
            "returncode": int(rc),
            "measurement_valid": False,
            "completion_model_ready": False,
            "required_dispatch_regime": required_dispatch_regime,
            "observed_dispatch_regime": payload.get(
                "observed_dispatch_regime"
            ),
            "task_results": [],
            **aggregate,
            "expected_task_count": cell.colocation_count,
            "completed_task_count": 0,
            "failure_count": 0,
            "all_tasks_ready": False,
            "near_synchronous_start": False,
            "eta_source": None,
            "service_semantics": "dispatch_regime_preflight_only",
            "log_path": str(combined_log),
            "remote_group_dir": remote_group_dir,
            "stderr_tail": (err or "")[-1000:],
        }

    payload_tasks = {
        str(row.get("task_id") or ""): dict(row)
        for row in payload.get("tasks") or []
        if isinstance(row, Mapping)
    }
    task_results = []
    for task in cell.tasks:
        measured = payload_tasks.get(task.task_id)
        if measured is None:
            measured = {
                "task_id": task.task_id,
                "returncode": None,
                "measurement_valid": False,
                "status": "MISSING",
            }
        task_results.append({**asdict(task), **measured})
    aggregate = aggregate_task_results(
        task_results,
        expected_count=cell.colocation_count,
        external_load_telemetry=(
            telemetry if isinstance(telemetry, Mapping) else None
        ),
    )
    valid = bool(
        int(rc) == 0
        and aggregate["all_tasks_ready"]
        and aggregate["near_synchronous_start"]
        and aggregate["dispatch_state_ready"]
    )
    dispatch_regime = str(
        aggregate.get("dispatch_external_cpu_regime") or ""
    ).strip()
    dispatch_external_present = bool(
        dispatch_regime and dispatch_regime != "cpu_external_idle"
    )
    run_window_external_present = float(
        aggregate.get("run_window_external_cpu_core_equiv") or 0.0
    ) > 2.0
    if cell.colocation_count > 1:
        observed_effective_resource_state = (
            "controlled_colocation_external"
            if dispatch_external_present
            else "controlled_colocation"
        )
        run_window_effective_resource_state = (
            "controlled_colocation_external"
            if run_window_external_present
            else "controlled_colocation"
        )
    else:
        observed_effective_resource_state = (
            "cpu_resident_external" if dispatch_external_present else "empty"
        )
        run_window_effective_resource_state = (
            "cpu_resident_external" if run_window_external_present else "empty"
        )
    return {
        **_manifest_row(cell, init_cache_state=init_cache_state),
        "run_id": group_run_id,
        "launched": True,
        "status": "READY" if valid else "NOT_READY",
        "returncode": int(rc),
        "measurement_valid": valid,
        "completion_model_ready": bool(aggregate["all_tasks_ready"]),
        "required_dispatch_regime": required_dispatch_regime,
        "observed_dispatch_regime": dispatch_regime or None,
        "observed_effective_resource_state": observed_effective_resource_state,
        "run_window_effective_resource_state": (
            run_window_effective_resource_state
        ),
        "task_results": task_results,
        **aggregate,
        "eta_source": "task_native_durable_csv",
        "service_semantics": "natural_completion_colocation",
        "log_path": str(combined_log),
        "remote_group_dir": remote_group_dir,
        "stderr_tail": (err or "")[-1000:],
    }


def _exception_row(
    cell: NativeCpuColocationCell,
    *,
    init_cache_state: str,
    exc: Exception,
) -> dict[str, Any]:
    return {
        **_manifest_row(cell, init_cache_state=init_cache_state),
        "launched": True,
        "status": "EXCEPTION",
        "measurement_valid": False,
        "expected_task_count": cell.colocation_count,
        "completed_task_count": 0,
        "failure_count": cell.colocation_count,
        "all_tasks_ready": False,
        "near_synchronous_start": False,
        "error": f"{type(exc).__name__}: {exc}",
    }


def _deploy_remote_bundle(node: str, raw_dir: Path) -> None:
    from .remote_workload_selected_profile_probe import (
        _run_remote,
        _write_remote_text,
    )

    _run_remote(
        node,
        (
            f"mkdir -p {REMOTE_EXP_DIR} {REMOTE_GROUP_ROOT} && "
            f"touch {REMOTE_ROOT}/algorithm/__init__.py "
            f"{REMOTE_EXP_DIR}/__init__.py"
        ),
        raw_dir / "deploy_mkdir",
        timeout_s=45,
    )
    local_dir = REPO_ROOT / "algorithm" / "experiments"
    for name in (
        "progress_units.py",
        "file_progress_completion_wrapper.py",
        "file_progress_cpu_benchmark.py",
        "native_cpu_colocation_completion_matrix.py",
    ):
        _write_remote_text(
            node,
            f"{REMOTE_EXP_DIR}/{name}",
            (local_dir / name).read_text(encoding="utf-8"),
            raw_dir / f"deploy_{name}",
        )


def _runtime_task_spec(
    *,
    cell: NativeCpuColocationCell,
    task: NativeCpuColocationTask,
    group_run_id: str,
) -> dict[str, Any]:
    if cell.kind == "freqduet":
        python_path, artifact_globs, child_script = _freqduet_task_command(
            task=task
        )
    elif cell.kind == "sumo":
        python_path, artifact_globs, child_script = _sumo_task_command(
            task=task,
            node=cell.node,
            group_run_id=group_run_id,
        )
    elif cell.kind in {"light", "cpu_bench"}:
        python_path, artifact_globs, child_script = _cpu_benchmark_task_command(
            task=task,
            kind=cell.kind,
            group_run_id=group_run_id,
        )
    else:
        raise ValueError(f"unsupported native CPU kind {cell.kind!r}")

    argv = [
        python_path,
        "-u",
        "-m",
        WRAPPER_MODULE,
        "--progress-file-glob",
        task.progress_file_glob,
        "--total",
        str(task.total_outer_units),
        "--unit",
        ("episode" if cell.kind in DEFAULT_KINDS else "step"),
        "--poll-s",
        "0.5",
        "--csv-has-header",
        "--skip-dispatch-state-sample",
    ]
    for pattern in artifact_globs:
        argv.extend(["--artifact-glob", pattern])
    argv.extend(["--", "bash", "-lc", child_script])
    return {
        **asdict(task),
        "kind": cell.kind,
        "argv": argv,
        "env": {"PYTHONPATH": REMOTE_ROOT},
    }


def _freqduet_task_command(
    *, task: NativeCpuColocationTask
) -> tuple[str, tuple[str, ...], str]:
    output = task.output_dir
    artifact_globs = (
        f"{output}/logs/*/checkpoints/*.pt",
        f"{output}/logs/*/*.json",
    )
    script = f"""
set -euo pipefail
cd {_shell_quote(FREQDUET_ROOT)}
export CUDA_VISIBLE_DEVICES=
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export TORCH_NUM_THREADS=1 FREQDUET_TORCH_THREADS=1
export PYGAME_HIDE_SUPPORT_PROMPT=1 SDL_VIDEODRIVER=dummy
unset FREQDUET_SUPPRESS_HEAVY_ARTIFACTS || true
test -x {_shell_quote(FREQDUET_PYTHON)}
test -r configs_freqduet/F_freqduet_timetable_hiro.yaml
rm -rf {_shell_quote(output)}
mkdir -p {_shell_quote(output + '/logs')}
exec {_shell_quote(FREQDUET_PYTHON)} -u runner_v3.py \
  --config configs_freqduet/F_freqduet_timetable_hiro.yaml \
  --episodes {int(task.total_outer_units)} \
  --seed {int(task.seed)} \
  --upper-warmup-eps 0 \
  --no-resume \
  --logs-dir {_shell_quote(output + '/logs')}
""".strip()
    return FREQDUET_PYTHON, artifact_globs, script


def _sumo_task_command(
    *,
    task: NativeCpuColocationTask,
    node: str,
    group_run_id: str,
) -> tuple[str, tuple[str, ...], str]:
    output = task.output_dir
    tasks = [
        {
            "kind": "baseline",
            "method": "nohold",
            "policy_kind": "nohold",
            "policy_seed": int(task.seed),
            "sumo_seed": int(task.seed) + episode_index,
            "episode_idx": episode_index,
            "max_steps": 600,
        }
        for episode_index in range(int(task.total_outer_units))
    ]
    tasks_json = json.dumps(tasks, separators=(",", ":"))
    artifact_globs = (
        f"{output}/metrics.csv",
        f"{output}/runtime_manifest.json",
    )
    script = f"""
set -euo pipefail
ROOT={_shell_quote(SUMO_ROOT)}
PY={_shell_quote(SUMO_PYTHON)}
RT={_shell_quote(SUMO_RUNTIME)}
SRC={_shell_quote(SUMO_SOURCE)}
OUT={_shell_quote(output)}
export SUMO_HOME="$RT/sumo"
export PATH="$(dirname "$PY"):$SUMO_HOME/bin:$PATH"
export PYTHONPATH="$SUMO_HOME/tools:$RT:${{PYTHONPATH:-}}"
export LIBSUMO_AS_TRACI=1 CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
test -x "$PY"
test -x "$SUMO_HOME/bin/sumo"
test -r "$SRC/online_control/control_sim_traci_period.sumocfg"
rm -rf "$OUT"
mkdir -p "$OUT"
printf '%s\\n' {_shell_quote(tasks_json)} > "$OUT/tasks.json"
"$PY" -u "$ROOT/scripts/write_run_manifest.py" \
  --output "$OUT/runtime_manifest.json" --expected_sumo 1.25.0 \
  --label {_shell_quote(group_run_id + '-' + node + '-' + task.task_id)}
SB=$(mktemp -d "${{TMPDIR:-/tmp}}/sumo-{_safe_id(task.task_id)}.XXXXXX")
cleanup() {{ rm -rf -- "$SB"; }}
trap cleanup EXIT
ln -s "$ROOT" "$SB/offline-sumo"
P="$SB/sumo-rl/_standalone_f543609/SUMO_ruiguang"
mkdir -p "$P"
for d in b_network c_social_vehicle_rou d_bus_rou e_passenger_rou; do
  ln -s "$SRC/$d" "$P/$d"
done
cp -a "$SRC/online_control" "$P/"
"$PY" -u "$SB/offline-sumo/eval_revision_metrics.py" \
  --tasks_json "$OUT/tasks.json" --out_csv "$OUT/metrics.csv" \
  --n_workers 1 --expected_sumo_version 1.25.0
""".strip()
    return SUMO_PYTHON, artifact_globs, script


def _cpu_benchmark_task_command(
    *,
    task: NativeCpuColocationTask,
    kind: str,
    group_run_id: str,
) -> tuple[str, tuple[str, ...], str]:
    output = task.output_dir
    mode = "light" if kind == "light" else "cpu"
    work_items = 1_000 if kind == "light" else 400_000
    sleep_s = 0.05 if kind == "light" else 0.0
    checkpoint_interval = max(1, int(task.total_outer_units) // 4)
    artifact_globs = (
        f"{output}/checkpoints/*.bin",
        f"{output}/final.json",
    )
    script = f"""
set -euo pipefail
OUT={_shell_quote(output)}
rm -rf "$OUT"
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec python3 -u {REMOTE_EXP_DIR}/file_progress_cpu_benchmark.py \
  --progress-file "$OUT/progress.csv" \
  --output-dir "$OUT" \
  --steps {int(task.total_outer_units)} \
  --mode {mode} \
  --work-items {work_items} \
  --sleep-s {sleep_s} \
  --seed {int(task.seed)} \
  --checkpoint-interval {checkpoint_interval} \
  --checkpoint-bytes {8 * 1024 * 1024} \
  --label {_shell_quote(group_run_id + '-' + task.task_id)}
""".strip()
    return "python3", artifact_globs, script


def _task_output_paths(
    *,
    run_id: str,
    node: str,
    kind: str,
    profile: int,
    task_index: int,
    seed: int,
) -> tuple[str, str]:
    task_leaf = _safe_id(f"task{task_index + 1:03d}_seed{seed}")
    if kind == "freqduet":
        output = (
            f"{FREQDUET_ROOT}/results_freqduet/{run_id}/"
            f"{node}/p{profile}/{task_leaf}"
        )
        return output, f"{output}/logs/*/diagnostics.csv"
    if kind == "sumo":
        output = (
            f"{SUMO_ROOT}/experiment_output/{run_id}/"
            f"{node}/p{profile}/{task_leaf}"
        )
        return output, f"{output}/metrics.csv"
    if kind in {"light", "cpu_bench"}:
        output = (
            f"{CONTROLLED_CPU_BENCH_ROOT}/{run_id}/"
            f"{node}/p{profile}/{task_leaf}"
        )
        return output, f"{output}/progress.csv"
    raise ValueError(f"unsupported native CPU kind {kind!r}")


def _total_outer_units(kind: str) -> int:
    if kind == "light":
        return 300
    if kind == "cpu_bench":
        return 120
    return 6


def _run_remote_group(spec_path: Path) -> int:
    """Remote-only coordinator for one node/profile group."""

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    _validate_remote_group_spec(spec)
    group_dir = Path(str(spec["group_dir"]))
    group_dir.mkdir(parents=True, exist_ok=True)
    gate_path = group_dir / "start.gate"
    if gate_path.exists():
        gate_path.unlink()

    dispatch_state = _sample_dispatch_cpu_state()
    required_dispatch_regime = str(
        spec.get("required_dispatch_regime") or ""
    ).strip()
    observed_dispatch_regime = str(
        dispatch_state.get("dispatch_external_cpu_regime") or ""
    ).strip()
    if (
        required_dispatch_regime
        and observed_dispatch_regime != required_dispatch_regime
    ):
        payload = {
            "schema_version": 1,
            "group_id": spec["group_id"],
            "node": spec["node"],
            "kind": spec["kind"],
            "profile": int(spec["profile"]),
            "colocation_count": int(spec["colocation_count"]),
            "allocation_workers": 1,
            "status": "DEFERRED_REGIME_MISMATCH",
            "controlled_tasks_launched": False,
            "required_dispatch_regime": required_dispatch_regime,
            "observed_dispatch_regime": observed_dispatch_regime or None,
            "tasks": [],
            "external_load_telemetry": _dispatch_only_external_load_telemetry(
                dispatch_state
            ),
        }
        print(
            GROUP_RESULT_PREFIX
            + json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            flush=True,
        )
        return 0

    records: list[dict[str, Any]] = []
    try:
        for raw_task in spec["tasks"]:
            task = dict(raw_task)
            task_id = str(task["task_id"])
            start_path = group_dir / f"{task_id}.start.json"
            task_spec_path = group_dir / f"{task_id}.task.json"
            task_log_path = group_dir / f"{task_id}.completion.log"
            if start_path.exists():
                start_path.unlink()
            worker_spec = {
                "gate_path": str(gate_path),
                "start_path": str(start_path),
                "argv": task["argv"],
                "env": task.get("env") or {},
            }
            _atomic_write_json(task_spec_path, worker_spec)
            log_handle = task_log_path.open("w", encoding="utf-8")
            try:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        str(Path(__file__).resolve()),
                        "--remote-task-spec-file",
                        str(task_spec_path),
                    ],
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                )
            except BaseException:
                log_handle.close()
                raise
            records.append(
                {
                    "task": task,
                    "process": process,
                    "log_handle": log_handle,
                    "log_path": task_log_path,
                    "start_path": start_path,
                    "finished_monotonic_s": None,
                }
            )
    except BaseException:
        for record in records:
            process = record["process"]
            if process.poll() is None:
                process.terminate()
            record["log_handle"].close()
        raise

    busy_start, total_start = _cpu_counters()
    release_wall_s = time.time()
    release_monotonic_s = time.perf_counter()
    gate_path.write_text(
        json.dumps(
            {
                "release_wall_s": release_wall_s,
                "release_monotonic_s": release_monotonic_s,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    pending = set(range(len(records)))
    try:
        while pending:
            for index in tuple(pending):
                if records[index]["process"].poll() is None:
                    continue
                records[index]["finished_monotonic_s"] = time.perf_counter()
                pending.remove(index)
            if pending:
                time.sleep(0.05)
    except BaseException:
        for index in pending:
            process = records[index]["process"]
            if process.poll() is None:
                process.terminate()
        raise
    finally:
        for record in records:
            record["log_handle"].close()

    finished_monotonic_s = time.perf_counter()
    busy_end, total_end = _cpu_counters()
    task_rows = [
        _remote_task_result(
            record,
            release_monotonic_s=release_monotonic_s,
        )
        for record in records
    ]
    telemetry = _group_external_load_telemetry(
        task_rows=task_rows,
        dispatch_state=dispatch_state,
        busy_start=busy_start,
        total_start=total_start,
        busy_end=busy_end,
        total_end=total_end,
        measurement_wall_s=max(
            1e-9, finished_monotonic_s - release_monotonic_s
        ),
    )
    payload = {
        "schema_version": 1,
        "group_id": spec["group_id"],
        "node": spec["node"],
        "kind": spec["kind"],
        "profile": int(spec["profile"]),
        "colocation_count": int(spec["colocation_count"]),
        "allocation_workers": 1,
        "status": "COMPLETE",
        "controlled_tasks_launched": True,
        "required_dispatch_regime": required_dispatch_regime or None,
        "observed_dispatch_regime": observed_dispatch_regime or None,
        "release_wall_s": release_wall_s,
        "release_monotonic_s": release_monotonic_s,
        "tasks": task_rows,
        "external_load_telemetry": telemetry,
    }
    print(
        GROUP_RESULT_PREFIX
        + json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


def _run_remote_dispatch_state() -> int:
    payload = {
        "schema_version": 1,
        "measurement_protocol": MEASUREMENT_PROTOCOL,
        **_sample_dispatch_cpu_state(),
    }
    print(
        DISPATCH_STATE_PREFIX
        + json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


def _run_remote_task(spec_path: Path) -> int:
    """Wait on a common gate, record the actual release, then exec a wrapper."""

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    gate_path = Path(str(spec["gate_path"]))
    deadline = time.monotonic() + 120.0
    while not gate_path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"start gate was not released: {gate_path}")
        time.sleep(0.005)
    _atomic_write_json(
        Path(str(spec["start_path"])),
        {
            "started_wall_s": time.time(),
            "started_monotonic_s": time.perf_counter(),
        },
    )
    argv = [str(value) for value in spec.get("argv") or []]
    if not argv:
        raise ValueError("remote task argv is empty")
    env = os.environ.copy()
    for key, value in (spec.get("env") or {}).items():
        text_key = str(key)
        text_value = str(value)
        if text_key == "PYTHONPATH" and env.get(text_key):
            text_value = f"{text_value}:{env[text_key]}"
        env[text_key] = text_value
    os.execvpe(argv[0], argv, env)
    return 127


def _remote_task_result(
    record: Mapping[str, Any],
    *,
    release_monotonic_s: float,
) -> dict[str, Any]:
    task = dict(record["task"])
    process = record["process"]
    log_path = Path(record["log_path"])
    text = log_path.read_text(encoding="utf-8", errors="replace")
    completion_model = _parse_prefixed_json(text, COMPLETION_MODEL_PREFIX)
    resource_state = _parse_prefixed_json(text, RESOURCE_STATE_PREFIX)
    start_payload = None
    start_path = Path(record["start_path"])
    if start_path.exists():
        try:
            candidate = json.loads(start_path.read_text(encoding="utf-8"))
            start_payload = candidate if isinstance(candidate, dict) else None
        except (OSError, ValueError, json.JSONDecodeError):
            start_payload = None
    started_monotonic_s = (
        _finite_nonnegative(start_payload.get("started_monotonic_s"))
        if start_payload
        else None
    )
    launch_offset_s = (
        max(0.0, started_monotonic_s - release_monotonic_s)
        if started_monotonic_s is not None
        else None
    )
    finished = _finite_nonnegative(record.get("finished_monotonic_s"))
    flow_time_s = (
        max(0.0, finished - release_monotonic_s)
        if finished is not None
        else None
    )
    model_ready = bool(
        isinstance(completion_model, Mapping)
        and completion_model.get("completion_model_ready")
    )
    interval_sample_count = int(
        (completion_model or {}).get("interval_sample_count") or 0
    )
    interval_gate_ready = interval_sample_count >= MIN_INTERVAL_SAMPLE_COUNT
    progress_durable = bool(
        isinstance(completion_model, Mapping)
        and completion_model.get("progress_source") == "durable_csv"
    )
    returncode = int(process.returncode)
    valid = bool(
        returncode == 0
        and model_ready
        and progress_durable
        and interval_gate_ready
    )
    eta_s = _model_eta_s(
        completion_model if isinstance(completion_model, Mapping) else {}
    )
    return {
        "task_id": str(task["task_id"]),
        "task_index": int(task["task_index"]),
        "seed": int(task["seed"]),
        "output_dir": str(task["output_dir"]),
        "progress_file_glob": str(task["progress_file_glob"]),
        "allocation_workers": 1,
        "launched": True,
        "returncode": returncode,
        "status": "READY" if valid else "NOT_READY",
        "measurement_valid": valid,
        "completion_model_ready": model_ready,
        "interval_sample_count": interval_sample_count,
        "min_interval_sample_count": MIN_INTERVAL_SAMPLE_COUNT,
        "interval_gate_ready": interval_gate_ready,
        "durable_outer_progress_observed": progress_durable,
        "completion_model": completion_model,
        "eta_s": eta_s,
        "launch_offset_s": launch_offset_s,
        "flow_time_s": flow_time_s,
        "started_wall_s": (
            start_payload.get("started_wall_s") if start_payload else None
        ),
        "resource_telemetry_ready": isinstance(resource_state, Mapping),
        "observed_resource_state": resource_state,
        "remote_log_path": str(log_path),
    }


def _group_external_load_telemetry(
    *,
    task_rows: Sequence[Mapping[str, Any]],
    dispatch_state: Mapping[str, Any],
    busy_start: int,
    total_start: int,
    busy_end: int,
    total_end: int,
    measurement_wall_s: float,
) -> dict[str, Any]:
    logical_cpus = max(
        1,
        int(
            dispatch_state.get("dispatch_logical_cpu_count")
            or os.cpu_count()
            or 1
        ),
    )
    delta_total = max(1, int(total_end) - int(total_start))
    busy_fraction = max(
        0.0,
        min(1.0, float(int(busy_end) - int(busy_start)) / delta_total),
    )
    system_busy_core_equiv = busy_fraction * logical_cpus
    own_cpu_values = []
    for row in task_rows:
        state = row.get("observed_resource_state")
        if not isinstance(state, Mapping):
            continue
        own_cpu_s = _finite_nonnegative(state.get("own_cpu_s"))
        if own_cpu_s is not None:
            own_cpu_values.append(own_cpu_s)
    controlled_cpu_s = sum(own_cpu_values)
    controlled_cpu_core_equiv = controlled_cpu_s / max(
        1e-9, float(measurement_wall_s)
    )
    external_cpu_core_equiv = max(
        0.0, system_busy_core_equiv - controlled_cpu_core_equiv
    )
    ready = len(own_cpu_values) == len(task_rows) and bool(task_rows)
    dispatch_external_cpu_core_equiv = _finite_nonnegative(
        dispatch_state.get("dispatch_external_cpu_core_equiv")
    )
    dispatch_sample_count = int(
        dispatch_state.get("dispatch_sample_count") or 0
    )
    dispatch_external_cpu_regime = str(
        dispatch_state.get("dispatch_external_cpu_regime") or ""
    ).strip()
    dispatch_ready = bool(
        dispatch_state.get("dispatch_state_ready")
        and dispatch_external_cpu_core_equiv is not None
        and dispatch_sample_count >= DISPATCH_CPU_SAMPLE_COUNT
        and dispatch_external_cpu_regime
    )
    dispatch_fields = {
        key: value
        for key, value in dispatch_state.items()
        if str(key).startswith("dispatch_")
    }
    return {
        "schema_version": 3,
        "measurement_protocol": MEASUREMENT_PROTOCOL,
        "resource_telemetry_ready": ready,
        **dispatch_fields,
        "dispatch_state_ready": dispatch_ready,
        "dispatch_state_observed_before_controlled_group": bool(
            dispatch_state.get("dispatch_state_observed_before_controlled_group")
        ),
        "dispatch_sample_count": dispatch_sample_count,
        "dispatch_logical_cpu_count": logical_cpus,
        "dispatch_external_cpu_regime": (
            dispatch_external_cpu_regime or None
        ),
        "dispatch_external_cpu_core_equiv": (
            dispatch_external_cpu_core_equiv
        ),
        "dispatch_external_cpu_bucket": (
            _external_cpu_bucket(dispatch_external_cpu_core_equiv)
            if dispatch_external_cpu_core_equiv is not None
            else None
        ),
        "logical_cpus": logical_cpus,
        "measurement_wall_s": float(measurement_wall_s),
        "run_window_system_busy_fraction": busy_fraction,
        "run_window_system_busy_core_equiv": system_busy_core_equiv,
        "run_window_external_cpu_core_equiv": external_cpu_core_equiv,
        "run_window_external_cpu_bucket": _external_cpu_bucket(
            external_cpu_core_equiv
        ),
        # Compatibility aliases retain their run-window meaning.
        "system_busy_fraction": busy_fraction,
        "system_busy_core_equiv": system_busy_core_equiv,
        "controlled_task_cpu_s": controlled_cpu_s,
        "controlled_task_cpu_core_equiv": controlled_cpu_core_equiv,
        "controlled_task_accounted_count": len(own_cpu_values),
        "expected_controlled_task_count": len(task_rows),
        "external_cpu_core_equiv": external_cpu_core_equiv,
        "external_cpu_bucket": _external_cpu_bucket(external_cpu_core_equiv),
        "external_load_excludes_controlled_colocation_tasks": True,
    }


def _dispatch_only_external_load_telemetry(
    dispatch_state: Mapping[str, Any],
) -> dict[str, Any]:
    logical_cpus = max(
        1,
        int(
            dispatch_state.get("dispatch_logical_cpu_count")
            or os.cpu_count()
            or 1
        ),
    )
    dispatch_sample_count = int(
        dispatch_state.get("dispatch_sample_count") or 0
    )
    dispatch_regime = str(
        dispatch_state.get("dispatch_external_cpu_regime") or ""
    ).strip()
    dispatch_core_equiv = _finite_nonnegative(
        dispatch_state.get("dispatch_external_cpu_core_equiv")
    )
    dispatch_fields = {
        key: value
        for key, value in dispatch_state.items()
        if str(key).startswith("dispatch_")
    }
    dispatch_ready = bool(
        dispatch_state.get("dispatch_state_ready")
        and dispatch_sample_count >= DISPATCH_CPU_SAMPLE_COUNT
        and dispatch_regime in DISPATCH_CPU_REGIMES
        and dispatch_core_equiv is not None
    )
    return {
        "schema_version": 3,
        "measurement_protocol": MEASUREMENT_PROTOCOL,
        "resource_telemetry_ready": False,
        **dispatch_fields,
        "dispatch_state_ready": dispatch_ready,
        "dispatch_state_observed_before_controlled_group": bool(
            dispatch_state.get(
                "dispatch_state_observed_before_controlled_group"
            )
        ),
        "dispatch_sample_count": dispatch_sample_count,
        "dispatch_logical_cpu_count": logical_cpus,
        "dispatch_external_cpu_regime": dispatch_regime or None,
        "dispatch_external_cpu_core_equiv": dispatch_core_equiv,
        "dispatch_external_cpu_bucket": (
            _external_cpu_bucket(dispatch_core_equiv)
            if dispatch_core_equiv is not None
            else None
        ),
        "run_window_external_cpu_core_equiv": None,
        "run_window_external_cpu_bucket": None,
        "external_cpu_core_equiv": None,
        "external_cpu_bucket": None,
        "controlled_task_accounted_count": 0,
        "expected_controlled_task_count": 0,
        "external_load_excludes_controlled_colocation_tasks": True,
        "dispatch_only_preflight": True,
    }


def _validate_remote_group_spec(spec: Mapping[str, Any]) -> None:
    node = str(spec.get("node") or "")
    kind = str(spec.get("kind") or "")
    profile = int(spec.get("profile") or 0)
    count = int(spec.get("colocation_count") or 0)
    allocation_workers = int(spec.get("allocation_workers") or 0)
    group_dir = str(spec.get("group_dir") or "")
    tasks = list(spec.get("tasks") or [])
    required_dispatch_regime = str(
        spec.get("required_dispatch_regime") or ""
    ).strip()
    if node not in DEFAULT_NODES:
        raise ValueError(f"uncontrolled node {node!r}")
    if kind not in SUPPORTED_KINDS:
        raise ValueError(f"uncontrolled benchmark kind {kind!r}")
    if profile <= 0 or count != profile or len(tasks) != count:
        raise ValueError("profile must equal the independent task count")
    if allocation_workers != 1:
        raise ValueError("allocation_workers must remain 1")
    if (
        required_dispatch_regime
        and required_dispatch_regime not in DISPATCH_CPU_REGIMES
    ):
        raise ValueError(
            f"unsupported required dispatch regime {required_dispatch_regime!r}"
        )
    if not group_dir.startswith(f"{REMOTE_GROUP_ROOT}/"):
        raise ValueError("group_dir is outside the controlled experiment root")
    for task in tasks:
        if int(task.get("allocation_workers") or 0) != 1:
            raise ValueError("task allocation_workers must remain 1")
        argv = [str(value) for value in task.get("argv") or []]
        if WRAPPER_MODULE not in argv:
            raise ValueError("task must use the durable completion wrapper")
        output_dir = str(task.get("output_dir") or "")
        expected_root = {
            "freqduet": FREQDUET_ROOT,
            "sumo": SUMO_ROOT,
            "light": CONTROLLED_CPU_BENCH_ROOT,
            "cpu_bench": CONTROLLED_CPU_BENCH_ROOT,
        }[kind]
        if not output_dir.startswith(f"{expected_root}/"):
            raise ValueError("task output is outside the controlled benchmark root")


def _cpu_counters() -> tuple[int, int]:
    fields = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()
    values = [int(value) for value in fields[1:]]
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    return sum(values) - idle, sum(values)


def _sample_dispatch_cpu_state(
    *,
    window_s: float = DISPATCH_CPU_SAMPLE_WINDOW_S,
    sample_count: int = DISPATCH_CPU_SAMPLE_COUNT,
) -> dict[str, Any]:
    """Measure a multi-window state before any controlled group member exists."""

    state = dict(
        _sample_child_dispatch_cpu_state(
            window_s=window_s,
            sample_count=sample_count,
        )
    )
    observed_before_child = bool(
        state.pop("dispatch_state_observed_before_child", False)
    )
    state["dispatch_state_observed_before_controlled_group"] = (
        observed_before_child
    )
    return state


def _task_measurement_valid(row: Mapping[str, Any]) -> bool:
    returncode = row.get("returncode")
    if returncode is None:
        return False
    try:
        natural_exit = int(returncode) == 0
    except (TypeError, ValueError):
        return False
    if not natural_exit:
        return False
    if row.get("measurement_valid") is not None:
        return bool(row.get("measurement_valid"))
    model = row.get("completion_model")
    return bool(
        isinstance(model, Mapping)
        and model.get("completion_model_ready")
    )


def _protocol_source_sha256() -> dict[str, str]:
    local_dir = REPO_ROOT / "algorithm" / "experiments"
    return {
        name: hashlib.sha256((local_dir / name).read_bytes()).hexdigest()
        for name in PROTOCOL_SOURCE_FILES
    }


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _model_eta_s(model: Mapping[str, Any]) -> float | None:
    predicted = _finite_positive(model.get("predicted_total_s"))
    if predicted is not None:
        return predicted
    total_units = _finite_positive(model.get("total_units"))
    unit_s = _finite_positive(
        model.get("completion_unit_s", model.get("amortized_unit_s"))
    )
    if total_units is None or unit_s is None:
        return None
    startup = _finite_nonnegative(model.get("startup_overhead_s"), default=0.0)
    terminal = _finite_nonnegative(
        model.get(
            "terminal_overhead_s",
            model.get("finalization_after_last_progress_s"),
        ),
        default=0.0,
    )
    return startup + total_units * unit_s + terminal


def _parse_prefixed_json(
    text: str | None, prefix: str
) -> dict[str, Any] | None:
    latest = None
    for line in str(text or "").splitlines():
        if not line.startswith(prefix):
            continue
        try:
            value = json.loads(line[len(prefix) :])
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            latest = value
    return latest


def _external_cpu_bucket(core_equiv: float) -> str:
    value = max(0.0, float(core_equiv))
    if value <= 2.0:
        return "external_c0_2"
    if value <= 16.0:
        return "external_c3_16"
    if value <= 32.0:
        return "external_c17_32"
    if value <= 64.0:
        return "external_c33_64"
    return "external_c65p"


def _finite_positive(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result > 0.0 else None


def _finite_nonnegative(
    value: Any, *, default: float | None = None
) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) and result >= 0.0 else default


def _normalize_profiles(
    profiles: Iterable[int] | None,
) -> tuple[int, ...]:
    if profiles is None:
        raw_values: Iterable[Any] = DEFAULT_PROFILES
    elif isinstance(profiles, str):
        raw_values = (
            value.strip() for value in profiles.split(",") if value.strip()
        )
    else:
        raw_values = profiles
    raw = tuple(
        int(str(value).strip().lower().removeprefix("p"))
        for value in raw_values
    )
    values = sorted(set(raw))
    if not values or any(value <= 0 for value in values):
        raise ValueError("profiles must contain positive colocation counts")
    return tuple(values)


def _ordered_selection(
    values: Iterable[str] | None,
    *,
    allowed: Sequence[str],
    label: str,
) -> tuple[str, ...]:
    if values is None:
        requested = set(allowed)
    elif isinstance(values, str):
        requested = {
            value.strip() for value in values.split(",") if value.strip()
        }
    else:
        requested = {str(value) for value in values}
    unknown = sorted(requested.difference(allowed))
    if unknown:
        raise ValueError(f"unsupported {label}(s): {','.join(unknown)}")
    return tuple(value for value in allowed if value in requested)


def _parse_csv_set(raw: str) -> set[str] | None:
    values = {value.strip() for value in str(raw or "").split(",") if value.strip()}
    return values or None


def _parse_profile_csv(raw: str) -> tuple[int, ...] | None:
    values = [
        int(value.strip().lower().removeprefix("p"))
        for value in str(raw or "").split(",")
        if value.strip()
    ]
    return tuple(values) or None


def _safe_id(value: str) -> str:
    return "".join(
        ch if ch.isalnum() or ch in "_.-" else "_" for ch in str(value)
    )[:180]


def _shell_quote(value: str) -> str:
    import shlex

    return shlex.quote(str(value))


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                dict(payload),
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Manifest or launch controlled native CPU co-location profiles. "
            "Every pN is N independent tasks, never N allocation workers."
        )
    )
    parser.add_argument("--allow-launch", action="store_true")
    parser.add_argument(
        "--run-id", default="native_cpu_colocation_completion_20260727"
    )
    parser.add_argument("--nodes", default=",".join(DEFAULT_NODES))
    parser.add_argument(
        "--profiles",
        "--profile-counts",
        dest="profiles",
        default=",".join(str(value) for value in DEFAULT_PROFILES),
        help="Comma-separated colocation counts; pN means N independent tasks.",
    )
    parser.add_argument("--kinds", default=",".join(DEFAULT_KINDS))
    parser.add_argument(
        "--max-parallel", type=int, default=DEFAULT_MAX_PARALLEL
    )
    parser.add_argument(
        "--tasks-per-node", type=int, default=DEFAULT_TASKS_PER_NODE
    )
    parser.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE)
    parser.add_argument(
        "--init-cache-state",
        choices=("cold", "warm", "unspecified"),
        default="unspecified",
    )
    parser.add_argument(
        "--required-dispatch-regime",
        choices=("", *DISPATCH_CPU_REGIMES),
        default="",
        help=(
            "Optional uniform preflight regime for every selected exact cell; "
            "a mismatch defers before controlled child launch."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ARTIFACT_ROOT
            / "native_cpu_colocation_completion_matrix_20260727.json"
        ),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    if raw_argv == ["--remote-dispatch-state"]:
        return _run_remote_dispatch_state()
    if len(raw_argv) == 2 and raw_argv[0] == "--remote-group-spec-file":
        return _run_remote_group(Path(raw_argv[1]))
    if len(raw_argv) == 2 and raw_argv[0] == "--remote-task-spec-file":
        return _run_remote_task(Path(raw_argv[1]))

    args = build_parser().parse_args(raw_argv)
    selected_nodes = _parse_csv_set(args.nodes)
    selected_profiles = _parse_profile_csv(args.profiles)
    selected_kinds = _parse_csv_set(args.kinds)
    required_dispatch_regimes = None
    if args.required_dispatch_regime:
        selected_cells = colocation_cells(
            run_id=str(args.run_id),
            nodes=selected_nodes,
            profiles=selected_profiles,
            kinds=selected_kinds,
            tasks_per_node=int(args.tasks_per_node),
            seed_base=int(args.seed_base),
        )
        required_dispatch_regimes = {
            _calibration_cell_id_for_cell(cell): str(
                args.required_dispatch_regime
            )
            for cell in selected_cells
        }
    result = build_native_cpu_colocation_completion_matrix(
        allow_launch=bool(args.allow_launch),
        run_id=str(args.run_id),
        nodes=selected_nodes,
        profiles=selected_profiles,
        kinds=selected_kinds,
        max_parallel=int(args.max_parallel),
        tasks_per_node=int(args.tasks_per_node),
        seed_base=int(args.seed_base),
        init_cache_state=str(args.init_cache_state),
        required_dispatch_regimes=required_dispatch_regimes,
    )
    _atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if not args.allow_launch or result["all_completion_models_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
