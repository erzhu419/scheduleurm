"""Merge theorem-facing native FreqDuet/SUMO completion LCBs into cache v2.

The native co-location axis is distinct from the allocation-worker axis:
``pN`` means N independent one-worker tasks launched as one controlled action.
Rows remain node- and state-scoped and therefore never update the legacy
``(workload_key, profile)`` fallback index.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from simulation.service_cache import ProfileRecord, ServiceRateCache


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_BASE = (
    ARTIFACT_ROOT
    / "service_cache_v2_live_merged_cpu_all_state_lcb_20260727.json"
)
DEFAULT_GATES = (
    ARTIFACT_ROOT
    / "native_cpu_colocation_state_targeted_p1_extended_lcb_gate_20260727formal_v6_state_targeted_p1_ext9.json",
    ARTIFACT_ROOT
    / "native_cpu_colocation_state_targeted_p2_lcb_gate_20260727formal_v6_state_targeted_p2.json",
    ARTIFACT_ROOT
    / "native_cpu_colocation_state_targeted_p4_merged_lcb_gate_20260727formal_v6_state_targeted_p4.json",
)
DEFAULT_OUTPUT = (
    ARTIFACT_ROOT
    / "service_cache_v2_live_merged_cpu_native_all_state_lcb_20260727.json"
)
DEFAULT_REPORT = (
    ARTIFACT_ROOT
    / "native_cpu_colocation_lcb_cache_merge_20260727.json"
)
EXPECTED_WORKLOADS = {
    "freqduet_cpu_native": "freqduet",
    "sumo_eval_cpu_native": "sumo",
}
EXPECTED_NODES = tuple(f"node{index:03d}" for index in range(1, 7))
EXPECTED_PROFILES = (1, 2, 4)
IDLE_STATES = {
    1: "empty",
    2: "controlled_colocation",
    4: "controlled_colocation",
}
EXTERNAL_STATES = {
    1: "cpu_resident_external",
    2: "controlled_colocation_external",
    4: "controlled_colocation_external",
}
COMMAND_FINGERPRINT = (
    "native_cpu_natural_completion_colocation_"
    "wave_max_split_conformal_v1"
)
ETA_SOURCE = (
    "task_native_durable_progress_natural_completion_"
    "wave_max_split_conformal_lcb"
)


def build_native_cpu_colocation_lcb_cache_merge(
    *,
    base_cache_path: Path,
    gate_paths: Sequence[Path],
) -> dict[str, Any]:
    """Return a validated cache snapshot containing all 36 native CPU rows."""

    cache = (
        ServiceRateCache.load(base_cache_path)
        if base_cache_path.exists()
        else ServiceRateCache()
    )
    rows_with_source: list[tuple[Mapping[str, Any], Path]] = []
    gate_diagnostics = []
    for path in gate_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        _validate_gate(payload, path=path)
        gate_rows = list(payload.get("rows") or ())
        rows_with_source.extend((row, path) for row in gate_rows)
        gate_diagnostics.append(
            {
                "path": str(path),
                "row_count": len(gate_rows),
                "profiles": sorted(
                    {
                        int(row.get("colocation_count") or 0)
                        for row in gate_rows
                    }
                ),
                "simultaneous_margin": (
                    payload.get("simultaneous_wave_max_bound") or {}
                ).get("conformal_ratio_margin"),
            }
        )

    expected_keys = {
        (workload_key, node, profile)
        for workload_key in EXPECTED_WORKLOADS
        for node in EXPECTED_NODES
        for profile in EXPECTED_PROFILES
    }
    observed: dict[tuple[str, str, int], tuple[Mapping[str, Any], Path]] = {}
    duplicates = []
    for row, source in rows_with_source:
        key = (
            str(row.get("workload_key") or ""),
            str(row.get("node") or ""),
            int(row.get("colocation_count") or 0),
        )
        if key in observed:
            duplicates.append(key)
        observed[key] = (row, source)
    missing = sorted(expected_keys - set(observed))
    unexpected = sorted(set(observed) - expected_keys)
    if duplicates or missing or unexpected:
        raise ValueError(
            "native CPU gate family is not an exact 2x6x3 matrix: "
            f"duplicates={duplicates}, missing={missing}, "
            f"unexpected={unexpected}"
        )

    inserted = []
    for key in sorted(expected_keys):
        row, source = observed[key]
        record = _profile_record(row, source=source)
        cache.add(record, force_replace=True)
        inserted.append(record.snapshot())

        if cache.get(record.workload_key, record.profile) is not None:
            raise ValueError(
                "state-scoped native CPU row leaked into legacy profile index: "
                f"{key}"
            )
        exact = cache.lookup_statewise_exact(
            record.workload_key,
            workload_env=record.workload_env,
            node_bucket=record.node_bucket,
            resource_state=record.resource_state,
            allocation_workers=record.allocation_workers,
            colocation_count=record.colocation_count,
            resident_mix=record.resident_mix,
        )
        if not exact.is_exact or exact.record != record:
            raise ValueError(f"exact statewise round-trip failed for {key}")

    return {
        "gate": "native_cpu_colocation_lcb_cache_merge",
        "schema_version": 1,
        "status": "PASS",
        "pass": True,
        "base_cache_path": str(base_cache_path),
        "gate_paths": [str(path) for path in gate_paths],
        "gate_diagnostics": gate_diagnostics,
        "expected_row_count": len(expected_keys),
        "inserted_count": len(inserted),
        "workload_keys": sorted(EXPECTED_WORKLOADS),
        "nodes": list(EXPECTED_NODES),
        "profiles": list(EXPECTED_PROFILES),
        "profile_axis": "colocation_count",
        "allocation_workers": 1,
        "legacy_index_unchanged_for_native_rows": True,
        "inserted_rows": inserted,
        "service_cache_snapshot": cache.snapshot(),
        "claim_boundary": (
            "The merge closes only the measured native FreqDuet and SUMO "
            "2-workload x 6-node x {1,2,4}-colocation family. Idle and organic "
            "external-load states remain separate exact tuples. These rows do "
            "not substitute for legacy ablation/surrogate workload keys, other "
            "hardware classes, higher co-location profiles, migration, or "
            "future workloads."
        ),
    }


def _validate_gate(payload: Mapping[str, Any], *, path: Path) -> None:
    if not payload.get("pass"):
        raise ValueError(f"native CPU lower-service gate did not pass: {path}")
    simultaneous = payload.get("simultaneous_wave_max_bound") or {}
    if not simultaneous.get("ready"):
        raise ValueError(
            f"native CPU gate lacks a simultaneous wave-level bound: {path}"
        )
    if int(payload.get("row_count") or 0) != 12:
        raise ValueError(f"native CPU gate must contain exactly 12 rows: {path}")
    if not payload.get("all_rows_ready"):
        raise ValueError(f"native CPU gate contains an unready row: {path}")
    if not payload.get("all_lower_service_valid_on_holdout"):
        raise ValueError(
            f"native CPU gate contains an uncovered lower-service row: {path}"
        )


def _profile_record(
    row: Mapping[str, Any],
    *,
    source: Path,
) -> ProfileRecord:
    workload_key = str(row.get("workload_key") or "")
    workload_env = str(row.get("workload_env") or "")
    node = str(row.get("node") or "")
    workers = int(row.get("allocation_workers") or 0)
    colocation = int(row.get("colocation_count") or 0)
    observed_state = str(
        row.get("observed_effective_resource_state") or ""
    )
    run_state = str(row.get("run_window_effective_resource_state") or "")
    dispatch_regime = str(row.get("dispatch_external_cpu_regime") or "")
    total_units = float(row.get("aggregate_service_units") or 0.0)
    lower_rate = float(
        row.get("simultaneous_lower_service_units_per_s")
        or row.get("lower_service_units_per_s")
        or 0.0
    )
    point_eta = float(row.get("point_predicted_makespan_s") or 0.0)
    upper_eta = float(
        row.get("simultaneous_conservative_upper_makespan_s") or 0.0
    )
    phase = row.get("operational_point_group_phase_model") or {}
    startup = float(phase.get("startup_overhead_s") or 0.0)
    completion_unit = float(
        phase.get("aggregate_completion_unit_s") or 0.0
    )
    terminal = float(phase.get("terminal_overhead_s") or 0.0)

    if EXPECTED_WORKLOADS.get(workload_key) != workload_env:
        raise ValueError(
            f"native CPU workload/env mismatch: {workload_key}/{workload_env}"
        )
    if node not in EXPECTED_NODES:
        raise ValueError(f"unexpected native CPU node: {node!r}")
    if workers != 1 or colocation not in EXPECTED_PROFILES:
        raise ValueError(
            "native CPU row has the wrong profile dimensions: "
            f"workers={workers}, colocation={colocation}"
        )
    if observed_state != run_state:
        raise ValueError(
            f"dispatch/run-window state mismatch for {workload_key}/{node}/"
            f"p{colocation}: {observed_state!r} != {run_state!r}"
        )
    external = observed_state.endswith("_external")
    expected_state = (
        EXTERNAL_STATES[colocation] if external else IDLE_STATES[colocation]
    )
    if observed_state != expected_state:
        raise ValueError(
            f"unexpected effective state for {workload_key}/{node}/"
            f"p{colocation}: {observed_state!r}"
        )
    dispatch_is_external = bool(
        dispatch_regime and dispatch_regime != "cpu_external_idle"
    )
    if dispatch_is_external != external:
        raise ValueError(
            f"dispatch regime/state mismatch for {workload_key}/{node}/"
            f"p{colocation}: {dispatch_regime!r}/{observed_state!r}"
        )
    if not row.get("simultaneous_lower_service_valid_on_holdout"):
        raise ValueError(
            f"simultaneous lower service is not holdout-valid for "
            f"{workload_key}/{node}/p{colocation}"
        )
    expected_units = 6.0 * float(colocation)
    if not math.isclose(total_units, expected_units, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"aggregate service units differ from 6*p for "
            f"{workload_key}/{node}/p{colocation}"
        )
    phase_eta = startup + total_units * completion_unit + terminal
    if (
        lower_rate <= 0.0
        or point_eta <= 0.0
        or upper_eta < point_eta
        or completion_unit <= 0.0
        or not math.isclose(
            phase_eta,
            point_eta,
            rel_tol=1e-9,
            abs_tol=1e-6,
        )
    ):
        raise ValueError(
            f"incomplete phase/LCB model for {workload_key}/{node}/"
            f"p{colocation}"
        )

    per_task_rate = lower_rate / float(colocation)
    return ProfileRecord(
        workload_key=workload_key,
        command_fingerprint=COMMAND_FINGERPRINT,
        resource_kind="cpu",
        node_bucket=f"{node}:cpu_hpc_192c",
        profile=colocation,
        allocation_workers=workers,
        colocation_count=colocation,
        unit="episode",
        total_units=total_units,
        aggregate_rate=lower_rate,
        per_task_rates=tuple(per_task_rate for _ in range(colocation)),
        source=str(source),
        workload_env=workload_env,
        resource_state=observed_state,
        resident_mix=(
            "organic_external_measured" if external else ""
        ),
        eta_source=ETA_SOURCE,
        stable_rate_ready=True,
        hardware_class="cpu_hpc_192c",
        completion_model_ready=True,
        completion_model_sample_count=(
            int(row.get("training_sample_count") or 0)
            + int(row.get("calibration_sample_count") or 0)
            + 1
        ),
        startup_overhead_s=startup,
        completion_unit_s=completion_unit,
        finalization_overhead_s=terminal,
        checkpoint_observed_s=0.0,
        save_observed_s=terminal,
        completion_total_wall_s=point_eta,
        completion_group_total_units=total_units,
        completion_model_relative_error=float(
            row.get("point_relative_error") or 0.0
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-cache", type=Path, default=DEFAULT_BASE)
    parser.add_argument(
        "--gates",
        default=",".join(str(path) for path in DEFAULT_GATES),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    gate_paths = tuple(
        Path(item.strip())
        for item in str(args.gates).split(",")
        if item.strip()
    )
    report = build_native_cpu_colocation_lcb_cache_merge(
        base_cache_path=args.base_cache,
        gate_paths=gate_paths,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            report["service_cache_snapshot"],
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    args.report_output.write_text(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key != "service_cache_snapshot"
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
