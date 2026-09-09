"""Merge the pre-registered q00/q10 phase-aware CPU action family."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from simulation.service_cache import ProfileRecord, ServiceRateCache


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_BASE = (
    ARTIFACT_ROOT
    / "service_cache_v2_live_merged_cpu_native_all_state_lcb_20260727.json"
)
DEFAULT_CPU_GATE = ARTIFACT_ROOT / "critical_cpu_colocation_lcb_gate_20260802.json"
DEFAULT_LIGHT_GATE = ARTIFACT_ROOT / "critical_light_colocation_lcb_gate_20260802.json"
DEFAULT_CPU_EXPECTED_GATE = (
    ARTIFACT_ROOT / "critical_cpu_expected_completion_gate_20260802.json"
)
DEFAULT_OUTPUT = ARTIFACT_ROOT / "service_cache_v2_critical_phase_20260802.json"
DEFAULT_REPORT = ARTIFACT_ROOT / "critical_cpu_phase_cache_merge_20260802.json"
ETA_SOURCE = (
    "task_native_durable_progress_csv_natural_completion_actual_io_"
    "wave_max_split_conformal_lcb"
)
COMMAND_FINGERPRINT = "critical_cpu_phase_actual_io_v1"
CONTRACTS = {
    "cpu_heavy_local_bench": {
        "env": "cpu",
        "nodes": ("node001", "node002"),
        "profiles": (1, 8, 9),
        "units_per_task": 120.0,
    },
    "light_control_local": {
        "env": "light_control",
        "nodes": ("node003", "node005"),
        "profiles": (1, 13),
        "units_per_task": 300.0,
    },
}


def build_critical_cpu_phase_cache_merge(
    *,
    base_cache_path: Path,
    cpu_gate_path: Path,
    light_gate_path: Path,
    cpu_expected_gate_path: Path,
) -> dict[str, Any]:
    cache = ServiceRateCache.load(base_cache_path)
    source_rows: list[tuple[Mapping[str, Any], Path]] = []
    for path in (cpu_gate_path, light_gate_path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        _validate_lower_service_gate(payload, path=path)
        source_rows.extend((row, path) for row in payload.get("rows") or ())
    expected_payload = json.loads(
        cpu_expected_gate_path.read_text(encoding="utf-8")
    )
    _validate_expected_gate(expected_payload, path=cpu_expected_gate_path)
    expected_point_rows = {
        (
            str(row.get("workload_key") or ""),
            str(row.get("node") or ""),
            int(row.get("colocation_count") or 0),
        ): row
        for row in expected_payload.get("rows") or ()
    }

    expected = {
        (workload, node, profile)
        for workload, contract in CONTRACTS.items()
        for node in contract["nodes"]
        for profile in contract["profiles"]
    }
    observed: dict[tuple[str, str, int], tuple[Mapping[str, Any], Path]] = {}
    duplicates = []
    for row, source in source_rows:
        key = (
            str(row.get("workload_key") or ""),
            str(row.get("node") or ""),
            int(row.get("colocation_count") or 0),
        )
        if key in observed:
            duplicates.append(key)
        observed[key] = (row, source)
    missing = sorted(expected - set(observed))
    unexpected = sorted(set(observed) - expected)
    if duplicates or missing or unexpected:
        raise ValueError(
            "critical q00/q10 gate family is not exact: "
            f"duplicates={duplicates}, missing={missing}, unexpected={unexpected}"
        )

    inserted = []
    for key in sorted(expected):
        row, source = observed[key]
        point_row = expected_point_rows.get(key) if key[0] == "cpu_heavy_local_bench" else None
        if key[0] == "cpu_heavy_local_bench" and point_row is None:
            raise ValueError(f"missing expected-completion row for {key}")
        record = _profile_record(
            row,
            source=source,
            point_row=point_row,
            point_source=cpu_expected_gate_path if point_row is not None else None,
        )
        legacy_before = cache.get(record.workload_key, record.profile)
        cache.add(record, force_replace=True)
        if cache.get(record.workload_key, record.profile) != legacy_before:
            raise ValueError(f"statewise row changed legacy index for {key}")
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
        inserted.append(record.snapshot())

    return {
        "gate": "critical_cpu_phase_cache_merge",
        "schema_version": 1,
        "status": "PASS",
        "pass": True,
        "base_cache_path": str(base_cache_path),
        "gate_paths": [
            str(cpu_gate_path),
            str(light_gate_path),
            str(cpu_expected_gate_path),
        ],
        "expected_row_count": len(expected),
        "inserted_count": len(inserted),
        "profile_axis": "colocation_count",
        "allocation_workers": 1,
        "legacy_index_unchanged": True,
        "inserted_rows": inserted,
        "service_cache_snapshot": cache.snapshot(),
        "claim_boundary": (
            "This merge admits only the pre-registered q00/q10 controlled CPU "
            "actions with actual checkpoint I/O, natural completion, exact node "
            "and run-window state, and simultaneous split-conformal lower service."
        ),
    }


def _validate_lower_service_gate(payload: Mapping[str, Any], *, path: Path) -> None:
    if not payload.get("lower_service_pass", payload.get("pass")):
        raise ValueError(f"lower-service gate did not pass: {path}")
    simultaneous = payload.get("simultaneous_wave_max_bound") or {}
    if not simultaneous.get("ready"):
        raise ValueError(f"gate lacks simultaneous wave bound: {path}")
    if not payload.get("all_rows_ready"):
        raise ValueError(f"gate contains unready rows: {path}")
    if not payload.get("all_lower_service_valid_on_holdout"):
        raise ValueError(f"gate contains uncovered lower service: {path}")


def _validate_expected_gate(payload: Mapping[str, Any], *, path: Path) -> None:
    if not payload.get("pass"):
        raise ValueError(f"expected-completion gate did not pass: {path}")
    if payload.get("estimand") != "expected_group_natural_completion_makespan":
        raise ValueError(f"unexpected completion estimand: {path}")
    if not payload.get("all_rows_ready"):
        raise ValueError(f"expected-completion gate has unready rows: {path}")


def _profile_record(
    row: Mapping[str, Any],
    *,
    source: Path,
    point_row: Mapping[str, Any] | None,
    point_source: Path | None,
) -> ProfileRecord:
    workload = str(row.get("workload_key") or "")
    contract = CONTRACTS.get(workload)
    if contract is None:
        raise ValueError(f"unregistered critical workload {workload!r}")
    env = str(row.get("workload_env") or "")
    node = str(row.get("node") or "")
    profile = int(row.get("colocation_count") or 0)
    workers = int(row.get("allocation_workers") or 0)
    observed_state = str(row.get("observed_effective_resource_state") or "")
    run_state = str(row.get("run_window_effective_resource_state") or "")
    dispatch_bucket = str(row.get("dispatch_external_cpu_bucket") or "")
    run_bucket = str(row.get("run_window_external_cpu_bucket") or "")
    total_units = float(row.get("aggregate_service_units") or 0.0)
    lower_rate = float(
        row.get("simultaneous_lower_service_units_per_s")
        or row.get("lower_service_units_per_s")
        or 0.0
    )
    upper_eta = float(
        row.get("simultaneous_conservative_upper_makespan_s") or 0.0
    )
    if point_row is None:
        phase = row.get("operational_point_group_phase_model") or {}
        point_eta = float(row.get("point_predicted_makespan_s") or 0.0)
        point_error = float(row.get("point_relative_error") or 0.0)
        point_sample_count = (
            int(row.get("training_sample_count") or 0)
            + int(row.get("calibration_sample_count") or 0)
        )
    else:
        phase = point_row.get("frozen_expected_phase_model") or {}
        point_eta = float(
            point_row.get("predicted_expected_makespan_s") or 0.0
        )
        point_error = float(
            point_row.get("expected_makespan_relative_error") or 0.0
        )
        point_sample_count = int(point_row.get("fit_sample_count") or 0)
    startup = float(phase.get("startup_overhead_s") or 0.0)
    unit_s = float(phase.get("aggregate_completion_unit_s") or 0.0)
    terminal = float(phase.get("terminal_overhead_s") or 0.0)

    if env != contract["env"] or node not in contract["nodes"]:
        raise ValueError(f"workload fabric mismatch for {workload}/{node}/{env}")
    if workers != 1 or profile not in contract["profiles"]:
        raise ValueError(f"profile-axis mismatch for {workload}/{node}/p{profile}")
    expected_state = "empty" if profile == 1 else "controlled_colocation"
    if observed_state != expected_state or run_state != expected_state:
        raise ValueError(f"dispatch/run state mismatch for {workload}/{node}/p{profile}")
    if dispatch_bucket != "external_c0_2" or run_bucket != "external_c0_2":
        raise ValueError(f"external load entered {workload}/{node}/p{profile}")
    if point_row is not None:
        point_identity = (
            str(point_row.get("workload_key") or ""),
            str(point_row.get("workload_env") or ""),
            str(point_row.get("node") or ""),
            int(point_row.get("allocation_workers") or 0),
            int(point_row.get("colocation_count") or 0),
            str(point_row.get("observed_effective_resource_state") or ""),
            str(point_row.get("run_window_effective_resource_state") or ""),
            str(point_row.get("dispatch_external_cpu_bucket") or ""),
            str(point_row.get("run_window_external_cpu_bucket") or ""),
        )
        lower_identity = (
            workload,
            env,
            node,
            workers,
            profile,
            observed_state,
            run_state,
            dispatch_bucket,
            run_bucket,
        )
        if point_identity != lower_identity:
            raise ValueError(f"point/lower identity mismatch for {workload}/{node}/p{profile}")
    expected_units = float(contract["units_per_task"]) * profile
    if not math.isclose(total_units, expected_units, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"service-unit mismatch for {workload}/{node}/p{profile}")
    phase_eta = startup + total_units * unit_s + terminal
    if (
        lower_rate <= 0.0
        or point_eta <= 0.0
        or upper_eta < point_eta
        or unit_s <= 0.0
        or not math.isclose(phase_eta, point_eta, rel_tol=1e-9, abs_tol=1e-6)
        or not row.get("simultaneous_lower_service_valid_on_holdout")
    ):
        raise ValueError(f"incomplete phase/LCB row for {workload}/{node}/p{profile}")

    return ProfileRecord(
        workload_key=workload,
        command_fingerprint=COMMAND_FINGERPRINT,
        resource_kind="cpu",
        node_bucket=f"{node}:cpu_hpc_192c",
        profile=profile,
        allocation_workers=1,
        colocation_count=profile,
        unit="step",
        total_units=total_units,
        aggregate_rate=lower_rate,
        per_task_rates=tuple(lower_rate / profile for _ in range(profile)),
        source=(
            f"{source}|expected_completion={point_source}"
            if point_source is not None
            else str(source)
        ),
        workload_env=env,
        resource_state=expected_state,
        resident_mix="",
        eta_source=ETA_SOURCE,
        stable_rate_ready=True,
        hardware_class="cpu_hpc_192c",
        completion_model_ready=True,
        completion_model_sample_count=point_sample_count,
        startup_overhead_s=startup,
        completion_unit_s=unit_s,
        finalization_overhead_s=terminal,
        completion_total_wall_s=point_eta,
        completion_group_total_units=total_units,
        completion_model_relative_error=point_error,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-cache", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--cpu-gate", type=Path, default=DEFAULT_CPU_GATE)
    parser.add_argument("--light-gate", type=Path, default=DEFAULT_LIGHT_GATE)
    parser.add_argument(
        "--cpu-expected-gate",
        type=Path,
        default=DEFAULT_CPU_EXPECTED_GATE,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    report = build_critical_cpu_phase_cache_merge(
        base_cache_path=args.base_cache,
        cpu_gate_path=args.cpu_gate,
        light_gate_path=args.light_gate,
        cpu_expected_gate_path=args.cpu_expected_gate,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report["service_cache_snapshot"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    compact = {key: value for key, value in report.items() if key != "service_cache_snapshot"}
    args.report_output.write_text(
        json.dumps(compact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(compact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
