"""Simultaneous split-conformal lower-service gate for CPU worker profiles."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import fmean
import tempfile
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_TRAINING = (
    ARTIFACT_ROOT
    / "native_cpu_allocation_worker_state_campaign_20260727_r3.json"
)
DEFAULT_EXTENSION = (
    ARTIFACT_ROOT
    / "native_cpu_allocation_worker_state_campaign_20260727_split_extension.json"
)
DEFAULT_P1 = (
    ARTIFACT_ROOT
    / "native_cpu_allocation_worker_empty_p1_natural_20260727_r0_r12.json"
)
DEFAULT_P1_REPAIR = (
    ARTIFACT_ROOT
    / "native_cpu_allocation_worker_empty_p1_repair_20260727_r8.json"
)
DEFAULT_PROFILES = (1, 2, 4, 8, 16, 32, 64, 96, 128, 180, 192)
IDLE_REGIME = "cpu_external_idle"
TOTAL_UNITS = 160.0
LOGICAL_CPUS = 192
IDLE_RUN_WINDOW_MAX_FRACTION = 0.02


def build_native_cpu_allocation_worker_lcb_gate(
    *,
    campaign_paths: Sequence[Path],
    profiles: Sequence[int] = DEFAULT_PROFILES,
    training_replicates: Sequence[int] = (0, 1, 2),
    calibration_replicates: Sequence[int] = tuple(range(3, 12)),
    holdout_replicate: int = 12,
    miscoverage_alpha: float = 0.1,
    max_point_relative_error: float = 0.10,
) -> dict[str, Any]:
    if not 0.0 < float(miscoverage_alpha) < 1.0:
        raise ValueError("miscoverage_alpha must lie in (0, 1)")
    expected_profiles = tuple(sorted({int(value) for value in profiles}))
    training_ids = tuple(sorted({int(value) for value in training_replicates}))
    calibration_ids = tuple(
        sorted({int(value) for value in calibration_replicates})
    )
    holdout_id = int(holdout_replicate)
    if set(training_ids) & set(calibration_ids):
        raise ValueError("training and calibration replicates must be disjoint")
    if holdout_id in set(training_ids) | set(calibration_ids):
        raise ValueError("holdout replicate must be disjoint")

    source_rows: list[dict[str, Any]] = []
    source_artifacts = []
    planned_nodes: dict[tuple[int, int], str] = {}
    node_mismatches = []
    for path in campaign_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        source_artifacts.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "status": payload.get("status"),
            }
        )
        for raw_row in payload.get("rows") or []:
            row = dict(raw_row)
            if row.get("target_regime") != IDLE_REGIME:
                continue
            key = (
                int(row.get("workers") or 0),
                int(row.get("replicate") or 0),
            )
            node = str(row.get("node") or "")
            expected_node = planned_nodes.setdefault(key, node)
            if row.get("measurement_valid") and node != expected_node:
                node_mismatches.append(
                    {
                        "profile": key[0],
                        "replicate": key[1],
                        "expected_node": expected_node,
                        "observed_node": node,
                        "source": str(path),
                    }
                )
                continue
            if row.get("measurement_valid"):
                source_rows.append(row)

    indexed, duplicates = _index_rows(source_rows)
    required_replicates = training_ids + calibration_ids + (holdout_id,)
    missing = [
        {"profile": profile, "replicate": replicate}
        for profile in expected_profiles
        for replicate in required_replicates
        if (profile, replicate) not in indexed
    ]

    training_bases: dict[int, float] = {}
    training_phase_models: dict[int, dict[str, float]] = {}
    phase_ready = True
    idle_state_ready = True
    for profile in expected_profiles:
        values = []
        models = []
        for replicate in training_ids:
            row = indexed.get((profile, replicate))
            if not row:
                continue
            values.append(_wall_s(row))
            models.append(dict(row.get("completion_model") or {}))
            phase_ready = phase_ready and _phase_model_ready(row)
            idle_state_ready = idle_state_ready and _idle_state_ready(row)
        if len(values) == len(training_ids) and all(value > 0.0 for value in values):
            training_bases[profile] = fmean(values)
            training_phase_models[profile] = {
                field: fmean(
                    float(model.get(field) or 0.0)
                    for model in models
                )
                for field in (
                    "startup_overhead_s",
                    "completion_unit_s",
                    "terminal_overhead_s",
                    "checkpoint_observed_s",
                    "save_observed_s",
                )
            }

    wave_scores = []
    calibration_phase_ready = True
    calibration_state_ready = True
    for replicate in calibration_ids:
        profile_scores = []
        for profile in expected_profiles:
            row = indexed.get((profile, replicate))
            base = training_bases.get(profile, 0.0)
            if not row or base <= 0.0:
                continue
            profile_scores.append(_wall_s(row) / base - 1.0)
            calibration_phase_ready = (
                calibration_phase_ready and _phase_model_ready(row)
            )
            calibration_state_ready = (
                calibration_state_ready and _idle_state_ready(row)
            )
        if len(profile_scores) == len(expected_profiles):
            wave_scores.append(max(profile_scores))

    conformal_rank = min(
        len(wave_scores),
        max(
            1,
            int(
                math.ceil(
                    (len(wave_scores) + 1)
                    * (1.0 - float(miscoverage_alpha))
                )
            ),
        ),
    )
    simultaneous_margin = (
        max(0.0, sorted(wave_scores)[conformal_rank - 1])
        if wave_scores
        else math.inf
    )

    rows = []
    for profile in expected_profiles:
        base = training_bases.get(profile, 0.0)
        holdout = indexed.get((profile, holdout_id))
        actual = _wall_s(holdout) if holdout else 0.0
        upper = (
            base * (1.0 + simultaneous_margin)
            if base > 0.0 and math.isfinite(simultaneous_margin)
            else 0.0
        )
        point_error = (
            abs(base - actual) / actual
            if actual > 0.0
            else math.inf
        )
        holdout_phase_ready = bool(
            holdout and _phase_model_ready(holdout)
        )
        holdout_state_ready = bool(
            holdout and _idle_state_ready(holdout)
        )
        lower_valid = bool(
            holdout
            and upper >= actual
            and holdout_phase_ready
            and holdout_state_ready
        )
        phase_model = training_phase_models.get(profile, {})
        rows.append(
            {
                "profile_axis": "allocation_workers",
                "allocation_workers": profile,
                "colocation_count": 1,
                "resource_state": "empty",
                "dispatch_external_cpu_regime": IDLE_REGIME,
                "training_sample_count": len(training_ids),
                "calibration_wave_count": len(wave_scores),
                "holdout_replicate": holdout_id,
                "point_eta_s": base,
                "holdout_actual_completion_s": actual,
                "point_relative_error": point_error,
                "startup_overhead_s": float(
                    phase_model.get("startup_overhead_s") or 0.0
                ),
                "completion_unit_s": float(
                    phase_model.get("completion_unit_s") or 0.0
                ),
                "finalization_overhead_s": float(
                    phase_model.get("terminal_overhead_s") or 0.0
                ),
                "checkpoint_observed_s": float(
                    phase_model.get("checkpoint_observed_s") or 0.0
                ),
                "save_observed_s": float(
                    phase_model.get("save_observed_s") or 0.0
                ),
                "simultaneous_conservative_eta_s": upper,
                "realized_service_units_per_s": (
                    TOTAL_UNITS / actual if actual > 0.0 else 0.0
                ),
                "lower_service_units_per_s": (
                    TOTAL_UNITS / upper if upper > 0.0 else 0.0
                ),
                "lower_service_valid_on_holdout": lower_valid,
                "holdout_phase_model_ready": holdout_phase_ready,
                "holdout_idle_state_ready": holdout_state_ready,
                "lower_service_bound_kind": (
                    "wave_max_simultaneous_split_conformal"
                ),
            }
        )

    point_errors = sorted(
        float(row["point_relative_error"])
        for row in rows
        if math.isfinite(float(row["point_relative_error"]))
    )
    point_max = max(point_errors) if point_errors else math.inf
    all_lower_valid = bool(rows) and all(
        row["lower_service_valid_on_holdout"] for row in rows
    )
    sample_counts_ready = bool(
        not missing
        and not duplicates
        and not node_mismatches
        and len(training_bases) == len(expected_profiles)
        and len(wave_scores) == len(calibration_ids)
    )
    passed = bool(
        sample_counts_ready
        and phase_ready
        and calibration_phase_ready
        and idle_state_ready
        and calibration_state_ready
        and all_lower_valid
        and point_max <= float(max_point_relative_error)
    )
    return {
        "gate": "native_cpu_allocation_worker_lcb_gate",
        "schema_version": 1,
        "status": "PASS" if passed else "FAIL",
        "pass": passed,
        "profiles": list(expected_profiles),
        "training_replicates": list(training_ids),
        "calibration_replicates": list(calibration_ids),
        "holdout_replicate": holdout_id,
        "miscoverage_alpha": float(miscoverage_alpha),
        "nominal_simultaneous_coverage": 1.0 - float(miscoverage_alpha),
        "conformal_rank": conformal_rank,
        "wave_max_scores": wave_scores,
        "simultaneous_ratio_margin": simultaneous_margin,
        "max_point_relative_error_limit": float(max_point_relative_error),
        "point_max_relative_error": point_max,
        "sample_counts_ready": sample_counts_ready,
        "phase_models_ready": bool(
            phase_ready and calibration_phase_ready
        ),
        "idle_state_certificates_ready": bool(
            idle_state_ready and calibration_state_ready
        ),
        "all_lower_service_valid_on_holdout": all_lower_valid,
        "missing_cells": missing,
        "duplicate_cells": duplicates,
        "node_mismatches": node_mismatches,
        "source_artifacts": source_artifacts,
        "rows": rows,
        "claim_boundary": (
            "The gate covers the declared node001-node006 homogeneous CPU class, "
            "the measured empty dispatch regime, and allocation-worker profiles "
            "listed here. It does not relabel light/moderate external-load rows "
            "as empty and does not extrapolate to unmeasured worker counts."
        ),
    }


def _index_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[tuple[int, int], dict[str, Any]], list[dict[str, int]]]:
    indexed: dict[tuple[int, int], dict[str, Any]] = {}
    duplicates = []
    for row in rows:
        key = (
            int(row.get("workers") or 0),
            int(row.get("replicate") or 0),
        )
        if key in indexed:
            duplicates.append(
                {"profile": key[0], "replicate": key[1]}
            )
            continue
        indexed[key] = dict(row)
    return indexed, duplicates


def _wall_s(row: Mapping[str, Any] | None) -> float:
    model = (row or {}).get("completion_model") or {}
    return max(0.0, float(model.get("total_wall_s") or 0.0))


def _phase_model_ready(row: Mapping[str, Any]) -> bool:
    model = row.get("completion_model") or {}
    return bool(
        row.get("measurement_valid")
        and row.get("completion_model_ready")
        and model.get("completion_model_ready")
        and model.get("natural_exit")
        and not model.get("stopped_on_stable")
        and float(model.get("checkpoint_observed_s") or 0.0) > 0.0
        and float(model.get("save_observed_s") or 0.0) > 0.0
        and int(model.get("progress_observation_count") or 0) >= 2
    )


def _idle_state_ready(row: Mapping[str, Any]) -> bool:
    preflight = row.get("dispatch_preflight") or {}
    observed = row.get("observed_resource_state") or {}
    return bool(
        row.get("target_regime") == IDLE_REGIME
        and preflight.get("ready")
        and preflight.get("dispatch_external_cpu_regime") == IDLE_REGIME
        and float(observed.get("external_cpu_core_equiv") or 0.0)
        <= IDLE_RUN_WINDOW_MAX_FRACTION * LOGICAL_CPUS
    )


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _markdown(report: Mapping[str, Any], json_path: Path) -> str:
    lines = [
        "# CPU allocation-worker lower-service certificate",
        "",
        f"- Gate: `{report.get('status')}`",
        f"- Source: `{json_path}`",
        (
            "- Simultaneous split-conformal margin: "
            f"`{float(report.get('simultaneous_ratio_margin') or 0.0):.6f}`"
        ),
        (
            "- Maximum fresh-holdout point error: "
            f"`{float(report.get('point_max_relative_error') or 0.0):.4%}`"
        ),
        "",
        "| workers | point ETA (s) | holdout (s) | conservative ETA (s) | lower service (step/s) | covered |",
        "|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in report.get("rows") or []:
        lines.append(
            "| {allocation_workers} | {point_eta_s:.3f} | "
            "{holdout_actual_completion_s:.3f} | "
            "{simultaneous_conservative_eta_s:.3f} | "
            "{lower_service_units_per_s:.6f} | {covered} |".format(
                **row,
                covered=(
                    "yes"
                    if row.get("lower_service_valid_on_holdout")
                    else "no"
                ),
            )
        )
    lines.extend(["", str(report.get("claim_boundary") or ""), ""])
    return "\n".join(lines)


def _parse_paths(raw: str) -> tuple[Path, ...]:
    return tuple(
        Path(item.strip())
        for item in str(raw or "").split(",")
        if item.strip()
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--campaigns",
        default=(
            f"{DEFAULT_TRAINING},{DEFAULT_EXTENSION},"
            f"{DEFAULT_P1},{DEFAULT_P1_REPAIR}"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ARTIFACT_ROOT
            / "native_cpu_allocation_worker_lcb_gate_20260727.json"
        ),
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=(
            ARTIFACT_ROOT
            / "native_cpu_allocation_worker_lcb_table_20260727.csv"
        ),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=(
            REPO_ROOT
            / "md"
            / "native_cpu_allocation_worker_lcb_20260727.md"
        ),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    report = build_native_cpu_allocation_worker_lcb_gate(
        campaign_paths=_parse_paths(args.campaigns)
    )
    _atomic_write_json(args.output, report)
    _write_csv(args.csv_output, report.get("rows") or [])
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(
        _markdown(report, args.output),
        encoding="utf-8",
    )
    print(args.output)
    return 0 if report.get("pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
