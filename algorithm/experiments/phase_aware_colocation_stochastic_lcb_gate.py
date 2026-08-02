"""Split-conformal completion and lower-service gate for native co-location groups."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from statistics import median
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from .file_progress_completion_wrapper import (
    DISPATCH_CPU_SAMPLE_COUNT,
    MEASUREMENT_PROTOCOL,
)


def build_colocation_stochastic_lcb_gate(
    *,
    training_paths: Sequence[Path],
    calibration_paths: Sequence[Path],
    holdout_path: Path,
    miscoverage_alpha: float = 0.10,
    p95_relative_error_limit: float = 0.10,
    max_relative_error_limit: float = 0.15,
    min_interval_sample_count: int = 5,
    min_training_sample_count: int = 3,
    expected_cell_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Fit group point/bound models and score one untouched holdout."""

    alpha = float(miscoverage_alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("miscoverage_alpha must lie strictly between zero and one")
    if not training_paths or not calibration_paths:
        raise ValueError("training and calibration paths are required")

    training = [_load(path) for path in training_paths]
    calibration = [_load(path) for path in calibration_paths]
    holdout = _load(holdout_path)
    holdout_rows = _rows_by_cell(holdout)
    expected_cells = set(
        str(value).strip()
        for value in (expected_cell_ids or holdout_rows.keys())
        if str(value).strip()
    )
    artifact_sets = [
        *[
            (f"training:{index}", set(_rows_by_cell(payload)))
            for index, payload in enumerate(training, start=1)
        ],
        *[
            (f"calibration:{index}", set(_rows_by_cell(payload)))
            for index, payload in enumerate(calibration, start=1)
        ],
        ("holdout:1", set(holdout_rows)),
    ]
    artifact_cell_set_diagnostics = [
        {
            "artifact": label,
            "missing_cell_ids": sorted(expected_cells - observed),
            "unexpected_cell_ids": sorted(observed - expected_cells),
            "exact_match": observed == expected_cells,
        }
        for label, observed in artifact_sets
    ]
    artifact_cell_sets_ready = bool(expected_cells) and all(
        row["exact_match"] for row in artifact_cell_set_diagnostics
    )
    result_rows: list[dict[str, Any]] = []

    for cell_id in sorted(expected_cells):
        holdout_row = holdout_rows.get(cell_id)
        if holdout_row is None:
            result_rows.append(
                {
                    "calibration_cell_id": cell_id,
                    "ready": False,
                    "missing_holdout_cell": True,
                    "lower_service_valid_on_holdout": False,
                }
            )
            continue
        raw_train_rows = _matching_rows(training, cell_id)
        raw_calibration_rows = _matching_rows(calibration, cell_id)
        target_signature = _state_signature(holdout_row)
        train_rows = [
            row for row in raw_train_rows
            if _state_signature(row) == target_signature
        ]
        calibration_rows = [
            row for row in raw_calibration_rows
            if _state_signature(row) == target_signature
        ]
        all_rows = [*train_rows, *calibration_rows, holdout_row]
        minimum_training_count = max(1, int(min_training_sample_count))
        minimum_sample_counts_ready = bool(
            len(train_rows) >= minimum_training_count
            and calibration_rows
        )
        all_ready = bool(
            all(
                _row_ready(
                    row,
                    min_interval_sample_count=min_interval_sample_count,
                )
                for row in all_rows
            )
        )
        signatures = [_state_signature(row) for row in all_rows]
        statewise_match = bool(
            train_rows
            and calibration_rows
            and signatures
            and len(set(signatures)) == 1
        )
        total_units = _aggregate_units(holdout_row)
        frozen_conformal_phase_model = _fit_group_phase_medians(train_rows)
        conformal_base_prediction = _predict_group_total(
            frozen_conformal_phase_model,
            total_units,
        )
        raw_scores = [
            _one_sided_ratio_score(
                actual=_actual_total(row),
                predicted=_predict_group_total(
                    frozen_conformal_phase_model,
                    _aggregate_units(row),
                ),
            )
            for row in calibration_rows
            if _actual_total(row) > 0.0
        ]
        scores = [score for score in raw_scores if math.isfinite(score)]
        nonfinite_score_count = len(raw_scores) - len(scores)
        conformal_rank = int(math.ceil((len(scores) + 1) * (1.0 - alpha)))
        finite_sample_ready = bool(
            scores
            and nonfinite_score_count == 0
            and 1 <= conformal_rank <= len(scores)
        )
        margin = (
            max(0.0, sorted(scores)[conformal_rank - 1])
            if finite_sample_ready
            else math.inf
        )
        upper_total = (
            conformal_base_prediction * (1.0 + margin)
            if math.isfinite(margin)
            else math.inf
        )
        operational_point_rows = [*train_rows, *calibration_rows]
        operational_point_phase_model = _fit_group_phase_medians(
            operational_point_rows
        )
        operational_point_prediction = _predict_group_total(
            operational_point_phase_model,
            total_units,
        )
        actual_total = _actual_total(holdout_row)
        point_error = (
            abs(operational_point_prediction - actual_total) / actual_total
            if operational_point_prediction > 0.0 and actual_total > 0.0
            else math.inf
        )
        covered = bool(
            math.isfinite(upper_total)
            and upper_total > 0.0
            and actual_total > 0.0
            and actual_total <= upper_total * (1.0 + 1e-12)
        )
        lower_service = (
            total_units / upper_total
            if total_units > 0.0 and math.isfinite(upper_total)
            else 0.0
        )
        realized_service = (
            total_units / actual_total
            if total_units > 0.0 and actual_total > 0.0
            else 0.0
        )
        row_ready = bool(
            minimum_sample_counts_ready
            and all_ready
            and statewise_match
            and finite_sample_ready
            and conformal_base_prediction > 0.0
            and operational_point_prediction > 0.0
            and actual_total > 0.0
        )
        result_rows.append(
            {
                "calibration_cell_id": cell_id,
                "node": holdout_row.get("node"),
                "workload_key": holdout_row.get("workload_key"),
                "workload_env": holdout_row.get("workload_env"),
                "allocation_workers": int(
                    holdout_row.get("allocation_workers") or 1
                ),
                "colocation_count": int(
                    holdout_row.get("colocation_count") or 1
                ),
                "profile_axis": holdout_row.get("profile_axis"),
                "observed_effective_resource_state": holdout_row.get(
                    "observed_effective_resource_state",
                    holdout_row.get("resource_state"),
                ),
                "dispatch_state_ready": bool(
                    holdout_row.get("dispatch_state_ready")
                ),
                "measurement_protocol": holdout_row.get(
                    "measurement_protocol"
                ),
                "dispatch_sample_count": holdout_row.get(
                    "dispatch_sample_count"
                ),
                "dispatch_external_cpu_regime": holdout_row.get(
                    "dispatch_external_cpu_regime"
                ),
                "dispatch_external_cpu_fraction": holdout_row.get(
                    "dispatch_external_cpu_fraction"
                ),
                "dispatch_external_cpu_fraction_p95": holdout_row.get(
                    "dispatch_external_cpu_fraction_p95"
                ),
                "dispatch_external_cpu_core_equiv": holdout_row.get(
                    "dispatch_external_cpu_core_equiv"
                ),
                "dispatch_external_cpu_core_equiv_p95": holdout_row.get(
                    "dispatch_external_cpu_core_equiv_p95"
                ),
                "dispatch_external_cpu_bucket": holdout_row.get(
                    "dispatch_external_cpu_bucket"
                ),
                "run_window_effective_resource_state": holdout_row.get(
                    "run_window_effective_resource_state"
                ),
                "run_window_external_cpu_core_equiv": holdout_row.get(
                    "run_window_external_cpu_core_equiv",
                    holdout_row.get("external_cpu_core_equiv"),
                ),
                "run_window_external_cpu_bucket": holdout_row.get(
                    "run_window_external_cpu_bucket",
                    holdout_row.get("external_cpu_bucket"),
                ),
                "init_cache_state": holdout_row.get("init_cache_state"),
                "aggregate_service_units": total_units,
                "training_sample_count": len(train_rows),
                "raw_training_sample_count": len(raw_train_rows),
                "minimum_training_sample_count": minimum_training_count,
                "calibration_sample_count": len(calibration_rows),
                "raw_calibration_sample_count": len(raw_calibration_rows),
                "excluded_training_state_count": (
                    len(raw_train_rows) - len(train_rows)
                ),
                "excluded_calibration_state_count": (
                    len(raw_calibration_rows) - len(calibration_rows)
                ),
                "minimum_sample_counts_ready": minimum_sample_counts_ready,
                "all_task_phase_models_ready": all_ready,
                "min_interval_sample_count": int(min_interval_sample_count),
                "observed_min_interval_sample_counts": [
                    _minimum_interval_count(row) for row in all_rows
                ],
                "statewise_match": statewise_match,
                "state_signature": list(signatures[-1]) if signatures else None,
                "frozen_group_phase_model": frozen_conformal_phase_model,
                "frozen_conformal_group_phase_model": (
                    frozen_conformal_phase_model
                ),
                "conformal_base_predicted_makespan_s": (
                    conformal_base_prediction
                ),
                "operational_point_sample_count": len(
                    operational_point_rows
                ),
                "operational_point_group_phase_model": (
                    operational_point_phase_model
                ),
                "operational_point_predicted_makespan_s": (
                    operational_point_prediction
                ),
                "point_predicted_makespan_s": (
                    operational_point_prediction
                ),
                "actual_makespan_s": actual_total,
                "point_relative_error": (
                    point_error if math.isfinite(point_error) else None
                ),
                "miscoverage_alpha": alpha,
                "conformal_rank": conformal_rank,
                "calibration_ratio_scores": scores,
                "nonfinite_calibration_score_count": nonfinite_score_count,
                "conformal_ratio_margin": (
                    margin if math.isfinite(margin) else None
                ),
                "finite_sample_conformal_ready": finite_sample_ready,
                "conservative_upper_makespan_s": (
                    upper_total if math.isfinite(upper_total) else None
                ),
                "conservative_holdout_covered": covered,
                "lower_service_units_per_s": lower_service,
                "realized_service_units_per_s": realized_service,
                "lower_service_valid_on_holdout": bool(
                    row_ready
                    and covered
                    and lower_service <= realized_service * (1.0 + 1e-12)
                ),
                "ready": row_ready,
            }
        )

    simultaneous = _build_wave_max_simultaneous_bound(
        calibration_payloads=calibration,
        holdout_rows=holdout_rows,
        result_rows=result_rows,
        expected_cells=expected_cells,
        miscoverage_alpha=alpha,
        min_interval_sample_count=min_interval_sample_count,
    )
    if simultaneous["ready"]:
        simultaneous_margin = float(simultaneous["conformal_ratio_margin"])
        for row in result_rows:
            base_prediction = max(
                0.0,
                float(row.get("conformal_base_predicted_makespan_s") or 0.0),
            )
            actual_total = max(
                0.0,
                float(row.get("actual_makespan_s") or 0.0),
            )
            total_units = max(
                0.0,
                float(row.get("aggregate_service_units") or 0.0),
            )
            simultaneous_upper = base_prediction * (
                1.0 + simultaneous_margin
            )
            simultaneous_covered = bool(
                row.get("ready")
                and simultaneous_upper > 0.0
                and actual_total > 0.0
                and actual_total <= simultaneous_upper * (1.0 + 1e-12)
            )
            simultaneous_lower = (
                total_units / simultaneous_upper
                if total_units > 0.0 and simultaneous_upper > 0.0
                else 0.0
            )
            row["cellwise_conservative_upper_makespan_s"] = row.get(
                "conservative_upper_makespan_s"
            )
            row["cellwise_lower_service_units_per_s"] = row.get(
                "lower_service_units_per_s"
            )
            row["cellwise_lower_service_valid_on_holdout"] = row.get(
                "lower_service_valid_on_holdout"
            )
            row["simultaneous_conservative_upper_makespan_s"] = (
                simultaneous_upper
            )
            row["simultaneous_lower_service_units_per_s"] = (
                simultaneous_lower
            )
            row["simultaneous_lower_service_valid_on_holdout"] = (
                simultaneous_covered
                and simultaneous_lower
                <= float(row.get("realized_service_units_per_s") or 0.0)
                * (1.0 + 1e-12)
            )
            row["conservative_upper_makespan_s"] = simultaneous_upper
            row["lower_service_units_per_s"] = simultaneous_lower
            row["lower_service_valid_on_holdout"] = row[
                "simultaneous_lower_service_valid_on_holdout"
            ]
            row["lower_service_bound_kind"] = (
                "wave_max_simultaneous_split_conformal"
            )

    point_errors = sorted(
        float(row["point_relative_error"])
        for row in result_rows
        if row.get("ready") and row.get("point_relative_error") is not None
    )
    p95 = _nearest_rank(point_errors, 0.95)
    maximum = max(point_errors, default=math.inf)
    all_rows_ready = bool(result_rows) and len(point_errors) == len(result_rows)
    all_lower_valid = bool(result_rows) and all(
        bool(row.get("lower_service_valid_on_holdout"))
        for row in result_rows
    )
    lower_service_pass = bool(
        artifact_cell_sets_ready and all_rows_ready and all_lower_valid
    )
    point_eta_pass = bool(
        all_rows_ready
        and p95 <= float(p95_relative_error_limit)
        and maximum <= float(max_relative_error_limit)
    )
    passed = lower_service_pass and point_eta_pass
    return {
        "gate": "phase_aware_colocation_stochastic_lcb_gate",
        "schema_version": 1,
        "training_paths": [str(path) for path in training_paths],
        "calibration_paths": [str(path) for path in calibration_paths],
        "holdout_path": str(holdout_path),
        "miscoverage_alpha": alpha,
        "nominal_cellwise_coverage": 1.0 - alpha,
        "expected_cell_ids": sorted(expected_cells),
        "expected_cell_count": len(expected_cells),
        "artifact_cell_sets_ready": artifact_cell_sets_ready,
        "artifact_cell_set_diagnostics": artifact_cell_set_diagnostics,
        "row_count": len(result_rows),
        "ready_row_count": len(point_errors),
        "all_rows_ready": all_rows_ready,
        "point_p95_relative_error": p95 if math.isfinite(p95) else None,
        "point_max_relative_error": maximum if math.isfinite(maximum) else None,
        "p95_relative_error_limit": float(p95_relative_error_limit),
        "max_relative_error_limit": float(max_relative_error_limit),
        "min_training_sample_count": max(1, int(min_training_sample_count)),
        "all_lower_service_valid_on_holdout": all_lower_valid,
        "lower_service_pass": lower_service_pass,
        "point_eta_pass": point_eta_pass,
        "simultaneous_wave_max_bound": simultaneous,
        "pass": passed,
        "status": "PASS" if passed else "FAIL",
        "rows": result_rows,
        "claim_boundary": (
            "The bound has finite-sample split-conformal cellwise coverage for "
            "the exact native workload, node, observed hardware-normalized "
            "multi-window external-load regime, "
            "allocation, co-location count, total work, and cache-state cell. "
            "The regime is sampled before the controlled group starts, and "
            "the categorical run-window state and external-load bucket are "
            "also part of the exact calibration signature. "
            "Rows from other observed states are excluded before fitting and "
            "calibration. The conformal base model uses training groups only, "
            "and its calibration residuals remain frozen. The operational "
            "point ETA separately uses all pre-holdout training and calibration "
            "groups; it has no conformal coverage claim and cannot alter the "
            "lower-service bound. "
            "When every calibration artifact contains the same exact cell "
            "family, a second split-conformal construction calibrates the "
            "maximum normalized residual across each complete calibration "
            "wave. Under exchangeability of these pre-registered vector waves, "
            "the reported wave-max lower services have simultaneous family "
            "coverage at the same nominal level; the scalar cellwise values "
            "remain recorded separately for audit. "
            "Every group member has a task-native phase-aware natural-completion "
            "model. This is not joint coverage over arbitrary load states."
        ),
    }


def _build_wave_max_simultaneous_bound(
    *,
    calibration_payloads: Sequence[Mapping[str, Any]],
    holdout_rows: Mapping[str, Mapping[str, Any]],
    result_rows: Sequence[Mapping[str, Any]],
    expected_cells: set[str],
    miscoverage_alpha: float,
    min_interval_sample_count: int,
) -> dict[str, Any]:
    result_by_cell = {
        str(row.get("calibration_cell_id") or ""): row
        for row in result_rows
    }
    wave_scores = []
    wave_diagnostics = []
    for wave_index, payload in enumerate(calibration_payloads, start=1):
        rows = _rows_by_cell(payload)
        scores = []
        errors = []
        for cell_id in sorted(expected_cells):
            row = rows.get(cell_id)
            holdout_row = holdout_rows.get(cell_id)
            result = result_by_cell.get(cell_id)
            if row is None or holdout_row is None or result is None:
                errors.append(f"{cell_id}:missing")
                continue
            if _state_signature(row) != _state_signature(holdout_row):
                errors.append(f"{cell_id}:state_mismatch")
                continue
            if not _row_ready(
                row,
                min_interval_sample_count=min_interval_sample_count,
            ):
                errors.append(f"{cell_id}:not_ready")
                continue
            predicted = _predict_group_total(
                result.get("frozen_conformal_group_phase_model") or {},
                _aggregate_units(row),
            )
            score = _one_sided_ratio_score(
                actual=_actual_total(row),
                predicted=predicted,
            )
            if not math.isfinite(score):
                errors.append(f"{cell_id}:nonfinite_score")
                continue
            scores.append(float(score))
        ready = not errors and len(scores) == len(expected_cells)
        wave_score = max(scores) if ready and scores else None
        if wave_score is not None:
            wave_scores.append(wave_score)
        wave_diagnostics.append(
            {
                "calibration_wave": wave_index,
                "ready": ready,
                "cell_score_count": len(scores),
                "wave_max_ratio_score": wave_score,
                "errors": errors,
            }
        )
    rank = int(
        math.ceil(
            (len(wave_scores) + 1)
            * (1.0 - float(miscoverage_alpha))
        )
    )
    ready = bool(
        expected_cells
        and len(wave_scores) == len(calibration_payloads)
        and wave_scores
        and 1 <= rank <= len(wave_scores)
    )
    margin = (
        max(0.0, sorted(wave_scores)[rank - 1])
        if ready
        else None
    )
    return {
        "construction": "wave_max_simultaneous_split_conformal",
        "ready": ready,
        "exchangeability_unit": "complete_exact_cell_calibration_wave",
        "expected_cell_count": len(expected_cells),
        "calibration_wave_count": len(calibration_payloads),
        "ready_calibration_wave_count": len(wave_scores),
        "miscoverage_alpha": float(miscoverage_alpha),
        "nominal_simultaneous_family_coverage": (
            1.0 - float(miscoverage_alpha)
        ),
        "conformal_rank": rank,
        "wave_max_ratio_scores": wave_scores,
        "conformal_ratio_margin": margin,
        "wave_diagnostics": wave_diagnostics,
        "assumption": (
            "The complete pre-registered calibration vectors and the future "
            "holdout vector are exchangeable at the wave level within the fixed "
            "exact cell family and dispatch-state targets."
        ),
    }


def _matching_rows(
    payloads: Sequence[dict[str, Any]],
    cell_id: str,
) -> list[dict[str, Any]]:
    result = []
    for payload in payloads:
        rows = _rows_by_cell(payload)
        if cell_id in rows:
            result.append(rows[cell_id])
    return result


def _rows_by_cell(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in payload.get("rows") or []:
        row = dict(raw)
        cell_id = str(row.get("calibration_cell_id") or "").strip()
        if not cell_id:
            raise ValueError("co-location row is missing calibration_cell_id")
        if cell_id in result:
            raise ValueError(f"duplicate calibration cell {cell_id!r}")
        result[cell_id] = row
    return result


def _row_ready(
    row: Mapping[str, Any],
    *,
    min_interval_sample_count: int,
) -> bool:
    tasks = list(row.get("task_results") or [])
    expected = int(row.get("colocation_count") or 0)
    if (
        not row.get("measurement_valid")
        or not row.get("all_tasks_ready")
        or not row.get("near_synchronous_start")
        or not row.get("dispatch_state_ready")
        or row.get("measurement_protocol") != MEASUREMENT_PROTOCOL
        or int(row.get("dispatch_sample_count") or 0)
        < DISPATCH_CPU_SAMPLE_COUNT
        or not str(row.get("dispatch_external_cpu_regime") or "").strip()
        or str(row.get("eta_source") or "").strip() == "history"
        or expected <= 0
        or len(tasks) != expected
    ):
        return False
    for task in tasks:
        model = task.get("completion_model") or {}
        if (
            not task.get("measurement_valid")
            or int(task.get("returncode") or 0) != 0
            or not model.get("completion_model_ready")
            or not model.get("natural_exit")
            or int(model.get("interval_sample_count") or 0)
            < max(1, int(min_interval_sample_count))
        ):
            return False
    return True


def _state_signature(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("node"),
        row.get("workload_key"),
        row.get("workload_env"),
        row.get("observed_effective_resource_state", row.get("resource_state")),
        row.get("dispatch_external_cpu_regime"),
        row.get("dispatch_external_cpu_bucket"),
        row.get(
            "run_window_effective_resource_state",
            row.get("observed_effective_resource_state", row.get("resource_state")),
        ),
        row.get(
            "run_window_external_cpu_bucket",
            row.get("external_cpu_bucket"),
        ),
        int(row.get("allocation_workers") or 1),
        int(row.get("colocation_count") or 1),
        float(row.get("aggregate_service_units") or 0.0),
        row.get("init_cache_state", "unspecified"),
        row.get("profile_axis"),
    )


def _group_phase_components(row: Mapping[str, Any]) -> dict[str, float]:
    tasks = list(row.get("task_results") or [])
    startup = 0.0
    terminal = 0.0
    for task in tasks:
        model = task.get("completion_model") or {}
        startup = max(
            startup,
            max(0.0, float(task.get("launch_offset_s") or 0.0))
            + max(0.0, float(model.get("startup_overhead_s") or 0.0)),
        )
        terminal = max(
            terminal,
            max(0.0, float(model.get("terminal_overhead_s") or 0.0)),
        )
    makespan = _actual_total(row)
    units = _aggregate_units(row)
    loop_wall = max(0.0, makespan - startup - terminal)
    return {
        "startup_overhead_s": startup,
        "aggregate_completion_unit_s": loop_wall / units if units > 0.0 else 0.0,
        "terminal_overhead_s": terminal,
    }


def _fit_group_phase_medians(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    if not rows:
        return {
            "startup_overhead_s": 0.0,
            "aggregate_completion_unit_s": 0.0,
            "terminal_overhead_s": 0.0,
        }
    components = [_group_phase_components(row) for row in rows]
    return {
        key: float(median(component[key] for component in components))
        for key in (
            "startup_overhead_s",
            "aggregate_completion_unit_s",
            "terminal_overhead_s",
        )
    }


def _predict_group_total(model: Mapping[str, Any], units: float) -> float:
    return (
        max(0.0, float(model.get("startup_overhead_s") or 0.0))
        + max(0.0, float(units))
        * max(0.0, float(model.get("aggregate_completion_unit_s") or 0.0))
        + max(0.0, float(model.get("terminal_overhead_s") or 0.0))
    )


def _aggregate_units(row: Mapping[str, Any]) -> float:
    return max(0.0, float(row.get("aggregate_service_units") or 0.0))


def _actual_total(row: Mapping[str, Any]) -> float:
    return max(0.0, float(row.get("makespan_s") or 0.0))


def _minimum_interval_count(row: Mapping[str, Any]) -> int:
    counts = [
        int((task.get("completion_model") or {}).get("interval_sample_count") or 0)
        for task in (row.get("task_results") or [])
    ]
    return min(counts, default=0)


def _one_sided_ratio_score(*, actual: float, predicted: float) -> float:
    if actual <= 0.0 or predicted <= 0.0:
        return math.inf
    return actual / predicted - 1.0


def _nearest_rank(values: Sequence[float], quantile: float) -> float:
    if not values:
        return math.inf
    rank = max(1, int(math.ceil(float(quantile) * len(values))))
    return float(values[min(len(values) - 1, rank - 1)])


def _load(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", type=Path, nargs="+", required=True)
    parser.add_argument("--calibration", type=Path, nargs="+", required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--p95-limit", type=float, default=0.10)
    parser.add_argument("--max-limit", type=float, default=0.15)
    parser.add_argument("--min-interval-samples", type=int, default=5)
    parser.add_argument("--expected-cell-id", action="append", default=[])
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    result = build_colocation_stochastic_lcb_gate(
        training_paths=args.training,
        calibration_paths=args.calibration,
        holdout_path=args.holdout,
        miscoverage_alpha=args.alpha,
        p95_relative_error_limit=args.p95_limit,
        max_relative_error_limit=args.max_limit,
        min_interval_sample_count=args.min_interval_samples,
        expected_cell_ids=args.expected_cell_id or None,
    )
    _atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
