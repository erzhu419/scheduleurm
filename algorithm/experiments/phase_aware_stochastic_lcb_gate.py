"""Split-conformal validation for phase-aware completion ETA and service LCBs."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
from statistics import median
import tempfile
from typing import Any, Iterable, Sequence

from .file_progress_completion_wrapper import (
    DISPATCH_CPU_SAMPLE_COUNT,
    MEASUREMENT_PROTOCOL,
)


def build_stochastic_completion_lcb_gate(
    *,
    training_paths: Sequence[Path],
    calibration_paths: Sequence[Path],
    holdout_path: Path,
    miscoverage_alpha: float = 0.10,
    p95_relative_error_limit: float = 0.10,
    max_relative_error_limit: float = 0.15,
    min_interval_sample_count: int = 5,
    min_training_sample_count: int = 3,
) -> dict[str, Any]:
    """Fit point and bound models, then score one untouched holdout.

    The training runs determine the frozen conformal base model. Distinct
    calibration runs supply one-sided ratio nonconformity scores. The
    operational point ETA may use every pre-holdout row, but it never replaces
    the frozen base model or calibration scores used by the finite-sample
    split-conformal upper bound.
    """

    alpha = float(miscoverage_alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("miscoverage_alpha must lie strictly between zero and one")
    if not training_paths:
        raise ValueError("at least one training path is required")
    if not calibration_paths:
        raise ValueError("at least one calibration path is required")

    training = [_load_artifact(path) for path in training_paths]
    calibration = [_load_artifact(path) for path in calibration_paths]
    holdout = _load_artifact(holdout_path)
    holdout_rows = _rows_by_calibration_cell(holdout)
    rows: list[dict[str, Any]] = []

    for cell_id in sorted(holdout_rows):
        holdout_row = holdout_rows[cell_id]
        raw_training_rows = [
            rows_by_cell[cell_id]
            for rows_by_cell in (_rows_by_calibration_cell(item) for item in training)
            if cell_id in rows_by_cell
        ]
        raw_calibration_rows = [
            rows_by_cell[cell_id]
            for rows_by_cell in (_rows_by_calibration_cell(item) for item in calibration)
            if cell_id in rows_by_cell
        ]
        target_signature = _state_signature(holdout_row)
        training_rows = [
            row for row in raw_training_rows
            if _state_signature(row) == target_signature
        ]
        calibration_rows = [
            row for row in raw_calibration_rows
            if _state_signature(row) == target_signature
        ]
        minimum_training_count = max(1, int(min_training_sample_count))
        minimum_sample_counts_ready = bool(
            len(training_rows) >= minimum_training_count
            and calibration_rows
        )
        all_sample_rows = [*training_rows, *calibration_rows, holdout_row]
        all_completion_ready = bool(
            all(
                _completion_row_ready(
                    row,
                    min_interval_sample_count=min_interval_sample_count,
                )
                for row in all_sample_rows
            )
        )
        signatures = [_state_signature(row) for row in all_sample_rows]
        statewise_match = bool(
            training_rows
            and calibration_rows
            and signatures
            and len(set(signatures)) == 1
        )

        target_model = holdout_row.get("completion_model") or {}
        total_units = max(0.0, float(target_model.get("total_units") or 0.0))
        frozen_conformal_phase_model = _fit_phase_medians(training_rows)
        conformal_base_prediction = _predict_total(
            frozen_conformal_phase_model,
            total_units,
        )
        raw_calibration_scores = [
            _one_sided_ratio_score(
                actual=_actual_total(row),
                predicted=_predict_total(
                    frozen_conformal_phase_model,
                    float((row.get("completion_model") or {}).get("total_units") or 0.0),
                ),
            )
            for row in calibration_rows
            if _actual_total(row) > 0.0
        ]
        calibration_scores = [
            score for score in raw_calibration_scores if math.isfinite(score)
        ]
        all_calibration_scores_finite = bool(
            raw_calibration_scores
            and len(calibration_scores) == len(raw_calibration_scores)
        )
        conformal_rank = int(math.ceil((len(calibration_scores) + 1) * (1.0 - alpha)))
        finite_sample_conformal_ready = bool(
            all_calibration_scores_finite
            and calibration_scores
            and conformal_rank >= 1
            and conformal_rank <= len(calibration_scores)
        )
        conformal_ratio_margin = (
            max(0.0, sorted(calibration_scores)[conformal_rank - 1])
            if finite_sample_conformal_ready
            else math.inf
        )
        conservative_upper = (
            conformal_base_prediction * (1.0 + conformal_ratio_margin)
            if math.isfinite(conformal_ratio_margin)
            else math.inf
        )
        operational_point_rows = [*training_rows, *calibration_rows]
        operational_point_phase_model = _fit_phase_medians(
            operational_point_rows
        )
        operational_point_prediction = _predict_total(
            operational_point_phase_model,
            total_units,
        )
        actual = _actual_total(holdout_row)
        point_relative_error = (
            abs(operational_point_prediction - actual) / actual
            if operational_point_prediction > 0.0 and actual > 0.0
            else math.inf
        )
        conservative_covered = bool(
            math.isfinite(conservative_upper)
            and conservative_upper > 0.0
            and actual > 0.0
            and actual <= conservative_upper * (1.0 + 1e-12)
        )
        lower_service = (
            total_units / conservative_upper
            if total_units > 0.0 and math.isfinite(conservative_upper)
            else 0.0
        )
        realized_service = total_units / actual if total_units > 0.0 and actual > 0.0 else 0.0
        row_ready = bool(
            minimum_sample_counts_ready
            and all_completion_ready
            and statewise_match
            and finite_sample_conformal_ready
            and conformal_base_prediction > 0.0
            and operational_point_prediction > 0.0
            and actual > 0.0
        )
        rows.append(
            {
                "calibration_cell_id": cell_id,
                "node": holdout_row.get("node"),
                "workload_key": holdout_row.get("workload_key"),
                "workload_env": holdout_row.get("workload_env"),
                "allocation_workers": holdout_row.get("allocation_workers"),
                "colocation_count": holdout_row.get("colocation_count"),
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
                "total_units": total_units,
                "training_sample_count": len(training_rows),
                "raw_training_sample_count": len(raw_training_rows),
                "minimum_training_sample_count": minimum_training_count,
                "calibration_sample_count": len(calibration_rows),
                "raw_calibration_sample_count": len(raw_calibration_rows),
                "excluded_training_state_count": (
                    len(raw_training_rows) - len(training_rows)
                ),
                "excluded_calibration_state_count": (
                    len(raw_calibration_rows) - len(calibration_rows)
                ),
                "minimum_sample_counts_ready": minimum_sample_counts_ready,
                "all_completion_ready": all_completion_ready,
                "min_interval_sample_count": int(min_interval_sample_count),
                "observed_interval_sample_counts": [
                    int((row.get("completion_model") or {}).get("interval_sample_count") or 0)
                    for row in all_sample_rows
                ],
                "statewise_match": statewise_match,
                "state_signature": list(signatures[-1]) if signatures else None,
                "frozen_phase_model": frozen_conformal_phase_model,
                "frozen_conformal_phase_model": (
                    frozen_conformal_phase_model
                ),
                "conformal_base_predicted_total_s": (
                    conformal_base_prediction
                ),
                "operational_point_sample_count": len(
                    operational_point_rows
                ),
                "operational_point_phase_model": (
                    operational_point_phase_model
                ),
                "operational_point_predicted_total_s": (
                    operational_point_prediction
                ),
                "point_predicted_total_s": operational_point_prediction,
                "actual_total_s": actual,
                "point_absolute_error_s": (
                    abs(operational_point_prediction - actual)
                    if math.isfinite(point_relative_error)
                    else None
                ),
                "point_relative_error": (
                    point_relative_error if math.isfinite(point_relative_error) else None
                ),
                "miscoverage_alpha": alpha,
                "conformal_rank": conformal_rank,
                "calibration_ratio_scores": calibration_scores,
                "nonfinite_calibration_score_count": (
                    len(raw_calibration_scores) - len(calibration_scores)
                ),
                "all_calibration_scores_finite": (
                    all_calibration_scores_finite
                ),
                "conformal_ratio_margin": (
                    conformal_ratio_margin
                    if math.isfinite(conformal_ratio_margin)
                    else None
                ),
                "finite_sample_conformal_ready": finite_sample_conformal_ready,
                "conservative_upper_total_s": (
                    conservative_upper if math.isfinite(conservative_upper) else None
                ),
                "conservative_holdout_covered": conservative_covered,
                "lower_service_units_per_s": lower_service,
                "realized_service_units_per_s": realized_service,
                "lower_service_valid_on_holdout": bool(
                    row_ready
                    and conservative_covered
                    and lower_service <= realized_service * (1.0 + 1e-12)
                ),
                "ready": row_ready,
            }
        )

    point_errors = sorted(
        float(row["point_relative_error"])
        for row in rows
        if row.get("ready") and row.get("point_relative_error") is not None
    )
    p95 = _nearest_rank(point_errors, 0.95)
    maximum = max(point_errors, default=math.inf)
    all_rows_ready = bool(rows) and len(point_errors) == len(rows)
    holdout_coverage = (
        sum(bool(row.get("conservative_holdout_covered")) for row in rows) / len(rows)
        if rows
        else 0.0
    )
    all_lower_service_valid = bool(rows) and all(
        bool(row.get("lower_service_valid_on_holdout")) for row in rows
    )
    passed = bool(
        all_rows_ready
        and p95 <= float(p95_relative_error_limit)
        and maximum <= float(max_relative_error_limit)
        and all_lower_service_valid
    )
    return {
        "gate": "phase_aware_stochastic_lcb_gate",
        "schema_version": 1,
        "training_paths": [str(path) for path in training_paths],
        "calibration_paths": [str(path) for path in calibration_paths],
        "holdout_path": str(holdout_path),
        "miscoverage_alpha": alpha,
        "nominal_cellwise_coverage": 1.0 - alpha,
        "min_interval_sample_count": int(min_interval_sample_count),
        "min_training_sample_count": max(1, int(min_training_sample_count)),
        "row_count": len(rows),
        "ready_row_count": len(point_errors),
        "all_rows_ready": all_rows_ready,
        "point_p95_relative_error": p95 if math.isfinite(p95) else None,
        "point_max_relative_error": maximum if math.isfinite(maximum) else None,
        "p95_relative_error_limit": float(p95_relative_error_limit),
        "max_relative_error_limit": float(max_relative_error_limit),
        "empirical_holdout_coverage": holdout_coverage,
        "all_lower_service_valid_on_holdout": all_lower_service_valid,
        "pass": passed,
        "status": "PASS" if passed else "FAIL",
        "rows": rows,
        "claim_boundary": (
            "The one-sided completion bound has finite-sample split-conformal "
            "cellwise coverage conditional on exchangeability within the exact "
            "workload, node, hardware-normalized multi-window dispatch regime, "
            "allocation/colocation, "
            "total-unit, and cache-state cell. The dispatch state is measured before "
            "the controlled child starts; the legacy absolute core bucket and "
            "run-window load remain audited outcomes rather than conditioning "
            "variables. Rows from other dispatch states are "
            "excluded before fitting and calibration. The conformal base model "
            "uses training rows only, and its calibration residuals remain frozen. "
            "The operational point ETA separately uses all pre-holdout training "
            "and calibration rows; it has no conformal coverage claim and cannot "
            "alter the lower-service bound. The final run is untouched holdout "
            "evidence. This "
            "does not imply simultaneous coverage over arbitrary future cells or "
            "coverage after a resource-state distribution shift."
        ),
    }


def _load_artifact(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _rows_by_calibration_cell(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in payload.get("rows") or []:
        key = _calibration_cell_id(row)
        if key in result:
            raise ValueError(f"duplicate calibration cell {key!r}")
        result[key] = row
    return result


def _calibration_cell_id(row: dict[str, Any]) -> str:
    explicit = str(row.get("calibration_cell_id") or "").strip()
    if explicit:
        return explicit
    cell_id = str(row.get("cell_id") or "").strip()
    return re.sub(r"_seed\d+$", "", cell_id)


def _completion_row_ready(
    row: dict[str, Any],
    *,
    min_interval_sample_count: int,
) -> bool:
    model = row.get("completion_model") or {}
    measurement_valid = row.get("measurement_valid")
    try:
        natural_return = int(row.get("returncode")) == 0
    except (TypeError, ValueError):
        natural_return = False
    try:
        natural_child_return = int(model.get("child_returncode")) == 0
    except (TypeError, ValueError):
        natural_child_return = False
    return bool(
        natural_return
        and natural_child_return
        and bool(model.get("natural_exit"))
        and model.get("completion_model_ready")
        and int(model.get("interval_sample_count") or 0)
        >= max(1, int(min_interval_sample_count))
        and (measurement_valid is None or bool(measurement_valid))
        and bool(row.get("dispatch_state_ready"))
        and row.get("measurement_protocol") == MEASUREMENT_PROTOCOL
        and int(row.get("dispatch_sample_count") or 0)
        >= DISPATCH_CPU_SAMPLE_COUNT
        and bool(
            str(row.get("dispatch_external_cpu_regime") or "").strip()
        )
        and bool(str(row.get("dispatch_external_cpu_bucket") or "").strip())
        and str(row.get("eta_source") or "").strip() != "history"
    )


def _state_signature(row: dict[str, Any]) -> tuple[Any, ...]:
    model = row.get("completion_model") or {}
    return (
        row.get("node"),
        row.get("workload_key"),
        row.get("workload_env"),
        row.get("observed_effective_resource_state", row.get("resource_state")),
        row.get("dispatch_external_cpu_regime"),
        int(row.get("allocation_workers") or row.get("workers") or 1),
        int(row.get("colocation_count") or 1),
        int(model.get("total_units") or row.get("total_outer_units") or 0),
        row.get("init_cache_state", "unspecified"),
    )


def _fit_phase_medians(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {
            "startup_overhead_s": 0.0,
            "completion_unit_s": 0.0,
            "terminal_overhead_s": 0.0,
        }
    models = [row.get("completion_model") or {} for row in rows]
    return {
        "startup_overhead_s": float(
            median(max(0.0, float(model.get("startup_overhead_s") or 0.0)) for model in models)
        ),
        "completion_unit_s": float(
            median(max(0.0, float(model.get("completion_unit_s") or 0.0)) for model in models)
        ),
        "terminal_overhead_s": float(
            median(max(0.0, float(model.get("terminal_overhead_s") or 0.0)) for model in models)
        ),
    }


def _predict_total(model: dict[str, float], total_units: float) -> float:
    return (
        max(0.0, float(model.get("startup_overhead_s") or 0.0))
        + max(0.0, float(total_units))
        * max(0.0, float(model.get("completion_unit_s") or 0.0))
        + max(0.0, float(model.get("terminal_overhead_s") or 0.0))
    )


def _actual_total(row: dict[str, Any]) -> float:
    return max(0.0, float((row.get("completion_model") or {}).get("total_wall_s") or 0.0))


def _one_sided_ratio_score(*, actual: float, predicted: float) -> float:
    if actual <= 0.0 or predicted <= 0.0:
        return math.inf
    return actual / predicted - 1.0


def _nearest_rank(values: list[float], quantile: float) -> float:
    if not values:
        return math.inf
    rank = max(1, int(math.ceil(float(quantile) * len(values))))
    return float(values[min(len(values) - 1, rank - 1)])


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
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
    parser.add_argument("--min-training-samples", type=int, default=3)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    result = build_stochastic_completion_lcb_gate(
        training_paths=args.training,
        calibration_paths=args.calibration,
        holdout_path=args.holdout,
        miscoverage_alpha=args.alpha,
        p95_relative_error_limit=args.p95_limit,
        max_relative_error_limit=args.max_limit,
        min_interval_sample_count=max(1, int(args.min_interval_samples)),
        min_training_sample_count=max(1, int(args.min_training_samples)),
    )
    _atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
