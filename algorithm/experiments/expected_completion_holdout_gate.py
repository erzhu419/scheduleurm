"""Prospective expected-completion gate for exact phase-aware action cells."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from statistics import fmean, stdev
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from .phase_aware_colocation_stochastic_lcb_gate import (
    _group_phase_components,
    _row_ready,
    _rows_by_cell,
    _state_signature,
)


def build_expected_completion_holdout_gate(
    *,
    fit_paths: Sequence[Path],
    validation_paths: Sequence[Path],
    p95_relative_error_limit: float = 0.10,
    max_relative_error_limit: float = 0.15,
    min_interval_sample_count: int = 5,
) -> dict[str, Any]:
    """Validate a frozen sample-mean E[JCT] model on future complete waves."""

    fit = [_load(path) for path in fit_paths]
    validation = [_load(path) for path in validation_paths]
    artifact_rows = [
        *(('fit', index, _rows_by_cell(payload)) for index, payload in enumerate(fit, 1)),
        *(('validation', index, _rows_by_cell(payload)) for index, payload in enumerate(validation, 1)),
    ]
    expected_cells = set(artifact_rows[0][2]) if artifact_rows else set()
    set_checks = [
        {
            "artifact": f"{role}:{index}",
            "exact_match": set(rows) == expected_cells,
            "missing_cell_ids": sorted(expected_cells - set(rows)),
            "unexpected_cell_ids": sorted(set(rows) - expected_cells),
        }
        for role, index, rows in artifact_rows
    ]
    exact_sets = bool(expected_cells) and all(row["exact_match"] for row in set_checks)
    result_rows = []
    for cell_id in sorted(expected_cells):
        fit_rows = [_rows_by_cell(payload)[cell_id] for payload in fit]
        validation_rows = [_rows_by_cell(payload)[cell_id] for payload in validation]
        reference = validation_rows[0]
        signature = _state_signature(reference)
        statewise = all(
            _state_signature(row) == signature
            for row in (*fit_rows, *validation_rows)
        )
        ready = statewise and all(
            _row_ready(row, min_interval_sample_count=min_interval_sample_count)
            for row in (*fit_rows, *validation_rows)
        )
        components = [_group_phase_components(row) for row in fit_rows]
        phase_model = {
            key: fmean(component[key] for component in components)
            for key in (
                "startup_overhead_s",
                "aggregate_completion_unit_s",
                "terminal_overhead_s",
            )
        }
        units = float(reference.get("aggregate_service_units") or 0.0)
        predicted = (
            phase_model["startup_overhead_s"]
            + units * phase_model["aggregate_completion_unit_s"]
            + phase_model["terminal_overhead_s"]
        )
        actuals = [float(row.get("makespan_s") or 0.0) for row in validation_rows]
        validation_mean = fmean(actuals) if actuals else 0.0
        expected_error = (
            abs(predicted - validation_mean) / validation_mean
            if predicted > 0.0 and validation_mean > 0.0
            else math.inf
        )
        individual_errors = [
            abs(predicted - value) / value if value > 0.0 else math.inf
            for value in actuals
        ]
        standard_error = (
            stdev(actuals) / math.sqrt(len(actuals)) if len(actuals) >= 2 else None
        )
        result_rows.append(
            {
                "calibration_cell_id": cell_id,
                "node": reference.get("node"),
                "workload_key": reference.get("workload_key"),
                "workload_env": reference.get("workload_env"),
                "allocation_workers": int(reference.get("allocation_workers") or 1),
                "colocation_count": int(reference.get("colocation_count") or 1),
                "profile_axis": reference.get("profile_axis"),
                "observed_effective_resource_state": reference.get(
                    "observed_effective_resource_state"
                ),
                "run_window_effective_resource_state": reference.get(
                    "run_window_effective_resource_state"
                ),
                "dispatch_external_cpu_bucket": reference.get(
                    "dispatch_external_cpu_bucket"
                ),
                "run_window_external_cpu_bucket": reference.get(
                    "run_window_external_cpu_bucket"
                ),
                "aggregate_service_units": units,
                "state_signature": list(signature),
                "statewise_match": statewise,
                "fit_sample_count": len(fit_rows),
                "validation_sample_count": len(validation_rows),
                "frozen_expected_phase_model": phase_model,
                "predicted_expected_makespan_s": predicted,
                "validation_mean_makespan_s": validation_mean,
                "validation_standard_error_s": standard_error,
                "validation_actual_makespans_s": actuals,
                "expected_makespan_relative_error": expected_error,
                "individual_relative_errors": individual_errors,
                "ready": ready and math.isfinite(expected_error),
            }
        )

    errors = sorted(
        float(row["expected_makespan_relative_error"])
        for row in result_rows
        if row["ready"]
    )
    p95 = _nearest_rank(errors, 0.95)
    maximum = max(errors, default=math.inf)
    all_ready = bool(result_rows) and len(errors) == len(result_rows)
    passed = bool(
        exact_sets
        and all_ready
        and p95 <= float(p95_relative_error_limit)
        and maximum <= float(max_relative_error_limit)
    )
    return {
        "gate": "expected_completion_holdout_gate",
        "schema_version": 1,
        "status": "PASS" if passed else "FAIL",
        "pass": passed,
        "estimand": "expected_group_natural_completion_makespan",
        "estimator": "arithmetic_mean_of_exact_phase_components",
        "fit_paths": [str(path) for path in fit_paths],
        "validation_paths": [str(path) for path in validation_paths],
        "fit_wave_count": len(fit_paths),
        "prospective_validation_wave_count": len(validation_paths),
        "expected_cell_count": len(expected_cells),
        "artifact_cell_sets_ready": exact_sets,
        "artifact_cell_set_diagnostics": set_checks,
        "all_rows_ready": all_ready,
        "point_p95_relative_error": p95 if math.isfinite(p95) else None,
        "point_max_relative_error": maximum if math.isfinite(maximum) else None,
        "p95_relative_error_limit": float(p95_relative_error_limit),
        "max_relative_error_limit": float(max_relative_error_limit),
        "rows": result_rows,
        "claim_boundary": (
            "PASS validates the frozen expected natural-completion makespan "
            "model on complete future waves for the exact declared cells. It "
            "does not claim a 10% error bound for every individual completion; "
            "online task ETA is updated separately from task-native progress."
        ),
    }


def _load(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _nearest_rank(values: Sequence[float], quantile: float) -> float:
    if not values:
        return math.inf
    rank = max(1, int(math.ceil(float(quantile) * len(values))))
    return float(values[min(len(values) - 1, rank - 1)])


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", type=Path, nargs="+", required=True)
    parser.add_argument("--validation", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--p95-limit", type=float, default=0.10)
    parser.add_argument("--max-limit", type=float, default=0.15)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    report = build_expected_completion_holdout_gate(
        fit_paths=args.fit,
        validation_paths=args.validation,
        p95_relative_error_limit=args.p95_limit,
        max_relative_error_limit=args.max_limit,
    )
    _atomic_write(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
