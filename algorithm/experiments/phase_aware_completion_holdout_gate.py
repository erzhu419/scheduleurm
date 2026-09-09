"""Independent holdout validation for phase-aware completion ETA models."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable


def build_completion_holdout_gate(
    *,
    fit_path: Path,
    holdout_path: Path,
    p95_relative_error_limit: float = 0.10,
    max_relative_error_limit: float = 0.15,
) -> dict[str, Any]:
    fit = json.loads(fit_path.read_text(encoding="utf-8"))
    holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
    fit_rows = {str(row.get("cell_id")): row for row in fit.get("rows") or []}
    holdout_rows = {str(row.get("cell_id")): row for row in holdout.get("rows") or []}
    rows = []
    for cell_id in sorted(set(fit_rows) & set(holdout_rows)):
        fit_row = fit_rows[cell_id]
        holdout_row = holdout_rows[cell_id]
        model = fit_row.get("completion_model") or {}
        holdout_model = holdout_row.get("completion_model") or {}
        fit_dispatch_bucket = fit_row.get("dispatch_external_cpu_bucket")
        holdout_dispatch_bucket = holdout_row.get(
            "dispatch_external_cpu_bucket"
        )
        resource_telemetry_required = bool(
            "observed_resource_state" in fit_row
            or "observed_resource_state" in holdout_row
            or "dispatch_state_ready" in fit_row
            or "dispatch_state_ready" in holdout_row
        )
        resource_state_match = bool(
            not resource_telemetry_required
            or (
                fit_row.get("resource_telemetry_ready")
                and holdout_row.get("resource_telemetry_ready")
                and fit_row.get("dispatch_state_ready")
                and holdout_row.get("dispatch_state_ready")
                and fit_dispatch_bucket
                and fit_dispatch_bucket == holdout_dispatch_bucket
            )
        )
        completion_ready = bool(
            model.get("completion_model_ready")
            and holdout_model.get("completion_model_ready")
        )
        ready = bool(completion_ready and resource_state_match)
        total_units = max(0.0, float(holdout_model.get("total_units") or 0.0))
        predicted = (
            max(0.0, float(model.get("startup_overhead_s") or 0.0))
            + total_units * max(0.0, float(model.get("completion_unit_s") or 0.0))
            + max(0.0, float(model.get("terminal_overhead_s") or 0.0))
        )
        actual = max(0.0, float(holdout_model.get("total_wall_s") or 0.0))
        relative_error = abs(predicted - actual) / actual if ready and actual > 0.0 else math.inf
        rows.append({
            "cell_id": cell_id,
            "node": holdout_row.get("node"),
            "workload_key": holdout_row.get("workload_key"),
            "workers": holdout_row.get("workers"),
            "resource_state": holdout_row.get("resource_state"),
            "fit_dispatch_external_cpu_core_equiv": fit_row.get(
                "dispatch_external_cpu_core_equiv"
            ),
            "holdout_dispatch_external_cpu_core_equiv": holdout_row.get(
                "dispatch_external_cpu_core_equiv"
            ),
            "fit_dispatch_external_cpu_bucket": fit_dispatch_bucket,
            "holdout_dispatch_external_cpu_bucket": (
                holdout_dispatch_bucket
            ),
            "fit_run_window_external_cpu_core_equiv": fit_row.get(
                "run_window_external_cpu_core_equiv",
                fit_row.get("external_cpu_core_equiv"),
            ),
            "holdout_run_window_external_cpu_core_equiv": holdout_row.get(
                "run_window_external_cpu_core_equiv",
                holdout_row.get("external_cpu_core_equiv"),
            ),
            "fit_run_window_external_cpu_bucket": fit_row.get(
                "run_window_external_cpu_bucket",
                fit_row.get("external_cpu_bucket"),
            ),
            "holdout_run_window_external_cpu_bucket": holdout_row.get(
                "run_window_external_cpu_bucket",
                holdout_row.get("external_cpu_bucket"),
            ),
            "resource_telemetry_required": resource_telemetry_required,
            "resource_state_match": resource_state_match,
            "fit_run_id": fit_row.get("run_id"),
            "holdout_run_id": holdout_row.get("run_id"),
            "predicted_total_s": predicted,
            "actual_total_s": actual,
            "absolute_error_s": abs(predicted - actual) if math.isfinite(relative_error) else None,
            "relative_error": relative_error if math.isfinite(relative_error) else None,
            "completion_ready": completion_ready,
            "ready": ready and actual > 0.0,
        })
    errors = sorted(
        float(row["relative_error"])
        for row in rows
        if row.get("ready") and row.get("relative_error") is not None
    )
    p95 = _nearest_rank(errors, 0.95)
    maximum = max(errors, default=math.inf)
    all_rows_ready = bool(rows) and len(errors) == len(rows)
    passed = bool(
        all_rows_ready
        and p95 <= float(p95_relative_error_limit)
        and maximum <= float(max_relative_error_limit)
    )
    return {
        "gate": "phase_aware_completion_holdout_gate",
        "fit_path": str(fit_path),
        "holdout_path": str(holdout_path),
        "row_count": len(rows),
        "ready_row_count": len(errors),
        "all_rows_ready": all_rows_ready,
        "p95_relative_error": p95 if math.isfinite(p95) else None,
        "max_relative_error": maximum if math.isfinite(maximum) else None,
        "p95_relative_error_limit": float(p95_relative_error_limit),
        "max_relative_error_limit": float(max_relative_error_limit),
        "pass": passed,
        "status": "PASS" if passed else "FAIL",
        "rows": rows,
        "claim_boundary": (
            "The fit model is frozen from the first natural-completion run and "
            "scored against a distinct second run of the same exact cell. This "
            "is an out-of-sample run holdout, not the algebraic in-run residual. "
            "When resource telemetry is present, fit and holdout must occupy the "
            "same external-CPU bucket measured before child launch. Run-window "
            "load is retained only as an audited outcome and is never a "
            "conditioning variable."
        ),
    }


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
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--p95-limit", type=float, default=0.10)
    parser.add_argument("--max-limit", type=float, default=0.15)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    result = build_completion_holdout_gate(
        fit_path=args.fit,
        holdout_path=args.holdout,
        p95_relative_error_limit=args.p95_limit,
        max_relative_error_limit=args.max_limit,
    )
    _atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
