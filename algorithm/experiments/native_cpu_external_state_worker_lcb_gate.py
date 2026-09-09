"""Simultaneous lower-service gate for organic light/moderate CPU states.

Unlike the controlled resident campaign, these rows retain the external load
that was already present on node004/node006.  The dispatch regime is therefore
part of every cell key and is rechecked before each target launch.  Training,
calibration, and holdout replicates are fixed before any completion outcomes
are read.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from statistics import fmean
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from .native_cpu_allocation_worker_lcb_gate import (
    _phase_model_ready,
    _wall_s,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_CAMPAIGNS = (
    ARTIFACT_ROOT
    / "native_cpu_allocation_worker_state_campaign_20260727_r3.json",
    ARTIFACT_ROOT
    / "native_cpu_allocation_worker_state_campaign_20260727_split_extension.json",
    ARTIFACT_ROOT
    / "native_cpu_allocation_worker_external_state_extension_20260727_r4_r12.json",
    ARTIFACT_ROOT
    / "native_cpu_allocation_worker_external_state_repair_20260727_moderate_w96_r10.json",
    ARTIFACT_ROOT
    / "native_cpu_allocation_worker_external_state_repair_20260727_light_w96_r10.json",
)
REGIME_STATES = {
    "cpu_external_light": "cpu_external_light",
    "cpu_external_moderate": "cpu_external_moderate",
}
DEFAULT_PROFILES = (2, 4, 8, 16, 32, 64, 96, 128)
TOTAL_UNITS = 160.0


def build_native_cpu_external_state_worker_lcb_gate(
    *,
    campaign_paths: Sequence[Path],
    regimes: Sequence[str] = tuple(REGIME_STATES),
    profiles: Sequence[int] = DEFAULT_PROFILES,
    training_replicates: Sequence[int] = (0, 1, 2),
    calibration_replicates: Sequence[int] = tuple(range(3, 12)),
    holdout_replicate: int = 12,
    miscoverage_alpha: float = 0.1,
    max_point_relative_error: float = 0.15,
) -> dict[str, Any]:
    expected_regimes = tuple(dict.fromkeys(str(value) for value in regimes))
    if any(value not in REGIME_STATES for value in expected_regimes):
        raise ValueError("unsupported external CPU regime")
    expected_profiles = tuple(sorted({int(value) for value in profiles}))
    training_ids = tuple(int(value) for value in training_replicates)
    calibration_ids = tuple(int(value) for value in calibration_replicates)
    holdout_id = int(holdout_replicate)
    required_ids = training_ids + calibration_ids + (holdout_id,)
    if set(training_ids) & set(calibration_ids):
        raise ValueError("training and calibration replicates must be disjoint")
    if holdout_id in set(training_ids) | set(calibration_ids):
        raise ValueError("holdout replicate must be disjoint")

    source_rows: list[dict[str, Any]] = []
    source_artifacts = []
    planned_nodes: dict[tuple[str, int, int], str] = {}
    node_mismatches = []
    for path in campaign_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        source_artifacts.append(
            {"path": str(path), "status": payload.get("status")}
        )
        for raw_row in payload.get("rows") or []:
            row = dict(raw_row)
            regime = str(row.get("target_regime"))
            if regime not in expected_regimes:
                continue
            key = (
                regime,
                int(row.get("workers") or 0),
                int(row.get("replicate") or 0),
            )
            node = str(row.get("node") or "")
            expected_node = planned_nodes.setdefault(key, node)
            if row.get("measurement_valid") and node != expected_node:
                node_mismatches.append(
                    {
                        "dispatch_external_cpu_regime": key[0],
                        "allocation_workers": key[1],
                        "replicate": key[2],
                        "expected_node": expected_node,
                        "observed_node": node,
                        "source": str(path),
                    }
                )
                continue
            if row.get("measurement_valid"):
                source_rows.append(row)

    indexed: dict[tuple[str, int, int], dict[str, Any]] = {}
    duplicates = []
    for row in source_rows:
        key = (
            str(row.get("target_regime")),
            int(row.get("workers") or 0),
            int(row.get("replicate") or 0),
        )
        if key in indexed:
            duplicates.append(
                {
                    "dispatch_external_cpu_regime": key[0],
                    "allocation_workers": key[1],
                    "replicate": key[2],
                }
            )
        else:
            indexed[key] = row

    expected_cells = tuple(
        (regime, profile)
        for regime in expected_regimes
        for profile in expected_profiles
    )
    missing = [
        {
            "dispatch_external_cpu_regime": regime,
            "allocation_workers": profile,
            "replicate": replicate,
        }
        for regime, profile in expected_cells
        for replicate in required_ids
        if (regime, profile, replicate) not in indexed
    ]

    bases: dict[tuple[str, int], float] = {}
    phase_models: dict[tuple[str, int], dict[str, float]] = {}
    source_nodes: dict[tuple[str, int], set[str]] = {}
    training_ready = True
    for regime, profile in expected_cells:
        rows = [
            indexed.get((regime, profile, replicate))
            for replicate in training_ids
        ]
        valid_rows = [row for row in rows if row]
        training_ready = bool(
            training_ready
            and len(valid_rows) == len(training_ids)
            and all(_phase_model_ready(row) for row in valid_rows)
            and all(_external_state_ready(row, regime) for row in valid_rows)
        )
        if len(valid_rows) != len(training_ids):
            continue
        key = (regime, profile)
        bases[key] = fmean(_wall_s(row) for row in valid_rows)
        source_nodes[key] = {
            str(row.get("node")) for row in valid_rows if row.get("node")
        }
        phase_models[key] = {
            field: fmean(
                float((row.get("completion_model") or {}).get(field) or 0.0)
                for row in valid_rows
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
    calibration_ready = True
    for replicate in calibration_ids:
        scores = []
        for regime, profile in expected_cells:
            row = indexed.get((regime, profile, replicate))
            base = bases.get((regime, profile), 0.0)
            if not row or base <= 0.0:
                continue
            calibration_ready = bool(
                calibration_ready
                and _phase_model_ready(row)
                and _external_state_ready(row, regime)
            )
            scores.append(_wall_s(row) / base - 1.0)
            if row.get("node"):
                source_nodes.setdefault((regime, profile), set()).add(
                    str(row["node"])
                )
        if len(scores) == len(expected_cells):
            wave_scores.append(max(scores))

    rank = min(
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
    margin = (
        max(0.0, sorted(wave_scores)[rank - 1])
        if wave_scores
        else math.inf
    )

    result_rows = []
    for regime, profile in expected_cells:
        key = (regime, profile)
        base = bases.get(key, 0.0)
        holdout = indexed.get((regime, profile, holdout_id))
        actual = _wall_s(holdout)
        upper = (
            base * (1.0 + margin)
            if base > 0.0 and math.isfinite(margin)
            else 0.0
        )
        phase = phase_models.get(key, {})
        covered = bool(
            holdout
            and _phase_model_ready(holdout)
            and _external_state_ready(holdout, regime)
            and upper >= actual > 0.0
        )
        measured_nodes = set(source_nodes.get(key, set()))
        if holdout and holdout.get("node"):
            measured_nodes.add(str(holdout["node"]))
        result_rows.append(
            {
                "dispatch_external_cpu_regime": regime,
                "resource_state": REGIME_STATES[regime],
                "allocation_workers": profile,
                "colocation_count": 1,
                "measured_nodes": sorted(measured_nodes),
                "point_eta_s": base,
                "holdout_actual_completion_s": actual,
                "point_relative_error": (
                    abs(base - actual) / actual
                    if actual > 0.0
                    else math.inf
                ),
                "simultaneous_conservative_eta_s": upper,
                "lower_service_units_per_s": (
                    TOTAL_UNITS / upper if upper > 0.0 else 0.0
                ),
                "realized_service_units_per_s": (
                    TOTAL_UNITS / actual if actual > 0.0 else 0.0
                ),
                "lower_service_valid_on_holdout": covered,
                "training_sample_count": len(training_ids),
                "calibration_wave_count": len(wave_scores),
                "holdout_replicate": holdout_id,
                "startup_overhead_s": float(
                    phase.get("startup_overhead_s") or 0.0
                ),
                "completion_unit_s": float(
                    phase.get("completion_unit_s") or 0.0
                ),
                "finalization_overhead_s": float(
                    phase.get("terminal_overhead_s") or 0.0
                ),
                "checkpoint_observed_s": float(
                    phase.get("checkpoint_observed_s") or 0.0
                ),
                "save_observed_s": float(
                    phase.get("save_observed_s") or 0.0
                ),
                "lower_service_bound_kind": (
                    "wave_max_simultaneous_split_conformal"
                ),
            }
        )

    finite_errors = [
        float(row["point_relative_error"])
        for row in result_rows
        if math.isfinite(float(row["point_relative_error"]))
    ]
    point_max = max(finite_errors, default=math.inf)
    sample_ready = bool(
        not missing
        and not duplicates
        and not node_mismatches
        and len(bases) == len(expected_cells)
        and len(wave_scores) == len(calibration_ids)
    )
    all_covered = bool(result_rows) and all(
        row["lower_service_valid_on_holdout"] for row in result_rows
    )
    passed = bool(
        sample_ready
        and training_ready
        and calibration_ready
        and all_covered
        and point_max <= float(max_point_relative_error)
    )
    return {
        "gate": "native_cpu_external_state_worker_lcb_gate",
        "schema_version": 1,
        "status": "PASS" if passed else "FAIL",
        "pass": passed,
        "campaign_paths": [str(path) for path in campaign_paths],
        "source_artifacts": source_artifacts,
        "regimes": list(expected_regimes),
        "profiles": list(expected_profiles),
        "training_replicates": list(training_ids),
        "calibration_replicates": list(calibration_ids),
        "holdout_replicate": holdout_id,
        "miscoverage_alpha": float(miscoverage_alpha),
        "conformal_rank": rank,
        "wave_max_scores": wave_scores,
        "simultaneous_ratio_margin": margin,
        "point_max_relative_error": point_max,
        "max_point_relative_error_limit": float(max_point_relative_error),
        "sample_counts_ready": sample_ready,
        "all_lower_service_valid_on_holdout": all_covered,
        "missing_cells": missing,
        "duplicate_cells": duplicates,
        "node_mismatches": node_mismatches,
        "rows": result_rows,
        "claim_boundary": (
            "The certificate covers only the observed node004 moderate and "
            "node006 light dispatch-regime processes. It does not identify "
            "organic external workload composition and is not extrapolated to "
            "unmeasured nodes or heavy/saturated states."
        ),
    }


def _external_state_ready(
    row: Mapping[str, Any],
    expected_regime: str,
) -> bool:
    preflight = row.get("dispatch_preflight") or {}
    observed = row.get("observed_resource_state") or {}
    return bool(
        row.get("target_regime") == expected_regime
        and preflight.get("ready")
        and preflight.get("dispatch_external_cpu_regime") == expected_regime
        and float(observed.get("external_cpu_core_equiv") or 0.0) > 0.0
    )


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
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
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


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
        default=",".join(str(path) for path in DEFAULT_CAMPAIGNS),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ARTIFACT_ROOT
            / "native_cpu_external_state_worker_lcb_gate_20260727.json"
        ),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    report = build_native_cpu_external_state_worker_lcb_gate(
        campaign_paths=_parse_paths(args.campaigns)
    )
    _atomic_write_json(args.output, report)
    print(args.output)
    return 0 if report.get("pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
