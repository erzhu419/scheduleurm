"""Split-conformal lower-service gate for controlled half/full CPU load."""
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
from .native_cpu_controlled_resident_worker_campaign import (
    FULL_LOAD,
    HALF_LOAD,
    LOAD_SPECS,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_CAMPAIGNS = (
    ARTIFACT_ROOT
    / "native_cpu_controlled_resident_worker_campaign_20260727_r13.json",
    ARTIFACT_ROOT
    / "native_cpu_controlled_resident_worker_campaign_20260727_p1_r13.json",
    ARTIFACT_ROOT
    / "native_cpu_controlled_resident_worker_campaign_20260727_repair_r8_w2.json",
    ARTIFACT_ROOT
    / "native_cpu_controlled_resident_worker_campaign_20260727_repair_r9_node001_w64.json",
    ARTIFACT_ROOT
    / "native_cpu_controlled_resident_worker_campaign_20260727_repair_r12_node001_w2_w64.json",
    ARTIFACT_ROOT
    / "native_cpu_controlled_resident_worker_campaign_20260727_repair_r12_node002_w4_w16.json",
)


def build_native_cpu_controlled_resident_worker_lcb_gate(
    *,
    campaign_paths: Sequence[Path],
    training_replicates: Sequence[int] = (0, 1, 2),
    calibration_replicates: Sequence[int] = tuple(range(3, 12)),
    holdout_replicate: int = 12,
    miscoverage_alpha: float = 0.1,
    max_point_relative_error: float = 0.15,
) -> dict[str, Any]:
    accepted = []
    source_artifacts = []
    planned_nodes: dict[tuple[str, int, int], str] = {}
    node_mismatches = []
    for campaign_path in campaign_paths:
        payload = json.loads(campaign_path.read_text(encoding="utf-8"))
        source_artifacts.append(
            {"path": str(campaign_path), "status": payload.get("status")}
        )
        for raw_row in payload.get("rows") or []:
            row = dict(raw_row)
            key = (
                str(row.get("load_class")),
                int(row.get("workers") or 0),
                int(row.get("replicate") or 0),
            )
            node = str(row.get("node") or "")
            expected_node = planned_nodes.setdefault(key, node)
            if row.get("measurement_valid") and node != expected_node:
                node_mismatches.append(
                    {
                        "load_class": key[0],
                        "allocation_workers": key[1],
                        "replicate": key[2],
                        "expected_node": expected_node,
                        "observed_node": node,
                        "source": str(campaign_path),
                    }
                )
                continue
            if row.get("measurement_valid"):
                accepted.append(row)
    indexed = {}
    duplicates = []
    for row in accepted:
        key = (
            str(row.get("load_class")),
            int(row.get("workers") or 0),
            int(row.get("replicate") or 0),
        )
        if key in indexed:
            duplicates.append(
                {
                    "load_class": key[0],
                    "allocation_workers": key[1],
                    "replicate": key[2],
                }
            )
        else:
            indexed[key] = row
    expected_cells = tuple(
        (load_class, int(profile))
        for load_class in (HALF_LOAD, FULL_LOAD)
        for profile in LOAD_SPECS[load_class]["profiles"]
    )
    training_ids = tuple(int(value) for value in training_replicates)
    calibration_ids = tuple(int(value) for value in calibration_replicates)
    holdout_id = int(holdout_replicate)
    required_ids = training_ids + calibration_ids + (holdout_id,)
    missing = [
        {
            "load_class": load_class,
            "allocation_workers": profile,
            "replicate": replicate,
        }
        for load_class, profile in expected_cells
        for replicate in required_ids
        if (load_class, profile, replicate) not in indexed
    ]

    bases = {}
    phase_models = {}
    training_ready = True
    for load_class, profile in expected_cells:
        rows = [
            indexed.get((load_class, profile, replicate))
            for replicate in training_ids
        ]
        valid_rows = [row for row in rows if row]
        training_ready = bool(
            training_ready
            and len(valid_rows) == len(training_ids)
            and all(_phase_model_ready(row) for row in valid_rows)
            and all(_resident_state_ready(row) for row in valid_rows)
        )
        if len(valid_rows) != len(training_ids):
            continue
        bases[(load_class, profile)] = fmean(
            _wall_s(row) for row in valid_rows
        )
        phase_models[(load_class, profile)] = {
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
        for load_class, profile in expected_cells:
            row = indexed.get((load_class, profile, replicate))
            base = bases.get((load_class, profile), 0.0)
            if not row or base <= 0.0:
                continue
            calibration_ready = bool(
                calibration_ready
                and _phase_model_ready(row)
                and _resident_state_ready(row)
            )
            scores.append(_wall_s(row) / base - 1.0)
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

    rows = []
    for load_class, profile in expected_cells:
        key = (load_class, profile)
        base = bases.get(key, 0.0)
        holdout = indexed.get((*key, holdout_id))
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
            and _resident_state_ready(holdout)
            and upper >= actual > 0.0
        )
        rows.append(
            {
                "load_class": load_class,
                "resource_state": LOAD_SPECS[load_class]["resource_state"],
                "resident_workers": int(
                    LOAD_SPECS[load_class]["resident_workers"]
                ),
                "dispatch_external_cpu_regime": LOAD_SPECS[load_class][
                    "expected_regime"
                ],
                "allocation_workers": profile,
                "colocation_count": 1,
                "point_eta_s": base,
                "holdout_actual_completion_s": actual,
                "point_relative_error": (
                    abs(base - actual) / actual
                    if actual > 0.0
                    else math.inf
                ),
                "simultaneous_conservative_eta_s": upper,
                "lower_service_units_per_s": (
                    160.0 / upper if upper > 0.0 else 0.0
                ),
                "realized_service_units_per_s": (
                    160.0 / actual if actual > 0.0 else 0.0
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

    point_max = max(
        (
            float(row["point_relative_error"])
            for row in rows
            if math.isfinite(float(row["point_relative_error"]))
        ),
        default=math.inf,
    )
    all_covered = bool(rows) and all(
        row["lower_service_valid_on_holdout"] for row in rows
    )
    sample_ready = bool(
        not missing
        and not duplicates
        and not node_mismatches
        and len(bases) == len(expected_cells)
        and len(wave_scores) == len(calibration_ids)
    )
    passed = bool(
        sample_ready
        and training_ready
        and calibration_ready
        and all_covered
        and point_max <= float(max_point_relative_error)
    )
    return {
        "gate": "native_cpu_controlled_resident_worker_lcb_gate",
        "schema_version": 1,
        "status": "PASS" if passed else "FAIL",
        "pass": passed,
        "campaign_paths": [str(path) for path in campaign_paths],
        "source_artifacts": source_artifacts,
        "training_replicates": list(training_ids),
        "calibration_replicates": list(calibration_ids),
        "holdout_replicate": holdout_id,
        "miscoverage_alpha": float(miscoverage_alpha),
        "conformal_rank": rank,
        "wave_max_scores": wave_scores,
        "simultaneous_ratio_margin": margin,
        "point_max_relative_error": point_max,
        "max_point_relative_error_limit": float(
            max_point_relative_error
        ),
        "sample_counts_ready": sample_ready,
        "all_lower_service_valid_on_holdout": all_covered,
        "missing_cells": missing,
        "duplicate_cells": duplicates,
        "node_mismatches": node_mismatches,
        "rows": rows,
        "claim_boundary": (
            "The certificate covers only controlled 96-core and 180-core "
            "resident states and target profiles satisfying the declared "
            "192-core capacity rule. Organic external load is a separate state."
        ),
    }


def _resident_state_ready(row: Mapping[str, Any]) -> bool:
    load_class = str(row.get("load_class") or "")
    if load_class not in LOAD_SPECS:
        return False
    spec = LOAD_SPECS[load_class]
    preflight = row.get("dispatch_preflight") or {}
    observed = row.get("observed_resource_state") or {}
    external_cores = float(
        observed.get("external_cpu_core_equiv") or 0.0
    )
    resident = float(spec["resident_workers"])
    tolerance = max(8.0, 0.20 * resident)
    return bool(
        preflight.get("ready")
        and preflight.get("dispatch_external_cpu_regime")
        == spec["expected_regime"]
        and int(row.get("resident_workers") or 0)
        == int(spec["resident_workers"])
        and int(row.get("resident_workers") or 0)
        + int(row.get("workers") or 0)
        <= 192
        and abs(external_cores - resident) <= tolerance
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
            / "native_cpu_controlled_resident_worker_lcb_gate_20260727.json"
        ),
    )
    return parser


def _parse_paths(raw: str) -> tuple[Path, ...]:
    return tuple(
        Path(item.strip())
        for item in str(raw or "").split(",")
        if item.strip()
    )


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    result = build_native_cpu_controlled_resident_worker_lcb_gate(
        campaign_paths=_parse_paths(args.campaigns)
    )
    _atomic_write_json(args.output, result)
    print(args.output)
    return 0 if result.get("pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
