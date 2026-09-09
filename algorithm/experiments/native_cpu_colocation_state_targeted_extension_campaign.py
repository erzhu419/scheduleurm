"""Pre-registered calibration extension for an exact state-targeted CPU campaign."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable, Mapping

from .file_progress_completion_wrapper import MEASUREMENT_PROTOCOL
from .native_cpu_colocation_stochastic_calibration_campaign import (
    _acquire_node_locks,
    _release_node_locks,
)
from .native_cpu_colocation_state_targeted_campaign import (
    ARTIFACT_ROOT,
    DEFAULT_KINDS,
    DEFAULT_NODES,
    _atomic_write_json,
    _deploy_nodes,
    _file_sha256_map,
    _parse_csv,
    _run_node_lanes,
    _safe_id,
    _write_composite_artifacts,
)
from .phase_aware_colocation_stochastic_lcb_gate import (
    build_colocation_stochastic_lcb_gate,
)


DEFAULT_ADDITIONAL_CALIBRATION_REPEATS = 9


def run_state_targeted_colocation_extension(
    *,
    base_campaign_path: Path,
    tag: str,
    allow_launch: bool = False,
    additional_calibration_repeats: int = (
        DEFAULT_ADDITIONAL_CALIBRATION_REPEATS
    ),
    max_parallel: int = 6,
    max_attempts_per_cell: int = 60,
    seed_base: int = 1_727_000,
    miscoverage_alpha: float = 0.10,
) -> dict[str, Any]:
    """Add calibration to every exact cell, then draw one new holdout vector."""

    started = time.time()
    safe_tag = _safe_id(tag)
    base_path = Path(base_campaign_path).resolve()
    base = json.loads(base_path.read_text(encoding="utf-8"))
    validation = _validate_base_campaign(base, base_path=base_path)
    calibration_repeats = max(1, int(additional_calibration_repeats))
    manifest = {
        "campaign": "native_cpu_colocation_state_targeted_extension_campaign",
        "schema_version": 1,
        "tag": safe_tag,
        "allow_launch": bool(allow_launch),
        "measurement_protocol": MEASUREMENT_PROTOCOL,
        "base_campaign_path": str(base_path),
        "base_campaign_sha256": hashlib.sha256(
            base_path.read_bytes()
        ).hexdigest(),
        "base_validation": validation,
        "profile": validation["profile"],
        "nodes": validation["nodes"],
        "kinds": validation["kinds"],
        "expected_cell_ids": validation["expected_cell_ids"],
        "target_dispatch_regimes": validation[
            "target_dispatch_regimes"
        ],
        "base_training_repeat_count": len(validation["training_paths"]),
        "base_calibration_repeat_count": len(
            validation["calibration_paths"]
        ),
        "additional_calibration_repeat_count": calibration_repeats,
        "new_holdout_repeat_count": 1,
        "max_parallel": int(max_parallel),
        "max_attempts_per_cell": int(max_attempts_per_cell),
        "performance_outcome_fields_used_for_selection": [],
        "old_holdout_used_for_final_gate": False,
        "old_holdout_used_for_selection": False,
        "training_model_remains_frozen": True,
        "user_task_control_operations": [],
        "legacy_scheduler_limits_bypassed": True,
    }
    if not validation["ready"]:
        return {
            **manifest,
            "status": "BASE_CAMPAIGN_GAP",
            "pass": False,
            "elapsed_wall_s": time.time() - started,
        }
    if not allow_launch:
        return {
            **manifest,
            "status": "MANIFEST",
            "pass": False,
            "elapsed_wall_s": time.time() - started,
        }

    lock_handles: list[Any] = []
    lock_rows: list[dict[str, Any]] = []
    try:
        lock_handles, lock_rows = _acquire_node_locks(
            validation["nodes"],
            tag=safe_tag,
        )
        deployment_rows = _deploy_nodes(
            validation["nodes"],
            tag=safe_tag,
            max_parallel=int(max_parallel),
        )
        if not all(row["ready"] for row in deployment_rows):
            return {
                **manifest,
                "node_lock_rows": lock_rows,
                "deployment_rows": deployment_rows,
                "status": "DEPLOYMENT_GAP",
                "pass": False,
                "elapsed_wall_s": time.time() - started,
            }

        lane_results = _run_node_lanes(
            nodes=validation["nodes"],
            kinds=validation["kinds"],
            profile=validation["profile"],
            tag=safe_tag,
            target_regimes=validation["target_dispatch_regimes"],
            max_parallel=int(max_parallel),
            max_attempts_per_cell=int(max_attempts_per_cell),
            seed_base=int(seed_base),
            init_cache_state=validation["init_cache_state"],
            role_specs=(
                ("calibration", calibration_repeats, 3),
                ("holdout", 1, 4),
            ),
        )
        if not all(row.get("ready") for row in lane_results):
            return {
                **manifest,
                "node_lock_rows": lock_rows,
                "deployment_rows": deployment_rows,
                "lane_results": _compact_lane_results(lane_results),
                "status": "TARGETED_MEASUREMENT_GAP",
                "pass": False,
                "elapsed_wall_s": time.time() - started,
            }
        extension_paths = _write_composite_artifacts(
            tag=safe_tag,
            profile=validation["profile"],
            lane_results=lane_results,
            expected_cell_ids=validation["expected_cell_ids"],
            target_regimes=validation["target_dispatch_regimes"],
            role_repeat_counts={
                "calibration": calibration_repeats,
                "holdout": 1,
            },
        )
        training_paths = [
            Path(path) for path in validation["training_paths"]
        ]
        base_calibration_paths = [
            Path(path) for path in validation["calibration_paths"]
        ]
        all_calibration_paths = [
            *base_calibration_paths,
            *extension_paths["calibration"],
        ]
        new_holdout_path = extension_paths["holdout"][0]
        gate = build_colocation_stochastic_lcb_gate(
            training_paths=training_paths,
            calibration_paths=all_calibration_paths,
            holdout_path=new_holdout_path,
            miscoverage_alpha=float(miscoverage_alpha),
            expected_cell_ids=validation["expected_cell_ids"],
        )
        gate_path = ARTIFACT_ROOT / (
            f"native_cpu_colocation_state_targeted_p"
            f"{validation['profile']}_extended_lcb_gate_{safe_tag}.json"
        )
        _atomic_write_json(gate_path, gate)
        evidence_paths = [
            *training_paths,
            *all_calibration_paths,
            new_holdout_path,
        ]
        gate_pass = bool(gate.get("pass"))
        return {
            **manifest,
            "node_lock_rows": lock_rows,
            "deployment_rows": deployment_rows,
            "lane_results": _compact_lane_results(lane_results),
            "training_paths": [str(path) for path in training_paths],
            "base_calibration_paths": [
                str(path) for path in base_calibration_paths
            ],
            "extension_calibration_paths": [
                str(path) for path in extension_paths["calibration"]
            ],
            "calibration_paths": [
                str(path) for path in all_calibration_paths
            ],
            "holdout_path": str(new_holdout_path),
            "gate_path": str(gate_path),
            "lcb_gate_path": str(gate_path),
            "gate_pass": gate_pass,
            "gate": gate,
            "evidence_sha256": _file_sha256_map(evidence_paths),
            "status": "PASS" if gate_pass else "GATE_FAIL",
            "pass": gate_pass,
            "elapsed_wall_s": time.time() - started,
            "claim_boundary": (
                "The original three training vectors and their conformal base "
                "models remain frozen. Nine additional complete calibration "
                "vectors are collected for every exact cell under the original "
                "pre-registered node regimes, followed by one new holdout vector. "
                "The inspected base holdout is excluded from both calibration and "
                "the final gate. No completion outcome participates in attempt "
                "admission or state matching."
            ),
        }
    finally:
        _release_node_locks(lock_handles)


def _validate_base_campaign(
    payload: Mapping[str, Any],
    *,
    base_path: Path,
) -> dict[str, Any]:
    errors = []
    if payload.get("campaign") != (
        "native_cpu_colocation_state_targeted_campaign"
    ):
        errors.append("campaign_identity_mismatch")
    if payload.get("measurement_protocol") != MEASUREMENT_PROTOCOL:
        errors.append("measurement_protocol_mismatch")
    nodes = tuple(str(value) for value in payload.get("nodes") or [])
    kinds = tuple(str(value) for value in payload.get("kinds") or [])
    if not nodes or not set(nodes).issubset(DEFAULT_NODES):
        errors.append("node_scope_invalid")
    if not kinds or not set(kinds).issubset(DEFAULT_KINDS):
        errors.append("kind_scope_invalid")
    expected_cell_ids = tuple(
        str(value) for value in payload.get("expected_cell_ids") or []
    )
    targets = {
        str(key): str(value)
        for key, value in (
            payload.get("target_dispatch_regimes") or {}
        ).items()
    }
    if set(targets) != set(nodes):
        errors.append("target_dispatch_regime_node_set_mismatch")
    training_paths = tuple(
        str(Path(path).resolve())
        for path in payload.get("training_paths") or []
    )
    calibration_paths = tuple(
        str(Path(path).resolve())
        for path in payload.get("calibration_paths") or []
    )
    if len(training_paths) != 3:
        errors.append("base_training_count_not_three")
    if len(calibration_paths) != 9:
        errors.append("base_calibration_count_not_nine")
    for path in (*training_paths, *calibration_paths):
        if not Path(path).is_file():
            errors.append(f"evidence_missing:{path}")
    if not expected_cell_ids:
        errors.append("expected_cell_set_empty")
    return {
        "ready": not errors,
        "errors": errors,
        "base_path": str(base_path),
        "profile": int(payload.get("profile") or 1),
        "nodes": nodes,
        "kinds": kinds,
        "expected_cell_ids": expected_cell_ids,
        "target_dispatch_regimes": targets,
        "training_paths": training_paths,
        "calibration_paths": calibration_paths,
        "init_cache_state": str(
            payload.get("init_cache_state") or "warm"
        ),
    }


def _compact_lane_results(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            key: value
            for key, value in row.items()
            if key != "selected"
        }
        for row in rows
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-launch", action="store_true")
    parser.add_argument("--base-campaign", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument(
        "--additional-calibration-repeats",
        type=int,
        default=DEFAULT_ADDITIONAL_CALIBRATION_REPEATS,
    )
    parser.add_argument("--max-parallel", type=int, default=6)
    parser.add_argument("--max-attempts-per-cell", type=int, default=60)
    parser.add_argument("--seed-base", type=int, default=1_727_000)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    result = run_state_targeted_colocation_extension(
        base_campaign_path=args.base_campaign,
        tag=args.tag,
        allow_launch=args.allow_launch,
        additional_calibration_repeats=(
            args.additional_calibration_repeats
        ),
        max_parallel=args.max_parallel,
        max_attempts_per_cell=args.max_attempts_per_cell,
        seed_base=args.seed_base,
        miscoverage_alpha=args.alpha,
    )
    _atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if not args.allow_launch or result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
