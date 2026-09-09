"""State-targeted p2/p4 campaign anchored to an exact p1 certificate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Iterable, Mapping

from .file_progress_completion_wrapper import MEASUREMENT_PROTOCOL
from .native_cpu_colocation_stochastic_calibration_campaign import (
    _acquire_node_locks,
    _release_node_locks,
    _validate_p1_prerequisite,
)
from .native_cpu_colocation_state_targeted_campaign import (
    ARTIFACT_ROOT,
    DEFAULT_KINDS,
    DEFAULT_NODES,
    CALIBRATION_REPEATS,
    HOLDOUT_REPEATS,
    TRAINING_REPEATS,
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


def run_state_targeted_profile_campaign(
    *,
    p1_campaign_path: Path,
    p1_gate_path: Path,
    tag: str,
    profile: int,
    allow_launch: bool = False,
    nodes: Iterable[str] = DEFAULT_NODES,
    kinds: Iterable[str] = DEFAULT_KINDS,
    max_parallel: int = 6,
    max_attempts_per_cell: int = 60,
    seed_base: int = 2_127_000,
    init_cache_state: str = "warm",
    miscoverage_alpha: float = 0.10,
) -> dict[str, Any]:
    """Run a full split-conformal profile under p1-certified regimes."""

    started = time.time()
    safe_tag = _safe_id(tag)
    selected_nodes = tuple(str(value) for value in nodes)
    selected_kinds = tuple(str(value) for value in kinds)
    selected_profile = int(profile)
    if selected_profile <= 1:
        raise ValueError("state-targeted profile campaign requires profile > 1")
    prerequisite = _validate_p1_prerequisite(
        campaign_path=Path(p1_campaign_path),
        gate_path=Path(p1_gate_path),
        required_nodes=selected_nodes,
        required_kinds=selected_kinds,
        required=True,
    )
    target_result = _target_regimes_from_p1_gate(
        Path(p1_gate_path),
        nodes=selected_nodes,
        kinds=selected_kinds,
    )
    target_regimes = target_result["target_dispatch_regimes"]
    manifest = {
        "campaign": "native_cpu_colocation_state_targeted_profile_campaign",
        "schema_version": 1,
        "tag": safe_tag,
        "allow_launch": bool(allow_launch),
        "measurement_protocol": MEASUREMENT_PROTOCOL,
        "profile": selected_profile,
        "profile_axis": "colocation_count",
        "nodes": list(selected_nodes),
        "kinds": list(selected_kinds),
        "training_repeat_count": TRAINING_REPEATS,
        "calibration_repeat_count": CALIBRATION_REPEATS,
        "holdout_repeat_count": HOLDOUT_REPEATS,
        "max_parallel": int(max_parallel),
        "max_attempts_per_cell": int(max_attempts_per_cell),
        "seed_base": int(seed_base),
        "init_cache_state": str(init_cache_state),
        "p1_prerequisite": prerequisite,
        "p1_target_extraction": target_result,
        "target_dispatch_regimes": target_regimes,
        "target_regime_source": "exact_passed_p1_gate",
        "performance_outcome_fields_used_for_selection": [],
        "legacy_scheduler_limits_bypassed": True,
        "user_task_control_operations": [],
    }
    if not prerequisite["ready"] or not target_result["ready"]:
        return {
            **manifest,
            "status": "P1_PREREQUISITE_GAP",
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
            selected_nodes,
            tag=safe_tag,
        )
        deployment_rows = _deploy_nodes(
            selected_nodes,
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
            nodes=selected_nodes,
            kinds=selected_kinds,
            profile=selected_profile,
            tag=safe_tag,
            target_regimes=target_regimes,
            max_parallel=int(max_parallel),
            max_attempts_per_cell=int(max_attempts_per_cell),
            seed_base=int(seed_base),
            init_cache_state=init_cache_state,
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
        expected_cell_ids = tuple(
            sorted(
                cell_id
                for lane in lane_results
                for cell_id in (
                    lane["selected"]["training"][1].keys()
                )
            )
        )
        paths = _write_composite_artifacts(
            tag=safe_tag,
            profile=selected_profile,
            lane_results=lane_results,
            expected_cell_ids=expected_cell_ids,
            target_regimes=target_regimes,
        )
        gate = build_colocation_stochastic_lcb_gate(
            training_paths=paths["training"],
            calibration_paths=paths["calibration"],
            holdout_path=paths["holdout"][0],
            miscoverage_alpha=float(miscoverage_alpha),
            expected_cell_ids=expected_cell_ids,
        )
        gate_path = ARTIFACT_ROOT / (
            f"native_cpu_colocation_state_targeted_p{selected_profile}_"
            f"lcb_gate_{safe_tag}.json"
        )
        _atomic_write_json(gate_path, gate)
        evidence_paths = [
            *paths["training"],
            *paths["calibration"],
            *paths["holdout"],
        ]
        gate_pass = bool(gate.get("pass"))
        return {
            **manifest,
            "node_lock_rows": lock_rows,
            "deployment_rows": deployment_rows,
            "lane_results": _compact_lane_results(lane_results),
            "expected_cell_ids": list(expected_cell_ids),
            "training_paths": [str(path) for path in paths["training"]],
            "calibration_paths": [
                str(path) for path in paths["calibration"]
            ],
            "holdout_path": str(paths["holdout"][0]),
            "gate_path": str(gate_path),
            "lcb_gate_path": str(gate_path),
            "gate_pass": gate_pass,
            "gate": gate,
            "evidence_sha256": _file_sha256_map(evidence_paths),
            "status": "PASS" if gate_pass else "GATE_FAIL",
            "pass": gate_pass,
            "elapsed_wall_s": time.time() - started,
            "claim_boundary": (
                "This profile certificate is limited to the exact workload, "
                "node, co-location count, total work, warm-cache protocol, and "
                "external-load regime certified by the referenced p1 evidence. "
                "Every state mismatch is deferred before child launch. The p1 "
                "lower service is a prerequisite, not reused as the p2/p4 rate."
            ),
        }
    finally:
        _release_node_locks(lock_handles)


def _target_regimes_from_p1_gate(
    path: Path,
    *,
    nodes: Iterable[str],
    kinds: Iterable[str],
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required_nodes = tuple(str(value) for value in nodes)
    required_kinds = tuple(str(value) for value in kinds)
    workload_keys = {
        "freqduet": "freqduet_cpu_native",
        "sumo": "sumo_eval_cpu_native",
    }
    required_keys = {workload_keys[kind] for kind in required_kinds}
    by_node: dict[str, list[Mapping[str, Any]]] = {
        node: [] for node in required_nodes
    }
    for row in payload.get("rows") or []:
        node = str(row.get("node") or "")
        if (
            node in by_node
            and row.get("workload_key") in required_keys
            and int(row.get("colocation_count") or 0) == 1
        ):
            by_node[node].append(row)
    errors = []
    targets = {}
    for node, rows in by_node.items():
        observed_keys = {str(row.get("workload_key") or "") for row in rows}
        regimes = {
            str(row.get("dispatch_external_cpu_regime") or "")
            for row in rows
        }
        if observed_keys != required_keys:
            errors.append(f"{node}:p1_workload_scope_mismatch")
        if len(regimes) != 1 or "" in regimes:
            errors.append(f"{node}:p1_regime_not_unique")
        if not all(
            row.get("ready")
            and row.get("lower_service_valid_on_holdout")
            and float(row.get("lower_service_units_per_s") or 0.0) > 0.0
            for row in rows
        ):
            errors.append(f"{node}:p1_lower_service_not_ready")
        if len(regimes) == 1 and "" not in regimes:
            targets[node] = next(iter(regimes))
    return {
        "ready": not errors and set(targets) == set(required_nodes),
        "errors": errors,
        "p1_gate_path": str(Path(path).resolve()),
        "target_dispatch_regimes": targets,
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
    parser.add_argument("--p1-campaign", type=Path, required=True)
    parser.add_argument("--p1-gate", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--profile", type=int, required=True)
    parser.add_argument("--nodes", default=",".join(DEFAULT_NODES))
    parser.add_argument("--kinds", default=",".join(DEFAULT_KINDS))
    parser.add_argument("--max-parallel", type=int, default=6)
    parser.add_argument("--max-attempts-per-cell", type=int, default=60)
    parser.add_argument("--seed-base", type=int, default=2_127_000)
    parser.add_argument(
        "--init-cache-state",
        choices=("cold", "warm", "unspecified"),
        default="warm",
    )
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    result = run_state_targeted_profile_campaign(
        p1_campaign_path=args.p1_campaign,
        p1_gate_path=args.p1_gate,
        tag=args.tag,
        profile=args.profile,
        allow_launch=args.allow_launch,
        nodes=_parse_csv(args.nodes),
        kinds=_parse_csv(args.kinds),
        max_parallel=args.max_parallel,
        max_attempts_per_cell=args.max_attempts_per_cell,
        seed_base=args.seed_base,
        init_cache_state=args.init_cache_state,
        miscoverage_alpha=args.alpha,
    )
    _atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if not args.allow_launch or result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
