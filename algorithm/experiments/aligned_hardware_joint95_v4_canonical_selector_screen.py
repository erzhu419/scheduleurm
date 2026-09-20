"""Freeze and run the canonical static screen against joint95 v4 only.

The registration reuses the already frozen seed-13041 queue roster and the
canonical K36/K72 versus K4096 design.  It reads neither the v4 results
directory nor the future v4 capacity result.  Runtime first requires a complete
v4 19+1 collection before it reads that capacity result or invokes a selector.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from algorithm.experiments import aligned_hardware_joint95_canonical_selector_screen as base
from algorithm.experiments import critical_gpu_joint95_v4_dispatcher as v4_dispatcher


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
STEM = "aligned_hardware_joint95_v4_canonical_selector_screen_20260913"
DEFAULT_REGISTRATION = ARTIFACTS / f"{STEM}_protocol.json"
DEFAULT_FIXED_SOURCES = ARTIFACTS / f"{STEM}_fixed_sources.json"
DEFAULT_RESULT = ARTIFACTS / f"{STEM}.json"
DEFAULT_CAPACITY_REGISTRATION = ARTIFACTS / (
    "aligned_hardware_joint95_v4_capacity_closure_20260913_protocol.json"
)
DEFAULT_CAPACITY_RESULT = ARTIFACTS / (
    "aligned_hardware_joint95_v4_capacity_closure_20260913.json"
)
DEFAULT_COLLECTION_MANIFEST = ARTIFACTS / (
    "critical_gpu_joint95_v4_collection_manifest_20260913.json"
)
DEFAULT_RESULTS_DIR = ARTIFACTS / "critical_gpu_joint95_v4_results"
DEFAULT_REUSED_ROSTER = base.DEFAULT_ROSTER
DEFAULT_LEGACY_REGISTRATION = base.DEFAULT_REGISTRATION

EXPERIMENT = "aligned_hardware_joint95_v4_canonical_selector_screen"
GATE = "aligned_hardware_joint95_v4_canonical_selector_screen_runner"
REGISTRATION_STATUS = "REGISTERED_AWAITING_JOINT95_V4_CAPACITY_CLOSURE"
CAPACITY_EXPERIMENT = "aligned_hardware_joint95_v4_capacity_closure"
CAPACITY_GATE = "aligned_hardware_joint95_v4_capacity_closure_runner"
PASS_DECISION = base.PASS_DECISION
THIS_SOURCE = (
    "algorithm/experiments/aligned_hardware_joint95_v4_canonical_selector_screen.py"
)
FIXED_SOURCE_PATHS = (
    "algorithm/canonical_count_space_bnb_frame_candidates.py",
    "algorithm/experiments/aligned_hardware_frame_model.py",
    "algorithm/experiments/aligned_hardware_lcb_frame_envelope_shadow.py",
    "algorithm/experiments/aligned_hardware_joint95_capacity_closure.py",
    "algorithm/experiments/aligned_hardware_joint95_canonical_selector_screen.py",
    "algorithm/experiments/aligned_hardware_joint95_static_selector_screen.py",
    "algorithm/experiments/aligned_hardware_joint95_v4_capacity_closure.py",
    "algorithm/experiments/critical_gpu_joint95_v4_dispatcher.py",
    THIS_SOURCE,
)


class V4CanonicalScreenRefusal(RuntimeError):
    """The v4 data gate or frozen canonical-screen contract is not satisfied."""


def _json(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V4CanonicalScreenRefusal(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise V4CanonicalScreenRefusal(f"JSON root must be an object: {path}")
    return value


def _write_new(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(dict(payload), stream, indent=2, sort_keys=True)
        stream.write("\n")


def _capacity_contract(source: Mapping[str, Any]) -> dict[str, Any]:
    paths = source.get("source_paths") or {}
    scope = source.get("registration_scope") or {}
    capacity = source.get("fixed_capacity_contract") or {}
    if (
        source.get("schema_version") != 1
        or source.get("gate") != CAPACITY_GATE
        or source.get("experiment") != CAPACITY_EXPERIMENT
        or source.get("status") != "REGISTERED_AWAITING_JOINT95_V4_DATA"
        or source.get("registered") is not True
        or Path(str(paths.get("collection_manifest") or "")).resolve()
        != DEFAULT_COLLECTION_MANIFEST.resolve()
        or Path(str(paths.get("joint_results_directory") or "")).resolve()
        != DEFAULT_RESULTS_DIR.resolve()
        or capacity.get("capacity_lanes") != list(base.LANE_COUNTS)
        or capacity.get("loads") != list(base.LOADS)
        or scope.get("v4_joint_visit_values_read") is not False
        or scope.get("v4_joint_results_directory_read") is not False
    ):
        raise V4CanonicalScreenRefusal("joint95 v4 capacity registration is invalid")
    return _json(source)


def _reused_roster_contract() -> dict[str, Any]:
    legacy_protocol = _read_object(DEFAULT_LEGACY_REGISTRATION)
    base._validate_protocol(legacy_protocol)
    roster = _read_object(DEFAULT_REUSED_ROSTER)
    queues = base._validate_roster(roster, legacy_protocol)
    if (
        roster.get("seed") != base.ROSTER_SEED
        or len(queues) != base.ROSTER_SIZE
        or base.ROSTER_SEED != 13041
    ):
        raise V4CanonicalScreenRefusal("reused canonical roster differs")
    return {
        "path": str(DEFAULT_REUSED_ROSTER.resolve()),
        "seed": base.ROSTER_SEED,
        "size": base.ROSTER_SIZE,
        "candidate_draws": roster.get("candidate_draws"),
        "coordinate_ranges_inclusive": _json(roster.get("coordinate_ranges_inclusive")),
        "queues": [list(queue) for queue in queues],
        "reuse_role": "fixed design object only; no v1 joint observation is reused",
    }


def _load_reused_queues(contract: Mapping[str, Any]) -> tuple[tuple[int, ...], ...]:
    roster = _read_object(Path(str(contract.get("path") or "")))
    queues = tuple(tuple(int(value) for value in row) for row in roster.get("queues") or ())
    if (
        roster.get("seed") != 13041
        or contract.get("seed") != 13041
        or contract.get("size") != 512
        or roster.get("candidate_draws") != contract.get("candidate_draws")
        or roster.get("coordinate_ranges_inclusive")
        != contract.get("coordinate_ranges_inclusive")
        or [list(queue) for queue in queues] != contract.get("queues")
        or len(queues) != 512
        or len(set(queues)) != 512
    ):
        raise V4CanonicalScreenRefusal("reused seed-13041 roster changed")
    return queues


def _current_fixed_sources() -> dict[str, str]:
    return {name: (ROOT / name).read_text(encoding="utf-8") for name in FIXED_SOURCE_PATHS}


def _protocol(
    *,
    capacity_registration: Path,
    capacity_result: Path,
    capacity_contract: Mapping[str, Any],
    roster_contract: Mapping[str, Any],
) -> dict[str, Any]:
    audit = base.legacy._uniform_margin_scaling_audit(capacity_contract)
    if (
        audit.get("all_nine_action_durations_uniformly_scaled") is not True
        or audit.get("issue_count") != 0
        or audit.get("joint_result_values_read") is not False
    ):
        raise V4CanonicalScreenRefusal("v3 common-margin scaling contract is invalid")
    return {
        "schema_version": 1,
        "gate": GATE,
        "experiment": EXPERIMENT,
        "status": REGISTRATION_STATUS,
        "registered": True,
        "source_paths": {
            "capacity_registration": str(Path(capacity_registration).resolve()),
            "capacity_result": str(Path(capacity_result).resolve()),
            "collection_manifest": str(DEFAULT_COLLECTION_MANIFEST.resolve()),
            "joint_results_directory": str(DEFAULT_RESULTS_DIR.resolve()),
            "reused_seed13041_roster": str(DEFAULT_REUSED_ROSTER.resolve()),
        },
        "capacity_registration_contract": _json(capacity_contract),
        "uniform_margin_scaling_audit": audit,
        "queue_roster_contract": _json(roster_contract),
        "selector_contract": {
            "family": "canonical-v3 cold count-space branch-and-bound selector",
            "fixed_budget_by_lane": {"8": 36, "16": 72},
            "cold_canonical_reference_budget": 4096,
            "lanes": [8, 16],
            "physical_capacity_normalized_loads": [0.5, 0.8, 0.95],
            "registered_queue_count": 512,
            "static_only": True,
        },
        "v4_data_gate": {
            "calibration_waves": list(range(74, 93)),
            "holdout_wave": 93,
            "required_collection_status": "COMPLETE",
            "required_successful_calibration_count": 19,
            "holdout_release_required": True,
            "holdout_success_required": True,
        },
        "v1_contribution": {
            "joint_units": 0,
            "joint_scores": 0,
            "capacity_rows": 0,
            "holdout_rows": 0,
            "result_directory_read_during_registration": False,
        },
        "v2_contribution": {
            "joint_units": 0,
            "joint_scores": 0,
            "capacity_rows": 0,
            "holdout_rows": 0,
            "result_directory_read_during_registration": False,
        },
        "v3_contribution": {
            "joint_units": 0,
            "joint_scores": 0,
            "capacity_rows": 0,
            "holdout_rows": 0,
            "result_directory_read_during_registration": False,
        },
        "decision_rule": dict(base.DECISION_RULE),
        "claim_boundary": base.CLAIM_BOUNDARY,
        "registration_scope": {
            "v4_capacity_protocol_read": True,
            "v4_results_directory_read": False,
            "v4_joint_visit_values_read": False,
            "v4_capacity_result_read": False,
            "selector_invoked": False,
            "exact_reference_invoked": False,
            "roster_reused": True,
            "roster_rng_initializations": 0,
            "roster_rng_draws": 0,
            "paper_edited": False,
        },
    }


def register(
    registration: Path = DEFAULT_REGISTRATION,
    fixed_sources: Path = DEFAULT_FIXED_SOURCES,
    *,
    capacity_registration: Path = DEFAULT_CAPACITY_REGISTRATION,
    capacity_result: Path = DEFAULT_CAPACITY_RESULT,
) -> dict[str, Any]:
    if any(Path(path).exists() for path in (registration, fixed_sources, DEFAULT_RESULT)):
        raise V4CanonicalScreenRefusal("v4 canonical-screen artifacts already exist")
    capacity_contract = _capacity_contract(_read_object(Path(capacity_registration)))
    protocol = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        **_protocol(
            capacity_registration=capacity_registration,
            capacity_result=capacity_result,
            capacity_contract=capacity_contract,
            roster_contract=_reused_roster_contract(),
        ),
    }
    _write_new(Path(registration), protocol)
    _write_new(Path(fixed_sources), _current_fixed_sources())
    return protocol


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    selector = protocol.get("selector_contract") or {}
    gate = protocol.get("v4_data_gate") or {}
    v1 = protocol.get("v1_contribution") or {}
    v2 = protocol.get("v2_contribution") or {}
    v3 = protocol.get("v3_contribution") or {}
    if (
        protocol.get("schema_version") != 1
        or protocol.get("gate") != GATE
        or protocol.get("experiment") != EXPERIMENT
        or protocol.get("status") != REGISTRATION_STATUS
        or protocol.get("registered") is not True
        or selector.get("fixed_budget_by_lane") != {"8": 36, "16": 72}
        or selector.get("cold_canonical_reference_budget") != 4096
        or gate.get("calibration_waves") != list(range(74, 93))
        or gate.get("holdout_wave") != 93
        or any(v1.get(key) != 0 for key in ("joint_units", "joint_scores", "capacity_rows", "holdout_rows"))
        or any(v2.get(key) != 0 for key in ("joint_units", "joint_scores", "capacity_rows", "holdout_rows"))
        or any(v3.get(key) != 0 for key in ("joint_units", "joint_scores", "capacity_rows", "holdout_rows"))
    ):
        raise V4CanonicalScreenRefusal("v4 canonical-screen registration changed")


def require_v4_complete(protocol: Mapping[str, Any]) -> dict[str, Any]:
    paths = protocol.get("source_paths") or {}
    manifest = v4_dispatcher.load_registered_manifest(Path(paths["collection_manifest"]))
    state = v4_dispatcher.collection_status(
        manifest, Path(paths["joint_results_directory"])
    )
    complete = (
        state.get("status") == "COMPLETE"
        and state.get("errors") == []
        and state.get("successful_calibration_count") == 19
        and state.get("holdout_released") is True
        and state.get("holdout_success") is True
        and state.get("next_wave") is None
    )
    if not complete:
        raise V4CanonicalScreenRefusal(
            "joint95 v4 collection is incomplete at the data gate: "
            f"status={state.get('status')!r}, calibration="
            f"{state.get('successful_calibration_count')!r}"
        )
    return state


def run_registered(
    registration: Path = DEFAULT_REGISTRATION,
    fixed_sources: Path = DEFAULT_FIXED_SOURCES,
    output: Path = DEFAULT_RESULT,
) -> dict[str, Any]:
    output = Path(output)
    if output.exists():
        raise V4CanonicalScreenRefusal(f"result already exists: {output}")
    protocol = _read_object(Path(registration))
    _validate_protocol(protocol)
    if _read_object(Path(fixed_sources)) != _current_fixed_sources():
        raise V4CanonicalScreenRefusal("registered v4 canonical-screen sources changed")
    collection = require_v4_complete(protocol)

    paths = protocol["source_paths"]
    capacity_protocol = _read_object(Path(paths["capacity_registration"]))
    if _capacity_contract(capacity_protocol) != protocol["capacity_registration_contract"]:
        raise V4CanonicalScreenRefusal("v4 capacity registration changed")
    capacity = _read_object(Path(paths["capacity_result"]))
    if (
        capacity.get("gate") != CAPACITY_GATE
        or capacity.get("experiment") != CAPACITY_EXPERIMENT
        or capacity.get("status") != "PASS"
        or capacity.get("protocol") != capacity_protocol
        or capacity.get("capacity_established") is not True
        or capacity.get("joint95_v4_audit", {}).get("holdout_covered") is not True
    ):
        raise V4CanonicalScreenRefusal("v4 capacity closure has not established capacity")
    snapshot = (capacity.get("physical_complete_frame_envelope") or {}).get("model")
    model = base.legacy.closure._model_from_snapshot(snapshot or {})
    scale = float((capacity.get("capacity") or {}).get("common_capacity_scale", 0.0))
    if len(model.options) != base.OPTION_COUNT or not math.isfinite(scale) or scale <= 0.0:
        raise V4CanonicalScreenRefusal("v3 physical envelope is invalid")
    queues = _load_reused_queues(protocol["queue_roster_contract"])
    capacity_rows = base.legacy._physical_capacity_rows(capacity_protocol, model, scale)

    started = time.perf_counter()
    lane_rows = []
    lane_checks = []
    for lanes in base.LANE_COUNTS:
        row, checks = base._screen_lane(model, queues, lanes, capacity_rows[str(lanes)])
        lane_rows.append(row)
        lane_checks.append(checks)
    candidate_total = base._new_work()
    reference_total = base._new_work()
    for row in lane_rows:
        for field in base.WORK_FIELDS:
            candidate_total[field] += row["candidate_actual_work"][field]
            reference_total[field] += row["cold_canonical_reference_actual_work"][field]
    comparisons = {
        field: {
            "candidate": candidate_total[field],
            "cold_canonical_reference": reference_total[field],
            "strictly_lower": candidate_total[field] < reference_total[field],
        }
        for field in ("lp_calls", "lp_matrix_coefficients", "candidate_evaluations")
    }
    aggregate_compute = all(row["strictly_lower"] for row in comparisons.values())
    checks = {
        "joint95_v4_complete": True,
        "v1_joint_contribution_is_zero": True,
        "reused_seed13041_roster_has_512_queues": len(queues) == 512,
        "every_lane_execution_check_passes": all(all(row.values()) for row in lane_checks),
        "selector_call_count_is_exact": sum(row["state_count"] for row in lane_rows) == 1024,
        "no_arrivals_or_trajectories": True,
    }
    decision = base._decision(all(checks.values()), lane_rows, aggregate_compute)
    report = {
        "schema_version": 1,
        "gate": GATE,
        "experiment": EXPERIMENT,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "protocol": protocol,
        "checks": checks,
        "v4_collection_status": collection,
        "v1_contribution": protocol["v1_contribution"],
        "v2_contribution": protocol["v2_contribution"],
        "v3_contribution": protocol["v3_contribution"],
        "physical_complete_frame_envelope": capacity["physical_complete_frame_envelope"],
        "physical_capacity_normalized_loads": capacity_rows,
        "lane_summaries": lane_rows,
        "aggregate_actual_compute": {
            "candidate": candidate_total,
            "cold_canonical_reference": reference_total,
            "comparison": comparisons,
            "gate_passed": aggregate_compute,
        },
        "scientific_decision": decision,
        "physical_trajectory_advance": decision == PASS_DECISION,
        "baseline_trajectory_advance": decision == PASS_DECISION,
        "claim_boundary": base.CLAIM_BOUNDARY,
        "timing": {"wall_s": time.perf_counter() - started, "scientific_gate": False},
    }
    _write_new(output, report)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--register", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    parser.add_argument("--fixed-sources", type=Path, default=DEFAULT_FIXED_SOURCES)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--capacity-registration", type=Path, default=DEFAULT_CAPACITY_REGISTRATION)
    parser.add_argument("--capacity-result", type=Path, default=DEFAULT_CAPACITY_RESULT)
    args = parser.parse_args(argv)
    try:
        payload = (
            register(
                args.registration,
                args.fixed_sources,
                capacity_registration=args.capacity_registration,
                capacity_result=args.capacity_result,
            )
            if args.register
            else run_registered(args.registration, args.fixed_sources, args.output)
        )
    except V4CanonicalScreenRefusal as exc:
        raise SystemExit(f"REFUSED: {exc}") from exc
    print(json.dumps({"status": payload["status"], "experiment": payload["experiment"]}, indent=2))


if __name__ == "__main__":
    main()
