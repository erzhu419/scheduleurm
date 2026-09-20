"""Post-result static selector diagnostic on the frozen joint95-v5 envelope.

This sidecar preserves the negative v5 point-boundary capacity decision.  It
uses the final physical envelope's own capacity boundary to define loads, then
compares the already frozen K36/K72 canonical searches with cold K4096 on the
already frozen seed-13041 queue roster.  Registration reads the completed v5
result, but invokes no selector and builds no trajectory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from algorithm.experiments import (
    aligned_hardware_joint95_canonical_selector_screen as canonical,
)
from algorithm.experiments import (
    aligned_hardware_joint95_v5_canonical_selector_screen as parent,
)
from algorithm.experiments import (
    aligned_hardware_joint95_v5_capacity_closure as capacity,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
STEM = "aligned_hardware_joint95_v5_envelope_normalized_static_diagnostic_20260919"
DEFAULT_REGISTRATION = ARTIFACTS / f"{STEM}_protocol.json"
DEFAULT_FIXED_SOURCES = ARTIFACTS / f"{STEM}_fixed_sources.json"
DEFAULT_RESULT = ARTIFACTS / f"{STEM}.json"
DEFAULT_CAPACITY_REGISTRATION = capacity.DEFAULT_REGISTRATION
DEFAULT_CAPACITY_RESULT = capacity.DEFAULT_RESULT
DEFAULT_PARENT_REGISTRATION = parent.DEFAULT_REGISTRATION
DEFAULT_PARENT_FIXED_SOURCES = parent.DEFAULT_FIXED_SOURCES

EXPERIMENT = "aligned_hardware_joint95_v5_envelope_normalized_static_diagnostic"
GATE = "aligned_hardware_joint95_v5_envelope_normalized_static_diagnostic_runner"
REGISTRATION_STATUS = "REGISTERED_POSTRESULT_AWAITING_STATIC_DIAGNOSTIC"
NEGATIVE_CAPACITY_DECISION = "JOINT95_NO_POSITIVE_CAPACITY_MARGIN_STOP"
PASS_DECISION = canonical.PASS_DECISION
DECISION_RULE = canonical.DECISION_RULE
NEXT_REQUIRED_EVIDENCE = (
    "register and run a same-envelope static selector screen over a fully "
    "materialized fixed queue roster before any new trajectory"
)
CLAIM_BOUNDARY = (
    "A post-result development diagnostic on the immutable joint95-v5 physical "
    "complete-frame envelope and the previously frozen seed-13041 finite queue "
    "roster. Loads 0.5, 0.8 and 0.95 are fractions of that conservative "
    "envelope's own common capacity boundary. A pass is only a necessary input "
    "to a separately registered trajectory experiment; it does not reverse the "
    "v5 point-boundary capacity failure, establish capacity at the old loads, or "
    "make a trajectory, stability, deployment, or physical multi-GPU claim."
)
PROHIBITED = (
    "changing the joint95-v5 result or its negative capacity decision; replacing "
    "the final physical envelope; changing the seed-13041 roster, K36/K72/K4096 "
    "budgets, or normalized loads after registration; run-time RNG; arrivals or "
    "trajectories; server pulls; and paper edits"
)
THIS_SOURCE = (
    "algorithm/experiments/"
    "aligned_hardware_joint95_v5_envelope_normalized_static_diagnostic.py"
)
FIXED_SOURCE_PATHS = (
    "algorithm/canonical_count_space_bnb_frame_candidates.py",
    "algorithm/experiments/aligned_hardware_count_space_bnb_scaling_screen.py",
    "algorithm/experiments/aligned_hardware_frame_model.py",
    "algorithm/experiments/aligned_hardware_lcb_frame_envelope_shadow.py",
    "algorithm/experiments/aligned_hardware_joint95_capacity_closure.py",
    "algorithm/experiments/aligned_hardware_joint95_static_selector_screen.py",
    "algorithm/experiments/aligned_hardware_joint95_canonical_selector_screen.py",
    "algorithm/experiments/aligned_hardware_joint95_v5_capacity_closure.py",
    "algorithm/experiments/aligned_hardware_joint95_v5_canonical_selector_screen.py",
    THIS_SOURCE,
)


class EnvelopeDiagnosticRefusal(RuntimeError):
    """The registered post-result diagnostic contract is not satisfied."""


def _json(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EnvelopeDiagnosticRefusal(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EnvelopeDiagnosticRefusal(f"JSON root must be an object: {path}")
    return value


def _write_new(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(dict(payload), stream, indent=2, sort_keys=True)
        stream.write("\n")


def _current_fixed_sources() -> dict[str, str]:
    return {name: (ROOT / name).read_text(encoding="utf-8") for name in FIXED_SOURCE_PATHS}


def _parent_design_contract(source: Mapping[str, Any]) -> dict[str, Any]:
    try:
        parent._validate_protocol(source)
    except parent.V5CanonicalScreenRefusal as exc:
        raise EnvelopeDiagnosticRefusal(str(exc)) from exc
    selector = source.get("selector_contract") or {}
    roster = source.get("queue_roster_contract") or {}
    if (
        selector.get("fixed_budget_by_lane") != {"8": 36, "16": 72}
        or selector.get("cold_canonical_reference_budget") != 4096
        or selector.get("lanes") != [8, 16]
        or selector.get("physical_capacity_normalized_loads") != [0.5, 0.8, 0.95]
        or selector.get("registered_queue_count") != 512
        or roster.get("seed") != 13041
        or roster.get("size") != 512
    ):
        raise EnvelopeDiagnosticRefusal("frozen v5 selector design differs")
    queues = parent._load_reused_queues(roster)
    if len(queues) != 512:
        raise EnvelopeDiagnosticRefusal("frozen queue roster does not contain 512 states")
    return {
        "parent_protocol": _json(source),
        "selector_contract": _json(selector),
        "queue_roster_contract": _json(roster),
    }


def _validated_capacity_snapshot(
    result: Mapping[str, Any], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        registered = parent._capacity_contract(protocol)
    except parent.V5CanonicalScreenRefusal as exc:
        raise EnvelopeDiagnosticRefusal(str(exc)) from exc
    audit = result.get("joint95_v5_audit") or {}
    accounting = result.get("joint_data_accounting") or {}
    physical = result.get("physical_complete_frame_envelope") or {}
    candidate = result.get("candidate_support") or {}
    observed_capacity = result.get("capacity") or {}
    expected_accounting = {
        "v5_calibration_units": 19,
        "v5_holdout_units": 1,
        "v1_joint_units": 0,
        "v1_joint_scores": 0,
        "v2_joint_units": 0,
        "v2_joint_scores": 0,
        "v3_joint_units": 0,
        "v3_joint_scores": 0,
        "v4_joint_units": 0,
        "v4_joint_scores": 0,
    }
    if (
        result.get("schema_version") != 1
        or result.get("gate") != capacity.GATE
        or result.get("experiment") != capacity.EXPERIMENT
        or result.get("status") != "PASS"
        or result.get("protocol") != protocol
        or result.get("scientific_decision") != NEGATIVE_CAPACITY_DECISION
        or result.get("capacity_established") is not False
        or result.get("closure_passed") is not False
        or audit.get("holdout_covered") is not True
        or accounting != expected_accounting
        or candidate.get("status") != "CANDIDATE_SUPPORT_EVIDENCE_UNAVAILABLE"
        or candidate.get("next_required_evidence") != NEXT_REQUIRED_EVIDENCE
        or observed_capacity.get("all_registered_capacity_margins_positive") is not False
    ):
        raise EnvelopeDiagnosticRefusal("formal v5 negative capacity artifact differs")
    try:
        model = capacity.base._model_from_snapshot(physical.get("model") or {})
        scale = float(observed_capacity["common_capacity_scale"])
    except (KeyError, TypeError, ValueError, capacity.ClosureRefusal) as exc:
        raise EnvelopeDiagnosticRefusal("v5 physical envelope is invalid") from exc
    recomputed = capacity.base._capacity_analysis(protocol, model)
    if observed_capacity != recomputed:
        raise EnvelopeDiagnosticRefusal("v5 capacity analysis does not reproduce")
    if (
        len(model.options) != canonical.OPTION_COUNT
        or not math.isfinite(scale)
        or scale <= 0.0
    ):
        raise EnvelopeDiagnosticRefusal("v5 common physical capacity boundary is invalid")
    return {
        "capacity_registration_contract": registered,
        "capacity_result_snapshot": _json(result),
        "common_capacity_scale": scale,
    }


def build_registration(
    *,
    capacity_registration: Path = DEFAULT_CAPACITY_REGISTRATION,
    capacity_result: Path = DEFAULT_CAPACITY_RESULT,
    parent_registration: Path = DEFAULT_PARENT_REGISTRATION,
    parent_fixed_sources: Path = DEFAULT_PARENT_FIXED_SOURCES,
) -> dict[str, Any]:
    capacity_protocol = _read_object(Path(capacity_registration))
    result = _read_object(Path(capacity_result))
    parent_protocol = _read_object(Path(parent_registration))
    design = _parent_design_contract(parent_protocol)
    if _read_object(Path(parent_fixed_sources)) != parent._current_fixed_sources():
        raise EnvelopeDiagnosticRefusal("frozen parent selector sources changed")
    try:
        collection = parent.require_v5_complete(parent_protocol)
    except parent.V5CanonicalScreenRefusal as exc:
        raise EnvelopeDiagnosticRefusal(str(exc)) from exc
    frozen = _validated_capacity_snapshot(result, capacity_protocol)
    scale = float(frozen["common_capacity_scale"])
    normalized = [
        {
            "rho": load,
            "point_boundary_equivalent_load": load * scale,
            "formula": "rho * common_capacity_scale * (R/2) * lambda_point_R2",
        }
        for load in canonical.LOADS
    ]
    return {
        "schema_version": 1,
        "gate": GATE,
        "experiment": EXPERIMENT,
        "status": REGISTRATION_STATUS,
        "registered": True,
        "post_result_development": True,
        "source_paths": {
            "capacity_registration": str(Path(capacity_registration).resolve()),
            "capacity_result": str(Path(capacity_result).resolve()),
            "parent_selector_registration": str(Path(parent_registration).resolve()),
            "parent_selector_fixed_sources": str(Path(parent_fixed_sources).resolve()),
        },
        **frozen,
        "parent_design_contract": design,
        "normalized_load_contract": normalized,
        "v5_collection_snapshot": {
            "status": collection.get("status"),
            "successful_calibration_count": collection.get(
                "successful_calibration_count"
            ),
            "holdout_released": collection.get("holdout_released"),
            "holdout_success": collection.get("holdout_success"),
            "errors": collection.get("errors"),
        },
        "decision_rule": dict(DECISION_RULE),
        "claim_boundary": CLAIM_BOUNDARY,
        "prohibited": PROHIBITED,
        "registration_scope": {
            "v5_capacity_result_read": True,
            "v5_joint_results_read_for_completion_gate": True,
            "post_result_design_declared": True,
            "selector_invoked": False,
            "exact_reference_invoked": False,
            "trajectory_built": False,
            "arrival_generated": False,
            "rng_draws": 0,
            "server_pulls": 0,
            "paper_edited": False,
        },
    }


def register(
    registration: Path = DEFAULT_REGISTRATION,
    fixed_sources: Path = DEFAULT_FIXED_SOURCES,
    **kwargs: Any,
) -> dict[str, Any]:
    if any(Path(path).exists() for path in (registration, fixed_sources, DEFAULT_RESULT)):
        raise EnvelopeDiagnosticRefusal("diagnostic artifact already exists")
    protocol = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        **build_registration(**kwargs),
    }
    _write_new(Path(fixed_sources), _current_fixed_sources())
    _write_new(Path(registration), protocol)
    return protocol


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    scope = protocol.get("registration_scope") or {}
    design = protocol.get("parent_design_contract") or {}
    if (
        protocol.get("schema_version") != 1
        or protocol.get("gate") != GATE
        or protocol.get("experiment") != EXPERIMENT
        or protocol.get("status") != REGISTRATION_STATUS
        or protocol.get("registered") is not True
        or protocol.get("post_result_development") is not True
        or protocol.get("decision_rule") != DECISION_RULE
        or protocol.get("claim_boundary") != CLAIM_BOUNDARY
        or protocol.get("prohibited") != PROHIBITED
        or scope.get("v5_capacity_result_read") is not True
        or scope.get("selector_invoked") is not False
        or scope.get("trajectory_built") is not False
        or scope.get("rng_draws") != 0
        or (design.get("selector_contract") or {}).get("fixed_budget_by_lane")
        != {"8": 36, "16": 72}
        or (design.get("queue_roster_contract") or {}).get("size") != 512
    ):
        raise EnvelopeDiagnosticRefusal("registered diagnostic protocol changed")
    snapshot = protocol.get("capacity_result_snapshot") or {}
    if (
        snapshot.get("scientific_decision") != NEGATIVE_CAPACITY_DECISION
        or snapshot.get("capacity_established") is not False
        or snapshot.get("closure_passed") is not False
    ):
        raise EnvelopeDiagnosticRefusal("protocol does not retain the v5 negative result")
    try:
        scale = float(protocol["common_capacity_scale"])
    except (KeyError, TypeError, ValueError) as exc:
        raise EnvelopeDiagnosticRefusal("registered common capacity scale is invalid") from exc
    expected_loads = [
        {
            "rho": load,
            "point_boundary_equivalent_load": load * scale,
            "formula": "rho * common_capacity_scale * (R/2) * lambda_point_R2",
        }
        for load in canonical.LOADS
    ]
    if (
        not math.isfinite(scale)
        or scale <= 0.0
        or protocol.get("normalized_load_contract") != expected_loads
    ):
        raise EnvelopeDiagnosticRefusal("registered normalized load contract changed")


def run_registered(
    registration: Path = DEFAULT_REGISTRATION,
    fixed_sources: Path = DEFAULT_FIXED_SOURCES,
    output: Path = DEFAULT_RESULT,
) -> dict[str, Any]:
    output = Path(output)
    if output.exists():
        raise EnvelopeDiagnosticRefusal(f"diagnostic result already exists: {output}")
    protocol = _read_object(Path(registration))
    _validate_protocol(protocol)
    if _read_object(Path(fixed_sources)) != _current_fixed_sources():
        raise EnvelopeDiagnosticRefusal("registered diagnostic sources changed")
    paths = protocol["source_paths"]
    if (
        _read_object(Path(paths["parent_selector_fixed_sources"]))
        != parent._current_fixed_sources()
    ):
        raise EnvelopeDiagnosticRefusal("frozen parent selector sources changed")
    parent_protocol = _read_object(Path(paths["parent_selector_registration"]))
    if _parent_design_contract(parent_protocol) != protocol["parent_design_contract"]:
        raise EnvelopeDiagnosticRefusal("frozen parent selector design changed")
    try:
        collection = parent.require_v5_complete(parent_protocol)
    except parent.V5CanonicalScreenRefusal as exc:
        raise EnvelopeDiagnosticRefusal(str(exc)) from exc
    capacity_protocol = _read_object(Path(paths["capacity_registration"]))
    result = _read_object(Path(paths["capacity_result"]))
    frozen = _validated_capacity_snapshot(result, capacity_protocol)
    if (
        frozen["capacity_registration_contract"]
        != protocol["capacity_registration_contract"]
        or frozen["capacity_result_snapshot"] != protocol["capacity_result_snapshot"]
        or frozen["common_capacity_scale"] != protocol["common_capacity_scale"]
    ):
        raise EnvelopeDiagnosticRefusal("frozen v5 capacity evidence changed")
    model = capacity.base._model_from_snapshot(
        (result["physical_complete_frame_envelope"] or {})["model"]
    )
    scale = float(protocol["common_capacity_scale"])
    capacity_rows = canonical.legacy._physical_capacity_rows(
        capacity_protocol, model, scale
    )
    normalized_rows_positive = all(
        row.get("positive_effective_slack") is True
        for rows in capacity_rows.values()
        for row in rows
    )
    queues = parent._load_reused_queues(
        protocol["parent_design_contract"]["queue_roster_contract"]
    )

    started = time.perf_counter()
    lane_rows = []
    lane_checks = []
    for lanes in canonical.LANE_COUNTS:
        row, checks = canonical._screen_lane(
            model, queues, lanes, capacity_rows[str(lanes)]
        )
        lane_rows.append(row)
        lane_checks.append(checks)
    candidate_total = canonical._new_work()
    reference_total = canonical._new_work()
    for row in lane_rows:
        for field in canonical.WORK_FIELDS:
            candidate_total[field] += row["candidate_actual_work"][field]
            reference_total[field] += row[
                "cold_canonical_reference_actual_work"
            ][field]
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
        "v5_collection_complete": collection.get("status") == "COMPLETE",
        "original_negative_capacity_decision_retained": True,
        "original_capacity_established_remains_false": True,
        "normalized_capacity_rows_positive": normalized_rows_positive,
        "seed13041_roster_has_512_queues": len(queues) == 512,
        "every_lane_execution_check_passes": all(
            all(row.values()) for row in lane_checks
        ),
        "selector_call_count_is_exact": sum(row["state_count"] for row in lane_rows)
        == 1024,
        "no_arrivals_or_trajectories": True,
        "v1_v4_joint_contribution_is_zero": all(
            result["joint_data_accounting"].get(f"v{version}_joint_units") == 0
            and result["joint_data_accounting"].get(f"v{version}_joint_scores") == 0
            for version in range(1, 5)
        ),
    }
    execution_passed = all(checks.values())
    decision = canonical._decision(execution_passed, lane_rows, aggregate_compute)
    report = {
        "schema_version": 1,
        "gate": GATE,
        "experiment": EXPERIMENT,
        "status": "PASS" if execution_passed else "FAIL",
        "protocol": protocol,
        "checks": checks,
        "v5_collection_status": collection,
        "retained_v5_capacity_decision": NEGATIVE_CAPACITY_DECISION,
        "retained_v5_capacity_established": False,
        "retained_v5_closure_passed": False,
        "physical_complete_frame_envelope": result[
            "physical_complete_frame_envelope"
        ],
        "common_capacity_scale": scale,
        "normalized_load_contract": protocol["normalized_load_contract"],
        "physical_envelope_capacity_normalized_loads": capacity_rows,
        "lane_summaries": lane_rows,
        "aggregate_actual_compute": {
            "candidate": candidate_total,
            "cold_canonical_reference": reference_total,
            "comparison": comparisons,
            "gate_passed": aggregate_compute,
        },
        "scientific_decision": decision,
        "trajectory_registration_prerequisite_passed": decision == PASS_DECISION,
        "capacity_closure_reclassified": False,
        "trajectory_runs": 0,
        "arrival_generations": 0,
        "rng_draws": 0,
        "server_pulls": 0,
        "claim_boundary": CLAIM_BOUNDARY,
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
    args = parser.parse_args(argv)
    try:
        payload = (
            register(args.registration, args.fixed_sources)
            if args.register
            else run_registered(args.registration, args.fixed_sources, args.output)
        )
    except EnvelopeDiagnosticRefusal as exc:
        raise SystemExit(f"REFUSED: {exc}") from exc
    print(
        json.dumps(
            {
                "status": payload["status"],
                "experiment": payload["experiment"],
                "scientific_decision": payload.get("scientific_decision"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
