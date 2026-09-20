"""Conditional same-envelope static screen after the joint-95 capacity audit.

Registration reads only the frozen capacity-closure protocol and the two old
static queue rosters.  It materializes one independent queue roster before the
future joint margin is known; it does not inspect the joint result ledger or a
capacity outcome, construct a margin/capacity value, or invoke a selector.

Execution is deliberately conditional.  It first revalidates all nineteen
calibration visits, the explicit holdout release, and the independent holdout,
then requires a matching capacity-closure artifact with passing coverage.  The
old point-normalized load gate is not inherited.  Loads are instead fixed
fractions of the final joint-envelope common capacity boundary.  On those
loads, R8/K36 and R16/K72 are compared with separate cold K4096 exact calls on
every registered queue.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from itertools import product
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

from algorithm.count_space_bnb_frame_candidates import (
    CountSpaceBnbSelection,
    select_count_space_bnb_frame,
)
from algorithm.experiments import aligned_hardware_joint95_capacity_closure as closure
from algorithm.experiments.aligned_hardware_count_space_bnb_scaling_screen import (
    LOCAL_CHOICE_COUNT,
    OPTION_COUNT,
    SCORE_TOLERANCE,
    _selection_reconstructs,
    _selection_work_holds,
)
from algorithm.experiments.aligned_hardware_frame_model import (
    FrameModel,
    capacity_certificate,
)
from algorithm.experiments.aligned_hardware_lcb_frame_envelope_shadow import (
    _uniform_capacity_slack,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
DEFAULT_REGISTRATION = ARTIFACTS / (
    "aligned_hardware_joint95_static_selector_screen_20260913_protocol.json"
)
DEFAULT_ROSTER = ARTIFACTS / (
    "aligned_hardware_joint95_static_selector_screen_20260913_roster.json"
)
DEFAULT_FIXED_SOURCES = ARTIFACTS / (
    "aligned_hardware_joint95_static_selector_screen_20260913_fixed_sources.json"
)
DEFAULT_RESULT = ARTIFACTS / (
    "aligned_hardware_joint95_static_selector_screen_20260913.json"
)
DEFAULT_CAPACITY_REGISTRATION = ARTIFACTS / (
    "aligned_hardware_joint95_capacity_closure_20260913_protocol.json"
)
DEFAULT_CAPACITY_RESULT = ARTIFACTS / (
    "aligned_hardware_joint95_capacity_closure_20260913.json"
)
DEFAULT_PREVIOUS_ROSTER = ARTIFACTS / (
    "aligned_hardware_count_space_bnb_confirmation_20260913_roster.json"
)

EXPERIMENT = "aligned_hardware_joint95_static_selector_screen_v1"
GATE = "aligned_hardware_joint95_static_selector_screen_runner"
CAPACITY_EXPERIMENT = closure.EXPERIMENT
CAPACITY_GATE = closure.GATE
REGISTRATION_STATUS = "REGISTERED_AWAITING_JOINT95_CAPACITY_CLOSURE"
ROSTER_SEED = 10041
ROSTER_SIZE = 512
ROSTER_RANGES = ((0, 48), (0, 30), (0, 30))
OLD_CNN_LEVELS = (0, 1, 2, 3, 4, 5, 8, 16, 24, 40)
OLD_JAX_LEVELS = (0, 1, 2, 3, 4, 5, 8, 12, 18, 25)
OLD_LLM_LEVELS = (0, 1, 2, 3, 4, 5, 8, 12, 18, 24)
PREVIOUS_ROSTER_EXPERIMENT = "aligned_hardware_count_space_bnb_confirmation_v1"
PREVIOUS_ROSTER_SEED = 260913
LANE_COUNTS = (8, 16)
LOADS = (0.50, 0.80, 0.95)
FIXED_BUDGET_BY_LANE = {8: 36, 16: 72}
EXACT_REFERENCE_BUDGET = 4096
THIS_SOURCE = (
    "algorithm/experiments/aligned_hardware_joint95_static_selector_screen.py"
)
FIXED_SOURCE_PATHS = (
    "algorithm/count_space_bnb_frame_candidates.py",
    "algorithm/experiments/aligned_hardware_count_space_bnb_scaling_screen.py",
    "algorithm/experiments/aligned_hardware_frame_model.py",
    "algorithm/experiments/aligned_hardware_joint95_capacity_closure.py",
    "algorithm/experiments/aligned_hardware_lcb_frame_envelope_shadow.py",
    THIS_SOURCE,
)

DECISION_RULE = {
    "INVALID_EXECUTION": "one or more registered execution checks fail",
    "STATIC_SUPPORT_AND_COMPUTE_PASS_ADVANCE_TO_SEPARATE_PHYSICAL_TRAJECTORY": (
        "R8/K36 and R16/K72 both have valid finite-roster certificates, actual "
        "action regret and returned epsilon strictly below eta at all three "
        "physical-capacity-normalized loads, and strictly lower aggregate LP, "
        "matrix, and candidate-evaluation work than their cold K4096 references"
    ),
    "STATIC_SUPPORT_FAIL_STOP": (
        "at least one registered lane/load support residual is nonpositive"
    ),
    "STATIC_COMPUTE_FAIL_STOP": (
        "support passes, but at least one actual-compute comparison is not strict"
    ),
}

CLAIM_BOUNDARY = (
    "An exhaustive comparison on one independently generated and preregistered "
    "512-base-queue roster for synthetic identical-lane R8 and R16 replicas of "
    "the joint-95 jtl110gpu2 complete-frame envelope. It is conditional on the "
    "registered one-future-visit joint coverage audit and is not an all-queue "
    "theorem, failure-rate estimate, physical multi-GPU trajectory, stability "
    "proof, or deployment decision. A pass advances only to a separately "
    "registered physical-time trajectory experiment."
)

PROHIBITED = (
    "reading joint visit values, the joint result ledger, or the capacity-closure "
    "outcome during registration; changing the queue roster, loads, budgets, "
    "reference, duration composition, or gates after joint95 outcomes; ordered-"
    "action or count-vector enumeration; run-time RNG; arrivals or trajectories; "
    "server pulls; and paper edits"
)


class StaticScreenRefusal(RuntimeError):
    """The frozen protocol is invalid or its conditional evidence is not ready."""


def _json(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StaticScreenRefusal(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise StaticScreenRefusal(f"JSON root must be an object: {path}")
    return value


def _write_new(path: Path, payload: Mapping[str, Any], *, compact: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(
            dict(payload),
            stream,
            indent=None if compact else 2,
            separators=(",", ":") if compact else None,
            sort_keys=not compact,
        )
        stream.write("\n")


def _current_fixed_sources() -> dict[str, str]:
    return {name: (ROOT / name).read_text(encoding="utf-8") for name in FIXED_SOURCE_PATHS}


def _old_base_grid() -> set[tuple[int, int, int]]:
    return set(product(OLD_CNN_LEVELS, OLD_JAX_LEVELS, OLD_LLM_LEVELS))


def _validated_previous_roster(payload: Mapping[str, Any]) -> tuple[tuple[int, ...], ...]:
    queues = tuple(tuple(int(value) for value in queue) for queue in payload.get("queues") or ())
    valid = (
        payload.get("experiment") == PREVIOUS_ROSTER_EXPERIMENT
        and payload.get("seed") == PREVIOUS_ROSTER_SEED
        and payload.get("coordinate_ranges_inclusive")
        == [list(bounds) for bounds in ROSTER_RANGES]
        and payload.get("requested_queue_count") == ROSTER_SIZE
        and payload.get("old_grid_excluded") is True
        and payload.get("zero_queue_excluded") is True
        and len(queues) == len(set(queues)) == ROSTER_SIZE
        and all(
            len(queue) == len(ROSTER_RANGES)
            and queue != (0, 0, 0)
            and queue not in _old_base_grid()
            and all(
                lower <= value <= upper
                for value, (lower, upper) in zip(queue, ROSTER_RANGES, strict=True)
            )
            for queue in queues
        )
    )
    if not valid:
        raise StaticScreenRefusal("prior 512-queue exclusion roster is invalid")
    return queues


def generate_roster(previous_queues: Sequence[Sequence[int]]) -> dict[str, Any]:
    """Materialize the only registration-time random object."""

    old_grid = _old_base_grid()
    previous = {tuple(int(value) for value in queue) for queue in previous_queues}
    if len(previous) != ROSTER_SIZE:
        raise StaticScreenRefusal("prior exclusion roster must contain 512 unique queues")
    rng = random.Random(ROSTER_SEED)
    queues: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    candidate_draws = 0
    while len(queues) < ROSTER_SIZE:
        candidate_draws += 1
        queue = tuple(
            rng.randint(lower, upper) for lower, upper in ROSTER_RANGES
        )
        if (
            queue == (0, 0, 0)
            or queue in old_grid
            or queue in previous
            or queue in seen
        ):
            continue
        seen.add(queue)
        queues.append(queue)
    return {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "seed": ROSTER_SEED,
        "coordinate_ranges_inclusive": [list(bounds) for bounds in ROSTER_RANGES],
        "requested_queue_count": ROSTER_SIZE,
        "candidate_draws": candidate_draws,
        "rng_integer_draws": candidate_draws * len(ROSTER_RANGES),
        "zero_queue_excluded": True,
        "old_1000_grid_excluded": True,
        "prior_512_roster_excluded": True,
        "old_1000_grid_intersection_count": sum(queue in old_grid for queue in queues),
        "prior_512_roster_intersection_count": sum(queue in previous for queue in queues),
        "queues": [list(queue) for queue in queues],
    }


def _uniform_margin_scaling_audit(source: Mapping[str, Any]) -> dict[str, Any]:
    """Audit the symbolic common-margin factorization without choosing a margin."""

    mappings = list(source.get("action_mapping") or [])
    envelopes = list(source.get("zero_margin_action_envelopes") or [])
    mapping_by_id = {str(row.get("action_id")): row for row in mappings}
    envelope_by_id = {str(row.get("action_id")): row for row in envelopes}
    issues: list[str] = []
    if len(mappings) != 9 or len(mapping_by_id) != 9:
        issues.append("ACTION_MAPPING_IS_NOT_EXACTLY_NINE_UNIQUE_ACTIONS")
    if len(envelopes) != 9 or len(envelope_by_id) != 9:
        issues.append("ZERO_MARGIN_ENVELOPE_IS_NOT_EXACTLY_NINE_UNIQUE_ACTIONS")
    if set(mapping_by_id) != set(envelope_by_id):
        issues.append("ACTION_MAPPING_AND_ZERO_MARGIN_ENVELOPE_DIFFER")

    rows = []
    for action_id in sorted(set(mapping_by_id) & set(envelope_by_id)):
        mapping = mapping_by_id[action_id]
        envelope = envelope_by_id[action_id]
        kind = str(mapping.get("action_kind") or "")
        functional_ids = [str(value) for value in mapping.get("functional_ids") or []]
        components = list(envelope.get("components") or [])
        component_ids = [str(row.get("functional_id")) for row in components]
        component_values = []
        component_zero_identity = True
        for component in components:
            try:
                predicted = float(component["predicted_time_s"])
                upper = float(component["upper_time_s"])
            except (KeyError, TypeError, ValueError):
                predicted = math.nan
                upper = math.nan
            component_values.append(predicted)
            component_zero_identity &= (
                math.isfinite(predicted)
                and predicted > 0.0
                and math.isclose(upper, predicted, rel_tol=0.0, abs_tol=SCORE_TOLERANCE)
            )
        expected_components = 2 if kind == "loaded" else 1 if kind == "dedicated" else -1
        try:
            duration = float(envelope["upper_complete_frame_duration_s"])
            multiplier = float(envelope["margin_multiplier"])
        except (KeyError, TypeError, ValueError):
            duration = math.nan
            multiplier = math.nan
        sum_components = sum(component_values)
        row_issues = []
        if component_ids != functional_ids:
            row_issues.append("FUNCTIONAL_COMPONENT_ORDER_MISMATCH")
        if len(components) != expected_components:
            row_issues.append("ACTION_COMPONENT_COUNT_MISMATCH")
        if not component_zero_identity:
            row_issues.append("ZERO_MARGIN_COMPONENT_IDENTITY_FAIL")
        if not math.isclose(multiplier, 1.0, rel_tol=0.0, abs_tol=SCORE_TOLERANCE):
            row_issues.append("ZERO_MARGIN_MULTIPLIER_IS_NOT_ONE")
        if not (
            math.isfinite(duration)
            and math.isclose(duration, sum_components, rel_tol=0.0, abs_tol=SCORE_TOLERANCE)
        ):
            row_issues.append("ACTION_DURATION_IS_NOT_COMPONENT_SUM")
        if not math.isfinite(duration) or duration <= 1.0:
            row_issues.append("ONE_SECOND_SERVICE_DENOMINATOR_CLIPPING_REACHABLE")
        issues.extend(f"{action_id}:{issue}" for issue in row_issues)
        rows.append({
            "action_id": action_id,
            "action_kind": kind,
            "component_count": len(components),
            "functional_ids": functional_ids,
            "zero_margin_duration_s": duration,
            "symbolic_duration_identity": (
                "sum_f[p_f*(1+m)] = (1+m)*sum_f[p_f]"
            ),
            "loaded_sum_included": kind == "loaded" and len(components) == 2,
            "one_second_denominator_clipping_reachable_for_m_ge_0": (
                not math.isfinite(duration) or duration <= 1.0
            ),
            "issues": row_issues,
        })
    durations = [row["zero_margin_duration_s"] for row in rows]
    if len(durations) == 9 and len(set(durations)) != 9:
        issues.append("NINE_ACTIONS_DO_NOT_HAVE_NINE_DISTINCT_DURATION_ROOTS")
    loaded_count = sum(row["action_kind"] == "loaded" for row in rows)
    dedicated_count = sum(row["action_kind"] == "dedicated" for row in rows)
    if loaded_count != 2 or dedicated_count != 7:
        issues.append("EXPECTED_SEVEN_DEDICATED_AND_TWO_LOADED_ACTIONS")
    return {
        "audit_performed_before_joint_margin_known": True,
        "joint_result_values_read": False,
        "capacity_outcome_read": False,
        "common_margin_symbol": "m >= 0",
        "functional_upper_identity": "u_f(m) = p_f*(1+m)",
        "action_duration_identity": "tau_a(m) = tau_a(0)*(1+m)",
        "frame_duration_identity": (
            "max_a tau_a(m) = (1+m)*max_a tau_a(0)"
        ),
        "joint_score_implication": (
            "the maximum of the nine registered functional ratio scores "
            "simultaneously bounds every component; summing the two loaded "
            "components therefore preserves the same multiplier"
        ),
        "dedicated_action_count": dedicated_count,
        "loaded_action_count": loaded_count,
        "action_count": len(rows),
        "minimum_zero_margin_duration_s": min(durations) if durations else None,
        "one_second_denominator_clipping_issue": any(
            row["one_second_denominator_clipping_reachable_for_m_ge_0"]
            for row in rows
        ),
        "all_nine_action_durations_uniformly_scaled": not issues and len(rows) == 9,
        "issue_count": len(issues),
        "issues": issues,
        "actions": rows,
    }


def _capacity_registration_contract(source: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version": 1,
        "gate": CAPACITY_GATE,
        "experiment": CAPACITY_EXPERIMENT,
        "status": "REGISTERED_AWAITING_JOINT95_DATA",
        "registered": True,
    }
    for key, value in required.items():
        if source.get(key) != value:
            raise StaticScreenRefusal(f"capacity registration {key} differs")
    capacity = source.get("fixed_capacity_contract") or {}
    if (
        capacity.get("base_lanes") != 2
        or capacity.get("capacity_lanes") != list(LANE_COUNTS)
        or capacity.get("loads") != list(LOADS)
        or len(capacity.get("fixed_point_boundary_arrivals_base_R2") or []) != 3
    ):
        raise StaticScreenRefusal("capacity registration lane/load/direction contract differs")
    audit = _uniform_margin_scaling_audit(source)
    if not audit["all_nine_action_durations_uniformly_scaled"]:
        raise StaticScreenRefusal(f"common-margin scaling audit failed: {audit['issues']}")
    keys = (
        "schema_version",
        "gate",
        "experiment",
        "status",
        "registered",
        "source_paths",
        "registration_scope",
        "collection_contract",
        "predictor_anchor",
        "score_and_envelope_contract",
        "action_mapping",
        "zero_margin_action_envelopes",
        "point_reference_model",
        "fixed_capacity_contract",
        "run_preconditions",
        "claim_boundary",
    )
    return {key: _json(source.get(key)) for key in keys}


def _protocol_constants(
    *,
    capacity_registration_path: Path,
    capacity_result_path: Path,
    previous_roster_path: Path,
    source_contract: Mapping[str, Any],
    previous_roster: Mapping[str, Any],
    roster: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "gate": GATE,
        "experiment": EXPERIMENT,
        "status": REGISTRATION_STATUS,
        "registered": True,
        "source_paths": {
            "capacity_closure_registration": str(capacity_registration_path.resolve()),
            "capacity_closure_result": str(capacity_result_path.resolve()),
            "joint_results_directory": str(
                Path(source_contract["source_paths"]["joint_results_directory"]).resolve()
            ),
            "previous_count_space_confirmation_roster": str(previous_roster_path.resolve()),
        },
        "capacity_closure_registration_contract": _json(source_contract),
        "uniform_margin_scaling_audit": _uniform_margin_scaling_audit(source_contract),
        "queue_roster_contract": {
            "seed": ROSTER_SEED,
            "size": ROSTER_SIZE,
            "coordinate_ranges_inclusive": [list(bounds) for bounds in ROSTER_RANGES],
            "candidate_draws": roster["candidate_draws"],
            "rng_integer_draws": roster["rng_integer_draws"],
            "base_queues": _json(roster["queues"]),
            "lane_scaling_rule": "Q_R = (R/2) * Q_base coordinatewise",
            "zero_queue_excluded": True,
            "old_1000_grid": {
                "cnn_levels": list(OLD_CNN_LEVELS),
                "jax_levels": list(OLD_JAX_LEVELS),
                "llm_levels": list(OLD_LLM_LEVELS),
                "size": len(_old_base_grid()),
                "intersection_count": roster["old_1000_grid_intersection_count"],
            },
            "previous_512_roster": {
                "experiment": previous_roster["experiment"],
                "seed": previous_roster["seed"],
                "size": previous_roster["requested_queue_count"],
                "candidate_draws": previous_roster["candidate_draws"],
                "queues": _json(previous_roster["queues"]),
                "intersection_count": roster["prior_512_roster_intersection_count"],
            },
        },
        "model_contract": {
            "node": "jtl110gpu2",
            "base_lanes": 2,
            "lane_counts": list(LANE_COUNTS),
            "action_count": 9,
            "duration_catalogue": (
                "the capacity-closure nine-action complete-frame envelope rebuilt "
                "from the frozen predictor anchor and final rank-19 joint margin"
            ),
            "synthetic_lane_rule": (
                "replicate the same hardware-local lane option catalogue without "
                "adding a physical multi-GPU interference claim"
            ),
        },
        "physical_load_contract": {
            "loads": list(LOADS),
            "common_boundary_source": "capacity_closure.capacity.common_capacity_scale",
            "base_arrival_source": (
                "capacity_closure_registration.fixed_capacity_contract."
                "fixed_point_boundary_arrivals_base_R2"
            ),
            "common_boundary_arrival_formula": (
                "lambda_boundary_R = common_capacity_scale * (R/2) * lambda_point_R2"
            ),
            "registered_load_arrival_formula": (
                "lambda_R(load) = load * lambda_boundary_R"
            ),
            "eta_formula": (
                "max_w min_i[R*sum_a w_a requirements[a,i]/max(1,tau_joint95[a]) "
                "- lambda_R(load)[i]] - projection_support_loss_R"
            ),
            "old_point_load_capacity_gate_required": False,
            "reason": (
                "the registered loads are redefined relative to the final physical "
                "joint-envelope common boundary"
            ),
        },
        "selector_contract": {
            "fixed_budget_by_lane": {str(key): value for key, value in FIXED_BUDGET_BY_LANE.items()},
            "cold_exact_reference_budget": EXACT_REFERENCE_BUDGET,
            "candidate": "one fresh count-space B&B call per state at the fixed budget",
            "reference": "one separate cold count-space B&B K4096 call per state",
            "candidate_epsilon": "returned (upper_bound-score)/queue_L1",
            "actual_normalized_action_regret": (
                "(cold_exact_reference_score-candidate_score)/queue_L1"
            ),
            "support_gate": (
                "for every R/load case both eta-max(candidate_epsilon) and "
                "eta-max(actual_normalized_action_regret) exceed 1e-10"
            ),
            "certificate_gate": (
                "every candidate and reference reconstructs; every returned bound "
                "is valid; every K4096 reference is exact; actual regret is "
                "nonnegative and no larger than returned epsilon"
            ),
            "actual_compute_gate": (
                "for each lane and in aggregate, fixed-budget totals are strictly "
                "below cold-reference totals for LP calls, LP matrix coefficients, "
                "and candidate evaluations"
            ),
            "enumeration_gate": (
                "ordered-action evaluations and count-vector enumerations both equal zero"
            ),
        },
        "expected_work": {
            "registered_base_queues": ROSTER_SIZE,
            "scaled_queue_states": ROSTER_SIZE * len(LANE_COUNTS),
            "candidate_selector_calls": ROSTER_SIZE * len(LANE_COUNTS),
            "cold_reference_selector_calls": ROSTER_SIZE * len(LANE_COUNTS),
            "capacity_structural_calls": len(LANE_COUNTS),
            "capacity_slack_calls": len(LANE_COUNTS) * len(LOADS),
            "ordered_action_evaluations": 0,
            "count_vector_enumerations": 0,
            "run_rng_initializations": 0,
            "arrival_tapes": 0,
            "trajectory_frames": 0,
        },
        "run_preconditions": [
            "the frozen capacity-closure source protocol is unchanged",
            "waves 14..32 are complete dispatcher-valid calibration terminals",
            "the explicit rank-19 holdout release is valid and unchanged",
            "wave 33 is a complete dispatcher-valid independent holdout terminal",
            "the recomputed joint95 holdout coverage audit passes",
            "a matching capacity-closure result exists and reproduces the joint envelope and capacity analysis",
            "the final common capacity boundary is positive and finite",
        ],
        "decision_rule": dict(DECISION_RULE),
        "claim_boundary": CLAIM_BOUNDARY,
        "prohibited": PROHIBITED,
        "registration_scope": {
            "capacity_closure_protocol_read": True,
            "previous_roster_read": True,
            "joint_results_directory_read": False,
            "joint_visit_values_read": False,
            "capacity_closure_result_read": False,
            "joint_margin_computed": False,
            "capacity_computed": False,
            "selector_invoked": False,
            "exact_reference_invoked": False,
            "arrival_or_trajectory_constructed": False,
            "server_pull_performed": False,
            "roster_rng_initializations": 1,
            "roster_rng_integer_draws": roster["rng_integer_draws"],
            "other_rng_initializations": 0,
            "other_rng_draws": 0,
        },
    }


def register(
    registration: Path = DEFAULT_REGISTRATION,
    roster_path: Path = DEFAULT_ROSTER,
    fixed_sources_path: Path = DEFAULT_FIXED_SOURCES,
    *,
    capacity_registration_path: Path = DEFAULT_CAPACITY_REGISTRATION,
    capacity_result_path: Path = DEFAULT_CAPACITY_RESULT,
    previous_roster_path: Path = DEFAULT_PREVIOUS_ROSTER,
) -> dict[str, Any]:
    """Freeze the queue roster and protocol without reading any joint outcome."""

    registration = Path(registration)
    roster_path = Path(roster_path)
    fixed_sources_path = Path(fixed_sources_path)
    if any(path.exists() for path in (registration, roster_path, fixed_sources_path)):
        raise StaticScreenRefusal("static-screen registration artifacts already exist")
    source = _read_object(Path(capacity_registration_path))
    source_contract = _capacity_registration_contract(source)
    previous_payload = _read_object(Path(previous_roster_path))
    previous = _validated_previous_roster(previous_payload)
    roster = generate_roster(previous)
    protocol = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        **_protocol_constants(
            capacity_registration_path=Path(capacity_registration_path),
            capacity_result_path=Path(capacity_result_path),
            previous_roster_path=Path(previous_roster_path),
            source_contract=source_contract,
            previous_roster=previous_payload,
            roster=roster,
        ),
    }
    _write_new(roster_path, roster, compact=True)
    _write_new(fixed_sources_path, _current_fixed_sources())
    _write_new(registration, protocol, compact=True)
    return protocol


def _validate_roster(
    roster: Mapping[str, Any], protocol: Mapping[str, Any]
) -> tuple[tuple[int, int, int], ...]:
    contract = protocol.get("queue_roster_contract") or {}
    queues = tuple(tuple(int(value) for value in row) for row in roster.get("queues") or ())
    previous = {
        tuple(int(value) for value in row)
        for row in (contract.get("previous_512_roster") or {}).get("queues") or ()
    }
    old_grid = _old_base_grid()
    valid = (
        roster.get("schema_version") == 1
        and roster.get("experiment") == EXPERIMENT
        and roster.get("seed") == ROSTER_SEED
        and roster.get("coordinate_ranges_inclusive")
        == [list(bounds) for bounds in ROSTER_RANGES]
        and roster.get("requested_queue_count") == ROSTER_SIZE
        and roster.get("candidate_draws") == contract.get("candidate_draws")
        and roster.get("rng_integer_draws") == contract.get("rng_integer_draws")
        and roster.get("queues") == contract.get("base_queues")
        and len(queues) == len(set(queues)) == ROSTER_SIZE
        and len(previous) == ROSTER_SIZE
        and all(
            queue != (0, 0, 0)
            and queue not in old_grid
            and queue not in previous
            and len(queue) == len(ROSTER_RANGES)
            and all(
                lower <= value <= upper
                for value, (lower, upper) in zip(queue, ROSTER_RANGES, strict=True)
            )
            for queue in queues
        )
        and roster.get("old_1000_grid_intersection_count") == 0
        and roster.get("prior_512_roster_intersection_count") == 0
    )
    if not valid:
        raise StaticScreenRefusal("registered static-screen queue roster changed or is invalid")
    return queues


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    if (
        protocol.get("schema_version") != 1
        or protocol.get("gate") != GATE
        or protocol.get("experiment") != EXPERIMENT
        or protocol.get("status") != REGISTRATION_STATUS
        or protocol.get("registered") is not True
        or protocol.get("decision_rule") != DECISION_RULE
        or protocol.get("claim_boundary") != CLAIM_BOUNDARY
        or protocol.get("prohibited") != PROHIBITED
    ):
        raise StaticScreenRefusal("static-screen registration is invalid")
    audit = protocol.get("uniform_margin_scaling_audit") or {}
    if (
        audit.get("all_nine_action_durations_uniformly_scaled") is not True
        or audit.get("issue_count") != 0
        or audit.get("joint_result_values_read") is not False
        or audit.get("capacity_outcome_read") is not False
    ):
        raise StaticScreenRefusal("registered common-margin scaling audit is invalid")


def _runtime_context(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Revalidate complete joint evidence before accepting the closure result."""

    paths = protocol.get("source_paths") or {}
    source_path = Path(paths["capacity_closure_registration"])
    source = _read_object(source_path)
    contract = _capacity_registration_contract(source)
    if contract != protocol.get("capacity_closure_registration_contract"):
        raise StaticScreenRefusal("capacity-closure registration changed after static registration")
    if _uniform_margin_scaling_audit(source) != protocol.get("uniform_margin_scaling_audit"):
        raise StaticScreenRefusal("common-margin scaling audit changed after registration")

    try:
        current = closure._current_source_snapshot(source)
        closure._validate_registered_sources(source, current)
        joint = closure._complete_joint_evidence(source, current["manifest"])
    except (closure.ClosureRefusal, KeyError, TypeError, ValueError) as exc:
        raise StaticScreenRefusal(
            f"joint95 calibration/release/holdout evidence is incomplete or invalid: {exc}"
        ) from exc
    if (
        joint.get("release", {}).get("audit_passed") is not True
        or joint.get("holdout_covered") is not True
        or len(joint.get("calibration") or ()) != 19
        or int((joint.get("holdout") or {}).get("joint_wave", -1)) != closure.HOLDOUT_WAVE
    ):
        raise StaticScreenRefusal("joint95 complete-evidence coverage audit did not pass")

    capacity_result = _read_object(Path(paths["capacity_closure_result"]))
    if (
        capacity_result.get("schema_version") != 1
        or capacity_result.get("gate") != CAPACITY_GATE
        or capacity_result.get("experiment") != CAPACITY_EXPERIMENT
        or capacity_result.get("status") != "PASS"
        or capacity_result.get("protocol") != source
        or capacity_result.get("joint95_audit") != joint
    ):
        raise StaticScreenRefusal("capacity-closure artifact does not match the complete joint audit")

    margin = float(joint["one_sided_margin"])
    envelope_model, envelopes = closure._planned_baseline_model(
        current["point_model"],
        source["predictor_anchor"],
        source["action_mapping"],
        margin=margin,
    )
    expected_model = _json(envelope_model.snapshot())
    physical = capacity_result.get("physical_complete_frame_envelope") or {}
    if physical.get("model") != expected_model or physical.get("actions") != _json(envelopes):
        raise StaticScreenRefusal("capacity-closure artifact changed the joint-envelope catalogue")
    recomputed_capacity = closure._capacity_analysis(source, envelope_model)
    if capacity_result.get("capacity") != recomputed_capacity:
        raise StaticScreenRefusal("capacity-closure capacity analysis does not reproduce")
    scale = float(recomputed_capacity["common_capacity_scale"])
    if not math.isfinite(scale) or scale <= 0.0:
        raise StaticScreenRefusal("final joint-envelope common capacity boundary is not positive")

    # This screen intentionally does not consult capacity_established or the old
    # point-normalized all_registered_capacity_margins_positive field.
    return {
        "source": source,
        "joint": joint,
        "capacity_closure": capacity_result,
        "envelope_model": envelope_model,
        "envelopes": envelopes,
        "common_capacity_scale": scale,
    }


def _physical_capacity_rows(
    source: Mapping[str, Any], model: FrameModel, common_scale: float
) -> dict[str, list[dict[str, Any]]]:
    capacity = source.get("fixed_capacity_contract") or {}
    base = tuple(float(value) for value in capacity["fixed_point_boundary_arrivals_base_R2"])
    result: dict[str, list[dict[str, Any]]] = {}
    for lanes in LANE_COUNTS:
        lane_model = replace(model, lanes=lanes)
        structural = capacity_certificate(lane_model, LOADS[0])
        boundary = tuple(common_scale * lanes / 2 * value for value in base)
        rows = []
        for load in LOADS:
            arrivals = tuple(load * value for value in boundary)
            slack = _uniform_capacity_slack(lane_model, arrivals, structural)
            rows.append({
                "lanes": lanes,
                "load": load,
                "load_normalization": "fraction_of_final_joint_envelope_common_capacity_boundary",
                "common_capacity_scale": common_scale,
                "common_boundary_arrival_rates": list(boundary),
                **slack,
                "capacity_projection_passed": bool(structural["capacity_projection_passed"]),
                "positive_effective_slack": (
                    bool(structural["capacity_projection_passed"])
                    and float(slack["effective_slack_eta"]) > SCORE_TOLERANCE
                ),
            })
        result[str(lanes)] = rows
    return result


def _certificate_valid(selection: CountSpaceBnbSelection, queue: Sequence[int]) -> bool:
    queue_norm = sum(queue)
    return (
        selection.score <= selection.upper_bound + SCORE_TOLERANCE
        and selection.gap >= -SCORE_TOLERANCE
        and math.isclose(
            selection.gap,
            max(0.0, selection.upper_bound - selection.score),
            rel_tol=0.0,
            abs_tol=SCORE_TOLERANCE,
        )
        and math.isclose(
            selection.epsilon,
            selection.gap / queue_norm if queue_norm else 0.0,
            rel_tol=0.0,
            abs_tol=SCORE_TOLERANCE,
        )
        and selection.dual_certified_lp_calls + selection.safe_fallback_lp_calls
        == selection.lp_calls
    )


def _work_row(selection: CountSpaceBnbSelection) -> dict[str, int]:
    return {
        "lp_calls": int(selection.lp_calls),
        "lp_matrix_coefficients": int(selection.lp_matrix_coefficients),
        "candidate_evaluations": int(selection.candidate_evaluations),
        "residual_fill_option_probes": int(selection.residual_fill_option_probes),
        "dual_column_checks": int(selection.dual_column_checks),
    }


def _add_work(target: dict[str, int], selection: CountSpaceBnbSelection) -> None:
    for key, value in _work_row(selection).items():
        target[key] += value


def _new_work() -> dict[str, int]:
    return {
        "lp_calls": 0,
        "lp_matrix_coefficients": 0,
        "candidate_evaluations": 0,
        "residual_fill_option_probes": 0,
        "dual_column_checks": 0,
    }


def _selection_witness(
    base_queue: Sequence[int],
    queue: Sequence[int],
    candidate: CountSpaceBnbSelection,
    reference: CountSpaceBnbSelection,
    value: float,
) -> dict[str, Any]:
    return {
        "base_queue": list(base_queue),
        "scaled_queue": list(queue),
        "queue_l1": sum(queue),
        "value": value,
        "candidate": {
            "option_counts": list(candidate.option_counts),
            "score": candidate.score,
            "upper_bound": candidate.upper_bound,
            "gap": candidate.gap,
            "epsilon": candidate.epsilon,
            "exact": candidate.exact,
            "work": _work_row(candidate),
        },
        "cold_reference": {
            "option_counts": list(reference.option_counts),
            "score": reference.score,
            "upper_bound": reference.upper_bound,
            "gap": reference.gap,
            "epsilon": reference.epsilon,
            "exact": reference.exact,
            "work": _work_row(reference),
        },
    }


def _screen_lane(
    model: FrameModel,
    base_queues: Sequence[Sequence[int]],
    lanes: int,
    capacity_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, bool]]:
    budget = FIXED_BUDGET_BY_LANE[lanes]
    factor = lanes // 2
    candidate_work = _new_work()
    reference_work = _new_work()
    max_epsilon = -1.0
    max_regret = -1.0
    epsilon_witness = None
    regret_witness = None
    candidate_exact_count = 0
    action_match_count = 0
    valid_candidate_count = 0
    valid_reference_count = 0
    regret_bounded_count = 0

    for base_queue in base_queues:
        queue = tuple(factor * int(value) for value in base_queue)
        candidate = select_count_space_bnb_frame(model.options, queue, lanes, budget)
        reference = select_count_space_bnb_frame(
            model.options, queue, lanes, EXACT_REFERENCE_BUDGET
        )
        candidate_valid = (
            _selection_reconstructs(model, queue, lanes, candidate)
            and _selection_work_holds(queue, lanes, budget, candidate)
            and _certificate_valid(candidate, queue)
        )
        reference_valid = (
            _selection_reconstructs(model, queue, lanes, reference)
            and _selection_work_holds(
                queue, lanes, EXACT_REFERENCE_BUDGET, reference
            )
            and _certificate_valid(reference, queue)
            and reference.exact
            and math.isclose(
                reference.score,
                reference.upper_bound,
                rel_tol=0.0,
                abs_tol=SCORE_TOLERANCE,
            )
        )
        queue_norm = sum(queue)
        regret = (reference.score - candidate.score) / queue_norm
        regret_bounded = (
            regret >= -SCORE_TOLERANCE
            and regret <= candidate.epsilon + SCORE_TOLERANCE
        )
        candidate_exact_count += int(candidate.exact)
        action_match_count += int(candidate.option_counts == reference.option_counts)
        valid_candidate_count += int(candidate_valid)
        valid_reference_count += int(reference_valid)
        regret_bounded_count += int(regret_bounded)
        _add_work(candidate_work, candidate)
        _add_work(reference_work, reference)
        if candidate.epsilon > max_epsilon:
            max_epsilon = candidate.epsilon
            epsilon_witness = _selection_witness(
                base_queue, queue, candidate, reference, candidate.epsilon
            )
        if regret > max_regret:
            max_regret = regret
            regret_witness = _selection_witness(
                base_queue, queue, candidate, reference, regret
            )

    compute_comparisons = {
        key: {
            "candidate": candidate_work[key],
            "cold_exact_reference": reference_work[key],
            "candidate_fraction_of_reference": (
                candidate_work[key] / reference_work[key]
                if reference_work[key]
                else None
            ),
            "strictly_lower": candidate_work[key] < reference_work[key],
        }
        for key in ("lp_calls", "lp_matrix_coefficients", "candidate_evaluations")
    }
    support_rows = []
    for capacity in capacity_rows:
        eta = float(capacity["effective_slack_eta"])
        epsilon_residual = eta - max_epsilon
        regret_residual = eta - max_regret
        support_rows.append({
            "lanes": lanes,
            "load": float(capacity["load"]),
            "load_normalization": capacity["load_normalization"],
            "arrival_rates": _json(capacity["arrival_rates"]),
            "effective_slack_eta": eta,
            "maximum_candidate_epsilon": max_epsilon,
            "maximum_actual_normalized_action_regret": max_regret,
            "eta_minus_candidate_epsilon": epsilon_residual,
            "eta_minus_actual_action_regret": regret_residual,
            "candidate_epsilon_strictly_below_eta": (
                epsilon_residual > SCORE_TOLERANCE
            ),
            "actual_action_regret_strictly_below_eta": (
                regret_residual > SCORE_TOLERANCE
            ),
            "support_residual_gate_passed": (
                capacity["capacity_projection_passed"]
                and epsilon_residual > SCORE_TOLERANCE
                and regret_residual > SCORE_TOLERANCE
            ),
        })
    lane_checks = {
        "scaled_queue_construction_count_is_exact": len(base_queues) == ROSTER_SIZE,
        "every_candidate_certificate_valid": valid_candidate_count == ROSTER_SIZE,
        "every_cold_reference_is_valid_and_exact": valid_reference_count == ROSTER_SIZE,
        "every_actual_regret_is_nonnegative_and_bounded_by_epsilon": (
            regret_bounded_count == ROSTER_SIZE
        ),
    }
    support_gate_passed = all(
        row["support_residual_gate_passed"] for row in support_rows
    )
    compute_gate_passed = all(
        row["strictly_lower"] for row in compute_comparisons.values()
    )
    return ({
        "lanes": lanes,
        "queue_scale": factor,
        "state_count": len(base_queues),
        "fixed_lp_budget": budget,
        "cold_exact_reference_budget": EXACT_REFERENCE_BUDGET,
        "raw_ordered_actions_per_state": LOCAL_CHOICE_COUNT**lanes,
        "represented_count_vectors_per_state": math.comb(OPTION_COUNT + lanes, lanes),
        "candidate_exact_state_count": candidate_exact_count,
        "candidate_action_matches_reference_count": action_match_count,
        "valid_candidate_certificate_count": valid_candidate_count,
        "valid_exact_reference_count": valid_reference_count,
        "regret_bounded_by_epsilon_count": regret_bounded_count,
        "maximum_candidate_epsilon": max_epsilon,
        "maximum_actual_normalized_action_regret": max_regret,
        "epsilon_witness": epsilon_witness,
        "action_regret_witness": regret_witness,
        "support_by_physical_load": support_rows,
        "candidate_actual_work": candidate_work,
        "cold_reference_actual_work": reference_work,
        "actual_compute_comparison": compute_comparisons,
        "checks": lane_checks,
        "support_gate_passed": support_gate_passed,
        "actual_compute_gate_passed": compute_gate_passed,
    }, lane_checks)


def _decision(execution_passed: bool, lanes: Sequence[Mapping[str, Any]], aggregate_compute: bool) -> str:
    if not execution_passed:
        return "INVALID_EXECUTION"
    if not all(row["support_gate_passed"] for row in lanes):
        return "STATIC_SUPPORT_FAIL_STOP"
    if not aggregate_compute or not all(row["actual_compute_gate_passed"] for row in lanes):
        return "STATIC_COMPUTE_FAIL_STOP"
    return "STATIC_SUPPORT_AND_COMPUTE_PASS_ADVANCE_TO_SEPARATE_PHYSICAL_TRAJECTORY"


def run_registered(
    registration: Path = DEFAULT_REGISTRATION,
    roster_path: Path = DEFAULT_ROSTER,
    fixed_sources_path: Path = DEFAULT_FIXED_SOURCES,
    output: Path = DEFAULT_RESULT,
) -> dict[str, Any]:
    output = Path(output)
    if output.exists():
        raise StaticScreenRefusal(f"static-screen result already exists: {output}")
    protocol = _read_object(Path(registration))
    _validate_protocol(protocol)
    roster_payload = _read_object(Path(roster_path))
    base_queues = _validate_roster(roster_payload, protocol)
    fixed_sources = _read_object(Path(fixed_sources_path))
    if fixed_sources != _current_fixed_sources():
        raise StaticScreenRefusal("registered static-screen sources changed")

    # Do not invoke capacity or selector code until the future joint evidence
    # and the matching capacity-closure artifact are complete and valid.
    context = _runtime_context(protocol)
    model = context["envelope_model"]
    if (
        len(model.options) != OPTION_COUNT
        or len(set(max(1.0, option.duration) for option in model.options)) != OPTION_COUNT
    ):
        raise StaticScreenRefusal("joint envelope does not retain nine duration roots")
    capacity_by_lane = _physical_capacity_rows(
        context["source"], model, context["common_capacity_scale"]
    )

    started = time.perf_counter()
    lane_rows = []
    lane_checks = []
    for lanes in LANE_COUNTS:
        row, checks = _screen_lane(
            model, base_queues, lanes, capacity_by_lane[str(lanes)]
        )
        lane_rows.append(row)
        lane_checks.append(checks)

    candidate_total = _new_work()
    reference_total = _new_work()
    for row in lane_rows:
        for key in candidate_total:
            candidate_total[key] += row["candidate_actual_work"][key]
            reference_total[key] += row["cold_reference_actual_work"][key]
    aggregate_comparisons = {
        key: {
            "candidate": candidate_total[key],
            "cold_exact_reference": reference_total[key],
            "candidate_fraction_of_reference": (
                candidate_total[key] / reference_total[key]
                if reference_total[key]
                else None
            ),
            "strictly_lower": candidate_total[key] < reference_total[key],
        }
        for key in ("lp_calls", "lp_matrix_coefficients", "candidate_evaluations")
    }
    aggregate_compute = all(row["strictly_lower"] for row in aggregate_comparisons.values())
    checks = {
        "registered_roster_is_explicit_unique_and_disjoint": len(base_queues) == ROSTER_SIZE,
        "joint95_calibration_release_holdout_and_coverage_revalidated": True,
        "matching_capacity_closure_artifact_reproduced": True,
        "common_margin_uniformly_scales_all_nine_durations": True,
        "old_point_load_capacity_gate_not_used": True,
        "physical_loads_are_common_boundary_normalized": all(
            row["load_normalization"]
            == "fraction_of_final_joint_envelope_common_capacity_boundary"
            for rows in capacity_by_lane.values()
            for row in rows
        ),
        "every_lane_execution_check_passes": all(
            all(row.values()) for row in lane_checks
        ),
        "selector_call_count_is_exact": (
            sum(row["state_count"] for row in lane_rows)
            == ROSTER_SIZE * len(LANE_COUNTS)
        ),
        "no_ordered_or_count_vector_enumeration": True,
        "no_run_rng_arrivals_or_trajectories": True,
    }
    execution_passed = all(checks.values())
    decision = _decision(execution_passed, lane_rows, aggregate_compute)
    report = {
        "schema_version": 1,
        "gate": GATE,
        "experiment": EXPERIMENT,
        "status": "PASS" if execution_passed else "FAIL",
        "registration_path": str(Path(registration).resolve()),
        "protocol": protocol,
        "checks": checks,
        "joint95_dependency": {
            "capacity_closure_result": str(
                Path(protocol["source_paths"]["capacity_closure_result"]).resolve()
            ),
            "one_sided_margin": context["joint"]["one_sided_margin"],
            "holdout_covered": context["joint"]["holdout_covered"],
            "common_capacity_scale": context["common_capacity_scale"],
            "old_point_load_capacity_gate_consulted": False,
        },
        "physical_complete_frame_envelope": {
            "model": _json(model.snapshot()),
            "actions": _json(context["envelopes"]),
        },
        "physical_capacity_normalized_loads": capacity_by_lane,
        "lane_summaries": lane_rows,
        "aggregate_actual_compute": {
            "candidate": candidate_total,
            "cold_exact_reference": reference_total,
            "comparison": aggregate_comparisons,
            "gate_passed": aggregate_compute,
        },
        "work_accounting": {
            "candidate_selector_calls": ROSTER_SIZE * len(LANE_COUNTS),
            "cold_reference_selector_calls": ROSTER_SIZE * len(LANE_COUNTS),
            "capacity_structural_calls": len(LANE_COUNTS),
            "capacity_slack_calls": len(LANE_COUNTS) * len(LOADS),
            "ordered_action_evaluations": 0,
            "count_vector_enumerations": 0,
            "run_rng_initializations": 0,
            "arrival_tapes": 0,
            "trajectory_frames": 0,
        },
        "scientific_decision": decision,
        "physical_trajectory_advance": (
            decision
            == "STATIC_SUPPORT_AND_COMPUTE_PASS_ADVANCE_TO_SEPARATE_PHYSICAL_TRAJECTORY"
        ),
        "claim_boundary": CLAIM_BOUNDARY,
        "next_step": (
            "register a separate physical-time paired trajectory before execution"
            if decision
            == "STATIC_SUPPORT_AND_COMPUTE_PASS_ADVANCE_TO_SEPARATE_PHYSICAL_TRAJECTORY"
            else "preserve this finite-roster outcome and do not tune on its queues"
        ),
        "timing": {
            "total_screen_wall_s": time.perf_counter() - started,
            "timing_is_scientific_gate": False,
        },
    }
    _write_new(output, report)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    register_parser = sub.add_parser("register")
    register_parser.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    register_parser.add_argument("--roster", type=Path, default=DEFAULT_ROSTER)
    register_parser.add_argument("--fixed-sources", type=Path, default=DEFAULT_FIXED_SOURCES)
    register_parser.add_argument(
        "--capacity-registration", type=Path, default=DEFAULT_CAPACITY_REGISTRATION
    )
    register_parser.add_argument("--capacity-result", type=Path, default=DEFAULT_CAPACITY_RESULT)
    register_parser.add_argument("--previous-roster", type=Path, default=DEFAULT_PREVIOUS_ROSTER)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    run_parser.add_argument("--roster", type=Path, default=DEFAULT_ROSTER)
    run_parser.add_argument("--fixed-sources", type=Path, default=DEFAULT_FIXED_SOURCES)
    run_parser.add_argument("--output", type=Path, default=DEFAULT_RESULT)
    args = parser.parse_args(argv)
    try:
        if args.command == "register":
            payload = register(
                args.registration,
                args.roster,
                args.fixed_sources,
                capacity_registration_path=args.capacity_registration,
                capacity_result_path=args.capacity_result,
                previous_roster_path=args.previous_roster,
            )
            output = args.registration
        else:
            payload = run_registered(
                args.registration, args.roster, args.fixed_sources, args.output
            )
            output = args.output
    except StaticScreenRefusal as exc:
        parser.exit(2, f"REFUSED: {exc}\n")
    print(json.dumps({
        "experiment": payload["experiment"],
        "status": payload.get("status", REGISTRATION_STATUS),
        "scientific_decision": payload.get("scientific_decision"),
        "output": str(output),
    }))


if __name__ == "__main__":
    main()
