"""Development budget frontier for the frozen joint95-v5 static diagnostic.

The same seed-13041 roster is reused to identify a fixed R8/R16 budget pair.
The grid was chosen after an exploratory read-only scout on this roster, so the
result is development evidence only.  One cumulative canonical B&B trace per
state supplies all registered checkpoints and an exact K4096 endpoint.  No
arrivals, trajectories, server data, or paper sources are touched.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from algorithm.canonical_count_space_bnb_frame_candidates import (
    trace_canonical_count_space_bnb_frame,
)
from algorithm.experiments import (
    aligned_hardware_joint95_v5_envelope_normalized_static_diagnostic as source,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
STEM = "aligned_hardware_joint95_v5_envelope_budget_frontier_development_v2_20260919"
DEFAULT_REGISTRATION = ARTIFACTS / f"{STEM}_protocol.json"
DEFAULT_FIXED_SOURCES = ARTIFACTS / f"{STEM}_fixed_sources.json"
DEFAULT_RESULT = ARTIFACTS / f"{STEM}.json"
DEFAULT_SOURCE_REGISTRATION = source.DEFAULT_REGISTRATION
DEFAULT_SOURCE_FIXED_SOURCES = source.DEFAULT_FIXED_SOURCES
DEFAULT_SOURCE_RESULT = source.DEFAULT_RESULT

EXPERIMENT = "aligned_hardware_joint95_v5_envelope_budget_frontier_development_v2"
GATE = "aligned_hardware_joint95_v5_envelope_budget_frontier_development_v2_runner"
REGISTRATION_STATUS = "REGISTERED_POSTSCOUT_AWAITING_DEVELOPMENT_FRONTIER_V2"
SOURCE_DECISION = "CANONICAL_PHYSICAL_SELECTOR_SUPPORT_FAIL_STOP"
PASS_DECISION = (
    "DEVELOPMENT_ROSTER_ACTION_IDENTICAL_BUDGET_PAIR_READY_FOR_"
    "INDEPENDENT_STATIC_CONFIRMATION"
)
SUPPORT_ONLY_DECISION = (
    "SUPPORT_COMPUTE_PAIR_LACKS_DEVELOPMENT_ROSTER_ACTION_IDENTITY_STOP"
)
NO_PAIR_DECISION = "NO_REGISTERED_SUPPORT_COMPUTE_BUDGET_PAIR_STOP"
INVALID_DECISION = "INVALID_EXECUTION"
REFERENCE_BUDGET = 4096
LANE_BUDGETS = {
    8: (36, 43, 44, 48),
    16: (9, 18, 36, 67, 68, 72),
}
WORK_GATE_FIELDS = ("lp_calls", "lp_matrix_coefficients", "candidate_evaluations")
THIS_SOURCE = (
    "algorithm/experiments/"
    "aligned_hardware_joint95_v5_envelope_budget_frontier_development.py"
)
TRACE_RULE = (
    "one cumulative canonical count-space B&B trace per state and lane; "
    "all registered checkpoints precede a K4096 exact endpoint"
)
BUDGET_SELECTION_RULE = {
    "support_compute_minimum": (
        "smallest registered K whose valid returned certificate has maximum "
        "epsilon and actual normalized regret strictly below eta at every "
        "registered load, and whose aggregate LP calls, matrix cells, and "
        "candidate evaluations are each strictly below the exact endpoint"
    ),
    "development_roster_action_identity_minimum": (
        "smallest support-compute K whose selected option indices equal the "
        "exact canonical endpoint on all 512 states"
    ),
    "confirmation_budget": "development_roster_action_identity_minimum",
    "score_tolerance": source.canonical.SCORE_TOLERANCE,
}
EXPECTED_WORK = {
    "base_queue_states": 512,
    "lane_counts": [8, 16],
    "cumulative_traces": 1024,
    "checkpoint_reads": 512 * sum(len(v) + 1 for v in LANE_BUDGETS.values()),
    "ordered_action_enumerations": 0,
    "count_vector_enumerations": 0,
    "arrival_generations": 0,
    "trajectory_runs": 0,
    "rng_draws": 0,
    "server_pulls": 0,
}
REGISTRATION_SCOPE = {
    "source_outcome_read": True,
    "exploratory_scout_preceded_registration": True,
    "selector_invoked_during_registration": False,
    "trajectory_built": False,
    "rng_draws": 0,
    "server_pulls": 0,
    "paper_edited": False,
}

DECISION_RULE = {
    INVALID_DECISION: "one or more registered execution-integrity checks fail",
    NO_PAIR_DECISION: (
        "at least one lane has no registered checkpoint whose returned epsilon and "
        "actual regret pass every normalized-load eta and whose three work totals "
        "are strictly below the exact endpoint"
    ),
    SUPPORT_ONLY_DECISION: (
        "both lanes have a support-and-compute checkpoint, but at least one lane "
        "has no such checkpoint matching the exact canonical action on all 512 "
        "development-roster states"
    ),
    PASS_DECISION: (
        "for each lane, select the smallest registered checkpoint that passes the "
        "support, strict-work, and 512-state development-roster exact-canonical-"
        "action identity gates"
    ),
}
CLAIM_BOUNDARY = (
    "An outcome-informed development frontier on the already used seed-13041 "
    "512-state roster and immutable joint95-v5 envelope. It may identify one fixed "
    "budget per lane for a new, disjoint, preregistered static confirmation. It is "
    "not independent confirmation, trajectory authorization, a capacity "
    "reclassification, an all-queue theorem, or a physical multi-GPU result."
)
PROHIBITED = (
    "changing the frozen v5 diagnostic or its negative decisions; treating this "
    "same-roster development frontier as confirmation; changing the registered "
    "budget grid or selection rule after execution; arrivals, trajectories, run-time "
    "RNG, server pulls, and paper edits"
)


class BudgetFrontierRefusal(RuntimeError):
    """The registered development frontier contract is not satisfied."""


def _json(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BudgetFrontierRefusal(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BudgetFrontierRefusal(f"JSON root must be an object: {path}")
    return value


def _write_new(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(dict(payload), stream, indent=2, sort_keys=True)
        stream.write("\n")


def _current_fixed_sources() -> dict[str, str]:
    return {
        **source._current_fixed_sources(),
        THIS_SOURCE: (ROOT / THIS_SOURCE).read_text(encoding="utf-8"),
    }


def _source_contract(
    result: Mapping[str, Any], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        source._validate_protocol(protocol)
    except source.EnvelopeDiagnosticRefusal as exc:
        raise BudgetFrontierRefusal(str(exc)) from exc
    lane_rows = result.get("lane_summaries") or ()
    by_lane = {row.get("lanes"): row for row in lane_rows}
    checks = result.get("checks") or {}
    if (
        result.get("schema_version") != 1
        or result.get("gate") != source.GATE
        or result.get("experiment") != source.EXPERIMENT
        or result.get("status") != "PASS"
        or result.get("protocol") != protocol
        or result.get("scientific_decision") != SOURCE_DECISION
        or result.get("trajectory_registration_prerequisite_passed") is not False
        or result.get("capacity_closure_reclassified") is not False
        or result.get("retained_v5_capacity_established") is not False
        or result.get("retained_v5_closure_passed") is not False
        or not checks
        or not all(checks.values())
        or set(by_lane) != set(LANE_BUDGETS)
        or by_lane[8].get("fixed_lp_budget") != 36
        or by_lane[8].get("support_gate_passed") is not False
        or by_lane[8].get("actual_compute_gate_passed") is not True
        or by_lane[16].get("fixed_lp_budget") != 72
        or by_lane[16].get("support_gate_passed") is not True
        or by_lane[16].get("actual_compute_gate_passed") is not False
    ):
        raise BudgetFrontierRefusal("frozen v5 static diagnostic differs")
    return {
        "experiment": result["experiment"],
        "status": result["status"],
        "scientific_decision": result["scientific_decision"],
        "trajectory_registration_prerequisite_passed": False,
        "capacity_closure_reclassified": False,
        "common_capacity_scale": result["common_capacity_scale"],
        "normalized_load_contract": _json(result["normalized_load_contract"]),
        "physical_complete_frame_envelope": _json(
            result["physical_complete_frame_envelope"]
        ),
        "physical_envelope_capacity_normalized_loads": _json(
            result["physical_envelope_capacity_normalized_loads"]
        ),
        "parent_design_contract": _json(protocol["parent_design_contract"]),
        "source_lane_outcomes": [
            {
                "lanes": lanes,
                "fixed_lp_budget": by_lane[lanes]["fixed_lp_budget"],
                "maximum_candidate_epsilon": by_lane[lanes][
                    "maximum_candidate_epsilon"
                ],
                "maximum_actual_normalized_support_regret": by_lane[lanes][
                    "maximum_actual_normalized_support_regret"
                ],
                "candidate_action_matches_reference_count": by_lane[lanes][
                    "candidate_action_matches_reference_count"
                ],
                "support_gate_passed": by_lane[lanes]["support_gate_passed"],
                "actual_compute_gate_passed": by_lane[lanes][
                    "actual_compute_gate_passed"
                ],
            }
            for lanes in sorted(LANE_BUDGETS)
        ],
    }


def build_registration(
    *,
    source_registration: Path = DEFAULT_SOURCE_REGISTRATION,
    source_fixed_sources: Path = DEFAULT_SOURCE_FIXED_SOURCES,
    source_result: Path = DEFAULT_SOURCE_RESULT,
) -> dict[str, Any]:
    protocol = _read_object(Path(source_registration))
    result = _read_object(Path(source_result))
    if _read_object(Path(source_fixed_sources)) != source._current_fixed_sources():
        raise BudgetFrontierRefusal("frozen v5 diagnostic sources changed")
    contract = _source_contract(result, protocol)
    return {
        "schema_version": 1,
        "gate": GATE,
        "experiment": EXPERIMENT,
        "status": REGISTRATION_STATUS,
        "registered": True,
        "post_scout_development": True,
        "independent_confirmation": False,
        "source_paths": {
            "diagnostic_registration": str(Path(source_registration).resolve()),
            "diagnostic_fixed_sources": str(Path(source_fixed_sources).resolve()),
            "diagnostic_result": str(Path(source_result).resolve()),
        },
        "source_contract": contract,
        "lane_budget_grid": {
            str(lanes): list(budgets) for lanes, budgets in LANE_BUDGETS.items()
        },
        "exact_endpoint_budget": REFERENCE_BUDGET,
        "trace_rule": TRACE_RULE,
        "budget_selection_rule": dict(BUDGET_SELECTION_RULE),
        "expected_work": _json(EXPECTED_WORK),
        "decision_rule": dict(DECISION_RULE),
        "claim_boundary": CLAIM_BOUNDARY,
        "prohibited": PROHIBITED,
        "registration_scope": dict(REGISTRATION_SCOPE),
    }


def register(
    registration: Path = DEFAULT_REGISTRATION,
    fixed_sources: Path = DEFAULT_FIXED_SOURCES,
    **kwargs: Any,
) -> dict[str, Any]:
    if any(Path(path).exists() for path in (registration, fixed_sources, DEFAULT_RESULT)):
        raise BudgetFrontierRefusal("budget-frontier artifact already exists")
    protocol = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        **build_registration(**kwargs),
    }
    _write_new(Path(fixed_sources), _current_fixed_sources())
    _write_new(Path(registration), protocol)
    return protocol


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    expected_grid = {
        str(lanes): list(budgets) for lanes, budgets in LANE_BUDGETS.items()
    }
    if (
        protocol.get("schema_version") != 1
        or protocol.get("gate") != GATE
        or protocol.get("experiment") != EXPERIMENT
        or protocol.get("status") != REGISTRATION_STATUS
        or protocol.get("registered") is not True
        or protocol.get("post_scout_development") is not True
        or protocol.get("independent_confirmation") is not False
        or protocol.get("lane_budget_grid") != expected_grid
        or protocol.get("exact_endpoint_budget") != REFERENCE_BUDGET
        or protocol.get("trace_rule") != TRACE_RULE
        or protocol.get("budget_selection_rule") != BUDGET_SELECTION_RULE
        or protocol.get("expected_work") != EXPECTED_WORK
        or protocol.get("decision_rule") != DECISION_RULE
        or protocol.get("claim_boundary") != CLAIM_BOUNDARY
        or protocol.get("prohibited") != PROHIBITED
        or protocol.get("registration_scope") != REGISTRATION_SCOPE
    ):
        raise BudgetFrontierRefusal("registered budget-frontier protocol changed")


def _work_comparison(
    candidate: Mapping[str, int], reference: Mapping[str, int]
) -> dict[str, dict[str, Any]]:
    return {
        field: {
            "candidate": candidate[field],
            "exact_endpoint": reference[field],
            "candidate_fraction_of_endpoint": (
                candidate[field] / reference[field] if reference[field] else None
            ),
            "strictly_lower": candidate[field] < reference[field],
        }
        for field in WORK_GATE_FIELDS
    }


def _empty_budget_state() -> dict[str, Any]:
    return {
        "work": source.canonical._new_work(),
        "valid_candidate_count": 0,
        "regret_bounded_by_epsilon_count": 0,
        "score_exact_count": 0,
        "canonical_exact_count": 0,
        "action_match_count": 0,
        "maximum_epsilon": -1.0,
        "maximum_actual_regret": -1.0,
        "epsilon_witness": None,
        "regret_witness": None,
    }


def _screen_lane(
    model: Any,
    base_queues: Sequence[Sequence[int]],
    lanes: int,
    capacity_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, bool]]:
    budgets = LANE_BUDGETS[lanes]
    checkpoints = (*budgets, REFERENCE_BUDGET)
    factor = lanes // 2
    states = {budget: _empty_budget_state() for budget in budgets}
    reference_work = source.canonical._new_work()
    valid_references = 0

    for base_queue in base_queues:
        queue = tuple(factor * int(value) for value in base_queue)
        snapshots = trace_canonical_count_space_bnb_frame(
            model.options, queue, lanes, checkpoints
        )
        reference = snapshots[-1]
        reference_valid = (
            source.canonical._selection_reconstructs(model, queue, lanes, reference)
            and source.canonical._selection_work_holds(
                queue, lanes, REFERENCE_BUDGET, reference
            )
            and source.canonical._certificate_valid(reference, queue)
            and reference.score_exact
            and reference.canonical_exact
            and math.isclose(
                reference.score,
                reference.upper_bound,
                rel_tol=0.0,
                abs_tol=source.canonical.SCORE_TOLERANCE,
            )
        )
        valid_references += int(reference_valid)
        source.canonical._add_work(reference_work, reference)
        norm = sum(queue)

        for budget, candidate in zip(budgets, snapshots[:-1], strict=True):
            state = states[budget]
            valid = (
                source.canonical._selection_reconstructs(model, queue, lanes, candidate)
                and source.canonical._selection_work_holds(
                    queue, lanes, budget, candidate
                )
                and source.canonical._certificate_valid(candidate, queue)
            )
            regret = (reference.score - candidate.score) / norm
            bounded = (
                regret >= -source.canonical.SCORE_TOLERANCE
                and regret
                <= candidate.epsilon + source.canonical.SCORE_TOLERANCE
            )
            state["valid_candidate_count"] += int(valid)
            state["regret_bounded_by_epsilon_count"] += int(bounded)
            state["score_exact_count"] += int(candidate.score_exact)
            state["canonical_exact_count"] += int(candidate.canonical_exact)
            state["action_match_count"] += int(
                candidate.option_indices == reference.option_indices
            )
            source.canonical._add_work(state["work"], candidate)
            if candidate.epsilon > state["maximum_epsilon"]:
                state["maximum_epsilon"] = candidate.epsilon
                state["epsilon_witness"] = source.canonical._selection_witness(
                    base_queue, queue, candidate, reference, candidate.epsilon
                )
            if regret > state["maximum_actual_regret"]:
                state["maximum_actual_regret"] = regret
                state["regret_witness"] = source.canonical._selection_witness(
                    base_queue, queue, candidate, reference, regret
                )

    rows = []
    state_count = len(base_queues)
    for budget in budgets:
        state = states[budget]
        comparisons = _work_comparison(state["work"], reference_work)
        support_rows = []
        for capacity in capacity_rows:
            eta = float(capacity["effective_slack_eta"])
            epsilon_residual = eta - state["maximum_epsilon"]
            regret_residual = eta - state["maximum_actual_regret"]
            support_rows.append({
                "lanes": lanes,
                "load": float(capacity["load"]),
                "effective_slack_eta": eta,
                "maximum_candidate_epsilon": state["maximum_epsilon"],
                "maximum_actual_normalized_support_regret": state[
                    "maximum_actual_regret"
                ],
                "eta_minus_candidate_epsilon": epsilon_residual,
                "eta_minus_actual_support_regret": regret_residual,
                "support_gate_passed": (
                    bool(capacity["capacity_projection_passed"])
                    and epsilon_residual > source.canonical.SCORE_TOLERANCE
                    and regret_residual > source.canonical.SCORE_TOLERANCE
                ),
            })
        execution_passed = (
            state["valid_candidate_count"] == state_count
            and state["regret_bounded_by_epsilon_count"] == state_count
            and valid_references == state_count
        )
        support_passed = all(row["support_gate_passed"] for row in support_rows)
        compute_passed = all(row["strictly_lower"] for row in comparisons.values())
        action_identity = state["action_match_count"] == state_count
        support_compute = execution_passed and support_passed and compute_passed
        rows.append({
            "lanes": lanes,
            "lp_budget": budget,
            "state_count": state_count,
            "valid_candidate_certificate_count": state["valid_candidate_count"],
            "valid_exact_endpoint_count": valid_references,
            "regret_bounded_by_epsilon_count": state[
                "regret_bounded_by_epsilon_count"
            ],
            "candidate_score_exact_state_count": state["score_exact_count"],
            "candidate_canonical_exact_state_count": state[
                "canonical_exact_count"
            ],
            "candidate_action_matches_endpoint_count": state["action_match_count"],
            "maximum_candidate_epsilon": state["maximum_epsilon"],
            "maximum_actual_normalized_support_regret": state[
                "maximum_actual_regret"
            ],
            "epsilon_witness": state["epsilon_witness"],
            "support_regret_witness": state["regret_witness"],
            "support_by_physical_load": support_rows,
            "candidate_actual_work": state["work"],
            "exact_endpoint_actual_work": reference_work,
            "actual_compute_comparison": comparisons,
            "execution_gate_passed": execution_passed,
            "support_gate_passed": support_passed,
            "actual_compute_gate_passed": compute_passed,
            "development_roster_exact_canonical_action_identity_gate_passed": (
                action_identity
            ),
            "support_compute_eligible": support_compute,
            "development_roster_action_identity_confirmation_eligible": (
                support_compute and action_identity
            ),
        })

    support_choices = [
        row["lp_budget"] for row in rows if row["support_compute_eligible"]
    ]
    action_identity_choices = [
        row["lp_budget"]
        for row in rows
        if row["development_roster_action_identity_confirmation_eligible"]
    ]
    checks = {
        "state_count_is_512": state_count == 512,
        "every_exact_endpoint_is_valid_score_exact_and_canonical_exact": (
            valid_references == state_count
        ),
        "every_registered_candidate_execution_is_valid": all(
            row["execution_gate_passed"] for row in rows
        ),
    }
    return ({
        "lanes": lanes,
        "queue_scale": factor,
        "registered_budgets": list(budgets),
        "exact_endpoint_budget": REFERENCE_BUDGET,
        "state_count": state_count,
        "budget_rows": rows,
        "minimum_support_compute_budget": min(support_choices, default=None),
        "minimum_development_roster_action_identity_budget": min(
            action_identity_choices, default=None
        ),
        "confirmation_budget": min(action_identity_choices, default=None),
        "checks": checks,
    }, checks)


def _decision(execution_passed: bool, lane_rows: Sequence[Mapping[str, Any]]) -> str:
    if not execution_passed:
        return INVALID_DECISION
    if any(row.get("minimum_support_compute_budget") is None for row in lane_rows):
        return NO_PAIR_DECISION
    if any(
        row.get("minimum_development_roster_action_identity_budget") is None
        for row in lane_rows
    ):
        return SUPPORT_ONLY_DECISION
    return PASS_DECISION


def run_registered(
    registration: Path = DEFAULT_REGISTRATION,
    fixed_sources: Path = DEFAULT_FIXED_SOURCES,
    output: Path = DEFAULT_RESULT,
) -> dict[str, Any]:
    output = Path(output)
    if output.exists():
        raise BudgetFrontierRefusal(f"budget-frontier result already exists: {output}")
    protocol = _read_object(Path(registration))
    _validate_protocol(protocol)
    if _read_object(Path(fixed_sources)) != _current_fixed_sources():
        raise BudgetFrontierRefusal("registered budget-frontier sources changed")

    paths = protocol["source_paths"]
    source_protocol = _read_object(Path(paths["diagnostic_registration"]))
    source_result = _read_object(Path(paths["diagnostic_result"]))
    if (
        _read_object(Path(paths["diagnostic_fixed_sources"]))
        != source._current_fixed_sources()
        or _source_contract(source_result, source_protocol)
        != protocol["source_contract"]
    ):
        raise BudgetFrontierRefusal("frozen v5 diagnostic changed after registration")

    model = source.capacity.base._model_from_snapshot(
        source_result["physical_complete_frame_envelope"]["model"]
    )
    if (
        len(model.options) != source.canonical.OPTION_COUNT
        or len(set(max(1.0, option.duration) for option in model.options))
        != source.canonical.OPTION_COUNT
    ):
        raise BudgetFrontierRefusal("v5 envelope does not retain nine duration roots")
    base_queues = source.parent._load_reused_queues(
        source_protocol["parent_design_contract"]["queue_roster_contract"]
    )
    capacity_by_lane = source_result[
        "physical_envelope_capacity_normalized_loads"
    ]

    started = time.perf_counter()
    lane_rows = []
    lane_checks = []
    for lanes in sorted(LANE_BUDGETS):
        row, checks = _screen_lane(
            model, base_queues, lanes, capacity_by_lane[str(lanes)]
        )
        lane_rows.append(row)
        lane_checks.append(checks)
    checks = {
        "source_static_diagnostic_is_valid_and_unchanged": True,
        "source_negative_capacity_decision_retained": True,
        "seed13041_development_roster_has_512_states": len(base_queues) == 512,
        "every_lane_execution_check_passes": all(
            all(row.values()) for row in lane_checks
        ),
        "cumulative_trace_count_is_exact": sum(
            row["state_count"] for row in lane_rows
        )
        == 1024,
        "no_arrivals_trajectories_rng_or_server_pulls": True,
    }
    execution_passed = all(checks.values())
    decision = _decision(execution_passed, lane_rows)
    selected = {
        str(row["lanes"]): row["confirmation_budget"] for row in lane_rows
    }
    report = {
        "schema_version": 1,
        "gate": GATE,
        "experiment": EXPERIMENT,
        "status": "PASS" if execution_passed else "FAIL",
        "protocol": protocol,
        "checks": checks,
        "source_scientific_decision": SOURCE_DECISION,
        "retained_v5_capacity_established": False,
        "retained_v5_closure_passed": False,
        "capacity_closure_reclassified": False,
        "lane_frontiers": lane_rows,
        "selected_confirmation_budget_by_lane": selected,
        "scientific_decision": decision,
        "independent_static_confirmation_registration_authorized": (
            decision == PASS_DECISION
        ),
        "trajectory_registration_prerequisite_passed": False,
        "independent_confirmation_completed": False,
        "trajectory_runs": 0,
        "arrival_generations": 0,
        "rng_draws": 0,
        "server_pulls": 0,
        "claim_boundary": CLAIM_BOUNDARY,
        "timing": {
            "wall_s": time.perf_counter() - started,
            "scientific_gate": False,
        },
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
    except BudgetFrontierRefusal as exc:
        raise SystemExit(f"REFUSED: {exc}") from exc
    print(json.dumps({
        "status": payload["status"],
        "experiment": payload["experiment"],
        "scientific_decision": payload.get("scientific_decision"),
        "selected_confirmation_budget_by_lane": payload.get(
            "selected_confirmation_budget_by_lane"
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
