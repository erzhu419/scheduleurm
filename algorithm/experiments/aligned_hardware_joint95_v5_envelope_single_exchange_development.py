"""Post-result development replay of the R8/K48 single-exchange polish.

The method was designed after the seed-260919 confirmation exposed one R8/K48
action mismatch.  This runner therefore consumes both seed-13041 and seed-
260919 rosters as development data.  It applies the fixed one-exchange rule to
R8/K48 only, charges every unique count-vector exchange attempt, and compares
against separate cold K4096 references.  R16/K9 is unchanged and retained from
the two frozen source results.  No new roster, arrival, or trajectory is built.
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
    select_canonical_count_space_bnb_frame,
)
from algorithm.canonical_count_space_bnb_single_exchange_frame_candidates import (
    select_canonical_count_space_bnb_single_exchange_frame,
    total_work_row,
)
from algorithm.experiments import (
    aligned_hardware_joint95_v5_envelope_budget_pair_confirmation as source,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
STEM = "aligned_hardware_joint95_v5_envelope_single_exchange_development_20260919"
DEFAULT_REGISTRATION = ARTIFACTS / f"{STEM}_protocol.json"
DEFAULT_FIXED_SOURCES = ARTIFACTS / f"{STEM}_fixed_sources.json"
DEFAULT_RESULT = ARTIFACTS / f"{STEM}.json"
DEFAULT_SOURCE_REGISTRATION = source.DEFAULT_REGISTRATION
DEFAULT_SOURCE_ROSTER = source.DEFAULT_ROSTER
DEFAULT_SOURCE_FIXED_SOURCES = source.DEFAULT_FIXED_SOURCES
DEFAULT_SOURCE_RESULT = source.DEFAULT_RESULT

canonical = source.canonical
EXPERIMENT = "aligned_hardware_joint95_v5_envelope_single_exchange_development_v1"
GATE = "aligned_hardware_joint95_v5_envelope_single_exchange_development_runner"
REGISTRATION_STATUS = "REGISTERED_POSTRESULT_AWAITING_DEVELOPMENT_REPLAY"
SOURCE_DECISION = source.ACTION_FAIL_DECISION
PASS_DECISION = "SINGLE_EXCHANGE_READY_FOR_NEW_INDEPENDENT_STATIC_CONFIRMATION"
INVALID_DECISION = "INVALID_EXECUTION"
SUPPORT_FAIL_DECISION = "SINGLE_EXCHANGE_DEVELOPMENT_SUPPORT_FAIL_STOP"
ACTION_FAIL_DECISION = "SINGLE_EXCHANGE_DEVELOPMENT_ACTION_IDENTITY_FAIL_STOP"
COMPUTE_FAIL_DECISION = "SINGLE_EXCHANGE_DEVELOPMENT_COMPUTE_FAIL_STOP"
R8_LANES = 8
R8_BUDGET = 48
R16_LANES = 16
R16_BUDGET = 9
REFERENCE_BUDGET = 4096
ROSTER_SIZE = 512
ROSTER_NAMES = ("seed13041", "seed260919")
THIS_SOURCE = (
    "algorithm/experiments/"
    "aligned_hardware_joint95_v5_envelope_single_exchange_development.py"
)
POLISH_SOURCE = "algorithm/canonical_count_space_bnb_single_exchange_frame_candidates.py"
METHOD_CONTRACT = {
    "lane": R8_LANES,
    "base_lp_budget": R8_BUDGET,
    "trigger": "base score_exact is false",
    "neighborhood": (
        "every unique option-count vector obtained by moving one lane from each "
        "occupied real-option or idle symbol to each different real-option or idle symbol"
    ),
    "choice": (
        "maximize exact Fraction support score; break score ties by ascending "
        "(duration, sorted option indices with idle last)"
    ),
    "reference_independence": (
        "the polish reads no K4096 action, score, bound, or work value"
    ),
    "certificate": (
        "retain the base certified upper bound and exactness flags; update only the "
        "feasible incumbent action, score, gap, and epsilon"
    ),
    "charged_work": (
        "add every unique exchange attempt, including infeasible attempts, to "
        "candidate evaluations and residual option probes; add no LP or matrix work"
    ),
    "r16_disposition": "R16/K9 is unchanged and no polish is applied",
}
EXPECTED_WORK = {
    "development_rosters": 2,
    "states_per_roster": ROSTER_SIZE,
    "r8_candidate_calls": 2 * ROSTER_SIZE,
    "r8_cold_reference_calls": 2 * ROSTER_SIZE,
    "r16_selector_calls": 0,
    "run_rng_initializations": 0,
    "arrival_generations": 0,
    "trajectory_runs": 0,
    "server_pulls": 0,
}
REGISTRATION_SCOPE = {
    "failed_confirmation_outcome_read": True,
    "both_consumed_rosters_read": True,
    "selector_invoked": False,
    "reference_invoked": False,
    "new_roster_generated": False,
    "trajectory_built": False,
    "rng_draws": 0,
    "server_pulls": 0,
    "paper_edited": False,
}
DECISION_RULE = {
    INVALID_DECISION: "one or more registered execution-integrity checks fail",
    SUPPORT_FAIL_DECISION: (
        "the polished R8/K48 certificate or actual regret fails an envelope-normalized "
        "support gate on at least one consumed roster"
    ),
    ACTION_FAIL_DECISION: (
        "support passes but a polished R8/K48 action differs from the exact canonical "
        "reference on at least one consumed-roster state"
    ),
    COMPUTE_FAIL_DECISION: (
        "support and action identity pass but charged LP, matrix, or candidate-"
        "evaluation work is not strictly below cold K4096 per roster or in aggregate"
    ),
    PASS_DECISION: (
        "R8/K48 plus the frozen single-exchange polish passes support, action identity, "
        "and charged-work gates on both consumed rosters and in aggregate; retained "
        "R16/K9 source evidence remains passing"
    ),
}
CLAIM_BOUNDARY = (
    "An outcome-informed development replay on two consumed 512-state rosters. A pass "
    "can authorize preregistration of one new disjoint static confirmation for the "
    "fixed K48-plus-polish/K9 method. It is not independent confirmation, trajectory "
    "authorization, candidate-side canonical exactness, all-queue policy equivalence, "
    "capacity at the old point loads, stability, or a physical multi-GPU result."
)
PROHIBITED = (
    "changing the trigger, neighborhood, tie-break, work accounting, budgets, loads, "
    "or consumed rosters after registration; reading K4096 inside the polish; using "
    "these consumed rosters as confirmation; generating a new roster; arrivals, "
    "trajectories, run-time RNG, server pulls, and paper edits"
)


class SingleExchangeDevelopmentRefusal(RuntimeError):
    """The registered post-result development contract is not satisfied."""


def _json(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SingleExchangeDevelopmentRefusal(
            f"cannot read JSON object {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise SingleExchangeDevelopmentRefusal(f"JSON root must be an object: {path}")
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
        POLISH_SOURCE: (ROOT / POLISH_SOURCE).read_text(encoding="utf-8"),
        THIS_SOURCE: (ROOT / THIS_SOURCE).read_text(encoding="utf-8"),
    }


def _validated_source_contract(
    result: Mapping[str, Any],
    protocol: Mapping[str, Any],
    source_roster: Path,
) -> dict[str, Any]:
    try:
        source._validate_protocol(protocol)
    except source.BudgetPairConfirmationRefusal as exc:
        raise SingleExchangeDevelopmentRefusal(str(exc)) from exc
    r8 = next((row for row in result.get("lane_summaries") or () if row.get("lanes") == 8), None)
    r16 = next((row for row in result.get("lane_summaries") or () if row.get("lanes") == 16), None)
    if (
        result.get("schema_version") != 1
        or result.get("gate") != source.GATE
        or result.get("experiment") != source.EXPERIMENT
        or result.get("status") != "PASS"
        or result.get("protocol") != protocol
        or result.get("scientific_decision") != SOURCE_DECISION
        or result.get("independent_static_confirmation_passed") is not False
        or result.get("trajectory_registration_prerequisite_passed") is not False
        or result.get("capacity_closure_reclassified") is not False
        or not result.get("checks")
        or not all(result["checks"].values())
        or r8 is None
        or r8.get("fixed_lp_budget") != R8_BUDGET
        or r8.get("support_gate_passed") is not True
        or r8.get("actual_compute_gate_passed") is not True
        or r8.get("independent_roster_action_identity_gate_passed") is not False
        or r8.get("candidate_action_mismatch_count") != 1
        or r8.get("positive_regret_action_mismatch_count") != 1
        or r16 is None
        or r16.get("fixed_lp_budget") != R16_BUDGET
        or r16.get("support_gate_passed") is not True
        or r16.get("actual_compute_gate_passed") is not True
        or r16.get("independent_roster_action_identity_gate_passed") is not True
        or r16.get("candidate_action_matches_reference_count") != ROSTER_SIZE
        or r16.get("lane_confirmation_gate_passed") is not True
    ):
        raise SingleExchangeDevelopmentRefusal(
            "frozen failed independent confirmation differs"
        )
    development_path = Path(protocol["source_paths"]["development_result"])
    development_result = _read_object(development_path)
    development_protocol = _read_object(
        Path(protocol["source_paths"]["development_registration"])
    )
    try:
        development_contract = source._validated_development_contract(
            development_result, development_protocol
        )
    except source.BudgetPairConfirmationRefusal as exc:
        raise SingleExchangeDevelopmentRefusal(str(exc)) from exc
    if development_contract != protocol.get("development_contract"):
        raise SingleExchangeDevelopmentRefusal(
            "embedded development contract differs from its authoritative source"
        )
    dev_r16 = next(
        row
        for row in development_result["lane_frontiers"]
        if row["lanes"] == R16_LANES
    )
    dev_r16_k9 = next(row for row in dev_r16["budget_rows"] if row["lp_budget"] == 9)
    if (
        dev_r16_k9.get("support_gate_passed") is not True
        or dev_r16_k9.get("actual_compute_gate_passed") is not True
        or dev_r16_k9.get(
            "development_roster_exact_canonical_action_identity_gate_passed"
        )
        is not True
    ):
        raise SingleExchangeDevelopmentRefusal("retained seed13041 R16/K9 evidence differs")
    return {
        "experiment": result["experiment"],
        "status": result["status"],
        "scientific_decision": result["scientific_decision"],
        "independent_static_confirmation_passed": False,
        "trajectory_registration_prerequisite_passed": False,
        "capacity_closure_reclassified": False,
        "physical_complete_frame_envelope": _json(
            development_contract["physical_complete_frame_envelope"]
        ),
        "physical_envelope_capacity_normalized_loads": _json(
            development_contract["physical_envelope_capacity_normalized_loads"]
        ),
        "seed13041_roster_contract": _json(
            development_contract["development_roster_contract"]
        ),
        "seed260919_roster_contract": {
            "path": str(Path(source_roster).resolve()),
            "seed": protocol["queue_roster_contract"]["seed"],
            "size": ROSTER_SIZE,
            "queues": _json(protocol["queue_roster_contract"]["queues"]),
            "payload": _json(protocol["queue_roster_contract"]),
        },
        "retained_r16_evidence": {
            "seed13041": {
                "fixed_lp_budget": 9,
                "maximum_candidate_epsilon": dev_r16_k9[
                    "maximum_candidate_epsilon"
                ],
                "candidate_action_matches_reference_count": dev_r16_k9[
                    "candidate_action_matches_endpoint_count"
                ],
                "support_gate_passed": True,
                "actual_compute_gate_passed": True,
            },
            "seed260919": {
                "fixed_lp_budget": 9,
                "maximum_candidate_epsilon": r16["maximum_candidate_epsilon"],
                "candidate_action_matches_reference_count": r16[
                    "candidate_action_matches_reference_count"
                ],
                "support_gate_passed": True,
                "actual_compute_gate_passed": True,
            },
        },
        "development_source_contract": development_contract,
    }


def _load_consumed_rosters(
    contract: Mapping[str, Any],
) -> dict[str, tuple[tuple[int, int, int], ...]]:
    rows = {}
    for name in ROSTER_NAMES:
        roster_contract = contract[f"{name}_roster_contract"]
        payload = _read_object(Path(roster_contract["path"]))
        queues = tuple(tuple(int(value) for value in row) for row in payload.get("queues") or ())
        frozen_payload = roster_contract.get("payload")
        if (
            payload.get("seed") != roster_contract.get("seed")
            or [list(queue) for queue in queues] != roster_contract.get("queues")
            or len(queues) != ROSTER_SIZE
            or len(set(queues)) != ROSTER_SIZE
            or (frozen_payload is not None and payload != frozen_payload)
        ):
            raise SingleExchangeDevelopmentRefusal(f"consumed roster changed: {name}")
        rows[name] = queues
    return rows


def build_registration(
    *,
    source_registration: Path = DEFAULT_SOURCE_REGISTRATION,
    source_roster: Path = DEFAULT_SOURCE_ROSTER,
    source_fixed_sources: Path = DEFAULT_SOURCE_FIXED_SOURCES,
    source_result: Path = DEFAULT_SOURCE_RESULT,
) -> dict[str, Any]:
    protocol = _read_object(Path(source_registration))
    result = _read_object(Path(source_result))
    if _read_object(Path(source_fixed_sources)) != source._current_fixed_sources():
        raise SingleExchangeDevelopmentRefusal("frozen confirmation sources changed")
    contract = _validated_source_contract(result, protocol, Path(source_roster))
    source_roster_payload = _read_object(Path(source_roster))
    if source_roster_payload != contract["seed260919_roster_contract"]["payload"]:
        raise SingleExchangeDevelopmentRefusal("seed260919 source roster changed")
    rosters = _load_consumed_rosters(contract)
    if any(len(rows) != ROSTER_SIZE for rows in rosters.values()):
        raise SingleExchangeDevelopmentRefusal("consumed roster size changed")
    return {
        "schema_version": 1,
        "gate": GATE,
        "experiment": EXPERIMENT,
        "status": REGISTRATION_STATUS,
        "registered": True,
        "post_result_development": True,
        "independent_confirmation": False,
        "source_paths": {
            "failed_confirmation_registration": str(Path(source_registration).resolve()),
            "failed_confirmation_roster": str(Path(source_roster).resolve()),
            "failed_confirmation_fixed_sources": str(Path(source_fixed_sources).resolve()),
            "failed_confirmation_result": str(Path(source_result).resolve()),
        },
        "source_contract": contract,
        "method_contract": dict(METHOD_CONTRACT),
        "consumed_rosters": {
            name: [list(row) for row in rows] for name, rows in rosters.items()
        },
        "loads": list(canonical.LOADS),
        "expected_work": dict(EXPECTED_WORK),
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
        raise SingleExchangeDevelopmentRefusal("development artifact already exists")
    protocol = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        **build_registration(**kwargs),
    }
    _write_new(Path(fixed_sources), _current_fixed_sources())
    _write_new(Path(registration), protocol)
    return protocol


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    rosters = protocol.get("consumed_rosters") or {}
    if (
        protocol.get("schema_version") != 1
        or protocol.get("gate") != GATE
        or protocol.get("experiment") != EXPERIMENT
        or protocol.get("status") != REGISTRATION_STATUS
        or protocol.get("registered") is not True
        or protocol.get("post_result_development") is not True
        or protocol.get("independent_confirmation") is not False
        or protocol.get("method_contract") != METHOD_CONTRACT
        or set(rosters) != set(ROSTER_NAMES)
        or any(len(rows) != ROSTER_SIZE for rows in rosters.values())
        or protocol.get("loads") != list(canonical.LOADS)
        or protocol.get("expected_work") != EXPECTED_WORK
        or protocol.get("decision_rule") != DECISION_RULE
        or protocol.get("claim_boundary") != CLAIM_BOUNDARY
        or protocol.get("prohibited") != PROHIBITED
        or protocol.get("registration_scope") != REGISTRATION_SCOPE
    ):
        raise SingleExchangeDevelopmentRefusal(
            "registered single-exchange protocol changed"
        )


def _add_work(target: dict[str, int], work: Mapping[str, int]) -> None:
    for field in canonical.WORK_FIELDS:
        target[field] += int(work[field])


def _screen_roster(
    model: Any,
    base_queues: Sequence[Sequence[int]],
    capacity_rows: Sequence[Mapping[str, Any]],
    roster_name: str,
) -> tuple[dict[str, Any], dict[str, bool]]:
    candidate_work = canonical._new_work()
    base_candidate_work = canonical._new_work()
    reference_work = canonical._new_work()
    valid_candidates = 0
    valid_references = 0
    bounded_regrets = 0
    action_matches = 0
    max_epsilon = -1.0
    max_regret = -1.0
    epsilon_witness = None
    regret_witness = None
    trigger_count = 0
    attempts = 0
    feasible = 0
    changed = 0

    for base_queue in base_queues:
        queue = tuple(4 * int(value) for value in base_queue)
        polished = select_canonical_count_space_bnb_single_exchange_frame(
            model.options, queue, R8_LANES, R8_BUDGET
        )
        candidate = polished.selection
        reference = select_canonical_count_space_bnb_frame(
            model.options, queue, R8_LANES, REFERENCE_BUDGET
        )
        candidate_valid = (
            canonical._selection_work_holds(
                queue, R8_LANES, R8_BUDGET, polished.base_selection
            )
            and canonical._selection_reconstructs(model, queue, R8_LANES, candidate)
            and canonical._certificate_valid(candidate, queue)
        )
        reference_valid = (
            canonical._selection_work_holds(
                queue, R8_LANES, REFERENCE_BUDGET, reference
            )
            and canonical._selection_reconstructs(model, queue, R8_LANES, reference)
            and canonical._certificate_valid(reference, queue)
            and reference.score_exact
            and reference.canonical_exact
            and math.isclose(
                reference.score,
                reference.upper_bound,
                rel_tol=0.0,
                abs_tol=canonical.SCORE_TOLERANCE,
            )
        )
        regret = (reference.score - candidate.score) / sum(queue)
        bounded = (
            regret >= -canonical.SCORE_TOLERANCE
            and regret <= candidate.epsilon + canonical.SCORE_TOLERANCE
        )
        valid_candidates += int(candidate_valid)
        valid_references += int(reference_valid)
        bounded_regrets += int(bounded)
        action_matches += int(candidate.option_indices == reference.option_indices)
        trigger_count += int(polished.polish_triggered)
        attempts += polished.exchange_attempts
        feasible += polished.feasible_exchange_count
        changed += int(polished.incumbent_changed)
        canonical._add_work(base_candidate_work, polished.base_selection)
        _add_work(candidate_work, total_work_row(polished))
        canonical._add_work(reference_work, reference)
        if candidate.epsilon > max_epsilon:
            max_epsilon = candidate.epsilon
            epsilon_witness = canonical._selection_witness(
                base_queue, queue, candidate, reference, candidate.epsilon
            )
        if regret > max_regret:
            max_regret = regret
            regret_witness = canonical._selection_witness(
                base_queue, queue, candidate, reference, regret
            )

    comparisons = source.development._work_comparison(candidate_work, reference_work)
    support_rows = []
    for capacity in capacity_rows:
        eta = float(capacity["effective_slack_eta"])
        epsilon_residual = eta - max_epsilon
        regret_residual = eta - max_regret
        support_rows.append({
            "load": float(capacity["load"]),
            "effective_slack_eta": eta,
            "maximum_candidate_epsilon": max_epsilon,
            "maximum_actual_normalized_support_regret": max_regret,
            "eta_minus_candidate_epsilon": epsilon_residual,
            "eta_minus_actual_support_regret": regret_residual,
            "support_gate_passed": (
                bool(capacity["capacity_projection_passed"])
                and epsilon_residual > canonical.SCORE_TOLERANCE
                and regret_residual > canonical.SCORE_TOLERANCE
            ),
        })
    expected_charged_work = dict(base_candidate_work)
    expected_charged_work["candidate_evaluations"] += attempts
    expected_charged_work["residual_fill_option_probes"] += attempts
    checks = {
        "state_count_is_512": len(base_queues) == ROSTER_SIZE,
        "every_polished_candidate_is_valid": valid_candidates == ROSTER_SIZE,
        "every_reference_is_valid_score_exact_and_canonical_exact": (
            valid_references == ROSTER_SIZE
        ),
        "every_regret_is_nonnegative_and_bounded_by_epsilon": (
            bounded_regrets == ROSTER_SIZE
        ),
        "polish_attempts_are_fully_charged": candidate_work == expected_charged_work,
    }
    support_passed = all(row["support_gate_passed"] for row in support_rows)
    action_passed = action_matches == ROSTER_SIZE
    compute_passed = all(row["strictly_lower"] for row in comparisons.values())
    return ({
        "roster": roster_name,
        "seed": 13041 if roster_name == "seed13041" else 260919,
        "state_count": len(base_queues),
        "lanes": R8_LANES,
        "base_lp_budget": R8_BUDGET,
        "cold_reference_budget": REFERENCE_BUDGET,
        "polish_trigger_state_count": trigger_count,
        "exchange_attempt_count": attempts,
        "feasible_exchange_count": feasible,
        "changed_incumbent_state_count": changed,
        "candidate_action_matches_reference_count": action_matches,
        "maximum_candidate_epsilon": max_epsilon,
        "maximum_actual_normalized_support_regret": max_regret,
        "epsilon_witness": epsilon_witness,
        "support_regret_witness": regret_witness,
        "support_by_load": support_rows,
        "candidate_charged_work": candidate_work,
        "cold_reference_actual_work": reference_work,
        "actual_compute_comparison": comparisons,
        "support_gate_passed": support_passed,
        "action_identity_gate_passed": action_passed,
        "charged_compute_gate_passed": compute_passed,
        "development_gate_passed": (
            all(checks.values()) and support_passed and action_passed and compute_passed
        ),
        "checks": checks,
    }, checks)


def _decision(
    execution_passed: bool,
    rows: Sequence[Mapping[str, Any]],
    aggregate_compute_passed: bool,
    retained_r16_passed: bool,
) -> str:
    if not execution_passed or not retained_r16_passed:
        return INVALID_DECISION
    if not all(row["support_gate_passed"] for row in rows):
        return SUPPORT_FAIL_DECISION
    if not all(row["action_identity_gate_passed"] for row in rows):
        return ACTION_FAIL_DECISION
    if not aggregate_compute_passed or not all(
        row["charged_compute_gate_passed"] for row in rows
    ):
        return COMPUTE_FAIL_DECISION
    return PASS_DECISION


def run_registered(
    registration: Path = DEFAULT_REGISTRATION,
    fixed_sources: Path = DEFAULT_FIXED_SOURCES,
    output: Path = DEFAULT_RESULT,
) -> dict[str, Any]:
    output = Path(output)
    if output.exists():
        raise SingleExchangeDevelopmentRefusal(f"development result exists: {output}")
    protocol = _read_object(Path(registration))
    _validate_protocol(protocol)
    if _read_object(Path(fixed_sources)) != _current_fixed_sources():
        raise SingleExchangeDevelopmentRefusal("registered development sources changed")
    paths = protocol["source_paths"]
    source_protocol = _read_object(Path(paths["failed_confirmation_registration"]))
    source_result = _read_object(Path(paths["failed_confirmation_result"]))
    if (
        _read_object(Path(paths["failed_confirmation_fixed_sources"]))
        != source._current_fixed_sources()
        or _validated_source_contract(
            source_result,
            source_protocol,
            Path(paths["failed_confirmation_roster"]),
        )
        != protocol["source_contract"]
    ):
        raise SingleExchangeDevelopmentRefusal("failed confirmation changed after registration")
    rosters = _load_consumed_rosters(protocol["source_contract"])
    if {
        name: [list(row) for row in rows] for name, rows in rosters.items()
    } != protocol["consumed_rosters"]:
        raise SingleExchangeDevelopmentRefusal("consumed roster snapshot changed")
    contract = protocol["source_contract"]
    model = source.development.source.capacity.base._model_from_snapshot(
        contract["physical_complete_frame_envelope"]["model"]
    )
    capacity_rows = contract["physical_envelope_capacity_normalized_loads"]["8"]

    started = time.perf_counter()
    rows = []
    row_checks = []
    for name in ROSTER_NAMES:
        row, checks = _screen_roster(model, rosters[name], capacity_rows, name)
        rows.append(row)
        row_checks.append(checks)
    candidate_total = canonical._new_work()
    reference_total = canonical._new_work()
    for row in rows:
        _add_work(candidate_total, row["candidate_charged_work"])
        _add_work(reference_total, row["cold_reference_actual_work"])
    aggregate_comparisons = source.development._work_comparison(
        candidate_total, reference_total
    )
    aggregate_compute_passed = all(
        row["strictly_lower"] for row in aggregate_comparisons.values()
    )
    retained_r16 = protocol["source_contract"]["retained_r16_evidence"]
    retained_r16_passed = all(
        row["candidate_action_matches_reference_count"] == ROSTER_SIZE
        and row["support_gate_passed"]
        and row["actual_compute_gate_passed"]
        for row in retained_r16.values()
    )
    checks = {
        "failed_confirmation_is_valid_and_unchanged": True,
        "both_consumed_rosters_are_exact_and_unchanged": all(
            len(rows_) == ROSTER_SIZE for rows_ in rosters.values()
        ),
        "every_r8_execution_check_passes": all(
            all(check.values()) for check in row_checks
        ),
        "retained_r16_k9_source_evidence_passes": retained_r16_passed,
        "r8_candidate_and_reference_call_counts_are_exact": sum(
            row["state_count"] for row in rows
        )
        == 1024,
        "no_new_roster_arrivals_trajectories_rng_or_server_pulls": True,
    }
    execution_passed = all(checks.values())
    decision = _decision(
        execution_passed, rows, aggregate_compute_passed, retained_r16_passed
    )
    passed = decision == PASS_DECISION
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
        "r8_consumed_roster_summaries": rows,
        "r8_aggregate_charged_compute": {
            "candidate": candidate_total,
            "cold_reference": reference_total,
            "comparison": aggregate_comparisons,
            "gate_passed": aggregate_compute_passed,
        },
        "retained_r16_k9_evidence": retained_r16,
        "scientific_decision": decision,
        "new_independent_static_confirmation_registration_authorized": passed,
        "trajectory_registration_prerequisite_passed": False,
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
    except SingleExchangeDevelopmentRefusal as exc:
        raise SystemExit(f"REFUSED: {exc}") from exc
    print(json.dumps({
        "status": payload["status"],
        "experiment": payload["experiment"],
        "scientific_decision": payload.get("scientific_decision"),
        "new_independent_static_confirmation_registration_authorized": payload.get(
            "new_independent_static_confirmation_registration_authorized"
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
