"""Post-result development replay of the R16/K9 bulk canonical-tie polish.

The fixed method was designed after the seed-270919 independent confirmation
exposed three zero-regret R16/K9 action ties.  This runner consumes the three
already used seed-13041, seed-260919, and seed-270919 rosters.  It applies the
fixed seven-pair bulk substitution rule to R16/K9, charges every structural
pair check, and compares against separate cold K4096 references.  Passing can
authorize one new independent static registration; no new roster or trajectory
is created here.
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
from algorithm.canonical_score_tie_bulk_substitution_frame_candidates import (
    select_canonical_score_tie_bulk_substitution_frame,
    total_work_row,
)
from algorithm.experiments import (
    aligned_hardware_joint95_v5_envelope_single_exchange_confirmation as source,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
STEM = "aligned_hardware_joint95_v5_envelope_r16_bulk_tie_development_20260919"
DEFAULT_REGISTRATION = ARTIFACTS / f"{STEM}_protocol.json"
DEFAULT_FIXED_SOURCES = ARTIFACTS / f"{STEM}_fixed_sources.json"
DEFAULT_RESULT = ARTIFACTS / f"{STEM}.json"
DEFAULT_SOURCE_REGISTRATION = source.DEFAULT_REGISTRATION
DEFAULT_SOURCE_ROSTER = source.DEFAULT_ROSTER
DEFAULT_SOURCE_FIXED_SOURCES = source.DEFAULT_FIXED_SOURCES
DEFAULT_SOURCE_RESULT = source.DEFAULT_RESULT

canonical = source.canonical
EXPERIMENT = "aligned_hardware_joint95_v5_envelope_r16_bulk_tie_development_v1"
GATE = "aligned_hardware_joint95_v5_envelope_r16_bulk_tie_development_runner"
REGISTRATION_STATUS = "REGISTERED_POSTRESULT_AWAITING_DEVELOPMENT_REPLAY"
SOURCE_DECISION = source.ACTION_FAIL_DECISION
PASS_DECISION = "R16_BULK_TIE_READY_FOR_NEW_INDEPENDENT_STATIC_CONFIRMATION"
INVALID_DECISION = "INVALID_EXECUTION"
SUPPORT_FAIL_DECISION = "R16_BULK_TIE_DEVELOPMENT_SUPPORT_FAIL_STOP"
ACTION_FAIL_DECISION = "R16_BULK_TIE_DEVELOPMENT_ACTION_IDENTITY_FAIL_STOP"
COMPUTE_FAIL_DECISION = "R16_BULK_TIE_DEVELOPMENT_COMPUTE_FAIL_STOP"
R16_LANES = 16
R16_BUDGET = 9
REFERENCE_BUDGET = 4096
ROSTER_SIZE = 512
ROSTER_NAMES = ("seed13041", "seed260919", "seed270919")
THIS_SOURCE = (
    "algorithm/experiments/"
    "aligned_hardware_joint95_v5_envelope_r16_bulk_tie_development.py"
)
BULK_SOURCE = "algorithm/canonical_score_tie_bulk_substitution_frame_candidates.py"
METHOD_CONTRACT = {
    "lane": R16_LANES,
    "base_lp_budget": R16_BUDGET,
    "trigger": "base score_exact is true and canonical_exact is false",
    "structural_pairs": (
        "all destination < source option pairs whose integer requirement sums match; "
        "the frozen nine-option model yields exactly seven pairs"
    ),
    "candidate": (
        "one pass; for each active source replace all copies with the destination, "
        "then retain the feasible exact-score candidate with the smallest canonical key"
    ),
    "reference_independence": (
        "the polish reads no K4096 action, score, bound, or work value"
    ),
    "certificate": (
        "retain the K9 upper bound and exactness flags; update only a feasible incumbent"
    ),
    "charged_work": (
        "charge every structural pair check, including inactive and infeasible pairs, "
        "to candidate evaluations and residual option probes"
    ),
    "r8_disposition": "the fixed R8/K48 single-exchange method is unchanged",
}
EXPECTED_WORK = {
    "development_rosters": 3,
    "states_per_roster": ROSTER_SIZE,
    "r16_candidate_calls": 3 * ROSTER_SIZE,
    "r16_cold_reference_calls": 3 * ROSTER_SIZE,
    "r8_selector_calls": 0,
    "run_rng_initializations": 0,
    "arrival_generations": 0,
    "trajectory_runs": 0,
    "server_pulls": 0,
}
REGISTRATION_SCOPE = {
    "failed_confirmation_outcome_read": True,
    "three_consumed_rosters_read": True,
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
        "the polished R16/K9 certificate or actual regret fails a support gate"
    ),
    ACTION_FAIL_DECISION: (
        "support passes but a polished R16/K9 action differs from its exact reference"
    ),
    COMPUTE_FAIL_DECISION: (
        "support and action identity pass but charged work is not strictly lower"
    ),
    PASS_DECISION: (
        "R16/K9 plus the frozen bulk-tie polish passes support, action identity, and "
        "charged-work gates on all three consumed rosters and in aggregate, while "
        "the retained R8 source evidence remains passing"
    ),
}
CLAIM_BOUNDARY = (
    "An outcome-informed development replay on three consumed 512-state rosters. "
    "A pass can authorize preregistration of one new disjoint static confirmation "
    "for fixed R8/K48-single-exchange and R16/K9-bulk-tie methods. It is not "
    "independent confirmation, trajectory authorization, candidate-side canonical "
    "exactness, all-queue policy equivalence, capacity, stability, or a physical result."
)
PROHIBITED = (
    "changing the trigger, structural pairs, one-pass bulk rule, tie-break, work "
    "accounting, budgets, loads, or consumed rosters after registration; reading "
    "K4096 inside the polish; using these rosters as confirmation; new rosters, "
    "arrivals, trajectories, run-time RNG, server pulls, and paper edits"
)


class R16BulkTieDevelopmentRefusal(RuntimeError):
    """The registered post-result development contract is not satisfied."""


def _json(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise R16BulkTieDevelopmentRefusal(
            f"cannot read JSON object {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise R16BulkTieDevelopmentRefusal(f"JSON root must be an object: {path}")
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
        BULK_SOURCE: (ROOT / BULK_SOURCE).read_text(encoding="utf-8"),
        THIS_SOURCE: (ROOT / THIS_SOURCE).read_text(encoding="utf-8"),
    }


def _roster_contract(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(Path(path).resolve()),
        "seed": payload["seed"],
        "size": ROSTER_SIZE,
        "queues": _json(payload["queues"]),
        "payload": _json(payload),
    }


def _validated_source_contract(
    result: Mapping[str, Any],
    protocol: Mapping[str, Any],
    source_roster: Path,
) -> dict[str, Any]:
    try:
        source._validate_protocol(protocol)
    except source.SingleExchangeConfirmationRefusal as exc:
        raise R16BulkTieDevelopmentRefusal(str(exc)) from exc
    paths = protocol.get("source_paths") or {}
    dev_protocol = _read_object(Path(paths["development_registration"]))
    dev_result = _read_object(Path(paths["development_result"]))
    if (
        _read_object(Path(paths["development_fixed_sources"]))
        != source.development._current_fixed_sources()
    ):
        raise R16BulkTieDevelopmentRefusal(
            "failed confirmation development source changed"
        )
    try:
        development_contract = source._validated_development_contract(
            dev_result, dev_protocol
        )
    except source.SingleExchangeConfirmationRefusal as exc:
        raise R16BulkTieDevelopmentRefusal(str(exc)) from exc
    if development_contract != protocol.get("development_contract"):
        raise R16BulkTieDevelopmentRefusal(
            "failed confirmation embedded development contract differs"
        )
    lane_rows = result.get("lane_summaries") or ()
    by_lane = {row.get("lanes"): row for row in lane_rows}
    r8 = by_lane.get(8)
    r16 = by_lane.get(16)
    aggregate = result.get("aggregate_actual_compute") or {}
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
        or result.get("independent_roster_seed") != source.ROSTER_SEED
        or result.get("independent_roster_state_count") != ROSTER_SIZE
        or not result.get("checks")
        or not all(result["checks"].values())
        or set(by_lane) != {8, 16}
        or r8 is None
        or r8.get("fixed_lp_budget") != 48
        or r8.get("lane_confirmation_gate_passed") is not True
        or r8.get("candidate_action_matches_reference_count") != ROSTER_SIZE
        or r16 is None
        or r16.get("fixed_lp_budget") != R16_BUDGET
        or r16.get("support_gate_passed") is not True
        or r16.get("actual_compute_gate_passed") is not True
        or r16.get("independent_roster_action_identity_gate_passed") is not False
        or r16.get("candidate_score_exact_state_count") != ROSTER_SIZE
        or r16.get("candidate_canonical_exact_state_count") != 0
        or r16.get("candidate_action_matches_reference_count") != 509
        or r16.get("candidate_action_mismatch_count") != 3
        or r16.get("score_tie_action_mismatch_count") != 3
        or r16.get("positive_regret_action_mismatch_count") != 0
        or aggregate.get("gate_passed") is not True
    ):
        raise R16BulkTieDevelopmentRefusal(
            "frozen failed bulk-tie source confirmation differs"
        )
    source_roster_payload = _read_object(Path(source_roster))
    try:
        exclusions = source._load_exclusion_rosters(protocol["source_paths"])
        validated_source_queues = source._validate_roster(
            source_roster_payload, protocol, exclusions
        )
    except (KeyError, source.SingleExchangeConfirmationRefusal) as exc:
        raise R16BulkTieDevelopmentRefusal(str(exc)) from exc
    if (
        source_roster_payload != protocol.get("queue_roster_contract")
        or [list(row) for row in validated_source_queues]
        != source_roster_payload.get("queues")
    ):
        raise R16BulkTieDevelopmentRefusal("seed270919 source roster changed")
    old_r8_rows = {
        row["roster"]: row
        for row in dev_result.get("r8_consumed_roster_summaries") or ()
    }
    if set(old_r8_rows) != {"seed13041", "seed260919"}:
        raise R16BulkTieDevelopmentRefusal("retained older R8 evidence changed")
    retained_r8 = {}
    for name, row in old_r8_rows.items():
        if (
            row.get("candidate_action_matches_reference_count") != ROSTER_SIZE
            or row.get("support_gate_passed") is not True
            or row.get("charged_compute_gate_passed") is not True
            or row.get("development_gate_passed") is not True
        ):
            raise R16BulkTieDevelopmentRefusal(
                f"retained R8 evidence is not passing: {name}"
            )
        retained_r8[name] = {
            "candidate_action_matches_reference_count": ROSTER_SIZE,
            "support_gate_passed": True,
            "charged_compute_gate_passed": True,
        }
    retained_r8["seed270919"] = {
        "candidate_action_matches_reference_count": ROSTER_SIZE,
        "support_gate_passed": True,
        "charged_compute_gate_passed": True,
    }
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
        "seed260919_roster_contract": _json(
            development_contract["prior_confirmation_roster_contract"]
        ),
        "seed270919_roster_contract": _roster_contract(
            Path(source_roster), source_roster_payload
        ),
        "retained_r8_evidence": retained_r8,
        "source_r16_failure": {
            "candidate_score_exact_state_count": ROSTER_SIZE,
            "candidate_canonical_exact_state_count": 0,
            "candidate_action_matches_reference_count": 509,
            "score_tie_action_mismatch_count": 3,
            "positive_regret_action_mismatch_count": 0,
        },
    }


def _load_consumed_rosters(
    contract: Mapping[str, Any],
) -> dict[str, tuple[tuple[int, int, int], ...]]:
    rosters = {}
    for name in ROSTER_NAMES:
        frozen = contract[f"{name}_roster_contract"]
        payload = _read_object(Path(frozen["path"]))
        queues = tuple(
            tuple(int(value) for value in row) for row in payload.get("queues") or ()
        )
        if (
            payload.get("seed") != frozen.get("seed")
            or [list(row) for row in queues] != frozen.get("queues")
            or len(queues) != ROSTER_SIZE
            or len(set(queues)) != ROSTER_SIZE
            or (frozen.get("payload") is not None and payload != frozen["payload"])
        ):
            raise R16BulkTieDevelopmentRefusal(f"consumed roster changed: {name}")
        rosters[name] = queues
    return rosters


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
        raise R16BulkTieDevelopmentRefusal("frozen source confirmation changed")
    contract = _validated_source_contract(result, protocol, Path(source_roster))
    rosters = _load_consumed_rosters(contract)
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
        "method_contract": _json(METHOD_CONTRACT),
        "consumed_rosters": {
            name: [list(row) for row in rows] for name, rows in rosters.items()
        },
        "loads": list(canonical.LOADS),
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
        raise R16BulkTieDevelopmentRefusal("development artifact already exists")
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
        raise R16BulkTieDevelopmentRefusal(
            "registered R16 bulk-tie protocol changed"
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
    base_work = canonical._new_work()
    reference_work = canonical._new_work()
    valid_candidates = valid_references = bounded_regrets = action_matches = 0
    trigger_count = pair_checks = attempts = feasible = accepted = changed = 0
    max_epsilon = max_regret = -1.0
    epsilon_witness = regret_witness = first_mismatch = None
    structural_pair_count = sum(
        1
        for source_index in range(len(model.options))
        for destination in range(source_index)
        if sum(model.options[source_index].requirements)
        == sum(model.options[destination].requirements)
    )

    for base_queue in base_queues:
        queue = tuple(8 * int(value) for value in base_queue)
        polished = select_canonical_score_tie_bulk_substitution_frame(
            model.options, queue, R16_LANES, R16_BUDGET
        )
        base = polished.base_selection
        candidate = polished.selection
        reference = select_canonical_count_space_bnb_frame(
            model.options, queue, R16_LANES, REFERENCE_BUDGET
        )
        charged = total_work_row(polished)
        expected = {field: int(getattr(base, field)) for field in canonical.WORK_FIELDS}
        expected["candidate_evaluations"] += polished.pair_checks
        expected["residual_fill_option_probes"] += polished.pair_checks
        relation = (
            polished.polish_triggered == (base.score_exact and not base.canonical_exact)
            and polished.pair_checks
            == structural_pair_count * int(polished.polish_triggered)
            and (
                polished.polish_triggered
                or (
                    polished.bulk_candidate_attempts == 0
                    and polished.feasible_bulk_candidate_count == 0
                    and polished.accepted_bulk_substitution_count == 0
                    and polished.incumbent_changed is False
                    and candidate == base
                )
            )
            and charged == expected
            and candidate.upper_bound == base.upper_bound
            and candidate.score_exact == base.score_exact
            and candidate.canonical_exact == base.canonical_exact
            and all(
                getattr(candidate, field) == getattr(base, field)
                for field in canonical.WORK_FIELDS
            )
            and candidate.score + canonical.SCORE_TOLERANCE >= base.score
            and polished.incumbent_changed
            == (candidate.option_indices != base.option_indices)
            and polished.accepted_bulk_substitution_count
            == int(polished.incumbent_changed)
        )
        candidate_valid = (
            relation
            and canonical._selection_reconstructs(model, queue, R16_LANES, base)
            and canonical._selection_reconstructs(model, queue, R16_LANES, candidate)
            and canonical._selection_work_holds(queue, R16_LANES, R16_BUDGET, base)
            and canonical._certificate_valid(base, queue)
            and canonical._certificate_valid(candidate, queue)
        )
        reference_valid = (
            canonical._selection_reconstructs(model, queue, R16_LANES, reference)
            and canonical._selection_work_holds(
                queue, R16_LANES, REFERENCE_BUDGET, reference
            )
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
        action_match = candidate.option_indices == reference.option_indices
        action_matches += int(action_match)
        if not action_match and first_mismatch is None:
            first_mismatch = canonical._selection_witness(
                base_queue, queue, candidate, reference, regret
            )
        trigger_count += int(polished.polish_triggered)
        pair_checks += polished.pair_checks
        attempts += polished.bulk_candidate_attempts
        feasible += polished.feasible_bulk_candidate_count
        accepted += polished.accepted_bulk_substitution_count
        changed += int(polished.incumbent_changed)
        canonical._add_work(base_work, base)
        _add_work(candidate_work, charged)
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

    comparisons = source.previous.development._work_comparison(
        candidate_work, reference_work
    )
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
    expected_work = dict(base_work)
    expected_work["candidate_evaluations"] += pair_checks
    expected_work["residual_fill_option_probes"] += pair_checks
    checks = {
        "state_count_is_512": len(base_queues) == ROSTER_SIZE,
        "every_candidate_method_relation_and_certificate_is_valid": (
            valid_candidates == ROSTER_SIZE
        ),
        "every_reference_is_valid_score_exact_and_canonical_exact": (
            valid_references == ROSTER_SIZE
        ),
        "every_regret_is_nonnegative_and_bounded_by_epsilon": (
            bounded_regrets == ROSTER_SIZE
        ),
        "all_seven_structural_pair_checks_are_charged_per_triggered_state": (
            structural_pair_count == 7
            and pair_checks == structural_pair_count * trigger_count
            and candidate_work == expected_work
        ),
    }
    support_passed = all(row["support_gate_passed"] for row in support_rows)
    action_passed = action_matches == ROSTER_SIZE
    compute_passed = all(row["strictly_lower"] for row in comparisons.values())
    return ({
        "roster": roster_name,
        "seed": int(roster_name.removeprefix("seed")),
        "state_count": len(base_queues),
        "lanes": R16_LANES,
        "base_lp_budget": R16_BUDGET,
        "cold_reference_budget": REFERENCE_BUDGET,
        "polish_trigger_state_count": trigger_count,
        "structural_pair_check_count": pair_checks,
        "active_bulk_candidate_attempt_count": attempts,
        "feasible_bulk_candidate_count": feasible,
        "accepted_bulk_substitution_count": accepted,
        "changed_incumbent_state_count": changed,
        "candidate_action_matches_reference_count": action_matches,
        "candidate_action_mismatch_count": ROSTER_SIZE - action_matches,
        "first_action_mismatch_witness": first_mismatch,
        "maximum_candidate_epsilon": max_epsilon,
        "maximum_actual_normalized_support_regret": max_regret,
        "epsilon_witness": epsilon_witness,
        "support_regret_witness": regret_witness,
        "support_by_load": support_rows,
        "candidate_base_work": base_work,
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
    retained_r8_passed: bool,
) -> str:
    if not execution_passed or not retained_r8_passed:
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
        raise R16BulkTieDevelopmentRefusal(f"development result exists: {output}")
    protocol = _read_object(Path(registration))
    _validate_protocol(protocol)
    if _read_object(Path(fixed_sources)) != _current_fixed_sources():
        raise R16BulkTieDevelopmentRefusal("registered development sources changed")
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
        raise R16BulkTieDevelopmentRefusal(
            "failed confirmation changed after registration"
        )
    rosters = _load_consumed_rosters(protocol["source_contract"])
    if {
        name: [list(row) for row in rows] for name, rows in rosters.items()
    } != protocol["consumed_rosters"]:
        raise R16BulkTieDevelopmentRefusal("consumed roster snapshot changed")
    contract = protocol["source_contract"]
    model = source.previous.development.source.capacity.base._model_from_snapshot(
        contract["physical_complete_frame_envelope"]["model"]
    )
    capacity_rows = contract["physical_envelope_capacity_normalized_loads"]["16"]

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
    aggregate_comparisons = source.previous.development._work_comparison(
        candidate_total, reference_total
    )
    aggregate_compute_passed = all(
        row["strictly_lower"] for row in aggregate_comparisons.values()
    )
    retained_r8 = contract["retained_r8_evidence"]
    retained_r8_passed = all(
        row["candidate_action_matches_reference_count"] == ROSTER_SIZE
        and row["support_gate_passed"]
        and row["charged_compute_gate_passed"]
        for row in retained_r8.values()
    )
    checks = {
        "failed_confirmation_is_valid_and_unchanged": True,
        "three_consumed_rosters_are_exact_and_unchanged": all(
            len(rows_) == ROSTER_SIZE for rows_ in rosters.values()
        ),
        "every_r16_execution_check_passes": all(
            all(check.values()) for check in row_checks
        ),
        "retained_r8_source_evidence_passes": retained_r8_passed,
        "r16_candidate_and_reference_call_counts_are_exact": sum(
            row["state_count"] for row in rows
        )
        == 3 * ROSTER_SIZE,
        "no_new_roster_arrivals_trajectories_rng_or_server_pulls": True,
    }
    execution_passed = all(checks.values())
    decision = _decision(
        execution_passed, rows, aggregate_compute_passed, retained_r8_passed
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
        "r16_consumed_roster_summaries": rows,
        "r16_aggregate_charged_compute": {
            "candidate": candidate_total,
            "cold_reference": reference_total,
            "comparison": aggregate_comparisons,
            "gate_passed": aggregate_compute_passed,
        },
        "retained_r8_evidence": retained_r8,
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
    except R16BulkTieDevelopmentRefusal as exc:
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
