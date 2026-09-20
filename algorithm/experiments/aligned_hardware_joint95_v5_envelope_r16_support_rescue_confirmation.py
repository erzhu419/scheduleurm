"""Independent static confirmation of the fixed conditional R16 rescue method.

Registration alone draws the seed-290919 roster. Execution reads that frozen
payload, applies R8/K48 single exchange and R16/K9 with conditional fresh K144
rescue, and independently evaluates cold K4096 references. A scientific pass
authorizes only registration of a later envelope-normalized trajectory study.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

from algorithm.experiments import (
    aligned_hardware_joint95_v5_envelope_r16_support_rescue_development as development,
)


source = development.source
canonical = development.canonical
previous = source.previous
ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
STEM = "aligned_hardware_joint95_v5_envelope_r16_support_rescue_confirmation_20260920"
DEFAULT_REGISTRATION = ARTIFACTS / f"{STEM}_protocol.json"
DEFAULT_ROSTER = ARTIFACTS / f"{STEM}_roster.json"
DEFAULT_FIXED_SOURCES = ARTIFACTS / f"{STEM}_fixed_sources.json"
DEFAULT_RESULT = ARTIFACTS / f"{STEM}.json"
DEFAULT_DEVELOPMENT_REGISTRATION = development.DEFAULT_REGISTRATION
DEFAULT_DEVELOPMENT_FIXED_SOURCES = development.DEFAULT_FIXED_SOURCES
DEFAULT_DEVELOPMENT_RESULT = development.DEFAULT_RESULT
DEFAULT_BULK_CONFIRMATION_REGISTRATION = source.DEFAULT_REGISTRATION
DEFAULT_BULK_CONFIRMATION_ROSTER = source.DEFAULT_ROSTER
DEFAULT_BULK_CONFIRMATION_FIXED_SOURCES = source.DEFAULT_FIXED_SOURCES
EXPERIMENT = "aligned_hardware_joint95_v5_envelope_r16_support_rescue_confirmation_v1"
GATE = "aligned_hardware_joint95_v5_envelope_r16_support_rescue_confirmation_runner"
REGISTRATION_STATUS = "REGISTERED_AWAITING_INDEPENDENT_STATIC_CONFIRMATION"
SOURCE_DECISION = development.PASS_DECISION
PASS_DECISION = "INDEPENDENT_ENVELOPE_R16_SUPPORT_RESCUE_STATIC_CONFIRMATION_PASS"
INVALID_DECISION = source.INVALID_DECISION
SUPPORT_FAIL_DECISION = source.SUPPORT_FAIL_DECISION
ACTION_FAIL_DECISION = source.ACTION_FAIL_DECISION
COMPUTE_FAIL_DECISION = source.COMPUTE_FAIL_DECISION
ROSTER_SEED = 290919
ROSTER_SIZE = 512
ROSTER_CANDIDATE_DRAWS = 558
ROSTER_RNG_INTEGER_DRAWS = 1674
ROSTER_RANGES = source.ROSTER_RANGES
R8_LANES, R16_LANES = 8, 16
FIXED_BUDGET_BY_LANE = {8: 48, 16: 9}
PRIMARY_BUDGET = development.PRIMARY_BUDGET
RESCUE_BUDGET = development.RESCUE_BUDGET
REFERENCE_BUDGET = development.REFERENCE_BUDGET
SUPPORT_EPSILON_LIMIT = development.SUPPORT_EPSILON_LIMIT
LOADS = canonical.LOADS
WORK_GATE_FIELDS = source.WORK_GATE_FIELDS
EXCLUSION_NAMES = (
    "count_space_confirmation", "joint95_static", "canonical_development",
    "budget_pair_confirmation", "single_exchange_confirmation", "bulk_tie_confirmation",
)
THIS_SOURCE = (
    "algorithm/experiments/"
    "aligned_hardware_joint95_v5_envelope_r16_support_rescue_confirmation.py"
)
ROSTER_GENERATION_RULE = (
    "draw integer coordinates from random.Random(290919); reject zero, the old "
    "1,000-state grid, all six prior 512-state rosters and within-roster duplicates; "
    "preserve accepted order; execution reads the complete registered payload"
)
SELECTOR_CONTRACT = {
    "fixed_budget_by_lane": {"8": 48, "16": 9},
    "r8_method_contract": source.SELECTOR_CONTRACT["r8_method_contract"],
    "r16_method_contract": development.METHOD_CONTRACT,
    "r16_rescue_lp_budget": RESCUE_BUDGET,
    "r16_support_epsilon_limit": SUPPORT_EPSILON_LIMIT,
    "score_tolerance": development.RESCUE_SCORE_TOLERANCE,
    "r16_trigger": development.METHOD_CONTRACT["rescue_trigger"],
    "r16_rescue_gate": "every triggered rescue is score-exact and canonical-exact",
    "cold_canonical_reference_budget": REFERENCE_BUDGET,
    "reference": "separate fresh cold K4096 per state and lane; never read by either candidate",
    "support_gate": "epsilon and actual normalized regret strictly below eta at all three frozen loads",
    "action_identity_gate": "candidate option indices equal the exact canonical reference on every state",
    "actual_work_gate": (
        "LP calls, matrix coefficients and candidate evaluations each strictly lower "
        "than the reference per lane and in aggregate; cumulative primary plus every "
        "executed rescue, including all structural checks and R8 exchange attempts"
    ),
}
EXPECTED_WORK = {
    "registered_base_queues": ROSTER_SIZE,
    "scaled_queue_states": 2 * ROSTER_SIZE,
    "r8_composite_candidate_calls": ROSTER_SIZE,
    "r16_primary_composite_calls": ROSTER_SIZE,
    "r16_rescue_composite_calls": "data-dependent and fully charged",
    "cold_reference_selector_calls": 2 * ROSTER_SIZE,
    "ordered_action_evaluations": 0,
    "run_rng_initializations": 0,
    "arrival_generations": 0,
    "trajectory_runs": 0,
    "server_pulls": 0,
}
REGISTRATION_SCOPE = {
    "post_result_development_outcome_read": True,
    "roster_rng_initializations": 1,
    "roster_rng_integer_draws": ROSTER_RNG_INTEGER_DRAWS,
    "selector_invoked": False,
    "reference_invoked": False,
    "trajectory_built": False,
    "arrival_generated": False,
    "server_pulls": 0,
    "paper_edited": False,
}
DECISION_RULE = {
    INVALID_DECISION: "a registered execution-integrity or triggered-rescue exactness check fails",
    SUPPORT_FAIL_DECISION: "valid execution fails support on at least one lane and load",
    ACTION_FAIL_DECISION: "support passes but an action differs from its exact reference",
    COMPUTE_FAIL_DECISION: "support and action identity pass but a per-lane or aggregate work gate is not strict",
    PASS_DECISION: "execution, all-load support, action identity and all strict charged-work gates pass",
}
CLAIM_BOUNDARY = (
    "Independent finite-roster static confirmation of fixed R8/K48 single exchange "
    "and conditional R16/K9-to-K144 support rescue on the immutable joint95-v5 "
    "physical envelope. A pass authorizes only registration of a separate "
    "envelope-normalized trajectory study. It establishes neither trajectory "
    "performance, capacity reclassification, all-queue action identity, stability, "
    "nor a physical multi-GPU result."
)
PROHIBITED = (
    "changing source evidence, method, budgets, support limit, loads, seed, roster, "
    "exclusions, reference, accounting or gates after registration; warm-start or "
    "reference sharing; execution RNG; arrivals, trajectories, server pulls, "
    "capacity reclassification and paper edits"
)
SOURCE_DEVELOPMENT_OUTCOMES = {
    "seed13041": {
        "candidate_score_exact_state_count": 511,
        "candidate_canonical_exact_state_count": 0,
        "rescue_trigger_state_count": 0,
        "exact_rescue_state_count": 0,
        "candidate_action_matches_reference_count": 512,
        "candidate_action_mismatch_count": 0,
        "maximum_candidate_epsilon": 0.00519007334806459,
        "maximum_actual_normalized_support_regret": 0.0,
        "primary_work": (4608, 238080, 8697),
        "rescue_work": (0, 0, 0),
        "candidate_work": (4608, 238080, 8697),
        "reference_work": (8981, 722631, 9493),
    },
    "seed260919": {
        "candidate_score_exact_state_count": 512,
        "candidate_canonical_exact_state_count": 0,
        "rescue_trigger_state_count": 0,
        "exact_rescue_state_count": 0,
        "candidate_action_matches_reference_count": 512,
        "candidate_action_mismatch_count": 0,
        "maximum_candidate_epsilon": 0.0,
        "maximum_actual_normalized_support_regret": 0.0,
        "primary_work": (4608, 238080, 8704),
        "rescue_work": (0, 0, 0),
        "candidate_work": (4608, 238080, 8704),
        "reference_work": (9029, 728529, 9541),
    },
    "seed270919": {
        "candidate_score_exact_state_count": 512,
        "candidate_canonical_exact_state_count": 0,
        "rescue_trigger_state_count": 0,
        "exact_rescue_state_count": 0,
        "candidate_action_matches_reference_count": 512,
        "candidate_action_mismatch_count": 0,
        "maximum_candidate_epsilon": 0.0,
        "maximum_actual_normalized_support_regret": 0.0,
        "primary_work": (4608, 238080, 8704),
        "rescue_work": (0, 0, 0),
        "candidate_work": (4608, 238080, 8704),
        "reference_work": (9277, 759090, 9789),
    },
    "seed280919": {
        "candidate_score_exact_state_count": 512,
        "candidate_canonical_exact_state_count": 1,
        "rescue_trigger_state_count": 1,
        "exact_rescue_state_count": 1,
        "candidate_action_matches_reference_count": 512,
        "candidate_action_mismatch_count": 0,
        "maximum_candidate_epsilon": 0.0,
        "maximum_actual_normalized_support_regret": 0.0,
        "primary_work": (4608, 238080, 8697),
        "rescue_work": (130, 14622, 131),
        "candidate_work": (4738, 252702, 8828),
        "reference_work": (8990, 722697, 9502),
    },
}
SOURCE_AGGREGATE_GATE_WORK = {
    "candidate": (18562, 966942, 34933),
    "cold_reference": (36277, 2932947, 38325),
}


class R16SupportRescueConfirmationRefusal(RuntimeError):
    """The fixed independent static confirmation contract is not satisfied."""


def _json(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _read_object(path: Path) -> dict[str, Any]:
    try:
        return source._read_object(Path(path))
    except source.R16BulkTieConfirmationRefusal as exc:
        raise R16SupportRescueConfirmationRefusal(str(exc)) from exc


def _write_new(path: Path, payload: Mapping[str, Any], *, compact: bool = False) -> None:
    source._write_new(path, payload, compact=compact)


def _current_fixed_sources() -> dict[str, str]:
    return {**development._current_fixed_sources(), THIS_SOURCE: (ROOT / THIS_SOURCE).read_text(encoding="utf-8")}


def _gate_work(work: Mapping[str, Any]) -> tuple[int, int, int]:
    return tuple(int(work[field]) for field in WORK_GATE_FIELDS)


def _validated_development_contract(result: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    try:
        development._validate_protocol(protocol)
        paths = protocol["source_paths"]
        if _read_object(Path(paths["failed_confirmation_fixed_sources"])) != source._current_fixed_sources():
            raise R16SupportRescueConfirmationRefusal("frozen source confirmation changed")
        authoritative = development._validated_source_contract(
            _read_object(Path(paths["failed_confirmation_result"])),
            _read_object(Path(paths["failed_confirmation_registration"])),
            Path(paths["failed_confirmation_roster"]),
        )
        consumed = development._load_consumed_rosters(authoritative)
    except development.R16SupportRescueDevelopmentRefusal as exc:
        raise R16SupportRescueConfirmationRefusal(str(exc)) from exc
    rows = result.get("r16_consumed_roster_summaries") or ()
    expected_false = ("trajectory_registration_prerequisite_passed", "capacity_closure_reclassified",
                      "retained_v5_capacity_established", "retained_v5_closure_passed")
    if (
        result.get("schema_version") != 1 or result.get("gate") != development.GATE
        or result.get("experiment") != development.EXPERIMENT or result.get("status") != "PASS"
        or result.get("scientific_decision") != SOURCE_DECISION
        or result.get("source_scientific_decision") != development.SOURCE_DECISION
        or result.get("protocol") != protocol
        or result.get("new_independent_static_confirmation_registration_authorized") is not True
        or any(result.get(key) is not False for key in expected_false)
        or any(result.get(key) != 0 for key in ("trajectory_runs", "arrival_generations", "rng_draws", "server_pulls"))
        or not result.get("checks") or not all(result["checks"].values())
        or authoritative != protocol.get("source_contract")
        or {name: [list(q) for q in roster] for name, roster in consumed.items()} != protocol.get("consumed_rosters")
        or len(rows) != 4 or {row.get("roster") for row in rows} != set(development.ROSTER_NAMES)
        or result.get("retained_r8_evidence") != development.SOURCE_R8_EVIDENCE
    ):
        raise R16SupportRescueConfirmationRefusal("frozen support-rescue development result differs")
    totals = {name: development._new_work() for name in ("primary", "rescue", "candidate", "cold_reference")}
    for row in rows:
        roster_name = str(row.get("roster"))
        expected_outcome = SOURCE_DEVELOPMENT_OUTCOMES[roster_name]
        if (
            row.get("state_count") != ROSTER_SIZE or row.get("lanes") != R16_LANES
            or row.get("primary_lp_budget") != PRIMARY_BUDGET or row.get("rescue_lp_budget") != RESCUE_BUDGET
            or row.get("cold_reference_budget") != REFERENCE_BUDGET
            or row.get("support_epsilon_limit") != SUPPORT_EPSILON_LIMIT
            or row.get("candidate_action_matches_reference_count") != ROSTER_SIZE
            or row.get("candidate_action_mismatch_count") != 0
            or row.get("rescue_trigger_state_count") != row.get("exact_rescue_state_count")
            or not row.get("checks") or not all(row["checks"].values())
            or any(row.get(key) is not True for key in ("support_gate_passed", "action_identity_gate_passed",
                                                       "charged_compute_gate_passed", "development_gate_passed"))
        ):
            raise R16SupportRescueConfirmationRefusal("development roster gates changed")
        work = {"primary": row["primary_charged_work"], "rescue": row["rescue_charged_work"],
                "candidate": row["candidate_total_charged_work"], "cold_reference": row["cold_reference_actual_work"]}
        observed_outcome = {
            key: row.get(key) for key in (
                "candidate_score_exact_state_count",
                "candidate_canonical_exact_state_count",
                "rescue_trigger_state_count",
                "exact_rescue_state_count",
                "candidate_action_matches_reference_count",
                "candidate_action_mismatch_count",
                "maximum_candidate_epsilon",
                "maximum_actual_normalized_support_regret",
            )
        }
        observed_outcome.update({
            "primary_work": _gate_work(work["primary"]),
            "rescue_work": _gate_work(work["rescue"]),
            "candidate_work": _gate_work(work["candidate"]),
            "reference_work": _gate_work(work["cold_reference"]),
        })
        if observed_outcome != expected_outcome:
            raise R16SupportRescueConfirmationRefusal(
                "development roster outcome changed"
            )
        expected_support = []
        for capacity in authoritative[
            "physical_envelope_capacity_normalized_loads"
        ]["16"]:
            eta = float(capacity["effective_slack_eta"])
            epsilon_residual = eta - expected_outcome[
                "maximum_candidate_epsilon"
            ]
            regret_residual = eta - expected_outcome[
                "maximum_actual_normalized_support_regret"
            ]
            expected_support.append({
                "load": float(capacity["load"]),
                "effective_slack_eta": eta,
                "maximum_candidate_epsilon": expected_outcome[
                    "maximum_candidate_epsilon"
                ],
                "maximum_actual_normalized_support_regret": expected_outcome[
                    "maximum_actual_normalized_support_regret"
                ],
                "eta_minus_candidate_epsilon": epsilon_residual,
                "eta_minus_actual_support_regret": regret_residual,
                "support_gate_passed": (
                    bool(capacity["capacity_projection_passed"])
                    and epsilon_residual > canonical.SCORE_TOLERANCE
                    and regret_residual > canonical.SCORE_TOLERANCE
                ),
            })
        if row.get("support_by_load") != expected_support:
            raise R16SupportRescueConfirmationRefusal(
                "development support rows changed"
            )
        if any(work["candidate"][field] != work["primary"][field] + work["rescue"][field]
               for field in canonical.WORK_FIELDS):
            raise R16SupportRescueConfirmationRefusal("development rescue work is not cumulative")
        comparison = previous.development._work_comparison(work["candidate"], work["cold_reference"])
        if comparison != row["actual_compute_comparison"] or not all(x["strictly_lower"] for x in comparison.values()):
            raise R16SupportRescueConfirmationRefusal("development charged-work comparison changed")
        for name, counts in work.items():
            development._add_work(totals[name], counts)
    aggregate = result.get("r16_aggregate_charged_compute") or {}
    expected_comparison = previous.development._work_comparison(totals["candidate"], totals["cold_reference"])
    if (any(aggregate.get(name) != work for name, work in totals.items())
            or aggregate.get("comparison") != expected_comparison or aggregate.get("gate_passed") is not True
            or not all(row["strictly_lower"] for row in expected_comparison.values())
            or _gate_work(aggregate.get("candidate") or {})
            != SOURCE_AGGREGATE_GATE_WORK["candidate"]
            or _gate_work(aggregate.get("cold_reference") or {})
            != SOURCE_AGGREGATE_GATE_WORK["cold_reference"]):
        raise R16SupportRescueConfirmationRefusal("development aggregate charged work changed")
    return {
        "experiment": development.EXPERIMENT, "status": "PASS", "scientific_decision": SOURCE_DECISION,
        "method_contract": _json(development.METHOD_CONTRACT),
        "physical_complete_frame_envelope": authoritative["physical_complete_frame_envelope"],
        "physical_envelope_capacity_normalized_loads": authoritative["physical_envelope_capacity_normalized_loads"],
        "consumed_roster_contracts": {name: authoritative[f"{name}_roster_contract"] for name in development.ROSTER_NAMES},
        "development_result_payload": _json(result),
    }


def _old_grid() -> set[tuple[int, int, int]]:
    return source._old_grid()


def _load_exclusion_rosters(paths: Mapping[str, Any]) -> dict[str, tuple[tuple[int, int, int], ...]]:
    try:
        exclusions = source._load_exclusion_rosters(paths)
        bulk_protocol = _read_object(Path(paths["bulk_confirmation_registration"]))
        source._validate_protocol(bulk_protocol)
        if _read_object(Path(paths["bulk_confirmation_fixed_sources"])) != source._current_fixed_sources():
            raise R16SupportRescueConfirmationRefusal("bulk-tie confirmation frozen sources changed")
        bulk_payload = _read_object(Path(paths["bulk_confirmation_roster"]))
        exclusions["bulk_tie_confirmation"] = source._validate_roster(bulk_payload, bulk_protocol, exclusions)
    except source.R16BulkTieConfirmationRefusal as exc:
        raise R16SupportRescueConfirmationRefusal(str(exc)) from exc
    if set(exclusions) != set(EXCLUSION_NAMES) or len(set().union(*(set(rows) for rows in exclusions.values()))) != 3072:
        raise R16SupportRescueConfirmationRefusal("the six prior 512-state rosters are not disjoint")
    return exclusions


def _exclusion_contract(exclusions: Mapping[str, Sequence[Sequence[int]]]) -> dict[str, Any]:
    contract = source._exclusion_contract({key: rows for key, rows in exclusions.items() if key != "bulk_tie_confirmation"})
    contract["bulk_tie_confirmation"] = {"experiment": source.EXPERIMENT, "seed": source.ROSTER_SEED,
                                         "size": ROSTER_SIZE, "queues": _json(exclusions["bulk_tie_confirmation"])}
    return contract


def generate_roster(exclusions: Mapping[str, Sequence[Sequence[int]]]) -> dict[str, Any]:
    sets = {name: {tuple(q) for q in rows} for name, rows in exclusions.items()}
    union = set().union(*sets.values())
    if (set(sets) != set(EXCLUSION_NAMES) or any(len(rows) != ROSTER_SIZE for rows in sets.values())
            or len(union) != 3072):
        raise R16SupportRescueConfirmationRefusal("six complete disjoint prior rosters are required")
    old, seen, queues = _old_grid(), set(), []
    rng = random.Random(ROSTER_SEED)
    draws = 0
    while len(queues) < ROSTER_SIZE:
        draws += 1
        queue = tuple(rng.randint(lower, upper) for lower, upper in ROSTER_RANGES)
        if queue == (0, 0, 0) or queue in old or queue in union or queue in seen:
            continue
        seen.add(queue)
        queues.append(list(queue))
    return {
        "schema_version": 1, "experiment": EXPERIMENT, "seed": ROSTER_SEED,
        "coordinate_ranges_inclusive": _json(ROSTER_RANGES), "requested_queue_count": ROSTER_SIZE,
        "candidate_draws": draws, "rng_integer_draws": draws * len(ROSTER_RANGES),
        "generation_rule": ROSTER_GENERATION_RULE, "zero_queue_excluded": True,
        "old_1000_grid_excluded": True, "all_six_prior_512_rosters_excluded": True,
        "old_1000_grid_intersection_count": 0, "prior_roster_union_size": len(union),
        "prior_roster_intersection_count": {name: len(seen & values) for name, values in sets.items()},
        "queues": queues,
    }


def build_registration(
    *, development_registration: Path = DEFAULT_DEVELOPMENT_REGISTRATION,
    development_fixed_sources: Path = DEFAULT_DEVELOPMENT_FIXED_SOURCES,
    development_result: Path = DEFAULT_DEVELOPMENT_RESULT,
    bulk_confirmation_registration: Path = DEFAULT_BULK_CONFIRMATION_REGISTRATION,
    bulk_confirmation_roster: Path = DEFAULT_BULK_CONFIRMATION_ROSTER,
    bulk_confirmation_fixed_sources: Path = DEFAULT_BULK_CONFIRMATION_FIXED_SOURCES,
) -> dict[str, Any]:
    dev_protocol, dev_result = _read_object(development_registration), _read_object(development_result)
    if _read_object(development_fixed_sources) != development._current_fixed_sources():
        raise R16SupportRescueConfirmationRefusal("frozen development sources changed")
    contract = _validated_development_contract(dev_result, dev_protocol)
    bulk_protocol = _read_object(bulk_confirmation_registration)
    paths = {**bulk_protocol["source_paths"],
             "development_registration": str(Path(development_registration).resolve()),
             "development_fixed_sources": str(Path(development_fixed_sources).resolve()),
             "development_result": str(Path(development_result).resolve()),
             "bulk_confirmation_registration": str(Path(bulk_confirmation_registration).resolve()),
             "bulk_confirmation_roster": str(Path(bulk_confirmation_roster).resolve()),
             "bulk_confirmation_fixed_sources": str(Path(bulk_confirmation_fixed_sources).resolve())}
    exclusions = _load_exclusion_rosters(paths)
    mapped = {"seed13041": "canonical_development", "seed260919": "budget_pair_confirmation",
              "seed270919": "single_exchange_confirmation", "seed280919": "bulk_tie_confirmation"}
    for name, key in mapped.items():
        frozen = contract["consumed_roster_contracts"][name]
        if _json(exclusions[key]) != frozen["queues"]:
            raise R16SupportRescueConfirmationRefusal("prior roster differs from development evidence")
    roster = generate_roster(exclusions)
    if (
        roster["candidate_draws"] != ROSTER_CANDIDATE_DRAWS
        or roster["rng_integer_draws"] != ROSTER_RNG_INTEGER_DRAWS
    ):
        raise R16SupportRescueConfirmationRefusal(
            "seed290919 roster generation count changed"
        )
    model = contract["physical_complete_frame_envelope"]["model"]
    if (
        len(model["options"]) != canonical.OPTION_COUNT
        or RESCUE_BUDGET != R16_LANES * len(model["options"])
        or development.RESCUE_SCORE_TOLERANCE != canonical.SCORE_TOLERANCE
    ):
        raise R16SupportRescueConfirmationRefusal(
            "registered option count, rescue budget or score tolerance changed"
        )
    return {
        "schema_version": 1, "gate": GATE, "experiment": EXPERIMENT, "status": REGISTRATION_STATUS,
        "registered": True, "independent_confirmation": True, "source_paths": paths,
        "development_contract": contract, "exclusion_roster_contract": _exclusion_contract(exclusions),
        "queue_roster_contract": roster, "lane_counts": [8, 16], "queue_scale_by_lane": {"8": 4, "16": 8},
        "physical_envelope_capacity_normalized_loads": list(LOADS), "selector_contract": _json(SELECTOR_CONTRACT),
        "expected_work": _json(EXPECTED_WORK), "decision_rule": dict(DECISION_RULE),
        "claim_boundary": CLAIM_BOUNDARY, "prohibited": PROHIBITED,
        "registration_scope": dict(REGISTRATION_SCOPE),
    }


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    roster = protocol.get("queue_roster_contract") or {}
    expected = {
        "schema_version": 1, "gate": GATE, "experiment": EXPERIMENT, "status": REGISTRATION_STATUS,
        "registered": True, "independent_confirmation": True, "lane_counts": [8, 16],
        "queue_scale_by_lane": {"8": 4, "16": 8}, "physical_envelope_capacity_normalized_loads": list(LOADS),
        "selector_contract": SELECTOR_CONTRACT, "expected_work": EXPECTED_WORK, "decision_rule": DECISION_RULE,
        "claim_boundary": CLAIM_BOUNDARY, "prohibited": PROHIBITED,
        "registration_scope": REGISTRATION_SCOPE,
    }
    if any(protocol.get(key) != value for key, value in expected.items()) or set(protocol.get("exclusion_roster_contract") or {}) != set(EXCLUSION_NAMES):
        raise R16SupportRescueConfirmationRefusal("registered support-rescue confirmation protocol changed")


def _validate_roster(roster: Mapping[str, Any], protocol: Mapping[str, Any],
                     exclusions: Mapping[str, Sequence[Sequence[int]]]) -> tuple[tuple[int, int, int], ...]:
    queues = tuple(tuple(q) for q in roster.get("queues") or ())
    sets = {name: {tuple(q) for q in rows} for name, rows in exclusions.items()}
    union = set().union(*sets.values())
    old = _old_grid()
    expected = {
        "schema_version": 1, "experiment": EXPERIMENT, "seed": ROSTER_SEED,
        "coordinate_ranges_inclusive": _json(ROSTER_RANGES), "requested_queue_count": ROSTER_SIZE,
        "candidate_draws": ROSTER_CANDIDATE_DRAWS,
        "rng_integer_draws": ROSTER_RNG_INTEGER_DRAWS,
        "generation_rule": ROSTER_GENERATION_RULE, "zero_queue_excluded": True,
        "old_1000_grid_excluded": True, "all_six_prior_512_rosters_excluded": True,
        "old_1000_grid_intersection_count": 0, "prior_roster_union_size": 3072,
        "prior_roster_intersection_count": {name: 0 for name in EXCLUSION_NAMES},
    }
    if (
        roster != protocol.get("queue_roster_contract")
        or any(roster.get(key) != value for key, value in expected.items())
        or set(sets) != set(EXCLUSION_NAMES) or len(union) != 3072
        or any(len(rows) != ROSTER_SIZE for rows in sets.values())
        or len(queues) != ROSTER_SIZE or len(set(queues)) != ROSTER_SIZE
        or any(len(q) != 3 or q == (0, 0, 0) or q in old or q in union
               or any(not isinstance(value, int) or not lower <= value <= upper
                      for value, (lower, upper) in zip(q, ROSTER_RANGES)) for q in queues)
    ):
        raise R16SupportRescueConfirmationRefusal("registered independent roster changed or is invalid")
    return queues


def register(registration: Path = DEFAULT_REGISTRATION, roster_path: Path = DEFAULT_ROSTER,
             fixed_sources: Path = DEFAULT_FIXED_SOURCES, **kwargs: Any) -> dict[str, Any]:
    if any(Path(path).exists() for path in (registration, roster_path, fixed_sources, DEFAULT_RESULT)):
        raise R16SupportRescueConfirmationRefusal("confirmation artifact already exists")
    protocol = {"registered_at_utc": datetime.now(timezone.utc).isoformat(), **build_registration(**kwargs)}
    _write_new(Path(roster_path), protocol["queue_roster_contract"], compact=True)
    _write_new(Path(fixed_sources), _current_fixed_sources())
    _write_new(Path(registration), protocol)
    return protocol


def _screen_lane(model: Any, base_queues: Sequence[Sequence[int]], lanes: int,
                 capacity_rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, bool]]:
    if lanes == R8_LANES:
        return source._screen_lane(model, base_queues, lanes, capacity_rows)
    if lanes != R16_LANES:
        raise R16SupportRescueConfirmationRefusal("only the registered R8/R16 pair is supported")
    row, checks = development._screen_roster(model, base_queues, capacity_rows, f"seed{ROSTER_SEED}")
    row = dict(row)
    row["candidate_charged_work"] = row["candidate_total_charged_work"]
    row["cold_canonical_reference_actual_work"] = row["cold_reference_actual_work"]
    row["independent_roster_action_identity_gate_passed"] = row["action_identity_gate_passed"]
    row["actual_compute_gate_passed"] = row["charged_compute_gate_passed"]
    row["lane_confirmation_gate_passed"] = row.pop("development_gate_passed")
    return row, checks


def _decision(execution_passed: bool, rows: Sequence[Mapping[str, Any]], aggregate_compute_passed: bool) -> str:
    if not execution_passed:
        return INVALID_DECISION
    if not all(row["support_gate_passed"] for row in rows):
        return SUPPORT_FAIL_DECISION
    if not all(row["independent_roster_action_identity_gate_passed"] for row in rows):
        return ACTION_FAIL_DECISION
    if not aggregate_compute_passed or not all(row["actual_compute_gate_passed"] for row in rows):
        return COMPUTE_FAIL_DECISION
    return PASS_DECISION


def run_registered(registration: Path = DEFAULT_REGISTRATION, roster_path: Path = DEFAULT_ROSTER,
                   fixed_sources: Path = DEFAULT_FIXED_SOURCES, output: Path = DEFAULT_RESULT) -> dict[str, Any]:
    if Path(output).exists():
        raise R16SupportRescueConfirmationRefusal(f"confirmation result already exists: {output}")
    protocol = _read_object(registration)
    _validate_protocol(protocol)
    if _read_object(fixed_sources) != _current_fixed_sources():
        raise R16SupportRescueConfirmationRefusal("registered confirmation sources changed")
    paths = protocol["source_paths"]
    if (_read_object(Path(paths["development_fixed_sources"])) != development._current_fixed_sources()
            or _validated_development_contract(_read_object(Path(paths["development_result"])),
                                                _read_object(Path(paths["development_registration"])))
            != protocol["development_contract"]):
        raise R16SupportRescueConfirmationRefusal("development evidence changed after registration")
    exclusions = _load_exclusion_rosters(paths)
    if _exclusion_contract(exclusions) != protocol["exclusion_roster_contract"]:
        raise R16SupportRescueConfirmationRefusal("historical exclusion rosters changed after registration")
    queues = _validate_roster(_read_object(roster_path), protocol, exclusions)
    contract = protocol["development_contract"]
    model = previous.development.source.capacity.base._model_from_snapshot(contract["physical_complete_frame_envelope"]["model"])
    if (len(model.options) != canonical.OPTION_COUNT
            or len({max(1.0, option.duration) for option in model.options}) != canonical.OPTION_COUNT
            or R16_LANES * len(model.options) != RESCUE_BUDGET):
        raise R16SupportRescueConfirmationRefusal("frozen nine-option model or rescue budget changed")
    capacity = contract["physical_envelope_capacity_normalized_loads"]
    started = time.perf_counter()
    rows, execution_checks = [], []
    for lanes in (R8_LANES, R16_LANES):
        row, checks = _screen_lane(model, queues, lanes, capacity[str(lanes)])
        rows.append(row)
        execution_checks.append(checks)
    candidate_total, reference_total = development._new_work(), development._new_work()
    for row in rows:
        development._add_work(candidate_total, row["candidate_charged_work"])
        development._add_work(reference_total, row["cold_canonical_reference_actual_work"])
    comparison = previous.development._work_comparison(candidate_total, reference_total)
    aggregate_passed = all(row["strictly_lower"] for row in comparison.values())
    checks = {
        "passed_development_is_valid_and_unchanged": True,
        "registered_roster_is_disjoint_from_grid_and_six_prior_rosters": len(queues) == len(set(queues)) == ROSTER_SIZE,
        "every_lane_execution_check_passes": all(all(row.values()) for row in execution_checks),
        "candidate_and_reference_state_counts_are_exact": sum(row["state_count"] for row in rows) == 2 * ROSTER_SIZE,
        "no_arrivals_trajectories_execution_rng_or_server_pulls": True,
    }
    execution_passed = all(checks.values())
    decision = _decision(execution_passed, rows, aggregate_passed)
    passed = decision == PASS_DECISION
    report = {
        "schema_version": 1, "gate": GATE, "experiment": EXPERIMENT,
        "status": "PASS" if execution_passed else "FAIL", "protocol": protocol, "checks": checks,
        "development_scientific_decision": SOURCE_DECISION,
        "retained_v5_capacity_established": False, "retained_v5_closure_passed": False,
        "capacity_closure_reclassified": False, "independent_roster_seed": ROSTER_SEED,
        "independent_roster_state_count": ROSTER_SIZE, "lane_summaries": rows,
        "aggregate_actual_compute": {"candidate": candidate_total, "cold_canonical_reference": reference_total,
                                     "comparison": comparison, "gate_passed": aggregate_passed},
        "scientific_decision": decision, "independent_static_confirmation_passed": passed,
        "envelope_normalized_trajectory_registration_authorized": passed,
        "trajectory_registration_prerequisite_passed": passed,
        "trajectory_runs": 0, "arrival_generations": 0, "rng_draws_during_execution": 0,
        "server_pulls": 0, "claim_boundary": CLAIM_BOUNDARY,
        "timing": {"wall_s": time.perf_counter() - started, "scientific_gate": False},
    }
    _write_new(Path(output), report)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--register", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    parser.add_argument("--roster", type=Path, default=DEFAULT_ROSTER)
    parser.add_argument("--fixed-sources", type=Path, default=DEFAULT_FIXED_SOURCES)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULT)
    args = parser.parse_args(argv)
    try:
        payload = (register(args.registration, args.roster, args.fixed_sources) if args.register else
                   run_registered(args.registration, args.roster, args.fixed_sources, args.output))
    except R16SupportRescueConfirmationRefusal as exc:
        raise SystemExit(f"REFUSED: {exc}") from exc
    print(json.dumps({"status": payload["status"], "experiment": payload["experiment"],
                      "scientific_decision": payload.get("scientific_decision")}, indent=2))


if __name__ == "__main__":
    main()
