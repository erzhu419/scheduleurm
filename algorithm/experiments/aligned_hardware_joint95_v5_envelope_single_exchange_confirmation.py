"""Independent static confirmation of the fixed joint95-v5 polished method.

Registration freezes a new 512-state base-queue roster before any selector
call.  The roster excludes the old 1,000-state grid and all four earlier
512-state rosters, including the failed seed-260919 confirmation used to
develop the polish.  Execution compares fixed R8/K48-plus-single-exchange and
R16/K9 calls with separate cold K4096 canonical references on the immutable v5
envelope.  No arrivals or trajectories run.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from itertools import product
import json
import math
from pathlib import Path
import random
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
    aligned_hardware_joint95_v5_envelope_single_exchange_development as development,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
STEM = "aligned_hardware_joint95_v5_envelope_single_exchange_confirmation_20260919"
DEFAULT_REGISTRATION = ARTIFACTS / f"{STEM}_protocol.json"
DEFAULT_ROSTER = ARTIFACTS / f"{STEM}_roster.json"
DEFAULT_FIXED_SOURCES = ARTIFACTS / f"{STEM}_fixed_sources.json"
DEFAULT_RESULT = ARTIFACTS / f"{STEM}.json"
DEFAULT_DEVELOPMENT_REGISTRATION = development.DEFAULT_REGISTRATION
DEFAULT_DEVELOPMENT_FIXED_SOURCES = development.DEFAULT_FIXED_SOURCES
DEFAULT_DEVELOPMENT_RESULT = development.DEFAULT_RESULT
previous = development.source
DEFAULT_COUNT_ROSTER = previous.DEFAULT_COUNT_ROSTER
DEFAULT_STATIC_ROSTER = previous.DEFAULT_STATIC_ROSTER
DEFAULT_CANONICAL_ROSTER = previous.DEFAULT_CANONICAL_ROSTER
DEFAULT_CANONICAL_PROTOCOL = previous.DEFAULT_CANONICAL_PROTOCOL
DEFAULT_PRIOR_CONFIRMATION_REGISTRATION = previous.DEFAULT_REGISTRATION
DEFAULT_PRIOR_CONFIRMATION_ROSTER = previous.DEFAULT_ROSTER
DEFAULT_PRIOR_CONFIRMATION_FIXED_SOURCES = previous.DEFAULT_FIXED_SOURCES

canonical = development.canonical
EXPERIMENT = "aligned_hardware_joint95_v5_envelope_single_exchange_confirmation_v1"
GATE = "aligned_hardware_joint95_v5_envelope_single_exchange_confirmation_runner"
REGISTRATION_STATUS = "REGISTERED_AWAITING_INDEPENDENT_STATIC_CONFIRMATION"
SOURCE_DECISION = development.PASS_DECISION
PASS_DECISION = "INDEPENDENT_ENVELOPE_SINGLE_EXCHANGE_STATIC_CONFIRMATION_PASS"
INVALID_DECISION = "INVALID_EXECUTION"
SUPPORT_FAIL_DECISION = "INDEPENDENT_STATIC_SUPPORT_FAIL_STOP"
ACTION_FAIL_DECISION = "INDEPENDENT_STATIC_ACTION_IDENTITY_FAIL_STOP"
COMPUTE_FAIL_DECISION = "INDEPENDENT_STATIC_COMPUTE_FAIL_STOP"
ROSTER_SEED = 270919
ROSTER_SIZE = 512
ROSTER_CANDIDATE_DRAWS = 549
ROSTER_RNG_INTEGER_DRAWS = 1647
ROSTER_RANGES = canonical.ROSTER_RANGES
R8_LANES = 8
R16_LANES = 16
FIXED_BUDGET_BY_LANE = {8: 48, 16: 9}
REFERENCE_BUDGET = 4096
LOADS = canonical.LOADS
WORK_GATE_FIELDS = previous.WORK_GATE_FIELDS
THIS_SOURCE = (
    "algorithm/experiments/"
    "aligned_hardware_joint95_v5_envelope_single_exchange_confirmation.py"
)
ROSTER_GENERATION_RULE = (
    "draw pseudorandom integer coordinates from random.Random(270919); reject the "
    "all-zero queue, the old 1,000-state grid, every queue in each of the four "
    "prior 512-state rosters, and within-roster duplicates; preserve accepted order"
)
SELECTOR_CONTRACT = {
    "family": "fixed canonical-v2 count-space selectors with R8 single exchange",
    "fixed_budget_by_lane": {"8": 48, "16": 9},
    "cold_canonical_reference_budget": REFERENCE_BUDGET,
    "r8_candidate": (
        "one fresh cold K48 call plus the frozen unique count-space single-exchange "
        "polish per registered queue"
    ),
    "r8_method_contract": json.loads(json.dumps(development.METHOD_CONTRACT)),
    "r16_candidate": "one fresh cold K9 base-selector call per registered queue",
    "reference": "one separate fresh cold K4096 call per registered queue and lane",
    "reference_gate": "every reference is score-exact and canonical-exact",
    "support_gate": (
        "candidate epsilon and actual normalized support regret are strictly below "
        "eta at every envelope-normalized load"
    ),
    "heldout_roster_action_identity_gate": (
        "candidate option indices equal the exact canonical reference on all states"
    ),
    "actual_work_gate": (
        "charged candidate LP calls, matrix coefficients, and candidate evaluations "
        "are each strictly below the reference per lane and in aggregate; R8 charges "
        "every exchange attempt to candidate evaluations and residual option probes"
    ),
}
EXPECTED_WORK = {
    "registered_base_queues": ROSTER_SIZE,
    "scaled_queue_states": ROSTER_SIZE * len(FIXED_BUDGET_BY_LANE),
    "r8_composite_candidate_calls": ROSTER_SIZE,
    "r16_base_candidate_calls": ROSTER_SIZE,
    "cold_reference_selector_calls": ROSTER_SIZE * len(FIXED_BUDGET_BY_LANE),
    "ordered_action_evaluations": 0,
    "exchange_attempts": "data-dependent and fully charged during execution",
    "run_rng_initializations": 0,
    "arrival_generations": 0,
    "trajectory_runs": 0,
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
    INVALID_DECISION: "one or more registered execution-integrity checks fail",
    SUPPORT_FAIL_DECISION: (
        "execution is valid but at least one lane/load support gate fails"
    ),
    ACTION_FAIL_DECISION: (
        "support passes but at least one candidate action differs from its exact "
        "canonical reference on the independent roster"
    ),
    COMPUTE_FAIL_DECISION: (
        "support and independent-roster action identity pass, but at least one "
        "registered work comparison is not strict"
    ),
    PASS_DECISION: (
        "both lanes pass valid-certificate, all-load support, independent-roster "
        "exact-reference action identity, and strict actual-work gates"
    ),
}
CLAIM_BOUNDARY = (
    "A preregistered independent finite-roster static confirmation of fixed "
    "R8/K48-plus-single-exchange and R16/K9 on the immutable joint95-v5 physical "
    "envelope. It can authorize "
    "registration of a separate envelope-normalized trajectory experiment. It does "
    "not prove candidate-side canonical exactness, action identity outside this "
    "roster, capacity at the old point-boundary loads, stability, an all-queue "
    "theorem, a physical multi-GPU result, or deployment readiness."
)
PROHIBITED = (
    "changing the frozen v5 capacity result, post-result development result, method, "
    "budgets, loads, "
    "seed, roster, exclusions, reference, or gates after registration; reusing any "
    "prior roster state; cumulative or warm-start candidate/reference sharing; "
    "run-time RNG; arrivals, trajectories, server pulls, and paper edits"
)

SOURCE_R8_OUTCOME = {
    "seed13041": {
        "polish_trigger_state_count": 7,
        "exchange_attempt_count": 198,
        "feasible_exchange_count": 163,
        "changed_incumbent_state_count": 0,
        "candidate_action_matches_reference_count": 512,
        "candidate_work": {
            "lp_calls": 9139,
            "lp_matrix_coefficients": 741117,
            "candidate_evaluations": 9849,
        },
        "reference_work": {
            "lp_calls": 9448,
            "lp_matrix_coefficients": 777270,
            "candidate_evaluations": 9960,
        },
    },
    "seed260919": {
        "polish_trigger_state_count": 2,
        "exchange_attempt_count": 45,
        "feasible_exchange_count": 41,
        "changed_incumbent_state_count": 1,
        "candidate_action_matches_reference_count": 512,
        "candidate_work": {
            "lp_calls": 9156,
            "lp_matrix_coefficients": 742884,
            "candidate_evaluations": 9713,
        },
        "reference_work": {
            "lp_calls": 9229,
            "lp_matrix_coefficients": 751425,
            "candidate_evaluations": 9741,
        },
    },
}
SOURCE_R8_AGGREGATE = {
    "candidate": {
        "lp_calls": 18295,
        "lp_matrix_coefficients": 1484001,
        "candidate_evaluations": 19562,
    },
    "reference": {
        "lp_calls": 18677,
        "lp_matrix_coefficients": 1528695,
        "candidate_evaluations": 19701,
    },
}


class SingleExchangeConfirmationRefusal(RuntimeError):
    """The independent static confirmation contract is not satisfied."""


def _json(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SingleExchangeConfirmationRefusal(
            f"cannot read JSON object {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise SingleExchangeConfirmationRefusal(f"JSON root must be an object: {path}")
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
    return {
        **development._current_fixed_sources(),
        THIS_SOURCE: (ROOT / THIS_SOURCE).read_text(encoding="utf-8"),
    }


def _validated_development_contract(
    result: Mapping[str, Any], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        development._validate_protocol(protocol)
    except development.SingleExchangeDevelopmentRefusal as exc:
        raise SingleExchangeConfirmationRefusal(str(exc)) from exc
    source_paths = protocol.get("source_paths") or {}
    failed_protocol = _read_object(
        Path(source_paths["failed_confirmation_registration"])
    )
    failed_result = _read_object(Path(source_paths["failed_confirmation_result"]))
    if (
        _read_object(Path(source_paths["failed_confirmation_fixed_sources"]))
        != previous._current_fixed_sources()
    ):
        raise SingleExchangeConfirmationRefusal(
            "post-result development source confirmation changed"
        )
    try:
        authoritative = development._validated_source_contract(
            failed_result,
            failed_protocol,
            Path(source_paths["failed_confirmation_roster"]),
        )
        consumed = development._load_consumed_rosters(authoritative)
    except development.SingleExchangeDevelopmentRefusal as exc:
        raise SingleExchangeConfirmationRefusal(str(exc)) from exc
    consumed_snapshot = {
        name: [list(row) for row in rows] for name, rows in consumed.items()
    }
    lane_rows = result.get("r8_consumed_roster_summaries") or ()
    by_roster = {row.get("roster"): row for row in lane_rows}
    if (
        result.get("schema_version") != 1
        or result.get("gate") != development.GATE
        or result.get("experiment") != development.EXPERIMENT
        or result.get("status") != "PASS"
        or result.get("protocol") != protocol
        or result.get("scientific_decision") != SOURCE_DECISION
        or result.get(
            "new_independent_static_confirmation_registration_authorized"
        )
        is not True
        or result.get("trajectory_registration_prerequisite_passed") is not False
        or result.get("capacity_closure_reclassified") is not False
        or result.get("retained_v5_capacity_established") is not False
        or result.get("retained_v5_closure_passed") is not False
        or not result.get("checks")
        or not all(result["checks"].values())
        or authoritative != protocol.get("source_contract")
        or consumed_snapshot != protocol.get("consumed_rosters")
        or protocol.get("method_contract") != development.METHOD_CONTRACT
        or set(by_roster) != set(SOURCE_R8_OUTCOME)
        or any(
            not row.get("checks")
            or not all(row["checks"].values())
            or row.get("support_gate_passed") is not True
            or row.get("action_identity_gate_passed") is not True
            or row.get("charged_compute_gate_passed") is not True
            or row.get("development_gate_passed") is not True
            for row in by_roster.values()
        )
    ):
        raise SingleExchangeConfirmationRefusal(
            "frozen single-exchange development result differs"
        )
    for name, expected in SOURCE_R8_OUTCOME.items():
        row = by_roster[name]
        observed = {
            key: row.get(key)
            for key in (
                "polish_trigger_state_count",
                "exchange_attempt_count",
                "feasible_exchange_count",
                "changed_incumbent_state_count",
                "candidate_action_matches_reference_count",
            )
        }
        observed["candidate_work"] = {
            field: row["candidate_charged_work"][field]
            for field in WORK_GATE_FIELDS
        }
        observed["reference_work"] = {
            field: row["cold_reference_actual_work"][field]
            for field in WORK_GATE_FIELDS
        }
        if observed != expected:
            raise SingleExchangeConfirmationRefusal(
                f"frozen single-exchange outcome changed: {name}"
            )
    aggregate = result.get("r8_aggregate_charged_compute") or {}
    observed_aggregate = {
        "candidate": {
            field: (aggregate.get("candidate") or {}).get(field)
            for field in WORK_GATE_FIELDS
        },
        "reference": {
            field: (aggregate.get("cold_reference") or {}).get(field)
            for field in WORK_GATE_FIELDS
        },
    }
    retained_r16 = result.get("retained_r16_k9_evidence") or {}
    if (
        observed_aggregate != SOURCE_R8_AGGREGATE
        or aggregate.get("gate_passed") is not True
        or set(retained_r16) != set(development.ROSTER_NAMES)
        or any(
            row.get("candidate_action_matches_reference_count") != ROSTER_SIZE
            or row.get("support_gate_passed") is not True
            or row.get("actual_compute_gate_passed") is not True
            for row in retained_r16.values()
        )
    ):
        raise SingleExchangeConfirmationRefusal(
            "frozen single-exchange aggregate or R16 evidence changed"
        )
    return {
        "experiment": result["experiment"],
        "status": result["status"],
        "scientific_decision": result["scientific_decision"],
        "method_contract": _json(development.METHOD_CONTRACT),
        "new_independent_static_confirmation_registration_authorized": True,
        "trajectory_registration_prerequisite_passed": False,
        "capacity_closure_reclassified": False,
        "physical_complete_frame_envelope": _json(
            authoritative["physical_complete_frame_envelope"]
        ),
        "physical_envelope_capacity_normalized_loads": _json(
            authoritative["physical_envelope_capacity_normalized_loads"]
        ),
        "development_roster_contract": _json(
            authoritative["seed13041_roster_contract"]
        ),
        "prior_confirmation_roster_contract": _json(
            authoritative["seed260919_roster_contract"]
        ),
        "source_r8_outcome": _json(SOURCE_R8_OUTCOME),
        "source_r8_aggregate": _json(SOURCE_R8_AGGREGATE),
    }


def _old_grid() -> set[tuple[int, int, int]]:
    return set(product(
        canonical.OLD_CNN_LEVELS,
        canonical.OLD_JAX_LEVELS,
        canonical.OLD_LLM_LEVELS,
    ))


def _validated_prior_roster(
    payload: Mapping[str, Any], *, experiment: str, seed: int,
    expected_queues: Sequence[Sequence[int]],
) -> tuple[tuple[int, int, int], ...]:
    queues = tuple(
        tuple(int(value) for value in row) for row in payload.get("queues") or ()
    )
    if (
        payload.get("experiment") != experiment
        or payload.get("seed") != seed
        or payload.get("requested_queue_count") != ROSTER_SIZE
        or payload.get("coordinate_ranges_inclusive")
        != [list(bounds) for bounds in ROSTER_RANGES]
        or len(queues) != ROSTER_SIZE
        or len(set(queues)) != ROSTER_SIZE
        or [list(queue) for queue in queues] != _json(expected_queues)
        or any(
            len(queue) != len(ROSTER_RANGES)
            or queue == (0, 0, 0)
            or any(
                not lower <= value <= upper
                for value, (lower, upper) in zip(
                    queue, ROSTER_RANGES, strict=True
                )
            )
            for queue in queues
        )
    ):
        raise SingleExchangeConfirmationRefusal(
            f"prior roster is invalid: experiment={experiment}, seed={seed}"
        )
    return queues


def _load_exclusion_rosters(paths: Mapping[str, Any]) -> dict[str, tuple[tuple[int, int, int], ...]]:
    historical_protocol = _read_object(
        Path(paths["canonical_development_protocol"])
    )
    try:
        canonical._validate_protocol(historical_protocol)
    except canonical.CanonicalStaticScreenRefusal as exc:
        raise SingleExchangeConfirmationRefusal(str(exc)) from exc
    historical = historical_protocol.get("queue_roster_contract") or {}
    count = _validated_prior_roster(
        _read_object(Path(paths["count_roster"])),
        experiment=canonical.legacy.PREVIOUS_ROSTER_EXPERIMENT,
        seed=canonical.legacy.PREVIOUS_ROSTER_SEED,
        expected_queues=(historical.get("old_count_512_roster") or {}).get(
            "queues"
        ) or (),
    )
    static = _validated_prior_roster(
        _read_object(Path(paths["joint95_static_roster"])),
        experiment=canonical.legacy.EXPERIMENT,
        seed=canonical.legacy.ROSTER_SEED,
        expected_queues=(historical.get("old_joint95_static_512_roster") or {}).get(
            "queues"
        ) or (),
    )
    current = _validated_prior_roster(
        _read_object(Path(paths["canonical_development_roster"])),
        experiment=canonical.EXPERIMENT,
        seed=canonical.ROSTER_SEED,
        expected_queues=historical.get("base_queues") or (),
    )
    prior_protocol = _read_object(Path(paths["prior_confirmation_registration"]))
    try:
        previous._validate_protocol(prior_protocol)
    except previous.BudgetPairConfirmationRefusal as exc:
        raise SingleExchangeConfirmationRefusal(str(exc)) from exc
    if (
        _read_object(Path(paths["prior_confirmation_fixed_sources"]))
        != previous._current_fixed_sources()
    ):
        raise SingleExchangeConfirmationRefusal(
            "prior confirmation frozen sources changed"
        )
    prior = _validated_prior_roster(
        _read_object(Path(paths["prior_confirmation_roster"])),
        experiment=previous.EXPERIMENT,
        seed=previous.ROSTER_SEED,
        expected_queues=(prior_protocol.get("queue_roster_contract") or {}).get(
            "queues"
        )
        or (),
    )
    return {
        "count_space_confirmation": count,
        "joint95_static": static,
        "canonical_development": current,
        "budget_pair_confirmation": prior,
    }


def _exclusion_contract(
    exclusions: Mapping[str, Sequence[Sequence[int]]]
) -> dict[str, Any]:
    metadata = {
        "count_space_confirmation": {
            "experiment": canonical.legacy.PREVIOUS_ROSTER_EXPERIMENT,
            "seed": canonical.legacy.PREVIOUS_ROSTER_SEED,
        },
        "joint95_static": {
            "experiment": canonical.legacy.EXPERIMENT,
            "seed": canonical.legacy.ROSTER_SEED,
        },
        "canonical_development": {
            "experiment": canonical.EXPERIMENT,
            "seed": canonical.ROSTER_SEED,
        },
        "budget_pair_confirmation": {
            "experiment": previous.EXPERIMENT,
            "seed": previous.ROSTER_SEED,
        },
    }
    return {
        name: {
            **metadata[name],
            "size": len(rows),
            "queues": [list(row) for row in rows],
        }
        for name, rows in exclusions.items()
    }


def generate_roster(
    exclusions: Mapping[str, Sequence[Sequence[int]]]
) -> dict[str, Any]:
    sets = {
        name: {tuple(int(value) for value in row) for row in rows}
        for name, rows in exclusions.items()
    }
    if set(sets) != {
        "count_space_confirmation",
        "joint95_static",
        "canonical_development",
        "budget_pair_confirmation",
    } or any(len(rows) != ROSTER_SIZE for rows in sets.values()):
        raise SingleExchangeConfirmationRefusal("exclusion rosters are incomplete")
    union = set().union(*sets.values())
    if len(union) != 4 * ROSTER_SIZE:
        raise SingleExchangeConfirmationRefusal("prior 512-state rosters are not disjoint")
    old = _old_grid()
    rng = random.Random(ROSTER_SEED)
    queues: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    draws = 0
    while len(queues) < ROSTER_SIZE:
        draws += 1
        queue = tuple(
            rng.randint(lower, upper) for lower, upper in ROSTER_RANGES
        )
        if (
            queue == (0, 0, 0)
            or queue in old
            or queue in union
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
        "candidate_draws": draws,
        "rng_integer_draws": draws * len(ROSTER_RANGES),
        "generation_rule": ROSTER_GENERATION_RULE,
        "zero_queue_excluded": True,
        "old_1000_grid_excluded": True,
        "all_four_prior_512_rosters_excluded": True,
        "old_1000_grid_intersection_count": sum(q in old for q in queues),
        "prior_roster_union_size": len(union),
        "prior_roster_intersection_count": {
            name: sum(q in rows for q in queues) for name, rows in sets.items()
        },
        "queues": [list(queue) for queue in queues],
    }


def build_registration(
    *,
    development_registration: Path = DEFAULT_DEVELOPMENT_REGISTRATION,
    development_fixed_sources: Path = DEFAULT_DEVELOPMENT_FIXED_SOURCES,
    development_result: Path = DEFAULT_DEVELOPMENT_RESULT,
    count_roster: Path = DEFAULT_COUNT_ROSTER,
    static_roster: Path = DEFAULT_STATIC_ROSTER,
    canonical_roster: Path = DEFAULT_CANONICAL_ROSTER,
    canonical_protocol: Path = DEFAULT_CANONICAL_PROTOCOL,
    prior_confirmation_registration: Path = DEFAULT_PRIOR_CONFIRMATION_REGISTRATION,
    prior_confirmation_roster: Path = DEFAULT_PRIOR_CONFIRMATION_ROSTER,
    prior_confirmation_fixed_sources: Path = DEFAULT_PRIOR_CONFIRMATION_FIXED_SOURCES,
) -> dict[str, Any]:
    dev_protocol = _read_object(Path(development_registration))
    dev_result = _read_object(Path(development_result))
    if (
        _read_object(Path(development_fixed_sources))
        != development._current_fixed_sources()
    ):
        raise SingleExchangeConfirmationRefusal("frozen development sources changed")
    contract = _validated_development_contract(dev_result, dev_protocol)
    source_paths = {
        "development_registration": str(Path(development_registration).resolve()),
        "development_fixed_sources": str(Path(development_fixed_sources).resolve()),
        "development_result": str(Path(development_result).resolve()),
        "count_roster": str(Path(count_roster).resolve()),
        "joint95_static_roster": str(Path(static_roster).resolve()),
        "canonical_development_roster": str(Path(canonical_roster).resolve()),
        "canonical_development_protocol": str(Path(canonical_protocol).resolve()),
        "prior_confirmation_registration": str(
            Path(prior_confirmation_registration).resolve()
        ),
        "prior_confirmation_roster": str(Path(prior_confirmation_roster).resolve()),
        "prior_confirmation_fixed_sources": str(
            Path(prior_confirmation_fixed_sources).resolve()
        ),
    }
    exclusions = _load_exclusion_rosters(source_paths)
    if list(exclusions["canonical_development"]) != [
        tuple(row) for row in contract["development_roster_contract"]["queues"]
    ]:
        raise SingleExchangeConfirmationRefusal(
            "seed13041 roster differs from the development source contract"
        )
    prior_contract = contract["prior_confirmation_roster_contract"]
    if (
        str(Path(prior_confirmation_roster).resolve()) != prior_contract["path"]
        or [list(row) for row in exclusions["budget_pair_confirmation"]]
        != prior_contract["queues"]
        or _read_object(Path(prior_confirmation_roster))
        != prior_contract["payload"]
    ):
        raise SingleExchangeConfirmationRefusal(
            "seed260919 roster differs from the development source contract"
        )
    roster = generate_roster(exclusions)
    if (
        roster["candidate_draws"] != ROSTER_CANDIDATE_DRAWS
        or roster["rng_integer_draws"] != ROSTER_RNG_INTEGER_DRAWS
    ):
        raise SingleExchangeConfirmationRefusal(
            "registered roster draw accounting changed"
        )
    return {
        "schema_version": 1,
        "gate": GATE,
        "experiment": EXPERIMENT,
        "status": REGISTRATION_STATUS,
        "registered": True,
        "independent_confirmation": True,
        "source_paths": source_paths,
        "development_contract": contract,
        "exclusion_roster_contract": _exclusion_contract(exclusions),
        "queue_roster_contract": roster,
        "lane_counts": sorted(FIXED_BUDGET_BY_LANE),
        "queue_scale_by_lane": {
            str(lanes): lanes // 2 for lanes in FIXED_BUDGET_BY_LANE
        },
        "physical_envelope_capacity_normalized_loads": list(LOADS),
        "selector_contract": _json(SELECTOR_CONTRACT),
        "expected_work": _json(EXPECTED_WORK),
        "decision_rule": dict(DECISION_RULE),
        "claim_boundary": CLAIM_BOUNDARY,
        "prohibited": PROHIBITED,
        "registration_scope": dict(REGISTRATION_SCOPE),
    }


def register(
    registration: Path = DEFAULT_REGISTRATION,
    roster_path: Path = DEFAULT_ROSTER,
    fixed_sources: Path = DEFAULT_FIXED_SOURCES,
    **kwargs: Any,
) -> dict[str, Any]:
    paths = (Path(registration), Path(roster_path), Path(fixed_sources), DEFAULT_RESULT)
    if any(path.exists() for path in paths):
        raise SingleExchangeConfirmationRefusal("confirmation artifact already exists")
    protocol = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        **build_registration(**kwargs),
    }
    _write_new(Path(roster_path), protocol["queue_roster_contract"], compact=True)
    _write_new(Path(fixed_sources), _current_fixed_sources())
    _write_new(Path(registration), protocol)
    return protocol


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    scope = protocol.get("registration_scope") or {}
    exclusion_contract = protocol.get("exclusion_roster_contract") or {}
    if (
        protocol.get("schema_version") != 1
        or protocol.get("gate") != GATE
        or protocol.get("experiment") != EXPERIMENT
        or protocol.get("status") != REGISTRATION_STATUS
        or protocol.get("registered") is not True
        or protocol.get("independent_confirmation") is not True
        or set(exclusion_contract) != {
            "count_space_confirmation",
            "joint95_static",
            "canonical_development",
            "budget_pair_confirmation",
        }
        or any(
            row.get("size") != ROSTER_SIZE
            or len(row.get("queues") or ()) != ROSTER_SIZE
            for row in exclusion_contract.values()
        )
        or protocol.get("lane_counts") != [8, 16]
        or protocol.get("queue_scale_by_lane") != {"8": 4, "16": 8}
        or protocol.get("physical_envelope_capacity_normalized_loads")
        != list(LOADS)
        or protocol.get("selector_contract") != SELECTOR_CONTRACT
        or protocol.get("expected_work") != EXPECTED_WORK
        or protocol.get("decision_rule") != DECISION_RULE
        or protocol.get("claim_boundary") != CLAIM_BOUNDARY
        or protocol.get("prohibited") != PROHIBITED
        or scope != REGISTRATION_SCOPE
        or scope.get("selector_invoked") is not False
        or scope.get("trajectory_built") is not False
        or scope.get("paper_edited") is not False
    ):
        raise SingleExchangeConfirmationRefusal(
            "registered confirmation protocol changed"
        )


def _validate_roster(
    roster: Mapping[str, Any],
    protocol: Mapping[str, Any],
    exclusions: Mapping[str, Sequence[Sequence[int]]],
) -> tuple[tuple[int, int, int], ...]:
    expected = protocol.get("queue_roster_contract") or {}
    queues = tuple(
        tuple(int(value) for value in row) for row in roster.get("queues") or ()
    )
    union = {
        tuple(int(value) for value in row)
        for rows in exclusions.values()
        for row in rows
    }
    old = _old_grid()
    if (
        roster != expected
        or roster.get("schema_version") != 1
        or roster.get("experiment") != EXPERIMENT
        or roster.get("seed") != ROSTER_SEED
        or roster.get("coordinate_ranges_inclusive")
        != [list(bounds) for bounds in ROSTER_RANGES]
        or roster.get("requested_queue_count") != ROSTER_SIZE
        or roster.get("candidate_draws") != ROSTER_CANDIDATE_DRAWS
        or roster.get("rng_integer_draws") != ROSTER_RNG_INTEGER_DRAWS
        or roster.get("generation_rule") != ROSTER_GENERATION_RULE
        or len(queues) != ROSTER_SIZE
        or len(set(queues)) != ROSTER_SIZE
        or roster.get("all_four_prior_512_rosters_excluded") is not True
        or roster.get("prior_roster_union_size") != 4 * ROSTER_SIZE
        or roster.get("old_1000_grid_intersection_count") != 0
        or any(roster.get("prior_roster_intersection_count", {}).values())
        or any(
            queue == (0, 0, 0)
            or queue in old
            or queue in union
            or any(
                not lower <= value <= upper
                for value, (lower, upper) in zip(
                    queue, ROSTER_RANGES, strict=True
                )
            )
            for queue in queues
        )
    ):
        raise SingleExchangeConfirmationRefusal(
            "registered independent roster changed or is invalid"
        )
    return queues


def _add_work_row(target: dict[str, int], work: Mapping[str, int]) -> None:
    for field in canonical.WORK_FIELDS:
        target[field] += int(work[field])


def _polish_relation_holds(polished: Any, queue: Sequence[int], option_count: int) -> bool:
    base = polished.base_selection
    selected = polished.selection
    triggered = not base.score_exact
    occupied_symbols = sum(int(value > 0) for value in base.option_counts)
    occupied_symbols += int(sum(base.option_counts) < R8_LANES)
    expected_attempts = occupied_symbols * option_count if triggered else 0
    expected_gap = max(0.0, selected.upper_bound - selected.score)
    expected_epsilon = expected_gap / sum(queue)
    return bool(
        polished.polish_triggered is triggered
        and polished.exchange_attempts == expected_attempts
        and polished.feasible_exchange_count <= polished.exchange_attempts
        and polished.score_improving_exchange_count <= polished.feasible_exchange_count
        and polished.canonical_tie_improving_exchange_count
        <= polished.feasible_exchange_count
        and polished.incumbent_changed
        == (selected.option_indices != base.option_indices)
        and selected.upper_bound == base.upper_bound
        and selected.score_exact == base.score_exact
        and selected.canonical_exact == base.canonical_exact
        and all(
            getattr(selected, field) == getattr(base, field)
            for field in canonical.WORK_FIELDS
        )
        and selected.score + canonical.SCORE_TOLERANCE >= base.score
        and math.isclose(
            selected.gap,
            expected_gap,
            rel_tol=0.0,
            abs_tol=canonical.SCORE_TOLERANCE,
        )
        and math.isclose(
            selected.epsilon,
            expected_epsilon,
            rel_tol=0.0,
            abs_tol=canonical.SCORE_TOLERANCE,
        )
        and (
            triggered
            or (
                polished.exchange_attempts == 0
                and polished.feasible_exchange_count == 0
                and polished.incumbent_changed is False
                and selected == base
            )
        )
    )


def _screen_lane(
    model: Any,
    base_queues: Sequence[Sequence[int]],
    lanes: int,
    capacity_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, bool]]:
    budget = FIXED_BUDGET_BY_LANE[lanes]
    factor = lanes // 2
    candidate_work = canonical._new_work()
    base_candidate_work = canonical._new_work()
    reference_work = canonical._new_work()
    max_epsilon = -1.0
    max_regret = -1.0
    epsilon_witness = None
    regret_witness = None
    valid_candidates = 0
    valid_references = 0
    bounded_regrets = 0
    score_exact = 0
    canonical_exact = 0
    action_matches = 0
    score_tie_action_mismatches = 0
    positive_regret_action_mismatches = 0
    first_action_mismatch_witness = None
    polish_trigger_count = 0
    exchange_attempts = 0
    feasible_exchanges = 0
    changed_incumbents = 0
    polish_relations = 0

    for base_queue in base_queues:
        queue = tuple(factor * int(value) for value in base_queue)
        if lanes == R8_LANES:
            polished = select_canonical_count_space_bnb_single_exchange_frame(
                model.options, queue, lanes, budget
            )
            base_candidate = polished.base_selection
            candidate = polished.selection
            candidate_row = total_work_row(polished)
            expected_candidate_row = {
                field: int(getattr(base_candidate, field))
                for field in canonical.WORK_FIELDS
            }
            expected_candidate_row["candidate_evaluations"] += (
                polished.exchange_attempts
            )
            expected_candidate_row["residual_fill_option_probes"] += (
                polished.exchange_attempts
            )
            relation_holds = _polish_relation_holds(
                polished, queue, len(model.options)
            ) and candidate_row == expected_candidate_row
            polish_trigger_count += int(polished.polish_triggered)
            exchange_attempts += polished.exchange_attempts
            feasible_exchanges += polished.feasible_exchange_count
            changed_incumbents += int(polished.incumbent_changed)
            polish_relations += int(relation_holds)
        else:
            candidate = select_canonical_count_space_bnb_frame(
                model.options, queue, lanes, budget
            )
            base_candidate = candidate
            candidate_row = {
                field: int(getattr(candidate, field))
                for field in canonical.WORK_FIELDS
            }
            relation_holds = True
            polish_relations += 1
        reference = select_canonical_count_space_bnb_frame(
            model.options, queue, lanes, REFERENCE_BUDGET
        )
        candidate_valid = (
            canonical._selection_reconstructs(model, queue, lanes, base_candidate)
            and canonical._selection_reconstructs(model, queue, lanes, candidate)
            and canonical._selection_work_holds(
                queue, lanes, budget, base_candidate
            )
            and canonical._certificate_valid(base_candidate, queue)
            and canonical._certificate_valid(candidate, queue)
            and relation_holds
        )
        reference_valid = (
            canonical._selection_reconstructs(model, queue, lanes, reference)
            and canonical._selection_work_holds(
                queue, lanes, REFERENCE_BUDGET, reference
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
        norm = sum(queue)
        regret = (reference.score - candidate.score) / norm
        bounded = (
            regret >= -canonical.SCORE_TOLERANCE
            and regret <= candidate.epsilon + canonical.SCORE_TOLERANCE
        )
        valid_candidates += int(candidate_valid)
        valid_references += int(reference_valid)
        bounded_regrets += int(bounded)
        score_exact += int(candidate.score_exact)
        canonical_exact += int(candidate.canonical_exact)
        action_match = candidate.option_indices == reference.option_indices
        action_matches += int(action_match)
        if not action_match:
            positive = regret > canonical.SCORE_TOLERANCE
            positive_regret_action_mismatches += int(positive)
            score_tie_action_mismatches += int(not positive)
            if first_action_mismatch_witness is None:
                first_action_mismatch_witness = canonical._selection_witness(
                    base_queue, queue, candidate, reference, regret
                )
                first_action_mismatch_witness["mismatch_kind"] = (
                    "positive_regret" if positive else "score_tie"
                )
        canonical._add_work(base_candidate_work, base_candidate)
        _add_work_row(candidate_work, candidate_row)
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

    expected_charged_work = dict(base_candidate_work)
    if lanes == R8_LANES:
        expected_charged_work["candidate_evaluations"] += exchange_attempts
        expected_charged_work["residual_fill_option_probes"] += exchange_attempts
    comparisons = previous.development._work_comparison(
        candidate_work, reference_work
    )
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
            "maximum_actual_normalized_support_regret": max_regret,
            "eta_minus_candidate_epsilon": epsilon_residual,
            "eta_minus_actual_support_regret": regret_residual,
            "support_gate_passed": (
                bool(capacity["capacity_projection_passed"])
                and epsilon_residual > canonical.SCORE_TOLERANCE
                and regret_residual > canonical.SCORE_TOLERANCE
            ),
        })
    checks = {
        "scaled_queue_count_is_exact": len(base_queues) == ROSTER_SIZE,
        "every_candidate_certificate_and_method_relation_is_valid": (
            valid_candidates == ROSTER_SIZE
            and polish_relations == ROSTER_SIZE
        ),
        "every_reference_is_valid_score_exact_and_canonical_exact": (
            valid_references == ROSTER_SIZE
        ),
        "every_actual_regret_is_nonnegative_and_bounded_by_epsilon": (
            bounded_regrets == ROSTER_SIZE
        ),
        "candidate_work_is_exactly_base_plus_registered_exchange_charge": (
            candidate_work == expected_charged_work
        ),
        "single_exchange_is_applied_only_to_r8": (
            lanes == R8_LANES
            or (
                polish_trigger_count == 0
                and exchange_attempts == 0
                and feasible_exchanges == 0
                and changed_incumbents == 0
            )
        ),
    }
    support_passed = all(row["support_gate_passed"] for row in support_rows)
    action_identity_passed = action_matches == ROSTER_SIZE
    compute_passed = all(row["strictly_lower"] for row in comparisons.values())
    return ({
        "lanes": lanes,
        "queue_scale": factor,
        "state_count": len(base_queues),
        "fixed_lp_budget": budget,
        "cold_canonical_reference_budget": REFERENCE_BUDGET,
        "candidate_score_exact_state_count": score_exact,
        "candidate_canonical_exact_state_count": canonical_exact,
        "polish_trigger_state_count": polish_trigger_count,
        "exchange_attempt_count": exchange_attempts,
        "feasible_exchange_count": feasible_exchanges,
        "changed_incumbent_state_count": changed_incumbents,
        "candidate_action_matches_reference_count": action_matches,
        "candidate_action_mismatch_count": ROSTER_SIZE - action_matches,
        "score_tie_action_mismatch_count": score_tie_action_mismatches,
        "positive_regret_action_mismatch_count": (
            positive_regret_action_mismatches
        ),
        "first_action_mismatch_witness": first_action_mismatch_witness,
        "valid_candidate_certificate_count": valid_candidates,
        "valid_canonical_reference_count": valid_references,
        "regret_bounded_by_epsilon_count": bounded_regrets,
        "maximum_candidate_epsilon": max_epsilon,
        "maximum_actual_normalized_support_regret": max_regret,
        "epsilon_witness": epsilon_witness,
        "support_regret_witness": regret_witness,
        "support_by_physical_load": support_rows,
        "candidate_base_work": base_candidate_work,
        "candidate_charged_work": candidate_work,
        "cold_canonical_reference_actual_work": reference_work,
        "actual_compute_comparison": comparisons,
        "checks": checks,
        "support_gate_passed": support_passed,
        "independent_roster_action_identity_gate_passed": action_identity_passed,
        "actual_compute_gate_passed": compute_passed,
        "lane_confirmation_gate_passed": (
            all(checks.values())
            and support_passed
            and action_identity_passed
            and compute_passed
        ),
    }, checks)

def _decision(
    execution_passed: bool,
    lanes: Sequence[Mapping[str, Any]],
    aggregate_compute_passed: bool,
) -> str:
    if not execution_passed:
        return INVALID_DECISION
    if not all(row["support_gate_passed"] for row in lanes):
        return SUPPORT_FAIL_DECISION
    if not all(row["independent_roster_action_identity_gate_passed"] for row in lanes):
        return ACTION_FAIL_DECISION
    if (
        not aggregate_compute_passed
        or not all(row["actual_compute_gate_passed"] for row in lanes)
    ):
        return COMPUTE_FAIL_DECISION
    return PASS_DECISION


def run_registered(
    registration: Path = DEFAULT_REGISTRATION,
    roster_path: Path = DEFAULT_ROSTER,
    fixed_sources: Path = DEFAULT_FIXED_SOURCES,
    output: Path = DEFAULT_RESULT,
) -> dict[str, Any]:
    output = Path(output)
    if output.exists():
        raise SingleExchangeConfirmationRefusal(
            f"confirmation result already exists: {output}"
        )
    protocol = _read_object(Path(registration))
    _validate_protocol(protocol)
    if _read_object(Path(fixed_sources)) != _current_fixed_sources():
        raise SingleExchangeConfirmationRefusal("registered confirmation sources changed")
    paths = protocol["source_paths"]
    dev_protocol = _read_object(Path(paths["development_registration"]))
    dev_result = _read_object(Path(paths["development_result"]))
    if (
        _read_object(Path(paths["development_fixed_sources"]))
        != development._current_fixed_sources()
        or _validated_development_contract(dev_result, dev_protocol)
        != protocol["development_contract"]
    ):
        raise SingleExchangeConfirmationRefusal(
            "development evidence changed after registration"
        )
    exclusions = _load_exclusion_rosters(paths)
    if _exclusion_contract(exclusions) != protocol["exclusion_roster_contract"]:
        raise SingleExchangeConfirmationRefusal(
            "historical exclusion rosters changed after registration"
        )
    base_queues = _validate_roster(
        _read_object(Path(roster_path)), protocol, exclusions
    )
    contract = protocol["development_contract"]
    model = previous.development.source.capacity.base._model_from_snapshot(
        contract["physical_complete_frame_envelope"]["model"]
    )
    if (
        len(model.options) != canonical.OPTION_COUNT
        or len(set(max(1.0, option.duration) for option in model.options))
        != canonical.OPTION_COUNT
    ):
        raise SingleExchangeConfirmationRefusal(
            "v5 envelope does not retain nine duration roots"
        )
    capacity_by_lane = contract["physical_envelope_capacity_normalized_loads"]

    started = time.perf_counter()
    lane_rows = []
    lane_checks = []
    for lanes in sorted(FIXED_BUDGET_BY_LANE):
        row, checks = _screen_lane(
            model, base_queues, lanes, capacity_by_lane[str(lanes)]
        )
        lane_rows.append(row)
        lane_checks.append(checks)

    candidate_total = canonical._new_work()
    reference_total = canonical._new_work()
    for row in lane_rows:
        for field in canonical.WORK_FIELDS:
            candidate_total[field] += row["candidate_charged_work"][field]
            reference_total[field] += row[
                "cold_canonical_reference_actual_work"
            ][field]
    aggregate_comparisons = previous.development._work_comparison(
        candidate_total, reference_total
    )
    aggregate_compute_passed = all(
        row["strictly_lower"] for row in aggregate_comparisons.values()
    )
    checks = {
        "post_result_development_is_valid_and_unchanged": True,
        "v5_negative_capacity_decision_remains_unreclassified": True,
        "registered_roster_is_unique_and_disjoint_from_grid_and_four_prior_rosters": (
            len(base_queues) == len(set(base_queues)) == ROSTER_SIZE
        ),
        "every_lane_execution_check_passes": all(
            all(row.values()) for row in lane_checks
        ),
        "candidate_and_reference_call_counts_are_exact": sum(
            row["state_count"] for row in lane_rows
        )
        == 1024,
        "no_arrivals_trajectories_rng_or_server_pulls": True,
    }
    execution_passed = all(checks.values())
    decision = _decision(execution_passed, lane_rows, aggregate_compute_passed)
    passed = decision == PASS_DECISION
    report = {
        "schema_version": 1,
        "gate": GATE,
        "experiment": EXPERIMENT,
        "status": "PASS" if execution_passed else "FAIL",
        "protocol": protocol,
        "checks": checks,
        "development_scientific_decision": SOURCE_DECISION,
        "retained_v5_capacity_established": False,
        "retained_v5_closure_passed": False,
        "capacity_closure_reclassified": False,
        "independent_roster_seed": ROSTER_SEED,
        "independent_roster_state_count": ROSTER_SIZE,
        "lane_summaries": lane_rows,
        "aggregate_actual_compute": {
            "candidate": candidate_total,
            "cold_canonical_reference": reference_total,
            "comparison": aggregate_comparisons,
            "gate_passed": aggregate_compute_passed,
        },
        "scientific_decision": decision,
        "independent_static_confirmation_passed": passed,
        "envelope_normalized_trajectory_registration_authorized": passed,
        "trajectory_registration_prerequisite_passed": passed,
        "trajectory_runs": 0,
        "arrival_generations": 0,
        "rng_draws_during_execution": 0,
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
    parser.add_argument("--roster", type=Path, default=DEFAULT_ROSTER)
    parser.add_argument("--fixed-sources", type=Path, default=DEFAULT_FIXED_SOURCES)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULT)
    args = parser.parse_args(argv)
    try:
        payload = (
            register(args.registration, args.roster, args.fixed_sources)
            if args.register
            else run_registered(
                args.registration, args.roster, args.fixed_sources, args.output
            )
        )
    except SingleExchangeConfirmationRefusal as exc:
        raise SystemExit(f"REFUSED: {exc}") from exc
    print(json.dumps({
        "status": payload["status"],
        "experiment": payload["experiment"],
        "scientific_decision": payload.get("scientific_decision"),
        "trajectory_registration_prerequisite_passed": payload.get(
            "trajectory_registration_prerequisite_passed"
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
