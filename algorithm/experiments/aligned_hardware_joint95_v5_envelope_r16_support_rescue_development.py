"""Post-result development replay of an R16 support-triggered cold rescue.

The fixed method was designed after the seed-280919 independent confirmation
exposed one positive-regret R16/K9 state.  It consumes all four already used
512-state rosters.  Every state runs the frozen K9-plus-bulk primary; only a
primary whose epsilon lacks the frozen R16 support margin runs a separate cold
K144-plus-bulk rescue.  All work from both calls is charged, and every state is
compared with a separate cold K4096 reference.  Passing can authorize only one
new independent static registration.
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
    total_work_row as bulk_tie_total_work_row,
)
from algorithm.canonical_support_rescue_bulk_tie_frame_candidates import (
    SCORE_TOLERANCE as RESCUE_SCORE_TOLERANCE,
    select_canonical_support_rescue_bulk_tie_frame,
    total_work_row as rescue_total_work_row,
)
from algorithm.experiments import (
    aligned_hardware_joint95_v5_envelope_r16_bulk_tie_confirmation as source,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
STEM = "aligned_hardware_joint95_v5_envelope_r16_support_rescue_development_20260919"
DEFAULT_REGISTRATION = ARTIFACTS / f"{STEM}_protocol.json"
DEFAULT_FIXED_SOURCES = ARTIFACTS / f"{STEM}_fixed_sources.json"
DEFAULT_RESULT = ARTIFACTS / f"{STEM}.json"
DEFAULT_SOURCE_REGISTRATION = source.DEFAULT_REGISTRATION
DEFAULT_SOURCE_ROSTER = source.DEFAULT_ROSTER
DEFAULT_SOURCE_FIXED_SOURCES = source.DEFAULT_FIXED_SOURCES
DEFAULT_SOURCE_RESULT = source.DEFAULT_RESULT

canonical = source.canonical
EXPERIMENT = "aligned_hardware_joint95_v5_envelope_r16_support_rescue_development_v1"
GATE = "aligned_hardware_joint95_v5_envelope_r16_support_rescue_development_runner"
REGISTRATION_STATUS = "REGISTERED_POSTRESULT_AWAITING_DEVELOPMENT_REPLAY"
SOURCE_DECISION = source.SUPPORT_FAIL_DECISION
PASS_DECISION = "R16_SUPPORT_RESCUE_READY_FOR_NEW_INDEPENDENT_STATIC_CONFIRMATION"
INVALID_DECISION = "INVALID_EXECUTION"
SUPPORT_FAIL_DECISION = "R16_SUPPORT_RESCUE_DEVELOPMENT_SUPPORT_FAIL_STOP"
ACTION_FAIL_DECISION = "R16_SUPPORT_RESCUE_DEVELOPMENT_ACTION_IDENTITY_FAIL_STOP"
COMPUTE_FAIL_DECISION = "R16_SUPPORT_RESCUE_DEVELOPMENT_COMPUTE_FAIL_STOP"
R16_LANES = 16
PRIMARY_BUDGET = 9
RESCUE_BUDGET = 144
REFERENCE_BUDGET = 4096
ROSTER_SIZE = 512
ROSTER_NAMES = ("seed13041", "seed260919", "seed270919", "seed280919")
SUPPORT_EPSILON_LIMIT = 0.044725253635528484
THIS_SOURCE = (
    "algorithm/experiments/"
    "aligned_hardware_joint95_v5_envelope_r16_support_rescue_development.py"
)
RESCUE_SOURCE = "algorithm/canonical_support_rescue_bulk_tie_frame_candidates.py"
METHOD_CONTRACT = {
    "lane": R16_LANES,
    "primary_lp_budget": PRIMARY_BUDGET,
    "primary_method": "frozen K9 canonical score-tie bulk-substitution selector",
    "support_epsilon_limit": SUPPORT_EPSILON_LIMIT,
    "support_epsilon_limit_source": (
        "minimum effective_slack_eta over the frozen R16 v5 physical-envelope "
        "capacity rows (the load-0.95 row)"
    ),
    "rescue_trigger": (
        "support_epsilon_limit minus primary epsilon is not greater than the "
        "shared score tolerance"
    ),
    "score_tolerance": RESCUE_SCORE_TOLERANCE,
    "rescue_lp_budget": RESCUE_BUDGET,
    "rescue_budget_rule": (
        "16 lanes multiplied by the frozen nine-option model cardinality"
    ),
    "rescue_method": (
        "one separate fresh cold K144 canonical score-tie bulk-substitution call"
    ),
    "rescue_gate": (
        "every triggered rescue selection is score-exact and canonical-exact"
    ),
    "reference_independence": (
        "neither primary, trigger, nor rescue reads the K4096 action, score, bound, "
        "or work value"
    ),
    "charged_work": (
        "sum every work field from the primary and every actually executed rescue; "
        "each bulk call also charges all structural pair checks it performs"
    ),
    "r8_disposition": "retain the four frozen passing R8 source outcomes without rerun",
}
EXPECTED_WORK = {
    "development_rosters": 4,
    "states_per_roster": ROSTER_SIZE,
    "r16_primary_composite_calls": 4 * ROSTER_SIZE,
    "r16_rescue_composite_calls": "data-dependent and exactly counted",
    "r16_cold_reference_calls": 4 * ROSTER_SIZE,
    "r8_selector_calls": 0,
    "run_rng_initializations": 0,
    "new_rosters": 0,
    "arrival_generations": 0,
    "trajectory_runs": 0,
    "server_pulls": 0,
}
REGISTRATION_SCOPE = {
    "failed_confirmation_outcome_read": True,
    "four_consumed_rosters_read": True,
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
        "the conditional R16 method fails a frozen physical-envelope support gate"
    ),
    ACTION_FAIL_DECISION: (
        "support passes but a final R16 action differs from its exact reference"
    ),
    COMPUTE_FAIL_DECISION: (
        "support and action identity pass but charged LP, matrix, or candidate-"
        "evaluation work is not strictly below cold K4096 per roster or in aggregate"
    ),
    PASS_DECISION: (
        "the fixed conditional method passes support, action identity, and strict "
        "charged-work gates on all four consumed rosters and in aggregate, every "
        "triggered rescue is score/canonical exact, and retained R8 evidence passes"
    ),
}
CLAIM_BOUNDARY = (
    "An outcome-informed development replay on four consumed 512-state rosters. "
    "A pass can authorize preregistration of one new disjoint static confirmation "
    "for the fixed R8/K48-single-exchange and conditional R16 K9/K144 method. It is "
    "not independent confirmation, trajectory authorization, capacity "
    "reclassification, action identity outside these rosters, stability, an "
    "all-queue theorem, or a physical multi-GPU result."
)
PROHIBITED = (
    "changing the support limit, trigger tolerance, budgets, bulk rule, work "
    "accounting, loads, or consumed rosters after registration; reading K4096 inside "
    "the trigger or rescue; using these rosters as confirmation; new rosters, "
    "arrivals, trajectories, run-time RNG, server pulls, capacity reclassification, "
    "and paper edits"
)

SOURCE_R16_WITNESS = {
    "base_queue": [0, 0, 10],
    "candidate": {
        "canonical_exact": False,
        "epsilon": 0.10303913566677281,
        "gap": 8.243130853341825,
        "option_counts": [0, 0, 0, 0, 0, 0, 0, 16, 0],
        "option_indices": [7] * 16,
        "score": 207.66392042612205,
        "score_exact": False,
        "upper_bound": 215.90705127946387,
        "work": {
            "branch_splits": 0,
            "candidate_evaluations": 10,
            "canonical_key_bound_evaluations": 4,
            "canonical_tie_incumbent_updates": 0,
            "canonical_tie_pruned_regions": 2,
            "canonical_tie_singletons_resolved": 0,
            "dual_column_checks": 45,
            "lp_calls": 9,
            "lp_matrix_coefficients": 465,
            "residual_fill_option_probes": 45,
        },
    },
    "cold_canonical_reference": {
        "canonical_exact": True,
        "epsilon": 0.0,
        "gap": 0.0,
        "option_counts": [0, 0, 0, 0, 0, 0, 5, 1, 9],
        "option_indices": [6, 6, 6, 6, 6, 7] + [8] * 9 + [None],
        "score": 215.90705127946384,
        "score_exact": True,
        "upper_bound": 215.90705127946384,
        "work": {
            "branch_splits": 85,
            "candidate_evaluations": 131,
            "canonical_key_bound_evaluations": 386,
            "canonical_tie_incumbent_updates": 5,
            "canonical_tie_pruned_regions": 11,
            "canonical_tie_singletons_resolved": 0,
            "dual_column_checks": 1134,
            "lp_calls": 130,
            "lp_matrix_coefficients": 14622,
            "residual_fill_option_probes": 1134,
        },
    },
    "mismatch_kind": "positive_regret",
    "queue_l1": 80,
    "scaled_queue": [0, 0, 80],
    "value": 0.10303913566677245,
}
SOURCE_LANE_OUTCOMES = {
    "8": {
        "state_count": 512,
        "fixed_lp_budget": 48,
        "candidate_score_exact_state_count": 511,
        "candidate_canonical_exact_state_count": 508,
        "polish_trigger_state_count": 1,
        "exchange_attempt_count": 27,
        "feasible_exchange_count": 24,
        "structural_pair_check_count": 0,
        "active_bulk_candidate_attempt_count": 0,
        "feasible_bulk_candidate_count": 0,
        "accepted_bulk_substitution_count": 0,
        "changed_incumbent_state_count": 0,
        "candidate_action_matches_reference_count": 512,
        "candidate_action_mismatch_count": 0,
        "score_tie_action_mismatch_count": 0,
        "positive_regret_action_mismatch_count": 0,
        "maximum_candidate_epsilon": 0.007927837039167827,
        "maximum_actual_normalized_support_regret": 0.0,
        "support_gate_passed": True,
        "independent_roster_action_identity_gate_passed": True,
        "actual_compute_gate_passed": True,
        "lane_confirmation_gate_passed": True,
        "candidate_work": {
            "branch_splits": 2513,
            "candidate_evaluations": 9551,
            "canonical_key_bound_evaluations": 14447,
            "canonical_tie_incumbent_updates": 1,
            "canonical_tie_pruned_regions": 1153,
            "canonical_tie_singletons_resolved": 0,
            "dual_column_checks": 60974,
            "lp_calls": 9012,
            "lp_matrix_coefficients": 725166,
            "residual_fill_option_probes": 61001,
        },
        "reference_work": {
            "branch_splits": 2533,
            "candidate_evaluations": 9566,
            "canonical_key_bound_evaluations": 14487,
            "canonical_tie_incumbent_updates": 1,
            "canonical_tie_pruned_regions": 1155,
            "canonical_tie_singletons_resolved": 0,
            "dual_column_checks": 61352,
            "lp_calls": 9054,
            "lp_matrix_coefficients": 730080,
            "residual_fill_option_probes": 61352,
        },
        "first_action_mismatch_witness": None,
    },
    "16": {
        "state_count": 512,
        "fixed_lp_budget": 9,
        "candidate_score_exact_state_count": 511,
        "candidate_canonical_exact_state_count": 0,
        "polish_trigger_state_count": 511,
        "exchange_attempt_count": 0,
        "feasible_exchange_count": 0,
        "structural_pair_check_count": 3577,
        "active_bulk_candidate_attempt_count": 186,
        "feasible_bulk_candidate_count": 156,
        "accepted_bulk_substitution_count": 0,
        "changed_incumbent_state_count": 0,
        "candidate_action_matches_reference_count": 511,
        "candidate_action_mismatch_count": 1,
        "score_tie_action_mismatch_count": 0,
        "positive_regret_action_mismatch_count": 1,
        "maximum_candidate_epsilon": 0.10303913566677281,
        "maximum_actual_normalized_support_regret": 0.10303913566677245,
        "support_gate_passed": False,
        "independent_roster_action_identity_gate_passed": False,
        "actual_compute_gate_passed": True,
        "lane_confirmation_gate_passed": False,
        "candidate_work": {
            "branch_splits": 0,
            "candidate_evaluations": 8697,
            "canonical_key_bound_evaluations": 1687,
            "canonical_tie_incumbent_updates": 0,
            "canonical_tie_pruned_regions": 620,
            "canonical_tie_singletons_resolved": 0,
            "dual_column_checks": 23040,
            "lp_calls": 4608,
            "lp_matrix_coefficients": 238080,
            "residual_fill_option_probes": 26617,
        },
        "reference_work": {
            "branch_splits": 2534,
            "candidate_evaluations": 9502,
            "canonical_key_bound_evaluations": 15553,
            "canonical_tie_incumbent_updates": 5,
            "canonical_tie_pruned_regions": 1204,
            "canonical_tie_singletons_resolved": 0,
            "dual_column_checks": 60783,
            "lp_calls": 8990,
            "lp_matrix_coefficients": 722697,
            "residual_fill_option_probes": 60783,
        },
        "first_action_mismatch_witness": SOURCE_R16_WITNESS,
    },
}
SOURCE_AGGREGATE = {
    "candidate": {
        "branch_splits": 2513,
        "candidate_evaluations": 18248,
        "canonical_key_bound_evaluations": 16134,
        "canonical_tie_incumbent_updates": 1,
        "canonical_tie_pruned_regions": 1773,
        "canonical_tie_singletons_resolved": 0,
        "dual_column_checks": 84014,
        "lp_calls": 13620,
        "lp_matrix_coefficients": 963246,
        "residual_fill_option_probes": 87618,
    },
    "reference": {
        "branch_splits": 5067,
        "candidate_evaluations": 19068,
        "canonical_key_bound_evaluations": 30040,
        "canonical_tie_incumbent_updates": 6,
        "canonical_tie_pruned_regions": 2359,
        "canonical_tie_singletons_resolved": 0,
        "dual_column_checks": 122135,
        "lp_calls": 18044,
        "lp_matrix_coefficients": 1452777,
        "residual_fill_option_probes": 122135,
    },
}
SOURCE_R8_EVIDENCE = {
    name: {
        "candidate_action_matches_reference_count": ROSTER_SIZE,
        "support_gate_passed": True,
        "charged_compute_gate_passed": True,
    }
    for name in ROSTER_NAMES
}


class R16SupportRescueDevelopmentRefusal(RuntimeError):
    """The registered post-result support-rescue contract is not satisfied."""


def _json(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise R16SupportRescueDevelopmentRefusal(
            f"cannot read JSON object {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise R16SupportRescueDevelopmentRefusal(
            f"JSON root must be an object: {path}"
        )
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
        RESCUE_SOURCE: (ROOT / RESCUE_SOURCE).read_text(encoding="utf-8"),
        THIS_SOURCE: (ROOT / THIS_SOURCE).read_text(encoding="utf-8"),
    }


def _lane_outcome(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "state_count",
        "fixed_lp_budget",
        "candidate_score_exact_state_count",
        "candidate_canonical_exact_state_count",
        "polish_trigger_state_count",
        "exchange_attempt_count",
        "feasible_exchange_count",
        "structural_pair_check_count",
        "active_bulk_candidate_attempt_count",
        "feasible_bulk_candidate_count",
        "accepted_bulk_substitution_count",
        "changed_incumbent_state_count",
        "candidate_action_matches_reference_count",
        "candidate_action_mismatch_count",
        "score_tie_action_mismatch_count",
        "positive_regret_action_mismatch_count",
        "maximum_candidate_epsilon",
        "maximum_actual_normalized_support_regret",
        "support_gate_passed",
        "independent_roster_action_identity_gate_passed",
        "actual_compute_gate_passed",
        "lane_confirmation_gate_passed",
    )
    return {
        **{key: row.get(key) for key in keys},
        "candidate_work": _json(row.get("candidate_charged_work")),
        "reference_work": _json(row.get("cold_canonical_reference_actual_work")),
        "first_action_mismatch_witness": _json(
            row.get("first_action_mismatch_witness")
        ),
    }


def _roster_contract(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(Path(path).resolve()),
        "seed": payload["seed"],
        "size": ROSTER_SIZE,
        "queues": _json(payload["queues"]),
        "payload": _json(payload),
    }


def _complete_roster_contract(
    frozen: Mapping[str, Any],
) -> dict[str, Any]:
    path = Path(frozen["path"])
    payload = _read_object(path)
    queues = payload.get("queues")
    if (
        payload.get("seed") != frozen.get("seed")
        or queues != frozen.get("queues")
        or not isinstance(queues, list)
        or len(queues) != ROSTER_SIZE
        or len({tuple(row) for row in queues}) != ROSTER_SIZE
    ):
        raise R16SupportRescueDevelopmentRefusal(
            "consumed source roster differs from its frozen contract"
        )
    return _roster_contract(path, payload)


def _validated_source_contract(
    result: Mapping[str, Any],
    protocol: Mapping[str, Any],
    source_roster: Path,
) -> dict[str, Any]:
    try:
        source._validate_protocol(protocol)
    except source.R16BulkTieConfirmationRefusal as exc:
        raise R16SupportRescueDevelopmentRefusal(str(exc)) from exc
    paths = protocol.get("source_paths") or {}
    dev_protocol = _read_object(Path(paths["development_registration"]))
    dev_result = _read_object(Path(paths["development_result"]))
    if (
        _read_object(Path(paths["development_fixed_sources"]))
        != source.development._current_fixed_sources()
    ):
        raise R16SupportRescueDevelopmentRefusal(
            "failed confirmation development source changed"
        )
    try:
        development_contract = source._validated_development_contract(
            dev_result, dev_protocol
        )
    except source.R16BulkTieConfirmationRefusal as exc:
        raise R16SupportRescueDevelopmentRefusal(str(exc)) from exc
    if development_contract != protocol.get("development_contract"):
        raise R16SupportRescueDevelopmentRefusal(
            "failed confirmation embedded development contract differs"
        )

    lane_rows = result.get("lane_summaries") or ()
    by_lane = {str(row.get("lanes")): row for row in lane_rows}
    aggregate = result.get("aggregate_actual_compute") or {}
    observed_aggregate = {
        "candidate": _json(aggregate.get("candidate")),
        "reference": _json(aggregate.get("cold_canonical_reference")),
    }
    if (
        result.get("schema_version") != 1
        or result.get("gate") != source.GATE
        or result.get("experiment") != source.EXPERIMENT
        or result.get("status") != "PASS"
        or result.get("protocol") != protocol
        or result.get("scientific_decision") != SOURCE_DECISION
        or result.get("development_scientific_decision")
        != source.SOURCE_DECISION
        or result.get("independent_static_confirmation_passed") is not False
        or result.get("envelope_normalized_trajectory_registration_authorized")
        is not False
        or result.get("trajectory_registration_prerequisite_passed") is not False
        or result.get("capacity_closure_reclassified") is not False
        or result.get("retained_v5_capacity_established") is not False
        or result.get("retained_v5_closure_passed") is not False
        or result.get("independent_roster_seed") != source.ROSTER_SEED
        or result.get("independent_roster_state_count") != ROSTER_SIZE
        or result.get("trajectory_runs") != 0
        or result.get("arrival_generations") != 0
        or result.get("rng_draws_during_execution") != 0
        or result.get("server_pulls") != 0
        or not result.get("checks")
        or not all(result["checks"].values())
        or set(by_lane) != set(SOURCE_LANE_OUTCOMES)
        or any(
            not row.get("checks") or not all(row["checks"].values())
            for row in by_lane.values()
        )
        or any(
            _lane_outcome(by_lane[lane]) != expected
            for lane, expected in SOURCE_LANE_OUTCOMES.items()
        )
        or observed_aggregate != SOURCE_AGGREGATE
        or aggregate.get("gate_passed") is not True
    ):
        raise R16SupportRescueDevelopmentRefusal(
            "frozen failed support confirmation differs"
        )

    source_roster_payload = _read_object(Path(source_roster))
    try:
        exclusions = source._load_exclusion_rosters(protocol["source_paths"])
        validated_source_queues = source._validate_roster(
            source_roster_payload, protocol, exclusions
        )
    except (KeyError, source.R16BulkTieConfirmationRefusal) as exc:
        raise R16SupportRescueDevelopmentRefusal(str(exc)) from exc
    if (
        source_roster_payload != protocol.get("queue_roster_contract")
        or [list(row) for row in validated_source_queues]
        != source_roster_payload.get("queues")
    ):
        raise R16SupportRescueDevelopmentRefusal(
            "seed280919 source roster changed"
        )

    capacity_rows = development_contract[
        "physical_envelope_capacity_normalized_loads"
    ]["16"]
    observed_limit = min(float(row["effective_slack_eta"]) for row in capacity_rows)
    if (
        observed_limit != SUPPORT_EPSILON_LIMIT
        or not any(
            float(row["load"]) == 0.95
            and float(row["effective_slack_eta"]) == SUPPORT_EPSILON_LIMIT
            and row.get("capacity_projection_passed") is True
            for row in capacity_rows
        )
    ):
        raise R16SupportRescueDevelopmentRefusal(
            "frozen R16 support epsilon limit changed"
        )

    prior_r8 = development_contract.get("source_r8_evidence") or {}
    expected_prior = {
        name: SOURCE_R8_EVIDENCE[name]
        for name in ("seed13041", "seed260919", "seed270919")
    }
    r8 = by_lane["8"]
    if prior_r8 != expected_prior:
        raise R16SupportRescueDevelopmentRefusal(
            "retained prior R8 evidence changed"
        )
    retained_r8 = {
        **_json(prior_r8),
        "seed280919": {
            "candidate_action_matches_reference_count": r8[
                "candidate_action_matches_reference_count"
            ],
            "support_gate_passed": r8["support_gate_passed"],
            "charged_compute_gate_passed": r8["actual_compute_gate_passed"],
        },
    }
    if retained_r8 != SOURCE_R8_EVIDENCE:
        raise R16SupportRescueDevelopmentRefusal(
            "retained four-roster R8 evidence changed"
        )

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
        "seed13041_roster_contract": _complete_roster_contract(
            development_contract["seed13041_roster_contract"]
        ),
        "seed260919_roster_contract": _complete_roster_contract(
            development_contract["seed260919_roster_contract"]
        ),
        "seed270919_roster_contract": _complete_roster_contract(
            development_contract["seed270919_roster_contract"]
        ),
        "seed280919_roster_contract": _roster_contract(
            Path(source_roster), source_roster_payload
        ),
        "retained_r8_evidence": retained_r8,
        "source_lane_outcomes": _json(SOURCE_LANE_OUTCOMES),
        "source_aggregate": _json(SOURCE_AGGREGATE),
        "source_unique_r16_positive_regret_witness": _json(SOURCE_R16_WITNESS),
        "support_epsilon_limit": SUPPORT_EPSILON_LIMIT,
    }


def _load_consumed_rosters(
    contract: Mapping[str, Any],
) -> dict[str, tuple[tuple[int, int, int], ...]]:
    rosters = {}
    for name in ROSTER_NAMES:
        frozen = contract[f"{name}_roster_contract"]
        payload = _read_object(Path(frozen["path"]))
        queues = tuple(
            tuple(int(value) for value in row)
            for row in payload.get("queues") or ()
        )
        if (
            payload.get("seed") != frozen.get("seed")
            or [list(row) for row in queues] != frozen.get("queues")
            or len(queues) != ROSTER_SIZE
            or len(set(queues)) != ROSTER_SIZE
            or (
                frozen.get("payload") is not None
                and payload != frozen["payload"]
            )
        ):
            raise R16SupportRescueDevelopmentRefusal(
                f"consumed roster changed: {name}"
            )
        rosters[name] = queues
    if len(set().union(*(set(rows) for rows in rosters.values()))) != 4 * ROSTER_SIZE:
        raise R16SupportRescueDevelopmentRefusal(
            "the four consumed rosters are not pairwise disjoint"
        )
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
        raise R16SupportRescueDevelopmentRefusal(
            "frozen source confirmation changed"
        )
    contract = _validated_source_contract(result, protocol, Path(source_roster))
    option_count = len(
        contract["physical_complete_frame_envelope"]["model"]["options"]
    )
    if (
        RESCUE_SCORE_TOLERANCE != canonical.SCORE_TOLERANCE
        or RESCUE_BUDGET != R16_LANES * option_count
    ):
        raise R16SupportRescueDevelopmentRefusal(
            "rescue tolerance or fixed option-cardinality budget changed"
        )
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
            "failed_confirmation_registration": str(
                Path(source_registration).resolve()
            ),
            "failed_confirmation_roster": str(Path(source_roster).resolve()),
            "failed_confirmation_fixed_sources": str(
                Path(source_fixed_sources).resolve()
            ),
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
    if any(
        Path(path).exists()
        for path in (registration, fixed_sources, DEFAULT_RESULT)
    ):
        raise R16SupportRescueDevelopmentRefusal(
            "development artifact already exists"
        )
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
        raise R16SupportRescueDevelopmentRefusal(
            "registered R16 support-rescue protocol changed"
        )


def _new_work() -> dict[str, int]:
    return {field: 0 for field in canonical.WORK_FIELDS}


def _add_work(target: dict[str, int], work: Mapping[str, int]) -> None:
    for field in canonical.WORK_FIELDS:
        target[field] += int(work[field])


def _support_rescue_relation_holds(
    result: Any,
    queue: Sequence[int],
    options: Sequence[Any],
) -> bool:
    primary = result.primary_result
    rescue = result.rescue_result
    triggered = (
        SUPPORT_EPSILON_LIMIT - primary.selection.epsilon
        <= RESCUE_SCORE_TOLERANCE
    )
    primary_work = bulk_tie_total_work_row(primary)
    rescue_work = _new_work()
    primary_relation = source._bulk_tie_relation_holds(
        primary, queue, options
    )
    if rescue is not None:
        rescue_work = bulk_tie_total_work_row(rescue)
    expected_work = {
        field: primary_work[field] + rescue_work[field]
        for field in canonical.WORK_FIELDS
    }
    if not triggered:
        return bool(
            result.rescue_triggered is False
            and rescue is None
            and result.selection == primary.selection
            and primary_relation
            and rescue_total_work_row(result) == expected_work
        )
    return bool(
        result.rescue_triggered is True
        and rescue is not None
        and result.selection == rescue.selection
        and primary_relation
        and source._bulk_tie_relation_holds(rescue, queue, options)
        and rescue.selection.score_exact
        and rescue.selection.canonical_exact
        and rescue.selection.score + canonical.SCORE_TOLERANCE
        >= primary.selection.score
        and rescue.selection.upper_bound
        <= primary.selection.upper_bound + canonical.SCORE_TOLERANCE
        and rescue_total_work_row(result) == expected_work
    )


def _screen_roster(
    model: Any,
    base_queues: Sequence[Sequence[int]],
    capacity_rows: Sequence[Mapping[str, Any]],
    roster_name: str,
) -> tuple[dict[str, Any], dict[str, bool]]:
    primary_work = _new_work()
    rescue_work = _new_work()
    candidate_work = _new_work()
    reference_work = _new_work()
    valid_candidates = valid_references = bounded_regrets = action_matches = 0
    rescue_count = exact_rescue_count = method_relations = 0
    primary_bulk_triggers = primary_pair_checks = 0
    rescue_bulk_triggers = rescue_pair_checks = 0
    score_exact = canonical_exact = 0
    max_epsilon = max_regret = -1.0
    epsilon_witness = regret_witness = first_rescue = first_mismatch = None

    for base_queue in base_queues:
        queue = tuple(8 * int(value) for value in base_queue)
        composite = select_canonical_support_rescue_bulk_tie_frame(
            model.options,
            queue,
            R16_LANES,
            PRIMARY_BUDGET,
            RESCUE_BUDGET,
            SUPPORT_EPSILON_LIMIT,
        )
        primary = composite.primary_result
        rescue = composite.rescue_result
        candidate = composite.selection
        relation = _support_rescue_relation_holds(
            composite, queue, model.options
        )
        method_relations += int(relation)
        primary_row = bulk_tie_total_work_row(primary)
        rescue_row = _new_work()
        if rescue is not None:
            rescue_row = bulk_tie_total_work_row(rescue)
        charged = rescue_total_work_row(composite)
        primary_bulk_triggers += int(primary.polish_triggered)
        primary_pair_checks += primary.pair_checks
        rescue_count += int(composite.rescue_triggered)
        if rescue is not None:
            rescue_bulk_triggers += int(rescue.polish_triggered)
            rescue_pair_checks += rescue.pair_checks
            exact_rescue_count += int(
                rescue.selection.score_exact
                and rescue.selection.canonical_exact
            )
            if first_rescue is None:
                first_rescue = {
                    "base_queue": list(base_queue),
                    "scaled_queue": list(queue),
                    "primary_epsilon": primary.selection.epsilon,
                    "primary_score_exact": primary.selection.score_exact,
                    "primary_canonical_exact": primary.selection.canonical_exact,
                    "rescue_epsilon": rescue.selection.epsilon,
                    "rescue_score_exact": rescue.selection.score_exact,
                    "rescue_canonical_exact": rescue.selection.canonical_exact,
                    "primary_work": primary_row,
                    "rescue_work": rescue_row,
                }

        reference = select_canonical_count_space_bnb_frame(
            model.options, queue, R16_LANES, REFERENCE_BUDGET
        )
        candidate_valid = (
            relation
            and canonical._selection_reconstructs(
                model, queue, R16_LANES, primary.base_selection
            )
            and canonical._selection_reconstructs(
                model, queue, R16_LANES, primary.selection
            )
            and canonical._selection_work_holds(
                queue, R16_LANES, PRIMARY_BUDGET, primary.base_selection
            )
            and canonical._certificate_valid(primary.selection, queue)
            and (
                rescue is None
                or (
                    canonical._selection_reconstructs(
                        model, queue, R16_LANES, rescue.base_selection
                    )
                    and canonical._selection_reconstructs(
                        model, queue, R16_LANES, rescue.selection
                    )
                    and canonical._selection_work_holds(
                        queue,
                        R16_LANES,
                        RESCUE_BUDGET,
                        rescue.base_selection,
                    )
                    and canonical._certificate_valid(rescue.selection, queue)
                )
            )
            and canonical._certificate_valid(candidate, queue)
        )
        reference_valid = (
            canonical._selection_reconstructs(
                model, queue, R16_LANES, reference
            )
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
        if not action_match and first_mismatch is None:
            first_mismatch = canonical._selection_witness(
                base_queue, queue, candidate, reference, regret
            )
        _add_work(primary_work, primary_row)
        _add_work(rescue_work, rescue_row)
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
    expected_charged_work = {
        field: primary_work[field] + rescue_work[field]
        for field in canonical.WORK_FIELDS
    }
    checks = {
        "state_count_is_512": len(base_queues) == ROSTER_SIZE,
        "every_candidate_method_relation_and_certificate_is_valid": (
            valid_candidates == ROSTER_SIZE
            and method_relations == ROSTER_SIZE
        ),
        "every_reference_is_valid_score_exact_and_canonical_exact": (
            valid_references == ROSTER_SIZE
        ),
        "every_actual_regret_is_nonnegative_and_bounded_by_epsilon": (
            bounded_regrets == ROSTER_SIZE
        ),
        "every_triggered_rescue_is_score_exact_and_canonical_exact": (
            exact_rescue_count == rescue_count
        ),
        "candidate_work_is_exactly_primary_plus_executed_rescue": (
            candidate_work == expected_charged_work
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
        "primary_lp_budget": PRIMARY_BUDGET,
        "rescue_lp_budget": RESCUE_BUDGET,
        "support_epsilon_limit": SUPPORT_EPSILON_LIMIT,
        "cold_reference_budget": REFERENCE_BUDGET,
        "primary_bulk_trigger_state_count": primary_bulk_triggers,
        "primary_structural_pair_check_count": primary_pair_checks,
        "rescue_trigger_state_count": rescue_count,
        "exact_rescue_state_count": exact_rescue_count,
        "rescue_bulk_trigger_state_count": rescue_bulk_triggers,
        "rescue_structural_pair_check_count": rescue_pair_checks,
        "candidate_score_exact_state_count": score_exact,
        "candidate_canonical_exact_state_count": canonical_exact,
        "candidate_action_matches_reference_count": action_matches,
        "candidate_action_mismatch_count": ROSTER_SIZE - action_matches,
        "first_rescue_witness": first_rescue,
        "first_action_mismatch_witness": first_mismatch,
        "maximum_candidate_epsilon": max_epsilon,
        "maximum_actual_normalized_support_regret": max_regret,
        "epsilon_witness": epsilon_witness,
        "support_regret_witness": regret_witness,
        "support_by_load": support_rows,
        "primary_charged_work": primary_work,
        "rescue_charged_work": rescue_work,
        "candidate_total_charged_work": candidate_work,
        "cold_reference_actual_work": reference_work,
        "actual_compute_comparison": comparisons,
        "support_gate_passed": support_passed,
        "action_identity_gate_passed": action_passed,
        "charged_compute_gate_passed": compute_passed,
        "development_gate_passed": (
            all(checks.values())
            and support_passed
            and action_passed
            and compute_passed
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
        raise R16SupportRescueDevelopmentRefusal(
            f"development result exists: {output}"
        )
    protocol = _read_object(Path(registration))
    _validate_protocol(protocol)
    if _read_object(Path(fixed_sources)) != _current_fixed_sources():
        raise R16SupportRescueDevelopmentRefusal(
            "registered development sources changed"
        )
    paths = protocol["source_paths"]
    source_protocol = _read_object(
        Path(paths["failed_confirmation_registration"])
    )
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
        raise R16SupportRescueDevelopmentRefusal(
            "failed confirmation changed after registration"
        )
    rosters = _load_consumed_rosters(protocol["source_contract"])
    if {
        name: [list(row) for row in rows] for name, rows in rosters.items()
    } != protocol["consumed_rosters"]:
        raise R16SupportRescueDevelopmentRefusal(
            "consumed roster snapshot changed"
        )
    contract = protocol["source_contract"]
    model = source.previous.development.source.capacity.base._model_from_snapshot(
        contract["physical_complete_frame_envelope"]["model"]
    )
    capacity_rows = contract["physical_envelope_capacity_normalized_loads"]["16"]

    started = time.perf_counter()
    rows = []
    row_checks = []
    for name in ROSTER_NAMES:
        row, checks = _screen_roster(
            model, rosters[name], capacity_rows, name
        )
        rows.append(row)
        row_checks.append(checks)

    primary_total = _new_work()
    rescue_total = _new_work()
    candidate_total = _new_work()
    reference_total = _new_work()
    for row in rows:
        _add_work(primary_total, row["primary_charged_work"])
        _add_work(rescue_total, row["rescue_charged_work"])
        _add_work(candidate_total, row["candidate_total_charged_work"])
        _add_work(reference_total, row["cold_reference_actual_work"])
    aggregate_comparisons = source.previous.development._work_comparison(
        candidate_total, reference_total
    )
    aggregate_compute_passed = all(
        row["strictly_lower"] for row in aggregate_comparisons.values()
    )
    retained_r8 = contract["retained_r8_evidence"]
    retained_r8_passed = (
        retained_r8 == SOURCE_R8_EVIDENCE
        and all(
            row["candidate_action_matches_reference_count"] == ROSTER_SIZE
            and row["support_gate_passed"]
            and row["charged_compute_gate_passed"]
            for row in retained_r8.values()
        )
    )
    checks = {
        "failed_confirmation_is_valid_and_unchanged": True,
        "four_consumed_rosters_are_exact_disjoint_and_unchanged": (
            len(set().union(*(set(rows_) for rows_ in rosters.values())))
            == 4 * ROSTER_SIZE
        ),
        "every_r16_execution_check_passes": all(
            all(check.values()) for check in row_checks
        ),
        "retained_r8_source_evidence_passes": retained_r8_passed,
        "r16_primary_and_reference_call_counts_are_exact": sum(
            row["state_count"] for row in rows
        ) == 4 * ROSTER_SIZE,
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
            "primary": primary_total,
            "rescue": rescue_total,
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
            else run_registered(
                args.registration, args.fixed_sources, args.output
            )
        )
    except R16SupportRescueDevelopmentRefusal as exc:
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
