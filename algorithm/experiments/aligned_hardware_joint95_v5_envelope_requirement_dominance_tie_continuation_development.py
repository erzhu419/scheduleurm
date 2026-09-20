"""Post-result development of an eight-option dominance quotient trajectory.

The predecessor independent confirmation is retained as a formal compute FAIL:
its R16 candidate exactly tied the K4096 shadow in all three charged fields.
This development replays the two already observed seed batches and changes only
the candidate catalogue.  A queue-independent proof removes option 1 because
option 0 has identical requirements, strictly shorter effective duration and a
smaller canonical index.  Conditional tie continuation then searches the
eight-option quotient while every shadow and executed reference remains the
full nine-option K4096 selector.

A PASS authorizes only a preregistered confirmation on seeds 12101--12120.
It does not turn the older ``KNOWN_QUOTIENT_ACTION_STOP`` result into flow
evidence and does not authorize a paper claim.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from algorithm.canonical_count_space_bnb_requirement_dominance_frame_candidates import (
    PreparedCanonicalRequirementDominanceCatalogue,
    prepare_canonical_count_space_bnb_requirement_dominance_catalogue,
    select_prepared_canonical_count_space_bnb_requirement_dominance_score_exact_tie_continuation_frame,
)
from algorithm.experiments import (
    aligned_hardware_joint95_v5_envelope_tie_continuation_confirmation
    as confirmation,
)


development = confirmation.development
raw = confirmation.raw
ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
STEM = (
    "aligned_hardware_joint95_v5_envelope_"
    "requirement_dominance_tie_continuation_development_20260920"
)
DEFAULT_REGISTRATION = ARTIFACTS / f"{STEM}_protocol.json"
DEFAULT_FIXED_SOURCES = ARTIFACTS / f"{STEM}_fixed_sources.json"
DEFAULT_RESULT = ARTIFACTS / f"{STEM}.json"
DEFAULT_CONFIRMATION_REGISTRATION = confirmation.DEFAULT_REGISTRATION
DEFAULT_CONFIRMATION_FIXED_SOURCES = confirmation.DEFAULT_FIXED_SOURCES
DEFAULT_CONFIRMATION_RESULT = confirmation.DEFAULT_RESULT

EXPERIMENT = (
    "aligned_hardware_joint95_v5_envelope_"
    "requirement_dominance_tie_continuation_development_v1"
)
GATE = (
    "aligned_hardware_joint95_v5_envelope_"
    "requirement_dominance_tie_continuation_development_runner"
)
REGISTRATION_STATUS = (
    "REGISTERED_AWAITING_REQUIREMENT_DOMINANCE_TIE_CONTINUATION_DEVELOPMENT"
)
PASS_DECISION = "REQUIREMENT_DOMINANCE_TIE_CONTINUATION_DEVELOPMENT_PASS"
INVALID_DECISION = "INVALID_REQUIREMENT_DOMINANCE_TIE_CONTINUATION_DEVELOPMENT"
SUPPORT_FAIL_DECISION = (
    "REQUIREMENT_DOMINANCE_TIE_CONTINUATION_DEVELOPMENT_SUPPORT_FAIL_STOP"
)
ACTION_FAIL_DECISION = (
    "REQUIREMENT_DOMINANCE_TIE_CONTINUATION_DEVELOPMENT_ACTION_OR_REGRET_FAIL_STOP"
)
COMPUTE_FAIL_DECISION = (
    "REQUIREMENT_DOMINANCE_TIE_CONTINUATION_DEVELOPMENT_COMPUTE_FAIL_STOP"
)
FLOW_FAIL_DECISION = (
    "REQUIREMENT_DOMINANCE_TIE_CONTINUATION_DEVELOPMENT_FLOW_NONINFERIORITY_FAIL_STOP"
)

LANES = confirmation.LANES
LOADS = confirmation.LOADS
ARRIVAL_SEEDS = tuple(range(12061, 12101))
NEXT_CONFIRMATION_SEEDS = tuple(range(12101, 12121))
ARRIVAL_HORIZON_S = confirmation.ARRIVAL_HORIZON_S
PRIMARY_BUDGETS = confirmation.PRIMARY_BUDGETS
CONTINUATION_BUDGETS = confirmation.CONTINUATION_BUDGETS
REFERENCE_BUDGET = confirmation.REFERENCE_BUDGET
WORK_FIELDS = confirmation.WORK_FIELDS
WORK_GATE_FIELDS = confirmation.WORK_GATE_FIELDS
SCORE_TOLERANCE = development.SCORE_TOLERANCE
EXECUTION_WORKERS = confirmation.EXECUTION_WORKERS
ORIGINAL_OPTION_COUNT = 9
QUOTIENT_OPTION_COUNT = 8
RETAINED_ORIGINAL_INDICES = (0, 2, 3, 4, 5, 6, 7, 8)
REMOVED_ORIGINAL_INDICES = (1,)
THIS_SOURCE = (
    "algorithm/experiments/"
    "aligned_hardware_joint95_v5_envelope_"
    "requirement_dominance_tie_continuation_development.py"
)
REDUCER_SOURCE = "algorithm/requirement_dominance_frame_candidates.py"
QUOTIENT_SOURCE = (
    "algorithm/canonical_count_space_bnb_"
    "requirement_dominance_frame_candidates.py"
)


class RequirementDominanceDevelopmentRefusal(RuntimeError):
    pass


def _json(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RequirementDominanceDevelopmentRefusal(
            f"expected JSON object: {path}"
        )
    return payload


def _write_new(path: Path, payload: Mapping[str, Any], *, compact: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=None if compact else 2, sort_keys=True)
        handle.write("\n")


def _source_paths(
    confirmation_registration: Path = DEFAULT_CONFIRMATION_REGISTRATION,
    confirmation_fixed_sources: Path = DEFAULT_CONFIRMATION_FIXED_SOURCES,
    confirmation_result: Path = DEFAULT_CONFIRMATION_RESULT,
) -> dict[str, str]:
    return {
        "confirmation_registration": str(Path(confirmation_registration).resolve()),
        "confirmation_fixed_sources": str(Path(confirmation_fixed_sources).resolve()),
        "confirmation_result": str(Path(confirmation_result).resolve()),
    }


def _current_fixed_sources(
    confirmation_fixed_sources: Path = DEFAULT_CONFIRMATION_FIXED_SOURCES,
) -> dict[str, str]:
    rows = _read_object(Path(confirmation_fixed_sources))
    for relative in (REDUCER_SOURCE, QUOTIENT_SOURCE, THIS_SOURCE):
        if relative in rows:
            raise RequirementDominanceDevelopmentRefusal(
                "successor source already exists in predecessor bundle"
            )
        rows[relative] = (ROOT / relative).read_text(encoding="utf-8")
    return dict(sorted(rows.items()))


def _validated_confirmation_bundle(
    paths: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    if set(paths) != {
        "confirmation_registration",
        "confirmation_fixed_sources",
        "confirmation_result",
    }:
        raise RequirementDominanceDevelopmentRefusal("confirmation paths changed")
    protocol = _read_object(Path(paths["confirmation_registration"]))
    frozen = _read_object(Path(paths["confirmation_fixed_sources"]))
    result = _read_object(Path(paths["confirmation_result"]))

    source_paths = protocol.get("source_paths") or {}
    try:
        expected = confirmation.build_registration(
            development_registration=Path(source_paths["development_registration"]),
            development_fixed_sources=Path(source_paths["development_fixed_sources"]),
            development_result=Path(source_paths["development_result"]),
        )
        confirmation._validate_protocol(protocol, expected)
        expected_frozen = confirmation._current_fixed_sources(
            Path(source_paths["development_fixed_sources"])
        )
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        raise RequirementDominanceDevelopmentRefusal(
            f"confirmation lineage is invalid: {exc}"
        ) from exc

    normal_keys = sorted(
        key
        for key in frozen
        if not key.startswith(confirmation.HISTORICAL_SOURCE_PREFIX)
    )
    drift = [
        key
        for key in normal_keys
        if not (ROOT / key).is_file()
        or (ROOT / key).read_text(encoding="utf-8") != frozen[key]
    ]
    if (
        len(frozen) != 62
        or len(normal_keys) != 32
        or frozen != expected_frozen
        or drift
        or confirmation.THIS_SOURCE not in frozen
    ):
        raise RequirementDominanceDevelopmentRefusal(
            "confirmation 62-source bundle or current normal sources changed"
        )

    checks = result.get("checks") or {}
    cases = result.get("case_summaries") or ()
    work_gates = result.get("work_gates") or {}
    lane_rows = {int(row["lanes"]): row for row in work_gates.get("by_lane") or ()}
    expected_run_keys = [
        (lanes, load, seed)
        for lanes in LANES
        for load in LOADS
        for seed in confirmation.ARRIVAL_SEEDS
    ]
    observed_run_keys = [
        (int(row["lanes"]), float(row["load"]), int(row["seed"]))
        for row in result.get("runs") or ()
    ]
    r16_fields = (
        (lane_rows.get(16) or {}).get("comparison", {}).get("fields", {})
    )
    r8_fields = (
        (lane_rows.get(8) or {}).get("comparison", {}).get("fields", {})
    )
    if (
        result.get("protocol") != protocol
        or result.get("gate") != confirmation.GATE
        or result.get("experiment") != confirmation.EXPERIMENT
        or result.get("status") != "FAIL"
        or result.get("scientific_decision") != confirmation.COMPUTE_FAIL_DECISION
        or result.get("independent_confirmation_passed") is not False
        or result.get("trajectory_transition_passed") is not False
        or result.get("trajectory_or_flow_claim_authorized") is not False
        or result.get("paper_claim_authorized") is not False
        or result.get("trajectory_runs") != 240
        or result.get("arrival_generations") != 120
        or result.get("shadow_trajectory_runs") != 0
        or result.get("server_pulls") != 0
        or result.get("tie_continuation_trigger_count") != 3
        or result.get("tie_continuation_final_exact_count") != 3
        or result.get(
            "all_triggered_confirmation_states_reached_canonical_exact"
        )
        is not True
        or result.get("fixed_protocol_claim_only") is not True
        or not checks
        or not all(checks.values())
        or len(cases) != 6
        or not all(
            row.get("execution_integrity_passed") is True
            and row.get("trajectory_support_gate_passed") is True
            and row.get("same_state_action_and_regret_gate_passed") is True
            and row.get("paired_mean_flow_noninferiority_passed") is True
            and row.get("case_compute_gate_passed") is True
            and row.get("case_gate_passed") is True
            and row.get("seeds") == list(confirmation.ARRIVAL_SEEDS)
            for row in cases
        )
        or observed_run_keys != expected_run_keys
        or work_gates.get("gate_passed") is not False
        or work_gates.get("uniform_case_no_regression_gate_passed") is not True
        or set(lane_rows) != {8, 16}
        or lane_rows[8].get("gate_passed") is not True
        or lane_rows[16].get("gate_passed") is not False
        or not (work_gates.get("aggregate") or {}).get("gate_passed")
        or not all(
            r8_fields[field]["strictly_lower"] is True
            for field in WORK_GATE_FIELDS
        )
        or not all(
            r16_fields[field]["equal"] is True
            and r16_fields[field]["no_greater"] is True
            and r16_fields[field]["strictly_lower"] is False
            for field in WORK_GATE_FIELDS
        )
    ):
        raise RequirementDominanceDevelopmentRefusal(
            "predecessor is not the exact audited R16 compute-only FAIL"
        )

    contract = {
        "experiment": result["experiment"],
        "gate": result["gate"],
        "status": result["status"],
        "scientific_decision": result["scientific_decision"],
        "arrival_seeds": list(confirmation.ARRIVAL_SEEDS),
        "arrival_generations": result["arrival_generations"],
        "trajectory_runs": result["trajectory_runs"],
        "tie_continuation_trigger_count": result["tie_continuation_trigger_count"],
        "tie_continuation_final_exact_count": result[
            "tie_continuation_final_exact_count"
        ],
        "all_six_case_gates_passed": True,
        "aggregate_compute_gate_passed": True,
        "r8_lane_compute_gate_passed": True,
        "r16_lane_compute_gate_passed": False,
        "r16_gate_fields_equal_to_shadow": list(WORK_GATE_FIELDS),
        "fixed_source_keys": sorted(frozen),
        "fixed_source_count": len(frozen),
        "current_normal_source_keys": normal_keys,
        "current_normal_source_drift": [],
    }
    capacity = protocol.get("physical_envelope_capacity_normalized_loads")
    eta = protocol.get("lane_minimum_effective_slack_eta")
    if not isinstance(capacity, dict) or not isinstance(eta, dict):
        raise RequirementDominanceDevelopmentRefusal(
            "confirmation envelope is missing"
        )
    return contract, protocol, _json(capacity), _json(eta)


def _quotient_contract(
    prepared: PreparedCanonicalRequirementDominanceCatalogue,
) -> dict[str, Any]:
    reduction = asdict(prepared.reduction)
    proof = asdict(prepared.certificate_lift_proof)
    if (
        prepared.reduction.original_option_count != ORIGINAL_OPTION_COUNT
        or prepared.reduction.retained_option_count != QUOTIENT_OPTION_COUNT
        or prepared.reduction.removed_option_count != 1
        or prepared.reduction.retained_original_indices
        != RETAINED_ORIGINAL_INDICES
        or prepared.reduction.removed_original_indices
        != REMOVED_ORIGINAL_INDICES
        or prepared.reduction.reduction_option_probes != 9
        or prepared.reduction.reduction_requirement_coordinate_reads != 27
        or prepared.reduction.reduction_duration_comparisons != 10
        or prepared.certificate_lift_proof.removed_to_dominator_original_indices
        != ((1, 0),)
        or not all(
            value
            for key, value in proof.items()
            if key != "removed_to_dominator_original_indices"
        )
    ):
        raise RequirementDominanceDevelopmentRefusal(
            "the registered nine-to-eight dominance quotient changed"
        )
    return {
        "reduction": _json(reduction),
        "full_catalogue_certificate_lift_proof": _json(proof),
        "candidate_bnb_option_count": QUOTIENT_OPTION_COUNT,
        "shadow_and_reference_option_count": ORIGINAL_OPTION_COUNT,
        "reduction_prepared_once_per_execution": True,
        "reduction_work_charged_separately_from_bnb": True,
    }


METHOD_CONTRACT = {
    "name": "requirement_dominance_quotient_score_exact_tie_continuation",
    "candidate_catalogue": "eight retained options",
    "full_catalogue_action_indices_and_count_vector": True,
    "primary_budget_by_lane": {str(k): v for k, v in PRIMARY_BUDGETS.items()},
    "continuation_budget_by_lane": {
        str(k): v for k, v in CONTINUATION_BUDGETS.items()
    },
    "continuation_trigger": "primary score_exact and not canonical_exact",
    "continuation_uses_same_live_frontier": True,
    "same_state_shadow": "full nine-option cold K4096 on every candidate frame",
    "executed_reference": "full nine-option cold K4096",
}

REGISTRATION_SCOPE = {
    "post_result_development": True,
    "confirmation_compute_fail_read": True,
    "catalogue_reduction_and_lift_preparations": 1,
    "rng_initializations": 0,
    "rng_draws": 0,
    "arrival_generations": 0,
    "selector_calls": 0,
    "trajectory_runs": 0,
    "server_pulls": 0,
    "paper_or_readme_edits": 0,
}

CLAIM_BOUNDARY = (
    "A PASS is post-result development evidence on the already observed seeds "
    "12061--12100. It only authorizes preregistration of seeds 12101--12120 "
    "under the fixed eight-option dominance quotient, full nine-option K4096 "
    "shadows and references, and unchanged trajectory and compute gates. It is "
    "not independent confirmation, trajectory-flow evidence, or a paper claim. "
    "The older KNOWN_QUOTIENT_ACTION_STOP remains a valid negative result for "
    "the distinct K26/v4 consumed-state quotient experiment."
)


def _registration_body(
    paths: Mapping[str, str],
    contract: Mapping[str, Any],
    source_protocol: Mapping[str, Any],
    capacity: Mapping[str, Any],
    eta: Mapping[str, Any],
    quotient: Mapping[str, Any],
) -> dict[str, Any]:
    model_snapshot = source_protocol.get("physical_complete_frame_model") or {}
    if (
        len(model_snapshot.get("options") or ()) != ORIGINAL_OPTION_COUNT
        or set(capacity) != {"8", "16"}
        or set(eta) != {"8", "16"}
    ):
        raise RequirementDominanceDevelopmentRefusal(
            "confirmation model or envelope changed"
        )
    return {
        "schema_version": 1,
        "gate": GATE,
        "experiment": EXPERIMENT,
        "status": REGISTRATION_STATUS,
        "registered": True,
        "source_paths": paths,
        "confirmation_compute_fail_contract": contract,
        "physical_complete_frame_model": _json(model_snapshot),
        "physical_envelope_capacity_normalized_loads": capacity,
        "lane_minimum_effective_slack_eta": eta,
        "arrival_seeds": list(ARRIVAL_SEEDS),
        "next_confirmation_seeds": list(NEXT_CONFIRMATION_SEEDS),
        "arrival_horizon_s": ARRIVAL_HORIZON_S,
        "case_seed_tapes": len(LANES) * len(LOADS) * len(ARRIVAL_SEEDS),
        "trajectory_runs": 2 * len(LANES) * len(LOADS) * len(ARRIVAL_SEEDS),
        "method_contract": _json(METHOD_CONTRACT),
        "quotient_contract": quotient,
        "work_fields": list(WORK_FIELDS),
        "work_gate_fields": list(WORK_GATE_FIELDS),
        "scientific_gates": {
            "execution_action_support_regret_certificate_and_flow": (
                "identical to the predecessor confirmation"
            ),
            "case_compute": (
                "candidate <= full-catalogue same-state K4096 shadow in every gate field"
            ),
            "lane_compute": (
                "candidate < full-catalogue same-state K4096 shadow in every gate field"
            ),
            "aggregate_compute": (
                "candidate < full-catalogue same-state K4096 shadow in every gate field"
            ),
        },
        "registration_scope": dict(REGISTRATION_SCOPE),
        "claim_boundary": CLAIM_BOUNDARY,
        "paper_claim_authorized": False,
    }


def build_registration(**kwargs: Any) -> dict[str, Any]:
    paths = _source_paths(**kwargs)
    contract, source_protocol, capacity, eta = _validated_confirmation_bundle(paths)
    model_snapshot = source_protocol.get("physical_complete_frame_model") or {}
    model = raw.old_transition.baseline._model_from_snapshot(model_snapshot)
    if _json(model.snapshot()) != model_snapshot:
        raise RequirementDominanceDevelopmentRefusal(
            "frozen model reconstruction failed during registration"
        )
    prepared = prepare_canonical_count_space_bnb_requirement_dominance_catalogue(
        model.options
    )
    return _registration_body(
        paths,
        contract,
        source_protocol,
        capacity,
        eta,
        _quotient_contract(prepared),
    )


def _validate_protocol(protocol: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    observed = dict(protocol)
    registered_at = observed.pop("registered_at_utc", None)
    if not isinstance(registered_at, str) or observed != expected:
        raise RequirementDominanceDevelopmentRefusal("registered protocol changed")


def register(
    registration: Path = DEFAULT_REGISTRATION,
    fixed_sources: Path = DEFAULT_FIXED_SOURCES,
    result: Path = DEFAULT_RESULT,
    **kwargs: Any,
) -> dict[str, Any]:
    if any(Path(path).exists() for path in (registration, fixed_sources, result)):
        raise RequirementDominanceDevelopmentRefusal(
            "development artifact already exists"
        )
    body = build_registration(**kwargs)
    protocol = {"registered_at_utc": datetime.now(timezone.utc).isoformat(), **body}
    source_fixed = Path(body["source_paths"]["confirmation_fixed_sources"])
    _write_new(Path(fixed_sources), _current_fixed_sources(source_fixed))
    _write_new(Path(registration), protocol, compact=True)
    return protocol


def _quotient_selection_work_holds(
    queue: Sequence[int], lanes: int, budget: int, selection: Any
) -> bool:
    option_count = QUOTIENT_OPTION_COUNT
    dimension = len(queue)
    return bool(
        len(selection.option_counts) == ORIGINAL_OPTION_COUNT
        and selection.option_counts[1] == 0
        and selection.lp_node_budget == budget
        and selection.duration_root_count == option_count
        and selection.duration_root_count <= selection.lp_calls <= budget
        and selection.dual_certified_lp_calls + selection.safe_fallback_lp_calls
        == selection.lp_calls
        and selection.successful_lp_calls <= selection.lp_calls
        and selection.candidate_evaluations
        == selection.lp_calls + selection.singleton_candidate_evaluations + 1
        and selection.residual_fill_option_probes
        <= option_count * selection.lp_calls
        and selection.dual_column_checks <= option_count * selection.lp_calls
        and selection.lp_matrix_coefficients
        <= option_count * (1 + dimension + option_count) * selection.lp_calls
        and selection.inherited_unsolved_regions
        <= selection.frontier_regions_remaining
        and selection.represented_ordered_configurations
        == (option_count + 1) ** lanes
        and selection.represented_count_vectors
        == math.comb(option_count + lanes, lanes)
    )


def _full_catalogue_selection_valid(
    model: Any,
    queue: Sequence[int],
    budget: int,
    selection: Any,
) -> bool:
    canonical = raw.old_transition.confirmation.canonical
    return bool(
        canonical._selection_reconstructs(model, queue, model.lanes, selection)
        and _quotient_selection_work_holds(queue, model.lanes, budget, selection)
        and canonical._certificate_valid(selection, queue)
    )


def _candidate_call(
    prepared: PreparedCanonicalRequirementDominanceCatalogue,
    model: Any,
    queue: Sequence[int],
) -> dict[str, Any]:
    if tuple(prepared.reduced_options) != tuple(
        model.options[index] for index in RETAINED_ORIGINAL_INDICES
    ):
        raise RequirementDominanceDevelopmentRefusal(
            "prepared quotient does not match the full model catalogue"
        )
    lanes = int(model.lanes)
    initial_budget = PRIMARY_BUDGETS[lanes]
    continuation_budget = CONTINUATION_BUDGETS[lanes]
    continued = select_prepared_canonical_count_space_bnb_requirement_dominance_score_exact_tie_continuation_frame(
        prepared,
        queue,
        lanes,
        initial_budget,
        continuation_budget,
    )
    primary, selected = continued.primary_selection, continued.selection
    expected_trigger = bool(primary.score_exact and not primary.canonical_exact)
    lift = continued.certificate_lift
    proof = asdict(lift.proof)
    lift_valid = bool(
        continued.reduction == prepared.reduction
        and lift.proof == prepared.certificate_lift_proof
        and lift.primary_score_exact_lifted is primary.score_exact
        and lift.primary_canonical_exact_lifted is primary.canonical_exact
        and lift.selection_score_exact_lifted is selected.score_exact
        and lift.selection_canonical_exact_lifted is selected.canonical_exact
        and lift.proof.removed_to_dominator_original_indices == ((1, 0),)
        and all(
            value
            for key, value in proof.items()
            if key != "removed_to_dominator_original_indices"
        )
    )
    relation = bool(
        continued.initial_budget == initial_budget
        and continued.continuation_budget == continuation_budget
        and continued.continuation_triggered is expected_trigger
        and (
            (not expected_trigger and selected is primary)
            or (
                expected_trigger
                and selected.lp_node_budget == continuation_budget
                and selected.score + SCORE_TOLERANCE >= primary.score
                and selected.upper_bound
                <= primary.upper_bound + SCORE_TOLERANCE
                and all(
                    getattr(selected, field) >= getattr(primary, field)
                    for field in WORK_FIELDS
                )
            )
        )
    )
    valid = bool(
        relation
        and lift_valid
        and _full_catalogue_selection_valid(
            model, queue, initial_budget, primary
        )
        and _full_catalogue_selection_valid(
            model, queue, selected.lp_node_budget, selected
        )
    )
    return {
        "selection": selected,
        "primary_selection": primary,
        "work": raw._selection_work(selected),
        "valid": valid,
        "method_relation_valid": bool(relation and lift_valid),
        "full_catalogue_lift_valid": lift_valid,
        "continuation_triggered": expected_trigger,
        "initial_budget": initial_budget,
        "continuation_budget": continuation_budget,
    }


def _run_tape(
    task: tuple[
        Any,
        Mapping[str, Any],
        PreparedCanonicalRequirementDominanceCatalogue,
        int,
        float,
        int,
        float,
    ]
) -> dict[str, Any]:
    model, capacity, prepared, lanes, load, seed, horizon = task
    counts = {"calls": 0, "valid_lifts": 0}

    def quotient_candidate(current_model: Any, queue: Sequence[int]) -> dict[str, Any]:
        call = _candidate_call(prepared, current_model, queue)
        counts["calls"] += 1
        counts["valid_lifts"] += int(call["full_catalogue_lift_valid"])
        return call

    previous_candidate = development._candidate_call
    development._candidate_call = quotient_candidate
    try:
        row = development._run_tape(
            (model, capacity, lanes, load, seed, horizon)
        )
    finally:
        development._candidate_call = previous_candidate
    candidate = next(
        item for item in row["policies"] if item["policy"] == "candidate"
    )["metrics"]
    candidate.update(
        {
            "requirement_dominance_quotient_selector_call_count": counts["calls"],
            "full_catalogue_certificate_lift_valid_count": counts["valid_lifts"],
        }
    )
    return row


def _execute_tasks(tasks: Sequence[tuple[Any, ...]]) -> list[dict[str, Any]]:
    with ProcessPoolExecutor(max_workers=EXECUTION_WORKERS) as pool:
        return list(pool.map(_run_tape, tasks))


def _case_summary(
    runs: Sequence[Mapping[str, Any]], lanes: int, load: float, eta: float
) -> dict[str, Any]:
    summary = confirmation._paired_case_summary(
        runs, lanes, load, eta, ARRIVAL_SEEDS
    )
    members = [row for row in runs if row["lanes"] == lanes and row["load"] == load]
    metrics = [
        next(item for item in row["policies"] if item["policy"] == "candidate")[
            "metrics"
        ]
        for row in members
    ]
    summary.pop("candidate_fixed_budget", None)
    summary.update(
        {
            "candidate_method": (
                "eight_option_requirement_dominance_score_exact_tie_continuation"
            ),
            "candidate_bnb_option_count": QUOTIENT_OPTION_COUNT,
            "shadow_and_reference_option_count": ORIGINAL_OPTION_COUNT,
            "primary_budget": PRIMARY_BUDGETS[lanes],
            "continuation_budget": CONTINUATION_BUDGETS[lanes],
            "tie_continuation_trigger_count": sum(
                row["tie_continuation_trigger_count"] for row in metrics
            ),
            "tie_continuation_final_exact_count": sum(
                row["tie_continuation_final_exact_count"] for row in metrics
            ),
            "conditional_selector_relation_valid_count": sum(
                row["conditional_selector_relation_valid_count"] for row in metrics
            ),
            "full_catalogue_certificate_lift_valid_count": sum(
                row["full_catalogue_certificate_lift_valid_count"] for row in metrics
            ),
        }
    )
    return summary


def _decision(
    execution_passed: bool,
    cases: Sequence[Mapping[str, Any]],
    compute_passed: bool,
) -> str:
    if not execution_passed:
        return INVALID_DECISION
    if not all(row["trajectory_support_gate_passed"] for row in cases):
        return SUPPORT_FAIL_DECISION
    if not all(row["same_state_action_and_regret_gate_passed"] for row in cases):
        return ACTION_FAIL_DECISION
    if not compute_passed:
        return COMPUTE_FAIL_DECISION
    if not all(row["paired_mean_flow_noninferiority_passed"] for row in cases):
        return FLOW_FAIL_DECISION
    return PASS_DECISION


def run_registered(
    registration: Path = DEFAULT_REGISTRATION,
    fixed_sources: Path = DEFAULT_FIXED_SOURCES,
    output: Path = DEFAULT_RESULT,
) -> dict[str, Any]:
    if Path(output).exists():
        raise RequirementDominanceDevelopmentRefusal(f"result exists: {output}")
    protocol = _read_object(Path(registration))
    paths = protocol.get("source_paths") or {}
    source_contract, source_protocol, capacity, eta = (
        _validated_confirmation_bundle(paths)
    )
    model = raw.old_transition.baseline._model_from_snapshot(
        protocol["physical_complete_frame_model"]
    )
    if _json(model.snapshot()) != protocol["physical_complete_frame_model"]:
        raise RequirementDominanceDevelopmentRefusal(
            "frozen model reconstruction failed"
        )
    prepared = prepare_canonical_count_space_bnb_requirement_dominance_catalogue(
        model.options
    )
    if _quotient_contract(prepared) != protocol["quotient_contract"]:
        raise RequirementDominanceDevelopmentRefusal(
            "prepared execution quotient differs from registration"
        )

    expected = _registration_body(
        paths,
        source_contract,
        source_protocol,
        capacity,
        eta,
        _quotient_contract(prepared),
    )
    _validate_protocol(protocol, expected)
    if source_contract != protocol["confirmation_compute_fail_contract"]:
        raise RequirementDominanceDevelopmentRefusal(
            "confirmation compute FAIL contract changed"
        )
    registered_sources = _read_object(Path(fixed_sources))
    current_sources = _current_fixed_sources(
        Path(paths["confirmation_fixed_sources"])
    )
    if registered_sources != current_sources:
        raise RequirementDominanceDevelopmentRefusal("registered sources changed")

    capacity = protocol["physical_envelope_capacity_normalized_loads"]
    by_case: dict[tuple[int, float], Mapping[str, Any]] = {}
    for lanes in LANES:
        for row in capacity[str(lanes)]:
            key = (lanes, float(row["load"]))
            if (
                int(row["lanes"]) != lanes
                or row.get("capacity_projection_passed") is not True
                or row.get("conditional_negative_drift_certified") is not True
                or row.get("positive_effective_slack") is not True
            ):
                raise RequirementDominanceDevelopmentRefusal("capacity row changed")
            by_case[key] = row
    if set(by_case) != {(r, load) for r in LANES for load in LOADS}:
        raise RequirementDominanceDevelopmentRefusal("capacity grid changed")
    tasks = [
        (model, by_case[(lanes, load)], prepared, lanes, load, seed, ARRIVAL_HORIZON_S)
        for lanes in LANES
        for load in LOADS
        for seed in ARRIVAL_SEEDS
    ]
    started = time.perf_counter()
    runs = _execute_tasks(tasks)

    post_contract, post_protocol, post_capacity, post_eta = (
        _validated_confirmation_bundle(paths)
    )
    post_expected = _registration_body(
        paths,
        post_contract,
        post_protocol,
        post_capacity,
        post_eta,
        _quotient_contract(prepared),
    )
    _validate_protocol(protocol, post_expected)
    if (
        _read_object(Path(fixed_sources)) != registered_sources
        or registered_sources
        != _current_fixed_sources(Path(paths["confirmation_fixed_sources"]))
    ):
        raise RequirementDominanceDevelopmentRefusal(
            "registered sources changed during execution"
        )

    eta = protocol["lane_minimum_effective_slack_eta"]
    cases = [
        _case_summary(runs, lanes, load, float(eta[str(lanes)]))
        for lanes in LANES
        for load in LOADS
    ]
    work_gates = raw._aggregate_work_gates(runs, cases)
    policy_rows = [policy for run in runs for policy in run["policies"]]
    candidate_rows = [row for row in policy_rows if row["policy"] == "candidate"]
    reference_rows = [
        row for row in policy_rows if row["policy"] == "cold_k4096_reference"
    ]
    candidate_metrics = [row["metrics"] for row in candidate_rows]
    total_candidate_frames = sum(row["frame_count"] for row in candidate_metrics)
    total_reference_frames = sum(
        row["metrics"]["frame_count"] for row in reference_rows
    )
    trigger_count = sum(
        row["tie_continuation_trigger_count"] for row in candidate_metrics
    )
    expected_run_keys = [
        (lanes, load, seed)
        for lanes in LANES
        for load in LOADS
        for seed in ARRIVAL_SEEDS
    ]
    observed_run_keys = [
        (int(row["lanes"]), float(row["load"]), int(row["seed"]))
        for row in runs
    ]
    checks = {
        "formal_confirmation_compute_fail_and_62_sources_revalidated": True,
        "confirmation_current_normal_sources_have_zero_drift": (
            source_contract["current_normal_source_drift"] == []
        ),
        "protocol_and_65_sources_revalidated_after_execution": True,
        "all_240_registered_case_seed_tapes_execute_exactly_once": (
            observed_run_keys == expected_run_keys and len(runs) == 240
        ),
        "executed_trajectory_count_is_480": len(policy_rows) == 480,
        "each_tape_executes_candidate_and_full_reference_once": all(
            [row["policy"] for row in run["policies"]]
            == ["candidate", "cold_k4096_reference"]
            and run["arrival_tape"]["executed_policy_count"] == 2
            for run in runs
        ),
        "one_arrival_generation_per_common_tape": all(
            run["arrival_tape"]["arrival_generation_count"] == 1 for run in runs
        ),
        "same_state_shadow_never_executes_a_trajectory": all(
            run["arrival_tape"]["shadow_trajectory_runs"] == 0 for run in runs
        ),
        "one_prepared_quotient_call_per_candidate_frame": all(
            row["requirement_dominance_quotient_selector_call_count"]
            == row["frame_count"]
            for row in candidate_metrics
        ),
        "every_full_catalogue_certificate_lift_is_valid": all(
            row["full_catalogue_certificate_lift_valid_count"]
            == row["frame_count"]
            for row in candidate_metrics
        ),
        "every_conditional_selector_relation_is_valid": all(
            row["conditional_selector_relation_valid_count"] == row["frame_count"]
            for row in candidate_metrics
        ),
        "trigger_is_exactly_primary_score_exact_noncanonical": all(
            row["tie_continuation_trigger_count"]
            == row["primary_score_exact_noncanonical_count"]
            for row in candidate_metrics
        ),
        "every_candidate_frame_has_one_fresh_full_k4096_shadow": all(
            row["same_state_shadow_calls"]
            == row["same_state_shadow_valid_count"]
            == row["same_state_shadow_exact_count"]
            == row["frame_count"]
            for row in candidate_metrics
        ),
        "every_policy_fully_drains_and_flow_decomposes": all(
            row["flow_decomposition"]["status"] == "PASS" for row in policy_rows
        ),
        "every_case_contains_the_registered_40_seed_roster": all(
            row["seeds"] == list(ARRIVAL_SEEDS) and row["seed_count"] == 40
            for row in cases
        ),
        "every_case_execution_integrity_and_certificate_gate_passes": all(
            row["execution_integrity_passed"] for row in cases
        ),
        "no_capacity_recomputation_reclassification_or_server_pull": True,
    }
    execution_passed = all(checks.values())
    decision = _decision(execution_passed, cases, work_gates["gate_passed"])
    status = (
        "PASS"
        if decision == PASS_DECISION
        else ("INVALID" if decision == INVALID_DECISION else "FAIL")
    )
    passed = decision == PASS_DECISION
    report = {
        "schema_version": 1,
        "gate": GATE,
        "experiment": EXPERIMENT,
        "status": status,
        "scientific_decision": decision,
        "protocol": protocol,
        "confirmation_compute_fail_contract": source_contract,
        "checks": checks,
        "case_summaries": cases,
        "runs": runs,
        "work_gates": work_gates,
        "tie_continuation_trigger_count": trigger_count,
        "tie_continuation_final_exact_count": sum(
            row["tie_continuation_final_exact_count"] for row in candidate_metrics
        ),
        "all_triggered_development_states_reached_canonical_exact": trigger_count
        == sum(
            row["tie_continuation_final_exact_count"] for row in candidate_metrics
        ),
        "catalogue_preprocessing_work": protocol["quotient_contract"]["reduction"],
        "work_accounting": {
            "arrival_tapes": len(runs),
            "arrival_generations": len(runs),
            "executed_trajectory_runs": len(policy_rows),
            "candidate_trajectory_runs": len(candidate_rows),
            "cold_full_catalogue_reference_trajectory_runs": len(reference_rows),
            "shadow_trajectory_runs": 0,
            "quotient_candidate_selector_calls": total_candidate_frames,
            "same_state_full_catalogue_shadow_selector_calls": total_candidate_frames,
            "executed_full_catalogue_reference_selector_calls": total_reference_frames,
            "cold_restart_rescue_calls": 0,
            "server_pulls": 0,
        },
        "development_replay_passed": passed,
        "new_seed_independent_confirmation_authorized": passed,
        "authorized_confirmation_seeds": (
            list(NEXT_CONFIRMATION_SEEDS) if passed else []
        ),
        "independent_confirmation_eligible": False,
        "trajectory_or_flow_claim_authorized": False,
        "paper_claim_authorized": False,
        "trajectory_runs": len(policy_rows),
        "arrival_generations": len(runs),
        "shadow_trajectory_runs": 0,
        "server_pulls": 0,
        "claim_boundary": CLAIM_BOUNDARY,
        "timing": {"wall_s": time.perf_counter() - started},
    }
    _write_new(Path(output), report, compact=True)
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
            register(args.registration, args.fixed_sources, args.output)
            if args.register
            else run_registered(args.registration, args.fixed_sources, args.output)
        )
    except RequirementDominanceDevelopmentRefusal as exc:
        raise SystemExit(f"REFUSED: {exc}") from exc
    print(
        json.dumps(
            {
                "status": payload["status"],
                "experiment": payload["experiment"],
                "scientific_decision": payload.get("scientific_decision"),
                "tie_continuation_trigger_count": payload.get(
                    "tie_continuation_trigger_count"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
