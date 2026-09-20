"""Post-result scaling screen for residual-fill seeds and a fractional bound.

This development experiment follows the fixed R8/R16 action-regret witnesses.
It is not an independent preregistration.  Registration freezes sources and
constants without constructing the grid, solving an LP, or calling a selector.
Execution first audits the repaired fractional upper bound against all 100
ordered two-lane actions, then evaluates fixed total budgets on the existing
1,000-state scaling grid without enumerating the R8/R16 Cartesian families.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
from itertools import product
import json
import math
from pathlib import Path
import time

from algorithm.budgeted_frame_candidates import select_budgeted_frame
from algorithm.experiments.aligned_hardware_frame_model import (
    capacity_certificate,
    load_model,
)
from algorithm.residual_fill_frame_candidates import (
    certified_fractional_frame_upper,
    select_residual_fill_frame,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
DEFAULT_OUTPUT = ARTIFACTS / (
    "aligned_hardware_residual_fill_scaling_screen_development_20260912.json"
)
SCALING_RESULT = ARTIFACTS / (
    "aligned_hardware_cartesian_scaling_development_20260912.json"
)
SCALING_PROTOCOL = ARTIFACTS / (
    "aligned_hardware_cartesian_scaling_development_20260912_protocol.json"
)
VETO_RESULT = ARTIFACTS / (
    "aligned_hardware_scaling_action_regret_veto_development_20260912.json"
)
VETO_PROTOCOL = ARTIFACTS / (
    "aligned_hardware_scaling_action_regret_veto_development_20260912_protocol.json"
)
VETO_FIXED_SOURCES = ARTIFACTS / (
    "aligned_hardware_scaling_action_regret_veto_development_20260912_fixed_sources.json"
)

EXPERIMENT = "aligned_hardware_residual_fill_scaling_screen_v1"
SCALING_EXPERIMENT = "aligned_hardware_cartesian_scaling_v1"
SCALING_DECISION = "COMPUTE_SAVING_WITH_LOOSE_CERTIFICATE_STOP"
VETO_EXPERIMENT = "aligned_hardware_scaling_action_regret_veto_v1"
VETO_DECISION = "ACTION_REGRET_VETO_CERTIFICATE_ONLY_STOP"
THIS_SOURCE = (
    "algorithm/experiments/aligned_hardware_residual_fill_scaling_screen.py"
)
SELECTOR_SOURCE = "algorithm/residual_fill_frame_candidates.py"

BASE_CNN_LEVELS = (0, 1, 2, 3, 4, 5, 8, 16, 24, 40)
BASE_JAX_LEVELS = (0, 1, 2, 3, 4, 5, 8, 12, 18, 25)
BASE_LLM_LEVELS = (0, 1, 2, 3, 4, 5, 8, 12, 18, 24)
BASE_STATE_COUNT = 1000
LANE_COUNTS = (8, 16)
BUDGETS = (4096, 8192)
LOADS = (0.50, 0.80, 0.95)
OPTION_COUNT = 9
LOCAL_CHOICE_COUNT = 10
R2_ORDERED_ACTION_COUNT = 100
SCORE_TOLERANCE = 1e-10
WORK_FRACTION_LIMIT = 0.05

DECISION_RULE = {
    "INVALID_EXECUTION": "one or more frozen execution checks fail",
    "RESIDUAL_FILL_SCALING_ADVANCE": (
        "R8 and R16 each have a fixed budget that improves the same-budget "
        "legacy alpha, preserves every eta margin, and keeps candidate and "
        "fully charged logical work below five percent of 10^R"
    ),
    "CERTIFIED_IMPROVEMENT_WITH_HIGH_LOAD_FAIL_STOP": (
        "both lane counts improve over legacy but at least one has no fixed "
        "budget preserving the 0.95-load eta margin"
    ),
    "NO_CERTIFIED_SCALING_IMPROVEMENT_STOP": (
        "the repaired certificate does not improve both scaled lane counts"
    ),
}

CLAIM_BOUNDARY = (
    "A post-result deterministic development screen on the fixed synthetic "
    "R8/R16 1,000-state grid. The R2 audit covers the 100 ordered two-lane "
    "actions only. It is not an independent confirmation, physical multi-GPU "
    "result, trajectory result, all-queue theorem, or deployment decision."
)
PROHIBITED = (
    "R8/R16 Cartesian enumeration, trajectory or arrival simulation, RNG, "
    "server data, confirmation claims, paper edits, and post-result changes to "
    "the grid, budgets, eta gate, pair order, LP relaxation, or work threshold"
)


def _json(value):
    return json.loads(json.dumps(value))


def _paths(output=DEFAULT_OUTPUT):
    output = Path(output)
    return (
        output,
        output.with_name(output.stem + "_protocol.json"),
        output.with_name(output.stem + "_fixed_sources.json"),
    )


def _write_new(path, payload, *, compact=False):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(
            payload,
            stream,
            indent=None if compact else 2,
            separators=(",", ":") if compact else None,
        )
        stream.write("\n")


def _current_fixed_sources():
    upstream = json.loads(VETO_FIXED_SOURCES.read_text())
    changed = [
        name
        for name, text in upstream.items()
        if (ROOT / name).read_text() != text
    ]
    if changed:
        raise ValueError(f"frozen veto source changed: {changed}")
    return {
        **upstream,
        SELECTOR_SOURCE: (ROOT / SELECTOR_SOURCE).read_text(),
        THIS_SOURCE: (ROOT / THIS_SOURCE).read_text(),
    }


def _source_contract():
    scaling = json.loads(SCALING_RESULT.read_text())
    scaling_protocol = json.loads(SCALING_PROTOCOL.read_text())
    veto = json.loads(VETO_RESULT.read_text())
    veto_protocol = json.loads(VETO_PROTOCOL.read_text())
    if scaling.get("protocol") != scaling_protocol:
        raise ValueError("scaling result does not embed its protocol")
    if veto.get("protocol") != veto_protocol:
        raise ValueError("veto result does not embed its protocol")
    if (
        scaling.get("experiment") != SCALING_EXPERIMENT
        or scaling.get("status") != "PASS"
        or scaling.get("scientific_decision") != SCALING_DECISION
        or not all(scaling.get("checks", {}).values())
        or veto.get("experiment") != VETO_EXPERIMENT
        or veto.get("status") != "PASS"
        or veto.get("scientific_decision") != VETO_DECISION
        or not all(veto.get("checks", {}).values())
        or veto.get("post_result_diagnostic") is not True
    ):
        raise ValueError("residual-fill screen requires passing scaling and veto sources")
    k4096_source_rows = {}
    k4096_alphas = {}
    for lanes in LANE_COUNTS:
        source_row = _source_budget_row(scaling, lanes, 4096)
        k4096_source_rows[str(lanes)] = _json(source_row)
        k4096_alphas[str(lanes)] = source_row[
            "max_certified_normalized_gap"
        ]
    return {
        "scaling_result": SCALING_RESULT.name,
        "scaling_experiment": scaling["experiment"],
        "scaling_decision": scaling["scientific_decision"],
        "scaling_k4096_source_rows_by_lane": k4096_source_rows,
        "scaling_k4096_alpha_by_lane": k4096_alphas,
        "veto_result": VETO_RESULT.name,
        "veto_experiment": veto["experiment"],
        "veto_decision": veto["scientific_decision"],
    }, scaling_protocol["model"]


def _protocol_constants(model, source_contract):
    return {
        "experiment": EXPERIMENT,
        "role": "post-result development scaling screen",
        "post_result_development_screen": True,
        "independent_preregistration": False,
        "source_contract": source_contract,
        "model": _json(model.snapshot()),
        "base_levels": {
            "cnn": list(BASE_CNN_LEVELS),
            "jax": list(BASE_JAX_LEVELS),
            "llm": list(BASE_LLM_LEVELS),
        },
        "base_state_count": BASE_STATE_COUNT,
        "lane_counts": list(LANE_COUNTS),
        "queue_scale_by_lane": {
            str(lanes): lanes // 2 for lanes in LANE_COUNTS
        },
        "total_candidate_budgets": list(BUDGETS),
        "loads": list(LOADS),
        "option_count": OPTION_COUNT,
        "local_choice_count_including_idle": LOCAL_CHOICE_COUNT,
        "raw_ordered_action_count_by_lane": {
            str(lanes): LOCAL_CHOICE_COUNT**lanes for lanes in LANE_COUNTS
        },
        "r2_oracle_audit": {
            "states": BASE_STATE_COUNT,
            "ordered_actions_per_state": R2_ORDERED_ACTION_COUNT,
            "complete_enumeration": True,
            "purpose": (
                "detect any under-bound before the fractional certificate is "
                "used on non-enumerated R8/R16 action spaces"
            ),
        },
        "expected_calls": {
            "r2_fractional_upper": BASE_STATE_COUNT,
            "r2_oracle_action_evaluations": (
                BASE_STATE_COUNT * R2_ORDERED_ACTION_COUNT
            ),
            "residual_fill_selector": (
                BASE_STATE_COUNT * len(LANE_COUNTS) * len(BUDGETS)
            ),
            "fresh_legacy_k8192_selector": (
                BASE_STATE_COUNT * len(LANE_COUNTS)
            ),
            "capacity_certificate": len(LANE_COUNTS) * len(LOADS),
        },
        "pair_rule": {
            "order": (
                "ascending original anchor index, ascending distinct fill index, "
                "ascending anchor copies"
            ),
            "candidate": (
                "for each ordered pair and anchor count, use the maximum feasible "
                "fill count under residual lanes and visible queue"
            ),
            "deduplication": (
                "deduplicate by option-count vector before a candidate attempt"
            ),
            "budget_allocation": (
                "reserve ten legacy seeds, charge each new residual-fill multiset "
                "once, and pass all remaining candidate budget to the unchanged "
                "legacy selector"
            ),
        },
        "certificate_rule": (
            "at each option-duration threshold solve the lane-and-queue fractional "
            "count relaxation; repair proposed dual prices with exact Fractions, "
            "outward-round the dual objective, then use max(best score, "
            "min(legacy upper, repaired fractional upper))"
        ),
        "work_rule": (
            "candidate attempts include residual-fill candidates after deduplication "
            "plus legacy attempts; fully charged work additionally includes every "
            "generation probe, zero-fill and duplicate-producing probe through the "
            "probe total, coordinate/division/key work, and all LP counts, variables, "
            "constraint rows, coefficient cells, and objective terms"
        ),
        "score_tolerance": SCORE_TOLERANCE,
        "work_fraction_limit": WORK_FRACTION_LIMIT,
        "advance_gate": {
            "execution_checks": True,
            "same_budget_alpha_improvement": True,
            "all_load_residuals_positive": True,
            "candidate_attempt_fraction_below": WORK_FRACTION_LIMIT,
            "charged_work_fraction_below": WORK_FRACTION_LIMIT,
            "required_lane_counts": list(LANE_COUNTS),
        },
        "decision_rule": DECISION_RULE,
        "claim_boundary": CLAIM_BOUNDARY,
        "prohibited": PROHIBITED,
        "timing_is_scientific_gate": False,
    }


def register(output=DEFAULT_OUTPUT):
    """Freeze sources and rules without constructing a grid or running methods."""
    output, protocol_path, fixed_sources_path = _paths(output)
    if any(path.exists() for path in (output, protocol_path, fixed_sources_path)):
        raise ValueError("residual-fill screen artifacts already exist")
    fixed_sources = _current_fixed_sources()
    source_contract, source_model = _source_contract()
    model = load_model()
    if _json(model.snapshot()) != source_model:
        raise ValueError("current aligned model differs from the scaling source")
    protocol = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        **_protocol_constants(model, source_contract),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_new(protocol_path, protocol, compact=True)
    _write_new(fixed_sources_path, fixed_sources)
    return protocol


def _base_queues():
    return tuple(product(BASE_CNN_LEVELS, BASE_JAX_LEVELS, BASE_LLM_LEVELS))


def _score_action(model, queue, lanes, indices):
    if len(indices) != lanes:
        raise ValueError("one option or idle is required per lane")
    service = tuple(
        sum(
            model.options[index].requirements[coordinate]
            for index in indices
            if index is not None
        )
        for coordinate in range(len(queue))
    )
    if any(done > waiting for done, waiting in zip(service, queue, strict=True)):
        return None
    duration = max((
        1.0,
        *(
            model.options[index].duration
            for index in indices
            if index is not None
        ),
    ))
    numerator = sum(
        waiting * done for waiting, done in zip(queue, service, strict=True)
    )
    return numerator / duration


def _r2_oracle(model, queue):
    choices = tuple(range(len(model.options))) + (None,)
    best = 0.0
    evaluations = 0
    for indices in product(choices, repeat=2):
        evaluations += 1
        score = _score_action(model, queue, 2, indices)
        if score is not None:
            best = max(best, score)
    return best, evaluations


def _selection_reconstructs(model, queue, lanes, selection):
    score = _score_action(model, queue, lanes, selection.option_indices)
    if score is None:
        return False
    service = tuple(
        sum(
            model.options[index].requirements[coordinate]
            for index in selection.option_indices
            if index is not None
        )
        for coordinate in range(len(queue))
    )
    duration = max((
        1.0,
        *(
            model.options[index].duration
            for index in selection.option_indices
            if index is not None
        ),
    ))
    return (
        tuple(selection.service) == service
        and math.isclose(selection.duration, duration, rel_tol=0.0, abs_tol=SCORE_TOLERANCE)
        and math.isclose(selection.score, score, rel_tol=0.0, abs_tol=SCORE_TOLERANCE)
        and math.isclose(
            selection.gap,
            max(0.0, selection.upper_bound - selection.score),
            rel_tol=0.0,
            abs_tol=SCORE_TOLERANCE,
        )
    )


def _source_budget_row(scaling, lanes, budget):
    lane = next(row for row in scaling["lane_summaries"] if row["lanes"] == lanes)
    return next(
        row for row in lane["budget_summaries"] if row["budget"] == budget
    )


def _new_accumulator(budget):
    return {
        "budget": budget,
        "state_count": 0,
        "nonempty_state_count": 0,
        "alpha": -1.0,
        "alpha_witness": None,
        "exact_certificate_state_count": 0,
        "positive_gap_state_count": 0,
        "candidate_attempts": 0,
        "charged_logical_work": 0,
        "pair_candidate_attempts": 0,
        "pair_generation_probes": 0,
        "zero_fill_probes": 0,
        "duplicate_multiset_probes": 0,
        "threshold_lp_count": 0,
        "lp_variables": 0,
        "lp_constraint_rows": 0,
        "lp_variable_constraint_cells": 0,
        "pair_beats_reduced_budget_legacy_state_count": 0,
        "score_gain_over_reduced_budget_legacy_sum": 0.0,
    }


def _update(accumulator, base_queue, queue, selection):
    accumulator["state_count"] += 1
    accumulator["exact_certificate_state_count"] += int(
        selection.gap <= SCORE_TOLERANCE
    )
    accumulator["positive_gap_state_count"] += int(
        selection.gap > SCORE_TOLERANCE
    )
    accumulator["candidate_attempts"] += selection.attempts
    accumulator["charged_logical_work"] += selection.charged_logical_work
    accumulator["pair_candidate_attempts"] += selection.pair_candidate_attempts
    accumulator["pair_generation_probes"] += selection.pair_generation_probes
    accumulator["zero_fill_probes"] += selection.zero_fill_probes
    accumulator["duplicate_multiset_probes"] += selection.duplicate_multiset_probes
    accumulator["threshold_lp_count"] += selection.threshold_lp_count
    accumulator["lp_variables"] += selection.lp_total_variables
    accumulator["lp_constraint_rows"] += selection.lp_total_constraint_rows
    accumulator["lp_variable_constraint_cells"] += (
        selection.lp_variable_constraint_cells
    )
    gain = selection.score - selection.legacy_score
    accumulator["pair_beats_reduced_budget_legacy_state_count"] += int(
        gain > SCORE_TOLERANCE
    )
    accumulator["score_gain_over_reduced_budget_legacy_sum"] += gain
    norm = sum(queue)
    if not norm:
        return
    accumulator["nonempty_state_count"] += 1
    epsilon = selection.gap / norm
    if epsilon > accumulator["alpha"]:
        accumulator["alpha"] = epsilon
        accumulator["alpha_witness"] = {
            "base_queue": list(base_queue),
            "scaled_queue": list(queue),
            "queue_l1": norm,
            "score": selection.score,
            "upper_bound": selection.upper_bound,
            "gap": selection.gap,
            "value": epsilon,
            "option_indices": list(selection.option_indices),
            "legacy_score": selection.legacy_score,
            "legacy_upper_bound": selection.legacy_upper_bound,
            "fractional_upper_bound": selection.fractional_upper_bound,
            "pair_candidate_attempts": selection.pair_candidate_attempts,
            "legacy_budget": selection.legacy_budget,
        }


def _finalize(accumulator, lanes, baseline_alpha, capacity_rows):
    state_count = accumulator["state_count"]
    raw_count = LOCAL_CHOICE_COUNT**lanes
    alpha = max(0.0, accumulator.pop("alpha"))
    accumulator["max_certified_normalized_gap"] = alpha
    accumulator["exact_certificate_domain_fraction"] = (
        accumulator["exact_certificate_state_count"] / state_count
    )
    accumulator["mean_candidate_attempts"] = (
        accumulator["candidate_attempts"] / state_count
    )
    accumulator["mean_charged_logical_work"] = (
        accumulator["charged_logical_work"] / state_count
    )
    accumulator["mean_pair_candidate_attempts"] = (
        accumulator["pair_candidate_attempts"] / state_count
    )
    accumulator["mean_pair_generation_probes"] = (
        accumulator["pair_generation_probes"] / state_count
    )
    accumulator["candidate_attempt_fraction_of_raw_enumeration"] = (
        accumulator["candidate_attempts"] / (state_count * raw_count)
    )
    accumulator["charged_work_fraction_of_raw_enumeration"] = (
        accumulator["charged_logical_work"] / (state_count * raw_count)
    )
    accumulator["baseline_legacy_alpha"] = baseline_alpha
    accumulator["alpha_improvement_over_legacy"] = baseline_alpha - alpha
    accumulator["same_budget_alpha_improved"] = (
        alpha < baseline_alpha - SCORE_TOLERANCE
    )
    accumulator["mean_score_gain_over_reduced_budget_legacy"] = (
        accumulator.pop("score_gain_over_reduced_budget_legacy_sum") / state_count
    )
    accumulator["capacity_bridge"] = [
        {
            "load": row["load"],
            "effective_slack_eta": row["effective_slack_eta"],
            "finite_grid_alpha": alpha,
            "residual_margin": row["effective_slack_eta"] - alpha,
            "residual_margin_positive": (
                row["effective_slack_eta"] - alpha > SCORE_TOLERANCE
            ),
        }
        for row in capacity_rows
    ]
    accumulator["capacity_margin_passed"] = all(
        row["residual_margin_positive"]
        for row in accumulator["capacity_bridge"]
    )
    accumulator["computation_passed"] = (
        accumulator["candidate_attempt_fraction_of_raw_enumeration"]
        < WORK_FRACTION_LIMIT
        and accumulator["charged_work_fraction_of_raw_enumeration"]
        < WORK_FRACTION_LIMIT
    )
    accumulator["tradeoff_passed"] = (
        accumulator["same_budget_alpha_improved"]
        and accumulator["capacity_margin_passed"]
        and accumulator["computation_passed"]
    )
    return accumulator


def _capacity_row(certificate):
    return {
        "load": certificate["load_factor"],
        "effective_slack_eta": certificate["effective_slack_eta"],
        "capacity_projection_passed": certificate["capacity_projection_passed"],
        "conditional_model_drift_certified": certificate[
            "conditional_model_drift_certified"
        ],
        "physical_service_confidence_increased": certificate[
            "physical_service_confidence_increased"
        ],
    }


def _decision(execution_passed, lane_summaries):
    if not execution_passed:
        return "INVALID_EXECUTION"
    if all(
        any(row["tradeoff_passed"] for row in lane["budget_summaries"])
        for lane in lane_summaries
    ):
        return "RESIDUAL_FILL_SCALING_ADVANCE"
    if all(
        any(row["same_budget_alpha_improved"] for row in lane["budget_summaries"])
        for lane in lane_summaries
    ) and any(
        not any(row["capacity_margin_passed"] for row in lane["budget_summaries"])
        for lane in lane_summaries
    ):
        return "CERTIFIED_IMPROVEMENT_WITH_HIGH_LOAD_FAIL_STOP"
    return "NO_CERTIFIED_SCALING_IMPROVEMENT_STOP"


def run_registered(output=DEFAULT_OUTPUT):
    """Execute the frozen R2 audit and R8/R16 development screen once."""
    output, protocol_path, fixed_sources_path = _paths(output)
    if output.exists():
        raise ValueError("residual-fill screen result already exists")
    protocol = json.loads(protocol_path.read_text())
    if json.loads(fixed_sources_path.read_text()) != _current_fixed_sources():
        raise ValueError("registered residual-fill sources changed")
    source_contract, source_model = _source_contract()
    model = load_model()
    if _json(model.snapshot()) != source_model:
        raise ValueError("current aligned model differs from the scaling source")
    expected = _protocol_constants(model, source_contract)
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise ValueError("registered residual-fill protocol changed")

    started = time.perf_counter()
    base = _base_queues()
    checks = {
        "base_grid_is_complete_unique_and_ordered": (
            len(base) == BASE_STATE_COUNT
            and len(set(base)) == BASE_STATE_COUNT
            and base == tuple(sorted(base))
            and base[0] == (0, 0, 0)
            and base[-1]
            == (
                BASE_CNN_LEVELS[-1],
                BASE_JAX_LEVELS[-1],
                BASE_LLM_LEVELS[-1],
            )
        ),
        "r2_fractional_duals_are_exactly_feasible": True,
        "r2_fractional_upper_never_underbounds_oracle": True,
        "r2_oracle_evaluates_exactly_100_actions_per_state": True,
        "r2_lp_work_is_complete": True,
        "scaled_grids_are_exact_unique_and_ordered": True,
        "capacity_certificates_are_valid": True,
        "selections_reconstruct": True,
        "returned_gap_certificates_reconstruct": True,
        "total_candidate_budgets_hold": True,
        "pair_family_is_exhausted_at_both_budgets": True,
        "lp_work_is_fully_reported": True,
        "budget_curve_is_monotone": True,
        "legacy_k4096_source_is_reused": True,
        "legacy_k8192_call_count_is_exact": True,
        "capacity_call_count_is_exact": True,
        "residual_fill_selector_call_count_is_exact": True,
        "no_scaled_cartesian_oracle_enumeration": True,
    }

    r2_started = time.perf_counter()
    r2_min_slack = math.inf
    r2_min_witness = None
    r2_exact_count = 0
    r2_underbound_count = 0
    r2_oracle_evaluations = 0
    r2_fractional_calls = 0
    r2_lp_work = {
        "threshold_lp_count": 0,
        "total_variables": 0,
        "total_constraint_rows": 0,
        "variable_constraint_cells": 0,
        "objective_terms": 0,
        "repaired_threshold_count": 0,
    }
    for queue in base:
        oracle_score, evaluations = _r2_oracle(model, queue)
        certificate = certified_fractional_frame_upper(model.options, queue, 2)
        r2_fractional_calls += 1
        r2_oracle_evaluations += evaluations
        checks["r2_oracle_evaluates_exactly_100_actions_per_state"] &= (
            evaluations == R2_ORDERED_ACTION_COUNT
        )
        checks["r2_fractional_duals_are_exactly_feasible"] &= (
            certificate.exact_dual_feasible
        )
        slack = certificate.upper_bound - oracle_score
        checks["r2_fractional_upper_never_underbounds_oracle"] &= (
            slack >= -SCORE_TOLERANCE
        )
        r2_underbound_count += int(slack < -SCORE_TOLERANCE)
        r2_exact_count += int(abs(slack) <= SCORE_TOLERANCE)
        if slack < r2_min_slack:
            r2_min_slack = slack
            r2_min_witness = {
                "queue": list(queue),
                "oracle_score": oracle_score,
                "fractional_upper_bound": certificate.upper_bound,
                "slack": slack,
            }
        for key in r2_lp_work:
            source_key = (
                "threshold_lp_count" if key == "threshold_lp_count" else key
            )
            r2_lp_work[key] += getattr(certificate, source_key)
        checks["r2_lp_work_is_complete"] &= (
            certificate.threshold_lp_count == OPTION_COUNT
            and certificate.total_variables == 45
            and certificate.total_constraint_rows == 36
            and certificate.variable_constraint_cells == 180
            and certificate.objective_terms == 45
        )
    r2_audit = {
        "state_count": len(base),
        "ordered_actions_per_state": R2_ORDERED_ACTION_COUNT,
        "oracle_action_evaluations": r2_oracle_evaluations,
        "underbound_violation_count": r2_underbound_count,
        "minimum_slack": r2_min_slack,
        "minimum_slack_witness": r2_min_witness,
        "exact_bound_state_count": r2_exact_count,
        "exact_bound_domain_fraction": r2_exact_count / len(base),
        "lp_work": r2_lp_work,
        "wall_s": time.perf_counter() - r2_started,
    }

    capacity_by_lane = {}
    capacity_call_count = 0
    for lanes in LANE_COUNTS:
        rows = []
        for load in LOADS:
            row = _capacity_row(
                capacity_certificate(replace(model, lanes=lanes), load)
            )
            rows.append(row)
            capacity_call_count += 1
            checks["capacity_certificates_are_valid"] &= (
                row["capacity_projection_passed"]
                and row["conditional_model_drift_certified"]
                and row["effective_slack_eta"] > 0.0
                and row["physical_service_confidence_increased"] is False
            )
        capacity_by_lane[str(lanes)] = rows

    accumulators = {
        lanes: {budget: _new_accumulator(budget) for budget in BUDGETS}
        for lanes in LANE_COUNTS
    }
    legacy_8192 = {
        lanes: {"alpha": -1.0, "witness": None, "selector_calls": 0}
        for lanes in LANE_COUNTS
    }
    selector_calls = 0
    legacy_selector_calls = 0
    lane_wall = {}
    for lanes in LANE_COUNTS:
        lane_started = time.perf_counter()
        factor = lanes // 2
        queues = tuple(
            tuple(factor * coordinate for coordinate in queue) for queue in base
        )
        checks["scaled_grids_are_exact_unique_and_ordered"] &= (
            len(queues) == BASE_STATE_COUNT
            and len(set(queues)) == BASE_STATE_COUNT
            and queues == tuple(sorted(queues))
        )
        for base_queue, queue in zip(base, queues, strict=True):
            selections = []
            for budget in BUDGETS:
                selection = select_residual_fill_frame(
                    model.options, queue, lanes, budget
                )
                selector_calls += 1
                selections.append(selection)
                checks["selections_reconstruct"] &= _selection_reconstructs(
                    model, queue, lanes, selection
                )
                checks["returned_gap_certificates_reconstruct"] &= (
                    selection.score <= selection.upper_bound + SCORE_TOLERANCE
                    and math.isclose(
                        selection.gap,
                        max(0.0, selection.upper_bound - selection.score),
                        rel_tol=0.0,
                        abs_tol=SCORE_TOLERANCE,
                    )
                    and selection.lp_exact_dual_feasible
                )
                checks["total_candidate_budgets_hold"] &= (
                    selection.attempts <= budget
                    and selection.attempts
                    == selection.legacy_attempts
                    + selection.pair_candidate_attempts
                    and selection.legacy_budget
                    == budget - selection.pair_candidate_attempts
                )
                checks["pair_family_is_exhausted_at_both_budgets"] &= (
                    selection.pair_family_exhausted
                    and not selection.pair_budget_blocked_unique_candidate
                )
                checks["lp_work_is_fully_reported"] &= (
                    selection.threshold_lp_count > 0
                    and selection.lp_total_variables > 0
                    and selection.lp_total_constraint_rows > 0
                    and selection.lp_variable_constraint_cells
                    == selection.lp_total_variables * (len(queue) + 1)
                    and selection.charged_logical_work >= selection.attempts
                )
                _update(
                    accumulators[lanes][budget],
                    base_queue,
                    queue,
                    selection,
                )
            left, right = selections
            checks["budget_curve_is_monotone"] &= (
                right.score + SCORE_TOLERANCE >= left.score
                and right.upper_bound <= left.upper_bound + SCORE_TOLERANCE
                and right.gap <= left.gap + SCORE_TOLERANCE
            )

            direct = select_budgeted_frame(model.options, queue, lanes, 8192)
            legacy_selector_calls += 1
            norm = sum(queue)
            epsilon = direct.gap / norm if norm else 0.0
            row = legacy_8192[lanes]
            row["selector_calls"] += 1
            if epsilon > row["alpha"]:
                row["alpha"] = epsilon
                row["witness"] = {
                    "base_queue": list(base_queue),
                    "scaled_queue": list(queue),
                    "value": epsilon,
                    "score": direct.score,
                    "upper_bound": direct.upper_bound,
                    "option_indices": list(direct.option_indices),
                }
        lane_wall[str(lanes)] = time.perf_counter() - lane_started

    lane_summaries = []
    for lanes in LANE_COUNTS:
        source_4096 = source_contract[
            "scaling_k4096_source_rows_by_lane"
        ][str(lanes)]
        source_alpha = source_contract["scaling_k4096_alpha_by_lane"][str(lanes)]
        checks["legacy_k4096_source_is_reused"] &= (
            source_4096["state_count"] == BASE_STATE_COUNT
            and source_4096["budget"] == 4096
            and source_4096["max_certified_normalized_gap"] == source_alpha
        )
        baseline = {
            4096: source_alpha,
            8192: max(0.0, legacy_8192[lanes]["alpha"]),
        }
        budget_rows = [
            _finalize(
                accumulators[lanes][budget],
                lanes,
                baseline[budget],
                capacity_by_lane[str(lanes)],
            )
            for budget in BUDGETS
        ]
        lane_summaries.append({
            "lanes": lanes,
            "queue_scale": lanes // 2,
            "state_count": BASE_STATE_COUNT,
            "raw_ordered_actions_per_state": LOCAL_CHOICE_COUNT**lanes,
            "legacy_baselines": {
                "4096": {
                    "source": SCALING_RESULT.name,
                    "alpha": baseline[4096],
                    "alpha_witness": source_4096["alpha_witness"],
                },
                "8192": legacy_8192[lanes],
            },
            "budget_summaries": budget_rows,
            "alpha_plateau_between_fixed_budgets": math.isclose(
                budget_rows[0]["max_certified_normalized_gap"],
                budget_rows[1]["max_certified_normalized_gap"],
                rel_tol=0.0,
                abs_tol=SCORE_TOLERANCE,
            ),
            "first_tradeoff_budget": next(
                (row["budget"] for row in budget_rows if row["tradeoff_passed"]),
                None,
            ),
        })

    checks["legacy_k8192_call_count_is_exact"] &= (
        legacy_selector_calls == BASE_STATE_COUNT * len(LANE_COUNTS)
        and all(
            row["selector_calls"] == BASE_STATE_COUNT
            for row in legacy_8192.values()
        )
    )
    checks["capacity_call_count_is_exact"] &= (
        capacity_call_count == len(LANE_COUNTS) * len(LOADS)
    )
    checks["residual_fill_selector_call_count_is_exact"] &= (
        selector_calls == BASE_STATE_COUNT * len(LANE_COUNTS) * len(BUDGETS)
    )
    checks["r2_lp_work_is_complete"] &= (
        r2_fractional_calls == BASE_STATE_COUNT
        and r2_lp_work["threshold_lp_count"] == BASE_STATE_COUNT * OPTION_COUNT
        and r2_lp_work["total_variables"] == BASE_STATE_COUNT * 45
        and r2_lp_work["total_constraint_rows"] == BASE_STATE_COUNT * 36
        and r2_lp_work["variable_constraint_cells"] == BASE_STATE_COUNT * 180
        and r2_lp_work["objective_terms"] == BASE_STATE_COUNT * 45
    )
    checks["no_scaled_cartesian_oracle_enumeration"] &= (
        selector_calls == BASE_STATE_COUNT * len(LANE_COUNTS) * len(BUDGETS)
        and r2_oracle_evaluations
        == BASE_STATE_COUNT * R2_ORDERED_ACTION_COUNT
    )
    execution_passed = all(checks.values())
    decision = _decision(execution_passed, lane_summaries)
    result = {
        "experiment": EXPERIMENT,
        "status": "PASS" if execution_passed else "FAIL",
        "checks": checks,
        "protocol": protocol,
        "post_result_development_screen": True,
        "independent_preregistration": False,
        "r2_fractional_upper_oracle_audit": r2_audit,
        "capacity_by_lane": capacity_by_lane,
        "lane_summaries": lane_summaries,
        "work_accounting": {
            "residual_fill_selector_calls": selector_calls,
            "fresh_legacy_k8192_selector_calls": legacy_selector_calls,
            "capacity_certificate_calls": capacity_call_count,
            "r2_fractional_upper_calls": r2_fractional_calls,
            "r2_oracle_action_evaluations": r2_oracle_evaluations,
            "scaled_cartesian_action_enumerations": 0,
            "arrival_tapes": 0,
            "trajectory_frames": 0,
            "rng_initializations": 0,
            "random_draws": 0,
        },
        "scientific_decision": decision,
        "development_advance": decision == "RESIDUAL_FILL_SCALING_ADVANCE",
        "claim_boundary": CLAIM_BOUNDARY,
        "limitations": [
            "The mechanism and fixed budgets were selected after inspecting the R8/R16 scaling witnesses.",
            "The fractional relaxation can retain an integrality gap even when its repaired dual certificate is valid.",
            "Generation probes do not consume candidate K under the stated post-dedup budget, but every probe is included in charged logical work.",
            "The synthetic replicated lanes do not add physical multi-GPU interference.",
        ],
        "next_step": (
            "Preserve the fixed-grid high-load result and close this candidate family; "
            "do not tune it for adoption on the consumed development grid."
            if decision == "CERTIFIED_IMPROVEMENT_WITH_HIGH_LOAD_FAIL_STOP"
            else "Follow the registered decision rule without post-result retuning."
        ),
        "timing": {
            "r2_audit_wall_s": r2_audit["wall_s"],
            "lane_wall_s": lane_wall,
            "total_wall_s": time.perf_counter() - started,
            "timing_is_scientific_gate": False,
        },
    }
    _write_new(output, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--register", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    payload = register(args.output) if args.register else run_registered(args.output)
    print(json.dumps({
        "experiment": payload["experiment"],
        "status": payload.get("status", "REGISTERED"),
        "scientific_decision": payload.get("scientific_decision"),
        "output": str(args.output),
    }, separators=(",", ":"), sort_keys=True))


if __name__ == "__main__":
    main()
