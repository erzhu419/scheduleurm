"""Post-result scaling screen for count-space branch-and-bound frame search.

The experiment keeps the aligned nine-option model and the existing 1,000-state
scaling grid fixed.  Identical lane permutations are represented by integer
option counts.  One cumulative branch-and-bound trace is read at four fixed LP
budgets.  Only the two-lane control enumerates its 100 ordered raw actions; the
larger action spaces are assessed through the selector's certified upper bound.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from itertools import product
import json
import math
from pathlib import Path
import time

from algorithm.count_space_bnb_frame_candidates import (
    trace_count_space_bnb_frame,
)
from algorithm.experiments.aligned_hardware_frame_model import (
    capacity_certificate,
    load_model,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
DEFAULT_OUTPUT = ARTIFACTS / (
    "aligned_hardware_count_space_bnb_scaling_screen_development_20260913.json"
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

EXPERIMENT = "aligned_hardware_count_space_bnb_scaling_screen_v1"
SCALING_EXPERIMENT = "aligned_hardware_cartesian_scaling_v1"
SCALING_DECISION = "COMPUTE_SAVING_WITH_LOOSE_CERTIFICATE_STOP"
VETO_EXPERIMENT = "aligned_hardware_scaling_action_regret_veto_v1"
VETO_DECISION = "ACTION_REGRET_VETO_CERTIFICATE_ONLY_STOP"
THIS_SOURCE = (
    "algorithm/experiments/aligned_hardware_count_space_bnb_scaling_screen.py"
)
SELECTOR_SOURCE = "algorithm/count_space_bnb_frame_candidates.py"

BASE_CNN_LEVELS = (0, 1, 2, 3, 4, 5, 8, 16, 24, 40)
BASE_JAX_LEVELS = (0, 1, 2, 3, 4, 5, 8, 12, 18, 25)
BASE_LLM_LEVELS = (0, 1, 2, 3, 4, 5, 8, 12, 18, 24)
BASE_STATE_COUNT = 1000
LANE_COUNTS = (2, 4, 8, 16)
LP_CHECKPOINT_BUDGETS = (9, 18, 36, 72)
LOADS = (0.50, 0.80, 0.95)
OPTION_COUNT = 9
LOCAL_CHOICE_COUNT = OPTION_COUNT + 1
R2_ORDERED_ACTION_COUNT = LOCAL_CHOICE_COUNT**2
SCORE_TOLERANCE = 1e-10
WORK_FRACTION_LIMIT = 0.05

DECISION_RULE = {
    "INVALID_EXECUTION": "one or more frozen execution checks fail",
    "COUNT_SPACE_BNB_SCALING_ADVANCE": (
        "R4, R8, and R16 each have a fixed LP checkpoint whose alpha preserves "
        "all three lane-capacity eta margins and whose LP-call, LP-matrix, and "
        "candidate-evaluation fractions are each below five percent of 10^R"
    ),
    "CERTIFIED_WITH_HIGH_MATRIX_WORK_STOP": (
        "all scaled lane counts have a capacity-preserving checkpoint but at "
        "least one lacks a checkpoint passing every frozen work fraction"
    ),
    "COMPUTE_SAVING_WITH_LOOSE_CERTIFICATE_STOP": (
        "all scaled lane counts have a work-efficient checkpoint but at least "
        "one lacks a capacity-preserving checkpoint"
    ),
    "NO_COUNT_SPACE_SCALING_TRADEOFF_STOP": (
        "neither a common certificate conclusion nor a common work conclusion holds"
    ),
}

CLAIM_BOUNDARY = (
    "A post-result deterministic development screen on the fixed synthetic "
    "1,000-state scaling grid for identical replicas of the aligned nine-option "
    "lane model. The R2 oracle audit covers its 100 ordered actions. This is not "
    "an independent confirmation, physical multi-GPU measurement, trajectory "
    "result, all-queue theorem, or deployment decision."
)
PROHIBITED = (
    "R4/R8/R16 ordered-action enumeration, count-vector enumeration, arrivals, "
    "trajectories, RNG, server data, paper edits, and post-result changes to the "
    "grid, lane counts, LP checkpoints, capacity bridge, or work gate"
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
        raise ValueError(f"frozen action-regret source changed: {changed}")
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
        raise ValueError("action-regret result does not embed its protocol")
    expected_levels = {
        "cnn": list(BASE_CNN_LEVELS),
        "jax": list(BASE_JAX_LEVELS),
        "llm": list(BASE_LLM_LEVELS),
    }
    if (
        scaling.get("experiment") != SCALING_EXPERIMENT
        or scaling.get("status") != "PASS"
        or scaling.get("scientific_decision") != SCALING_DECISION
        or not all(scaling.get("checks", {}).values())
        or scaling_protocol.get("base_levels") != expected_levels
        or scaling_protocol.get("base_state_count") != BASE_STATE_COUNT
        or scaling_protocol.get("lane_counts") != list(LANE_COUNTS)
        or scaling_protocol.get("option_count") != OPTION_COUNT
        or veto.get("experiment") != VETO_EXPERIMENT
        or veto.get("status") != "PASS"
        or veto.get("scientific_decision") != VETO_DECISION
        or veto.get("post_result_diagnostic") is not True
        or not all(veto.get("checks", {}).values())
    ):
        raise ValueError("count-space screen requires the passing scaling and veto sources")
    return ({
        "scaling_result": SCALING_RESULT.name,
        "scaling_experiment": scaling["experiment"],
        "scaling_decision": scaling["scientific_decision"],
        "action_regret_result": VETO_RESULT.name,
        "action_regret_experiment": veto["experiment"],
        "action_regret_decision": veto["scientific_decision"],
    }, scaling_protocol["model"])


def _raw_action_counts():
    return {str(lanes): LOCAL_CHOICE_COUNT**lanes for lanes in LANE_COUNTS}


def _protocol_constants(model_snapshot, source_contract):
    return {
        "experiment": EXPERIMENT,
        "role": "post-result count-space branch-and-bound scaling screen",
        "post_result_development_screen": True,
        "independent_preregistration": False,
        "source_contract": source_contract,
        "model": _json(model_snapshot),
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
        "lp_checkpoint_budgets": list(LP_CHECKPOINT_BUDGETS),
        "checkpoint_execution": (
            "one cumulative B&B trace per state/lane; roots and nodes are not "
            "repaid between checkpoint reads"
        ),
        "loads": list(LOADS),
        "option_count": OPTION_COUNT,
        "distinct_duration_root_count": OPTION_COUNT,
        "local_choice_count_including_idle": LOCAL_CHOICE_COUNT,
        "raw_ordered_action_count_by_lane": _raw_action_counts(),
        "r2_oracle_audit": {
            "states": BASE_STATE_COUNT,
            "ordered_actions_per_state": R2_ORDERED_ACTION_COUNT,
            "complete_enumeration": True,
            "checkpoint_inequality": "selected score <= exact optimum <= returned upper",
        },
        "expected_calls": {
            "count_space_traces": BASE_STATE_COUNT * len(LANE_COUNTS),
            "checkpoint_reads": (
                BASE_STATE_COUNT * len(LANE_COUNTS) * len(LP_CHECKPOINT_BUDGETS)
            ),
            "capacity_certificate": len(LANE_COUNTS) * len(LOADS),
            "r2_oracle_action_evaluations": (
                BASE_STATE_COUNT * R2_ORDERED_ACTION_COUNT
            ),
            "scaled_ordered_action_evaluations": 0,
        },
        "certificate_rule": (
            "stratify by exact maximum effective duration; bound every unresolved "
            "integer-count box by an exactly checked, outward-rounded repaired LP "
            "dual or its inherited analytic upper bound"
        ),
        "candidate_rule": (
            "floor the node LP primal within its count box, then greedily fill "
            "remaining lanes and visible queue in original option order"
        ),
        "work_rule": {
            "denominator": "1,000 states times 10^R ordered actions per state",
            "reported_separately": [
                "LP calls",
                "LP matrix coefficient cells",
                "candidate evaluations",
                "residual-fill option probes",
                "dual column checks",
            ],
            "gate_components": [
                "LP-call fraction",
                "LP-matrix coefficient fraction",
                "candidate-evaluation fraction",
            ],
            "fraction_limit": WORK_FRACTION_LIMIT,
        },
        "advance_gate": {
            "execution_checks": True,
            "required_lane_counts": [4, 8, 16],
            "capacity": "one checkpoint has eta(load)-alpha>1e-10 at every load",
            "work": "the same checkpoint passes every frozen work fraction",
        },
        "decision_rule": DECISION_RULE,
        "claim_boundary": CLAIM_BOUNDARY,
        "prohibited": PROHIBITED,
        "timing_is_scientific_gate": False,
    }


def register(output=DEFAULT_OUTPUT):
    """Freeze sources and constants without constructing the grid or doing work."""
    output, protocol_path, fixed_sources_path = _paths(output)
    if any(path.exists() for path in (output, protocol_path, fixed_sources_path)):
        raise ValueError("count-space scaling artifacts already exist")
    fixed_sources = _current_fixed_sources()
    source_contract, source_model = _source_contract()
    protocol = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        **_protocol_constants(source_model, source_contract),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_new(protocol_path, protocol, compact=True)
    _write_new(fixed_sources_path, fixed_sources)
    return protocol


def base_queues():
    return tuple(product(BASE_CNN_LEVELS, BASE_JAX_LEVELS, BASE_LLM_LEVELS))


def scaled_queues(lanes):
    if lanes not in LANE_COUNTS:
        raise ValueError(f"lanes must be one of {LANE_COUNTS}")
    factor = lanes // 2
    return tuple(
        tuple(factor * coordinate for coordinate in queue)
        for queue in base_queues()
    )


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
        *(model.options[index].duration for index in indices if index is not None),
    ))
    return sum(
        waiting * done for waiting, done in zip(queue, service, strict=True)
    ) / duration


def _r2_oracle(model, queue):
    choices = (None, *range(len(model.options)))
    optimum = 0.0
    evaluations = 0
    for indices in product(choices, repeat=2):
        evaluations += 1
        score = _score_action(model, queue, 2, indices)
        if score is not None:
            optimum = max(optimum, score)
    return optimum, evaluations


def _selection_reconstructs(model, queue, lanes, selection):
    reconstructed_counts = tuple(
        selection.option_indices.count(index) for index in range(len(model.options))
    )
    score = _score_action(model, queue, lanes, selection.option_indices)
    service = tuple(
        sum(
            selection.option_counts[index]
            * model.options[index].requirements[coordinate]
            for index in range(len(model.options))
        )
        for coordinate in range(len(queue))
    )
    duration = max((
        1.0,
        *(
            model.options[index].duration
            for index, count in enumerate(selection.option_counts)
            if count
        ),
    ))
    return (
        score is not None
        and selection.option_counts == reconstructed_counts
        and tuple(index for index in selection.option_indices if index is not None)
        == tuple(sorted(index for index in selection.option_indices if index is not None))
        and selection.service == service
        and math.isclose(selection.duration, duration, rel_tol=0.0, abs_tol=SCORE_TOLERANCE)
        and math.isclose(selection.score, score, rel_tol=0.0, abs_tol=SCORE_TOLERANCE)
    )


def _selection_work_holds(queue, lanes, budget, selection):
    option_count = OPTION_COUNT
    dimension = len(queue)
    return (
        selection.lp_node_budget == budget
        and selection.duration_root_count == OPTION_COUNT
        and selection.duration_root_count <= selection.lp_calls <= budget
        and selection.dual_certified_lp_calls + selection.safe_fallback_lp_calls
        == selection.lp_calls
        and selection.successful_lp_calls <= selection.lp_calls
        and selection.candidate_evaluations
        == selection.lp_calls + selection.singleton_candidate_evaluations + 1
        and selection.residual_fill_option_probes <= option_count * selection.lp_calls
        and selection.dual_column_checks <= option_count * selection.lp_calls
        and selection.lp_matrix_coefficients
        <= option_count * (1 + dimension + option_count) * selection.lp_calls
        and selection.inherited_unsolved_regions
        <= selection.frontier_regions_remaining
        and selection.represented_ordered_configurations
        == LOCAL_CHOICE_COUNT**lanes
        and selection.represented_count_vectors
        == math.comb(OPTION_COUNT + lanes, lanes)
    )


def _new_accumulator(budget):
    return {
        "lp_node_budget": budget,
        "state_count": 0,
        "nonempty_state_count": 0,
        "alpha": -1.0,
        "alpha_witness": None,
        "normalized_gap_sum": 0.0,
        "exact_certificate_state_count": 0,
        "lp_calls": 0,
        "successful_lp_calls": 0,
        "dual_certified_lp_calls": 0,
        "safe_fallback_lp_calls": 0,
        "dual_repair_count": 0,
        "candidate_evaluations": 0,
        "singleton_candidate_evaluations": 0,
        "residual_fill_option_probes": 0,
        "branch_splits": 0,
        "lp_matrix_coefficients": 0,
        "dual_column_checks": 0,
        "frontier_regions_remaining": 0,
        "inherited_unsolved_regions": 0,
        "frontier_peak_sum": 0,
    }


def _update(accumulator, base_queue, queue, selection):
    accumulator["state_count"] += 1
    accumulator["exact_certificate_state_count"] += int(selection.exact)
    if sum(queue):
        accumulator["nonempty_state_count"] += 1
        accumulator["normalized_gap_sum"] += selection.epsilon
        if selection.epsilon > accumulator["alpha"]:
            accumulator["alpha"] = selection.epsilon
            accumulator["alpha_witness"] = {
                "base_queue": list(base_queue),
                "scaled_queue": list(queue),
                "queue_l1": sum(queue),
                "value": selection.epsilon,
                "score": selection.score,
                "upper_bound": selection.upper_bound,
                "gap": selection.gap,
                "exact": selection.exact,
                "option_counts": list(selection.option_counts),
                "option_indices": list(selection.option_indices),
            }
    for field in (
        "lp_calls",
        "successful_lp_calls",
        "dual_certified_lp_calls",
        "safe_fallback_lp_calls",
        "dual_repair_count",
        "candidate_evaluations",
        "singleton_candidate_evaluations",
        "residual_fill_option_probes",
        "branch_splits",
        "lp_matrix_coefficients",
        "dual_column_checks",
        "frontier_regions_remaining",
        "inherited_unsolved_regions",
    ):
        accumulator[field] += getattr(selection, field)
    accumulator["frontier_peak_sum"] += selection.frontier_peak


def _capacity_row(certificate):
    return {
        "load": certificate["load_factor"],
        "effective_slack_eta": certificate["effective_slack_eta"],
        "support_slack_delta": certificate["support_slack_delta"],
        "projection_support_loss_rate": certificate[
            "projection_support_loss_rate"
        ],
        "capacity_projection_passed": certificate["capacity_projection_passed"],
        "conditional_model_drift_certified": certificate[
            "conditional_model_drift_certified"
        ],
        "physical_service_confidence_increased": certificate[
            "physical_service_confidence_increased"
        ],
    }


def _finalize(accumulator, lanes, capacity_rows):
    state_count = accumulator["state_count"]
    nonempty = accumulator["nonempty_state_count"]
    raw_per_state = LOCAL_CHOICE_COUNT**lanes
    denominator = state_count * raw_per_state
    alpha = max(0.0, accumulator["alpha"])
    fractions = {
        "lp_call_fraction_of_raw_enumeration": (
            accumulator["lp_calls"] / denominator
        ),
        "lp_matrix_coefficient_fraction_of_raw_enumeration": (
            accumulator["lp_matrix_coefficients"] / denominator
        ),
        "candidate_evaluation_fraction_of_raw_enumeration": (
            accumulator["candidate_evaluations"] / denominator
        ),
        "residual_fill_probe_fraction_of_raw_enumeration": (
            accumulator["residual_fill_option_probes"] / denominator
        ),
        "dual_column_check_fraction_of_raw_enumeration": (
            accumulator["dual_column_checks"] / denominator
        ),
    }
    capacity_bridge = []
    for row in capacity_rows:
        residual = row["effective_slack_eta"] - alpha
        capacity_bridge.append({
            "load": row["load"],
            "effective_slack_eta": row["effective_slack_eta"],
            "finite_grid_alpha": alpha,
            "residual_margin": residual,
            "residual_margin_positive": residual > SCORE_TOLERANCE,
        })
    capacity_passed = all(
        row["residual_margin_positive"] for row in capacity_bridge
    )
    work_passed = all(
        fractions[field] < WORK_FRACTION_LIMIT
        for field in (
            "lp_call_fraction_of_raw_enumeration",
            "lp_matrix_coefficient_fraction_of_raw_enumeration",
            "candidate_evaluation_fraction_of_raw_enumeration",
        )
    )
    return {
        **accumulator,
        "max_certified_normalized_gap": alpha,
        "mean_certified_normalized_gap": (
            accumulator["normalized_gap_sum"] / nonempty if nonempty else 0.0
        ),
        "exact_certificate_domain_fraction": (
            accumulator["exact_certificate_state_count"] / state_count
        ),
        "raw_ordered_actions_per_state": raw_per_state,
        "represented_count_vectors_per_state": math.comb(
            OPTION_COUNT + lanes, lanes
        ),
        "represented_count_vector_fraction_of_ordered_actions": (
            math.comb(OPTION_COUNT + lanes, lanes) / raw_per_state
        ),
        "mean_lp_calls": accumulator["lp_calls"] / state_count,
        "mean_candidate_evaluations": (
            accumulator["candidate_evaluations"] / state_count
        ),
        **fractions,
        "capacity_bridge": capacity_bridge,
        "capacity_margin_passed": capacity_passed,
        "work_fraction_passed": work_passed,
        "tradeoff_passed": capacity_passed and work_passed,
    }


def _first_budget(rows, field):
    return next((row["lp_node_budget"] for row in rows if row[field]), None)


def _decision(execution_passed, lane_summaries):
    if not execution_passed:
        return "INVALID_EXECUTION"
    scaled = [row for row in lane_summaries if row["lanes"] != 2]
    if all(row["first_tradeoff_budget"] is not None for row in scaled):
        return "COUNT_SPACE_BNB_SCALING_ADVANCE"
    if all(row["first_quality_budget"] is not None for row in scaled):
        return "CERTIFIED_WITH_HIGH_MATRIX_WORK_STOP"
    if all(row["first_work_budget"] is not None for row in scaled):
        return "COMPUTE_SAVING_WITH_LOOSE_CERTIFICATE_STOP"
    return "NO_COUNT_SPACE_SCALING_TRADEOFF_STOP"


def run_registered(output=DEFAULT_OUTPUT):
    """Execute the frozen cumulative traces after source/protocol validation."""
    output, protocol_path, fixed_sources_path = _paths(output)
    if output.exists():
        raise ValueError("count-space scaling result already exists")
    protocol = json.loads(protocol_path.read_text())
    registered_sources = json.loads(fixed_sources_path.read_text())
    if registered_sources != _current_fixed_sources():
        raise ValueError("registered count-space scaling sources changed")
    source_contract, source_model = _source_contract()
    expected = _protocol_constants(source_model, source_contract)
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise ValueError("registered count-space scaling protocol changed")

    model = load_model()
    if _json(model.snapshot()) != source_model:
        raise ValueError("current aligned model differs from the registered source")

    started = time.perf_counter()
    base = base_queues()
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
        "model_has_nine_distinct_duration_roots": (
            len(model.options) == OPTION_COUNT
            and len(set(max(1.0, option.duration) for option in model.options))
            == OPTION_COUNT
        ),
        "scaled_grids_are_exact_unique_and_ordered": True,
        "capacity_certificates_are_valid": True,
        "selection_reconstruction_holds": True,
        "selection_certificate_holds": True,
        "selection_work_identities_hold": True,
        "checkpoint_curves_are_monotone": True,
        "r2_score_optimum_upper_holds_at_every_checkpoint": True,
        "r2_oracle_action_count_is_exact": True,
        "count_space_trace_call_count_is_exact": True,
        "checkpoint_read_count_is_exact": True,
        "capacity_call_count_is_exact": True,
        "no_scaled_ordered_action_enumeration": True,
    }

    capacity_started = time.perf_counter()
    capacity_by_lane = {}
    capacity_calls = 0
    for lanes in LANE_COUNTS:
        rows = []
        synthetic_model = replace(model, lanes=lanes)
        for load in LOADS:
            row = _capacity_row(capacity_certificate(synthetic_model, load))
            rows.append(row)
            capacity_calls += 1
            checks["capacity_certificates_are_valid"] &= (
                row["capacity_projection_passed"]
                and row["conditional_model_drift_certified"]
            )
        capacity_by_lane[str(lanes)] = rows
    capacity_wall_s = time.perf_counter() - capacity_started

    accumulators = {
        lanes: {
            budget: _new_accumulator(budget)
            for budget in LP_CHECKPOINT_BUDGETS
        }
        for lanes in LANE_COUNTS
    }
    trace_calls = 0
    checkpoint_reads = 0
    r2_oracle_calls = 0
    r2_oracle_evaluations = 0
    r2_checkpoint_comparisons = 0
    scaled_ordered_action_evaluations = 0
    counterfactual_repeated_lp_calls = 0
    actual_cumulative_lp_calls = 0
    actual_cumulative_matrix_coefficients = 0
    lane_wall_s = {}

    for lanes in LANE_COUNTS:
        lane_started = time.perf_counter()
        queues = scaled_queues(lanes)
        factor = lanes // 2
        checks["scaled_grids_are_exact_unique_and_ordered"] &= (
            len(queues) == BASE_STATE_COUNT
            and len(set(queues)) == BASE_STATE_COUNT
            and queues == tuple(sorted(queues))
            and all(
                queue
                == tuple(factor * coordinate for coordinate in base_queue)
                for base_queue, queue in zip(base, queues, strict=True)
            )
        )
        for base_queue, queue in zip(base, queues, strict=True):
            optimum = None
            if lanes == 2:
                optimum, evaluations = _r2_oracle(model, queue)
                r2_oracle_calls += 1
                r2_oracle_evaluations += evaluations
                checks["r2_oracle_action_count_is_exact"] &= (
                    evaluations == R2_ORDERED_ACTION_COUNT
                )

            trace = trace_count_space_bnb_frame(
                model.options,
                queue,
                lanes,
                LP_CHECKPOINT_BUDGETS,
            )
            trace_calls += 1
            checkpoint_reads += len(trace)
            checks["checkpoint_read_count_is_exact"] &= (
                len(trace) == len(LP_CHECKPOINT_BUDGETS)
            )
            previous_score = -math.inf
            previous_upper = math.inf
            previous_lp_calls = -1
            for budget, selection in zip(
                LP_CHECKPOINT_BUDGETS, trace, strict=True
            ):
                checks["selection_reconstruction_holds"] &= (
                    _selection_reconstructs(model, queue, lanes, selection)
                )
                checks["selection_certificate_holds"] &= (
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
                        selection.gap / sum(queue) if sum(queue) else 0.0,
                        rel_tol=0.0,
                        abs_tol=SCORE_TOLERANCE,
                    )
                )
                checks["selection_work_identities_hold"] &= (
                    _selection_work_holds(queue, lanes, budget, selection)
                )
                checks["checkpoint_curves_are_monotone"] &= (
                    selection.score + SCORE_TOLERANCE >= previous_score
                    and selection.upper_bound <= previous_upper + SCORE_TOLERANCE
                    and selection.lp_calls >= previous_lp_calls
                )
                if optimum is not None:
                    r2_checkpoint_comparisons += 1
                    checks[
                        "r2_score_optimum_upper_holds_at_every_checkpoint"
                    ] &= (
                        selection.score <= optimum + SCORE_TOLERANCE
                        and optimum <= selection.upper_bound + SCORE_TOLERANCE
                    )
                _update(
                    accumulators[lanes][budget],
                    base_queue,
                    queue,
                    selection,
                )
                counterfactual_repeated_lp_calls += selection.lp_calls
                previous_score = selection.score
                previous_upper = selection.upper_bound
                previous_lp_calls = selection.lp_calls
            actual_cumulative_lp_calls += trace[-1].lp_calls
            actual_cumulative_matrix_coefficients += (
                trace[-1].lp_matrix_coefficients
            )
        lane_wall_s[str(lanes)] = time.perf_counter() - lane_started

    lane_summaries = []
    for lanes in LANE_COUNTS:
        budget_rows = [
            _finalize(
                accumulators[lanes][budget],
                lanes,
                capacity_by_lane[str(lanes)],
            )
            for budget in LP_CHECKPOINT_BUDGETS
        ]
        lane_summaries.append({
            "lanes": lanes,
            "queue_scale": lanes // 2,
            "state_count": BASE_STATE_COUNT,
            "raw_ordered_actions_per_state": LOCAL_CHOICE_COUNT**lanes,
            "represented_count_vectors_per_state": math.comb(
                OPTION_COUNT + lanes, lanes
            ),
            "budget_summaries": budget_rows,
            "first_quality_budget": _first_budget(
                budget_rows, "capacity_margin_passed"
            ),
            "first_work_budget": _first_budget(
                budget_rows, "work_fraction_passed"
            ),
            "first_tradeoff_budget": _first_budget(
                budget_rows, "tradeoff_passed"
            ),
        })

    checks["r2_oracle_action_count_is_exact"] &= (
        r2_oracle_calls == BASE_STATE_COUNT
        and r2_oracle_evaluations
        == BASE_STATE_COUNT * R2_ORDERED_ACTION_COUNT
        and r2_checkpoint_comparisons
        == BASE_STATE_COUNT * len(LP_CHECKPOINT_BUDGETS)
    )
    checks["count_space_trace_call_count_is_exact"] &= (
        trace_calls == BASE_STATE_COUNT * len(LANE_COUNTS)
    )
    checks["checkpoint_read_count_is_exact"] &= (
        checkpoint_reads
        == BASE_STATE_COUNT * len(LANE_COUNTS) * len(LP_CHECKPOINT_BUDGETS)
    )
    checks["capacity_call_count_is_exact"] &= (
        capacity_calls == len(LANE_COUNTS) * len(LOADS)
    )
    checks["no_scaled_ordered_action_enumeration"] &= (
        scaled_ordered_action_evaluations == 0
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
        "capacity_by_lane": capacity_by_lane,
        "lane_summaries": lane_summaries,
        "work_accounting": {
            "count_space_trace_calls": trace_calls,
            "checkpoint_reads": checkpoint_reads,
            "actual_cumulative_lp_calls": actual_cumulative_lp_calls,
            "actual_cumulative_lp_matrix_coefficients": (
                actual_cumulative_matrix_coefficients
            ),
            "counterfactual_lp_calls_if_checkpoints_restarted": (
                counterfactual_repeated_lp_calls
            ),
            "lp_calls_avoided_by_cumulative_trace": (
                counterfactual_repeated_lp_calls - actual_cumulative_lp_calls
            ),
            "capacity_certificate_calls": capacity_calls,
            "r2_oracle_calls": r2_oracle_calls,
            "r2_oracle_action_evaluations": r2_oracle_evaluations,
            "r2_checkpoint_inequality_comparisons": r2_checkpoint_comparisons,
            "scaled_ordered_action_evaluations": scaled_ordered_action_evaluations,
            "count_vector_enumerations": 0,
            "arrival_tapes": 0,
            "trajectory_frames": 0,
            "rng_initializations": 0,
            "random_draws": 0,
        },
        "scientific_decision": decision,
        "development_advance": decision == "COUNT_SPACE_BNB_SCALING_ADVANCE",
        "claim_boundary": CLAIM_BOUNDARY,
        "limitations": [
            "The mechanism and LP budgets were designed after observing the earlier scaling failure.",
            "LP calls, matrix cells, and candidate evaluations are distinct logical-work units, not wall-clock equivalents.",
            "The finite grid does not establish a uniform guarantee over all integer queues.",
            "Synthetic lane replication does not measure physical multi-GPU interference.",
        ],
        "next_step": (
            "Freeze an independent queue set before any trajectory transition."
            if decision == "COUNT_SPACE_BNB_SCALING_ADVANCE"
            else (
                "Preserve the fixed R4 matrix-work failure without changing its "
                "gate or budgets; assess R8/R16 only on a newly frozen independent "
                "large-action-regime queue set before any trajectory transition."
            )
        ),
        "timing": {
            "capacity_wall_s": capacity_wall_s,
            "lane_wall_s": lane_wall_s,
            "total_wall_s": time.perf_counter() - started,
            "timing_is_scientific_gate": False,
        },
    }
    _write_new(output, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
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
    }))


if __name__ == "__main__":
    main()
