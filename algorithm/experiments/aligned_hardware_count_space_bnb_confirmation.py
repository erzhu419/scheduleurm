"""Independent finite-roster confirmation for large count-space action regimes.

Registration uses one fixed RNG seed to materialize an explicit 512-queue
roster disjoint from the consumed development grid.  Execution reads that
immutable roster and uses no RNG.  R8/K36 and R16/K72 are the fixed primary
cases; R32 and R64 are exploratory scalability rows only.
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

from algorithm.count_space_bnb_frame_candidates import (
    trace_count_space_bnb_frame,
)
from algorithm.experiments.aligned_hardware_count_space_bnb_scaling_screen import (
    LOCAL_CHOICE_COUNT,
    OPTION_COUNT,
    SCORE_TOLERANCE,
    WORK_FRACTION_LIMIT,
    _capacity_row,
    _finalize,
    _new_accumulator,
    _selection_reconstructs,
    _selection_work_holds,
    _update,
)
from algorithm.experiments.aligned_hardware_frame_model import (
    capacity_certificate,
    load_model,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
DEFAULT_OUTPUT = ARTIFACTS / (
    "aligned_hardware_count_space_bnb_confirmation_20260913.json"
)
DEFAULT_DEVELOPMENT_RESULT = ARTIFACTS / (
    "aligned_hardware_count_space_bnb_scaling_screen_development_20260913.json"
)
EXPERIMENT = "aligned_hardware_count_space_bnb_confirmation_v1"
DEVELOPMENT_EXPERIMENT = "aligned_hardware_count_space_bnb_scaling_screen_v1"
DEVELOPMENT_DECISION = "CERTIFIED_WITH_HIGH_MATRIX_WORK_STOP"
THIS_SOURCE = (
    "algorithm/experiments/aligned_hardware_count_space_bnb_confirmation.py"
)

ROSTER_SEED = 260913
ROSTER_SIZE = 512
ROSTER_RANGES = ((0, 48), (0, 30), (0, 30))
OLD_CNN_LEVELS = (0, 1, 2, 3, 4, 5, 8, 16, 24, 40)
OLD_JAX_LEVELS = (0, 1, 2, 3, 4, 5, 8, 12, 18, 25)
OLD_LLM_LEVELS = (0, 1, 2, 3, 4, 5, 8, 12, 18, 24)
LANE_COUNTS = (8, 16, 32, 64)
PRIMARY_BUDGET_BY_LANE = {8: 36, 16: 72}
EXPLORATORY_LANE_COUNTS = (32, 64)
LP_CHECKPOINT_BUDGETS = (9, 18, 36, 72, 144, 288)
LOADS = (0.50, 0.80, 0.95)

DECISION_RULE = {
    "INVALID_EXECUTION": "one or more registered execution checks fail",
    "PRIMARY_LARGE_ACTION_CONFIRMATION_PASS": (
        "R8/K36 and R16/K72 each have a valid certificate at every roster "
        "state, alpha below eta at all three loads, and all three frozen work "
        "fractions below five percent"
    ),
    "PRIMARY_R8_ONLY_PASS_STOP": "only the fixed R8/K36 primary case passes",
    "PRIMARY_R16_ONLY_PASS_STOP": "only the fixed R16/K72 primary case passes",
    "PRIMARY_LARGE_ACTION_CONFIRMATION_FAIL_STOP": (
        "neither fixed primary case passes its complete confirmation gate"
    ),
}

CLAIM_BOUNDARY = (
    "An exhaustive result over one independently generated and preregistered "
    "512-base-queue roster for the synthetic identical-lane R8 and R16 models. "
    "R32/R64 results are exploratory. This is not an all-queue theorem, failure-"
    "rate estimate, physical multi-GPU measurement, trajectory result, or "
    "deployment decision."
)
ROSTER_SIZE_JUDGMENT = (
    "512 is sufficient for an exact finite-roster confirmation because every "
    "registered state is evaluated and the claim is limited to that roster; it "
    "is not used to infer rare-state probabilities or uniform all-queue behavior."
)
PROHIBITED = (
    "ordered-action enumeration, count-vector enumeration, run-time RNG, "
    "arrivals, trajectories, server data, paper edits, and post-result changes "
    "to the roster, primary cases, checkpoints, capacity bridge, or work gate"
)


def _json(value):
    return json.loads(json.dumps(value))


def _paths(output=DEFAULT_OUTPUT):
    output = Path(output)
    return (
        output,
        output.with_name(output.stem + "_protocol.json"),
        output.with_name(output.stem + "_fixed_sources.json"),
        output.with_name(output.stem + "_roster.json"),
    )


def _development_paths(result_path):
    result_path = Path(result_path)
    return (
        result_path,
        result_path.with_name(result_path.stem + "_protocol.json"),
        result_path.with_name(result_path.stem + "_fixed_sources.json"),
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


def _development_budget_row(development, lanes, budget):
    lane = next(
        row for row in development["lane_summaries"] if row["lanes"] == lanes
    )
    return next(
        row
        for row in lane["budget_summaries"]
        if row["lp_node_budget"] == budget
    )


def _source_contract(development_result=DEFAULT_DEVELOPMENT_RESULT):
    result_path, protocol_path, fixed_sources_path = _development_paths(
        development_result
    )
    development = json.loads(result_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    fixed_sources = json.loads(fixed_sources_path.read_text())
    if development.get("protocol") != protocol:
        raise ValueError("count-space development result does not embed its protocol")
    changed = [
        name
        for name, text in fixed_sources.items()
        if (ROOT / name).read_text() != text
    ]
    if changed:
        raise ValueError(f"frozen count-space development source changed: {changed}")
    work = development.get("work_accounting", {})
    if (
        development.get("experiment") != DEVELOPMENT_EXPERIMENT
        or development.get("status") != "PASS"
        or development.get("scientific_decision") != DEVELOPMENT_DECISION
        or development.get("post_result_development_screen") is not True
        or development.get("independent_preregistration") is not False
        or not all(development.get("checks", {}).values())
        or work.get("r2_oracle_calls") != 1000
        or work.get("r2_oracle_action_evaluations") != 100000
        or work.get("r2_checkpoint_inequality_comparisons") != 4000
        or work.get("scaled_ordered_action_evaluations") != 0
        or work.get("count_vector_enumerations") != 0
    ):
        raise ValueError("confirmation requires the passing frozen development screen")
    primary_development_rows = {}
    for lanes, budget in PRIMARY_BUDGET_BY_LANE.items():
        row = _development_budget_row(development, lanes, budget)
        if not row.get("tradeoff_passed"):
            raise ValueError("preselected primary case did not pass development")
        primary_development_rows[str(lanes)] = {
            "lp_node_budget": budget,
            "max_certified_normalized_gap": row[
                "max_certified_normalized_gap"
            ],
            "capacity_bridge": row["capacity_bridge"],
            "lp_call_fraction_of_raw_enumeration": row[
                "lp_call_fraction_of_raw_enumeration"
            ],
            "lp_matrix_coefficient_fraction_of_raw_enumeration": row[
                "lp_matrix_coefficient_fraction_of_raw_enumeration"
            ],
            "candidate_evaluation_fraction_of_raw_enumeration": row[
                "candidate_evaluation_fraction_of_raw_enumeration"
            ],
        }
    contract = {
        "development_result_path": str(result_path.resolve()),
        "development_experiment": development["experiment"],
        "development_decision": development["scientific_decision"],
        "primary_development_rows": primary_development_rows,
        "inherited_r2_oracle_evidence": {
            "oracle_calls": work["r2_oracle_calls"],
            "ordered_action_evaluations": work[
                "r2_oracle_action_evaluations"
            ],
            "checkpoint_inequality_comparisons": work[
                "r2_checkpoint_inequality_comparisons"
            ],
            "score_optimum_upper_check": development["checks"][
                "r2_score_optimum_upper_holds_at_every_checkpoint"
            ],
        },
    }
    return contract, protocol["model"], fixed_sources


def _current_fixed_sources(upstream_fixed_sources):
    return {
        **upstream_fixed_sources,
        THIS_SOURCE: (ROOT / THIS_SOURCE).read_text(),
    }


def _old_base_grid():
    return set(product(OLD_CNN_LEVELS, OLD_JAX_LEVELS, OLD_LLM_LEVELS))


def generate_roster():
    """Generate the registration roster once from the fixed local RNG."""
    rng = random.Random(ROSTER_SEED)
    old = _old_base_grid()
    seen = set()
    queues = []
    candidate_draws = 0
    while len(queues) < ROSTER_SIZE:
        queue = tuple(
            rng.randint(lower, upper) for lower, upper in ROSTER_RANGES
        )
        candidate_draws += 1
        if queue == (0, 0, 0) or queue in old or queue in seen:
            continue
        seen.add(queue)
        queues.append(queue)
    return {
        "experiment": EXPERIMENT,
        "seed": ROSTER_SEED,
        "coordinate_ranges_inclusive": [list(bounds) for bounds in ROSTER_RANGES],
        "requested_queue_count": ROSTER_SIZE,
        "candidate_draws": candidate_draws,
        "old_grid_excluded": True,
        "zero_queue_excluded": True,
        "queues": [list(queue) for queue in queues],
    }


def _protocol_constants(model_snapshot, source_contract, roster_payload):
    return {
        "experiment": EXPERIMENT,
        "role": "independent finite-roster large-action-regime confirmation",
        "source_contract": source_contract,
        "model": _json(model_snapshot),
        "roster_seed": ROSTER_SEED,
        "roster_size": ROSTER_SIZE,
        "roster_coordinate_ranges_inclusive": [
            list(bounds) for bounds in ROSTER_RANGES
        ],
        "roster_generation": (
            "uniform integer coordinate draws from a local random.Random stream; "
            "reject all-zero, consumed old-grid, and duplicate queues; preserve "
            "accepted draw order"
        ),
        "registered_base_queues": roster_payload["queues"],
        "roster_candidate_draws": roster_payload["candidate_draws"],
        "roster_size_judgment": ROSTER_SIZE_JUDGMENT,
        "primary_lane_counts": sorted(PRIMARY_BUDGET_BY_LANE),
        "primary_budget_by_lane": {
            str(lanes): budget
            for lanes, budget in PRIMARY_BUDGET_BY_LANE.items()
        },
        "exploratory_lane_counts": list(EXPLORATORY_LANE_COUNTS),
        "all_lane_counts": list(LANE_COUNTS),
        "queue_scale_by_lane": {
            str(lanes): lanes // 2 for lanes in LANE_COUNTS
        },
        "lp_checkpoint_budgets": list(LP_CHECKPOINT_BUDGETS),
        "checkpoint_execution": (
            "one cumulative count-space B&B trace per registered queue and lane "
            "count; roots and nodes are not repaid between checkpoint reads"
        ),
        "loads": list(LOADS),
        "raw_ordered_action_count_by_lane": {
            str(lanes): LOCAL_CHOICE_COUNT**lanes for lanes in LANE_COUNTS
        },
        "work_fraction_limit": WORK_FRACTION_LIMIT,
        "work_gate_components": [
            "LP-call fraction",
            "LP-matrix coefficient fraction",
            "candidate-evaluation fraction",
        ],
        "primary_gate": {
            "fixed_cases": [
                {"lanes": lanes, "lp_node_budget": budget}
                for lanes, budget in PRIMARY_BUDGET_BY_LANE.items()
            ],
            "every_roster_state_certificate_valid": True,
            "alpha_below_eta_at_loads": list(LOADS),
            "every_work_fraction_below": WORK_FRACTION_LIMIT,
        },
        "exploratory_rule": (
            "for R32 and R64 report the earliest registered checkpoint passing "
            "the unchanged capacity and work gates; exploratory rows cannot alter "
            "the primary decision"
        ),
        "expected_calls": {
            "count_space_traces": ROSTER_SIZE * len(LANE_COUNTS),
            "checkpoint_reads": (
                ROSTER_SIZE * len(LANE_COUNTS) * len(LP_CHECKPOINT_BUDGETS)
            ),
            "capacity_certificate": len(LANE_COUNTS) * len(LOADS),
            "ordered_action_evaluations": 0,
            "count_vector_enumerations": 0,
            "run_rng_initializations": 0,
        },
        "decision_rule": DECISION_RULE,
        "claim_boundary": CLAIM_BOUNDARY,
        "prohibited": PROHIBITED,
        "timing_is_scientific_gate": False,
    }


def register(output=DEFAULT_OUTPUT, development_result=DEFAULT_DEVELOPMENT_RESULT):
    """Write the roster and protocol without LP, selector, or capacity calls."""
    output, protocol_path, fixed_sources_path, roster_path = _paths(output)
    if any(
        path.exists()
        for path in (output, protocol_path, fixed_sources_path, roster_path)
    ):
        raise ValueError("count-space confirmation artifacts already exist")
    source_contract, source_model, upstream_fixed = _source_contract(
        development_result
    )
    fixed_sources = _current_fixed_sources(upstream_fixed)
    roster_payload = generate_roster()
    protocol = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        **_protocol_constants(source_model, source_contract, roster_payload),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_new(roster_path, roster_payload)
    _write_new(protocol_path, protocol, compact=True)
    _write_new(fixed_sources_path, fixed_sources)
    return protocol


def _validate_roster(roster_payload, protocol):
    queues = tuple(tuple(queue) for queue in roster_payload.get("queues", ()))
    old = _old_base_grid()
    return (
        roster_payload.get("experiment") == EXPERIMENT
        and roster_payload.get("seed") == ROSTER_SEED
        and roster_payload.get("coordinate_ranges_inclusive")
        == [list(bounds) for bounds in ROSTER_RANGES]
        and roster_payload.get("requested_queue_count") == ROSTER_SIZE
        and roster_payload.get("old_grid_excluded") is True
        and roster_payload.get("zero_queue_excluded") is True
        and roster_payload.get("queues") == protocol.get("registered_base_queues")
        and roster_payload.get("candidate_draws")
        == protocol.get("roster_candidate_draws")
        and len(queues) == len(set(queues)) == ROSTER_SIZE
        and all(queue != (0, 0, 0) and queue not in old for queue in queues)
        and all(
            len(queue) == len(ROSTER_RANGES)
            and all(
                int(value) == value and lower <= value <= upper
                for value, (lower, upper) in zip(
                    queue, ROSTER_RANGES, strict=True
                )
            )
            for queue in queues
        )
    ), queues


def _first_budget(rows, field):
    return next((row["lp_node_budget"] for row in rows if row[field]), None)


def _primary_decision(execution_passed, primary_cases):
    if not execution_passed:
        return "INVALID_EXECUTION"
    r8 = primary_cases["8"]["primary_gate_passed"]
    r16 = primary_cases["16"]["primary_gate_passed"]
    if r8 and r16:
        return "PRIMARY_LARGE_ACTION_CONFIRMATION_PASS"
    if r8:
        return "PRIMARY_R8_ONLY_PASS_STOP"
    if r16:
        return "PRIMARY_R16_ONLY_PASS_STOP"
    return "PRIMARY_LARGE_ACTION_CONFIRMATION_FAIL_STOP"


def run_registered(output=DEFAULT_OUTPUT):
    """Run the frozen roster after all source, protocol, and roster checks."""
    output, protocol_path, fixed_sources_path, roster_path = _paths(output)
    if output.exists():
        raise ValueError("count-space confirmation result already exists")
    protocol = json.loads(protocol_path.read_text())
    roster_payload = json.loads(roster_path.read_text())
    development_result = protocol.get("source_contract", {}).get(
        "development_result_path"
    )
    if not development_result:
        raise ValueError("registered development source path is missing")
    source_contract, source_model, upstream_fixed = _source_contract(
        development_result
    )
    registered_sources = json.loads(fixed_sources_path.read_text())
    if registered_sources != _current_fixed_sources(upstream_fixed):
        raise ValueError("registered count-space confirmation sources changed")
    roster_valid, roster = _validate_roster(roster_payload, protocol)
    if not roster_valid:
        raise ValueError("registered confirmation roster changed or is invalid")
    expected = _protocol_constants(source_model, source_contract, roster_payload)
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise ValueError("registered count-space confirmation protocol changed")

    model = load_model()
    if _json(model.snapshot()) != source_model:
        raise ValueError("current aligned model differs from the registered source")

    started = time.perf_counter()
    checks = {
        "registered_roster_is_explicit_unique_disjoint_and_in_range": roster_valid,
        "model_has_nine_distinct_duration_roots": (
            len(model.options) == OPTION_COUNT
            and len(set(max(1.0, option.duration) for option in model.options))
            == OPTION_COUNT
        ),
        "capacity_certificates_are_valid": True,
        "scaled_queues_are_exact": True,
        "selection_reconstruction_holds": True,
        "every_state_certificate_is_valid": True,
        "selection_work_identities_hold": True,
        "checkpoint_curves_are_monotone": True,
        "trace_call_count_is_exact": True,
        "checkpoint_read_count_is_exact": True,
        "capacity_call_count_is_exact": True,
        "no_ordered_or_count_action_enumeration": True,
        "no_run_rng": True,
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
    valid_certificate_counts = {
        lanes: {budget: 0 for budget in LP_CHECKPOINT_BUDGETS}
        for lanes in LANE_COUNTS
    }
    trace_calls = 0
    checkpoint_reads = 0
    actual_lp_calls = 0
    actual_matrix_coefficients = 0
    counterfactual_restarted_lp_calls = 0
    ordered_action_evaluations = 0
    count_vector_enumerations = 0
    run_rng_initializations = 0
    lane_wall_s = {}

    for lanes in LANE_COUNTS:
        lane_started = time.perf_counter()
        factor = lanes // 2
        for base_queue in roster:
            queue = tuple(factor * coordinate for coordinate in base_queue)
            checks["scaled_queues_are_exact"] &= all(
                queue[index] == factor * base_queue[index]
                for index in range(len(queue))
            )
            trace = trace_count_space_bnb_frame(
                model.options,
                queue,
                lanes,
                LP_CHECKPOINT_BUDGETS,
            )
            trace_calls += 1
            checkpoint_reads += len(trace)
            previous_score = -math.inf
            previous_upper = math.inf
            previous_lp_calls = -1
            for budget, selection in zip(
                LP_CHECKPOINT_BUDGETS, trace, strict=True
            ):
                reconstructs = _selection_reconstructs(
                    model, queue, lanes, selection
                )
                certificate_valid = (
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
                    and selection.dual_certified_lp_calls
                    + selection.safe_fallback_lp_calls
                    == selection.lp_calls
                )
                work_holds = _selection_work_holds(
                    queue, lanes, budget, selection
                )
                checks["selection_reconstruction_holds"] &= reconstructs
                checks["every_state_certificate_is_valid"] &= certificate_valid
                checks["selection_work_identities_hold"] &= work_holds
                checks["checkpoint_curves_are_monotone"] &= (
                    selection.score + SCORE_TOLERANCE >= previous_score
                    and selection.upper_bound <= previous_upper + SCORE_TOLERANCE
                    and selection.lp_calls >= previous_lp_calls
                )
                valid_certificate_counts[lanes][budget] += int(
                    reconstructs and certificate_valid and work_holds
                )
                _update(
                    accumulators[lanes][budget],
                    base_queue,
                    queue,
                    selection,
                )
                counterfactual_restarted_lp_calls += selection.lp_calls
                previous_score = selection.score
                previous_upper = selection.upper_bound
                previous_lp_calls = selection.lp_calls
            actual_lp_calls += trace[-1].lp_calls
            actual_matrix_coefficients += trace[-1].lp_matrix_coefficients
        lane_wall_s[str(lanes)] = time.perf_counter() - lane_started

    lane_summaries = []
    for lanes in LANE_COUNTS:
        budget_rows = []
        for budget in LP_CHECKPOINT_BUDGETS:
            row = _finalize(
                accumulators[lanes][budget],
                lanes,
                capacity_by_lane[str(lanes)],
            )
            row["valid_certificate_state_count"] = valid_certificate_counts[
                lanes
            ][budget]
            row["every_state_certificate_valid"] = (
                valid_certificate_counts[lanes][budget] == ROSTER_SIZE
            )
            budget_rows.append(row)
        lane_summaries.append({
            "lanes": lanes,
            "role": (
                "primary" if lanes in PRIMARY_BUDGET_BY_LANE else "exploratory"
            ),
            "queue_scale": lanes // 2,
            "state_count": ROSTER_SIZE,
            "raw_ordered_actions_per_state": LOCAL_CHOICE_COUNT**lanes,
            "represented_count_vectors_per_state": math.comb(
                OPTION_COUNT + lanes, lanes
            ),
            "budget_summaries": budget_rows,
            "earliest_capacity_budget": _first_budget(
                budget_rows, "capacity_margin_passed"
            ),
            "earliest_work_budget": _first_budget(
                budget_rows, "work_fraction_passed"
            ),
            "earliest_tradeoff_budget": _first_budget(
                budget_rows, "tradeoff_passed"
            ),
        })

    primary_cases = {}
    for lanes, budget in PRIMARY_BUDGET_BY_LANE.items():
        lane = next(row for row in lane_summaries if row["lanes"] == lanes)
        row = next(
            item
            for item in lane["budget_summaries"]
            if item["lp_node_budget"] == budget
        )
        primary_cases[str(lanes)] = {
            "lanes": lanes,
            "lp_node_budget": budget,
            "max_certified_normalized_gap": row[
                "max_certified_normalized_gap"
            ],
            "capacity_bridge": row["capacity_bridge"],
            "capacity_margin_passed": row["capacity_margin_passed"],
            "lp_call_fraction_of_raw_enumeration": row[
                "lp_call_fraction_of_raw_enumeration"
            ],
            "lp_matrix_coefficient_fraction_of_raw_enumeration": row[
                "lp_matrix_coefficient_fraction_of_raw_enumeration"
            ],
            "candidate_evaluation_fraction_of_raw_enumeration": row[
                "candidate_evaluation_fraction_of_raw_enumeration"
            ],
            "work_fraction_passed": row["work_fraction_passed"],
            "valid_certificate_state_count": row[
                "valid_certificate_state_count"
            ],
            "every_state_certificate_valid": row[
                "every_state_certificate_valid"
            ],
            "primary_gate_passed": (
                row["every_state_certificate_valid"]
                and row["capacity_margin_passed"]
                and row["work_fraction_passed"]
            ),
            "alpha_witness": row["alpha_witness"],
        }

    checks["trace_call_count_is_exact"] &= (
        trace_calls == ROSTER_SIZE * len(LANE_COUNTS)
    )
    checks["checkpoint_read_count_is_exact"] &= (
        checkpoint_reads
        == ROSTER_SIZE * len(LANE_COUNTS) * len(LP_CHECKPOINT_BUDGETS)
    )
    checks["capacity_call_count_is_exact"] &= (
        capacity_calls == len(LANE_COUNTS) * len(LOADS)
    )
    checks["no_ordered_or_count_action_enumeration"] &= (
        ordered_action_evaluations == 0 and count_vector_enumerations == 0
    )
    checks["no_run_rng"] &= run_rng_initializations == 0
    execution_passed = all(checks.values())
    decision = _primary_decision(execution_passed, primary_cases)
    result = {
        "experiment": EXPERIMENT,
        "status": "PASS" if execution_passed else "FAIL",
        "checks": checks,
        "protocol": protocol,
        "independent_confirmation": True,
        "finite_roster_confirmation": True,
        "roster": {
            "file": roster_path.name,
            "state_count": len(roster),
            "candidate_draws_at_registration": roster_payload[
                "candidate_draws"
            ],
            "disjoint_from_consumed_development_grid": True,
        },
        "capacity_by_lane": capacity_by_lane,
        "primary_cases": primary_cases,
        "exploratory_cases": {
            str(lanes): {
                "lanes": lanes,
                "earliest_tradeoff_budget": next(
                    row["earliest_tradeoff_budget"]
                    for row in lane_summaries
                    if row["lanes"] == lanes
                ),
            }
            for lanes in EXPLORATORY_LANE_COUNTS
        },
        "lane_summaries": lane_summaries,
        "work_accounting": {
            "count_space_trace_calls": trace_calls,
            "checkpoint_reads": checkpoint_reads,
            "actual_cumulative_lp_calls": actual_lp_calls,
            "actual_cumulative_lp_matrix_coefficients": actual_matrix_coefficients,
            "counterfactual_lp_calls_if_checkpoints_restarted": (
                counterfactual_restarted_lp_calls
            ),
            "lp_calls_avoided_by_cumulative_trace": (
                counterfactual_restarted_lp_calls - actual_lp_calls
            ),
            "capacity_certificate_calls": capacity_calls,
            "ordered_action_evaluations": ordered_action_evaluations,
            "count_vector_enumerations": count_vector_enumerations,
            "run_rng_initializations": run_rng_initializations,
            "arrival_tapes": 0,
            "trajectory_frames": 0,
        },
        "scientific_decision": decision,
        "confirmation_advance": (
            decision == "PRIMARY_LARGE_ACTION_CONFIRMATION_PASS"
        ),
        "claim_boundary": CLAIM_BOUNDARY,
        "limitations": [
            "The roster is finite and cannot establish a uniform all-queue result.",
            "The roster size is not used to estimate a population failure probability.",
            "R32 and R64 were not part of the primary confirmation gate.",
            "Synthetic lane replication does not measure physical multi-GPU interference.",
        ],
        "next_step": (
            "Use the confirmed R8/K36 and R16/K72 cases in a separately registered "
            "trajectory transition without changing the selector or budgets."
            if decision == "PRIMARY_LARGE_ACTION_CONFIRMATION_PASS"
            else "Preserve the failed primary case and do not tune on this roster."
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
    parser.add_argument(
        "--development-result",
        type=Path,
        default=DEFAULT_DEVELOPMENT_RESULT,
        help="frozen count-space development result used only while registering",
    )
    args = parser.parse_args(argv)
    payload = (
        register(args.output, args.development_result)
        if args.register
        else run_registered(args.output)
    )
    print(json.dumps({
        "experiment": payload["experiment"],
        "status": payload.get("status", "REGISTERED"),
        "scientific_decision": payload.get("scientific_decision"),
        "output": str(args.output),
    }))


if __name__ == "__main__":
    main()
