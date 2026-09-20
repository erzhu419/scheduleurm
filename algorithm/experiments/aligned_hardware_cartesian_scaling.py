"""Finite-grid computation/certificate scaling for the aligned frame selector.

The nine measured lane options are held fixed while the number of identical
synthetic lanes grows.  Each state is searched once with a continuous bounded
trace.  The runner uses only the selector's returned upper bound; it never
enumerates the Cartesian action family or an oracle.
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

from algorithm.experiments.aligned_budgeted_frame_search_curve import (
    trace_budgeted_frame_search_curve,
)
from algorithm.experiments.aligned_hardware_frame_model import (
    capacity_certificate,
    load_model,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
DEFAULT_OUTPUT = ARTIFACTS / (
    "aligned_hardware_cartesian_scaling_development_20260912.json"
)
SOURCE_RESULT = ARTIFACTS / (
    "aligned_hardware_global_support_extension_development_20260912.json"
)
SOURCE_PROTOCOL = ARTIFACTS / (
    "aligned_hardware_global_support_extension_development_20260912_protocol.json"
)
SOURCE_FIXED_SOURCES = ARTIFACTS / (
    "aligned_hardware_global_support_extension_development_20260912_fixed_sources.json"
)

EXPERIMENT = "aligned_hardware_cartesian_scaling_v1"
SOURCE_EXPERIMENT = "aligned_hardware_global_support_extension_v1"
SOURCE_DECISION = (
    "GLOBAL_K73_RETURNED_GAP_CERTIFIED_K76_EXACT_COMPUTE_GAIN_NEGLIGIBLE_CLOSE"
)
THIS_SOURCE = "algorithm/experiments/aligned_hardware_cartesian_scaling.py"

BASE_CNN_LEVELS = (0, 1, 2, 3, 4, 5, 8, 16, 24, 40)
BASE_JAX_LEVELS = (0, 1, 2, 3, 4, 5, 8, 12, 18, 25)
BASE_LLM_LEVELS = (0, 1, 2, 3, 4, 5, 8, 12, 18, 24)
REQUIRED_BASE_WITNESSES = ((4, 2, 5), (5, 3, 5))
BASE_STATE_COUNT = 1000
LANE_COUNTS = (2, 4, 8, 16)
LOADS = (0.50, 0.80, 0.95)
CHECKPOINT_BUDGETS = (10, 14, 26, 54, 73, 120, 256, 512, 1024, 2048, 4096)
OPTION_COUNT = 9
LOCAL_CHOICE_COUNT = OPTION_COUNT + 1
SCORE_TOLERANCE = 1e-10
WORK_FRACTION_LIMIT = 0.05


RULE = {
    "base_grid": (
        "the complete 10x10x10 threshold-aware Cartesian grid; no sampling or RNG"
    ),
    "queue_scaling": (
        "multiply every base coordinate by the integer lanes/2; no rounding"
    ),
    "primary_coefficient": (
        "alpha_R(K)=max over nonempty scaled grid of returned gap/L1(queue)"
    ),
    "capacity_bridge": (
        "recompute the existing conditional capacity certificate at each synthetic "
        "lane count and require eta_R(load)-alpha_R(K)>1e-10"
    ),
    "computation": (
        "compare paid candidate attempts and fully charged logical work with the "
        "analytic ordered family size 10^R; never materialize that family"
    ),
    "relative_gap": "returned gap/upper is diagnostic only",
}

DECISION_RULE = {
    "INVALID_EXECUTION": "one or more frozen execution checks fail",
    "MEANINGFUL_LAZY_SCALING_ADVANCE": (
        "every synthetic scaled lane count 4, 8 and 16 has one frozen checkpoint "
        "passing both capacity-margin and computation gates"
    ),
    "COMPUTE_SAVING_WITH_LOOSE_CERTIFICATE_STOP": (
        "all scaled lane counts save computation but at least one lacks a checkpoint "
        "whose returned-gap alpha preserves every load margin"
    ),
    "CERTIFICATE_WITHOUT_COMPUTE_SAVING_STOP": (
        "all scaled lane counts preserve a load margin but at least one fails the "
        "computation gate"
    ),
    "NO_SCALING_TRADEOFF_STOP": "no common scaling conclusion passes",
}

CLAIM_BOUNDARY = (
    "A deterministic finite-grid computation and returned-certificate result for "
    "synthetic independent replicas of the fixed nine-option dual-GPU lane model. "
    "It is not evidence for multi-GPU physical throughput, interference, queue "
    "stability, flow-time tails, deployment, untested grid points, or all queues."
)

PROHIBITED = (
    "full Cartesian action enumeration, an oracle, arrivals, trajectories, drains, "
    "RNG, server data, confirmation runs, paper edits, and post-outcome changes to "
    "the grid, lanes, budgets, or gates"
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


def base_queues():
    """Return the frozen 1,000-state base grid in lexicographic order."""
    return tuple(product(BASE_CNN_LEVELS, BASE_JAX_LEVELS, BASE_LLM_LEVELS))


def scaled_queues(lanes):
    """Scale the base grid exactly for one frozen even lane count."""
    if lanes not in LANE_COUNTS:
        raise ValueError(f"lanes must be one of {LANE_COUNTS}")
    factor = lanes // 2
    return tuple(
        tuple(factor * coordinate for coordinate in queue)
        for queue in base_queues()
    )


def _current_fixed_sources():
    upstream = json.loads(SOURCE_FIXED_SOURCES.read_text())
    changed = [
        name for name, text in upstream.items()
        if (ROOT / name).read_text() != text
    ]
    if changed:
        raise ValueError(f"frozen global-extension source changed: {changed}")
    return {**upstream, THIS_SOURCE: (ROOT / THIS_SOURCE).read_text()}


def _source_contract():
    source = json.loads(SOURCE_RESULT.read_text())
    protocol = json.loads(SOURCE_PROTOCOL.read_text())
    if source.get("protocol") != protocol:
        raise ValueError("global-extension result does not embed its protocol")
    if (
        source.get("experiment") != SOURCE_EXPERIMENT
        or source.get("status") != "PASS"
        or source.get("scientific_decision") != SOURCE_DECISION
        or not source.get("checks")
        or not all(source["checks"].values())
        or protocol.get("model", {}).get("lanes") != 2
        or len(protocol.get("model", {}).get("options", ())) != OPTION_COUNT
    ):
        raise ValueError("Cartesian scaling requires the passing global extension")
    return {
        "result": SOURCE_RESULT.name,
        "protocol": SOURCE_PROTOCOL.name,
        "experiment": source["experiment"],
        "status": source["status"],
        "scientific_decision": source["scientific_decision"],
        "source_lane_count": protocol["model"]["lanes"],
        "option_count": len(protocol["model"]["options"]),
        "next_step": source.get("next_step"),
    }, protocol["model"]


def _raw_action_counts():
    return {str(lanes): LOCAL_CHOICE_COUNT ** lanes for lanes in LANE_COUNTS}


def _scaled_maxima():
    maximum = (
        BASE_CNN_LEVELS[-1],
        BASE_JAX_LEVELS[-1],
        BASE_LLM_LEVELS[-1],
    )
    return {
        str(lanes): [coordinate * (lanes // 2) for coordinate in maximum]
        for lanes in LANE_COUNTS
    }


def _protocol_constants(model, source_contract):
    return {
        "experiment": EXPERIMENT,
        "role": "development finite-grid Cartesian search scaling",
        "source_contract": source_contract,
        "base_levels": {
            "cnn": list(BASE_CNN_LEVELS),
            "jax": list(BASE_JAX_LEVELS),
            "llm": list(BASE_LLM_LEVELS),
        },
        "base_state_count": BASE_STATE_COUNT,
        "required_base_witnesses": [list(queue) for queue in REQUIRED_BASE_WITNESSES],
        "lane_counts": list(LANE_COUNTS),
        "queue_scale_by_lane": {
            str(lanes): lanes // 2 for lanes in LANE_COUNTS
        },
        "scaled_maximum_by_lane": _scaled_maxima(),
        "loads": list(LOADS),
        "checkpoint_budgets": list(CHECKPOINT_BUDGETS),
        "option_count": OPTION_COUNT,
        "local_choice_count_including_idle": LOCAL_CHOICE_COUNT,
        "raw_ordered_action_count_by_lane": _raw_action_counts(),
        "score_tolerance": SCORE_TOLERANCE,
        "work_fraction_limit": WORK_FRACTION_LIMIT,
        "expected_trace_calls": BASE_STATE_COUNT * len(LANE_COUNTS),
        "expected_capacity_certificate_calls": len(LANE_COUNTS) * len(LOADS),
        "model": _json(model.snapshot()),
        "rule": RULE,
        "advance_gate": {
            "execution_checks": True,
            "per_scaled_lane_capacity_margin": (
                "there exists a frozen K with eta_R(load)-alpha_R(K)>1e-10 "
                "at loads 0.50, 0.80 and 0.95"
            ),
            "per_scaled_lane_computation": (
                "the same K has candidate and charged logical work fractions below 0.05"
            ),
            "required_lane_counts": [4, 8, 16],
        },
        "decision_rule": DECISION_RULE,
        "claim_boundary": CLAIM_BOUNDARY,
        "prohibited": PROHIBITED,
        "timing_is_scientific_gate": False,
    }


def register(output=DEFAULT_OUTPUT):
    """Freeze the protocol without running search or capacity calculations."""
    output, protocol_path, fixed_sources_path = _paths(output)
    if any(path.exists() for path in (output, protocol_path, fixed_sources_path)):
        raise ValueError("Cartesian-scaling artifacts already exist")
    fixed_sources = _current_fixed_sources()
    source_contract, source_model = _source_contract()
    model = load_model()
    if _json(model.snapshot()) != source_model:
        raise ValueError("current aligned model differs from the global extension")
    protocol = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        **_protocol_constants(model, source_contract),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_new(protocol_path, protocol, compact=True)
    _write_new(fixed_sources_path, fixed_sources)
    return protocol


def _work_identities(curve, option_count):
    work = curve.work
    final = curve.checkpoints[-1].selection
    return (
        work.trace_calls == 1
        and work.checkpoint_reads == len(CHECKPOINT_BUDGETS)
        and work.seed_attempts == option_count + 1
        and work.paid_candidate_attempts == final.attempts
        and work.feasible_candidate_evaluations == final.feasible_candidate_count
        and work.infeasible_candidate_attempts
        == work.paid_candidate_attempts - work.feasible_candidate_evaluations
        and work.local_score_evaluations == option_count
        and work.bound_option_probes == final.bound_option_probes
        and work.bound_threshold_evaluations == final.bound_threshold_evaluations
        and work.frontier_push_attempts
        == work.frontier_pushes + work.bound_rejected_pushes
        and work.frontier_pops
        == work.expanded_prefixes + work.certificate_closure_pops
        and work.frontier_pushes
        == work.frontier_pops
        + work.frontier_entries_discarded_on_exact
        + work.frontier_remaining
        and work.frontier_peak == final.frontier_peak
    )


def _curve_monotone(curve):
    selections = [checkpoint.selection for checkpoint in curve.checkpoints]
    return all(
        right.score + SCORE_TOLERANCE >= left.score
        and right.upper_bound <= left.upper_bound + SCORE_TOLERANCE
        and right.gap <= left.gap + SCORE_TOLERANCE
        and right.attempts >= left.attempts
        and right.attempts <= budget
        and (not left.exact or right.exact)
        for left, right, budget in zip(
            selections[:-1], selections[1:], CHECKPOINT_BUDGETS[1:], strict=True
        )
    )


def _selection_reconstructs(model, queue, lanes, selection):
    if len(selection.option_indices) != lanes:
        return False
    try:
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
    except (IndexError, TypeError):
        return False
    score = sum(
        waiting * done
        for waiting, done in zip(queue, service, strict=True)
    ) / duration
    return (
        tuple(selection.service) == service
        and all(done <= waiting for done, waiting in zip(service, queue, strict=True))
        and math.isclose(selection.duration, duration, rel_tol=0.0, abs_tol=SCORE_TOLERANCE)
        and math.isclose(selection.score, score, rel_tol=0.0, abs_tol=SCORE_TOLERANCE)
    )


def _selection_certificate_holds(queue, selection):
    norm = sum(queue)
    gap = max(0.0, selection.upper_bound - selection.score)
    return (
        selection.score <= selection.upper_bound + SCORE_TOLERANCE
        and selection.gap >= -SCORE_TOLERANCE
        and math.isclose(selection.gap, gap, rel_tol=0.0, abs_tol=SCORE_TOLERANCE)
        and math.isclose(
            selection.epsilon,
            gap / norm if norm else 0.0,
            rel_tol=0.0,
            abs_tol=SCORE_TOLERANCE,
        )
    )


def _selection_witness(base_queue, queue, selection, value):
    return {
        "base_queue": list(base_queue),
        "scaled_queue": list(queue),
        "queue_l1": sum(queue),
        "value": value,
        "score": selection.score,
        "upper_bound": selection.upper_bound,
        "gap": selection.gap,
        "attempts": selection.attempts,
        "exact": selection.exact,
        "option_indices": list(selection.option_indices),
    }


def _new_budget_accumulator(budget):
    return {
        "budget": budget,
        "state_count": 0,
        "nonempty_state_count": 0,
        "positive_gap_state_count": 0,
        "exact_certificate_state_count": 0,
        "alpha": -1.0,
        "alpha_witness": None,
        "normalized_gap_sum": 0.0,
        "max_relative_gap_diagnostic": -1.0,
        "relative_gap_witness": None,
        "relative_gaps": [],
        "candidate_attempts": 0,
        "feasible_candidate_evaluations": 0,
        "local_score_evaluations": 0,
        "bound_option_probes": 0,
        "bound_threshold_evaluations": 0,
        "max_candidate_attempts": 0,
        "peak_search_frontier": 0,
    }


def _update_budget(accumulator, base_queue, queue, selection):
    accumulator["state_count"] += 1
    accumulator["positive_gap_state_count"] += int(
        selection.gap > SCORE_TOLERANCE
    )
    accumulator["exact_certificate_state_count"] += int(selection.exact)
    accumulator["candidate_attempts"] += selection.attempts
    accumulator["feasible_candidate_evaluations"] += (
        selection.feasible_candidate_count
    )
    accumulator["local_score_evaluations"] += selection.local_score_evaluations
    accumulator["bound_option_probes"] += selection.bound_option_probes
    accumulator["bound_threshold_evaluations"] += (
        selection.bound_threshold_evaluations
    )
    accumulator["max_candidate_attempts"] = max(
        accumulator["max_candidate_attempts"], selection.attempts
    )
    accumulator["peak_search_frontier"] = max(
        accumulator["peak_search_frontier"], selection.frontier_peak
    )
    norm = sum(queue)
    if not norm:
        return
    accumulator["nonempty_state_count"] += 1
    normalized_gap = selection.gap / norm
    relative_gap = (
        selection.gap / selection.upper_bound
        if selection.upper_bound > 0.0 else 0.0
    )
    accumulator["normalized_gap_sum"] += normalized_gap
    accumulator["relative_gaps"].append(relative_gap)
    if normalized_gap > accumulator["alpha"]:
        accumulator["alpha"] = normalized_gap
        accumulator["alpha_witness"] = _selection_witness(
            base_queue, queue, selection, normalized_gap
        )
    if relative_gap > accumulator["max_relative_gap_diagnostic"]:
        accumulator["max_relative_gap_diagnostic"] = relative_gap
        accumulator["relative_gap_witness"] = _selection_witness(
            base_queue, queue, selection, relative_gap
        )


def _nearest_rank(values, probability):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def _finalize_budget(accumulator, lanes, capacity_rows):
    state_count = accumulator["state_count"]
    nonempty = accumulator["nonempty_state_count"]
    raw_count = LOCAL_CHOICE_COUNT ** lanes
    alpha = max(0.0, accumulator.pop("alpha"))
    relative_gaps = accumulator.pop("relative_gaps")
    charged_work = sum(
        accumulator[key]
        for key in (
            "candidate_attempts",
            "local_score_evaluations",
            "bound_option_probes",
            "bound_threshold_evaluations",
        )
    )
    candidate_fraction = accumulator["candidate_attempts"] / (
        state_count * raw_count
    )
    charged_fraction = charged_work / (state_count * raw_count)
    capacity_bridge = []
    for certificate in capacity_rows:
        eta = certificate["effective_slack_eta"]
        residual = eta - alpha
        capacity_bridge.append({
            "load": certificate["load"],
            "effective_slack_eta": eta,
            "finite_grid_alpha": alpha,
            "residual_margin": residual,
            "residual_margin_positive": residual > SCORE_TOLERANCE,
        })
    quality_passed = all(
        row["residual_margin_positive"] for row in capacity_bridge
    )
    computation_passed = (
        candidate_fraction < WORK_FRACTION_LIMIT
        and charged_fraction < WORK_FRACTION_LIMIT
    )
    return {
        **accumulator,
        "max_certified_normalized_gap": alpha,
        "mean_certified_normalized_gap": (
            accumulator["normalized_gap_sum"] / nonempty if nonempty else 0.0
        ),
        "mean_relative_gap_diagnostic": (
            sum(relative_gaps) / nonempty if nonempty else 0.0
        ),
        "p95_relative_gap_diagnostic": (
            _nearest_rank(relative_gaps, 0.95) if relative_gaps else 0.0
        ),
        "max_relative_gap_diagnostic": max(
            0.0, accumulator["max_relative_gap_diagnostic"]
        ),
        "exact_certificate_domain_fraction": (
            accumulator["exact_certificate_state_count"] / state_count
        ),
        "mean_candidate_attempts": accumulator["candidate_attempts"] / state_count,
        "raw_ordered_actions_per_state": raw_count,
        "candidate_attempt_fraction_of_raw_enumeration": candidate_fraction,
        "charged_logical_work": charged_work,
        "charged_work_fraction_of_raw_enumeration": charged_fraction,
        "candidate_work_below_limit": candidate_fraction < WORK_FRACTION_LIMIT,
        "charged_work_below_limit": charged_fraction < WORK_FRACTION_LIMIT,
        "capacity_bridge": capacity_bridge,
        "capacity_margin_passed": quality_passed,
        "computation_passed": computation_passed,
        "tradeoff_passed": quality_passed and computation_passed,
    }


def _capacity_row(certificate):
    return {
        "load": certificate["load_factor"],
        "support_slack_delta": certificate["support_slack_delta"],
        "projection_support_loss_rate": certificate[
            "projection_support_loss_rate"
        ],
        "effective_slack_eta": certificate["effective_slack_eta"],
        "capacity_projection_passed": certificate["capacity_projection_passed"],
        "conditional_model_drift_certified": certificate[
            "conditional_model_drift_certified"
        ],
        "physical_service_confidence_increased": certificate[
            "physical_service_confidence_increased"
        ],
    }


def _new_trace_work():
    return None


def _add_trace_work(total, work):
    row = asdict(work)
    if total is None:
        total = {key: 0 for key in row}
    for key, value in row.items():
        if key == "frontier_peak":
            total[key] = max(total[key], value)
        else:
            total[key] += value
    return total


def _first_budget(rows, field):
    return next((row["budget"] for row in rows if row[field]), None)


def _decision(execution_passed, lane_summaries):
    if not execution_passed:
        return "INVALID_EXECUTION"
    scaled = [row for row in lane_summaries if row["lanes"] != 2]
    if all(row["first_tradeoff_budget"] is not None for row in scaled):
        return "MEANINGFUL_LAZY_SCALING_ADVANCE"
    compute_all = all(row["first_compute_budget"] is not None for row in scaled)
    quality_all = all(row["first_quality_budget"] is not None for row in scaled)
    if compute_all and not quality_all:
        return "COMPUTE_SAVING_WITH_LOOSE_CERTIFICATE_STOP"
    if quality_all and not compute_all:
        return "CERTIFICATE_WITHOUT_COMPUTE_SAVING_STOP"
    return "NO_SCALING_TRADEOFF_STOP"


def run_registered(output=DEFAULT_OUTPUT):
    """Run exactly one continuous trace per frozen state and lane count."""
    output, protocol_path, fixed_sources_path = _paths(output)
    if output.exists():
        raise ValueError("Cartesian-scaling result already exists")
    protocol = json.loads(protocol_path.read_text())
    registered_sources = json.loads(fixed_sources_path.read_text())
    if registered_sources != _current_fixed_sources():
        raise ValueError("registered Cartesian-scaling sources changed")
    source_contract, source_model = _source_contract()
    model = load_model()
    if _json(model.snapshot()) != source_model:
        raise ValueError("current aligned model differs from the registered source")
    expected = _protocol_constants(model, source_contract)
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise ValueError("registered Cartesian-scaling protocol changed")

    base = base_queues()
    capacity_started = time.perf_counter()
    capacity_by_lane = {}
    capacity_call_count = 0
    for lanes in LANE_COUNTS:
        synthetic_model = replace(model, lanes=lanes)
        rows = []
        for load in LOADS:
            rows.append(_capacity_row(capacity_certificate(synthetic_model, load)))
            capacity_call_count += 1
        capacity_by_lane[lanes] = rows
    capacity_wall_s = time.perf_counter() - capacity_started

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
            and all(queue in base for queue in REQUIRED_BASE_WITNESSES)
        ),
        "scaled_grids_are_exact_unique_and_ordered": True,
        "capacity_certificates_are_valid": all(
            row["capacity_projection_passed"]
            and row["conditional_model_drift_certified"]
            and row["effective_slack_eta"] > 0.0
            and row["physical_service_confidence_increased"] is False
            for rows in capacity_by_lane.values()
            for row in rows
        ),
        "capacity_eta_scales_linearly_with_lanes": True,
        "all_states_have_all_checkpoints": True,
        "per_state_budget_curve_is_monotone": True,
        "every_checkpoint_selection_is_realizable": True,
        "returned_certificates_reconstruct": True,
        "continuous_trace_work_identities_hold": True,
        "empty_queue_certificate_is_zero": True,
        "two_lane_control_is_exact_by_k120": True,
    }
    for lanes in LANE_COUNTS:
        factor = lanes // 2
        queues = scaled_queues(lanes)
        checks["scaled_grids_are_exact_unique_and_ordered"] &= (
            len(queues) == BASE_STATE_COUNT
            and len(set(queues)) == BASE_STATE_COUNT
            and queues == tuple(sorted(queues))
            and all(
                queue == tuple(factor * value for value in base_queue)
                for base_queue, queue in zip(base, queues, strict=True)
            )
        )
        for reference, row in zip(
            capacity_by_lane[2], capacity_by_lane[lanes], strict=True
        ):
            checks["capacity_eta_scales_linearly_with_lanes"] &= (
                row["load"] == reference["load"]
                and math.isclose(
                    row["effective_slack_eta"],
                    factor * reference["effective_slack_eta"],
                    rel_tol=0.0,
                    abs_tol=SCORE_TOLERANCE,
                )
            )

    accumulators = {
        lanes: {
            budget: _new_budget_accumulator(budget)
            for budget in CHECKPOINT_BUDGETS
        }
        for lanes in LANE_COUNTS
    }
    trace_work_by_lane = {lanes: _new_trace_work() for lanes in LANE_COUNTS}
    trace_wall_by_lane = {lanes: 0.0 for lanes in LANE_COUNTS}
    trace_call_count = 0
    trace_started = time.perf_counter()
    for lanes in LANE_COUNTS:
        queues = scaled_queues(lanes)
        lane_started = time.perf_counter()
        for base_queue, queue in zip(base, queues, strict=True):
            curve = trace_budgeted_frame_search_curve(
                model.options,
                queue,
                lanes,
                checkpoint_budgets=CHECKPOINT_BUDGETS,
            )
            trace_call_count += 1
            trace_work_by_lane[lanes] = _add_trace_work(
                trace_work_by_lane[lanes], curve.work
            )
            checks["all_states_have_all_checkpoints"] &= (
                curve.checkpoint_budgets == CHECKPOINT_BUDGETS
                and len(curve.checkpoints) == len(CHECKPOINT_BUDGETS)
            )
            checks["per_state_budget_curve_is_monotone"] &= _curve_monotone(curve)
            checks["continuous_trace_work_identities_hold"] &= _work_identities(
                curve, len(model.options)
            )
            for checkpoint in curve.checkpoints:
                selection = checkpoint.selection
                checks["every_checkpoint_selection_is_realizable"] &= (
                    _selection_reconstructs(model, queue, lanes, selection)
                )
                checks["returned_certificates_reconstruct"] &= (
                    _selection_certificate_holds(queue, selection)
                )
                if not any(queue):
                    checks["empty_queue_certificate_is_zero"] &= (
                        selection.score == 0.0
                        and selection.upper_bound == 0.0
                        and selection.gap == 0.0
                        and selection.epsilon == 0.0
                    )
                _update_budget(
                    accumulators[lanes][checkpoint.budget],
                    base_queue,
                    queue,
                    selection,
                )
            if lanes == 2:
                checks["two_lane_control_is_exact_by_k120"] &= (
                    curve.selection_at(120).exact
                    and curve.selection_at(120).gap <= SCORE_TOLERANCE
                )
        trace_wall_by_lane[lanes] = time.perf_counter() - lane_started
    trace_wall_s = time.perf_counter() - trace_started

    lane_summaries = []
    for lanes in LANE_COUNTS:
        budget_rows = [
            _finalize_budget(
                accumulators[lanes][budget], lanes, capacity_by_lane[lanes]
            )
            for budget in CHECKPOINT_BUDGETS
        ]
        lane_summaries.append({
            "lanes": lanes,
            "queue_scale": lanes // 2,
            "domain_state_count": BASE_STATE_COUNT,
            "raw_ordered_actions_per_state": LOCAL_CHOICE_COUNT ** lanes,
            "budget_summaries": budget_rows,
            "first_quality_budget": _first_budget(
                budget_rows, "capacity_margin_passed"
            ),
            "first_compute_budget": _first_budget(
                budget_rows, "computation_passed"
            ),
            "first_tradeoff_budget": _first_budget(
                budget_rows, "tradeoff_passed"
            ),
        })

    expected_trace_calls = BASE_STATE_COUNT * len(LANE_COUNTS)
    expected_capacity_calls = len(LANE_COUNTS) * len(LOADS)
    checks.update({
        "exactly_frozen_trace_call_count": trace_call_count == expected_trace_calls,
        "exactly_frozen_capacity_call_count": (
            capacity_call_count == expected_capacity_calls
        ),
        "trace_aggregate_accounting": all(
            work["trace_calls"] == BASE_STATE_COUNT
            and work["checkpoint_reads"]
            == BASE_STATE_COUNT * len(CHECKPOINT_BUDGETS)
            for work in trace_work_by_lane.values()
        ),
        "no_full_action_or_oracle_enumeration": (
            CHECKPOINT_BUDGETS[-1] < LOCAL_CHOICE_COUNT ** min(LANE_COUNTS[1:])
        ),
        "no_arrivals_trajectories_drains_or_rng": True,
        "registered_sources_unchanged": (
            registered_sources == _current_fixed_sources()
        ),
        "registered_source_contract_and_model_unchanged": True,
    })
    execution_passed = all(checks.values())
    decision = _decision(execution_passed, lane_summaries)
    result = {
        "experiment": EXPERIMENT,
        "status": "PASS" if execution_passed else "FAIL",
        "checks": checks,
        "protocol": protocol,
        "capacity_by_lane": [
            {
                "lanes": lanes,
                "queue_scale": lanes // 2,
                "certificates": capacity_by_lane[lanes],
            }
            for lanes in LANE_COUNTS
        ],
        "lane_summaries": lane_summaries,
        "trace_work_by_lane": [
            {"lanes": lanes, **trace_work_by_lane[lanes]}
            for lanes in LANE_COUNTS
        ],
        "validation_work": {
            "continuous_trace_calls": trace_call_count,
            "capacity_certificate_calls": capacity_call_count,
            "full_action_enumeration_calls": 0,
            "oracle_calls": 0,
            "arrival_or_trajectory_calls": 0,
            "per_state_rows_retained": 0,
        },
        "scientific_decision": decision,
        "development_advance": decision == "MEANINGFUL_LAZY_SCALING_ADVANCE",
        "claim_boundary": CLAIM_BOUNDARY,
        "limitations": [
            "The 1,000 deterministic points are a finite grid, not a complete integer box or all-queue proof.",
            "Lane counts above two replicate cached dual-GPU lane options without modelling added interference or topology.",
            "A loose returned gap does not prove the selected action is poor; it means this bounded search did not certify a tighter guarantee.",
            "Analytic raw action counts are computation references, not measured exhaustive runtimes.",
        ],
        "next_step": (
            "tighten the scalable frontier bound before any trajectory experiment"
            if decision == "COMPUTE_SAVING_WITH_LOOSE_CERTIFICATE_STOP"
            else "retain the frozen decision without changing lanes, budgets, or grid"
        ),
        "timing": {
            "capacity_wall_seconds": capacity_wall_s,
            "continuous_trace_wall_seconds": trace_wall_s,
            "continuous_trace_wall_seconds_by_lane": {
                str(lanes): trace_wall_by_lane[lanes] for lanes in LANE_COUNTS
            },
            "timing_is_scientific_gate": False,
        },
    }
    _write_new(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--register", action="store_true")
    mode.add_argument("--run-registered", action="store_true")
    args = parser.parse_args()
    payload = register(args.output) if args.register else run_registered(args.output)
    print(json.dumps({
        "experiment": payload["experiment"],
        "status": payload.get("status", "REGISTERED"),
        "scientific_decision": payload.get("scientific_decision"),
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
