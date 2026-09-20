"""Registered fixed-state shadow for a residual-capacity pruning sidecar.

The input is the compact 5,003-state roster already projected from the preceding
candidate-search diagnostic.  No arrivals are generated and no selected action
is fed into a simulator.  For each state, one continuous sidecar search is
traced through K26 and K120, with separate selector calls used only to verify
the trace checkpoints.

The v4 option order, feasibility, tie break and paid-attempt budget remain fixed.
The heap remains ordered only by the legacy v4 upper bound and stable serial.
The tighter eight-mask bound can prune a prefix only when it is strictly below
the incumbent and supplies the final all-frontier certificate.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from algorithm.residual_sidecar_frame_candidates import (
    select_residual_sidecar_frame,
)
from algorithm.experiments.residual_sidecar_frame_search_trace import (
    trace_residual_sidecar_frame_search,
)
from algorithm.experiments.aligned_hardware_frame_confirmation import _write_new
from algorithm.experiments.aligned_hardware_frame_model import load_model


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
DEFAULT_OUTPUT = ARTIFACTS / "aligned_hardware_residual_sidecar_shadow_development_20260911.json"
SOURCE_DIAGNOSTIC = ARTIFACTS / "aligned_hardware_candidate_search_diagnostic_development_20260910.json"
SOURCE_INTEGRATED_RESULT = ARTIFACTS / "aligned_hardware_residual_capacity_bound_shadow_development_20260910.json"
SOURCE_DIVERGENCE_RESULT = ARTIFACTS / "aligned_hardware_residual_bound_first_divergence_v2_development_20260910.json"
SOURCE_FIXED_SOURCES = ARTIFACTS / "aligned_hardware_residual_capacity_bound_shadow_development_20260910_fixed_sources.json"
SOURCE_STATE_ROSTER = ARTIFACTS / "aligned_hardware_duration_tie_order_shadow_development_20260910_registered_states.json"
LOADS = (0.50, 0.80, 0.95)
SEEDS = (41, 42, 43)
SOURCE_POLICY_BUDGETS = (26, 120)
SMALL_BUDGET = 26
EXACT_BUDGET = 120
STATE_COUNT = 5003
SCORE_TOLERANCE = 1e-10
AGGREGATE_TOLERANCE = 1e-12
NEW_SOURCES = (
    "algorithm/residual_sidecar_frame_candidates.py",
    "algorithm/experiments/residual_sidecar_frame_search_trace.py",
    "algorithm/experiments/aligned_hardware_residual_sidecar_shadow.py",
)


def _json(value):
    return json.loads(json.dumps(value))


def _paths(output):
    output = Path(output)
    return (
        output,
        output.with_name(output.stem + "_protocol.json"),
        output.with_name(output.stem + "_fixed_sources.json"),
    )


def _source_texts():
    sources = json.loads(SOURCE_FIXED_SOURCES.read_text())
    changed = [name for name, text in sources.items() if (ROOT / name).read_text() != text]
    if changed:
        raise ValueError(f"integrated residual-bound source changed: {changed}")
    return {**sources, **{name: (ROOT / name).read_text() for name in NEW_SOURCES}}


def _registration_context():
    """Validate and compact the already observed motivation for this screen."""
    integrated = json.loads(SOURCE_INTEGRATED_RESULT.read_text())
    divergence = json.loads(SOURCE_DIVERGENCE_RESULT.read_text())
    regression_indices = sorted(
        row["state_index"] for row in integrated["state_changes"]
        if row["support_outcome"] == "regressed"
    )
    divergence_indices = sorted(row["state_index"] for row in divergence["cases"])
    if (
        integrated.get("experiment")
        != "residual_capacity_binary_price_bound_shadow_v1"
        or integrated.get("status") != "PASS"
        or not all(integrated.get("checks", {}).values())
        or integrated.get("state_evaluation_count") != STATE_COUNT
        or integrated.get("shadow_advance")
        or integrated.get("certificate_only")
        or regression_indices != [424, 425, 437, 438]
        or divergence.get("experiment")
        != "residual_capacity_bound_first_divergence_v2"
        or divergence.get("status") != "PASS"
        or not all(divergence.get("checks", {}).values())
        or divergence.get("target_count") != 4
        or divergence_indices != regression_indices
        or divergence.get("mechanism")
        != "heap_reprioritization_is_the_first_search_change_on_all_four_regressions"
        or not divergence.get("pure_priority_first")
    ):
        raise ValueError("residual-sidecar shadow requires the closed integrated result and corrected four-state diagnosis")
    return {
        "integrated_result": SOURCE_INTEGRATED_RESULT.name,
        "integrated_status": integrated["status"],
        "integrated_shadow_advance": integrated["shadow_advance"],
        "integrated_certificate_only": integrated["certificate_only"],
        "integrated_regression_state_indices": regression_indices,
        "diagnosis_result": SOURCE_DIVERGENCE_RESULT.name,
        "diagnosis_status": divergence["status"],
        "diagnosis_target_state_indices": divergence_indices,
        "diagnosis_mechanism": divergence["mechanism"],
        "diagnosis_pure_priority_first": divergence["pure_priority_first"],
    }


def _validate_source_diagnostic():
    report = json.loads(SOURCE_DIAGNOSTIC.read_text())
    protocol = report["protocol"]
    expected_population_ids = [
        (load, budget) for load in LOADS for budget in SOURCE_POLICY_BUDGETS
    ]
    if (report["experiment"] != "aligned_hardware_candidate_search_diagnostic_v1"
            or report["status"] != "PASS" or not all(report["checks"].values())
            or report["development_advance"]
            or protocol["loads"] != list(LOADS)
            or protocol["seeds"] != list(SEEDS)
            or protocol["source_budgets"] != list(SOURCE_POLICY_BUDGETS)
            or protocol["small_budget"] != SMALL_BUDGET
            or protocol["exact_budget"] != EXACT_BUDGET
            or len(report["source_trajectory_replays"]) != 18
            or len(report["state_diagnostics"]) != STATE_COUNT
            or len({tuple(row["queue"]) for row in report["state_diagnostics"]}) != STATE_COUNT
            or [(row["load"], row["source_budget"])
                for row in report["summary_by_load_and_source_policy"]]
            != expected_population_ids):
        raise ValueError("residual-sidecar shadow requires the completed candidate-search diagnostic")
    model = load_model()
    if _json(model.snapshot()) != protocol["model"]:
        raise ValueError("residual-sidecar shadow model changed from its source diagnostic")
    sources = json.loads(SOURCE_FIXED_SOURCES.read_text())
    if any((ROOT / name).read_text() != text for name, text in sources.items()):
        raise ValueError("residual-sidecar shadow source diagnostic code changed")
    return report, model


def _compact_state(row):
    legacy_fields = (
        "option_indices", "service", "duration", "score", "upper_bound",
        "gap", "epsilon", "attempts", "exact",
    )
    return {
        "queue": row["queue"],
        "population_visits": [
            [
                visit["load"], visit["seed"], visit["source_budget"],
                visit["frame_count"], visit["visited_time_s"],
            ]
            for visit in row["population_visits"]
        ],
        "legacy_k26": {key: row["k26"][key] for key in legacy_fields},
        "support_reference_score": row["exact"]["score"],
        "legacy_first_exact_score_attempt": row["first_exact_score_attempt"],
    }


def _registered_state_roster(report):
    states = [_compact_state(row) for row in report["state_diagnostics"]]
    return {
        "source_experiment": SOURCE_DIAGNOSTIC.name,
        "state_count": len(states),
        "states": states,
    }


def _state_visits(state, load, source_policy_budget, seed=None):
    matching = [
        row for row in state["population_visits"]
        if row[0] == load and row[2] == source_policy_budget
        and (seed is None or row[1] == seed)
    ]
    return (
        sum(row[3] for row in matching),
        sum(row[4] for row in matching),
    )


def _physical_signature(model, option_indices, service, duration):
    reads = 0
    offsets = []
    for coordinate in range(len(model.workloads)):
        local = []
        for option_index in option_indices:
            if option_index is None:
                continue
            for offset in model.options[option_index].completion_offsets[coordinate]:
                reads += 1
                local.append(float(offset))
        offsets.append(tuple(sorted(local)))
    return (tuple(service), float(duration), tuple(offsets)), reads


def _realizable(model, queues, selected):
    service = tuple(
        sum(
            model.options[index].requirements[coordinate]
            for index in selected.option_indices if index is not None
        )
        for coordinate in range(len(queues))
    )
    duration = max((
        1.0,
        *(model.options[index].duration
          for index in selected.option_indices if index is not None),
    ))
    score = sum(q * count for q, count in zip(queues, service)) / duration
    return (
        service == selected.service
        and all(count <= q for count, q in zip(service, queues))
        and duration == selected.duration
        and score == selected.score
    )


def _legacy_order(model, queues):
    weights = [
        sum(q * count for q, count in zip(queues, option.requirements))
        for option in model.options
    ]
    copies = [
        min(
            model.lanes,
            min(q // count for q, count in zip(queues, option.requirements) if count),
        )
        for option in model.options
    ]
    return tuple(sorted(range(len(model.options)), key=lambda index: (
        copies[index] == 0, -weights[index], index,
    ))) + (None,)


def _work_structure_holds(model, selected):
    dimensions = len(model.workloads)
    masks = 1 << dimensions
    positions = len(model.options) + 1
    threshold_count = len({
        1.0, *(max(1.0, float(option.duration)) for option in model.options),
    })
    return (
        selected.bound_price_vector_count == masks
        and selected.bound_option_probes == threshold_count * positions
        and selected.bound_dual_table_option_probes
        == threshold_count * masks * positions
        and selected.bound_dual_mask_evaluations
        == selected.bound_threshold_evaluations * masks
        and selected.bound_dual_mask_sums == selected.upper_bound_calls * masks
        and selected.bound_dual_coordinate_products
        == dimensions * (masks * positions + selected.upper_bound_calls)
        and 0 <= selected.bound_strict_tightening_count
        <= selected.upper_bound_calls
        and 0.0 <= selected.bound_tightening_max <= selected.bound_tightening_sum
        and selected.upper_bound_calls
        == selected.legacy_rejected_prefixes
        + selected.virtual_legacy_serial_allocations
        and selected.virtual_legacy_serial_allocations
        == selected.safe_pruned_prefixes_on_generation
        + selected.frontier_pushes
        and selected.frontier_pushes
        == selected.paid_search_pops
        + selected.safe_pruned_prefixes_on_pop
        + selected.exact_stop_frontier_entries_cleared
        + selected.certificate_frontier_entries
        and selected.attempts
        == selected.seed_attempts + selected.paid_search_pops
        and selected.certificate_frontier_scans == 1
        and selected.upper_bound
        <= selected.legacy_priority_upper_bound + SCORE_TOLERANCE
    )


def _evaluate_state(model, state_index, state):
    queues = tuple(state["queue"])
    traced = trace_residual_sidecar_frame_search(
        model.options, queues, model.lanes,
        small_budget=SMALL_BUDGET, exact_budget=EXACT_BUDGET,
    )
    small_shadow = select_residual_sidecar_frame(
        model.options, queues, model.lanes, SMALL_BUDGET,
    )
    exact_shadow = select_residual_sidecar_frame(
        model.options, queues, model.lanes, EXACT_BUDGET,
    )
    if traced.small != small_shadow or traced.exact != exact_shadow:
        raise ValueError(f"residual-sidecar trace checkpoint mismatch at state {state_index}")
    reference_score = state["support_reference_score"]
    support_regret = reference_score - traced.small.score
    bound_slack = traced.small.upper_bound - reference_score
    identity_error = traced.small.gap - support_regret - bound_slack
    legacy = state["legacy_k26"]
    legacy_signature, legacy_reads = _physical_signature(
        model, legacy["option_indices"], legacy["service"], legacy["duration"],
    )
    small_signature, small_reads = _physical_signature(
        model, traced.small.option_indices, traced.small.service, traced.small.duration,
    )
    exact_signature, exact_reads = _physical_signature(
        model, traced.exact.option_indices, traced.exact.service, traced.exact.duration,
    )
    legacy_order = _legacy_order(model, queues)
    score_delta = traced.small.score - legacy["score"]
    upper_delta = traced.small.upper_bound - legacy["upper_bound"]
    gap_delta = traced.small.gap - legacy["gap"]
    attempt_delta = traced.small.attempts - legacy["attempts"]
    exact_support_matches = abs(traced.exact.score - reference_score) <= SCORE_TOLERANCE
    order_matches = (
        traced.small.ordered_option_indices
        == traced.exact.ordered_option_indices
        == legacy_order
    )
    work_structure_matches = (
        _work_structure_holds(model, traced.small)
        and _work_structure_holds(model, traced.exact)
    )
    execution_ok = (
        _realizable(model, queues, traced.small)
        and _realizable(model, queues, traced.exact)
        and traced.small.attempts <= SMALL_BUDGET
        and traced.exact.attempts <= EXACT_BUDGET
        and traced.small.seed_attempts == traced.exact.seed_attempts == len(model.options) + 1
        and traced.exact.exact and exact_support_matches
        and traced.small.score <= reference_score + SCORE_TOLERANCE
        and reference_score <= traced.small.upper_bound + SCORE_TOLERANCE
        and abs(identity_error) <= SCORE_TOLERANCE
        and score_delta >= -SCORE_TOLERANCE
        and upper_delta <= SCORE_TOLERANCE
        and gap_delta <= SCORE_TOLERANCE
        and order_matches and work_structure_matches
    )
    return {
        "state_index": state_index,
        "legacy_v4_order_preserved": order_matches,
        "k26_bound_strictly_activated": (
            traced.small.bound_strict_tightening_count > 0
        ),
        "k26_safe_pruning_activated": (
            traced.small.safe_pruned_prefixes_on_generation
            + traced.small.safe_pruned_prefixes_on_pop > 0
        ),
        "k26_safe_pruned_prefix_count": (
            traced.small.safe_pruned_prefixes_on_generation
            + traced.small.safe_pruned_prefixes_on_pop
        ),
        "k26_bound_strict_tightening_count": (
            traced.small.bound_strict_tightening_count
        ),
        "k26_bound_tightening_sum": traced.small.bound_tightening_sum,
        "k26_bound_tightening_max": traced.small.bound_tightening_max,
        "variant_k26": asdict(traced.small),
        "variant_k120": asdict(traced.exact),
        "legacy_finite_budget_support_regret": max(0.0, reference_score - legacy["score"]),
        "variant_finite_budget_support_regret": max(0.0, support_regret),
        "variant_frontier_bound_slack": max(0.0, bound_slack),
        "variant_gap_identity_error": identity_error,
        "first_exact_score_attempt": traced.first_exact_score_attempt,
        "first_k120_selected_physical_signature_attempt": (
            traced.first_k120_selected_physical_signature_attempt
        ),
        "support_score_delta_over_legacy_k26": score_delta,
        "upper_bound_delta_over_legacy_k26": upper_delta,
        "certified_gap_delta_over_legacy_k26": gap_delta,
        "candidate_attempt_delta_over_legacy_k26": attempt_delta,
        "support_outcome": (
            "improved" if score_delta > SCORE_TOLERANCE else
            "regressed" if score_delta < -SCORE_TOLERANCE else "unchanged"
        ),
        "selected_physical_signature_changed_from_legacy_k26": (
            small_signature != legacy_signature
        ),
        "variant_k26_matches_variant_k120_physical_signature": (
            small_signature == exact_signature
        ),
        "trace_matches_independent_k26": True,
        "trace_matches_independent_k120": True,
        "variant_exact_support_matches_registered_reference": exact_support_matches,
        "work_structure_checks_pass": work_structure_matches,
        "selection_and_bound_checks_pass": execution_ok,
        "work": {
            "continuous_trace_candidate_attempts": traced.exact.attempts,
            "continuous_trace_local_score_evaluations": traced.exact.local_score_evaluations,
            "continuous_trace_bound_option_probes": traced.exact.bound_option_probes,
            "continuous_trace_bound_threshold_evaluations": traced.exact.bound_threshold_evaluations,
            "continuous_trace_upper_bound_calls": traced.exact.upper_bound_calls,
            "continuous_trace_dual_table_option_probes": traced.exact.bound_dual_table_option_probes,
            "continuous_trace_dual_mask_evaluations": traced.exact.bound_dual_mask_evaluations,
            "continuous_trace_dual_mask_sums": traced.exact.bound_dual_mask_sums,
            "continuous_trace_dual_coordinate_products": traced.exact.bound_dual_coordinate_products,
            "continuous_trace_strict_tightening_count": traced.exact.bound_strict_tightening_count,
            "continuous_trace_tightening_sum": traced.exact.bound_tightening_sum,
            "continuous_trace_tightening_max": traced.exact.bound_tightening_max,
            "continuous_trace_generation_prunes": traced.exact.safe_pruned_prefixes_on_generation,
            "continuous_trace_lazy_pop_prunes": traced.exact.safe_pruned_prefixes_on_pop,
            "continuous_trace_legacy_rejected_prefixes": traced.exact.legacy_rejected_prefixes,
            "continuous_trace_frontier_pushes": traced.exact.frontier_pushes,
            "continuous_trace_paid_search_pops": traced.exact.paid_search_pops,
            "continuous_trace_exact_stop_frontier_entries_cleared": traced.exact.exact_stop_frontier_entries_cleared,
            "continuous_trace_virtual_legacy_serial_allocations": traced.exact.virtual_legacy_serial_allocations,
            "continuous_trace_certificate_frontier_scans": traced.exact.certificate_frontier_scans,
            "continuous_trace_certificate_frontier_entries": traced.exact.certificate_frontier_entries,
            "continuous_trace_feasible_candidate_evaluations": traced.exact.feasible_candidate_count,
            "continuous_trace_frontier_peak": traced.exact.frontier_peak,
            "continuous_trace_physical_signature_evaluations": traced.physical_signature_evaluations,
            "continuous_trace_completion_offset_reads": traced.physical_signature_completion_offset_reads,
            "independent_validation_candidate_attempts": small_shadow.attempts + exact_shadow.attempts,
            "independent_validation_local_score_evaluations": small_shadow.local_score_evaluations + exact_shadow.local_score_evaluations,
            "independent_validation_bound_option_probes": small_shadow.bound_option_probes + exact_shadow.bound_option_probes,
            "independent_validation_bound_threshold_evaluations": small_shadow.bound_threshold_evaluations + exact_shadow.bound_threshold_evaluations,
            "independent_validation_upper_bound_calls": small_shadow.upper_bound_calls + exact_shadow.upper_bound_calls,
            "independent_validation_dual_table_option_probes": small_shadow.bound_dual_table_option_probes + exact_shadow.bound_dual_table_option_probes,
            "independent_validation_dual_mask_evaluations": small_shadow.bound_dual_mask_evaluations + exact_shadow.bound_dual_mask_evaluations,
            "independent_validation_dual_mask_sums": small_shadow.bound_dual_mask_sums + exact_shadow.bound_dual_mask_sums,
            "independent_validation_dual_coordinate_products": small_shadow.bound_dual_coordinate_products + exact_shadow.bound_dual_coordinate_products,
            "independent_validation_strict_tightening_count": small_shadow.bound_strict_tightening_count + exact_shadow.bound_strict_tightening_count,
            "independent_validation_tightening_sum": small_shadow.bound_tightening_sum + exact_shadow.bound_tightening_sum,
            "independent_validation_tightening_max": max(small_shadow.bound_tightening_max, exact_shadow.bound_tightening_max),
            "independent_validation_generation_prunes": small_shadow.safe_pruned_prefixes_on_generation + exact_shadow.safe_pruned_prefixes_on_generation,
            "independent_validation_lazy_pop_prunes": small_shadow.safe_pruned_prefixes_on_pop + exact_shadow.safe_pruned_prefixes_on_pop,
            "independent_validation_legacy_rejected_prefixes": small_shadow.legacy_rejected_prefixes + exact_shadow.legacy_rejected_prefixes,
            "independent_validation_frontier_pushes": small_shadow.frontier_pushes + exact_shadow.frontier_pushes,
            "independent_validation_paid_search_pops": small_shadow.paid_search_pops + exact_shadow.paid_search_pops,
            "independent_validation_exact_stop_frontier_entries_cleared": small_shadow.exact_stop_frontier_entries_cleared + exact_shadow.exact_stop_frontier_entries_cleared,
            "independent_validation_virtual_legacy_serial_allocations": small_shadow.virtual_legacy_serial_allocations + exact_shadow.virtual_legacy_serial_allocations,
            "independent_validation_certificate_frontier_scans": small_shadow.certificate_frontier_scans + exact_shadow.certificate_frontier_scans,
            "independent_validation_certificate_frontier_entries": small_shadow.certificate_frontier_entries + exact_shadow.certificate_frontier_entries,
            "independent_validation_feasible_candidate_evaluations": small_shadow.feasible_candidate_count + exact_shadow.feasible_candidate_count,
            "independent_validation_frontier_peak": max(small_shadow.frontier_peak, exact_shadow.frontier_peak),
            "runner_physical_signature_evaluations": 3,
            "runner_completion_offset_reads": legacy_reads + small_reads + exact_reads,
            "legacy_order_option_key_evaluations": len(model.options),
        },
    }


def _metrics_for_population(states, evaluations, load, source_policy_budget, seed=None):
    weighted = []
    for state, evaluation in zip(states, evaluations, strict=True):
        count, visited_time = _state_visits(state, load, source_policy_budget, seed)
        if count:
            weighted.append((state, evaluation, count, visited_time))
    frame_count = sum(row[2] for row in weighted)
    visited_time = sum(row[3] for row in weighted)

    def average(function):
        return sum(count * function(state, evaluation)
                   for state, evaluation, count, _ in weighted) / frame_count

    def frames_where(function):
        return sum(count for state, evaluation, count, _ in weighted
                   if function(state, evaluation))

    def time_where(function):
        return sum(time for state, evaluation, _, time in weighted
                   if function(state, evaluation))

    def baseline_values(state, evaluation):
        selection = state["legacy_k26"]
        reference = state["support_reference_score"]
        regret = max(0.0, reference - selection["score"])
        queue_norm = sum(state["queue"])
        return selection, regret, queue_norm

    def variant_values(state, evaluation):
        selection = evaluation["variant_k26"]
        regret = evaluation["variant_finite_budget_support_regret"]
        queue_norm = sum(state["queue"])
        return selection, regret, queue_norm

    def summary(which):
        values = baseline_values if which == "legacy" else variant_values
        return {
            "strict_support_loss_frame_count": frames_where(
                lambda state, evaluation: values(state, evaluation)[1] > SCORE_TOLERANCE
            ),
            "exact_support_score_frame_fraction": frames_where(
                lambda state, evaluation: values(state, evaluation)[1] <= SCORE_TOLERANCE
            ) / frame_count,
            "exact_certified_frame_fraction": frames_where(
                lambda state, evaluation: values(state, evaluation)[0]["exact"]
            ) / frame_count,
            "mean_finite_budget_support_regret": average(
                lambda state, evaluation: values(state, evaluation)[1]
            ),
            "mean_finite_budget_support_regret_per_visible_job": average(
                lambda state, evaluation: (
                    values(state, evaluation)[1] / values(state, evaluation)[2]
                    if values(state, evaluation)[2] else 0.0
                )
            ),
            "mean_certified_gap": average(
                lambda state, evaluation: values(state, evaluation)[0]["gap"]
            ),
            "mean_certified_gap_per_visible_job": average(
                lambda state, evaluation: values(state, evaluation)[0]["epsilon"]
            ),
            "mean_candidate_attempts": average(
                lambda state, evaluation: values(state, evaluation)[0]["attempts"]
            ),
        }

    legacy_summary = summary("legacy")
    variant_summary = summary("variant")
    return {
        "load": load, "seed": seed,
        "source_policy_budget": source_policy_budget,
        "population_role": (
            "primary frozen K26-policy states" if source_policy_budget == SMALL_BUDGET
            else "frozen K120-policy state sensitivity"
        ),
        "frame_count": frame_count, "legacy_visited_time_s": visited_time,
        "unique_queue_state_count": len(weighted),
        "strict_bound_activation_frame_count": frames_where(
            lambda state, evaluation: evaluation["k26_bound_strictly_activated"]
        ),
        "strict_bound_activation_frame_fraction": frames_where(
            lambda state, evaluation: evaluation["k26_bound_strictly_activated"]
        ) / frame_count,
        "strict_bound_activation_time_fraction": time_where(
            lambda state, evaluation: evaluation["k26_bound_strictly_activated"]
        ) / visited_time,
        "safe_pruning_frame_count": frames_where(
            lambda state, evaluation: evaluation["k26_safe_pruning_activated"]
        ),
        "safe_pruning_frame_fraction": frames_where(
            lambda state, evaluation: evaluation["k26_safe_pruning_activated"]
        ) / frame_count,
        "safe_pruning_time_fraction": time_where(
            lambda state, evaluation: evaluation["k26_safe_pruning_activated"]
        ) / visited_time,
        "mean_safe_pruned_prefixes": average(
            lambda state, evaluation: evaluation["k26_safe_pruned_prefix_count"]
        ),
        "mean_k26_bound_tightening_sum": average(
            lambda state, evaluation: evaluation["k26_bound_tightening_sum"]
        ),
        "mean_k26_bound_tightening_max": average(
            lambda state, evaluation: evaluation["k26_bound_tightening_max"]
        ),
        "support_improved_frame_count": frames_where(
            lambda state, evaluation: evaluation["support_outcome"] == "improved"
        ),
        "support_regressed_frame_count": frames_where(
            lambda state, evaluation: evaluation["support_outcome"] == "regressed"
        ),
        "certificate_improved_frame_count": frames_where(
            lambda state, evaluation:
            evaluation["certified_gap_delta_over_legacy_k26"]
            < -SCORE_TOLERANCE
        ),
        "candidate_attempts_reduced_frame_count": frames_where(
            lambda state, evaluation:
            evaluation["candidate_attempt_delta_over_legacy_k26"] < 0
        ),
        "selected_physical_signature_changed_frame_fraction": frames_where(
            lambda state, evaluation:
            evaluation["selected_physical_signature_changed_from_legacy_k26"]
        ) / frame_count,
        "selected_physical_signature_changed_time_fraction": time_where(
            lambda state, evaluation:
            evaluation["selected_physical_signature_changed_from_legacy_k26"]
        ) / visited_time,
        "legacy": legacy_summary, "variant": variant_summary,
        "variant_minus_legacy": {
            key: variant_summary[key] - legacy_summary[key]
            for key in variant_summary
            if isinstance(variant_summary[key], (int, float))
        },
    }


def _source_summary_matches(source_summary, shadow_summary):
    exact_fields = {
        "frame_count": "frame_count",
        "unique_queue_state_count": "unique_queue_state_count",
        "strict_support_loss_frame_count": "strict_support_loss_frame_count",
    }
    if any(source_summary[left] != shadow_summary["legacy"][right]
           if left == "strict_support_loss_frame_count" else
           source_summary[left] != shadow_summary[right]
           for left, right in exact_fields.items()):
        return False
    float_pairs = (
        (source_summary["exact_score_found_by_k26_frame_fraction"],
         shadow_summary["legacy"]["exact_support_score_frame_fraction"]),
        (source_summary["k26_exact_certified_frame_fraction"],
         shadow_summary["legacy"]["exact_certified_frame_fraction"]),
        (source_summary["mean_finite_budget_support_regret"],
         shadow_summary["legacy"]["mean_finite_budget_support_regret"]),
        (source_summary["mean_finite_budget_support_regret_per_visible_job"],
         shadow_summary["legacy"]["mean_finite_budget_support_regret_per_visible_job"]),
        (source_summary["mean_certified_gap"],
         shadow_summary["legacy"]["mean_certified_gap"]),
    )
    return all(math.isclose(left, right, rel_tol=0.0, abs_tol=AGGREGATE_TOLERANCE)
               for left, right in float_pairs)


def _advance_checks(primary_summaries, sensitivity_summaries, evaluations,
                    execution_passed):
    def nonworse(summaries, metric, *, higher=False):
        if higher:
            return all(
                row["variant"][metric] + AGGREGATE_TOLERANCE
                >= row["legacy"][metric] for row in summaries
            )
        return all(
            row["variant"][metric]
            <= row["legacy"][metric] + AGGREGATE_TOLERANCE
            for row in summaries
        )

    return {
        "execution_passed": execution_passed,
        "no_registered_state_has_lower_k26_support": all(
            row["support_outcome"] != "regressed" for row in evaluations
        ),
        "no_registered_state_has_looser_k26_certificate": all(
            row["upper_bound_delta_over_legacy_k26"] <= SCORE_TOLERANCE
            for row in evaluations
        ),
        "no_registered_state_has_larger_k26_gap": all(
            row["certified_gap_delta_over_legacy_k26"] <= SCORE_TOLERANCE
            for row in evaluations
        ),
        "primary_mean_support_regret_nonworse_at_every_load": nonworse(
            primary_summaries, "mean_finite_budget_support_regret",
        ),
        "primary_strict_loss_count_nonworse_at_every_load": nonworse(
            primary_summaries, "strict_support_loss_frame_count",
        ),
        "primary_mean_certified_gap_nonworse_at_every_load": nonworse(
            primary_summaries, "mean_certified_gap",
        ),
        "primary_exact_certified_fraction_nonworse_at_every_load": nonworse(
            primary_summaries, "exact_certified_frame_fraction", higher=True,
        ),
        "primary_exact_support_fraction_nonworse_at_every_load": nonworse(
            primary_summaries, "exact_support_score_frame_fraction", higher=True,
        ),
        "primary_mean_candidate_attempts_nonworse_at_every_load": nonworse(
            primary_summaries, "mean_candidate_attempts",
        ),
        "sensitivity_mean_support_regret_nonworse_at_every_load": nonworse(
            sensitivity_summaries, "mean_finite_budget_support_regret",
        ),
        "sensitivity_strict_loss_count_nonworse_at_every_load": nonworse(
            sensitivity_summaries, "strict_support_loss_frame_count",
        ),
        "sensitivity_mean_certified_gap_nonworse_at_every_load": nonworse(
            sensitivity_summaries, "mean_certified_gap",
        ),
        "sensitivity_exact_certified_fraction_nonworse_at_every_load": nonworse(
            sensitivity_summaries, "exact_certified_frame_fraction", higher=True,
        ),
        "sensitivity_exact_support_fraction_nonworse_at_every_load": nonworse(
            sensitivity_summaries, "exact_support_score_frame_fraction", higher=True,
        ),
        "sensitivity_mean_candidate_attempts_nonworse_at_every_load": nonworse(
            sensitivity_summaries, "mean_candidate_attempts",
        ),
        "at_least_one_primary_load_strictly_improves_support": any(
            row["variant"]["mean_finite_budget_support_regret"]
            < row["legacy"]["mean_finite_budget_support_regret"] - AGGREGATE_TOLERANCE
            or row["variant"]["exact_support_score_frame_fraction"]
            > row["legacy"]["exact_support_score_frame_fraction"] + AGGREGATE_TOLERANCE
            or row["variant"]["strict_support_loss_frame_count"]
            < row["legacy"]["strict_support_loss_frame_count"]
            for row in primary_summaries
        ),
        "at_least_one_primary_load_strictly_improves_certificate_or_attempts": any(
            row["variant"]["mean_certified_gap"]
            < row["legacy"]["mean_certified_gap"] - AGGREGATE_TOLERANCE
            or row["variant"]["exact_certified_frame_fraction"]
            > row["legacy"]["exact_certified_frame_fraction"] + AGGREGATE_TOLERANCE
            or row["variant"]["mean_candidate_attempts"]
            < row["legacy"]["mean_candidate_attempts"] - AGGREGATE_TOLERANCE
            for row in primary_summaries
        ),
        "primary_exposure_contains_strict_bound_activation": any(
            row["strict_bound_activation_frame_count"] > 0
            for row in primary_summaries
        ),
        "primary_exposure_contains_safe_pruning": any(
            row["safe_pruning_frame_count"] > 0
            for row in primary_summaries
        ),
    }


def _decision(advance_checks):
    execution_passed = advance_checks["execution_passed"]
    outcome_checks = {
        key: value for key, value in advance_checks.items()
        if key not in {
            "at_least_one_primary_load_strictly_improves_support",
            "at_least_one_primary_load_strictly_improves_certificate_or_attempts",
        }
    }
    all_guardrails = all(outcome_checks.values())
    shadow_advance = (
        all_guardrails
        and advance_checks["at_least_one_primary_load_strictly_improves_support"]
    )
    certificate_or_work_only = (
        all_guardrails
        and not advance_checks["at_least_one_primary_load_strictly_improves_support"]
        and advance_checks[
            "at_least_one_primary_load_strictly_improves_certificate_or_attempts"
        ]
    )
    if not execution_passed:
        scientific_decision = "INVALID_EXECUTION"
        next_step = (
            "invalid execution; make no scientific decision and register a "
            "corrected version"
        )
    elif shadow_advance:
        scientific_decision = "ADVANCE_TO_FULL_DEVELOPMENT_TRAJECTORIES"
        next_step = (
            "preregister full development trajectories for the frozen "
            "legacy-priority residual-capacity sidecar"
        )
    elif certificate_or_work_only:
        scientific_decision = "CERTIFICATE_OR_WORK_ONLY_STOP"
        next_step = (
            "record certificate or candidate-attempt improvement and stop without "
            "full trajectories"
        )
    else:
        scientific_decision = "CLOSE_WITHOUT_TRAJECTORIES"
        next_step = (
            "close the legacy-priority residual-capacity sidecar without tuning "
            "or full trajectories"
        )
    return shadow_advance, certificate_or_work_only, scientific_decision, next_step


def register(output=DEFAULT_OUTPUT):
    output, protocol_path, sources_path = _paths(output)
    if any(path.exists() for path in (output, protocol_path, sources_path)):
        raise ValueError("residual-sidecar shadow artifacts already exist")
    diagnostic, model = _validate_source_diagnostic()
    registration_context = _registration_context()
    roster = json.loads(SOURCE_STATE_ROSTER.read_text())
    if roster != _registered_state_roster(diagnostic):
        raise ValueError("reused registered state roster does not match its source diagnostic")
    protocol = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "residual_capacity_binary_price_sidecar_shadow_v1",
        "role": (
            "post-outcome development fixed-state mechanism screen on consumed "
            "states; not independent confirmation"
        ),
        "registration_context": registration_context,
        "source_experiment": SOURCE_DIAGNOSTIC.name,
        "reused_registered_state_roster": SOURCE_STATE_ROSTER.name,
        "roster_validation": (
            "exact compact projection of all 5,003 source diagnostic states; the "
            "four motivating regressions receive no special filtering or weight"
        ),
        "loads": LOADS,
        "seeds": SEEDS,
        "source_policy_budgets": SOURCE_POLICY_BUDGETS,
        "small_checkpoint_budget": SMALL_BUDGET,
        "exact_checkpoint_budget": EXACT_BUDGET,
        "state_count": STATE_COUNT,
        "model": model.snapshot(),
        "registered_state_schema": {
            "population_visit": [
                "load", "seed", "source_policy_budget", "frame_count",
                "legacy_visited_time_s",
            ],
            "legacy_k26": [
                "option_indices", "service", "duration", "score", "upper_bound",
                "gap", "epsilon", "attempts", "exact",
            ],
            "support_reference_score": (
                "registered legacy K120 maximum one-frame support"
            ),
            "legacy_first_exact_score_attempt": (
                "first legacy-search paid attempt attaining registered support"
            ),
        },
        "rule": {
            "ordered_key": "(seed_copies[i] == 0, -weights[i], i)",
            "idle_position": "last",
            "heap_priority": "(-legacy_v4_upper_bound, virtual_legacy_serial)",
            "residual_bound_role": "safe pruning and final certificate only; never heap priority or tie-break",
            "price_vectors": "all lambda_c in {0, Q_c}; 2^3=8 fixed masks",
            "threshold_cap": "min(v4 weighted cap, v4 coordinate cap, min_lambda[lambda dot residual + suffix reduced max + (remaining-1) global reduced max])",
            "generation_gate": "legacy_v4_upper_bound > incumbent",
            "pruning_rule": "residual_upper_bound < incumbent",
            "equality_rule": "a residual upper bound equal to the incumbent is retained whenever the unchanged v4 generation gate retains it",
            "lazy_pruning": "an already queued prefix is removed without a paid attempt when its residual upper bound becomes strictly below the incumbent",
            "candidate_attempt": "idle, nine homogeneous seeds, and each surviving popped option choice; pruned prefixes are free",
            "certificate": "max(incumbent, maximum residual upper bound across every surviving frontier entry)",
            "unchanged": "homogeneous seeds, v4 option order, stable serial allocation point, feasibility, consider tie break, duration thresholds and search tree",
            "admissibility": "weak duality for every nonnegative price vector; the zero mask reproduces the v4 weighted cap",
        },
        "execution_invariants": {
            "same_budget_support": "sidecar K26 score is pointwise at least registered legacy K26",
            "same_budget_certificate": "sidecar K26 upper bound is pointwise no larger than registered legacy K26",
            "same_budget_gap": "sidecar K26 gap is pointwise no larger than registered legacy K26",
            "exact_checkpoint": "sidecar K120 is exact and matches registered legacy K120 support",
            "trace": "one continuous K26-to-K120 trace matches independent K26 and K120 selector calls",
            "work": "all prefix, attempt, bound-mask and frontier accounting identities close",
        },
        "source_population_summaries": diagnostic["summary_by_load_and_source_policy"],
        "source_work_accounting": {
            "single_continuous_k120_trace": diagnostic["work_accounting"][
                "single_continuous_k120_trace"
            ],
            "separate_k26_k120_selector_validation": diagnostic[
                "work_accounting"
            ]["separate_k26_k120_selector_validation"],
        },
        "source_population_cells": [
            {
                "load": row["load"],
                "seed": row["seed"],
                "source_policy_budget": row["source_budget"],
                "frame_count": row["frame_count"],
                "legacy_visited_time_s": row["elapsed_including_drain_s"],
            }
            for row in diagnostic["source_trajectory_replays"]
        ],
        "advance_gate": {
            "execution": "all execution invariants and all source reconstruction checks pass",
            "primary": (
                "at every K26-policy load, mean support regret, strict loss count, "
                "mean certified gap and mean candidate attempts are no worse, while "
                "exact-support and exact-certified fractions are no worse"
            ),
            "sensitivity_veto": "the same six nonworsening checks hold at every K120-policy load",
            "advance": "primary exposure contains safe pruning and at least one primary load strictly improves a support metric",
            "certificate_or_work_only": "all guardrails pass, no support metric strictly improves, and a certificate or candidate-attempt metric strictly improves",
        },
        "work_accounting": {
            "candidate_attempts": "paid candidate construction only; safe pruning is uncharged",
            "legacy_bound": "one v4 suffix table and one old upper bound per generated prefix",
            "dual_bound": "eight price masks, mask tables, eligible-threshold evaluations and residual mask sums",
            "frontier": "virtual serial allocations, pushes, generation prunes, lazy pop prunes, paid pops, exact-stop clears and final certificate entries",
            "certificate_scan": "one side-effect-free full-frontier scan per returned checkpoint",
            "runtime_interpretation": "fewer paid attempts do not imply lower wall time because the sidecar adds dual-bound work",
        },
        "stopping_rule": (
            "evaluate this fixed sidecar once on all reused states; execution failure "
            "requires a newly registered corrected version; support improvement under "
            "all guardrails advances to separately preregistered full development "
            "trajectories; certificate-or-work-only benefit stops here; every other "
            "valid outcome closes the sidecar without tuning masks, budgets or states"
        ),
        "prohibited_in_this_stage": (
            "arrival generation, trajectory simulation, policy feedback, new seeds, "
            "new states, mask or budget variants, confirmation claims and paper edits"
        ),
        "support_reference": (
            "registered legacy K120 score only; exact means maximum one-frame support "
            "and does not define a physical tie or flow oracle"
        ),
    }
    sources = _source_texts()
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_new(protocol_path, protocol)
    _write_new(sources_path, sources)
    return protocol


def run_registered(output=DEFAULT_OUTPUT):
    output, protocol_path, sources_path = _paths(output)
    if output.exists():
        raise ValueError("residual-sidecar shadow result already exists")
    protocol = json.loads(protocol_path.read_text())
    roster = json.loads(SOURCE_STATE_ROSTER.read_text())
    sources = json.loads(sources_path.read_text())
    expected = {
        "loads": list(LOADS), "seeds": list(SEEDS),
        "source_policy_budgets": list(SOURCE_POLICY_BUDGETS),
        "small_checkpoint_budget": SMALL_BUDGET,
        "exact_checkpoint_budget": EXACT_BUDGET,
        "state_count": STATE_COUNT,
    }
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise ValueError("registered residual-sidecar roster or budgets changed")
    if protocol.get("experiment") != "residual_capacity_binary_price_sidecar_shadow_v1":
        raise ValueError("registered residual-sidecar experiment changed")
    if protocol.get("reused_registered_state_roster") != SOURCE_STATE_ROSTER.name:
        raise ValueError("registered residual-sidecar roster reference changed")
    expected_rule = "(seed_copies[i] == 0, -weights[i], i)"
    if protocol["rule"]["ordered_key"] != expected_rule:
        raise ValueError("registered residual-sidecar order changed")
    if protocol["rule"]["price_vectors"] != "all lambda_c in {0, Q_c}; 2^3=8 fixed masks":
        raise ValueError("registered residual-sidecar price vectors changed")
    expected_sidecar_rule = {
        "heap_priority": "(-legacy_v4_upper_bound, virtual_legacy_serial)",
        "residual_bound_role": "safe pruning and final certificate only; never heap priority or tie-break",
        "generation_gate": "legacy_v4_upper_bound > incumbent",
        "pruning_rule": "residual_upper_bound < incumbent",
        "certificate": "max(incumbent, maximum residual upper bound across every surviving frontier entry)",
    }
    if any(protocol["rule"].get(key) != value
           for key, value in expected_sidecar_rule.items()):
        raise ValueError("registered residual-sidecar search rule changed")
    if protocol.get("registration_context") != _registration_context():
        raise ValueError("registered residual-sidecar source outcomes changed")
    if any((ROOT / name).read_text() != text for name, text in sources.items()):
        raise ValueError("registered residual-sidecar sources changed")
    diagnostic, model = _validate_source_diagnostic()
    if (_json(model.snapshot()) != protocol["model"]
            or roster != _registered_state_roster(diagnostic)
            or roster["state_count"] != STATE_COUNT):
        raise ValueError("registered residual-sidecar states or model changed")
    states = roster["states"]
    evaluations = [_evaluate_state(model, index, state)
                   for index, state in enumerate(states)]
    population_summaries = [
        _metrics_for_population(states, evaluations, load, budget)
        for load in LOADS for budget in SOURCE_POLICY_BUDGETS
    ]
    seed_summaries = [
        _metrics_for_population(states, evaluations, load, budget, seed)
        for load in LOADS for seed in SEEDS for budget in SOURCE_POLICY_BUDGETS
    ]
    source_summaries = protocol["source_population_summaries"]
    source_summary_by_id = {
        (row["load"], row["source_budget"]): row for row in source_summaries
    }
    execution_checks = {
        "registered_5003_unique_states_executed_once": (
            len(evaluations) == STATE_COUNT
            and [row["state_index"] for row in evaluations] == list(range(STATE_COUNT))
        ),
        "registered_18_population_cells_reconstruct": (
            len(seed_summaries) == 18
            and all(
                (observed["load"], observed["seed"], observed["source_policy_budget"],
                 observed["frame_count"])
                == (expected["load"], expected["seed"], expected["source_policy_budget"],
                    expected["frame_count"])
                and math.isclose(
                    observed["legacy_visited_time_s"], expected["legacy_visited_time_s"],
                    rel_tol=0.0, abs_tol=1e-8,
                )
                for observed, expected in zip(
                    seed_summaries, protocol["source_population_cells"], strict=True,
                )
            )
        ),
        "registered_six_population_aggregates_reconstruct_source": all(
            _source_summary_matches(
                source_summary_by_id[(row["load"], row["source_policy_budget"])], row,
            ) for row in population_summaries
        ),
        "legacy_v4_option_order_preserved_on_every_state": all(
            row["legacy_v4_order_preserved"] for row in evaluations
        ),
        "sidecar_bound_and_frontier_work_structure_closes_on_every_state": all(
            row["work_structure_checks_pass"] for row in evaluations
        ),
        "continuous_trace_matches_independent_variant_selectors": all(
            row["trace_matches_independent_k26"] and row["trace_matches_independent_k120"]
            for row in evaluations
        ),
        "variant_actions_feasible_and_bounds_valid": all(
            row["selection_and_bound_checks_pass"] for row in evaluations
        ),
        "variant_k120_closes_with_registered_support_on_every_state": all(
            row["variant_k120"]["exact"]
            and row["variant_exact_support_matches_registered_reference"]
            for row in evaluations
        ),
        "candidate_attempts_respect_registered_budgets": all(
            row["variant_k26"]["attempts"] <= SMALL_BUDGET
            and row["variant_k120"]["attempts"] <= EXACT_BUDGET
            for row in evaluations
        ),
        "gap_decomposition_holds_on_every_state": all(
            abs(row["variant_gap_identity_error"]) <= SCORE_TOLERANCE
            for row in evaluations
        ),
        "sidecar_k26_support_is_pointwise_nonworse": all(
            row["support_score_delta_over_legacy_k26"] >= -SCORE_TOLERANCE
            for row in evaluations
        ),
        "sidecar_k26_certificate_is_pointwise_nonworse": all(
            row["upper_bound_delta_over_legacy_k26"] <= SCORE_TOLERANCE
            for row in evaluations
        ),
        "sidecar_k26_gap_is_pointwise_nonworse": all(
            row["certified_gap_delta_over_legacy_k26"] <= SCORE_TOLERANCE
            for row in evaluations
        ),
        "no_arrivals_trajectories_or_randomness_used": True,
        "registered_sources_unchanged": all(
            (ROOT / name).read_text() == text for name, text in sources.items()
        ),
    }
    execution_passed = all(execution_checks.values())
    primary = [
        row for row in population_summaries
        if row["source_policy_budget"] == SMALL_BUDGET
    ]
    sensitivity = [
        row for row in population_summaries
        if row["source_policy_budget"] == EXACT_BUDGET
    ]
    advance_checks = _advance_checks(
        primary, sensitivity, evaluations, execution_passed,
    )
    (
        shadow_advance,
        certificate_or_work_only,
        scientific_decision,
        next_step,
    ) = _decision(advance_checks)
    changed_states = [row for row in evaluations if row["support_outcome"] != "unchanged"]
    retained_changes = [
        row for row in evaluations
        if row["support_outcome"] != "unchanged"
        or row["selected_physical_signature_changed_from_legacy_k26"]
    ]
    report = {
        "experiment": "residual_capacity_binary_price_sidecar_shadow_v1",
        "status": "PASS" if execution_passed else "FAIL",
        "checks": execution_checks,
        "protocol": protocol,
        "reused_registered_state_roster": SOURCE_STATE_ROSTER.name,
        "state_evaluation_count": len(evaluations),
        "strict_bound_activation_state_count": sum(
            row["k26_bound_strictly_activated"] for row in evaluations
        ),
        "safe_pruning_activation_state_count": sum(
            row["k26_safe_pruning_activated"] for row in evaluations
        ),
        "generation_pruning_state_count": sum(
            row["variant_k26"]["safe_pruned_prefixes_on_generation"] > 0
            for row in evaluations
        ),
        "lazy_pop_pruning_state_count": sum(
            row["variant_k26"]["safe_pruned_prefixes_on_pop"] > 0
            for row in evaluations
        ),
        "support_improved_state_count": sum(
            row["support_outcome"] == "improved" for row in evaluations
        ),
        "support_regressed_state_count": sum(
            row["support_outcome"] == "regressed" for row in evaluations
        ),
        "certificate_improved_state_count": sum(
            row["certified_gap_delta_over_legacy_k26"] < -SCORE_TOLERANCE
            for row in evaluations
        ),
        "candidate_attempts_reduced_state_count": sum(
            row["candidate_attempt_delta_over_legacy_k26"] < 0
            for row in evaluations
        ),
        "physical_action_changed_state_count": sum(
            row["selected_physical_signature_changed_from_legacy_k26"]
            for row in evaluations
        ),
        "retained_change_state_count": len(retained_changes),
        "state_changes": [
            {
                key: value for key, value in row.items()
                if key not in ("variant_k120", "work")
            }
            for row in retained_changes
        ],
        "summary_by_load_and_source_policy": population_summaries,
        "summary_by_load_seed_and_source_policy": seed_summaries,
        "changed_state_examples": [
            {
                "state_index": row["state_index"],
                "queue": states[row["state_index"]]["queue"],
                "primary_frame_visits": sum(
                    visit[3]
                    for visit in states[row["state_index"]]["population_visits"]
                    if visit[2] == SMALL_BUDGET
                ),
                "support_outcome": row["support_outcome"],
                "support_score_delta_over_legacy_k26": row["support_score_delta_over_legacy_k26"],
                "legacy_option_indices": states[row["state_index"]]["legacy_k26"]["option_indices"],
                "variant_option_indices": row["variant_k26"]["option_indices"],
            }
            for row in sorted(
                changed_states,
                key=lambda row: -sum(
                    visit[3]
                    for visit in states[row["state_index"]]["population_visits"]
                    if visit[2] == SMALL_BUDGET
                ),
            )[:12]
        ],
        "bound_activation_examples": [
            {
                "state_index": row["state_index"],
                "queue": states[row["state_index"]]["queue"],
                "primary_frame_visits": sum(
                    visit[3]
                    for visit in states[row["state_index"]]["population_visits"]
                    if visit[2] == SMALL_BUDGET
                ),
                "k26_bound_strict_tightening_count": row[
                    "k26_bound_strict_tightening_count"
                ],
                "k26_bound_tightening_sum": row["k26_bound_tightening_sum"],
                "k26_bound_tightening_max": row["k26_bound_tightening_max"],
                "k26_generation_prunes": row["variant_k26"][
                    "safe_pruned_prefixes_on_generation"
                ],
                "k26_lazy_pop_prunes": row["variant_k26"][
                    "safe_pruned_prefixes_on_pop"
                ],
                "upper_bound_delta_over_legacy_k26": row[
                    "upper_bound_delta_over_legacy_k26"
                ],
                "certified_gap_delta_over_legacy_k26": row[
                    "certified_gap_delta_over_legacy_k26"
                ],
                "candidate_attempt_delta_over_legacy_k26": row[
                    "candidate_attempt_delta_over_legacy_k26"
                ],
                "support_outcome": row["support_outcome"],
            }
            for row in sorted(
                (row for row in evaluations if row["k26_bound_strictly_activated"]),
                key=lambda row: -sum(
                    visit[3]
                    for visit in states[row["state_index"]]["population_visits"]
                    if visit[2] == SMALL_BUDGET
                ),
            )[:12]
        ],
        "work_accounting": {
            "continuous_trace": {
                "calls": len(evaluations),
                "price_masks_per_call": 1 << len(model.workloads),
                "candidate_attempts": sum(row["work"]["continuous_trace_candidate_attempts"] for row in evaluations),
                "local_score_evaluations": sum(row["work"]["continuous_trace_local_score_evaluations"] for row in evaluations),
                "bound_option_probes": sum(row["work"]["continuous_trace_bound_option_probes"] for row in evaluations),
                "bound_threshold_evaluations": sum(row["work"]["continuous_trace_bound_threshold_evaluations"] for row in evaluations),
                "upper_bound_calls": sum(row["work"]["continuous_trace_upper_bound_calls"] for row in evaluations),
                "dual_table_option_probes": sum(row["work"]["continuous_trace_dual_table_option_probes"] for row in evaluations),
                "dual_mask_evaluations": sum(row["work"]["continuous_trace_dual_mask_evaluations"] for row in evaluations),
                "dual_mask_sums": sum(row["work"]["continuous_trace_dual_mask_sums"] for row in evaluations),
                "dual_coordinate_products": sum(row["work"]["continuous_trace_dual_coordinate_products"] for row in evaluations),
                "strict_tightening_count": sum(row["work"]["continuous_trace_strict_tightening_count"] for row in evaluations),
                "tightening_sum": sum(row["work"]["continuous_trace_tightening_sum"] for row in evaluations),
                "maximum_tightening": max(row["work"]["continuous_trace_tightening_max"] for row in evaluations),
                "generation_prunes": sum(row["work"]["continuous_trace_generation_prunes"] for row in evaluations),
                "lazy_pop_prunes": sum(row["work"]["continuous_trace_lazy_pop_prunes"] for row in evaluations),
                "legacy_rejected_prefixes": sum(row["work"]["continuous_trace_legacy_rejected_prefixes"] for row in evaluations),
                "frontier_pushes": sum(row["work"]["continuous_trace_frontier_pushes"] for row in evaluations),
                "paid_search_pops": sum(row["work"]["continuous_trace_paid_search_pops"] for row in evaluations),
                "exact_stop_frontier_entries_cleared": sum(row["work"]["continuous_trace_exact_stop_frontier_entries_cleared"] for row in evaluations),
                "virtual_legacy_serial_allocations": sum(row["work"]["continuous_trace_virtual_legacy_serial_allocations"] for row in evaluations),
                "certificate_frontier_scans": sum(row["work"]["continuous_trace_certificate_frontier_scans"] for row in evaluations),
                "certificate_frontier_entries": sum(row["work"]["continuous_trace_certificate_frontier_entries"] for row in evaluations),
                "feasible_candidate_evaluations": sum(row["work"]["continuous_trace_feasible_candidate_evaluations"] for row in evaluations),
                "peak_search_frontier": max(row["work"]["continuous_trace_frontier_peak"] for row in evaluations),
                "physical_signature_evaluations": sum(row["work"]["continuous_trace_physical_signature_evaluations"] for row in evaluations),
                "completion_offset_reads": sum(row["work"]["continuous_trace_completion_offset_reads"] for row in evaluations),
            },
            "independent_selector_validation": {
                "calls": 2 * len(evaluations),
                "price_masks_per_call": 1 << len(model.workloads),
                "candidate_attempts": sum(row["work"]["independent_validation_candidate_attempts"] for row in evaluations),
                "local_score_evaluations": sum(row["work"]["independent_validation_local_score_evaluations"] for row in evaluations),
                "bound_option_probes": sum(row["work"]["independent_validation_bound_option_probes"] for row in evaluations),
                "bound_threshold_evaluations": sum(row["work"]["independent_validation_bound_threshold_evaluations"] for row in evaluations),
                "upper_bound_calls": sum(row["work"]["independent_validation_upper_bound_calls"] for row in evaluations),
                "dual_table_option_probes": sum(row["work"]["independent_validation_dual_table_option_probes"] for row in evaluations),
                "dual_mask_evaluations": sum(row["work"]["independent_validation_dual_mask_evaluations"] for row in evaluations),
                "dual_mask_sums": sum(row["work"]["independent_validation_dual_mask_sums"] for row in evaluations),
                "dual_coordinate_products": sum(row["work"]["independent_validation_dual_coordinate_products"] for row in evaluations),
                "strict_tightening_count": sum(row["work"]["independent_validation_strict_tightening_count"] for row in evaluations),
                "tightening_sum": sum(row["work"]["independent_validation_tightening_sum"] for row in evaluations),
                "maximum_tightening": max(row["work"]["independent_validation_tightening_max"] for row in evaluations),
                "generation_prunes": sum(row["work"]["independent_validation_generation_prunes"] for row in evaluations),
                "lazy_pop_prunes": sum(row["work"]["independent_validation_lazy_pop_prunes"] for row in evaluations),
                "legacy_rejected_prefixes": sum(row["work"]["independent_validation_legacy_rejected_prefixes"] for row in evaluations),
                "frontier_pushes": sum(row["work"]["independent_validation_frontier_pushes"] for row in evaluations),
                "paid_search_pops": sum(row["work"]["independent_validation_paid_search_pops"] for row in evaluations),
                "exact_stop_frontier_entries_cleared": sum(row["work"]["independent_validation_exact_stop_frontier_entries_cleared"] for row in evaluations),
                "virtual_legacy_serial_allocations": sum(row["work"]["independent_validation_virtual_legacy_serial_allocations"] for row in evaluations),
                "certificate_frontier_scans": sum(row["work"]["independent_validation_certificate_frontier_scans"] for row in evaluations),
                "certificate_frontier_entries": sum(row["work"]["independent_validation_certificate_frontier_entries"] for row in evaluations),
                "feasible_candidate_evaluations": sum(row["work"]["independent_validation_feasible_candidate_evaluations"] for row in evaluations),
                "peak_search_frontier": max(row["work"]["independent_validation_frontier_peak"] for row in evaluations),
            },
            "runner_physical_comparison": {
                "signature_evaluations": sum(row["work"]["runner_physical_signature_evaluations"] for row in evaluations),
                "completion_offset_reads": sum(row["work"]["runner_completion_offset_reads"] for row in evaluations),
            },
            "legacy_order_option_key_evaluations": sum(
                row["work"]["legacy_order_option_key_evaluations"] for row in evaluations
            ),
            "arrival_tapes": 0, "trajectory_frames": 0,
            "rng_initializations": 0, "random_draws": 0,
        },
        "shadow_advance_checks": advance_checks,
        "shadow_advance": shadow_advance,
        "certificate_or_work_only": certificate_or_work_only,
        "scientific_decision": scientific_decision,
        "next_step": next_step,
        "limitations": [
            "The state roster and the rule choice both use consumed development evidence; this is a mechanism screen, not independent confirmation.",
            "The sidecar preserves v4 heap priority and can skip prefixes only after the fixed residual bound proves they are strictly below the incumbent.",
            "The eight binary prices are a fixed Lagrangian relaxation and need not attain the tightest continuous dual bound.",
            "Candidate attempts exclude the added dual-bound work, so fewer attempts do not establish lower runtime.",
            "K120 certifies maximum one-frame support only. It neither defines a unique physical tie action nor predicts mean flow or P90.",
            "Legacy visited time is an exposure weight from the source policy and is not the duration of a counterfactual variant trajectory.",
        ],
    }
    if not all((ROOT / name).read_text() == text for name, text in sources.items()):
        raise ValueError("registered residual-sidecar sources changed during execution")
    _write_new(output, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--register", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = register(args.output) if args.register else run_registered(args.output)
    print(json.dumps({
        "status": "REGISTERED" if args.register else result["status"],
        "shadow_advance": None if args.register else result["shadow_advance"],
        "scientific_decision": (
            None if args.register else result["scientific_decision"]
        ),
        "output": str(args.output),
    }))
    raise SystemExit(0 if args.register or result["status"] == "PASS" else 1)
