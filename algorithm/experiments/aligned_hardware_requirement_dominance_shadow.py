"""Registered fixed-state screen for requirement-duration catalogue reduction.

The screen reuses the complete 5,003-state roster from the candidate-search
trace. It removes only local options whose exact requirement tuple is available
at a strictly shorter effective duration, then runs the frozen v4 search at K26
and at the structural exhaustive cap K99. Returned lane tuples use original
catalogue indices. No arrivals, trajectories, or random processes run here.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

from algorithm.experiments import aligned_hardware_permutation_quotient_shadow as base
from algorithm.experiments.aligned_hardware_frame_model import load_model
from algorithm.requirement_dominance_frame_candidates import (
    reduce_requirement_dominated_options,
    select_requirement_dominance_frame,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
DEFAULT_OUTPUT = ARTIFACTS / "aligned_hardware_requirement_dominance_shadow_development_20260911.json"
SOURCE_DIAGNOSTIC = ARTIFACTS / "aligned_hardware_candidate_search_diagnostic_development_20260910.json"
SOURCE_DIAGNOSTIC_PROTOCOL = ARTIFACTS / "aligned_hardware_candidate_search_diagnostic_development_20260910_protocol.json"
SOURCE_FIXED_SOURCES = ARTIFACTS / "aligned_hardware_candidate_search_diagnostic_development_20260910_fixed_sources.json"
REGISTERED_STATES = ARTIFACTS / "aligned_hardware_duration_tie_order_shadow_development_20260910_registered_states.json"
QUOTIENT_TRANSITION = ARTIFACTS / "aligned_hardware_permutation_quotient_transition_development_20260911.json"
QUOTIENT_TRANSITION_PROTOCOL = ARTIFACTS / "aligned_hardware_permutation_quotient_transition_development_20260911_protocol.json"
QUOTIENT_TRANSITION_FIXED_SOURCES = ARTIFACTS / "aligned_hardware_permutation_quotient_transition_development_20260911_fixed_sources.json"

LOADS = (0.50, 0.80, 0.95)
SEEDS = (41, 42, 43)
SOURCE_POLICY_BUDGETS = (26, 120)
SMALL_BUDGET = 26
LEGACY_EXACT_BUDGET = 120
REDUCED_EXACT_BUDGET = 99
STATE_COUNT = 5003
ORIGINAL_OPTION_COUNT = 9
RETAINED_OPTION_COUNT = 8
ORIGINAL_ORDERED_ACTION_COUNT = 100
REDUCED_ORDERED_ACTION_COUNT = 81
EXPECTED_RETAINED_ORIGINAL_INDICES = (0, 2, 3, 4, 5, 6, 7, 8)
EXPECTED_REMOVED_ORIGINAL_INDICES = (1,)
SCORE_TOLERANCE = 1e-10
AGGREGATE_TOLERANCE = 1e-12

NEW_SOURCES = (
    "algorithm/requirement_dominance_frame_candidates.py",
    "algorithm/experiments/aligned_hardware_requirement_dominance_shadow.py",
)
ROLE = "development fixed-state mechanism screen on consumed states only"
RULE = {
    "name": "strict_same_requirement_effective_duration_reduction",
    "dominance_relation": "remove a non-idle local option only when another option has the exact same integer requirement tuple and a strictly shorter max(idle_duration, duration)",
    "equal_duration_rule": "retain every option tied at the minimum effective duration",
    "retained_order": "preserve original catalogue order and map every returned lane tuple monotonically back to original catalogue indices",
    "budget": "K counts only the reduced v4 idle seed, homogeneous seeds, and lazy candidate construction or feasibility attempts",
    "unchanged": "v4 queue-local ordering among retained options, prefix tree, feasibility, support objective, suffix bound, strict pruning, duration tie, and original raw-index tie order",
    "support_argument": "replacement preserves service and feasibility and cannot increase frame duration, so reduced and original families have the same maximum one-frame support for every queue",
    "excluded_claim": "no completion-offset, FIFO-flow, mean-flow, or P90 dominance is claimed",
    "full_search_cap": "frozen two-lane cap is 9 reduced seeds plus 9 one-lane and 81 two-lane ordered attempts = 99",
}
REGISTERED_STATE_SCHEMA = base.REGISTERED_STATE_SCHEMA
EXACT_REFERENCE_SCHEMA = [
    "score",
    "exact",
]
KNOWN_PAIR_SCHEMA = {
    "identity": "exact queue plus baseline and variant physical signatures; original-index raw lane tuples are stored separately",
    "physical_signature": ["aggregate service", "frame duration", "sorted completion-offset multiset for each class"],
    "outcome_use": "the completed quotient transition identifies previously tested action pairs only; its flow effects do not parameterize this rule or gates",
}
SIX_METRICS = [
    "mean_finite_budget_support_regret",
    "strict_support_loss_frame_count",
    "mean_certified_gap",
    "mean_candidate_attempts",
    "exact_support_score_frame_fraction",
    "exact_certified_frame_fraction",
]
ADVANCE_GATE = {
    "all_execution_checks": True,
    "pointwise_k26_support_nonworse": True,
    "pointwise_k26_upper_bound_nonworse": True,
    "pointwise_k26_gap_nonworse": True,
    "primary_each_load_six_metrics_nonworse": SIX_METRICS,
    "k120_distribution_each_load_same_six_metric_veto": True,
    "at_least_one_primary_load_strict_support_improvement": True,
    "at_least_one_primary_load_strict_certificate_or_attempt_improvement": True,
    "known_quotient_action_pair_veto": "an exact queue plus baseline/variant physical-signature pair already tested by the closed quotient transition cannot advance; raw witnesses remain separate diagnostics",
}
STOPPING_RULE = (
    "execute the fixed 5,003-state screen once; invalid execution precedes all scientific decisions; "
    "any pointwise or population regression gives REGRESSION_STOP; absent both strict-gain conditions "
    "gives NO_GAIN_STOP, except unchanged actions with only certificate or work gain give "
    "EFFICIENCY_ONLY_STOP; any exact previously tested quotient pair gives "
    "KNOWN_QUOTIENT_ACTION_STOP; only otherwise-new changed actions advance"
)
PROHIBITED = "arrival generation, trajectories, policy feedback, new states or seeds, alternative dominance rules or budgets, outcome-tuned filtering, confirmation outcomes, and paper edits"
SUPPORT_REFERENCE = "registered legacy K120 maximum one-frame support; not a unique physical action or a flow-time oracle"


def _json(value):
    return json.loads(json.dumps(value))


def _paths(output):
    output = Path(output)
    return output, output.with_name(output.stem + "_protocol.json"), output.with_name(output.stem + "_fixed_sources.json")


def _write_new(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")


def _read_source_map(path, label):
    sources = json.loads(Path(path).read_text())
    changed = [name for name, text in sources.items() if (ROOT / name).read_text() != text]
    if changed:
        raise ValueError(f"{label} source changed: {changed}")
    return sources


def _source_texts():
    diagnostic = _read_source_map(SOURCE_FIXED_SOURCES, "candidate-search diagnostic")
    transition = _read_source_map(QUOTIENT_TRANSITION_FIXED_SOURCES, "quotient transition")
    conflicts = [name for name in diagnostic.keys() & transition.keys() if diagnostic[name] != transition[name]]
    if conflicts:
        raise ValueError(f"frozen predecessor source maps disagree: {conflicts}")
    return {**diagnostic, **transition, **{name: (ROOT / name).read_text() for name in NEW_SOURCES}}


def _validate_source_diagnostic():
    report, model = base._validate_source_diagnostic()
    registered = json.loads(SOURCE_DIAGNOSTIC_PROTOCOL.read_text())
    if report.get("protocol") != registered:
        raise ValueError("candidate-search result no longer matches its registered protocol")
    return report, model


def _validated_roster(diagnostic):
    return base._validated_roster(diagnostic)


def _source_population_cells(diagnostic):
    return base._source_population_cells(diagnostic)


def _physical_signature(model, option_indices):
    service = [
        sum(model.options[index].requirements[c] for index in option_indices if index is not None)
        for c in range(len(model.workloads))
    ]
    duration = max((1.0, *(model.options[index].duration for index in option_indices if index is not None)))
    offsets = []
    reads = 0
    for coordinate in range(len(model.workloads)):
        local = []
        for index in option_indices:
            if index is None:
                continue
            for offset in model.options[index].completion_offsets[coordinate]:
                reads += 1
                local.append(float(offset))
        offsets.append(sorted(local))
    return {"service": service, "duration": float(duration), "completion_offset_multisets": offsets}, reads


def _known_quotient_pairs(model):
    report = json.loads(QUOTIENT_TRANSITION.read_text())
    protocol = json.loads(QUOTIENT_TRANSITION_PROTOCOL.read_text())
    if (
        report.get("protocol") != protocol
        or report.get("experiment") != "aligned_hardware_permutation_quotient_authored_transition_v1"
        or report.get("status") != "PASS"
        or not all(report.get("checks", {}).values())
        or report.get("development_advance") is not False
        or report.get("scientific_decision") != "LOCAL_FLOW_TRADEOFF_STOP"
        or protocol.get("registered_state_roster") != REGISTERED_STATES.name
        or protocol.get("state_count") != 16
        or protocol.get("budget") != SMALL_BUDGET
        or protocol.get("loads") != list(LOADS)
        or protocol.get("seeds") != list(SEEDS)
        or protocol.get("model") != _json(model.snapshot())
        or len(protocol.get("case_roster", ())) != 16
        or len(report.get("cases", ())) != 16
    ):
        raise ValueError("completed quotient transition is unavailable or changed")
    _read_source_map(QUOTIENT_TRANSITION_FIXED_SOURCES, "quotient transition")
    completed = {row["state_index"]: row for row in report["cases"]}
    pairs = []
    for case in protocol["case_roster"]:
        row = completed.get(case["state_index"])
        if row is None or any(row.get(key) != case[key] for key in ("queue", "legacy_option_indices", "variant_option_indices")):
            raise ValueError("completed quotient transition case roster changed")
        baseline, _ = _physical_signature(model, case["legacy_option_indices"])
        variant, _ = _physical_signature(model, case["variant_option_indices"])
        pairs.append({
            "state_index": case["state_index"],
            "queue": case["queue"],
            "baseline_option_indices": case["legacy_option_indices"],
            "variant_option_indices": case["variant_option_indices"],
            "baseline_physical_signature": baseline,
            "variant_physical_signature": variant,
        })
    if [row["state_index"] for row in pairs] != sorted(row["state_index"] for row in pairs):
        raise ValueError("quotient transition case order changed")
    return pairs


def _reduction_snapshot(model):
    reduction = reduce_requirement_dominated_options(model.options)
    witnesses = []
    for removed_index in reduction.removed_original_indices:
        removed = model.options[removed_index]
        candidates = [
            index for index in reduction.retained_original_indices
            if tuple(model.options[index].requirements) == tuple(removed.requirements)
            and max(1.0, model.options[index].duration) < max(1.0, removed.duration)
        ]
        if not candidates:
            raise ValueError("removed option lacks a retained strict-duration witness")
        kept_index = min(candidates, key=lambda index: (max(1.0, model.options[index].duration), index))
        kept = model.options[kept_index]
        witnesses.append({
            "removed_original_index": removed_index,
            "removed_action_id": removed.action_id,
            "dominating_original_index": kept_index,
            "dominating_action_id": kept.action_id,
            "requirements": list(removed.requirements),
            "removed_effective_duration": max(1.0, removed.duration),
            "dominating_effective_duration": max(1.0, kept.duration),
        })
    choices = reduction.retained_option_count + 1
    exhaustive = reduction.retained_option_count + 1 + sum(choices**depth for depth in range(1, model.lanes + 1))
    return {
        "original_option_count": reduction.original_option_count,
        "retained_option_count": reduction.retained_option_count,
        "removed_option_count": reduction.removed_option_count,
        "retained_original_indices": list(reduction.retained_original_indices),
        "removed_original_indices": list(reduction.removed_original_indices),
        "retained_action_ids": [model.options[i].action_id for i in reduction.retained_original_indices],
        "removed_action_ids": [model.options[i].action_id for i in reduction.removed_original_indices],
        "dominance_witnesses": witnesses,
        "original_ordered_action_count": (reduction.original_option_count + 1) ** model.lanes,
        "reduced_ordered_action_count": choices**model.lanes,
        "legacy_registered_exact_budget": LEGACY_EXACT_BUDGET,
        "reduced_exhaustive_budget": exhaustive,
        "reduction_option_probes": reduction.reduction_option_probes,
        "reduction_requirement_coordinate_reads": reduction.reduction_requirement_coordinate_reads,
        "reduction_duration_comparisons": reduction.reduction_duration_comparisons,
    }


def _reduction_is_expected(snapshot):
    return (
        snapshot["original_option_count"] == ORIGINAL_OPTION_COUNT
        and snapshot["retained_option_count"] == RETAINED_OPTION_COUNT
        and snapshot["removed_option_count"] == 1
        and snapshot["retained_original_indices"] == list(EXPECTED_RETAINED_ORIGINAL_INDICES)
        and snapshot["removed_original_indices"] == list(EXPECTED_REMOVED_ORIGINAL_INDICES)
        and snapshot["original_ordered_action_count"] == ORIGINAL_ORDERED_ACTION_COUNT
        and snapshot["reduced_ordered_action_count"] == REDUCED_ORDERED_ACTION_COUNT
        and snapshot["legacy_registered_exact_budget"] == LEGACY_EXACT_BUDGET
        and snapshot["reduced_exhaustive_budget"] == REDUCED_EXACT_BUDGET
        and len(snapshot["dominance_witnesses"]) == 1
        and snapshot["dominance_witnesses"][0]["removed_original_index"] == 1
        and snapshot["dominance_witnesses"][0]["dominating_original_index"] == 0
    )


def _realizable(model, queue, selection):
    signature, _ = _physical_signature(model, selection.option_indices)
    score = sum(q * d for q, d in zip(queue, signature["service"])) / signature["duration"]
    return (
        len(selection.option_indices) == model.lanes
        and tuple(signature["service"]) == tuple(selection.service)
        and all(d <= q for d, q in zip(selection.service, queue))
        and signature["duration"] == selection.duration
        and score == selection.score
    )


def _mapping_and_work_hold(model, selection, reduction):
    retained = tuple(reduction["retained_original_indices"])
    inverse = {original: local for local, original in enumerate(retained)}
    if any(index is not None and index not in inverse for index in selection.option_indices):
        return False
    local = tuple(None if index is None else inverse[index] for index in selection.option_indices)
    mapped = tuple(None if index is None else retained[index] for index in local)
    thresholds = {1.0, *(max(1.0, model.options[index].duration) for index in retained)}
    return (
        mapped == tuple(selection.option_indices)
        and tuple(sorted(retained)) == retained
        and selection.original_option_count == reduction["original_option_count"]
        and selection.retained_option_count == reduction["retained_option_count"]
        and selection.removed_option_count == reduction["removed_option_count"]
        and tuple(selection.retained_original_indices) == retained
        and tuple(selection.removed_original_indices) == tuple(reduction["removed_original_indices"])
        and selection.seed_attempts == len(retained) + 1
        and selection.seed_attempts <= selection.attempts
        and selection.feasible_candidate_count <= selection.attempts
        and selection.local_score_evaluations == len(retained)
        and selection.bound_option_probes == len(thresholds) * (len(retained) + 1)
        and selection.reduction_option_probes == reduction["reduction_option_probes"]
        and selection.reduction_requirement_coordinate_reads == reduction["reduction_requirement_coordinate_reads"]
        and selection.reduction_duration_comparisons == reduction["reduction_duration_comparisons"]
    )


def _pair_record(state_index, queue, legacy_indices, variant_indices, legacy_signature, variant_signature):
    return {
        "state_index": state_index,
        "queue": list(queue),
        "baseline_option_indices": list(legacy_indices),
        "variant_option_indices": list(variant_indices),
        "baseline_physical_signature": legacy_signature,
        "variant_physical_signature": variant_signature,
    }


def _evaluate_state(model, reduction, known_pairs, state_index, state, source_exact):
    queue = tuple(state["queue"])
    small = select_requirement_dominance_frame(model.options, queue, model.lanes, SMALL_BUDGET)
    exact = select_requirement_dominance_frame(model.options, queue, model.lanes, REDUCED_EXACT_BUDGET)
    legacy = state["legacy_k26"]
    reference = state["support_reference_score"]
    regret = reference - small.score
    slack = small.upper_bound - reference
    gap_error = small.gap - regret - slack
    epsilon_error = small.epsilon - (small.gap / sum(queue) if sum(queue) else 0.0)
    legacy_signature, legacy_reads = _physical_signature(model, legacy["option_indices"])
    small_signature, small_reads = _physical_signature(model, small.option_indices)
    exact_signature, exact_reads = _physical_signature(model, exact.option_indices)
    pair = _pair_record(state_index, queue, legacy["option_indices"], small.option_indices, legacy_signature, small_signature)
    physical_pair = {key: value for key, value in pair.items() if key not in ("baseline_option_indices", "variant_option_indices")}
    known_physical = physical_pair in [
        {key: value for key, value in known.items() if key not in ("baseline_option_indices", "variant_option_indices")}
        for known in known_pairs
    ]
    score_delta = small.score - legacy["score"]
    upper_delta = small.upper_bound - legacy["upper_bound"]
    gap_delta = small.gap - legacy["gap"]
    valid = (
        _realizable(model, queue, small)
        and _realizable(model, queue, exact)
        and _mapping_and_work_hold(model, small, reduction)
        and _mapping_and_work_hold(model, exact, reduction)
        and small.attempts <= SMALL_BUDGET
        and exact.attempts <= REDUCED_EXACT_BUDGET
        and exact.exact
        and abs(exact.upper_bound - exact.score) <= SCORE_TOLERANCE
        and abs(exact.gap) <= SCORE_TOLERANCE
        and abs(exact.score - reference) <= SCORE_TOLERANCE
        and small.score <= reference + SCORE_TOLERANCE
        and reference <= small.upper_bound + SCORE_TOLERANCE
        and abs(gap_error) <= SCORE_TOLERANCE
        and abs(epsilon_error) <= SCORE_TOLERANCE
    )
    return {
        "state_index": state_index,
        "variant_k26": asdict(small),
        "variant_k99": asdict(exact),
        "variant_finite_budget_support_regret": max(0.0, regret),
        "variant_frontier_bound_slack": max(0.0, slack),
        "variant_gap_identity_error": gap_error,
        "variant_epsilon_error": epsilon_error,
        "support_score_delta_over_legacy_k26": score_delta,
        "upper_bound_delta_over_legacy_k26": upper_delta,
        "gap_delta_over_legacy_k26": gap_delta,
        "support_outcome": "improved" if score_delta > SCORE_TOLERANCE else "regressed" if score_delta < -SCORE_TOLERANCE else "unchanged",
        "raw_lane_witness_changed_from_legacy_k26": tuple(small.option_indices) != tuple(legacy["option_indices"]),
        "selected_physical_signature_changed_from_legacy_k26": small_signature != legacy_signature,
        "legacy_physical_signature": legacy_signature,
        "variant_physical_signature": small_signature,
        "variant_k99_physical_signature": exact_signature,
        "known_quotient_physical_pair": known_physical,
        "known_quotient_physical_and_raw_pair": pair in known_pairs,
        "variant_k99_support_matches_registered_reference": abs(exact.score - reference) <= SCORE_TOLERANCE,
        "variant_k99_selection_matches_legacy_k120": (
            tuple(exact.option_indices) == tuple(source_exact["option_indices"])
            and tuple(exact.service) == tuple(source_exact["service"])
            and exact.duration == source_exact["duration"]
            and abs(exact.score - source_exact["score"]) <= SCORE_TOLERANCE
            and abs(exact.upper_bound - source_exact["upper_bound"]) <= SCORE_TOLERANCE
            and abs(exact.gap - source_exact["gap"]) <= SCORE_TOLERANCE
            and abs(exact.epsilon - source_exact["epsilon"]) <= SCORE_TOLERANCE
            and exact.exact is source_exact["exact"]
        ),
        "selection_bound_mapping_and_work_checks_pass": valid,
        "physical_signature_completion_offset_reads": legacy_reads + small_reads + exact_reads,
    }


def _metrics_for_population(states, evaluations, load, source_policy_budget, seed=None):
    weighted = []
    for state, evaluation in zip(states, evaluations, strict=True):
        count, visited_time = base._state_visits(state, load, source_policy_budget, seed)
        if count:
            weighted.append((state, evaluation, count, visited_time))
    frame_count = sum(row[2] for row in weighted)
    visited_time = sum(row[3] for row in weighted)

    def average(function):
        return sum(count * function(state, evaluation) for state, evaluation, count, _ in weighted) / frame_count

    def frames_where(function):
        return sum(count for state, evaluation, count, _ in weighted if function(state, evaluation))

    def time_where(function):
        return sum(elapsed for state, evaluation, _, elapsed in weighted if function(state, evaluation))

    def values(which, state, evaluation):
        if which == "legacy":
            selection = state["legacy_k26"]
            regret = max(0.0, state["support_reference_score"] - selection["score"])
        else:
            selection = evaluation["variant_k26"]
            regret = evaluation["variant_finite_budget_support_regret"]
        return selection, regret, sum(state["queue"])

    def summary(which):
        return {
            "strict_support_loss_frame_count": frames_where(lambda state, evaluation: values(which, state, evaluation)[1] > SCORE_TOLERANCE),
            "exact_support_score_frame_fraction": frames_where(lambda state, evaluation: values(which, state, evaluation)[1] <= SCORE_TOLERANCE) / frame_count,
            "exact_certified_frame_fraction": frames_where(lambda state, evaluation: values(which, state, evaluation)[0]["exact"]) / frame_count,
            "mean_finite_budget_support_regret": average(lambda state, evaluation: values(which, state, evaluation)[1]),
            "mean_finite_budget_support_regret_per_visible_job": average(
                lambda state, evaluation: values(which, state, evaluation)[1] / values(which, state, evaluation)[2]
                if values(which, state, evaluation)[2] else 0.0
            ),
            "mean_certified_gap": average(lambda state, evaluation: values(which, state, evaluation)[0]["gap"]),
            "mean_certified_gap_per_visible_job": average(lambda state, evaluation: values(which, state, evaluation)[0]["epsilon"]),
            "mean_candidate_attempts": average(lambda state, evaluation: values(which, state, evaluation)[0]["attempts"]),
        }

    legacy = summary("legacy")
    variant = summary("variant")
    return {
        "load": load,
        "seed": seed,
        "source_policy_budget": source_policy_budget,
        "population_role": "primary frozen K26-policy states" if source_policy_budget == SMALL_BUDGET else "frozen K120-policy state sensitivity",
        "frame_count": frame_count,
        "legacy_visited_time_s": visited_time,
        "unique_queue_state_count": len(weighted),
        "support_improved_frame_count": frames_where(lambda _state, evaluation: evaluation["support_outcome"] == "improved"),
        "support_regressed_frame_count": frames_where(lambda _state, evaluation: evaluation["support_outcome"] == "regressed"),
        "upper_bound_worsened_frame_count": frames_where(lambda _state, evaluation: evaluation["upper_bound_delta_over_legacy_k26"] > SCORE_TOLERANCE),
        "gap_worsened_frame_count": frames_where(lambda _state, evaluation: evaluation["gap_delta_over_legacy_k26"] > SCORE_TOLERANCE),
        "selected_physical_signature_changed_frame_fraction": frames_where(lambda _state, evaluation: evaluation["selected_physical_signature_changed_from_legacy_k26"]) / frame_count,
        "selected_physical_signature_changed_time_fraction": time_where(lambda _state, evaluation: evaluation["selected_physical_signature_changed_from_legacy_k26"]) / visited_time,
        "raw_lane_witness_changed_frame_fraction": frames_where(lambda _state, evaluation: evaluation["raw_lane_witness_changed_from_legacy_k26"]) / frame_count,
        "raw_lane_witness_changed_time_fraction": time_where(lambda _state, evaluation: evaluation["raw_lane_witness_changed_from_legacy_k26"]) / visited_time,
        "known_quotient_pair_frame_fraction": frames_where(lambda _state, evaluation: evaluation["known_quotient_physical_and_raw_pair"]) / frame_count,
        "mean_variant_bound_option_probes": average(lambda _state, evaluation: evaluation["variant_k26"]["bound_option_probes"]),
        "mean_variant_bound_threshold_evaluations": average(lambda _state, evaluation: evaluation["variant_k26"]["bound_threshold_evaluations"]),
        "legacy": legacy,
        "variant": variant,
        "variant_minus_legacy": {key: variant[key] - legacy[key] for key in variant},
    }


def _source_summary_matches(source, shadow):
    if (
        source["frame_count"] != shadow["frame_count"]
        or source["unique_queue_state_count"] != shadow["unique_queue_state_count"]
        or source["strict_support_loss_frame_count"] != shadow["legacy"]["strict_support_loss_frame_count"]
    ):
        return False
    pairs = (
        (source["exact_score_found_by_k26_frame_fraction"], shadow["legacy"]["exact_support_score_frame_fraction"]),
        (source["k26_exact_certified_frame_fraction"], shadow["legacy"]["exact_certified_frame_fraction"]),
        (source["mean_finite_budget_support_regret"], shadow["legacy"]["mean_finite_budget_support_regret"]),
        (source["mean_finite_budget_support_regret_per_visible_job"], shadow["legacy"]["mean_finite_budget_support_regret_per_visible_job"]),
        (source["mean_certified_gap"], shadow["legacy"]["mean_certified_gap"]),
    )
    return all(math.isclose(left, right, rel_tol=0.0, abs_tol=AGGREGATE_TOLERANCE) for left, right in pairs)


def _six_metric_checks(prefix, summaries):
    return {
        f"{prefix}_mean_support_regret_nonworse_at_every_load": all(row["variant"]["mean_finite_budget_support_regret"] <= row["legacy"]["mean_finite_budget_support_regret"] + AGGREGATE_TOLERANCE for row in summaries),
        f"{prefix}_strict_support_loss_count_nonworse_at_every_load": all(row["variant"]["strict_support_loss_frame_count"] <= row["legacy"]["strict_support_loss_frame_count"] for row in summaries),
        f"{prefix}_mean_certified_gap_nonworse_at_every_load": all(row["variant"]["mean_certified_gap"] <= row["legacy"]["mean_certified_gap"] + AGGREGATE_TOLERANCE for row in summaries),
        f"{prefix}_mean_candidate_attempts_nonworse_at_every_load": all(row["variant"]["mean_candidate_attempts"] <= row["legacy"]["mean_candidate_attempts"] + AGGREGATE_TOLERANCE for row in summaries),
        f"{prefix}_exact_support_fraction_nonworse_at_every_load": all(row["variant"]["exact_support_score_frame_fraction"] + AGGREGATE_TOLERANCE >= row["legacy"]["exact_support_score_frame_fraction"] for row in summaries),
        f"{prefix}_exact_certified_fraction_nonworse_at_every_load": all(row["variant"]["exact_certified_frame_fraction"] + AGGREGATE_TOLERANCE >= row["legacy"]["exact_certified_frame_fraction"] for row in summaries),
    }


def _advance_checks(primary, sensitivity, evaluations, execution_passed):
    checks = {
        "execution_passed": execution_passed,
        "no_registered_state_has_lower_k26_support": all(row["support_score_delta_over_legacy_k26"] >= -SCORE_TOLERANCE for row in evaluations),
        "no_registered_state_has_higher_k26_upper_bound": all(row["upper_bound_delta_over_legacy_k26"] <= SCORE_TOLERANCE for row in evaluations),
        "no_registered_state_has_higher_k26_gap": all(row["gap_delta_over_legacy_k26"] <= SCORE_TOLERANCE for row in evaluations),
    }
    checks.update(_six_metric_checks("primary", primary))
    checks.update(_six_metric_checks("sensitivity", sensitivity))
    checks["at_least_one_primary_load_strictly_improves_support"] = any(
        row["variant"]["mean_finite_budget_support_regret"] < row["legacy"]["mean_finite_budget_support_regret"] - AGGREGATE_TOLERANCE
        or row["variant"]["exact_support_score_frame_fraction"] > row["legacy"]["exact_support_score_frame_fraction"] + AGGREGATE_TOLERANCE
        for row in primary
    )
    checks["at_least_one_primary_load_strictly_improves_certificate_or_attempts"] = any(
        row["variant"]["mean_certified_gap"] < row["legacy"]["mean_certified_gap"] - AGGREGATE_TOLERANCE
        or row["variant"]["exact_certified_frame_fraction"] > row["legacy"]["exact_certified_frame_fraction"] + AGGREGATE_TOLERANCE
        or row["variant"]["mean_candidate_attempts"] < row["legacy"]["mean_candidate_attempts"] - AGGREGATE_TOLERANCE
        for row in primary
    )
    return checks


def _decision(checks, evaluations):
    if not checks["execution_passed"]:
        return "INVALID_EXECUTION"
    gain_keys = {"at_least_one_primary_load_strictly_improves_support", "at_least_one_primary_load_strictly_improves_certificate_or_attempts"}
    if not all(value for key, value in checks.items() if key not in gain_keys):
        return "REGRESSION_STOP"
    support_gain = checks["at_least_one_primary_load_strictly_improves_support"]
    work_gain = checks["at_least_one_primary_load_strictly_improves_certificate_or_attempts"]
    changed = [row for row in evaluations if row["raw_lane_witness_changed_from_legacy_k26"] or row["selected_physical_signature_changed_from_legacy_k26"]]
    if not changed and work_gain:
        return "EFFICIENCY_ONLY_STOP"
    if not support_gain or not work_gain:
        return "NO_GAIN_STOP"
    if any(row["known_quotient_physical_pair"] for row in changed):
        return "KNOWN_QUOTIENT_ACTION_STOP"
    return "ADVANCE_NEW_ACTIONS_TO_AUTHORED_TRANSITION_SCREEN"


def register(output=DEFAULT_OUTPUT):
    output, protocol_path, sources_path = _paths(output)
    if any(path.exists() for path in (output, protocol_path, sources_path)):
        raise ValueError("requirement-dominance shadow artifacts already exist")
    # Registration performs all structural/source checks but never calls the selector.
    sources = _source_texts()
    diagnostic, model = _validate_source_diagnostic()
    _validated_roster(diagnostic)
    reduction = _reduction_snapshot(model)
    if not _reduction_is_expected(reduction):
        raise ValueError("registered 9-to-8 catalogue reduction changed")
    known_pairs = _known_quotient_pairs(model)
    protocol = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "aligned_hardware_requirement_dominance_state_shadow_v1",
        "role": ROLE,
        "source_experiment": SOURCE_DIAGNOSTIC.name,
        "source_protocol": SOURCE_DIAGNOSTIC_PROTOCOL.name,
        "registered_state_roster": REGISTERED_STATES.name,
        "closed_quotient_transition": QUOTIENT_TRANSITION.name,
        "closed_quotient_transition_protocol": QUOTIENT_TRANSITION_PROTOCOL.name,
        "loads": list(LOADS),
        "seeds": list(SEEDS),
        "source_policy_budgets": list(SOURCE_POLICY_BUDGETS),
        "small_checkpoint_budget": SMALL_BUDGET,
        "legacy_exact_reference_budget": LEGACY_EXACT_BUDGET,
        "reduced_exact_budget": REDUCED_EXACT_BUDGET,
        "state_count": STATE_COUNT,
        "model": model.snapshot(),
        "registered_state_schema": REGISTERED_STATE_SCHEMA,
        "legacy_k120_exact_reference_schema": EXACT_REFERENCE_SCHEMA,
        "rule": RULE,
        "catalogue_reduction": reduction,
        "known_pair_schema": KNOWN_PAIR_SCHEMA,
        "known_quotient_physical_pair_roster": known_pairs,
        "source_population_summaries": diagnostic["summary_by_load_and_source_policy"],
        "source_population_cells": _source_population_cells(diagnostic),
        "advance_gate": ADVANCE_GATE,
        "stopping_rule": STOPPING_RULE,
        "prohibited_in_this_stage": PROHIBITED,
        "support_reference": SUPPORT_REFERENCE,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_new(protocol_path, protocol)
    _write_new(sources_path, sources)
    return protocol


def run_registered(output=DEFAULT_OUTPUT):
    output, protocol_path, sources_path = _paths(output)
    if output.exists():
        raise ValueError("requirement-dominance shadow result already exists")
    protocol = json.loads(protocol_path.read_text())
    sources = json.loads(sources_path.read_text())
    expected = {
        "experiment": "aligned_hardware_requirement_dominance_state_shadow_v1",
        "source_experiment": SOURCE_DIAGNOSTIC.name,
        "source_protocol": SOURCE_DIAGNOSTIC_PROTOCOL.name,
        "registered_state_roster": REGISTERED_STATES.name,
        "closed_quotient_transition": QUOTIENT_TRANSITION.name,
        "closed_quotient_transition_protocol": QUOTIENT_TRANSITION_PROTOCOL.name,
        "loads": list(LOADS),
        "seeds": list(SEEDS),
        "source_policy_budgets": list(SOURCE_POLICY_BUDGETS),
        "small_checkpoint_budget": SMALL_BUDGET,
        "legacy_exact_reference_budget": LEGACY_EXACT_BUDGET,
        "reduced_exact_budget": REDUCED_EXACT_BUDGET,
        "state_count": STATE_COUNT,
    }
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise ValueError("registered requirement-dominance roster or constants changed")
    frozen = {
        "role": ROLE,
        "registered_state_schema": REGISTERED_STATE_SCHEMA,
        "legacy_k120_exact_reference_schema": EXACT_REFERENCE_SCHEMA,
        "rule": RULE,
        "known_pair_schema": KNOWN_PAIR_SCHEMA,
        "advance_gate": ADVANCE_GATE,
        "stopping_rule": STOPPING_RULE,
        "prohibited_in_this_stage": PROHIBITED,
        "support_reference": SUPPORT_REFERENCE,
    }
    if any(protocol.get(key) != value for key, value in frozen.items()):
        raise ValueError("registered requirement-dominance protocol rule changed")

    # The complete preflight, including source and predecessor validation, ends
    # before selector_started and therefore before the first selector call.
    if sources != _source_texts():
        raise ValueError("registered requirement-dominance sources changed")
    diagnostic, model = _validate_source_diagnostic()
    roster = _validated_roster(diagnostic)
    if protocol.get("model") != _json(model.snapshot()):
        raise ValueError("registered requirement-dominance model changed")
    reduction = _reduction_snapshot(model)
    if reduction != protocol.get("catalogue_reduction") or not _reduction_is_expected(reduction):
        raise ValueError("registered requirement-dominance reduction changed")
    known_pairs = _known_quotient_pairs(model)
    if known_pairs != protocol.get("known_quotient_physical_pair_roster"):
        raise ValueError("registered known quotient action-pair roster changed")
    if protocol.get("source_population_summaries") != diagnostic["summary_by_load_and_source_policy"]:
        raise ValueError("registered requirement-dominance source summaries changed")
    if protocol.get("source_population_cells") != _source_population_cells(diagnostic):
        raise ValueError("registered requirement-dominance source cells changed")

    states = roster["states"]
    selector_started = time.perf_counter()
    evaluations = [
        _evaluate_state(
            model,
            reduction,
            known_pairs,
            state_index,
            state,
            diagnostic["state_diagnostics"][state_index]["exact"],
        )
        for state_index, state in enumerate(states)
    ]
    selector_wall_s = time.perf_counter() - selector_started
    population = [
        _metrics_for_population(states, evaluations, load, budget)
        for load in LOADS for budget in SOURCE_POLICY_BUDGETS
    ]
    cells = [
        _metrics_for_population(states, evaluations, load, budget, seed)
        for load in LOADS for seed in SEEDS for budget in SOURCE_POLICY_BUDGETS
    ]
    source_by_id = {
        (row["load"], row["source_budget"]): row
        for row in protocol["source_population_summaries"]
    }
    execution_checks = {
        "registered_5003_unique_states_executed_once": len(evaluations) == STATE_COUNT and [row["state_index"] for row in evaluations] == list(range(STATE_COUNT)),
        "registered_18_population_cells_reconstruct": len(cells) == 18 and all(
            (observed["load"], observed["seed"], observed["source_policy_budget"], observed["frame_count"])
            == (expected_cell["load"], expected_cell["seed"], expected_cell["source_policy_budget"], expected_cell["frame_count"])
            and math.isclose(observed["legacy_visited_time_s"], expected_cell["legacy_visited_time_s"], rel_tol=0.0, abs_tol=1e-8)
            for observed, expected_cell in zip(cells, protocol["source_population_cells"], strict=True)
        ),
        "registered_six_population_aggregates_reconstruct_source": all(
            _source_summary_matches(source_by_id[(row["load"], row["source_policy_budget"])], row)
            for row in population
        ),
        "strict_requirement_duration_witness_and_9_to_8_reduction_hold": _reduction_is_expected(reduction),
        "ordered_100_to_81_and_exact_120_to_99_accounting_hold": (
            reduction["original_ordered_action_count"] == ORIGINAL_ORDERED_ACTION_COUNT
            and reduction["reduced_ordered_action_count"] == REDUCED_ORDERED_ACTION_COUNT
            and reduction["legacy_registered_exact_budget"] == LEGACY_EXACT_BUDGET
            and reduction["reduced_exhaustive_budget"] == REDUCED_EXACT_BUDGET
        ),
        "registered_16_known_quotient_action_pairs_reconstruct": len(known_pairs) == 16 and known_pairs == protocol["known_quotient_physical_pair_roster"],
        "original_index_raw_mapping_bound_and_work_identities_hold": all(row["selection_bound_mapping_and_work_checks_pass"] for row in evaluations),
        "k99_matches_registered_legacy_k120_support_on_every_state": all(
            row["variant_k99"]["exact"]
            and row["variant_k99_support_matches_registered_reference"]
            for row in evaluations
        ),
        "k26_support_bound_gap_and_epsilon_hold": all(abs(row["variant_gap_identity_error"]) <= SCORE_TOLERANCE and abs(row["variant_epsilon_error"]) <= SCORE_TOLERANCE for row in evaluations),
        "candidate_attempts_respect_k26_and_k99_budgets": all(row["variant_k26"]["attempts"] <= SMALL_BUDGET and row["variant_k99"]["attempts"] <= REDUCED_EXACT_BUDGET for row in evaluations),
        "no_arrivals_trajectories_or_randomness_used": True,
        "registered_sources_unchanged": all((ROOT / name).read_text() == text for name, text in sources.items()),
    }
    passed = all(execution_checks.values())
    primary = [row for row in population if row["source_policy_budget"] == SMALL_BUDGET]
    sensitivity = [row for row in population if row["source_policy_budget"] == LEGACY_EXACT_BUDGET]
    advance_checks = _advance_checks(primary, sensitivity, evaluations, passed)
    decision = _decision(advance_checks, evaluations)
    retained = [
        row for row in evaluations
        if row["support_outcome"] != "unchanged"
        or row["raw_lane_witness_changed_from_legacy_k26"]
        or row["selected_physical_signature_changed_from_legacy_k26"]
    ]
    changed = [
        row for row in retained
        if row["raw_lane_witness_changed_from_legacy_k26"]
        or row["selected_physical_signature_changed_from_legacy_k26"]
    ]
    report = {
        "experiment": "aligned_hardware_requirement_dominance_state_shadow_v1",
        "status": "PASS" if passed else "FAIL",
        "checks": execution_checks,
        "protocol": protocol,
        "registered_state_roster": REGISTERED_STATES.name,
        "state_evaluation_count": len(evaluations),
        "retained_change_state_count": len(retained),
        "changed_action_state_count": len(changed),
        "known_quotient_physical_pair_count": sum(row["known_quotient_physical_pair"] for row in changed),
        "known_quotient_physical_and_raw_pair_count": sum(row["known_quotient_physical_and_raw_pair"] for row in changed),
        "state_changes": [
            {
                "state_index": row["state_index"],
                "queue": states[row["state_index"]]["queue"],
                "primary_frame_visits": sum(visit[3] for visit in states[row["state_index"]]["population_visits"] if visit[2] == SMALL_BUDGET),
                "support_outcome": row["support_outcome"],
                "support_score_delta_over_legacy_k26": row["support_score_delta_over_legacy_k26"],
                "upper_bound_delta_over_legacy_k26": row["upper_bound_delta_over_legacy_k26"],
                "gap_delta_over_legacy_k26": row["gap_delta_over_legacy_k26"],
                "selected_physical_signature_changed_from_legacy_k26": row["selected_physical_signature_changed_from_legacy_k26"],
                "raw_lane_witness_changed_from_legacy_k26": row["raw_lane_witness_changed_from_legacy_k26"],
                "known_quotient_physical_pair": row["known_quotient_physical_pair"],
                "known_quotient_physical_and_raw_pair": row["known_quotient_physical_and_raw_pair"],
                "legacy_option_indices": states[row["state_index"]]["legacy_k26"]["option_indices"],
                "variant_option_indices": row["variant_k26"]["option_indices"],
                "legacy_physical_signature": row["legacy_physical_signature"],
                "variant_physical_signature": row["variant_physical_signature"],
                "variant_attempts": row["variant_k26"]["attempts"],
            }
            for row in retained
        ],
        "summary_by_load_and_source_policy": population,
        "summary_by_load_seed_and_source_policy": cells,
        "work_accounting": {
            "selector_calls": 2 * len(evaluations),
            "selector_evaluation_wall_s": selector_wall_s,
            "k26_paid_candidate_attempts": sum(row["variant_k26"]["attempts"] for row in evaluations),
            "k99_paid_candidate_attempts": sum(row["variant_k99"]["attempts"] for row in evaluations),
            "k99_legacy_k120_raw_selection_match_state_count": sum(
                row["variant_k99_selection_matches_legacy_k120"]
                for row in evaluations
            ),
            "local_score_evaluations": sum(row[checkpoint]["local_score_evaluations"] for row in evaluations for checkpoint in ("variant_k26", "variant_k99")),
            "bound_option_probes": sum(row[checkpoint]["bound_option_probes"] for row in evaluations for checkpoint in ("variant_k26", "variant_k99")),
            "bound_threshold_evaluations": sum(row[checkpoint]["bound_threshold_evaluations"] for row in evaluations for checkpoint in ("variant_k26", "variant_k99")),
            "reduction_option_probes": sum(row[checkpoint]["reduction_option_probes"] for row in evaluations for checkpoint in ("variant_k26", "variant_k99")),
            "reduction_requirement_coordinate_reads": sum(row[checkpoint]["reduction_requirement_coordinate_reads"] for row in evaluations for checkpoint in ("variant_k26", "variant_k99")),
            "reduction_duration_comparisons": sum(row[checkpoint]["reduction_duration_comparisons"] for row in evaluations for checkpoint in ("variant_k26", "variant_k99")),
            "physical_signature_evaluations": 3 * len(evaluations),
            "completion_offset_reads": sum(row["physical_signature_completion_offset_reads"] for row in evaluations),
            "arrival_tapes": 0,
            "trajectory_frames": 0,
            "rng_initializations": 0,
            "random_draws": 0,
        },
        "shadow_advance_checks": advance_checks,
        "scientific_decision": decision,
        "development_advance": decision == "ADVANCE_NEW_ACTIONS_TO_AUTHORED_TRANSITION_SCREEN",
        "next_step": "preregister one authored transition screen for the new changed actions" if decision == "ADVANCE_NEW_ACTIONS_TO_AUTHORED_TRANSITION_SCREEN" else "close this reduction branch without tuning or trajectories",
        "limitations": [
            "The registered states and mechanism rationale use consumed development evidence; this is not independent confirmation.",
            "Strict same-requirement duration dominance preserves maximum support, not completion offsets, mean flow, or P90.",
            "K99 certifies maximum one-frame support only and is not a flow-time oracle.",
            "The quotient transition contributes only an already-tested action-pair roster; its flow effects do not tune this rule or gates.",
            "Legacy visited-time weights are not variant trajectory time.",
        ],
    }
    if sources != _source_texts():
        raise ValueError("registered requirement-dominance sources changed during execution")
    _write_new(output, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--register", action="store_true")
    mode.add_argument("--run-registered", action="store_true")
    args = parser.parse_args()
    result = register(args.output) if args.register else run_registered(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.run_registered and result["status"] != "PASS":
        raise SystemExit(1)
