"""Registered fixed-state screen for two-lane anchor-star candidate generation.

The screen reuses the complete 5,003-state candidate-diagnostic roster. It
compares frozen v4 K26/K120 controls with a selector that pays for all-idle and
homogeneous seeds, then spends its remaining budget directly on complete
directed heterogeneous pairs. K82 covers the full two-lane support family.
No arrivals, trajectories, policy feedback, or random processes run here.
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
from algorithm.anchor_star_frame_candidates import select_anchor_star_frame


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
DEFAULT_OUTPUT = ARTIFACTS / "aligned_hardware_anchor_star_shadow_development_20260911.json"
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
ANCHOR_EXACT_BUDGET = 82
STATE_COUNT = 5003
OPTION_COUNT = 9
LANES = 2
LOCAL_CHOICE_COUNT = OPTION_COUNT + 1
HOMOGENEOUS_SEED_ATTEMPTS = OPTION_COUNT + 1
DIRECTED_HETEROGENEOUS_PAIR_ATTEMPTS = OPTION_COUNT * (OPTION_COUNT - 1)
ORDERED_ACTION_COUNT = LOCAL_CHOICE_COUNT**LANES
SCORE_TOLERANCE = 1e-10
AGGREGATE_TOLERANCE = 1e-12

NEW_SOURCES = (
    "algorithm/anchor_star_frame_candidates.py",
    "algorithm/experiments/aligned_hardware_anchor_star_shadow.py",
)
ROLE = "development fixed-state mechanism screen on consumed states only"
RULE = {
    "name": "two_lane_homogeneous_anchor_one_exchange_stars",
    "scope": "exactly two identical lanes, the static nine-option catalogue, and the currently visible queue",
    "paid_seeds": "all-idle plus one maximal feasible homogeneous frame attempt for each original catalogue option",
    "ordered_options": "reuse the frozen v4 queue-local order: individually unavailable options last, then descending queue-weighted requirements, then original catalogue index",
    "directed_pairs": "after seeds, maintain one cursor per individually feasible anchor; each cursor scans the same ordered catalogue and both heterogeneous orientations pay independently",
    "best_first_rule": "pop the live anchor cursor with largest admissible suffix upper bound; heap ties follow deterministic insertion order induced by original anchor index",
    "free_skips": "an anchor paired with itself is covered by its homogeneous seed, and an individually infeasible second option cannot form a feasible pair; both cursor advances are free",
    "paid_budget": "K counts the ten seed attempts and each constructed or feasibility-tested directed heterogeneous pair at most once",
    "bound": "each cursor upper-bounds all partners in its unvisited ordered suffix using duration-threshold weighted and coordinate maxima capped by the queue remaining after its anchor",
    "selection_tie": "among paid feasible candidates, maximize support, then minimize frame duration, then the padded original-index raw lane tuple with idle ordered last",
    "support_argument": "for two lanes, all-idle, maximal homogeneous seeds, and all directed pairs of distinct individually feasible options represent every support-relevant feasible frame",
    "exact_cap": "10 seed attempts plus 9*8 directed heterogeneous pair attempts equals K82",
    "exact_semantics": "an admissible cursor upper bound at or below incumbent support may close early; exact certifies support only, so K82 duration, raw witness, and physical signature remain diagnostics rather than exactness gates",
    "excluded_claim": "the one-frame support certificate does not establish completion-offset, FIFO-flow, mean-flow, or P90 improvement",
}
REGISTERED_STATE_SCHEMA = base.REGISTERED_STATE_SCHEMA
KNOWN_PAIR_SCHEMA = {
    "identity": "registered state index and exact queue plus baseline and variant FIFO-execution signatures; physical signatures and original-index raw lane tuples are retained separately",
    "physical_signature": ["aggregate service", "frame duration", "sorted completion-offset multiset for each class"],
    "fifo_execution_signature": ["aggregate service", "frame duration", "per-class completion offsets concatenated in raw lane and option-local order"],
    "outcome_use": "the completed quotient transition supplies only a roster of previously tested FIFO executions; a raw-only change stops without another drain only when its FIFO-execution signature is unchanged",
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
    "known_action_veto": "raw-only changes with unchanged FIFO execution and sets whose every FIFO-effective change matches one of the 16 completed quotient-transition executions stop; at least one new FIFO-effective pair is required to advance",
}
STOPPING_RULE = (
    "execute this frozen 5,003-state screen once; an invalid execution gives "
    "INVALID_EXECUTION and any pointwise or population regression gives "
    "REGRESSION_STOP; FIFO-equivalent raw-only changes or exclusively known FIFO executions give "
    "KNOWN_ACTION_STOP; without both strict primary gains give NO_GAIN_STOP; only "
    "a passing screen with both strict gains and at least one new FIFO-effective pair gives "
    "ADVANCE_AUTHORED_TRANSITION"
)
PROHIBITED = (
    "arrival generation, trajectory simulation, policy feedback, new states or seeds, "
    "alternative orders, bounds or budgets, outcome-tuned filtering, confirmation "
    "outcomes, and paper edits"
)
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


def _action_signatures(model, option_indices):
    service = [
        sum(model.options[index].requirements[c] for index in option_indices if index is not None)
        for c in range(len(model.workloads))
    ]
    duration = max((1.0, *(model.options[index].duration for index in option_indices if index is not None)))
    ordered_offsets = []
    reads = 0
    for coordinate in range(len(model.workloads)):
        local = []
        for index in option_indices:
            if index is None:
                continue
            for offset in model.options[index].completion_offsets[coordinate]:
                reads += 1
                local.append(float(offset))
        ordered_offsets.append(local)
    physical = {
        "service": service,
        "duration": float(duration),
        "completion_offset_multisets": [
            sorted(local) for local in ordered_offsets
        ],
    }
    fifo = {
        "service": service,
        "duration": float(duration),
        "ordered_completion_offsets": ordered_offsets,
    }
    return physical, fifo, reads


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
        baseline, baseline_fifo, _ = _action_signatures(
            model, case["legacy_option_indices"]
        )
        variant, variant_fifo, _ = _action_signatures(
            model, case["variant_option_indices"]
        )
        pairs.append({
            "state_index": case["state_index"],
            "queue": case["queue"],
            "baseline_option_indices": case["legacy_option_indices"],
            "variant_option_indices": case["variant_option_indices"],
            "baseline_physical_signature": baseline,
            "variant_physical_signature": variant,
            "baseline_fifo_execution_signature": baseline_fifo,
            "variant_fifo_execution_signature": variant_fifo,
        })
    if [row["state_index"] for row in pairs] != sorted(row["state_index"] for row in pairs):
        raise ValueError("quotient transition case order changed")
    return pairs


def _structure_snapshot(model):
    option_count = len(model.options)
    seed_attempts = option_count + 1
    directed_pairs = option_count * (option_count - 1)
    return {
        "option_count": option_count,
        "lanes": model.lanes,
        "local_choice_count_with_idle": option_count + 1,
        "represented_ordered_action_count": (option_count + 1) ** model.lanes,
        "homogeneous_seed_attempts": seed_attempts,
        "directed_heterogeneous_pair_attempts": directed_pairs,
        "legacy_registered_exact_budget": LEGACY_EXACT_BUDGET,
        "anchor_star_exhaustive_support_candidate_cap": seed_attempts + directed_pairs,
    }


def _structure_is_expected(snapshot):
    return snapshot == {
        "option_count": OPTION_COUNT,
        "lanes": LANES,
        "local_choice_count_with_idle": LOCAL_CHOICE_COUNT,
        "represented_ordered_action_count": ORDERED_ACTION_COUNT,
        "homogeneous_seed_attempts": HOMOGENEOUS_SEED_ATTEMPTS,
        "directed_heterogeneous_pair_attempts": DIRECTED_HETEROGENEOUS_PAIR_ATTEMPTS,
        "legacy_registered_exact_budget": LEGACY_EXACT_BUDGET,
        "anchor_star_exhaustive_support_candidate_cap": ANCHOR_EXACT_BUDGET,
    }


def _realizable(model, queue, selection, signature):
    if (
        len(selection.option_indices) != model.lanes
        or any(
            index is not None and not 0 <= index < len(model.options)
            for index in selection.option_indices
        )
    ):
        return False
    score = sum(
        q * completed
        for q, completed in zip(queue, signature["service"], strict=True)
    ) / signature["duration"]
    return (
        tuple(signature["service"]) == tuple(selection.service)
        and all(
            completed <= visible
            for completed, visible in zip(selection.service, queue, strict=True)
        )
        and math.isclose(
            signature["duration"],
            selection.duration,
            rel_tol=0.0,
            abs_tol=SCORE_TOLERANCE,
        )
        and math.isclose(
            score, selection.score, rel_tol=0.0, abs_tol=SCORE_TOLERANCE
        )
    )


def _work_identities(model, selection, budget):
    option_count = len(model.options)
    dimensions = len(model.workloads)
    thresholds = {1.0, *(max(1.0, option.duration) for option in model.options)}
    ordered = tuple(selection.ordered_option_indices)
    blocked = int(selection.attempts == budget and selection.frontier_remaining > 0)
    return (
        sorted(ordered) == list(range(option_count))
        and len(ordered) == option_count
        and selection.seed_attempts == option_count + 1
        and selection.attempts == selection.seed_attempts + selection.pair_attempts
        and selection.seed_attempts <= selection.attempts <= budget
        and 0 <= selection.pair_attempts <= option_count * (option_count - 1)
        and 0 <= selection.feasible_pair_count <= selection.pair_attempts
        and 0 <= selection.active_anchor_count <= option_count
        and selection.feasible_candidate_count
        == 1 + selection.active_anchor_count + selection.feasible_pair_count
        and selection.candidate_score_evaluations
        == selection.feasible_candidate_count
        and selection.anchor_node_pops
        == selection.pair_attempts
        + selection.free_self_skips
        + selection.free_locally_infeasible_skips
        and selection.upper_bound_calls
        == selection.active_anchor_count
        + selection.anchor_node_pops
        - selection.exhausted_anchor_cursors
        and selection.upper_bound_calls
        == selection.frontier_pushes + selection.bound_rejected_anchor_nodes
        and selection.frontier_pushes
        == selection.anchor_node_pops
        + selection.exact_stop_frontier_entries_cleared
        + selection.frontier_remaining
        and selection.bound_option_probes == len(thresholds) * option_count
        and selection.bound_queue_coordinate_evaluations
        == dimensions * selection.bound_threshold_evaluations
        and selection.local_score_evaluations == option_count
        and 0 <= selection.frontier_remaining <= selection.active_anchor_count
        and 0 <= selection.frontier_peak <= selection.active_anchor_count
        and 0 <= selection.free_self_skips <= selection.active_anchor_count
        and 0 <= selection.exhausted_anchor_cursors <= selection.active_anchor_count
        and 0
        <= selection.exact_stop_frontier_entries_cleared
        <= selection.active_anchor_count
        and selection.budget_blocked_pair_lookups in (0, 1)
        and selection.budget_blocked_pair_lookups == blocked
        and selection.exact == (selection.gap == 0.0)
        and selection.exact == (selection.frontier_remaining == 0)
        and selection.represented_ordered_action_count
        == (option_count + 1) ** model.lanes
        and selection.exhaustive_support_candidate_cap
        == option_count + 1 + option_count * (option_count - 1)
    )


def _pair_record(
    state_index,
    queue,
    legacy_indices,
    variant_indices,
    legacy_signature,
    variant_signature,
    legacy_fifo_signature,
    variant_fifo_signature,
):
    return {
        "state_index": state_index,
        "queue": list(queue),
        "baseline_option_indices": list(legacy_indices),
        "variant_option_indices": list(variant_indices),
        "baseline_physical_signature": legacy_signature,
        "variant_physical_signature": variant_signature,
        "baseline_fifo_execution_signature": legacy_fifo_signature,
        "variant_fifo_execution_signature": variant_fifo_signature,
    }


def _physical_pair(record):
    return {
        key: record[key]
        for key in (
            "state_index",
            "queue",
            "baseline_physical_signature",
            "variant_physical_signature",
        )
    }


def _fifo_pair(record):
    return {
        key: record[key]
        for key in (
            "state_index",
            "queue",
            "baseline_fifo_execution_signature",
            "variant_fifo_execution_signature",
        )
    }


def _evaluate_state(model, known_pairs, state_index, state, source_exact):
    queue = tuple(state["queue"])
    small = select_anchor_star_frame(model.options, queue, model.lanes, SMALL_BUDGET)
    exact = select_anchor_star_frame(model.options, queue, model.lanes, ANCHOR_EXACT_BUDGET)
    legacy = state["legacy_k26"]
    reference = state["support_reference_score"]
    regret = reference - small.score
    slack = small.upper_bound - reference
    gap_error = small.gap - regret - slack
    epsilon_error = small.epsilon - (small.gap / sum(queue) if sum(queue) else 0.0)
    legacy_signature, legacy_fifo, legacy_reads = _action_signatures(
        model, legacy["option_indices"]
    )
    small_signature, small_fifo, small_reads = _action_signatures(
        model, small.option_indices
    )
    exact_signature, exact_fifo, exact_reads = _action_signatures(
        model, exact.option_indices
    )
    pair = _pair_record(
        state_index,
        queue,
        legacy["option_indices"],
        small.option_indices,
        legacy_signature,
        small_signature,
        legacy_fifo,
        small_fifo,
    )
    physical_pair = _physical_pair(pair)
    fifo_pair = _fifo_pair(pair)
    known_physical = physical_pair in [_physical_pair(known) for known in known_pairs]
    known_fifo = fifo_pair in [_fifo_pair(known) for known in known_pairs]
    score_delta = small.score - legacy["score"]
    upper_delta = small.upper_bound - legacy["upper_bound"]
    gap_delta = small.gap - legacy["gap"]
    valid = (
        _realizable(model, queue, small, small_signature)
        and _realizable(model, queue, exact, exact_signature)
        and _work_identities(model, small, SMALL_BUDGET)
        and _work_identities(model, exact, ANCHOR_EXACT_BUDGET)
        and exact.exact
        and exact.frontier_remaining == 0
        and exact.budget_blocked_pair_lookups == 0
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
        "variant_k82": asdict(exact),
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
        "fifo_execution_signature_changed_from_legacy_k26": small_fifo != legacy_fifo,
        "legacy_physical_signature": legacy_signature,
        "variant_physical_signature": small_signature,
        "variant_k82_physical_signature": exact_signature,
        "legacy_fifo_execution_signature": legacy_fifo,
        "variant_fifo_execution_signature": small_fifo,
        "variant_k82_fifo_execution_signature": exact_fifo,
        "known_quotient_physical_pair": known_physical,
        "known_quotient_fifo_pair": known_fifo,
        "known_quotient_physical_and_raw_pair": pair in known_pairs,
        "variant_k82_support_matches_registered_reference": abs(exact.score - reference) <= SCORE_TOLERANCE,
        "variant_k82_selection_matches_legacy_k120": (
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
        "action_signature_completion_offset_reads": legacy_reads + small_reads + exact_reads,
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
        "fifo_execution_signature_changed_frame_fraction": frames_where(lambda _state, evaluation: evaluation["fifo_execution_signature_changed_from_legacy_k26"]) / frame_count,
        "fifo_execution_signature_changed_time_fraction": time_where(lambda _state, evaluation: evaluation["fifo_execution_signature_changed_from_legacy_k26"]) / visited_time,
        "known_quotient_physical_pair_frame_fraction": frames_where(lambda _state, evaluation: evaluation["known_quotient_physical_pair"]) / frame_count,
        "known_quotient_fifo_pair_frame_fraction": frames_where(lambda _state, evaluation: evaluation["known_quotient_fifo_pair"]) / frame_count,
        "known_quotient_physical_and_raw_pair_frame_fraction": frames_where(lambda _state, evaluation: evaluation["known_quotient_physical_and_raw_pair"]) / frame_count,
        "mean_variant_pair_attempts": average(lambda _state, evaluation: evaluation["variant_k26"]["pair_attempts"]),
        "mean_variant_candidate_score_evaluations": average(lambda _state, evaluation: evaluation["variant_k26"]["candidate_score_evaluations"]),
        "mean_variant_upper_bound_calls": average(lambda _state, evaluation: evaluation["variant_k26"]["upper_bound_calls"]),
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
    gain_keys = {
        "at_least_one_primary_load_strictly_improves_support",
        "at_least_one_primary_load_strictly_improves_certificate_or_attempts",
    }
    if not all(value for key, value in checks.items() if key not in gain_keys):
        return "REGRESSION_STOP"

    action_changes = [
        row
        for row in evaluations
        if row["raw_lane_witness_changed_from_legacy_k26"]
        or row["selected_physical_signature_changed_from_legacy_k26"]
    ]
    fifo_effective_changes = [
        row
        for row in action_changes
        if row["fifo_execution_signature_changed_from_legacy_k26"]
    ]
    if action_changes and not fifo_effective_changes:
        return "KNOWN_ACTION_STOP"
    if fifo_effective_changes and all(
        row["known_quotient_fifo_pair"] for row in fifo_effective_changes
    ):
        return "KNOWN_ACTION_STOP"
    if not (
        checks["at_least_one_primary_load_strictly_improves_support"]
        and checks[
            "at_least_one_primary_load_strictly_improves_certificate_or_attempts"
        ]
    ):
        return "NO_GAIN_STOP"
    if any(
        not row["known_quotient_fifo_pair"] for row in fifo_effective_changes
    ):
        return "ADVANCE_AUTHORED_TRANSITION"
    return "NO_GAIN_STOP"


def register(output=DEFAULT_OUTPUT):
    output, protocol_path, sources_path = _paths(output)
    if any(path.exists() for path in (output, protocol_path, sources_path)):
        raise ValueError("anchor-star shadow artifacts already exist")

    # Registration validates and freezes every source and predecessor relation,
    # but deliberately never calls select_anchor_star_frame.
    sources = _source_texts()
    diagnostic, model = _validate_source_diagnostic()
    _validated_roster(diagnostic)
    structure = _structure_snapshot(model)
    if not _structure_is_expected(structure):
        raise ValueError("registered anchor-star 10+72=82 structure changed")
    known_pairs = _known_quotient_pairs(model)
    protocol = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "aligned_hardware_anchor_star_state_shadow_v1",
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
        "anchor_star_exact_budget": ANCHOR_EXACT_BUDGET,
        "state_count": STATE_COUNT,
        "model": model.snapshot(),
        "registered_state_schema": REGISTERED_STATE_SCHEMA,
        "rule": RULE,
        "anchor_star_structure": structure,
        "known_pair_schema": KNOWN_PAIR_SCHEMA,
        "known_quotient_action_pair_roster": known_pairs,
        "source_population_summaries": diagnostic[
            "summary_by_load_and_source_policy"
        ],
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
        raise ValueError("anchor-star shadow result already exists")
    protocol = json.loads(protocol_path.read_text())
    sources = json.loads(sources_path.read_text())
    expected = {
        "experiment": "aligned_hardware_anchor_star_state_shadow_v1",
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
        "anchor_star_exact_budget": ANCHOR_EXACT_BUDGET,
        "state_count": STATE_COUNT,
    }
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise ValueError("registered anchor-star roster or constants changed")
    frozen = {
        "role": ROLE,
        "registered_state_schema": REGISTERED_STATE_SCHEMA,
        "rule": RULE,
        "known_pair_schema": KNOWN_PAIR_SCHEMA,
        "advance_gate": ADVANCE_GATE,
        "stopping_rule": STOPPING_RULE,
        "prohibited_in_this_stage": PROHIBITED,
        "support_reference": SUPPORT_REFERENCE,
    }
    if any(protocol.get(key) != value for key, value in frozen.items()):
        raise ValueError("registered anchor-star protocol rule changed")

    # This complete preflight ends before selector_started and therefore before
    # the first selector call.
    if sources != _source_texts():
        raise ValueError("registered anchor-star sources changed")
    diagnostic, model = _validate_source_diagnostic()
    roster = _validated_roster(diagnostic)
    if protocol.get("model") != _json(model.snapshot()):
        raise ValueError("registered anchor-star model changed")
    structure = _structure_snapshot(model)
    if (
        structure != protocol.get("anchor_star_structure")
        or not _structure_is_expected(structure)
    ):
        raise ValueError("registered anchor-star structure changed")
    known_pairs = _known_quotient_pairs(model)
    if known_pairs != protocol.get("known_quotient_action_pair_roster"):
        raise ValueError("registered known quotient action-pair roster changed")
    if (
        protocol.get("source_population_summaries")
        != diagnostic["summary_by_load_and_source_policy"]
    ):
        raise ValueError("registered anchor-star source summaries changed")
    if protocol.get("source_population_cells") != _source_population_cells(
        diagnostic
    ):
        raise ValueError("registered anchor-star source cells changed")

    states = roster["states"]
    selector_started = time.perf_counter()
    evaluations = [
        _evaluate_state(
            model,
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
        for load in LOADS
        for budget in SOURCE_POLICY_BUDGETS
    ]
    cells = [
        _metrics_for_population(states, evaluations, load, budget, seed)
        for load in LOADS
        for seed in SEEDS
        for budget in SOURCE_POLICY_BUDGETS
    ]
    source_by_id = {
        (row["load"], row["source_budget"]): row
        for row in protocol["source_population_summaries"]
    }
    execution_checks = {
        "registered_5003_unique_states_executed_once": (
            len(evaluations) == STATE_COUNT
            and [row["state_index"] for row in evaluations]
            == list(range(STATE_COUNT))
        ),
        "registered_18_population_cells_reconstruct": (
            len(cells) == 18
            and all(
                (
                    observed["load"],
                    observed["seed"],
                    observed["source_policy_budget"],
                    observed["frame_count"],
                )
                == (
                    expected_cell["load"],
                    expected_cell["seed"],
                    expected_cell["source_policy_budget"],
                    expected_cell["frame_count"],
                )
                and math.isclose(
                    observed["legacy_visited_time_s"],
                    expected_cell["legacy_visited_time_s"],
                    rel_tol=0.0,
                    abs_tol=1e-8,
                )
                for observed, expected_cell in zip(
                    cells, protocol["source_population_cells"], strict=True
                )
            )
        ),
        "registered_six_population_aggregates_reconstruct_source": all(
            _source_summary_matches(
                source_by_id[(row["load"], row["source_policy_budget"])], row
            )
            for row in population
        ),
        "two_lane_100_actions_and_10_plus_72_equals_82_structure_holds": (
            _structure_is_expected(structure)
        ),
        "registered_16_known_quotient_action_pairs_reconstruct": (
            len(known_pairs) == 16
            and known_pairs == protocol["known_quotient_action_pair_roster"]
        ),
        "original_index_bound_and_work_identities_hold": all(
            row["selection_bound_mapping_and_work_checks_pass"]
            for row in evaluations
        ),
        "k82_matches_registered_legacy_k120_support_on_every_state": all(
            row["variant_k82"]["exact"]
            and row["variant_k82_support_matches_registered_reference"]
            for row in evaluations
        ),
        "k26_support_bound_gap_and_epsilon_hold": all(
            abs(row["variant_gap_identity_error"]) <= SCORE_TOLERANCE
            and abs(row["variant_epsilon_error"]) <= SCORE_TOLERANCE
            for row in evaluations
        ),
        "candidate_attempts_respect_k26_and_k82_budgets": all(
            row["variant_k26"]["attempts"] <= SMALL_BUDGET
            and row["variant_k82"]["attempts"] <= ANCHOR_EXACT_BUDGET
            for row in evaluations
        ),
        "no_arrivals_trajectories_or_randomness_used": True,
        "registered_sources_unchanged": all(
            (ROOT / name).read_text() == text for name, text in sources.items()
        ),
    }
    passed = all(execution_checks.values())
    primary = [
        row
        for row in population
        if row["source_policy_budget"] == SMALL_BUDGET
    ]
    sensitivity = [
        row
        for row in population
        if row["source_policy_budget"] == LEGACY_EXACT_BUDGET
    ]
    advance_checks = _advance_checks(
        primary, sensitivity, evaluations, passed
    )
    decision = _decision(advance_checks, evaluations)
    reported = [
        row
        for row in evaluations
        if row["support_outcome"] != "unchanged"
        or row["raw_lane_witness_changed_from_legacy_k26"]
        or row["selected_physical_signature_changed_from_legacy_k26"]
    ]
    physical_changes = [
        row
        for row in evaluations
        if row["selected_physical_signature_changed_from_legacy_k26"]
    ]
    raw_only_changes = [
        row
        for row in evaluations
        if row["raw_lane_witness_changed_from_legacy_k26"]
        and not row["selected_physical_signature_changed_from_legacy_k26"]
    ]
    fifo_effective_changes = [
        row
        for row in evaluations
        if row["fifo_execution_signature_changed_from_legacy_k26"]
    ]
    fifo_equivalent_raw_only_changes = [
        row
        for row in raw_only_changes
        if not row["fifo_execution_signature_changed_from_legacy_k26"]
    ]
    state_completion_offset_reads = sum(
        row["action_signature_completion_offset_reads"]
        for row in evaluations
    )
    known_roster_completion_offset_reads = sum(
        len(offsets)
        for pair in known_pairs
        for signature_key in (
            "baseline_physical_signature",
            "variant_physical_signature",
        )
        for offsets in pair[signature_key]["completion_offset_multisets"]
    )
    report = {
        "experiment": "aligned_hardware_anchor_star_state_shadow_v1",
        "status": "PASS" if passed else "FAIL",
        "checks": execution_checks,
        "protocol": protocol,
        "registered_state_roster": REGISTERED_STATES.name,
        "state_evaluation_count": len(evaluations),
        "reported_state_change_count": len(reported),
        "physical_change_state_count": len(physical_changes),
        "raw_only_change_state_count": len(raw_only_changes),
        "fifo_effective_change_state_count": len(fifo_effective_changes),
        "fifo_equivalent_raw_only_change_state_count": len(
            fifo_equivalent_raw_only_changes
        ),
        "known_quotient_physical_pair_count": sum(
            row["known_quotient_physical_pair"] for row in physical_changes
        ),
        "known_quotient_fifo_pair_count": sum(
            row["known_quotient_fifo_pair"] for row in fifo_effective_changes
        ),
        "known_quotient_physical_and_raw_pair_count": sum(
            row["known_quotient_physical_and_raw_pair"] for row in reported
        ),
        "new_fifo_effective_pair_count": sum(
            not row["known_quotient_fifo_pair"]
            for row in fifo_effective_changes
        ),
        "state_changes": [
            {
                "state_index": row["state_index"],
                "queue": states[row["state_index"]]["queue"],
                "primary_frame_visits": sum(
                    visit[3]
                    for visit in states[row["state_index"]]["population_visits"]
                    if visit[2] == SMALL_BUDGET
                ),
                "support_outcome": row["support_outcome"],
                "support_score_delta_over_legacy_k26": row[
                    "support_score_delta_over_legacy_k26"
                ],
                "upper_bound_delta_over_legacy_k26": row[
                    "upper_bound_delta_over_legacy_k26"
                ],
                "gap_delta_over_legacy_k26": row[
                    "gap_delta_over_legacy_k26"
                ],
                "selected_physical_signature_changed_from_legacy_k26": row[
                    "selected_physical_signature_changed_from_legacy_k26"
                ],
                "raw_lane_witness_changed_from_legacy_k26": row[
                    "raw_lane_witness_changed_from_legacy_k26"
                ],
                "fifo_execution_signature_changed_from_legacy_k26": row[
                    "fifo_execution_signature_changed_from_legacy_k26"
                ],
                "known_quotient_physical_pair": row[
                    "known_quotient_physical_pair"
                ],
                "known_quotient_fifo_pair": row[
                    "known_quotient_fifo_pair"
                ],
                "known_quotient_physical_and_raw_pair": row[
                    "known_quotient_physical_and_raw_pair"
                ],
                "legacy_option_indices": states[row["state_index"]][
                    "legacy_k26"
                ]["option_indices"],
                "variant_option_indices": row["variant_k26"]["option_indices"],
                "legacy_physical_signature": row["legacy_physical_signature"],
                "variant_physical_signature": row["variant_physical_signature"],
                "legacy_fifo_execution_signature": row[
                    "legacy_fifo_execution_signature"
                ],
                "variant_fifo_execution_signature": row[
                    "variant_fifo_execution_signature"
                ],
                "variant_attempts": row["variant_k26"]["attempts"],
                "variant_pair_attempts": row["variant_k26"]["pair_attempts"],
            }
            for row in reported
        ],
        "summary_by_load_and_source_policy": population,
        "summary_by_load_seed_and_source_policy": cells,
        "work_accounting": {
            "selector_calls": 2 * len(evaluations),
            "selector_evaluation_wall_s": selector_wall_s,
            "k26_paid_candidate_attempts": sum(
                row["variant_k26"]["attempts"] for row in evaluations
            ),
            "k82_paid_candidate_attempts": sum(
                row["variant_k82"]["attempts"] for row in evaluations
            ),
            "k82_legacy_k120_raw_selection_match_state_count": sum(
                row["variant_k82_selection_matches_legacy_k120"]
                for row in evaluations
            ),
            "pair_attempts": sum(
                row[checkpoint]["pair_attempts"]
                for row in evaluations
                for checkpoint in ("variant_k26", "variant_k82")
            ),
            "candidate_score_evaluations": sum(
                row[checkpoint]["candidate_score_evaluations"]
                for row in evaluations
                for checkpoint in ("variant_k26", "variant_k82")
            ),
            "upper_bound_calls": sum(
                row[checkpoint]["upper_bound_calls"]
                for row in evaluations
                for checkpoint in ("variant_k26", "variant_k82")
            ),
            "bound_queue_coordinate_evaluations": sum(
                row[checkpoint]["bound_queue_coordinate_evaluations"]
                for row in evaluations
                for checkpoint in ("variant_k26", "variant_k82")
            ),
            "free_self_skips": sum(
                row[checkpoint]["free_self_skips"]
                for row in evaluations
                for checkpoint in ("variant_k26", "variant_k82")
            ),
            "free_locally_infeasible_skips": sum(
                row[checkpoint]["free_locally_infeasible_skips"]
                for row in evaluations
                for checkpoint in ("variant_k26", "variant_k82")
            ),
            "state_action_signature_evaluations": 3 * len(evaluations),
            "known_roster_action_signature_evaluations": 2
            * len(known_pairs),
            "action_signature_evaluations": 3 * len(evaluations)
            + 2 * len(known_pairs),
            "state_completion_offset_reads": state_completion_offset_reads,
            "known_roster_completion_offset_reads": (
                known_roster_completion_offset_reads
            ),
            "completion_offset_reads": state_completion_offset_reads
            + known_roster_completion_offset_reads,
            "arrival_tapes": 0,
            "trajectory_frames": 0,
            "rng_initializations": 0,
            "random_draws": 0,
        },
        "shadow_advance_checks": advance_checks,
        "scientific_decision": decision,
        "development_advance": decision == "ADVANCE_AUTHORED_TRANSITION",
        "next_step": (
            "preregister one authored transition screen for the newly observed FIFO-effective pairs"
            if decision == "ADVANCE_AUTHORED_TRANSITION"
            else "close this anchor-star branch without tuning or trajectories"
        ),
        "limitations": [
            "The fixed states and mechanism rationale use consumed development evidence; this is not independent confirmation.",
            "K82 certifies maximum one-frame support only and is not a flow-time oracle.",
            "Sorted completion-offset multisets do not determine FIFO tail behavior; redundancy is decided with raw-order FIFO-execution signatures.",
            "The quotient transition supplies only an already-tested FIFO-execution pair roster; its outcome effects do not tune this screen.",
            "Legacy visited-time weights are not variant trajectory time.",
        ],
    }
    if sources != _source_texts():
        raise ValueError("registered anchor-star sources changed during execution")
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
