"""Registered fixed-state shadow for the duration-aware v4 search order.

The input roster is the complete set of queue states frozen by the preceding
candidate-search diagnostic.  No arrivals are generated and no selected action
is fed into a simulator.  For each state, one continuous duration-aware search
is traced through K26 and K120, with separate selector calls used only to verify
the trace checkpoints.

This development stage tests one fixed composite search rule.  Reordering the
options also rebuilds the suffix-bound tables and changes heap expansion, so the
result is not interpreted as a causal isolation of sorting from bounding.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from algorithm.duration_tie_frame_candidates import select_duration_tie_frame
from algorithm.experiments.duration_tie_frame_search_trace import (
    trace_duration_tie_frame_search,
)
from algorithm.experiments.aligned_hardware_frame_confirmation import _write_new
from algorithm.experiments.aligned_hardware_frame_model import load_model


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
DEFAULT_OUTPUT = ARTIFACTS / "aligned_hardware_duration_tie_order_shadow_development_20260910.json"
SOURCE_DIAGNOSTIC = ARTIFACTS / "aligned_hardware_candidate_search_diagnostic_development_20260910.json"
SOURCE_FIXED_SOURCES = ARTIFACTS / "aligned_hardware_candidate_search_diagnostic_development_20260910_fixed_sources.json"
LOADS = (0.50, 0.80, 0.95)
SEEDS = (41, 42, 43)
SOURCE_POLICY_BUDGETS = (26, 120)
SMALL_BUDGET = 26
EXACT_BUDGET = 120
STATE_COUNT = 5003
SCORE_TOLERANCE = 1e-10
AGGREGATE_TOLERANCE = 1e-12
NEW_SOURCES = (
    "algorithm/duration_tie_frame_candidates.py",
    "algorithm/experiments/duration_tie_frame_search_trace.py",
    "algorithm/experiments/aligned_hardware_duration_tie_order_shadow.py",
)


def _json(value):
    return json.loads(json.dumps(value))


def _paths(output):
    output = Path(output)
    return (
        output,
        output.with_name(output.stem + "_protocol.json"),
        output.with_name(output.stem + "_registered_states.json"),
        output.with_name(output.stem + "_fixed_sources.json"),
    )


def _write_new_compact(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, separators=(",", ":"))
        stream.write("\n")


def _source_texts():
    sources = json.loads(SOURCE_FIXED_SOURCES.read_text())
    changed = [name for name, text in sources.items() if (ROOT / name).read_text() != text]
    if changed:
        raise ValueError(f"candidate-search diagnostic source changed: {changed}")
    return {**sources, **{name: (ROOT / name).read_text() for name in NEW_SOURCES}}


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
        raise ValueError("duration-order shadow requires the completed registered candidate-search diagnostic")
    model = load_model()
    if _json(model.snapshot()) != protocol["model"]:
        raise ValueError("duration-order shadow model changed from its source diagnostic")
    sources = json.loads(SOURCE_FIXED_SOURCES.read_text())
    if any((ROOT / name).read_text() != text for name, text in sources.items()):
        raise ValueError("duration-order shadow source diagnostic code changed")
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


def _order_diagnostic(model, queues):
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
    durations = [max(1.0, float(option.duration)) for option in model.options]
    legacy = tuple(sorted(range(len(model.options)), key=lambda index: (
        copies[index] == 0, -weights[index], index,
    )))
    variant = tuple(sorted(range(len(model.options)), key=lambda index: (
        copies[index] == 0, -weights[index], durations[index], index,
    )))
    inversions = sum(
        ((copies[left] == 0) == (copies[right] == 0))
        for left in range(len(model.options))
        for right in range(left + 1, len(model.options))
        if weights[left] == weights[right] and durations[left] > durations[right]
    )
    return legacy != variant, inversions


def _evaluate_state(model, state_index, state):
    queues = tuple(state["queue"])
    traced = trace_duration_tie_frame_search(
        model.options, queues, model.lanes,
        small_budget=SMALL_BUDGET, exact_budget=EXACT_BUDGET,
    )
    small_shadow = select_duration_tie_frame(
        model.options, queues, model.lanes, SMALL_BUDGET,
    )
    exact_shadow = select_duration_tie_frame(
        model.options, queues, model.lanes, EXACT_BUDGET,
    )
    if traced.small != small_shadow or traced.exact != exact_shadow:
        raise ValueError(f"duration-order trace checkpoint mismatch at state {state_index}")
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
    order_changed, inversion_count = _order_diagnostic(model, queues)
    score_delta = traced.small.score - legacy["score"]
    exact_support_matches = abs(traced.exact.score - reference_score) <= SCORE_TOLERANCE
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
    )
    return {
        "state_index": state_index,
        "duration_tie_order_changed": order_changed,
        "equal_weight_longer_before_shorter_pair_count": inversion_count,
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
        "equal_support_duration_change_s": (
            traced.small.duration - legacy["duration"]
            if abs(score_delta) <= SCORE_TOLERANCE
            and small_signature != legacy_signature else None
        ),
        "trace_matches_independent_k26": True,
        "trace_matches_independent_k120": True,
        "variant_exact_support_matches_registered_reference": exact_support_matches,
        "selection_and_bound_checks_pass": execution_ok,
        "work": {
            "continuous_trace_candidate_attempts": traced.exact.attempts,
            "continuous_trace_local_score_evaluations": traced.exact.local_score_evaluations,
            "continuous_trace_bound_option_probes": traced.exact.bound_option_probes,
            "continuous_trace_bound_threshold_evaluations": traced.exact.bound_threshold_evaluations,
            "continuous_trace_feasible_candidate_evaluations": traced.exact.feasible_candidate_count,
            "continuous_trace_frontier_peak": traced.exact.frontier_peak,
            "continuous_trace_physical_signature_evaluations": traced.physical_signature_evaluations,
            "continuous_trace_completion_offset_reads": traced.physical_signature_completion_offset_reads,
            "independent_validation_candidate_attempts": small_shadow.attempts + exact_shadow.attempts,
            "independent_validation_local_score_evaluations": small_shadow.local_score_evaluations + exact_shadow.local_score_evaluations,
            "independent_validation_bound_option_probes": small_shadow.bound_option_probes + exact_shadow.bound_option_probes,
            "independent_validation_bound_threshold_evaluations": small_shadow.bound_threshold_evaluations + exact_shadow.bound_threshold_evaluations,
            "independent_validation_feasible_candidate_evaluations": small_shadow.feasible_candidate_count + exact_shadow.feasible_candidate_count,
            "independent_validation_frontier_peak": max(small_shadow.frontier_peak, exact_shadow.frontier_peak),
            "runner_physical_signature_evaluations": 3,
            "runner_completion_offset_reads": legacy_reads + small_reads + exact_reads,
            "order_diagnostic_option_key_evaluations": 2 * len(model.options),
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
        "duration_tie_order_changed_frame_fraction": frames_where(
            lambda state, evaluation: evaluation["duration_tie_order_changed"]
        ) / frame_count,
        "support_improved_frame_count": frames_where(
            lambda state, evaluation: evaluation["support_outcome"] == "improved"
        ),
        "support_regressed_frame_count": frames_where(
            lambda state, evaluation: evaluation["support_outcome"] == "regressed"
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


def _advance_checks(primary_summaries, evaluations, execution_passed):
    return {
        "execution_passed": execution_passed,
        "no_registered_state_has_lower_k26_support": all(
            row["support_outcome"] != "regressed" for row in evaluations
        ),
        "primary_mean_support_regret_nonworse_at_every_load": all(
            row["variant"]["mean_finite_budget_support_regret"]
            <= row["legacy"]["mean_finite_budget_support_regret"] + AGGREGATE_TOLERANCE
            for row in primary_summaries
        ),
        "primary_strict_loss_fraction_nonworse_at_every_load": all(
            row["variant"]["strict_support_loss_frame_count"]
            <= row["legacy"]["strict_support_loss_frame_count"]
            for row in primary_summaries
        ),
        "primary_mean_certified_gap_nonworse_at_every_load": all(
            row["variant"]["mean_certified_gap"]
            <= row["legacy"]["mean_certified_gap"] + AGGREGATE_TOLERANCE
            for row in primary_summaries
        ),
        "primary_exact_certified_fraction_nonworse_at_every_load": all(
            row["variant"]["exact_certified_frame_fraction"] + AGGREGATE_TOLERANCE
            >= row["legacy"]["exact_certified_frame_fraction"]
            for row in primary_summaries
        ),
        "primary_exact_support_fraction_nonworse_at_every_load": all(
            row["variant"]["exact_support_score_frame_fraction"] + AGGREGATE_TOLERANCE
            >= row["legacy"]["exact_support_score_frame_fraction"]
            for row in primary_summaries
        ),
        "at_least_one_primary_load_strictly_improves_support": any(
            row["variant"]["mean_finite_budget_support_regret"]
            < row["legacy"]["mean_finite_budget_support_regret"] - AGGREGATE_TOLERANCE
            or row["variant"]["exact_support_score_frame_fraction"]
            > row["legacy"]["exact_support_score_frame_fraction"] + AGGREGATE_TOLERANCE
            for row in primary_summaries
        ),
    }


def register(output=DEFAULT_OUTPUT):
    output, protocol_path, states_path, sources_path = _paths(output)
    if any(path.exists() for path in (output, protocol_path, states_path, sources_path)):
        raise ValueError("duration-order shadow artifacts already exist")
    diagnostic, model = _validate_source_diagnostic()
    roster = _registered_state_roster(diagnostic)
    protocol = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "duration_aware_equal_weight_state_shadow_v1",
        "role": "development fixed-state mechanism screen on consumed states only",
        "source_experiment": SOURCE_DIAGNOSTIC.name,
        "registered_state_roster": states_path.name,
        "loads": LOADS, "seeds": SEEDS,
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
            "support_reference_score": "registered legacy K120 maximum one-frame support",
            "legacy_first_exact_score_attempt": (
                "first legacy-search paid attempt attaining the registered support score"
            ),
        },
        "rule": {
            "ordered_key": "(seed_copies[i] == 0, -weights[i], effective_duration[i], i)",
            "effective_duration": "max(idle_duration, option.duration)",
            "idle_position": "last",
            "unchanged": "homogeneous seed loop, paid attempts, feasibility, consider tie break, thresholds and admissible suffix-bound formula",
            "interpretation": "duration-aware ordered-search composite because reordered suffix tables also change heap expansion",
        },
        "source_population_summaries": diagnostic["summary_by_load_and_source_policy"],
        "source_population_cells": [
            {
                "load": row["load"], "seed": row["seed"],
                "source_policy_budget": row["source_budget"],
                "frame_count": row["frame_count"],
                "legacy_visited_time_s": row["elapsed_including_drain_s"],
            }
            for row in diagnostic["source_trajectory_replays"]
        ],
        "advance_gate": {
            "all_execution_checks": True,
            "no_registered_state_has_lower_k26_support": True,
            "primary_each_load_mean_support_regret_nonworse": True,
            "primary_each_load_strict_loss_fraction_nonworse": True,
            "primary_each_load_mean_certified_gap_nonworse": True,
            "primary_each_load_exact_certified_fraction_nonworse": True,
            "primary_each_load_exact_support_fraction_nonworse": True,
            "at_least_one_primary_load_strict_support_improvement": True,
        },
        "stopping_rule": "evaluate this one rule once on the registered states; K26-source populations decide advancement and K120-source populations are sensitivity veto only; on failure close the rule without tuning",
        "prohibited_in_this_stage": "arrival generation, trajectory simulation, policy feedback, new seeds, rule variants, confirmation outcomes and paper edits",
        "support_reference": "registered legacy K120 score only; exact means maximum one-frame support and does not define a physical tie or flow oracle",
        "variant_physical_rank": (
            "first_k120_selected_physical_signature_attempt is the first appearance of "
            "the variant K120-selected signature on its own search path; it is neither "
            "a unique physical optimum nor a flow oracle"
        ),
    }
    sources = _source_texts()
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_new(protocol_path, protocol)
    _write_new_compact(states_path, roster)
    _write_new(sources_path, sources)
    return protocol


def run_registered(output=DEFAULT_OUTPUT):
    output, protocol_path, states_path, sources_path = _paths(output)
    if output.exists():
        raise ValueError("duration-order shadow result already exists")
    protocol = json.loads(protocol_path.read_text())
    roster = json.loads(states_path.read_text())
    sources = json.loads(sources_path.read_text())
    expected = {
        "loads": list(LOADS), "seeds": list(SEEDS),
        "source_policy_budgets": list(SOURCE_POLICY_BUDGETS),
        "small_checkpoint_budget": SMALL_BUDGET,
        "exact_checkpoint_budget": EXACT_BUDGET,
        "state_count": STATE_COUNT,
    }
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise ValueError("registered duration-order roster or budgets changed")
    expected_rule = "(seed_copies[i] == 0, -weights[i], effective_duration[i], i)"
    if protocol["rule"]["ordered_key"] != expected_rule:
        raise ValueError("registered duration-order rule changed")
    if any((ROOT / name).read_text() != text for name, text in sources.items()):
        raise ValueError("registered duration-order sources changed")
    diagnostic, model = _validate_source_diagnostic()
    if (_json(model.snapshot()) != protocol["model"]
            or roster != _registered_state_roster(diagnostic)
            or roster["state_count"] != STATE_COUNT):
        raise ValueError("registered duration-order states or model changed")
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
        "duration_order_activates_on_registered_states": any(
            row["duration_tie_order_changed"] for row in evaluations
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
    advance_checks = _advance_checks(primary, evaluations, execution_passed)
    shadow_advance = all(advance_checks.values())
    changed_states = [row for row in evaluations if row["support_outcome"] != "unchanged"]
    retained_changes = [
        row for row in evaluations
        if row["support_outcome"] != "unchanged"
        or row["selected_physical_signature_changed_from_legacy_k26"]
    ]
    report = {
        "experiment": "duration_aware_equal_weight_state_shadow_v1",
        "status": "PASS" if execution_passed else "FAIL",
        "checks": execution_checks,
        "protocol": protocol,
        "registered_state_roster": states_path.name,
        "state_evaluation_count": len(evaluations),
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
        "work_accounting": {
            "continuous_trace": {
                "calls": len(evaluations),
                "candidate_attempts": sum(row["work"]["continuous_trace_candidate_attempts"] for row in evaluations),
                "local_score_evaluations": sum(row["work"]["continuous_trace_local_score_evaluations"] for row in evaluations),
                "bound_option_probes": sum(row["work"]["continuous_trace_bound_option_probes"] for row in evaluations),
                "bound_threshold_evaluations": sum(row["work"]["continuous_trace_bound_threshold_evaluations"] for row in evaluations),
                "feasible_candidate_evaluations": sum(row["work"]["continuous_trace_feasible_candidate_evaluations"] for row in evaluations),
                "peak_search_frontier": max(row["work"]["continuous_trace_frontier_peak"] for row in evaluations),
                "physical_signature_evaluations": sum(row["work"]["continuous_trace_physical_signature_evaluations"] for row in evaluations),
                "completion_offset_reads": sum(row["work"]["continuous_trace_completion_offset_reads"] for row in evaluations),
            },
            "independent_selector_validation": {
                "calls": 2 * len(evaluations),
                "candidate_attempts": sum(row["work"]["independent_validation_candidate_attempts"] for row in evaluations),
                "local_score_evaluations": sum(row["work"]["independent_validation_local_score_evaluations"] for row in evaluations),
                "bound_option_probes": sum(row["work"]["independent_validation_bound_option_probes"] for row in evaluations),
                "bound_threshold_evaluations": sum(row["work"]["independent_validation_bound_threshold_evaluations"] for row in evaluations),
                "feasible_candidate_evaluations": sum(row["work"]["independent_validation_feasible_candidate_evaluations"] for row in evaluations),
                "peak_search_frontier": max(row["work"]["independent_validation_frontier_peak"] for row in evaluations),
            },
            "runner_physical_comparison": {
                "signature_evaluations": sum(row["work"]["runner_physical_signature_evaluations"] for row in evaluations),
                "completion_offset_reads": sum(row["work"]["runner_completion_offset_reads"] for row in evaluations),
            },
            "order_diagnostic_option_key_evaluations": sum(
                row["work"]["order_diagnostic_option_key_evaluations"] for row in evaluations
            ),
            "arrival_tapes": 0, "trajectory_frames": 0,
            "rng_initializations": 0, "random_draws": 0,
        },
        "shadow_advance_checks": advance_checks,
        "shadow_advance": shadow_advance,
        "next_step": (
            "preregister full development trajectories for the frozen duration-aware rule"
            if shadow_advance else
            "close the duration-aware rule without tuning or full trajectories"
        ),
        "limitations": [
            "The state roster and the rule choice both use consumed development evidence; this is a mechanism screen, not independent confirmation.",
            "Reordering also changes suffix-bound tables and heap expansion, so the result concerns the composite search and does not isolate sorting from bounding.",
            "K120 certifies maximum one-frame support only. It neither defines a unique physical tie action nor predicts mean flow or P90.",
            "Legacy visited time is an exposure weight from the source policy and is not the duration of a counterfactual variant trajectory.",
        ],
    }
    if not all((ROOT / name).read_text() == text for name, text in sources.items()):
        raise ValueError("registered duration-order sources changed during execution")
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
        "output": str(args.output),
    }))
    raise SystemExit(0 if args.register or result["status"] == "PASS" else 1)
