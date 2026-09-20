"""Registered fixed-state screen for lane-permutation quotient caching.

The screen reuses the complete 5,003-state roster from the candidate-search
diagnostic.  It compares the frozen v4 K26 result with one fixed selector that
keeps the v4 logical heap and bounds but charges each lane-option multiset only
once.  No arrival, trajectory, or random process is executed here.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

from algorithm.experiments.aligned_hardware_frame_model import load_model
from algorithm.permutation_quotient_frame_candidates import (
    select_permutation_quotient_frame,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
DEFAULT_OUTPUT = ARTIFACTS / "aligned_hardware_permutation_quotient_shadow_development_20260911.json"
SOURCE_DIAGNOSTIC = ARTIFACTS / "aligned_hardware_candidate_search_diagnostic_development_20260910.json"
SOURCE_FIXED_SOURCES = ARTIFACTS / "aligned_hardware_candidate_search_diagnostic_development_20260910_fixed_sources.json"
REGISTERED_STATES = ARTIFACTS / "aligned_hardware_duration_tie_order_shadow_development_20260910_registered_states.json"
CLOSED_SIDECAR_RESULT = ARTIFACTS / "aligned_hardware_residual_sidecar_trajectories_development_20260911.json"
LOADS = (0.50, 0.80, 0.95)
SEEDS = (41, 42, 43)
SOURCE_POLICY_BUDGETS = (26, 120)
SMALL_BUDGET = 26
EXACT_QUOTIENT_BUDGET = 55
STATE_COUNT = 5003
ORDERED_ACTION_COUNT = 100
QUOTIENT_ORBIT_COUNT = 55
SCORE_TOLERANCE = 1e-10
AGGREGATE_TOLERANCE = 1e-12
NEW_SOURCES = (
    "algorithm/permutation_quotient_frame_candidates.py",
    "algorithm/experiments/aligned_hardware_permutation_quotient_shadow.py",
)

RULE = {
    "name": "stable_lane_permutation_candidate_cache",
    "equivalence_key": (
        "sort the padded tuple of original option indices ascending with None last"
    ),
    "paid_budget": (
        "all-idle and one homogeneous seed per option, then the first logical "
        "search occurrence of each previously unseen equivalence key"
    ),
    "free_reuse": (
        "reuse feasibility, service, numerator, duration, and score for later "
        "lane permutations; still expand the original logical prefix"
    ),
    "post_budget_drain": (
        "after K paid attempts, process consecutive frontier cache hits until "
        "the next cache miss or exact termination"
    ),
    "raw_witness": (
        "apply each encountered ordered tuple to the unchanged v4 "
        "score-duration-raw-index tie break"
    ),
    "unchanged": (
        "homogeneous seed order, queue-dependent local order, logical prefix "
        "tree, feasibility, support objective, suffix bound, and strict bound pruning"
    ),
    "cache_key_excludes": "service-duration aliases and completion-offset aliases",
    "exact_cap": "binomial(number_of_nonidle_options + lanes, lanes)",
}

ADVANCE_GATE = {
    "all_execution_checks": True,
    "pointwise_k26_support_nonworse": True,
    "pointwise_k26_upper_bound_nonworse": True,
    "pointwise_k26_gap_nonworse": True,
    "primary_each_load_six_metrics_nonworse": [
        "mean_finite_budget_support_regret",
        "strict_support_loss_frame_count",
        "mean_certified_gap",
        "mean_candidate_attempts",
        "exact_support_score_frame_fraction",
        "exact_certified_frame_fraction",
    ],
    "k120_distribution_each_load_same_six_metric_veto": True,
    "at_least_one_primary_load_strict_support_improvement": True,
    "at_least_one_primary_load_strict_certificate_or_attempt_improvement": True,
}
STOPPING_RULE = (
    "run this one fixed quotient-cache rule once on the registered 5,003 states; "
    "advance only if every pointwise and population gate passes; if only certificate "
    "or attempt work improves without support, raw-lane, or physical-action change, stop as "
    "efficiency-only; otherwise close without tuning K, order, bound, or roster"
)
PROHIBITED = (
    "arrival generation, trajectory simulation, policy feedback, new seeds, "
    "rule variants, confirmation outcomes, and paper edits"
)
SUPPORT_REFERENCE = (
    "registered legacy K120 maximum one-frame support; it is not a unique "
    "physical action or a flow-time oracle"
)
ROLE = "development fixed-state mechanism screen on consumed states only"
REGISTERED_STATE_SCHEMA = {
    "population_visit": [
        "load",
        "seed",
        "source_policy_budget",
        "frame_count",
        "legacy_visited_time_s",
    ],
    "legacy_k26": [
        "option_indices",
        "service",
        "duration",
        "score",
        "upper_bound",
        "gap",
        "epsilon",
        "attempts",
        "exact",
    ],
    "support_reference_score": "registered legacy K120 maximum one-frame support",
}


def _json(value):
    return json.loads(json.dumps(value))


def _paths(output):
    output = Path(output)
    return (
        output,
        output.with_name(output.stem + "_protocol.json"),
        output.with_name(output.stem + "_fixed_sources.json"),
    )


def _write_new(path, data, *, compact=False):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, separators=(",", ":") if compact else None, indent=None if compact else 2)
        stream.write("\n")


def _source_texts():
    sources = json.loads(SOURCE_FIXED_SOURCES.read_text())
    changed = [
        name for name, text in sources.items() if (ROOT / name).read_text() != text
    ]
    if changed:
        raise ValueError(f"candidate-search diagnostic source changed: {changed}")
    return {
        **sources,
        **{name: (ROOT / name).read_text() for name in NEW_SOURCES},
    }


def _validate_source_diagnostic():
    report = json.loads(SOURCE_DIAGNOSTIC.read_text())
    protocol = report["protocol"]
    expected_population_ids = [
        (load, budget) for load in LOADS for budget in SOURCE_POLICY_BUDGETS
    ]
    if (
        report["experiment"] != "aligned_hardware_candidate_search_diagnostic_v1"
        or report["status"] != "PASS"
        or not all(report["checks"].values())
        or report["development_advance"]
        or protocol["loads"] != list(LOADS)
        or protocol["seeds"] != list(SEEDS)
        or protocol["source_budgets"] != list(SOURCE_POLICY_BUDGETS)
        or protocol["small_budget"] != SMALL_BUDGET
        or protocol["exact_budget"] != 120
        or len(report["source_trajectory_replays"]) != 18
        or len(report["state_diagnostics"]) != STATE_COUNT
        or len({tuple(row["queue"]) for row in report["state_diagnostics"]})
        != STATE_COUNT
        or [
            (row["load"], row["source_budget"])
            for row in report["summary_by_load_and_source_policy"]
        ]
        != expected_population_ids
    ):
        raise ValueError(
            "permutation-quotient shadow requires the completed candidate-search diagnostic"
        )
    model = load_model()
    if _json(model.snapshot()) != protocol["model"]:
        raise ValueError("permutation-quotient model changed from its source diagnostic")
    sources = json.loads(SOURCE_FIXED_SOURCES.read_text())
    if any((ROOT / name).read_text() != text for name, text in sources.items()):
        raise ValueError("permutation-quotient source diagnostic code changed")
    return report, model


def _validate_closed_sidecar():
    report = json.loads(CLOSED_SIDECAR_RESULT.read_text())
    if (
        report.get("experiment")
        != "aligned_residual_sidecar_full_development_trajectories_v1"
        or report.get("status") != "PASS"
        or not all(report.get("checks", {}).values())
        or report.get("development_advance") is not False
        or report.get("scientific_decision") != "TRAJECTORY_TRADEOFF_STOP"
    ):
        raise ValueError("residual-sidecar trajectory branch is not closed")


def _compact_state(row):
    legacy_fields = (
        "option_indices",
        "service",
        "duration",
        "score",
        "upper_bound",
        "gap",
        "epsilon",
        "attempts",
        "exact",
    )
    return {
        "queue": row["queue"],
        "population_visits": [
            [
                visit["load"],
                visit["seed"],
                visit["source_budget"],
                visit["frame_count"],
                visit["visited_time_s"],
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


def _source_population_cells(diagnostic):
    return [
        {
            "load": row["load"],
            "seed": row["seed"],
            "source_policy_budget": row["source_budget"],
            "frame_count": row["frame_count"],
            "legacy_visited_time_s": row["elapsed_including_drain_s"],
        }
        for row in diagnostic["source_trajectory_replays"]
    ]


def _validated_roster(diagnostic):
    roster = json.loads(REGISTERED_STATES.read_text())
    if roster != _registered_state_roster(diagnostic):
        raise ValueError("registered permutation-quotient states changed")
    return roster


def _state_visits(state, load, source_policy_budget, seed=None):
    matching = [
        row
        for row in state["population_visits"]
        if row[0] == load
        and row[2] == source_policy_budget
        and (seed is None or row[1] == seed)
    ]
    return sum(row[3] for row in matching), sum(row[4] for row in matching)


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
            for index in selected.option_indices
            if index is not None
        )
        for coordinate in range(len(queues))
    )
    duration = max(
        (
            1.0,
            *(
                model.options[index].duration
                for index in selected.option_indices
                if index is not None
            ),
        )
    )
    score = sum(q * count for q, count in zip(queues, service)) / duration
    return (
        len(selected.option_indices) == model.lanes
        and service == selected.service
        and all(count <= q for count, q in zip(service, queues))
        and duration == selected.duration
        and score == selected.score
    )


def _work_identities(selection, option_count, lanes):
    return (
        selection.seed_attempts == option_count + 1
        and selection.attempts
        == selection.seed_attempts + selection.paid_search_pops
        and selection.cache_entries == selection.attempts
        and selection.cache_misses_after_seeds == selection.paid_search_pops
        and selection.cache_hits == selection.free_cache_hit_pops
        and selection.logical_search_pops
        == selection.paid_search_pops + selection.free_cache_hit_pops
        and selection.cached_feasible_reuses
        + selection.cached_infeasible_reuses
        == selection.free_cache_hit_pops
        and selection.candidate_score_evaluations
        == selection.feasible_candidate_count
        and selection.cache_key_evaluations
        == selection.seed_attempts
        + selection.logical_search_pops
        + selection.budget_blocked_cache_miss_lookups
        and selection.cache_key_item_reads
        == lanes * selection.cache_key_evaluations
        and selection.upper_bound_calls
        == selection.frontier_pushes + selection.bound_rejected_prefixes
        and selection.frontier_pushes
        == selection.logical_search_pops
        + selection.exact_stop_frontier_entries_cleared
        + selection.frontier_entries_remaining
        and selection.represented_ordered_configurations
        == (option_count + 1) ** lanes
        and selection.represented_lane_multisets
        == math.comb(option_count + lanes, lanes)
        and selection.symmetry_duplicate_ordered_configurations
        == selection.represented_ordered_configurations
        - selection.represented_lane_multisets
        and selection.cache_entries <= selection.represented_lane_multisets
        and selection.budget_blocked_cache_miss_lookups in (0, 1)
    )


def _evaluate_state(model, state_index, state):
    queues = tuple(state["queue"])
    small = select_permutation_quotient_frame(
        model.options, queues, model.lanes, SMALL_BUDGET
    )
    exact = select_permutation_quotient_frame(
        model.options, queues, model.lanes, EXACT_QUOTIENT_BUDGET
    )
    reference_score = state["support_reference_score"]
    legacy = state["legacy_k26"]
    support_regret = reference_score - small.score
    bound_slack = small.upper_bound - reference_score
    identity_error = small.gap - support_regret - bound_slack
    epsilon_error = small.epsilon - (
        small.gap / sum(queues) if sum(queues) else 0.0
    )
    legacy_signature, legacy_reads = _physical_signature(
        model,
        legacy["option_indices"],
        legacy["service"],
        legacy["duration"],
    )
    small_signature, small_reads = _physical_signature(
        model, small.option_indices, small.service, small.duration
    )
    exact_signature, exact_reads = _physical_signature(
        model, exact.option_indices, exact.service, exact.duration
    )
    score_delta = small.score - legacy["score"]
    upper_delta = small.upper_bound - legacy["upper_bound"]
    gap_delta = small.gap - legacy["gap"]
    execution_ok = (
        _realizable(model, queues, small)
        and _realizable(model, queues, exact)
        and _work_identities(small, len(model.options), model.lanes)
        and _work_identities(exact, len(model.options), model.lanes)
        and small.attempts <= SMALL_BUDGET
        and exact.attempts <= EXACT_QUOTIENT_BUDGET
        and small.represented_ordered_configurations == ORDERED_ACTION_COUNT
        and small.represented_lane_multisets == QUOTIENT_ORBIT_COUNT
        and exact.represented_ordered_configurations == ORDERED_ACTION_COUNT
        and exact.represented_lane_multisets == QUOTIENT_ORBIT_COUNT
        and exact.exact
        and exact.frontier_entries_remaining == 0
        and exact.budget_blocked_cache_miss_lookups == 0
        and abs(exact.score - reference_score) <= SCORE_TOLERANCE
        and small.score <= reference_score + SCORE_TOLERANCE
        and reference_score <= small.upper_bound + SCORE_TOLERANCE
        and abs(identity_error) <= SCORE_TOLERANCE
        and abs(epsilon_error) <= SCORE_TOLERANCE
    )
    return {
        "state_index": state_index,
        "variant_k26": asdict(small),
        "variant_k55": asdict(exact),
        "variant_finite_budget_support_regret": max(0.0, support_regret),
        "variant_frontier_bound_slack": max(0.0, bound_slack),
        "variant_gap_identity_error": identity_error,
        "variant_epsilon_error": epsilon_error,
        "support_score_delta_over_legacy_k26": score_delta,
        "upper_bound_delta_over_legacy_k26": upper_delta,
        "gap_delta_over_legacy_k26": gap_delta,
        "support_outcome": (
            "improved"
            if score_delta > SCORE_TOLERANCE
            else "regressed"
            if score_delta < -SCORE_TOLERANCE
            else "unchanged"
        ),
        "raw_lane_witness_changed_from_legacy_k26": (
            tuple(small.option_indices) != tuple(legacy["option_indices"])
        ),
        "selected_physical_signature_changed_from_legacy_k26": (
            small_signature != legacy_signature
        ),
        "variant_k26_matches_k55_physical_signature": (
            small_signature == exact_signature
        ),
        "variant_k55_support_matches_registered_reference": (
            abs(exact.score - reference_score) <= SCORE_TOLERANCE
        ),
        "selection_bound_and_work_checks_pass": execution_ok,
        "physical_signature_completion_offset_reads": (
            legacy_reads + small_reads + exact_reads
        ),
    }


def _metrics_for_population(states, evaluations, load, source_policy_budget, seed=None):
    weighted = []
    for state, evaluation in zip(states, evaluations, strict=True):
        count, visited_time = _state_visits(
            state, load, source_policy_budget, seed
        )
        if count:
            weighted.append((state, evaluation, count, visited_time))
    frame_count = sum(row[2] for row in weighted)
    visited_time = sum(row[3] for row in weighted)

    def average(function):
        return sum(
            count * function(state, evaluation)
            for state, evaluation, count, _ in weighted
        ) / frame_count

    def frames_where(function):
        return sum(
            count
            for state, evaluation, count, _ in weighted
            if function(state, evaluation)
        )

    def time_where(function):
        return sum(
            time
            for state, evaluation, _, time in weighted
            if function(state, evaluation)
        )

    def values(which, state, evaluation):
        if which == "legacy":
            selection = state["legacy_k26"]
            regret = max(
                0.0, state["support_reference_score"] - selection["score"]
            )
        else:
            selection = evaluation["variant_k26"]
            regret = evaluation["variant_finite_budget_support_regret"]
        return selection, regret, sum(state["queue"])

    def summary(which):
        return {
            "strict_support_loss_frame_count": frames_where(
                lambda state, evaluation: values(which, state, evaluation)[1]
                > SCORE_TOLERANCE
            ),
            "exact_support_score_frame_fraction": frames_where(
                lambda state, evaluation: values(which, state, evaluation)[1]
                <= SCORE_TOLERANCE
            )
            / frame_count,
            "exact_certified_frame_fraction": frames_where(
                lambda state, evaluation: values(which, state, evaluation)[0][
                    "exact"
                ]
            )
            / frame_count,
            "mean_finite_budget_support_regret": average(
                lambda state, evaluation: values(which, state, evaluation)[1]
            ),
            "mean_finite_budget_support_regret_per_visible_job": average(
                lambda state, evaluation: (
                    values(which, state, evaluation)[1]
                    / values(which, state, evaluation)[2]
                    if values(which, state, evaluation)[2]
                    else 0.0
                )
            ),
            "mean_certified_gap": average(
                lambda state, evaluation: values(which, state, evaluation)[0]["gap"]
            ),
            "mean_certified_gap_per_visible_job": average(
                lambda state, evaluation: values(which, state, evaluation)[0][
                    "epsilon"
                ]
            ),
            "mean_candidate_attempts": average(
                lambda state, evaluation: values(which, state, evaluation)[0][
                    "attempts"
                ]
            ),
        }

    legacy_summary = summary("legacy")
    variant_summary = summary("variant")
    return {
        "load": load,
        "seed": seed,
        "source_policy_budget": source_policy_budget,
        "population_role": (
            "primary frozen K26-policy states"
            if source_policy_budget == SMALL_BUDGET
            else "frozen K120-policy state sensitivity"
        ),
        "frame_count": frame_count,
        "legacy_visited_time_s": visited_time,
        "unique_queue_state_count": len(weighted),
        "support_improved_frame_count": frames_where(
            lambda _state, evaluation: evaluation["support_outcome"] == "improved"
        ),
        "support_regressed_frame_count": frames_where(
            lambda _state, evaluation: evaluation["support_outcome"] == "regressed"
        ),
        "upper_bound_worsened_frame_count": frames_where(
            lambda _state, evaluation: evaluation[
                "upper_bound_delta_over_legacy_k26"
            ]
            > SCORE_TOLERANCE
        ),
        "gap_worsened_frame_count": frames_where(
            lambda _state, evaluation: evaluation["gap_delta_over_legacy_k26"]
            > SCORE_TOLERANCE
        ),
        "selected_physical_signature_changed_frame_fraction": frames_where(
            lambda _state, evaluation: evaluation[
                "selected_physical_signature_changed_from_legacy_k26"
            ]
        )
        / frame_count,
        "selected_physical_signature_changed_time_fraction": time_where(
            lambda _state, evaluation: evaluation[
                "selected_physical_signature_changed_from_legacy_k26"
            ]
        )
        / visited_time,
        "raw_lane_witness_changed_frame_fraction": frames_where(
            lambda _state, evaluation: evaluation[
                "raw_lane_witness_changed_from_legacy_k26"
            ]
        )
        / frame_count,
        "raw_lane_witness_changed_time_fraction": time_where(
            lambda _state, evaluation: evaluation[
                "raw_lane_witness_changed_from_legacy_k26"
            ]
        )
        / visited_time,
        "mean_variant_cache_hits": average(
            lambda _state, evaluation: evaluation["variant_k26"]["cache_hits"]
        ),
        "mean_variant_logical_search_pops": average(
            lambda _state, evaluation: evaluation["variant_k26"][
                "logical_search_pops"
            ]
        ),
        "legacy": legacy_summary,
        "variant": variant_summary,
        "variant_minus_legacy": {
            key: variant_summary[key] - legacy_summary[key]
            for key in variant_summary
        },
    }


def _source_summary_matches(source_summary, shadow_summary):
    if (
        source_summary["frame_count"] != shadow_summary["frame_count"]
        or source_summary["unique_queue_state_count"]
        != shadow_summary["unique_queue_state_count"]
        or source_summary["strict_support_loss_frame_count"]
        != shadow_summary["legacy"]["strict_support_loss_frame_count"]
    ):
        return False
    float_pairs = (
        (
            source_summary["exact_score_found_by_k26_frame_fraction"],
            shadow_summary["legacy"]["exact_support_score_frame_fraction"],
        ),
        (
            source_summary["k26_exact_certified_frame_fraction"],
            shadow_summary["legacy"]["exact_certified_frame_fraction"],
        ),
        (
            source_summary["mean_finite_budget_support_regret"],
            shadow_summary["legacy"]["mean_finite_budget_support_regret"],
        ),
        (
            source_summary["mean_finite_budget_support_regret_per_visible_job"],
            shadow_summary["legacy"][
                "mean_finite_budget_support_regret_per_visible_job"
            ],
        ),
        (
            source_summary["mean_certified_gap"],
            shadow_summary["legacy"]["mean_certified_gap"],
        ),
    )
    return all(
        math.isclose(left, right, rel_tol=0.0, abs_tol=AGGREGATE_TOLERANCE)
        for left, right in float_pairs
    )


def _six_metric_checks(prefix, summaries):
    return {
        f"{prefix}_mean_support_regret_nonworse_at_every_load": all(
            row["variant"]["mean_finite_budget_support_regret"]
            <= row["legacy"]["mean_finite_budget_support_regret"]
            + AGGREGATE_TOLERANCE
            for row in summaries
        ),
        f"{prefix}_strict_support_loss_count_nonworse_at_every_load": all(
            row["variant"]["strict_support_loss_frame_count"]
            <= row["legacy"]["strict_support_loss_frame_count"]
            for row in summaries
        ),
        f"{prefix}_mean_certified_gap_nonworse_at_every_load": all(
            row["variant"]["mean_certified_gap"]
            <= row["legacy"]["mean_certified_gap"] + AGGREGATE_TOLERANCE
            for row in summaries
        ),
        f"{prefix}_mean_candidate_attempts_nonworse_at_every_load": all(
            row["variant"]["mean_candidate_attempts"]
            <= row["legacy"]["mean_candidate_attempts"] + AGGREGATE_TOLERANCE
            for row in summaries
        ),
        f"{prefix}_exact_support_fraction_nonworse_at_every_load": all(
            row["variant"]["exact_support_score_frame_fraction"]
            + AGGREGATE_TOLERANCE
            >= row["legacy"]["exact_support_score_frame_fraction"]
            for row in summaries
        ),
        f"{prefix}_exact_certified_fraction_nonworse_at_every_load": all(
            row["variant"]["exact_certified_frame_fraction"]
            + AGGREGATE_TOLERANCE
            >= row["legacy"]["exact_certified_frame_fraction"]
            for row in summaries
        ),
    }


def _advance_checks(primary, sensitivity, evaluations, execution_passed):
    checks = {
        "execution_passed": execution_passed,
        "no_registered_state_has_lower_k26_support": all(
            row["support_score_delta_over_legacy_k26"] >= -SCORE_TOLERANCE
            for row in evaluations
        ),
        "no_registered_state_has_higher_k26_upper_bound": all(
            row["upper_bound_delta_over_legacy_k26"] <= SCORE_TOLERANCE
            for row in evaluations
        ),
        "no_registered_state_has_higher_k26_gap": all(
            row["gap_delta_over_legacy_k26"] <= SCORE_TOLERANCE
            for row in evaluations
        ),
    }
    checks.update(_six_metric_checks("primary", primary))
    checks.update(_six_metric_checks("sensitivity", sensitivity))
    checks.update(
        {
            "at_least_one_primary_load_strictly_improves_support": any(
                row["variant"]["mean_finite_budget_support_regret"]
                < row["legacy"]["mean_finite_budget_support_regret"]
                - AGGREGATE_TOLERANCE
                or row["variant"]["exact_support_score_frame_fraction"]
                > row["legacy"]["exact_support_score_frame_fraction"]
                + AGGREGATE_TOLERANCE
                for row in primary
            ),
            "at_least_one_primary_load_strictly_improves_certificate_or_attempts": any(
                row["variant"]["mean_certified_gap"]
                < row["legacy"]["mean_certified_gap"] - AGGREGATE_TOLERANCE
                or row["variant"]["exact_certified_frame_fraction"]
                > row["legacy"]["exact_certified_frame_fraction"]
                + AGGREGATE_TOLERANCE
                or row["variant"]["mean_candidate_attempts"]
                < row["legacy"]["mean_candidate_attempts"] - AGGREGATE_TOLERANCE
                for row in primary
            ),
        }
    )
    return checks


def _decision(checks, primary, evaluations, states):
    if not checks["execution_passed"]:
        return "INVALID_EXECUTION"
    if all(checks.values()):
        return "ADVANCE_TO_AUTHORED_TRANSITION_SCREEN"
    non_improvement_keys = {
        "at_least_one_primary_load_strictly_improves_support",
        "at_least_one_primary_load_strictly_improves_certificate_or_attempts",
    }
    nonworse = all(
        value for key, value in checks.items() if key not in non_improvement_keys
    )
    registered_action_changed = any(
        evaluation["raw_lane_witness_changed_from_legacy_k26"]
        or evaluation["selected_physical_signature_changed_from_legacy_k26"]
        for evaluation in evaluations
    )
    if (
        nonworse
        and not checks["at_least_one_primary_load_strictly_improves_support"]
        and not registered_action_changed
        and checks[
            "at_least_one_primary_load_strictly_improves_certificate_or_attempts"
        ]
    ):
        return "EFFICIENCY_ONLY_STOP"
    return "CLOSE_WITHOUT_TRANSITION_SCREEN"


def register(output=DEFAULT_OUTPUT):
    output, protocol_path, sources_path = _paths(output)
    if any(path.exists() for path in (output, protocol_path, sources_path)):
        raise ValueError("permutation-quotient shadow artifacts already exist")
    diagnostic, model = _validate_source_diagnostic()
    roster = _validated_roster(diagnostic)
    _validate_closed_sidecar()
    if (
        (len(model.options) + 1) ** model.lanes != ORDERED_ACTION_COUNT
        or math.comb(len(model.options) + model.lanes, model.lanes)
        != QUOTIENT_ORBIT_COUNT
    ):
        raise ValueError("registered ordered-action or quotient-orbit count changed")
    protocol = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "aligned_hardware_permutation_quotient_state_shadow_v1",
        "role": ROLE,
        "source_experiment": SOURCE_DIAGNOSTIC.name,
        "registered_state_roster": REGISTERED_STATES.name,
        "closed_predecessor": CLOSED_SIDECAR_RESULT.name,
        "loads": list(LOADS),
        "seeds": list(SEEDS),
        "source_policy_budgets": list(SOURCE_POLICY_BUDGETS),
        "small_checkpoint_budget": SMALL_BUDGET,
        "exact_quotient_budget": EXACT_QUOTIENT_BUDGET,
        "state_count": STATE_COUNT,
        "ordered_action_count": ORDERED_ACTION_COUNT,
        "quotient_orbit_count": QUOTIENT_ORBIT_COUNT,
        "model": model.snapshot(),
        "registered_state_schema": REGISTERED_STATE_SCHEMA,
        "rule": RULE,
        "source_population_summaries": diagnostic[
            "summary_by_load_and_source_policy"
        ],
        "source_population_cells": _source_population_cells(diagnostic),
        "advance_gate": ADVANCE_GATE,
        "stopping_rule": STOPPING_RULE,
        "prohibited_in_this_stage": PROHIBITED,
        "support_reference": SUPPORT_REFERENCE,
    }
    sources = _source_texts()
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_new(protocol_path, protocol)
    _write_new(sources_path, sources)
    return protocol


def run_registered(output=DEFAULT_OUTPUT):
    output, protocol_path, sources_path = _paths(output)
    if output.exists():
        raise ValueError("permutation-quotient shadow result already exists")
    protocol = json.loads(protocol_path.read_text())
    sources = json.loads(sources_path.read_text())
    expected = {
        "experiment": "aligned_hardware_permutation_quotient_state_shadow_v1",
        "source_experiment": SOURCE_DIAGNOSTIC.name,
        "registered_state_roster": REGISTERED_STATES.name,
        "closed_predecessor": CLOSED_SIDECAR_RESULT.name,
        "loads": list(LOADS),
        "seeds": list(SEEDS),
        "source_policy_budgets": list(SOURCE_POLICY_BUDGETS),
        "small_checkpoint_budget": SMALL_BUDGET,
        "exact_quotient_budget": EXACT_QUOTIENT_BUDGET,
        "state_count": STATE_COUNT,
        "ordered_action_count": ORDERED_ACTION_COUNT,
        "quotient_orbit_count": QUOTIENT_ORBIT_COUNT,
    }
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise ValueError("registered permutation-quotient roster or constants changed")
    frozen_objects = {
        "role": ROLE,
        "registered_state_schema": REGISTERED_STATE_SCHEMA,
        "rule": RULE,
        "advance_gate": ADVANCE_GATE,
        "stopping_rule": STOPPING_RULE,
        "prohibited_in_this_stage": PROHIBITED,
        "support_reference": SUPPORT_REFERENCE,
    }
    if any(protocol.get(key) != value for key, value in frozen_objects.items()):
        raise ValueError("registered permutation-quotient protocol rule changed")
    if sources != _source_texts():
        raise ValueError("registered permutation-quotient sources changed")
    diagnostic, model = _validate_source_diagnostic()
    roster = _validated_roster(diagnostic)
    _validate_closed_sidecar()
    if _json(model.snapshot()) != protocol["model"]:
        raise ValueError("registered permutation-quotient model changed")
    if protocol["source_population_summaries"] != diagnostic[
        "summary_by_load_and_source_policy"
    ]:
        raise ValueError("registered permutation-quotient source summaries changed")
    if protocol["source_population_cells"] != _source_population_cells(diagnostic):
        raise ValueError("registered permutation-quotient source cells changed")
    states = roster["states"]
    selector_started = time.perf_counter()
    evaluations = [
        _evaluate_state(model, index, state) for index, state in enumerate(states)
    ]
    selector_evaluation_wall_s = time.perf_counter() - selector_started
    population_summaries = [
        _metrics_for_population(states, evaluations, load, budget)
        for load in LOADS
        for budget in SOURCE_POLICY_BUDGETS
    ]
    seed_summaries = [
        _metrics_for_population(states, evaluations, load, budget, seed)
        for load in LOADS
        for seed in SEEDS
        for budget in SOURCE_POLICY_BUDGETS
    ]
    source_summary_by_id = {
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
            len(seed_summaries) == 18
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
                    seed_summaries,
                    protocol["source_population_cells"],
                    strict=True,
                )
            )
        ),
        "registered_six_population_aggregates_reconstruct_source": all(
            _source_summary_matches(
                source_summary_by_id[
                    (row["load"], row["source_policy_budget"])
                ],
                row,
            )
            for row in population_summaries
        ),
        "ordered_100_to_quotient_55_accounting_holds": all(
            row["variant_k26"]["represented_ordered_configurations"]
            == ORDERED_ACTION_COUNT
            and row["variant_k26"]["represented_lane_multisets"]
            == QUOTIENT_ORBIT_COUNT
            and row["variant_k26"][
                "symmetry_duplicate_ordered_configurations"
            ]
            == ORDERED_ACTION_COUNT - QUOTIENT_ORBIT_COUNT
            for row in evaluations
        ),
        "candidate_cache_and_frontier_work_identities_hold": all(
            row["selection_bound_and_work_checks_pass"] for row in evaluations
        ),
        "k55_closes_with_registered_support_on_every_state": all(
            row["variant_k55"]["exact"]
            and row["variant_k55_support_matches_registered_reference"]
            for row in evaluations
        ),
        "k26_support_bound_gap_and_epsilon_hold": all(
            abs(row["variant_gap_identity_error"]) <= SCORE_TOLERANCE
            and abs(row["variant_epsilon_error"]) <= SCORE_TOLERANCE
            for row in evaluations
        ),
        "candidate_attempts_respect_k26_and_k55_budgets": all(
            row["variant_k26"]["attempts"] <= SMALL_BUDGET
            and row["variant_k55"]["attempts"] <= EXACT_QUOTIENT_BUDGET
            for row in evaluations
        ),
        "lane_permutation_cache_activates": any(
            row["variant_k26"]["cache_hits"] > 0 for row in evaluations
        ),
        "no_arrivals_trajectories_or_randomness_used": True,
        "registered_sources_unchanged": all(
            (ROOT / name).read_text() == text for name, text in sources.items()
        ),
    }
    execution_passed = all(execution_checks.values())
    primary = [
        row
        for row in population_summaries
        if row["source_policy_budget"] == SMALL_BUDGET
    ]
    sensitivity = [
        row
        for row in population_summaries
        if row["source_policy_budget"] == 120
    ]
    advance_checks = _advance_checks(
        primary, sensitivity, evaluations, execution_passed
    )
    decision = _decision(advance_checks, primary, evaluations, states)
    retained_changes = [
        row
        for row in evaluations
        if row["support_outcome"] != "unchanged"
        or row["raw_lane_witness_changed_from_legacy_k26"]
        or row["selected_physical_signature_changed_from_legacy_k26"]
    ]
    report = {
        "experiment": "aligned_hardware_permutation_quotient_state_shadow_v1",
        "status": "PASS" if execution_passed else "FAIL",
        "checks": execution_checks,
        "protocol": protocol,
        "registered_state_roster": REGISTERED_STATES.name,
        "state_evaluation_count": len(evaluations),
        "retained_change_state_count": len(retained_changes),
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
                "legacy_option_indices": states[row["state_index"]][
                    "legacy_k26"
                ]["option_indices"],
                "variant_option_indices": row["variant_k26"]["option_indices"],
                "variant_attempts": row["variant_k26"]["attempts"],
                "variant_cache_hits": row["variant_k26"]["cache_hits"],
            }
            for row in retained_changes
        ],
        "summary_by_load_and_source_policy": population_summaries,
        "summary_by_load_seed_and_source_policy": seed_summaries,
        "work_accounting": {
            "selector_calls": 2 * len(evaluations),
            "selector_evaluation_wall_s": selector_evaluation_wall_s,
            "k26_paid_candidate_attempts": sum(
                row["variant_k26"]["attempts"] for row in evaluations
            ),
            "k26_free_cache_hit_pops": sum(
                row["variant_k26"]["free_cache_hit_pops"]
                for row in evaluations
            ),
            "k26_logical_search_pops": sum(
                row["variant_k26"]["logical_search_pops"]
                for row in evaluations
            ),
            "k55_paid_candidate_attempts": sum(
                row["variant_k55"]["attempts"] for row in evaluations
            ),
            "k55_free_cache_hit_pops": sum(
                row["variant_k55"]["free_cache_hit_pops"]
                for row in evaluations
            ),
            "candidate_score_evaluations": sum(
                row[checkpoint]["candidate_score_evaluations"]
                for row in evaluations
                for checkpoint in ("variant_k26", "variant_k55")
            ),
            "cache_key_evaluations": sum(
                row[checkpoint]["cache_key_evaluations"]
                for row in evaluations
                for checkpoint in ("variant_k26", "variant_k55")
            ),
            "upper_bound_calls": sum(
                row[checkpoint]["upper_bound_calls"]
                for row in evaluations
                for checkpoint in ("variant_k26", "variant_k55")
            ),
            "frontier_pushes": sum(
                row[checkpoint]["frontier_pushes"]
                for row in evaluations
                for checkpoint in ("variant_k26", "variant_k55")
            ),
            "bound_option_probes": sum(
                row[checkpoint]["bound_option_probes"]
                for row in evaluations
                for checkpoint in ("variant_k26", "variant_k55")
            ),
            "bound_threshold_evaluations": sum(
                row[checkpoint]["bound_threshold_evaluations"]
                for row in evaluations
                for checkpoint in ("variant_k26", "variant_k55")
            ),
            "physical_signature_evaluations": 3 * len(evaluations),
            "completion_offset_reads": sum(
                row["physical_signature_completion_offset_reads"]
                for row in evaluations
            ),
            "arrival_tapes": 0,
            "trajectory_frames": 0,
            "rng_initializations": 0,
            "random_draws": 0,
        },
        "shadow_advance_checks": advance_checks,
        "scientific_decision": decision,
        "development_advance": decision
        == "ADVANCE_TO_AUTHORED_TRANSITION_SCREEN",
        "next_step": (
            "preregister the authored changed-state transition screen"
            if decision == "ADVANCE_TO_AUTHORED_TRANSITION_SCREEN"
            else "close this quotient-cache branch without tuning or trajectories"
        ),
        "limitations": [
            "The registered states and mechanism rationale use consumed development evidence; this is not independent confirmation.",
            "K55 certifies maximum one-frame support only and does not predict mean flow or P90 under policy feedback.",
            "A changed raw lane witness can pair FIFO jobs with completion offsets differently even when its option multiset is unchanged, so trajectory P90 remains unresolved.",
            "The quotient applies to the current identical, fully renewed lanes; heterogeneous persistent lanes would not share this symmetry.",
            "Legacy visited time weights counterfactual fixed-state outcomes and is not variant trajectory time.",
        ],
    }
    if sources != _source_texts():
        raise ValueError("registered permutation-quotient sources changed during execution")
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
