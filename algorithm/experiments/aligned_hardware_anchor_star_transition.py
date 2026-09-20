"""Close the anchor-star transition branch from already completed evidence.

The anchor-star shadow contains 1,413 changed raw actions.  Of these, 912 have
identical FIFO execution and therefore have exactly zero flow effect, while 14
FIFO-effective pairs exactly match completed permutation-quotient transition
cases.  Those completed cases already contain positive mean-flow and P90
counterexamples to the frozen conjunctive advance gate.  Consequently the 487
novel FIFO-effective pairs cannot change the decision and are not drained.

Both registration and registered execution are evidence-only operations: they
call no selector, create no arrival tape, and execute no drain or trajectory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from algorithm.experiments import aligned_hardware_anchor_star_shadow as anchor_shadow
from algorithm.experiments import (
    aligned_hardware_permutation_quotient_transition as quotient_transition,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
DEFAULT_OUTPUT = ARTIFACTS / "aligned_hardware_anchor_star_transition_development_20260911.json"
ANCHOR_SHADOW = ARTIFACTS / "aligned_hardware_anchor_star_shadow_development_20260911.json"
ANCHOR_SHADOW_PROTOCOL = ARTIFACTS / "aligned_hardware_anchor_star_shadow_development_20260911_protocol.json"
ANCHOR_SHADOW_SOURCES = ARTIFACTS / "aligned_hardware_anchor_star_shadow_development_20260911_fixed_sources.json"
CLOSED_TRANSITION = ARTIFACTS / "aligned_hardware_permutation_quotient_transition_development_20260911.json"
CLOSED_TRANSITION_PROTOCOL = ARTIFACTS / "aligned_hardware_permutation_quotient_transition_development_20260911_protocol.json"
CLOSED_TRANSITION_SOURCES = ARTIFACTS / "aligned_hardware_permutation_quotient_transition_development_20260911_fixed_sources.json"
REGISTERED_STATES = ARTIFACTS / "aligned_hardware_duration_tie_order_shadow_development_20260910_registered_states.json"

BUDGET = 26
LOADS = (0.50, 0.80, 0.95)
SEEDS = (41, 42, 43)
AGE_PROFILES = ("coincident", "unit_fifo")
TOLERANCE = 1e-9
CHANGED_STATE_COUNT = 1413
FIFO_ALIAS_COUNT = 912
KNOWN_REUSE_COUNT = 14
NOVEL_UNEXECUTED_COUNT = 487
KNOWN_MEAN_VETO_COUNT = 6
KNOWN_P90_VETO_COUNT = 12

ANCHOR_CHECK_KEYS = frozenset({
    "registered_5003_unique_states_executed_once",
    "registered_18_population_cells_reconstruct",
    "registered_six_population_aggregates_reconstruct_source",
    "two_lane_100_actions_and_10_plus_72_equals_82_structure_holds",
    "registered_16_known_quotient_action_pairs_reconstruct",
    "original_index_bound_and_work_identities_hold",
    "k82_matches_registered_legacy_k120_support_on_every_state",
    "k26_support_bound_gap_and_epsilon_hold",
    "candidate_attempts_respect_k26_and_k82_budgets",
    "no_arrivals_trajectories_or_randomness_used",
    "registered_sources_unchanged",
})
ANCHOR_ADVANCE_KEYS = frozenset({
    "execution_passed",
    "no_registered_state_has_lower_k26_support",
    "no_registered_state_has_higher_k26_upper_bound",
    "no_registered_state_has_higher_k26_gap",
    "primary_mean_support_regret_nonworse_at_every_load",
    "primary_strict_support_loss_count_nonworse_at_every_load",
    "primary_mean_certified_gap_nonworse_at_every_load",
    "primary_mean_candidate_attempts_nonworse_at_every_load",
    "primary_exact_support_fraction_nonworse_at_every_load",
    "primary_exact_certified_fraction_nonworse_at_every_load",
    "sensitivity_mean_support_regret_nonworse_at_every_load",
    "sensitivity_strict_support_loss_count_nonworse_at_every_load",
    "sensitivity_mean_certified_gap_nonworse_at_every_load",
    "sensitivity_mean_candidate_attempts_nonworse_at_every_load",
    "sensitivity_exact_support_fraction_nonworse_at_every_load",
    "sensitivity_exact_certified_fraction_nonworse_at_every_load",
    "at_least_one_primary_load_strictly_improves_support",
    "at_least_one_primary_load_strictly_improves_certificate_or_attempts",
})
TRANSITION_CHECK_KEYS = frozenset({
    "registered_16_unique_states_executed",
    "registered_64_complete_drains_executed",
    "registered_first_actions_support_and_flow_identities_hold",
    "all_drains_respect_registered_elapsed_bounds",
    "age_profiles_leave_each_arm_queue_action_path_unchanged",
    "two_profiles_reproduce_one_unique_mean_flow_effect_per_state",
    "nine_primary_load_seed_cells_reconstruct_positive_visits",
    "raw_lane_fifo_pairing_fixture_passes",
    "no_future_arrivals_or_randomness_used",
    "registered_sources_unchanged",
})
TRANSITION_CASE_CHECK_KEYS = frozenset({
    "registered_first_actions_reproduced",
    "variant_first_action_support_improvement_reproduced",
    "both_arms_and_profiles_complete_within_bound",
    "age_profiles_do_not_change_queue_or_action_paths",
    "age_profiles_have_identical_total_flow_effect",
    "paired_flow_component_identity_holds",
})

ROLE = "development evidence-reuse closure after an irreversible local-flow veto"
PARTITION_RULE = {
    "fifo_alias": (
        "unchanged FIFO execution signature implies identical first-frame job "
        "completion times, next queue, deterministic v4 continuation, and zero "
        "mean-flow and P90 effect for both registered age profiles"
    ),
    "known_reuse": (
        "exact state, queue, raw lane tuple, and FIFO execution pair matches a "
        "completed permutation-quotient authored-transition case"
    ),
    "novel_unexecuted": (
        "FIFO execution differs and has no exact completed transition match; it "
        "remains unmeasured because completed known cases already falsify a "
        "mandatory advance condition"
    ),
}
ADVANCE_GATE = {
    "all_changed_state_mean_flow_deltas_nonpositive": True,
    "all_changed_state_profile_nearest_rank_p90_deltas_nonpositive": True,
    "all_9_primary_load_seed_weighted_flow_area_deltas_nonpositive": True,
    "at_least_one_changed_state_has_strict_mean_flow_improvement": True,
    "at_least_one_primary_load_seed_cell_has_strict_weighted_flow_area_improvement": True,
    "logic": "all five conditions are conjunctive",
}
SHORT_CIRCUIT_RULE = (
    "an exact reused positive mean-flow or P90 witness permanently falsifies a "
    "mandatory universal gate; then record every witness, skip all novel drains, "
    "and close without tuning, trajectories, or paper claims"
)
PROHIBITED = (
    "selector calls, authored drains, arrival generation, trajectory simulation, "
    "RNG, new states, new age profiles, outcome-tuned filtering, confirmation, "
    "and paper edits"
)
INTERPRETATION = (
    "This closure proves only that the frozen anchor-star branch cannot pass its "
    "local transition gate. It does not estimate the unexecuted novel effects."
)
NEW_SOURCES = (
    "algorithm/experiments/aligned_hardware_anchor_star_transition.py",
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


def _write_new(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")


def _source_texts():
    frozen = json.loads(ANCHOR_SHADOW_SOURCES.read_text())
    if frozen != anchor_shadow._source_texts():
        raise ValueError("anchor-star shadow source map changed")
    closed = json.loads(CLOSED_TRANSITION_SOURCES.read_text())
    if closed != quotient_transition._source_texts():
        raise ValueError("closed quotient-transition source map changed")
    conflicts = [
        name for name in frozen.keys() & closed.keys() if frozen[name] != closed[name]
    ]
    if conflicts:
        raise ValueError(f"frozen predecessor source maps disagree: {conflicts}")
    return {
        **frozen,
        **{name: (ROOT / name).read_text() for name in NEW_SOURCES},
    }


def _action_signatures(model, option_indices):
    indices = tuple(option_indices)
    service = [
        sum(
            model.options[index].requirements[coordinate]
            for index in indices
            if index is not None
        )
        for coordinate in range(len(model.workloads))
    ]
    duration = max((
        1.0,
        *(model.options[index].duration for index in indices if index is not None),
    ))
    ordered_offsets = [
        [
            float(offset)
            for index in indices
            if index is not None
            for offset in model.options[index].completion_offsets[coordinate]
        ]
        for coordinate in range(len(model.workloads))
    ]
    return (
        {
            "service": service,
            "duration": float(duration),
            "completion_offset_multisets": [sorted(row) for row in ordered_offsets],
        },
        {
            "service": service,
            "duration": float(duration),
            "ordered_completion_offsets": ordered_offsets,
        },
    )


def _primary_visits(state):
    return [
        [visit[0], visit[1], visit[3], visit[4]]
        for visit in state["population_visits"]
        if visit[2] == BUDGET
    ]


def _close(left, right):
    try:
        return math.isclose(
            float(left), float(right), rel_tol=0.0, abs_tol=TOLERANCE
        )
    except (TypeError, ValueError):
        return False


def _validate_anchor_shadow():
    report = json.loads(ANCHOR_SHADOW.read_text())
    protocol = json.loads(ANCHOR_SHADOW_PROTOCOL.read_text())
    checks = report.get("checks", {})
    advance = report.get("shadow_advance_checks", {})
    if (
        report.get("protocol") != protocol
        or report.get("experiment") != "aligned_hardware_anchor_star_state_shadow_v1"
        or report.get("status") != "PASS"
        or frozenset(checks) != ANCHOR_CHECK_KEYS
        or not all(checks.values())
        or frozenset(advance) != ANCHOR_ADVANCE_KEYS
        or not all(advance.values())
        or report.get("scientific_decision") != "ADVANCE_AUTHORED_TRANSITION"
        or report.get("development_advance") is not True
        or report.get("reported_state_change_count") != CHANGED_STATE_COUNT
        or report.get("fifo_effective_change_state_count")
        != KNOWN_REUSE_COUNT + NOVEL_UNEXECUTED_COUNT
        or report.get("fifo_equivalent_raw_only_change_state_count")
        != FIFO_ALIAS_COUNT
        or report.get("known_quotient_fifo_pair_count") != KNOWN_REUSE_COUNT
        or report.get("new_fifo_effective_pair_count") != NOVEL_UNEXECUTED_COUNT
        or len(report.get("state_changes", ())) != CHANGED_STATE_COUNT
    ):
        raise ValueError("closure requires the completed advancing anchor-star shadow")
    diagnostic, model = anchor_shadow._validate_source_diagnostic()
    roster = anchor_shadow._validated_roster(diagnostic)
    if (
        protocol.get("registered_state_roster") != REGISTERED_STATES.name
        or protocol.get("model") != _json(model.snapshot())
        or report.get("registered_state_roster") != REGISTERED_STATES.name
        or protocol.get("role") != anchor_shadow.ROLE
        or protocol.get("loads") != list(anchor_shadow.LOADS)
        or protocol.get("seeds") != list(anchor_shadow.SEEDS)
        or protocol.get("source_policy_budgets")
        != list(anchor_shadow.SOURCE_POLICY_BUDGETS)
        or protocol.get("small_checkpoint_budget") != anchor_shadow.SMALL_BUDGET
        or protocol.get("legacy_exact_reference_budget")
        != anchor_shadow.LEGACY_EXACT_BUDGET
        or protocol.get("anchor_star_exact_budget")
        != anchor_shadow.ANCHOR_EXACT_BUDGET
        or protocol.get("state_count") != anchor_shadow.STATE_COUNT
        or protocol.get("rule") != anchor_shadow.RULE
        or protocol.get("advance_gate") != anchor_shadow.ADVANCE_GATE
        or protocol.get("stopping_rule") != anchor_shadow.STOPPING_RULE
        or protocol.get("prohibited_in_this_stage") != anchor_shadow.PROHIBITED
        or protocol.get("support_reference") != anchor_shadow.SUPPORT_REFERENCE
        or protocol.get("anchor_star_structure")
        != anchor_shadow._structure_snapshot(model)
    ):
        raise ValueError("anchor-star shadow model or state lineage changed")
    _source_texts()
    return report, protocol, roster, model


def _validate_closed_transition(model):
    report = json.loads(CLOSED_TRANSITION.read_text())
    protocol = json.loads(CLOSED_TRANSITION_PROTOCOL.read_text())
    checks = report.get("checks", {})
    _predecessor, _roster, validated_model, cases = (
        quotient_transition._validate_predecessor()
    )
    maximum_duration = max(
        1.0, *(float(option.duration) for option in validated_model.options)
    )
    maximum_jobs = max(sum(case["queue"]) for case in cases)
    expected_protocol = {
        "experiment": "aligned_hardware_permutation_quotient_authored_transition_v1",
        "role": quotient_transition.ROLE,
        "source_predecessor": quotient_transition.PREDECESSOR.name,
        "registered_state_roster": REGISTERED_STATES.name,
        "budget": BUDGET,
        "state_count": 16,
        "age_profiles": list(AGE_PROFILES),
        "arm_count": 2,
        "registered_drain_count": 64,
        "loads": list(LOADS),
        "seeds": list(SEEDS),
        "model": _json(model.snapshot()),
        "maximum_model_frame_duration_s": maximum_duration,
        "maximum_registered_queue_jobs": maximum_jobs,
        "global_elapsed_frame_barrier_bound_s": maximum_jobs * maximum_duration,
        "case_roster": cases,
        "rule": quotient_transition.RULE,
        "fifo_pairing_fixture": quotient_transition.FIXTURE,
        "advance_gate": quotient_transition.ADVANCE_GATE,
        "stopping_rule": quotient_transition.STOPPING_RULE,
        "prohibited_in_this_stage": quotient_transition.PROHIBITED,
        "interpretation": (
            "a conservative local transition veto; occurrence weights come from "
            "the frozen v4 trajectories and are not quotient-policy trajectory outcomes"
        ),
    }
    if (
        report.get("protocol") != protocol
        or report.get("experiment")
        != "aligned_hardware_permutation_quotient_authored_transition_v1"
        or report.get("status") != "PASS"
        or frozenset(checks) != TRANSITION_CHECK_KEYS
        or not all(checks.values())
        or report.get("scientific_decision") != "LOCAL_FLOW_TRADEOFF_STOP"
        or report.get("development_advance") is not False
        or protocol.get("registered_state_roster") != REGISTERED_STATES.name
        or protocol.get("budget") != BUDGET
        or protocol.get("age_profiles") != list(AGE_PROFILES)
        or protocol.get("loads") != list(LOADS)
        or protocol.get("seeds") != list(SEEDS)
        or protocol.get("state_count") != 16
        or protocol.get("registered_drain_count") != 64
        or protocol.get("model") != _json(model.snapshot())
        or len(protocol.get("case_roster", ())) != 16
        or len(report.get("cases", ())) != 16
        or frozenset(protocol) != frozenset({"registered_at_utc", *expected_protocol})
        or any(protocol.get(key) != value for key, value in expected_protocol.items())
    ):
        raise ValueError("completed quotient transition is unavailable or changed")
    try:
        datetime.fromisoformat(protocol["registered_at_utc"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("completed quotient transition timestamp changed") from error
    return report, protocol


def _case_entry(change, state, evidence_class):
    visits = _primary_visits(state)
    if (
        state["queue"] != change["queue"]
        or state["legacy_k26"]["option_indices"] != change["legacy_option_indices"]
        or sum(visit[2] for visit in visits) != change["primary_frame_visits"]
    ):
        raise ValueError("anchor-star changed-state lineage changed")
    return {
        "state_index": change["state_index"],
        "queue": change["queue"],
        "primary_population_visits": visits,
        "primary_frame_visits": change["primary_frame_visits"],
        "baseline_option_indices": change["legacy_option_indices"],
        "variant_option_indices": change["variant_option_indices"],
        "baseline_physical_signature": change["legacy_physical_signature"],
        "variant_physical_signature": change["variant_physical_signature"],
        "baseline_fifo_execution_signature": change[
            "legacy_fifo_execution_signature"
        ],
        "variant_fifo_execution_signature": change[
            "variant_fifo_execution_signature"
        ],
        "registered_support_score_delta": change[
            "support_score_delta_over_legacy_k26"
        ],
        "evidence_class": evidence_class,
    }


def _validate_reused_case(source, entry):
    case_checks = source.get("checks", {})
    if (
        source.get("state_index") != entry["state_index"]
        or source.get("queue") != entry["queue"]
        or source.get("legacy_option_indices") != entry["baseline_option_indices"]
        or source.get("variant_option_indices") != entry["variant_option_indices"]
        or source.get("primary_population_visits")
        != entry["primary_population_visits"]
        or not _close(
            source.get("registered_support_score_delta"),
            entry["registered_support_score_delta"],
        )
        or frozenset(case_checks) != TRANSITION_CASE_CHECK_KEYS
        or not all(case_checks.values())
    ):
        raise ValueError("known FIFO pair does not exactly match its completed case")
    profiles = source.get("profiles", ())
    if [row.get("profile") for row in profiles] != list(AGE_PROFILES):
        raise ValueError("known FIFO pair age profiles changed")
    normalized_profiles = []
    for row in profiles:
        baseline = row.get("baseline", {})
        variant = row.get("variant", {})
        if (
            row.get("baseline_first_action", {}).get("option_indices")
            != entry["baseline_option_indices"]
            or row.get("variant_first_action", {}).get("option_indices")
            != entry["variant_option_indices"]
            or not row.get("first_actions_realizable")
            or not _close(
                row.get("flow_area_delta_job_s"),
                variant.get("flow_area_job_s", 0.0)
                - baseline.get("flow_area_job_s", 0.0),
            )
            or not _close(
                row.get("mean_flow_delta_s"),
                variant.get("mean_flow_s", 0.0) - baseline.get("mean_flow_s", 0.0),
            )
            or not _close(
                row.get("p90_flow_delta_s"),
                variant.get("p90_flow_s", 0.0) - baseline.get("p90_flow_s", 0.0),
            )
        ):
            raise ValueError("completed known FIFO outcome changed")
        normalized_profiles.append({
            "profile": row["profile"],
            "flow_area_delta_job_s": row["flow_area_delta_job_s"],
            "mean_flow_delta_s": row["mean_flow_delta_s"],
            "p90_flow_delta_s": row["p90_flow_delta_s"],
            "makespan_delta_s": row["makespan_delta_s"],
        })
    if (
        not _close(
            source.get("unique_state_flow_area_delta_job_s"),
            normalized_profiles[0]["flow_area_delta_job_s"],
        )
        or not _close(
            source.get("unique_state_mean_flow_delta_s"),
            normalized_profiles[0]["mean_flow_delta_s"],
        )
        or not _close(
            normalized_profiles[0]["flow_area_delta_job_s"],
            normalized_profiles[1]["flow_area_delta_job_s"],
        )
    ):
        raise ValueError("completed known FIFO state summary changed")
    return {
        "state_index": entry["state_index"],
        "queue": entry["queue"],
        "evidence_source": CLOSED_TRANSITION.name,
        "unique_state_flow_area_delta_job_s": source[
            "unique_state_flow_area_delta_job_s"
        ],
        "unique_state_mean_flow_delta_s": source[
            "unique_state_mean_flow_delta_s"
        ],
        "profiles": normalized_profiles,
    }


def _partition_cases(shadow, roster, model, transition):
    source_cases = {row["state_index"]: row for row in transition["cases"]}
    aliases, known, novel, reused = [], [], [], []
    seen = set()
    for change in shadow["state_changes"]:
        index = change["state_index"]
        if index in seen or not 0 <= index < len(roster["states"]):
            raise ValueError("anchor-star changed-state indices changed")
        seen.add(index)
        baseline_physical, baseline_fifo = _action_signatures(
            model, change["legacy_option_indices"]
        )
        variant_physical, variant_fifo = _action_signatures(
            model, change["variant_option_indices"]
        )
        raw_changed = change["legacy_option_indices"] != change["variant_option_indices"]
        physical_changed = baseline_physical != variant_physical
        fifo_changed = baseline_fifo != variant_fifo
        if (
            not raw_changed
            or change["raw_lane_witness_changed_from_legacy_k26"] is not raw_changed
            or change["selected_physical_signature_changed_from_legacy_k26"]
            is not physical_changed
            or change["fifo_execution_signature_changed_from_legacy_k26"]
            is not fifo_changed
            or change["legacy_physical_signature"] != baseline_physical
            or change["variant_physical_signature"] != variant_physical
            or change["legacy_fifo_execution_signature"] != baseline_fifo
            or change["variant_fifo_execution_signature"] != variant_fifo
        ):
            raise ValueError("anchor-star changed-action signatures changed")
        if not fifo_changed:
            entry = _case_entry(change, roster["states"][index], "fifo_alias")
            aliases.append(entry)
        elif change["known_quotient_fifo_pair"]:
            if not change["known_quotient_physical_and_raw_pair"]:
                raise ValueError("known FIFO pair lost its exact raw match")
            entry = _case_entry(change, roster["states"][index], "known_reuse")
            source = source_cases.get(index)
            if source is None:
                raise ValueError("known FIFO pair has no completed case")
            known.append(entry)
            reused.append(_validate_reused_case(source, entry))
        else:
            entry = _case_entry(change, roster["states"][index], "novel_unexecuted")
            novel.append(entry)
    if (
        len(seen) != CHANGED_STATE_COUNT
        or len(aliases) != FIFO_ALIAS_COUNT
        or len(known) != KNOWN_REUSE_COUNT
        or len(novel) != NOVEL_UNEXECUTED_COUNT
        or len(reused) != KNOWN_REUSE_COUNT
        or any(
            row["baseline_fifo_execution_signature"]
            != row["variant_fifo_execution_signature"]
            for row in aliases
        )
    ):
        raise ValueError("registered anchor-star evidence partition changed")
    return aliases, known, novel, reused


def _hard_veto_witnesses(reused):
    mean = [
        {
            "state_index": row["state_index"],
            "queue": row["queue"],
            "mean_flow_delta_s": row["unique_state_mean_flow_delta_s"],
            "flow_area_delta_job_s": row["unique_state_flow_area_delta_job_s"],
        }
        for row in reused
        if row["unique_state_mean_flow_delta_s"] > TOLERANCE
    ]
    p90 = [
        {
            "state_index": row["state_index"],
            "queue": row["queue"],
            "profile": profile["profile"],
            "p90_flow_delta_s": profile["p90_flow_delta_s"],
        }
        for row in reused
        for profile in row["profiles"]
        if profile["p90_flow_delta_s"] > TOLERANCE
    ]
    if len(mean) != KNOWN_MEAN_VETO_COUNT or len(p90) != KNOWN_P90_VETO_COUNT:
        raise ValueError("completed known hard-veto witness roster changed")
    return {"positive_mean_flow": mean, "positive_p90": p90}


def _primary_cells(roster, aliases, known, novel):
    partitions = {"fifo_alias": aliases, "known_reuse": known, "novel_unexecuted": novel}
    rows = []
    for load in LOADS:
        for seed in SEEDS:
            source_visits = [
                visit
                for state in roster["states"]
                for visit in state["population_visits"]
                if visit[0] == load and visit[1] == seed and visit[2] == BUDGET
            ]
            counts = {
                label: sum(
                    visit[2]
                    for case in cases
                    for visit in case["primary_population_visits"]
                    if visit[0] == load and visit[1] == seed
                )
                for label, cases in partitions.items()
            }
            source_count = sum(visit[3] for visit in source_visits)
            changed_count = sum(counts.values())
            rows.append({
                "load": load,
                "seed": seed,
                "source_frame_count": source_count,
                "source_visited_time_s": sum(visit[4] for visit in source_visits),
                "changed_action_frame_count": changed_count,
                "unchanged_action_frame_count": source_count - changed_count,
                "fifo_alias_frame_count": counts["fifo_alias"],
                "known_reuse_frame_count": counts["known_reuse"],
                "novel_unexecuted_frame_count": counts["novel_unexecuted"],
            })
    if len(rows) != 9 or any(
        row["changed_action_frame_count"]
        != row["fifo_alias_frame_count"]
        + row["known_reuse_frame_count"]
        + row["novel_unexecuted_frame_count"]
        or row["unchanged_action_frame_count"] < 0
        or row["known_reuse_frame_count"] <= 0
        for row in rows
    ):
        raise ValueError("primary load-seed evidence weights changed")
    return rows


def _known_cell_contributions(cells, known, reused):
    outcomes = {row["state_index"]: row for row in reused}
    summaries = []
    for cell in cells:
        load, seed = cell["load"], cell["seed"]
        weighted_area = 0.0
        weighted_mean = 0.0
        weighted_p90 = dict.fromkeys(AGE_PROFILES, 0.0)
        for case in known:
            count = sum(
                visit[2]
                for visit in case["primary_population_visits"]
                if visit[0] == load and visit[1] == seed
            )
            outcome = outcomes[case["state_index"]]
            weighted_area += count * outcome["unique_state_flow_area_delta_job_s"]
            weighted_mean += count * outcome["unique_state_mean_flow_delta_s"]
            for profile in outcome["profiles"]:
                weighted_p90[profile["profile"]] += count * profile["p90_flow_delta_s"]
        known_visits = cell["known_reuse_frame_count"]
        summaries.append({
            **cell,
            "known_reuse_weighted_flow_area_delta_job_s": weighted_area,
            "known_reuse_mean_flow_delta_s_per_known_visit": weighted_mean / known_visits,
            "known_reuse_p90_delta_s_per_known_visit_by_profile": {
                profile: value / known_visits for profile, value in weighted_p90.items()
            },
            "complete_anchor_weighted_effect": "NOT_EVALUATED_AFTER_HARD_VETO",
        })
    return summaries


def _build_preflight():
    shadow, shadow_protocol, roster, model = _validate_anchor_shadow()
    transition, transition_protocol = _validate_closed_transition(model)
    aliases, known, novel, reused = _partition_cases(
        shadow, roster, model, transition
    )
    witnesses = _hard_veto_witnesses(reused)
    cells = _primary_cells(roster, aliases, known, novel)
    source_primary = [
        row
        for row in shadow["summary_by_load_seed_and_source_policy"]
        if row["source_policy_budget"] == BUDGET
    ]
    if len(source_primary) != 9 or any(
        (left["load"], left["seed"], left["source_frame_count"])
        != (right["load"], right["seed"], right["frame_count"])
        or not _close(left["source_visited_time_s"], right["legacy_visited_time_s"])
        for left, right in zip(cells, source_primary, strict=True)
    ):
        raise ValueError("primary load-seed source populations changed")
    return {
        "shadow": shadow,
        "shadow_protocol": shadow_protocol,
        "transition": transition,
        "transition_protocol": transition_protocol,
        "model": model,
        "aliases": aliases,
        "known": known,
        "novel": novel,
        "reused": reused,
        "witnesses": witnesses,
        "cells": cells,
    }


def _protocol_payload(preflight):
    return {
        "experiment": "aligned_hardware_anchor_star_transition_evidence_reuse_closure_v1",
        "role": ROLE,
        "source_anchor_shadow": ANCHOR_SHADOW.name,
        "source_anchor_shadow_protocol": ANCHOR_SHADOW_PROTOCOL.name,
        "closed_transition": CLOSED_TRANSITION.name,
        "closed_transition_protocol": CLOSED_TRANSITION_PROTOCOL.name,
        "closed_transition_fixed_sources": CLOSED_TRANSITION_SOURCES.name,
        "registered_state_roster": REGISTERED_STATES.name,
        "budget": BUDGET,
        "loads": list(LOADS),
        "seeds": list(SEEDS),
        "age_profiles": list(AGE_PROFILES),
        "model": _json(preflight["model"].snapshot()),
        "changed_state_count": CHANGED_STATE_COUNT,
        "fifo_alias_count": FIFO_ALIAS_COUNT,
        "known_reuse_count": KNOWN_REUSE_COUNT,
        "novel_unexecuted_count": NOVEL_UNEXECUTED_COUNT,
        "registered_selector_call_count": 0,
        "registered_drain_count": 0,
        "partition_rule": PARTITION_RULE,
        "fifo_alias_roster": preflight["aliases"],
        "known_reuse_roster": preflight["known"],
        "novel_unexecuted_roster": preflight["novel"],
        "primary_load_seed_weight_roster": preflight["cells"],
        "known_reuse_hard_veto_witnesses": preflight["witnesses"],
        "advance_gate": ADVANCE_GATE,
        "short_circuit_rule": SHORT_CIRCUIT_RULE,
        "prohibited_in_this_stage": PROHIBITED,
        "interpretation": INTERPRETATION,
    }


def _decision(execution_passed, witnesses):
    if not execution_passed:
        return "INVALID_EXECUTION"
    if witnesses["positive_mean_flow"]:
        return "KNOWN_LOCAL_FLOW_TRADEOFF_STOP"
    if witnesses["positive_p90"]:
        return "KNOWN_TAIL_TRADEOFF_STOP"
    return "NOVEL_EVIDENCE_REQUIRED"


def register(output=DEFAULT_OUTPUT):
    output, protocol_path, sources_path = _paths(output)
    if any(path.exists() for path in (output, protocol_path, sources_path)):
        raise ValueError("anchor-star transition-closure artifacts already exist")
    preflight = _build_preflight()
    protocol = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        **_protocol_payload(preflight),
    }
    sources = _source_texts()
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_new(protocol_path, protocol)
    _write_new(sources_path, sources)
    return protocol


def run_registered(output=DEFAULT_OUTPUT):
    output, protocol_path, sources_path = _paths(output)
    if output.exists():
        raise ValueError("anchor-star transition-closure result already exists")
    protocol = json.loads(protocol_path.read_text())
    sources = json.loads(sources_path.read_text())

    # All predecessor, source, partition, weight, and outcome checks complete
    # before constructing the evidence-only report. There is no execution path.
    preflight = _build_preflight()
    expected = _protocol_payload(preflight)
    if (
        frozenset(protocol) != frozenset({"registered_at_utc", *expected})
        or any(protocol.get(key) != value for key, value in expected.items())
    ):
        raise ValueError("registered anchor-star transition-closure protocol changed")
    try:
        datetime.fromisoformat(protocol["registered_at_utc"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "registered anchor-star transition-closure timestamp changed"
        ) from error
    if sources != _source_texts():
        raise ValueError("registered anchor-star transition-closure sources changed")

    witnesses = preflight["witnesses"]
    execution_checks = {
        "completed_anchor_shadow_and_closed_transition_validated": True,
        "registered_1413_changed_states_partition_as_912_plus_14_plus_487": (
            len(preflight["aliases"])
            + len(preflight["known"])
            + len(preflight["novel"])
            == CHANGED_STATE_COUNT
        ),
        "registered_912_fifo_aliases_have_identical_execution_signatures": all(
            row["baseline_fifo_execution_signature"]
            == row["variant_fifo_execution_signature"]
            for row in preflight["aliases"]
        ),
        "registered_14_known_pairs_exactly_reuse_completed_outcomes": (
            len(preflight["reused"]) == KNOWN_REUSE_COUNT
        ),
        "registered_487_novel_pairs_remain_unexecuted": (
            len(preflight["novel"]) == NOVEL_UNEXECUTED_COUNT
        ),
        "six_positive_mean_flow_hard_veto_witnesses_reconstruct": (
            len(witnesses["positive_mean_flow"]) == KNOWN_MEAN_VETO_COUNT
        ),
        "twelve_positive_p90_hard_veto_witnesses_reconstruct": (
            len(witnesses["positive_p90"]) == KNOWN_P90_VETO_COUNT
        ),
        "nine_primary_load_seed_weights_partition": len(preflight["cells"]) == 9,
        "advance_conjunction_is_false_before_novel_measurement": bool(
            witnesses["positive_mean_flow"] or witnesses["positive_p90"]
        ),
        "no_selector_drain_arrival_trajectory_or_randomness_used": True,
        "registered_sources_unchanged": sources == _source_texts(),
    }
    passed = all(execution_checks.values())
    decision = _decision(passed, witnesses)
    report = {
        "experiment": "aligned_hardware_anchor_star_transition_evidence_reuse_closure_v1",
        "status": "PASS" if passed else "FAIL",
        "checks": execution_checks,
        "protocol": protocol,
        "evidence_partition": {
            "changed_state_count": CHANGED_STATE_COUNT,
            "fifo_alias_zero_effect_count": len(preflight["aliases"]),
            "known_reused_outcome_count": len(preflight["reused"]),
            "novel_unexecuted_count": len(preflight["novel"]),
        },
        "known_reused_cases": preflight["reused"],
        "hard_veto_witnesses": witnesses,
        "summary_by_primary_load_seed": _known_cell_contributions(
            preflight["cells"], preflight["known"], preflight["reused"]
        ),
        "transition_advance_gate_status": {
            "execution_passed": passed,
            "all_1413_changed_state_mean_flow_deltas_nonpositive": False,
            "all_2826_changed_state_profile_nearest_rank_p90_deltas_nonpositive": False,
            "all_9_primary_load_seed_weighted_flow_area_deltas_nonpositive": "NOT_EVALUATED_AFTER_HARD_VETO",
            "strict_local_and_weighted_gain_conditions": "NOT_EVALUATED_AFTER_HARD_VETO",
            "advance_conjunction_satisfied": False,
        },
        "work_accounting": {
            "selector_calls": 0,
            "authored_drain_calls": 0,
            "arrival_tapes": 0,
            "trajectory_frames": 0,
            "novel_fifo_pairs_executed": 0,
            "completed_case_outcomes_reused": len(preflight["reused"]),
            "completed_drain_calls_reused": (
                len(preflight["reused"]) * len(AGE_PROFILES) * 2
            ),
            "fifo_alias_zero_effect_proofs": len(preflight["aliases"]),
            "skipped_novel_drain_calls": (
                len(preflight["novel"]) * len(AGE_PROFILES) * 2
            ),
            "rng_initializations": 0,
            "random_draws": 0,
        },
        "scientific_decision": decision,
        "development_advance": False,
        "novel_drain_skip_reason": (
            "Six exact reused states already violate the mandatory per-state mean-flow "
            "gate and twelve exact reused state-profile outcomes violate the mandatory "
            "P90 gate; no novel outcome can make either universal condition true."
        ),
        "next_step": (
            "close the anchor-star branch without novel drains, tuning, full "
            "trajectories, or paper changes"
        ),
        "limitations": [
            "The 487 novel FIFO-effective effects remain unmeasured because they cannot change the frozen conjunctive decision.",
            "Known outcomes are reused only for exact state, queue, raw-action, and FIFO-execution matches.",
            "The nine-cell complete weighted effects are intentionally not estimated after the hard veto.",
        ],
    }
    if decision != "KNOWN_LOCAL_FLOW_TRADEOFF_STOP":
        raise ValueError("registered hard-veto decision changed")
    if sources != _source_texts():
        raise ValueError("registered sources changed during evidence closure")
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
