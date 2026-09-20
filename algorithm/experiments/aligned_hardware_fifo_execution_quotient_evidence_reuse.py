"""Close the FIFO-execution quotient branch from completed local evidence.

The fixed-state shadow changes four of 5,003 K26 actions.  Every changed
queue/action pair exactly matches a completed permutation-quotient transition
case, so this stage reuses those four complete drains and executes no new
selector, drain, arrival process, trajectory, or random draw.

Registration freezes only lineage, directional match keys, occurrence weights,
and the inherited transition gate.  It deliberately does not open the prior
transition outcome; that outcome is read and validated only by the registered
execution.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from algorithm.experiments import (
    aligned_hardware_fifo_execution_quotient_shadow as fifo_shadow,
)
from algorithm.experiments import (
    aligned_hardware_permutation_quotient_transition as quotient_transition,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
DEFAULT_OUTPUT = ARTIFACTS / (
    "aligned_hardware_fifo_execution_quotient_evidence_reuse_development_20260911.json"
)
FIFO_SHADOW = ARTIFACTS / (
    "aligned_hardware_fifo_execution_quotient_shadow_development_20260911.json"
)
FIFO_SHADOW_PROTOCOL = ARTIFACTS / (
    "aligned_hardware_fifo_execution_quotient_shadow_development_20260911_protocol.json"
)
FIFO_SHADOW_SOURCES = ARTIFACTS / (
    "aligned_hardware_fifo_execution_quotient_shadow_development_20260911_fixed_sources.json"
)
CLOSED_TRANSITION = ARTIFACTS / (
    "aligned_hardware_permutation_quotient_transition_development_20260911.json"
)
CLOSED_TRANSITION_PROTOCOL = ARTIFACTS / (
    "aligned_hardware_permutation_quotient_transition_development_20260911_protocol.json"
)
CLOSED_TRANSITION_SOURCES = ARTIFACTS / (
    "aligned_hardware_permutation_quotient_transition_development_20260911_fixed_sources.json"
)
REGISTERED_STATES = fifo_shadow.REGISTERED_STATES

BUDGET = 26
LOADS = (0.50, 0.80, 0.95)
SEEDS = (41, 42, 43)
AGE_PROFILES = ("coincident", "unit_fifo")
TOLERANCE = 1e-9
STATE_COUNT = 5003
RAW_UNCHANGED_COUNT = 4999
KNOWN_REUSE_COUNT = 4
NOVEL_COUNT = 0

ROLE = "development evidence-reuse closure for the FIFO-execution quotient screen"
REUSE_RULE = {
    "match": (
        "same registered state index and queue, with the same directed baseline "
        "and variant raw, physical, and FIFO-execution signatures"
    ),
    "outcome": (
        "reuse the completed coincident and unit-FIFO drains in which only the "
        "first action differs and both arms then use v4 K26 until empty"
    ),
    "unchanged": (
        "a raw-unchanged first action has zero local transition effect under the "
        "same deterministic continuation"
    ),
}
ADVANCE_GATE = {
    "all_execution_checks": True,
    "all_unique_state_mean_flow_deltas_nonpositive": True,
    "all_state_profile_nearest_rank_p90_deltas_nonpositive": True,
    "all_9_primary_load_seed_weighted_flow_area_deltas_nonpositive": True,
    "at_least_one_unique_state_has_strict_mean_flow_improvement": True,
    "at_least_one_primary_load_seed_cell_has_strict_weighted_flow_area_improvement": True,
    "logic": "all conditions are conjunctive",
}
DECISION_RULE = (
    "invalid evidence gives INVALID_EXECUTION; a positive unique-state mean-flow "
    "or primary-cell weighted-flow effect gives KNOWN_LOCAL_FLOW_TRADEOFF_STOP; otherwise "
    "a positive state-profile nearest-rank P90 effect gives KNOWN_TAIL_TRADEOFF_STOP; "
    "absence of both required strict gains gives NO_LOCAL_FLOW_BENEFIT_STOP; only "
    "the full conjunction gives ADVANCE_TO_FULL_DEVELOPMENT_TRAJECTORIES"
)
PROHIBITED = (
    "selector calls, authored drains, arrival generation, trajectory simulation, "
    "RNG, new states, new age profiles, outcome-tuned filtering, and paper edits"
)
INTERPRETATION = (
    "The reused drains test the frozen one-action intervention followed by common "
    "v4 K26 continuation. They do not estimate a full FIFO-selector feedback trajectory."
)
NEW_SOURCES = (
    "algorithm/experiments/aligned_hardware_fifo_execution_quotient_evidence_reuse.py",
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


def _close(left, right):
    try:
        return math.isclose(
            float(left), float(right), rel_tol=0.0, abs_tol=TOLERANCE
        )
    except (TypeError, ValueError):
        return False


def _source_texts():
    shadow_sources = json.loads(FIFO_SHADOW_SOURCES.read_text())
    if shadow_sources != fifo_shadow._source_texts():
        raise ValueError("FIFO shadow source map changed")
    transition_sources = json.loads(CLOSED_TRANSITION_SOURCES.read_text())
    if transition_sources != quotient_transition._source_texts():
        raise ValueError("completed transition source map changed")
    conflicts = [
        name
        for name in shadow_sources.keys() & transition_sources.keys()
        if shadow_sources[name] != transition_sources[name]
    ]
    if conflicts:
        raise ValueError(f"frozen predecessor source maps disagree: {conflicts}")
    return {
        **transition_sources,
        **shadow_sources,
        **{name: (ROOT / name).read_text() for name in NEW_SOURCES},
    }


def _validate_shadow():
    report = json.loads(FIFO_SHADOW.read_text())
    protocol = json.loads(FIFO_SHADOW_PROTOCOL.read_text())
    fixed_sources = json.loads(FIFO_SHADOW_SOURCES.read_text())
    diagnostic, model = fifo_shadow._validate_source_diagnostic()
    roster = fifo_shadow._validated_roster(diagnostic)
    partition = report.get("evidence_partition", {})
    counts = partition.get("unique_state_counts", {})
    if (
        fixed_sources != fifo_shadow._source_texts()
        or report.get("protocol") != protocol
        or report.get("experiment")
        != "aligned_hardware_fifo_execution_quotient_state_shadow_v1"
        or report.get("status") != "PASS"
        or len(report.get("checks", {})) != 16
        or not all(report.get("checks", {}).values())
        or len(report.get("shadow_advance_checks", {})) != 18
        or not all(report.get("shadow_advance_checks", {}).values())
        or report.get("scientific_decision")
        != "ADVANCE_TO_EVIDENCE_REUSE_OR_AUTHORED_TRANSITION"
        or report.get("development_advance") is not True
        or report.get("state_evaluation_count") != STATE_COUNT
        or report.get("raw_change_state_count") != KNOWN_REUSE_COUNT
        or report.get("physical_change_state_count") != KNOWN_REUSE_COUNT
        or report.get("fifo_effective_change_state_count") != KNOWN_REUSE_COUNT
        or len(report.get("state_changes", ())) != KNOWN_REUSE_COUNT
        or len(partition.get("known_protocol_matches", ())) != KNOWN_REUSE_COUNT
        or len(partition.get("novel_fifo_effective_roster", ())) != NOVEL_COUNT
        or counts.get("RAW_UNCHANGED") != RAW_UNCHANGED_COUNT
        or counts.get("PHYSICAL_CHANGED_KNOWN_PROTOCOL") != KNOWN_REUSE_COUNT
        or sum(counts.values()) != STATE_COUNT
        or protocol.get("registered_state_roster") != REGISTERED_STATES.name
        or protocol.get("loads") != list(LOADS)
        or protocol.get("seeds") != list(SEEDS)
        or protocol.get("small_checkpoint_budget") != BUDGET
        or protocol.get("state_count") != STATE_COUNT
        or protocol.get("model") != _json(model.snapshot())
        or protocol.get("rule") != fifo_shadow.RULE
        or protocol.get("advance_gate") != fifo_shadow.ADVANCE_GATE
        or protocol.get("stopping_rule") != fifo_shadow.STOPPING_RULE
        or protocol.get("prohibited_in_this_stage") != fifo_shadow.PROHIBITED
    ):
        raise ValueError("closure requires the completed advancing FIFO shadow")
    return report, protocol, roster, model


def _validate_transition_protocol(model):
    protocol = json.loads(CLOSED_TRANSITION_PROTOCOL.read_text())
    fixed_sources = json.loads(CLOSED_TRANSITION_SOURCES.read_text())
    _predecessor, _roster, validated_model, cases = (
        quotient_transition._validate_predecessor()
    )
    maximum_duration = max(
        1.0, *(float(option.duration) for option in validated_model.options)
    )
    maximum_jobs = max(sum(case["queue"]) for case in cases)
    expected = {
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
        fixed_sources != quotient_transition._source_texts()
        or _json(validated_model.snapshot()) != _json(model.snapshot())
        or frozenset(protocol) != frozenset({"registered_at_utc", *expected})
        or any(protocol.get(key) != value for key, value in expected.items())
    ):
        raise ValueError("completed transition protocol is unavailable or changed")
    try:
        datetime.fromisoformat(protocol["registered_at_utc"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("completed transition registration timestamp changed") from error
    return protocol, cases


def _primary_visits(state):
    return [
        [visit[0], visit[1], visit[3], visit[4]]
        for visit in state["population_visits"]
        if visit[2] == BUDGET
    ]


def _match_roster(shadow_report, roster, model, transition_cases):
    cases_by_state = {case["state_index"]: case for case in transition_cases}
    if len(cases_by_state) != len(transition_cases):
        raise ValueError("completed transition protocol state indices are not unique")
    entries = []
    seen = set()
    for change in shadow_report["evidence_partition"]["known_protocol_matches"]:
        state_index = change.get("state_index")
        if (
            not isinstance(state_index, int)
            or not 0 <= state_index < len(roster["states"])
            or state_index in seen
        ):
            raise ValueError("FIFO reuse state indices changed")
        seen.add(state_index)
        state = roster["states"][state_index]
        baseline = change["legacy_signatures"]
        variant = change["variant_k26_signatures"]
        rebuilt_baseline, _ = fifo_shadow._action_signatures(
            model, baseline["raw"]
        )
        rebuilt_variant, _ = fifo_shadow._action_signatures(model, variant["raw"])
        source_case = cases_by_state.get(state_index)
        pair_id = f"quotient_case_{state_index}"
        if (
            change.get("evidence_partition")
            != "PHYSICAL_CHANGED_KNOWN_PROTOCOL"
            or change.get("known_protocol_fifo_pair_ids") != [pair_id]
            or change.get("known_protocol_raw_pair_ids") != [pair_id]
            or change.get("known_protocol_physical_pair_ids") != [pair_id]
            or change.get("queue") != state["queue"]
            or baseline != rebuilt_baseline
            or variant != rebuilt_variant
            or baseline["raw"] == variant["raw"]
            or baseline["fifo_execution"] == variant["fifo_execution"]
            or state["legacy_k26"]["option_indices"] != baseline["raw"]
            or source_case is None
            or source_case.get("queue") != state["queue"]
            or source_case.get("legacy_option_indices") != baseline["raw"]
            or source_case.get("variant_option_indices") != variant["raw"]
        ):
            raise ValueError("FIFO shadow change does not match its transition protocol")
        entries.append({
            "pair_id": pair_id,
            "state_index": state_index,
            "queue": state["queue"],
            "primary_population_visits": _primary_visits(state),
            "baseline": baseline,
            "variant": variant,
            "registered_support_score_delta": change[
                "support_score_delta_over_legacy_k26"
            ],
        })
    entries.sort(key=lambda row: row["state_index"])
    if len(entries) != KNOWN_REUSE_COUNT:
        raise ValueError("FIFO reuse roster no longer contains four exact matches")
    return entries


def _primary_cell_weights(shadow_report, match_roster):
    source_rows = [
        row
        for row in shadow_report["summary_by_load_seed_and_source_policy"]
        if row["source_policy_budget"] == BUDGET
    ]
    source_by_id = {(row["load"], row["seed"]): row for row in source_rows}
    cells = []
    for load in LOADS:
        for seed in SEEDS:
            source = source_by_id.get((load, seed))
            matched = sum(
                visit[2]
                for entry in match_roster
                for visit in entry["primary_population_visits"]
                if visit[0] == load and visit[1] == seed
            )
            if source is None or matched > source["frame_count"]:
                raise ValueError("primary FIFO reuse weights changed")
            cells.append({
                "load": load,
                "seed": seed,
                "source_frame_count": source["frame_count"],
                "source_visited_time_s": source["legacy_visited_time_s"],
                "matched_change_frame_count": matched,
                "raw_unchanged_frame_count": source["frame_count"] - matched,
            })
    if len(cells) != 9 or sum(row["matched_change_frame_count"] for row in cells) != 61:
        raise ValueError("primary FIFO reuse cell partition changed")
    return cells


def _build_preflight():
    shadow_report, shadow_protocol, roster, model = _validate_shadow()
    transition_protocol, transition_cases = _validate_transition_protocol(model)
    matches = _match_roster(shadow_report, roster, model, transition_cases)
    cells = _primary_cell_weights(shadow_report, matches)
    return {
        "shadow_report": shadow_report,
        "shadow_protocol": shadow_protocol,
        "transition_protocol": transition_protocol,
        "model": model,
        "matches": matches,
        "cells": cells,
    }


def _protocol_payload(preflight):
    return {
        "experiment": "aligned_hardware_fifo_execution_quotient_evidence_reuse_v1",
        "role": ROLE,
        "source_fifo_shadow": FIFO_SHADOW.name,
        "source_fifo_shadow_protocol": FIFO_SHADOW_PROTOCOL.name,
        "closed_transition_result": CLOSED_TRANSITION.name,
        "closed_transition_protocol": CLOSED_TRANSITION_PROTOCOL.name,
        "closed_transition_fixed_sources": CLOSED_TRANSITION_SOURCES.name,
        "registered_state_roster": REGISTERED_STATES.name,
        "budget": BUDGET,
        "loads": list(LOADS),
        "seeds": list(SEEDS),
        "age_profiles": list(AGE_PROFILES),
        "model": _json(preflight["model"].snapshot()),
        "state_count": STATE_COUNT,
        "raw_unchanged_count": RAW_UNCHANGED_COUNT,
        "known_reuse_count": KNOWN_REUSE_COUNT,
        "novel_count": NOVEL_COUNT,
        "registered_selector_call_count": 0,
        "registered_drain_count": 0,
        "reuse_rule": REUSE_RULE,
        "known_reuse_roster": preflight["matches"],
        "primary_load_seed_weight_roster": preflight["cells"],
        "advance_gate": ADVANCE_GATE,
        "decision_rule": DECISION_RULE,
        "prohibited_in_this_stage": PROHIBITED,
        "interpretation": INTERPRETATION,
    }


def _validate_transition_outcome(transition_protocol, matches):
    report = json.loads(CLOSED_TRANSITION.read_text())
    cases = report.get("cases", ())
    if (
        report.get("protocol") != transition_protocol
        or report.get("experiment")
        != "aligned_hardware_permutation_quotient_authored_transition_v1"
        or report.get("status") != "PASS"
        or len(report.get("checks", {})) != 10
        or not all(report.get("checks", {}).values())
        or len(cases) != 16
        or report.get("work_accounting", {}).get("future_arrivals") != 0
        or report.get("work_accounting", {}).get("random_draws") != 0
        or report.get("work_accounting", {}).get("rng_initializations") != 0
    ):
        raise ValueError("completed transition outcome is unavailable or changed")
    cases_by_state = {case["state_index"]: case for case in cases}
    if len(cases_by_state) != len(cases):
        raise ValueError("completed transition outcome states are not unique")
    reused = []
    for entry in matches:
        source = cases_by_state.get(entry["state_index"])
        checks = source.get("checks", {}) if source else {}
        if (
            source is None
            or source.get("queue") != entry["queue"]
            or source.get("legacy_option_indices") != entry["baseline"]["raw"]
            or source.get("variant_option_indices") != entry["variant"]["raw"]
            or source.get("primary_population_visits")
            != entry["primary_population_visits"]
            or not _close(
                source.get("registered_support_score_delta"),
                entry["registered_support_score_delta"],
            )
            or len(checks) != 6
            or not all(checks.values())
        ):
            raise ValueError("known FIFO pair does not exactly match its completed case")
        profiles = source.get("profiles", ())
        if [row.get("profile") for row in profiles] != list(AGE_PROFILES):
            raise ValueError("known FIFO outcome age profiles changed")
        normalized_profiles = []
        for row in profiles:
            baseline = row.get("baseline", {})
            variant = row.get("variant", {})
            if (
                row.get("baseline_first_action", {}).get("option_indices")
                != entry["baseline"]["raw"]
                or row.get("variant_first_action", {}).get("option_indices")
                != entry["variant"]["raw"]
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
                raise ValueError("completed FIFO outcome metrics changed")
            normalized_profiles.append({
                "profile": row["profile"],
                "flow_area_delta_job_s": row["flow_area_delta_job_s"],
                "mean_flow_delta_s": row["mean_flow_delta_s"],
                "p90_flow_delta_s": row["p90_flow_delta_s"],
                "makespan_delta_s": row["makespan_delta_s"],
                "baseline_frame_count": baseline["frame_count"],
                "variant_frame_count": variant["frame_count"],
                "baseline_continuation_selector_calls": baseline[
                    "continuation_selector_calls"
                ],
                "variant_continuation_selector_calls": variant[
                    "continuation_selector_calls"
                ],
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
            raise ValueError("completed FIFO state summary changed")
        reused.append({
            "pair_id": entry["pair_id"],
            "state_index": entry["state_index"],
            "queue": entry["queue"],
            "baseline_option_indices": entry["baseline"]["raw"],
            "variant_option_indices": entry["variant"]["raw"],
            "evidence_source": CLOSED_TRANSITION.name,
            "unique_state_flow_area_delta_job_s": source[
                "unique_state_flow_area_delta_job_s"
            ],
            "unique_state_mean_flow_delta_s": source[
                "unique_state_mean_flow_delta_s"
            ],
            "profiles": normalized_profiles,
        })
    return reused


def _cell_outcomes(cells, matches, reused):
    outcomes = {row["state_index"]: row for row in reused}
    rows = []
    for cell in cells:
        load, seed = cell["load"], cell["seed"]
        matched_count = cell["matched_change_frame_count"]
        total_area = 0.0
        total_mean = 0.0
        total_makespan = 0.0
        total_p90 = dict.fromkeys(AGE_PROFILES, 0.0)
        for match in matches:
            count = sum(
                visit[2]
                for visit in match["primary_population_visits"]
                if visit[0] == load and visit[1] == seed
            )
            outcome = outcomes[match["state_index"]]
            total_area += count * outcome["unique_state_flow_area_delta_job_s"]
            total_mean += count * outcome["unique_state_mean_flow_delta_s"]
            for profile in outcome["profiles"]:
                total_p90[profile["profile"]] += count * profile["p90_flow_delta_s"]
                if profile["profile"] == AGE_PROFILES[0]:
                    total_makespan += count * profile["makespan_delta_s"]
        rows.append({
            **cell,
            "occurrence_weighted_flow_area_delta_job_s": total_area,
            "occurrence_weighted_flow_area_delta_job_s_per_source_frame": (
                total_area / cell["source_frame_count"]
            ),
            "mean_flow_delta_s_per_matched_occurrence": (
                total_mean / matched_count if matched_count else None
            ),
            "makespan_delta_s_per_matched_occurrence": (
                total_makespan / matched_count if matched_count else None
            ),
            "p90_delta_s_per_matched_occurrence_by_profile": {
                profile: total / matched_count if matched_count else None
                for profile, total in total_p90.items()
            },
        })
    return rows


def _gate_status(reused, cells, execution_passed):
    return {
        "execution_passed": execution_passed,
        "all_5003_unique_state_mean_flow_deltas_nonpositive": all(
            row["unique_state_mean_flow_delta_s"] <= TOLERANCE for row in reused
        ),
        "all_10006_state_profile_nearest_rank_p90_deltas_nonpositive": all(
            profile["p90_flow_delta_s"] <= TOLERANCE
            for row in reused
            for profile in row["profiles"]
        ),
        "all_9_primary_load_seed_weighted_flow_area_deltas_nonpositive": all(
            row["occurrence_weighted_flow_area_delta_job_s"] <= TOLERANCE
            for row in cells
        ),
        "at_least_one_unique_state_has_strict_mean_flow_improvement": any(
            row["unique_state_mean_flow_delta_s"] < -TOLERANCE for row in reused
        ),
        "at_least_one_primary_load_seed_cell_has_strict_weighted_flow_area_improvement": any(
            row["occurrence_weighted_flow_area_delta_job_s"] < -TOLERANCE
            for row in cells
        ),
    }


def _decision(checks):
    if not checks["execution_passed"]:
        return "INVALID_EXECUTION"
    if (
        not checks["all_5003_unique_state_mean_flow_deltas_nonpositive"]
        or not checks[
            "all_9_primary_load_seed_weighted_flow_area_deltas_nonpositive"
        ]
    ):
        return "KNOWN_LOCAL_FLOW_TRADEOFF_STOP"
    if not checks[
        "all_10006_state_profile_nearest_rank_p90_deltas_nonpositive"
    ]:
        return "KNOWN_TAIL_TRADEOFF_STOP"
    if (
        not checks["at_least_one_unique_state_has_strict_mean_flow_improvement"]
        or not checks[
            "at_least_one_primary_load_seed_cell_has_strict_weighted_flow_area_improvement"
        ]
    ):
        return "NO_LOCAL_FLOW_BENEFIT_STOP"
    return "ADVANCE_TO_FULL_DEVELOPMENT_TRAJECTORIES"


def register(output=DEFAULT_OUTPUT):
    output, protocol_path, sources_path = _paths(output)
    if any(path.exists() for path in (output, protocol_path, sources_path)):
        raise ValueError("FIFO evidence-reuse artifacts already exist")

    # This preflight reads the advancing shadow and the completed transition
    # protocol, but never the completed transition outcome.
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
        raise ValueError("FIFO evidence-reuse result already exists")
    protocol = json.loads(protocol_path.read_text())
    sources = json.loads(sources_path.read_text())

    preflight = _build_preflight()
    expected = _protocol_payload(preflight)
    if (
        frozenset(protocol) != frozenset({"registered_at_utc", *expected})
        or any(protocol.get(key) != value for key, value in expected.items())
    ):
        raise ValueError("registered FIFO evidence-reuse protocol changed")
    try:
        datetime.fromisoformat(protocol["registered_at_utc"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("registered FIFO evidence-reuse timestamp changed") from error
    if sources != _source_texts():
        raise ValueError("registered FIFO evidence-reuse sources changed")

    reused = _validate_transition_outcome(
        preflight["transition_protocol"], preflight["matches"]
    )
    cells = _cell_outcomes(preflight["cells"], preflight["matches"], reused)
    p90_witnesses = [
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
    mean_witnesses = [
        {
            "state_index": row["state_index"],
            "queue": row["queue"],
            "mean_flow_delta_s": row["unique_state_mean_flow_delta_s"],
        }
        for row in reused
        if row["unique_state_mean_flow_delta_s"] > TOLERANCE
    ]
    execution_checks = {
        "advancing_fifo_shadow_and_transition_protocol_validated": True,
        "registered_5003_states_partition_as_4999_unchanged_plus_4_known": (
            RAW_UNCHANGED_COUNT + len(reused) + NOVEL_COUNT == STATE_COUNT
        ),
        "registered_four_directional_pairs_exactly_reuse_completed_outcomes": (
            len(reused) == KNOWN_REUSE_COUNT
        ),
        "all_fifo_effective_changes_have_completed_evidence": (
            len(reused) == KNOWN_REUSE_COUNT and NOVEL_COUNT == 0
        ),
        "nine_primary_load_seed_weights_reconstruct": (
            len(cells) == 9
            and sum(row["matched_change_frame_count"] for row in cells) == 61
        ),
        "all_completed_case_execution_and_flow_identities_hold": all(
            len(row["profiles"]) == len(AGE_PROFILES) for row in reused
        ),
        "no_selector_drain_arrival_trajectory_or_randomness_used": True,
        "registered_sources_unchanged": sources == _source_texts(),
    }
    passed = all(execution_checks.values())
    gates = _gate_status(reused, cells, passed)
    decision = _decision(gates)
    total_matched = sum(row["matched_change_frame_count"] for row in cells)
    total_area = sum(
        row["occurrence_weighted_flow_area_delta_job_s"] for row in cells
    )
    weighted_mean_numerator = sum(
        sum(
            visit[2]
            for visit in match["primary_population_visits"]
            if visit[0] == cell["load"] and visit[1] == cell["seed"]
        )
        * next(
            outcome["unique_state_mean_flow_delta_s"]
            for outcome in reused
            if outcome["state_index"] == match["state_index"]
        )
        for cell in cells
        for match in preflight["matches"]
    )
    report = {
        "experiment": "aligned_hardware_fifo_execution_quotient_evidence_reuse_v1",
        "status": "PASS" if passed else "FAIL",
        "checks": execution_checks,
        "protocol": protocol,
        "evidence_partition": {
            "raw_unchanged_zero_effect_count": RAW_UNCHANGED_COUNT,
            "known_reused_outcome_count": len(reused),
            "novel_unexecuted_count": NOVEL_COUNT,
        },
        "known_reused_cases": reused,
        "positive_mean_flow_witnesses": mean_witnesses,
        "positive_p90_witnesses": p90_witnesses,
        "summary_by_primary_load_seed": cells,
        "summary": {
            "unique_state_count": STATE_COUNT,
            "changed_state_count": len(reused),
            "matched_primary_occurrence_count": total_matched,
            "mean_flow_improved_changed_state_count": sum(
                row["unique_state_mean_flow_delta_s"] < -TOLERANCE
                for row in reused
            ),
            "mean_flow_worsened_changed_state_count": len(mean_witnesses),
            "positive_p90_state_profile_count": len(p90_witnesses),
            "occurrence_weighted_flow_area_delta_job_s": total_area,
            "occurrence_weighted_flow_area_delta_job_s_per_matched_occurrence": (
                total_area / total_matched
            ),
            "occurrence_weighted_mean_flow_delta_s_per_matched_occurrence": (
                weighted_mean_numerator / total_matched
            ),
        },
        "transition_advance_gate_status": gates,
        "work_accounting": {
            "selector_calls": 0,
            "authored_drain_calls": 0,
            "arrival_tapes": 0,
            "trajectory_frames": 0,
            "completed_case_outcomes_reused": len(reused),
            "completed_drain_calls_reused": (
                len(reused) * len(AGE_PROFILES) * 2
            ),
            "novel_fifo_pairs_executed": 0,
            "rng_initializations": 0,
            "random_draws": 0,
        },
        "scientific_decision": decision,
        "development_advance": decision
        == "ADVANCE_TO_FULL_DEVELOPMENT_TRAJECTORIES",
        "next_step": (
            "close the FIFO-execution quotient branch without tuning, new drains, "
            "full trajectories, or paper changes"
            if decision != "ADVANCE_TO_FULL_DEVELOPMENT_TRAJECTORIES"
            else "preregister full development trajectories"
        ),
        "limitations": [
            "This is deterministic reuse of consumed development evidence, not independent confirmation.",
            "The result covers a forced first action followed by common v4 K26 continuation, not full-policy feedback.",
            "Reused flow outcomes do not transfer selector work or cache accounting.",
        ],
    }
    if sources != _source_texts():
        raise ValueError("registered sources changed during FIFO evidence reuse")
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
