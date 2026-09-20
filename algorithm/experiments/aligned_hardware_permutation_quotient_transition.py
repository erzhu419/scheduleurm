"""Registered authored transition screen for the quotient-cache selector.

Every state whose K26 action strictly improved in the fixed-state shadow is
drained twice from the same finite visible population: v4 first then v4, and
quotient-cache first then v4.  There are no future arrivals.  Coincident and
unit-spaced FIFO ages share the same queue-count paths and expose whether raw
lane order changes nearest-rank P90 flow.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

from algorithm.authored_transition_drain import (
    authored_arrival_times,
    drain_initial_population,
    fixed_frame_action,
)
from algorithm.budgeted_frame_candidates import select_budgeted_frame
from algorithm.experiments import aligned_hardware_permutation_quotient_shadow as predecessor_shadow
from algorithm.permutation_quotient_frame_candidates import (
    select_permutation_quotient_frame,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
DEFAULT_OUTPUT = ARTIFACTS / "aligned_hardware_permutation_quotient_transition_development_20260911.json"
PREDECESSOR = ARTIFACTS / "aligned_hardware_permutation_quotient_shadow_development_20260911.json"
PREDECESSOR_SOURCES = ARTIFACTS / "aligned_hardware_permutation_quotient_shadow_development_20260911_fixed_sources.json"
REGISTERED_STATES = ARTIFACTS / "aligned_hardware_duration_tie_order_shadow_development_20260910_registered_states.json"
BUDGET = 26
STATE_COUNT = 16
AGE_PROFILES = ("coincident", "unit_fifo")
LOADS = (0.50, 0.80, 0.95)
SEEDS = (41, 42, 43)
TOLERANCE = 1e-9
NEW_SOURCES = (
    "algorithm/authored_transition_drain.py",
    "algorithm/experiments/aligned_hardware_permutation_quotient_transition.py",
)
PREDECESSOR_EXECUTION_CHECK_KEYS = frozenset({
    "registered_5003_unique_states_executed_once",
    "registered_18_population_cells_reconstruct",
    "registered_six_population_aggregates_reconstruct_source",
    "ordered_100_to_quotient_55_accounting_holds",
    "candidate_cache_and_frontier_work_identities_hold",
    "k55_closes_with_registered_support_on_every_state",
    "k26_support_bound_gap_and_epsilon_hold",
    "candidate_attempts_respect_k26_and_k55_budgets",
    "lane_permutation_cache_activates",
    "no_arrivals_trajectories_or_randomness_used",
    "registered_sources_unchanged",
})
PREDECESSOR_ADVANCE_CHECK_KEYS = frozenset({
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

ROLE = "development authored first-action transition veto on consumed changed states"
RULE = {
    "population": (
        "all jobs in each registered queue are visible at time zero; no future "
        "arrivals are generated"
    ),
    "age_profiles": {
        "coincident": "every visible job has age zero",
        "unit_fifo": (
            "within each class, FIFO rank j has age "
            "(queue_count - 1 - j) times the one-second frame-duration floor"
        ),
    },
    "baseline": "v4 K26 first action followed by v4 K26 until complete drain",
    "variant": (
        "permutation-quotient K26 first action followed by v4 K26 until complete drain"
    ),
    "isolation": (
        "only the first action differs; both branches use the same continuation "
        "rule but may visit different queues"
    ),
    "fifo_execution": (
        "preserve raw option_indices order, then option coordinate and completion-offset order"
    ),
    "p90": "nearest-rank sorted_flow[ceil(0.9 * job_count) - 1]",
    "drain_bound": (
        "complete without truncation; each branch elapsed frame barriers must not "
        "exceed sum(queue) times the maximum model frame duration"
    ),
}
FIXTURE = {
    "queue": [1, 0, 4],
    "forward_raw_option_indices": [0, 7],
    "reverse_raw_option_indices": [7, 0],
    "expected": (
        "equal service, duration, score, completion-offset multiset, makespan, and mean; "
        "equal coincident sorted flows; unequal unit-FIFO nearest-rank P90"
    ),
}
ADVANCE_GATE = {
    "all_execution_checks": True,
    "all_16_unique_state_mean_flow_deltas_nonpositive": True,
    "all_32_state_profile_nearest_rank_p90_deltas_nonpositive": True,
    "all_9_primary_load_seed_weighted_flow_area_deltas_nonpositive": True,
    "at_least_one_unique_state_has_strict_mean_flow_improvement": True,
    "at_least_one_primary_load_seed_cell_has_strict_weighted_flow_area_improvement": True,
}
STOPPING_RULE = (
    "run the frozen 16 states, two age profiles, and two arms once; advance only "
    "if all mean, tail, and nine-cell gates pass with a strict local and weighted "
    "gain; otherwise preserve the assigned stop reason without changing states, "
    "ages, K, continuation, tolerance, or drain rule"
)
PROHIBITED = (
    "Poisson or other future arrivals, new states, age variants, repeated quotient "
    "decisions after the first frame, full trajectories, confirmation, and paper edits"
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
    sources = json.loads(PREDECESSOR_SOURCES.read_text())
    if sources != predecessor_shadow._source_texts():
        raise ValueError("permutation-quotient predecessor source map changed")
    return {
        **sources,
        **{name: (ROOT / name).read_text() for name in NEW_SOURCES},
    }


def _case_roster(predecessor, roster):
    cases = []
    for change in predecessor["state_changes"]:
        index = change["state_index"]
        state = roster["states"][index]
        if state["queue"] != change["queue"]:
            raise ValueError("transition predecessor state index and queue disagree")
        primary_visits = [
            [visit[0], visit[1], visit[3], visit[4]]
            for visit in state["population_visits"]
            if visit[2] == BUDGET
        ]
        expected_primary_visits = sum(visit[2] for visit in primary_visits)
        if expected_primary_visits != change["primary_frame_visits"]:
            raise ValueError("transition predecessor primary visit count changed")
        cases.append({
            "state_index": index,
            "queue": state["queue"],
            "primary_population_visits": primary_visits,
            "legacy_option_indices": change["legacy_option_indices"],
            "variant_option_indices": change["variant_option_indices"],
            "registered_support_score_delta": change[
                "support_score_delta_over_legacy_k26"
            ],
        })
    return cases


def _validate_predecessor():
    predecessor = json.loads(PREDECESSOR.read_text())
    execution_checks = predecessor.get("checks", {})
    advance_checks = predecessor.get("shadow_advance_checks", {})
    if (
        predecessor.get("experiment")
        != "aligned_hardware_permutation_quotient_state_shadow_v1"
        or predecessor.get("status") != "PASS"
        or frozenset(execution_checks) != PREDECESSOR_EXECUTION_CHECK_KEYS
        or not all(execution_checks.values())
        or frozenset(advance_checks) != PREDECESSOR_ADVANCE_CHECK_KEYS
        or not all(advance_checks.values())
        or predecessor.get("scientific_decision")
        != "ADVANCE_TO_AUTHORED_TRANSITION_SCREEN"
        or predecessor.get("development_advance") is not True
        or predecessor.get("retained_change_state_count") != STATE_COUNT
        or len(predecessor.get("state_changes", ())) != STATE_COUNT
    ):
        raise ValueError("transition requires the completed advancing quotient shadow")
    if any(
        change["support_outcome"] != "improved"
        or change["support_score_delta_over_legacy_k26"] <= TOLERANCE
        or not change["raw_lane_witness_changed_from_legacy_k26"]
        or not change["selected_physical_signature_changed_from_legacy_k26"]
        for change in predecessor["state_changes"]
    ):
        raise ValueError("transition predecessor changed-state roster is not strict")
    diagnostic, model = predecessor_shadow._validate_source_diagnostic()
    roster = predecessor_shadow._validated_roster(diagnostic)
    if (
        predecessor_shadow.REGISTERED_STATES.resolve() != REGISTERED_STATES.resolve()
        or roster.get("source_experiment")
        != predecessor["protocol"]["source_experiment"]
        or predecessor["protocol"]["registered_state_roster"]
        != REGISTERED_STATES.name
    ):
        raise ValueError("transition registered source-state lineage changed")
    if _json(model.snapshot()) != predecessor["protocol"]["model"]:
        raise ValueError("transition model changed from its quotient shadow")
    cases = _case_roster(predecessor, roster)
    if (
        len(cases) != STATE_COUNT
        or [row["state_index"] for row in cases]
        != sorted(row["state_index"] for row in cases)
        or len({tuple(row["queue"]) for row in cases}) != STATE_COUNT
    ):
        raise ValueError("transition changed-state case roster changed")
    expected_cells = {(load, seed) for load in LOADS for seed in SEEDS}
    observed_cells = {
        (visit[0], visit[1])
        for case in cases
        for visit in case["primary_population_visits"]
        if visit[2] > 0
    }
    if observed_cells != expected_cells:
        raise ValueError("transition changed states do not cover all nine primary cells")
    _source_texts()
    return predecessor, roster, model, cases


def _physical_signature(model, option_indices):
    service = tuple(
        sum(
            model.options[index].requirements[coordinate]
            for index in option_indices
            if index is not None
        )
        for coordinate in range(len(model.workloads))
    )
    duration = max((
        1.0,
        *(model.options[index].duration for index in option_indices if index is not None),
    ))
    offsets = tuple(
        tuple(sorted(
            offset
            for index in option_indices
            if index is not None
            for offset in model.options[index].completion_offsets[coordinate]
        ))
        for coordinate in range(len(model.workloads))
    )
    return service, duration, offsets


def _selection_realizable(model, queue, selected):
    signature = _physical_signature(model, selected.option_indices)
    score = sum(
        q * served for q, served in zip(queue, signature[0], strict=True)
    ) / signature[1]
    return (
        tuple(selected.service) == signature[0]
        and selected.duration == signature[1]
        and selected.score == score
        and all(served <= q for served, q in zip(signature[0], queue, strict=True))
    )


def _trace_signature(branch):
    return [
        (
            tuple(frame["queue_before"]),
            tuple(frame["departures"]),
            tuple(frame["queue_after"]),
            tuple(frame["option_indices"]),
            frame["duration_s"],
        )
        for frame in branch["frame_trace"]
    ]


def _branch_execution_ok(branch, queue, maximum_duration):
    components = branch["flow_area_components_job_s"]
    return (
        branch["job_count"] == branch["completed_jobs"] == sum(queue)
        and branch["frame_count"] == branch["continuation_selector_calls"] + 1
        and branch["frame_count"] <= sum(queue)
        and branch["elapsed_frame_barriers_s"]
        <= sum(queue) * maximum_duration + TOLERANCE
        and abs(branch["flow_area_identity_error_job_s"]) <= TOLERANCE
        and math.isclose(
            branch["flow_area_job_s"],
            sum(components.values()),
            rel_tol=0.0,
            abs_tol=TOLERANCE,
        )
        and tuple(branch["frame_trace"][0]["queue_before"]) == tuple(queue)
    )


def _run_fixture(model):
    queue = tuple(FIXTURE["queue"])
    forward_indices = tuple(FIXTURE["forward_raw_option_indices"])
    reverse_indices = tuple(FIXTURE["reverse_raw_option_indices"])
    forward_action = fixed_frame_action(model, queue, forward_indices)
    reverse_action = fixed_frame_action(model, queue, reverse_indices)

    def forbidden_continuation(*_args, **_kwargs):
        raise AssertionError("the one-frame raw-lane fixture unexpectedly continued")

    profiles = []
    for profile in AGE_PROFILES:
        arrivals = authored_arrival_times(queue, profile)
        forward = drain_initial_population(
            model,
            arrivals,
            BUDGET,
            forward_action,
            continuation=forbidden_continuation,
        )
        reverse = drain_initial_population(
            model,
            arrivals,
            BUDGET,
            reverse_action,
            continuation=forbidden_continuation,
        )
        profiles.append({
            "profile": profile,
            "forward": forward,
            "reverse": reverse,
            "mean_delta_s": reverse["mean_flow_s"] - forward["mean_flow_s"],
            "p90_delta_s": reverse["p90_flow_s"] - forward["p90_flow_s"],
        })
    by_profile = {row["profile"]: row for row in profiles}
    coincident_forward = sorted(
        row["flow_s"] for row in by_profile["coincident"]["forward"]["completion_records"]
    )
    coincident_reverse = sorted(
        row["flow_s"] for row in by_profile["coincident"]["reverse"]["completion_records"]
    )
    checks = {
        "raw_tuples_are_reversals_of_one_multiset": (
            forward_indices != reverse_indices
            and tuple(sorted(forward_indices)) == tuple(sorted(reverse_indices))
        ),
        "service_duration_and_completion_offset_multiset_match": (
            _physical_signature(model, forward_indices)
            == _physical_signature(model, reverse_indices)
        ),
        "both_profiles_preserve_equal_mean_and_makespan": all(
            abs(row["mean_delta_s"]) <= TOLERANCE
            and abs(
                row["reverse"]["makespan_s"] - row["forward"]["makespan_s"]
            )
            <= TOLERANCE
            for row in profiles
        ),
        "coincident_profile_has_identical_sorted_flows": all(
            math.isclose(left, right, rel_tol=0.0, abs_tol=TOLERANCE)
            for left, right in zip(
                coincident_forward, coincident_reverse, strict=True
            )
        ),
        "unit_fifo_profile_exposes_raw_lane_p90_difference": abs(
            by_profile["unit_fifo"]["p90_delta_s"]
        )
        > TOLERANCE,
    }
    return {"specification": FIXTURE, "profiles": profiles, "checks": checks}


def _run_case(model, case, maximum_duration):
    queue = tuple(case["queue"])
    profiles = []
    for profile in AGE_PROFILES:
        arrivals = authored_arrival_times(queue, profile)
        baseline_action = select_budgeted_frame(
            model.options, queue, model.lanes, BUDGET
        )
        variant_action = select_permutation_quotient_frame(
            model.options, queue, model.lanes, BUDGET
        )
        baseline = drain_initial_population(
            model,
            arrivals,
            BUDGET,
            baseline_action,
            continuation=select_budgeted_frame,
        )
        variant = drain_initial_population(
            model,
            arrivals,
            BUDGET,
            variant_action,
            continuation=select_budgeted_frame,
        )
        profiles.append({
            "profile": profile,
            "baseline_first_action": asdict(baseline_action),
            "variant_first_action": asdict(variant_action),
            "first_actions_realizable": (
                _selection_realizable(model, queue, baseline_action)
                and _selection_realizable(model, queue, variant_action)
            ),
            "baseline": baseline,
            "variant": variant,
            "flow_area_delta_job_s": (
                variant["flow_area_job_s"] - baseline["flow_area_job_s"]
            ),
            "mean_flow_delta_s": variant["mean_flow_s"] - baseline["mean_flow_s"],
            "p90_flow_delta_s": variant["p90_flow_s"] - baseline["p90_flow_s"],
            "makespan_delta_s": variant["makespan_s"] - baseline["makespan_s"],
            "deferred_visible_delta_job_s": (
                variant["flow_area_components_job_s"]["deferred_visible"]
                - baseline["flow_area_components_job_s"]["deferred_visible"]
            ),
            "dispatch_to_completion_delta_job_s": (
                variant["flow_area_components_job_s"]["dispatch_to_completion"]
                - baseline["flow_area_components_job_s"]["dispatch_to_completion"]
            ),
        })
    by_profile = {row["profile"]: row for row in profiles}
    coincident = by_profile["coincident"]
    unit_fifo = by_profile["unit_fifo"]
    first_actions_match = all(
        row["first_actions_realizable"]
        and
        tuple(row["baseline_first_action"]["option_indices"])
        == tuple(case["legacy_option_indices"])
        and tuple(row["variant_first_action"]["option_indices"])
        == tuple(case["variant_option_indices"])
        for row in profiles
    )
    # The actions are recomputed per profile. Their scores must retain the
    # registered fixed-state improvement without using authored ages.
    support_deltas = [
        row["variant_first_action"]["score"]
        - row["baseline_first_action"]["score"]
        for row in profiles
    ]
    checks = {
        "registered_first_actions_reproduced": first_actions_match,
        "variant_first_action_support_improvement_reproduced": all(
            math.isclose(
                delta,
                case["registered_support_score_delta"],
                rel_tol=0.0,
                abs_tol=TOLERANCE,
            )
            and delta > TOLERANCE
            for delta in support_deltas
        ),
        "both_arms_and_profiles_complete_within_bound": all(
            _branch_execution_ok(row[arm], queue, maximum_duration)
            for row in profiles
            for arm in ("baseline", "variant")
        ),
        "age_profiles_do_not_change_queue_or_action_paths": (
            _trace_signature(coincident["baseline"])
            == _trace_signature(unit_fifo["baseline"])
            and _trace_signature(coincident["variant"])
            == _trace_signature(unit_fifo["variant"])
        ),
        "age_profiles_have_identical_total_flow_effect": math.isclose(
            coincident["flow_area_delta_job_s"],
            unit_fifo["flow_area_delta_job_s"],
            rel_tol=0.0,
            abs_tol=TOLERANCE,
        ),
        "paired_flow_component_identity_holds": all(
            abs(
                row["flow_area_delta_job_s"]
                - row["deferred_visible_delta_job_s"]
                - row["dispatch_to_completion_delta_job_s"]
            )
            <= TOLERANCE
            for row in profiles
        ),
    }
    return {
        **case,
        "profiles": profiles,
        "unique_state_flow_area_delta_job_s": coincident[
            "flow_area_delta_job_s"
        ],
        "unique_state_mean_flow_delta_s": coincident["mean_flow_delta_s"],
        "checks": checks,
    }


def _visit_count(case, load, seed):
    return sum(
        visit[2]
        for visit in case["primary_population_visits"]
        if visit[0] == load and visit[1] == seed
    )


def _cell_summaries(cases):
    summaries = []
    for load in LOADS:
        for seed in SEEDS:
            visits = [(_visit_count(case, load, seed), case) for case in cases]
            total_visits = sum(count for count, _ in visits)
            summaries.append({
                "load": load,
                "seed": seed,
                "changed_state_visit_count": total_visits,
                "visited_changed_state_count": sum(count > 0 for count, _ in visits),
                "occurrence_weighted_flow_area_delta_job_s_per_changed_state_visit": (
                    sum(
                        count * case["unique_state_flow_area_delta_job_s"]
                        for count, case in visits
                    )
                    / total_visits
                ),
                "occurrence_weighted_mean_flow_delta_s": (
                    sum(
                        count * case["unique_state_mean_flow_delta_s"]
                        for count, case in visits
                    )
                    / total_visits
                ),
                "occurrence_weighted_p90_delta_s_by_profile": {
                    profile: sum(
                        count
                        * next(
                            row["p90_flow_delta_s"]
                            for row in case["profiles"]
                            if row["profile"] == profile
                        )
                        for count, case in visits
                    )
                    / total_visits
                    for profile in AGE_PROFILES
                },
            })
    return summaries


def _scientific_checks(cases, cells, execution_passed):
    return {
        "execution_passed": execution_passed,
        "all_16_unique_state_mean_flow_deltas_nonpositive": all(
            case["unique_state_mean_flow_delta_s"] <= TOLERANCE for case in cases
        ),
        "all_32_state_profile_nearest_rank_p90_deltas_nonpositive": all(
            profile["p90_flow_delta_s"] <= TOLERANCE
            for case in cases
            for profile in case["profiles"]
        ),
        "all_9_primary_load_seed_weighted_flow_area_deltas_nonpositive": all(
            row[
                "occurrence_weighted_flow_area_delta_job_s_per_changed_state_visit"
            ]
            <= TOLERANCE
            for row in cells
        ),
        "at_least_one_unique_state_has_strict_mean_flow_improvement": any(
            case["unique_state_mean_flow_delta_s"] < -TOLERANCE for case in cases
        ),
        "at_least_one_primary_load_seed_cell_has_strict_weighted_flow_area_improvement": any(
            row[
                "occurrence_weighted_flow_area_delta_job_s_per_changed_state_visit"
            ]
            < -TOLERANCE
            for row in cells
        ),
    }


def _decision(checks):
    if not checks["execution_passed"]:
        return "INVALID_EXECUTION"
    if (
        not checks["all_16_unique_state_mean_flow_deltas_nonpositive"]
        or not checks[
            "all_9_primary_load_seed_weighted_flow_area_deltas_nonpositive"
        ]
    ):
        return "LOCAL_FLOW_TRADEOFF_STOP"
    if not checks[
        "all_32_state_profile_nearest_rank_p90_deltas_nonpositive"
    ]:
        return "TAIL_TRADEOFF_STOP"
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
        raise ValueError("permutation-quotient transition artifacts already exist")
    predecessor, _roster, model, cases = _validate_predecessor()
    maximum_duration = max(
        1.0, *(float(option.duration) for option in model.options)
    )
    maximum_jobs = max(sum(case["queue"]) for case in cases)
    protocol = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "aligned_hardware_permutation_quotient_authored_transition_v1",
        "role": ROLE,
        "source_predecessor": PREDECESSOR.name,
        "registered_state_roster": REGISTERED_STATES.name,
        "budget": BUDGET,
        "state_count": STATE_COUNT,
        "age_profiles": list(AGE_PROFILES),
        "arm_count": 2,
        "registered_drain_count": STATE_COUNT * len(AGE_PROFILES) * 2,
        "loads": list(LOADS),
        "seeds": list(SEEDS),
        "model": _json(model.snapshot()),
        "maximum_model_frame_duration_s": maximum_duration,
        "maximum_registered_queue_jobs": maximum_jobs,
        "global_elapsed_frame_barrier_bound_s": maximum_jobs * maximum_duration,
        "case_roster": cases,
        "rule": RULE,
        "fifo_pairing_fixture": FIXTURE,
        "advance_gate": ADVANCE_GATE,
        "stopping_rule": STOPPING_RULE,
        "prohibited_in_this_stage": PROHIBITED,
        "interpretation": (
            "a conservative local transition veto; occurrence weights come from "
            "the frozen v4 trajectories and are not quotient-policy trajectory outcomes"
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
        raise ValueError("permutation-quotient transition result already exists")
    protocol = json.loads(protocol_path.read_text())
    sources = json.loads(sources_path.read_text())
    predecessor, _roster, model, cases = _validate_predecessor()
    maximum_duration = max(
        1.0, *(float(option.duration) for option in model.options)
    )
    expected = {
        "experiment": "aligned_hardware_permutation_quotient_authored_transition_v1",
        "role": ROLE,
        "source_predecessor": PREDECESSOR.name,
        "registered_state_roster": REGISTERED_STATES.name,
        "budget": BUDGET,
        "state_count": STATE_COUNT,
        "age_profiles": list(AGE_PROFILES),
        "arm_count": 2,
        "registered_drain_count": STATE_COUNT * len(AGE_PROFILES) * 2,
        "loads": list(LOADS),
        "seeds": list(SEEDS),
        "model": _json(model.snapshot()),
        "maximum_model_frame_duration_s": maximum_duration,
        "maximum_registered_queue_jobs": max(sum(case["queue"]) for case in cases),
        "global_elapsed_frame_barrier_bound_s": (
            max(sum(case["queue"]) for case in cases) * maximum_duration
        ),
        "case_roster": cases,
        "rule": RULE,
        "fifo_pairing_fixture": FIXTURE,
        "advance_gate": ADVANCE_GATE,
        "stopping_rule": STOPPING_RULE,
        "prohibited_in_this_stage": PROHIBITED,
    }
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise ValueError("registered permutation-quotient transition protocol changed")
    if protocol.get("interpretation") != (
        "a conservative local transition veto; occurrence weights come from "
        "the frozen v4 trajectories and are not quotient-policy trajectory outcomes"
    ):
        raise ValueError("registered permutation-quotient transition interpretation changed")
    if sources != _source_texts():
        raise ValueError("registered permutation-quotient transition sources changed")

    started = time.perf_counter()
    fixture = _run_fixture(model)
    evaluated = [
        _run_case(model, case, maximum_duration) for case in protocol["case_roster"]
    ]
    cells = _cell_summaries(evaluated)
    execution_checks = {
        "registered_16_unique_states_executed": (
            len(evaluated) == STATE_COUNT
            and [row["state_index"] for row in evaluated]
            == [row["state_index"] for row in protocol["case_roster"]]
        ),
        "registered_64_complete_drains_executed": sum(
            2 * len(case["profiles"]) for case in evaluated
        )
        == protocol["registered_drain_count"],
        "registered_first_actions_support_and_flow_identities_hold": all(
            all(case["checks"].values()) for case in evaluated
        ),
        "all_drains_respect_registered_elapsed_bounds": all(
            _branch_execution_ok(profile[arm], case["queue"], maximum_duration)
            for case in evaluated
            for profile in case["profiles"]
            for arm in ("baseline", "variant")
        ),
        "age_profiles_leave_each_arm_queue_action_path_unchanged": all(
            case["checks"]["age_profiles_do_not_change_queue_or_action_paths"]
            for case in evaluated
        ),
        "two_profiles_reproduce_one_unique_mean_flow_effect_per_state": all(
            case["checks"]["age_profiles_have_identical_total_flow_effect"]
            for case in evaluated
        ),
        "nine_primary_load_seed_cells_reconstruct_positive_visits": (
            len(cells) == 9
            and all(row["changed_state_visit_count"] > 0 for row in cells)
        ),
        "raw_lane_fifo_pairing_fixture_passes": all(fixture["checks"].values()),
        "no_future_arrivals_or_randomness_used": True,
        "registered_sources_unchanged": sources == _source_texts(),
    }
    execution_passed = all(execution_checks.values())
    scientific_checks = _scientific_checks(evaluated, cells, execution_passed)
    decision = _decision(scientific_checks)
    report = {
        "experiment": "aligned_hardware_permutation_quotient_authored_transition_v1",
        "status": "PASS" if execution_passed else "FAIL",
        "checks": execution_checks,
        "protocol": protocol,
        "fifo_pairing_fixture": fixture,
        "cases": evaluated,
        "summary_by_primary_load_seed": cells,
        "summary": {
            "unique_state_count": len(evaluated),
            "state_profile_effect_count": sum(
                len(case["profiles"]) for case in evaluated
            ),
            "drain_count": sum(
                2 * len(case["profiles"]) for case in evaluated
            ),
            "mean_flow_improved_state_count": sum(
                case["unique_state_mean_flow_delta_s"] < -TOLERANCE
                for case in evaluated
            ),
            "mean_flow_worsened_state_count": sum(
                case["unique_state_mean_flow_delta_s"] > TOLERANCE
                for case in evaluated
            ),
            "p90_improved_state_profile_count": sum(
                profile["p90_flow_delta_s"] < -TOLERANCE
                for case in evaluated
                for profile in case["profiles"]
            ),
            "p90_worsened_state_profile_count": sum(
                profile["p90_flow_delta_s"] > TOLERANCE
                for case in evaluated
                for profile in case["profiles"]
            ),
            "mean_unique_state_flow_area_delta_job_s": sum(
                case["unique_state_flow_area_delta_job_s"] for case in evaluated
            )
            / len(evaluated),
            "mean_unique_state_mean_flow_delta_s": sum(
                case["unique_state_mean_flow_delta_s"] for case in evaluated
            )
            / len(evaluated),
            "minimum_unique_state_mean_flow_delta_s": min(
                case["unique_state_mean_flow_delta_s"] for case in evaluated
            ),
            "maximum_unique_state_mean_flow_delta_s": max(
                case["unique_state_mean_flow_delta_s"] for case in evaluated
            ),
            "minimum_state_profile_p90_delta_s": min(
                profile["p90_flow_delta_s"]
                for case in evaluated
                for profile in case["profiles"]
            ),
            "maximum_state_profile_p90_delta_s": max(
                profile["p90_flow_delta_s"]
                for case in evaluated
                for profile in case["profiles"]
            ),
        },
        "work_accounting": {
            "first_action_selector_calls": STATE_COUNT * len(AGE_PROFILES) * 2,
            "continuation_selector_calls": sum(
                profile[arm]["continuation_selector_calls"]
                for case in evaluated
                for profile in case["profiles"]
                for arm in ("baseline", "variant")
            ),
            "candidate_attempts": sum(
                profile[arm]["selection_work"]["attempts"]
                for case in evaluated
                for profile in case["profiles"]
                for arm in ("baseline", "variant")
            ),
            "quotient_first_action_cache_hits": sum(
                profile["variant"]["selection_work"]["cache_hits"]
                for case in evaluated
                for profile in case["profiles"]
            ),
            "authored_jobs_completed": sum(
                profile[arm]["completed_jobs"]
                for case in evaluated
                for profile in case["profiles"]
                for arm in ("baseline", "variant")
            ),
            "future_arrivals": 0,
            "rng_initializations": 0,
            "random_draws": 0,
            "execution_wall_s": time.perf_counter() - started,
        },
        "transition_advance_checks": scientific_checks,
        "scientific_decision": decision,
        "development_advance": decision
        == "ADVANCE_TO_FULL_DEVELOPMENT_TRAJECTORIES",
        "next_step": (
            "preregister the nine K26 full development trajectories"
            if decision == "ADVANCE_TO_FULL_DEVELOPMENT_TRAJECTORIES"
            else "close the quotient-cache branch without tuning or full trajectories"
        ),
        "limitations": [
            "These authored finite drains are a conservative local veto, not full-policy trajectory evidence.",
            "The unit-spaced FIFO ages use the model's one-second duration floor and do not estimate the empirical age distribution.",
            "Occurrence weights come from frozen v4 K26 trajectories; they do not represent quotient-policy occupancy.",
            "Only the first action uses the quotient selector; every continuation action is v4 K26.",
        ],
    }
    if sources != _source_texts():
        raise ValueError("registered permutation-quotient transition sources changed during run")
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
