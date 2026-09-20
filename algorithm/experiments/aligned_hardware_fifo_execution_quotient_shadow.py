"""Registered fixed-state screen for FIFO-execution quotient caching.

The screen reuses the complete 5,003-state roster from the candidate-search
diagnostic. It compares the frozen v4 K26 result with one selector that keeps
the v4 logical heap and bounds but charges each exact FIFO execution signature
once after the paid seeds. No arrival, trajectory, or random process runs here.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import product
import json
import math
from pathlib import Path
import time

from algorithm.experiments.aligned_hardware_frame_model import load_model
from algorithm.fifo_execution_quotient_frame_candidates import (
    fifo_execution_signature,
    select_fifo_execution_quotient_frame,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
DEFAULT_OUTPUT = ARTIFACTS / "aligned_hardware_fifo_execution_quotient_shadow_development_20260911.json"
SOURCE_DIAGNOSTIC = ARTIFACTS / "aligned_hardware_candidate_search_diagnostic_development_20260910.json"
SOURCE_FIXED_SOURCES = ARTIFACTS / "aligned_hardware_candidate_search_diagnostic_development_20260910_fixed_sources.json"
REGISTERED_STATES = ARTIFACTS / "aligned_hardware_duration_tie_order_shadow_development_20260910_registered_states.json"
KNOWN_TRANSITION_PROTOCOL = ARTIFACTS / "aligned_hardware_permutation_quotient_transition_development_20260911_protocol.json"
LOADS = (0.50, 0.80, 0.95)
SEEDS = (41, 42, 43)
SMALL_BUDGET = 26
LEGACY_EXACT_BUDGET = 120
SOURCE_POLICY_BUDGETS = (SMALL_BUDGET, LEGACY_EXACT_BUDGET)
EXACT_FIFO_BUDGET = 71
STATE_COUNT = 5003
ORDERED_ACTION_COUNT = 100
FIFO_SIGNATURE_COUNT = 71
SCORE_TOLERANCE = 1e-10
AGGREGATE_TOLERANCE = 1e-12
NEW_SOURCES = (
    "algorithm/fifo_execution_quotient_frame_candidates.py",
    "algorithm/experiments/aligned_hardware_fifo_execution_quotient_shadow.py",
)

RULE = {
    "name": "stable_fifo_execution_signature_candidate_cache",
    "equivalence_key": (
        "right-pad a logical raw lane prefix with idle, then use aggregate service, "
        "effective frame duration, and each class's completion offsets concatenated "
        "in raw lane order and stored within-option order"
    ),
    "paid_budget": (
        "all-idle and one homogeneous seed per option, then the first logical "
        "search occurrence of each previously unseen equivalence key"
    ),
    "free_reuse": (
        "reuse feasibility, service, numerator, duration, and score for later "
        "raw actions with exactly the same FIFO execution signature; still expand "
        "the original logical prefix"
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
    "seed_accounting": "all-idle and all nine homogeneous option seeds remain paid even if seed signatures collide",
    "cache_key_excludes": "sorted completion-offset multisets and raw option identity beyond their exact FIFO execution effect",
    "exact_cap": "the frozen two-lane catalogue has 100 raw actions and 71 FIFO execution signatures, so K71 exhausts support-relevant signatures",
    "exact_semantics": "exact certifies maximum one-frame support; because this selector freezes the legacy logical heap, order, and paid-candidate tie rule, execution integrity separately requires K71 raw, physical, and FIFO witnesses to match registered legacy K120 on every state",
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
    "run this fixed rule once on the registered 5,003 states; invalid execution gives "
    "INVALID_EXECUTION; all fixed gates plus both strict gains and at least one FIFO-effective "
    "change give ADVANCE_TO_EVIDENCE_REUSE_OR_AUTHORED_TRANSITION; nonworse certificate or "
    "attempt gain without support gain or FIFO-effective change gives EFFICIENCY_ONLY_STOP; "
    "all other outcomes give CLOSE_WITHOUT_TRANSITION_SCREEN"
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
SIGNATURE_SCHEMA = {
    "raw": "right-padded original option-index tuple, with None denoting idle",
    "physical": "aggregate service, duration, and sorted completion-offset multiset per class",
    "fifo_execution": "aggregate service, duration, and FIFO-ordered completion-offset sequence per class",
}
EVIDENCE_PARTITION = (
    "RAW_UNCHANGED",
    "RAW_CHANGED_FIFO_EQUIVALENT",
    "FIFO_CHANGED_PHYSICAL_EQUIVALENT_KNOWN_PROTOCOL",
    "FIFO_CHANGED_PHYSICAL_EQUIVALENT_NOVEL",
    "PHYSICAL_CHANGED_KNOWN_PROTOCOL",
    "PHYSICAL_CHANGED_NOVEL",
)
KNOWN_PROTOCOL_USE = (
    "read only registered queues and action arms from the prior authored-transition "
    "protocol; do not read its outcome artifact or use effects to tune this selector or gates"
)
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
        or protocol["exact_budget"] != LEGACY_EXACT_BUDGET
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
            "FIFO-execution-quotient shadow requires the completed candidate-search diagnostic"
        )
    model = load_model()
    if _json(model.snapshot()) != protocol["model"]:
        raise ValueError("FIFO-execution-quotient model changed from its source diagnostic")
    sources = json.loads(SOURCE_FIXED_SOURCES.read_text())
    if any((ROOT / name).read_text() != text for name, text in sources.items()):
        raise ValueError("FIFO-execution-quotient source diagnostic code changed")
    return report, model


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
        raise ValueError("registered FIFO-execution-quotient states changed")
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


def _service_duration(model, option_indices):
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
    return service, float(duration)


def _signature_dict(signature, offset_name):
    service, duration, offsets = signature
    return {
        "service": list(service),
        "duration": float(duration),
        offset_name: [list(coordinate) for coordinate in offsets],
    }


def _fifo_signature(model, option_indices):
    signature = fifo_execution_signature(
        model.options,
        option_indices,
        model.lanes,
        len(model.workloads),
    )
    reads = sum(
        len(model.options[index].completion_offsets[coordinate])
        for index in option_indices
        if index is not None
        for coordinate in range(len(model.workloads))
    )
    return _signature_dict(signature, "completion_offset_sequences"), reads


def _physical_signature(model, option_indices):
    service, duration = _service_duration(model, option_indices)
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
    signature = (service, duration, tuple(offsets))
    return _signature_dict(signature, "completion_offset_multisets"), reads


def _action_signatures(model, option_indices):
    physical, physical_reads = _physical_signature(model, option_indices)
    fifo, fifo_reads = _fifo_signature(model, option_indices)
    return {
        "raw": list(option_indices),
        "physical": physical,
        "fifo_execution": fifo,
    }, physical_reads + fifo_reads


def _structure_snapshot(model):
    choices = (None, *range(len(model.options)))
    signatures = {
        fifo_execution_signature(
            model.options, raw, model.lanes, len(model.workloads)
        )
        for raw in product(choices, repeat=model.lanes)
    }
    return {
        "option_count": len(model.options),
        "lanes": model.lanes,
        "represented_raw_action_count": (len(model.options) + 1) ** model.lanes,
        "represented_fifo_execution_signature_count": len(signatures),
        "small_checkpoint_budget": SMALL_BUDGET,
        "exact_fifo_execution_signature_budget": EXACT_FIFO_BUDGET,
    }


def _structure_is_expected(snapshot):
    return snapshot == {
        "option_count": 9,
        "lanes": 2,
        "represented_raw_action_count": ORDERED_ACTION_COUNT,
        "represented_fifo_execution_signature_count": FIFO_SIGNATURE_COUNT,
        "small_checkpoint_budget": SMALL_BUDGET,
        "exact_fifo_execution_signature_budget": EXACT_FIFO_BUDGET,
    }


def _known_protocol_action_pairs(model, roster):
    protocol = json.loads(KNOWN_TRANSITION_PROTOCOL.read_text())
    if (
        protocol.get("experiment")
        != "aligned_hardware_permutation_quotient_authored_transition_v1"
        or protocol.get("registered_state_roster") != REGISTERED_STATES.name
        or protocol.get("state_count") != 16
        or protocol.get("budget") != SMALL_BUDGET
        or protocol.get("loads") != list(LOADS)
        or protocol.get("seeds") != list(SEEDS)
        or protocol.get("model") != _json(model.snapshot())
        or len(protocol.get("case_roster", ())) != 16
    ):
        raise ValueError("known authored-transition action protocol changed")
    states = roster["states"]
    pairs = []
    case_indices = []
    for case in protocol["case_roster"]:
        state_index = case.get("state_index")
        if (
            not isinstance(state_index, int)
            or not 0 <= state_index < len(states)
            or case.get("queue") != states[state_index]["queue"]
        ):
            raise ValueError("known transition case no longer matches the state roster")
        case_indices.append(state_index)
        baseline, _ = _action_signatures(model, case["legacy_option_indices"])
        variant, _ = _action_signatures(model, case["variant_option_indices"])
        pairs.append({
            "pair_id": f"quotient_case_{state_index}",
            "source_kind": "registered_quotient_transition_case",
            "state_index": state_index,
            "queue": case["queue"],
            "baseline": baseline,
            "variant": variant,
        })
    if case_indices != sorted(set(case_indices)):
        raise ValueError("known transition cases are not unique and ordered")

    return pairs


def _known_protocol_fifo_fixture(model, roster):
    protocol = json.loads(KNOWN_TRANSITION_PROTOCOL.read_text())
    fixture = protocol.get("fifo_pairing_fixture")
    if not isinstance(fixture, dict):
        raise ValueError("known FIFO-pairing fixture changed")
    states = roster["states"]
    queue = fixture.get("queue")
    forward = fixture.get("forward_raw_option_indices")
    reverse = fixture.get("reverse_raw_option_indices")
    if (
        not isinstance(queue, list)
        or len(queue) != len(model.workloads)
        or not isinstance(forward, list)
        or not isinstance(reverse, list)
        or len(forward) != model.lanes
        or len(reverse) != model.lanes
        or forward != list(reversed(reverse))
        or queue not in [state["queue"] for state in states]
    ):
        raise ValueError("known FIFO-pairing fixture changed")
    forward_signatures, _ = _action_signatures(model, forward)
    reverse_signatures, _ = _action_signatures(model, reverse)
    audit = {
        "queue": queue,
        "forward": forward_signatures,
        "reverse": reverse_signatures,
        "raw_tuples_are_reversals": forward == list(reversed(reverse)),
        "physical_signatures_equal": (
            forward_signatures["physical"] == reverse_signatures["physical"]
        ),
        "fifo_execution_signatures_differ": (
            forward_signatures["fifo_execution"]
            != reverse_signatures["fifo_execution"]
        ),
    }
    if not all(
        audit[key]
        for key in (
            "raw_tuples_are_reversals",
            "physical_signatures_equal",
            "fifo_execution_signatures_differ",
        )
    ):
        raise ValueError("known FIFO-pairing structural witness changed")
    return audit


def _matching_pair_ids(known_pairs, queue, baseline, variant, signature):
    return [
        known["pair_id"]
        for known in known_pairs
        if (
        known["queue"] == list(queue)
        and known["baseline"][signature] == baseline[signature]
        and known["variant"][signature] == variant[signature]
        )
    ]


def _evidence_partition(raw_changed, fifo_changed, physical_changed, known_fifo):
    if not raw_changed:
        return "RAW_UNCHANGED"
    if not fifo_changed:
        return "RAW_CHANGED_FIFO_EQUIVALENT"
    if not physical_changed:
        return (
            "FIFO_CHANGED_PHYSICAL_EQUIVALENT_KNOWN_PROTOCOL"
            if known_fifo
            else "FIFO_CHANGED_PHYSICAL_EQUIVALENT_NOVEL"
        )
    return (
        "PHYSICAL_CHANGED_KNOWN_PROTOCOL"
        if known_fifo
        else "PHYSICAL_CHANGED_NOVEL"
    )


def _realizable(model, queues, selected):
    service, duration = _service_duration(model, selected.option_indices)
    score = sum(
        q * count for q, count in zip(queues, service, strict=True)
    ) / duration
    return (
        len(selected.option_indices) == model.lanes
        and all(
            index is None or 0 <= index < len(model.options)
            for index in selected.option_indices
        )
        and service == selected.service
        and all(count <= q for count, q in zip(service, queues, strict=True))
        and math.isclose(
            duration, selected.duration, rel_tol=0.0, abs_tol=SCORE_TOLERANCE
        )
        and math.isclose(
            score, selected.score, rel_tol=0.0, abs_tol=SCORE_TOLERANCE
        )
    )


def _work_identities(selection, option_count, lanes):
    return (
        selection.seed_attempts == option_count + 1
        and selection.attempts
        == selection.seed_attempts + selection.paid_search_pops
        and selection.cache_entries
        == selection.attempts - selection.seed_signature_collisions
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
        and selection.cache_key_requirement_coordinate_reads
        == len(selection.service) * selection.cache_key_duration_reads
        and selection.cache_key_completion_offset_reads
        >= selection.cache_key_duration_reads
        and selection.upper_bound_calls
        == selection.frontier_pushes + selection.bound_rejected_prefixes
        and selection.frontier_pushes
        == selection.logical_search_pops
        + selection.exact_stop_frontier_entries_cleared
        + selection.frontier_entries_remaining
        and selection.represented_ordered_configurations
        == (option_count + 1) ** lanes
        and 0 <= selection.seed_signature_collisions < selection.seed_attempts
        and selection.cache_entries <= FIFO_SIGNATURE_COUNT
        and selection.budget_blocked_cache_miss_lookups in (0, 1)
        and selection.exact == (selection.frontier_entries_remaining == 0)
    )


def _evaluate_state(model, known_pairs, state_index, state, source_exact):
    queues = tuple(state["queue"])
    small = select_fifo_execution_quotient_frame(
        model.options, queues, model.lanes, SMALL_BUDGET
    )
    exact = select_fifo_execution_quotient_frame(
        model.options, queues, model.lanes, EXACT_FIFO_BUDGET
    )
    reference_score = state["support_reference_score"]
    legacy = state["legacy_k26"]
    support_regret = reference_score - small.score
    bound_slack = small.upper_bound - reference_score
    identity_error = small.gap - support_regret - bound_slack
    epsilon_error = small.epsilon - (
        small.gap / sum(queues) if sum(queues) else 0.0
    )
    legacy_signatures, legacy_reads = _action_signatures(
        model, legacy["option_indices"]
    )
    small_signatures, small_reads = _action_signatures(model, small.option_indices)
    exact_signatures, exact_reads = _action_signatures(model, exact.option_indices)
    source_exact_signatures, source_exact_reads = _action_signatures(
        model, source_exact["option_indices"]
    )
    raw_changed = small_signatures["raw"] != legacy_signatures["raw"]
    physical_changed = (
        small_signatures["physical"] != legacy_signatures["physical"]
    )
    fifo_changed = (
        small_signatures["fifo_execution"]
        != legacy_signatures["fifo_execution"]
    )
    known_raw_ids = _matching_pair_ids(
        known_pairs, queues, legacy_signatures, small_signatures, "raw"
    )
    known_physical_ids = _matching_pair_ids(
        known_pairs, queues, legacy_signatures, small_signatures, "physical"
    )
    known_fifo_ids = _matching_pair_ids(
        known_pairs, queues, legacy_signatures, small_signatures, "fifo_execution"
    )
    known_raw = bool(known_raw_ids)
    known_physical = bool(known_physical_ids)
    known_fifo = bool(known_fifo_ids)
    score_delta = small.score - legacy["score"]
    upper_delta = small.upper_bound - legacy["upper_bound"]
    gap_delta = small.gap - legacy["gap"]
    execution_ok = (
        _realizable(model, queues, small)
        and _realizable(model, queues, exact)
        and _work_identities(small, len(model.options), model.lanes)
        and _work_identities(exact, len(model.options), model.lanes)
        and small.attempts <= SMALL_BUDGET
        and exact.attempts <= EXACT_FIFO_BUDGET
        and small.represented_ordered_configurations == ORDERED_ACTION_COUNT
        and exact.represented_ordered_configurations == ORDERED_ACTION_COUNT
        and exact.exact
        and exact.frontier_entries_remaining == 0
        and exact.budget_blocked_cache_miss_lookups == 0
        and abs(exact.gap) <= SCORE_TOLERANCE
        and abs(exact.upper_bound - exact.score) <= SCORE_TOLERANCE
        and abs(exact.epsilon) <= SCORE_TOLERANCE
        and abs(exact.score - reference_score) <= SCORE_TOLERANCE
        and small.score <= reference_score + SCORE_TOLERANCE
        and reference_score <= small.upper_bound + SCORE_TOLERANCE
        and abs(identity_error) <= SCORE_TOLERANCE
        and abs(epsilon_error) <= SCORE_TOLERANCE
    )
    return {
        "state_index": state_index,
        "variant_k26": asdict(small),
        "variant_k71": asdict(exact),
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
        "raw_lane_witness_changed_from_legacy_k26": raw_changed,
        "physical_signature_changed_from_legacy_k26": physical_changed,
        "fifo_execution_signature_changed_from_legacy_k26": fifo_changed,
        "legacy_signatures": legacy_signatures,
        "variant_k26_signatures": small_signatures,
        "variant_k71_signatures": exact_signatures,
        "known_protocol_raw_pair_match": known_raw,
        "known_protocol_physical_pair_match": known_physical,
        "known_protocol_fifo_pair_match": known_fifo,
        "known_protocol_raw_pair_ids": known_raw_ids,
        "known_protocol_physical_pair_ids": known_physical_ids,
        "known_protocol_fifo_pair_ids": known_fifo_ids,
        "evidence_partition": _evidence_partition(
            raw_changed, fifo_changed, physical_changed, known_fifo
        ),
        "variant_k71_support_matches_registered_reference": (
            abs(exact.score - reference_score) <= SCORE_TOLERANCE
        ),
        "variant_k71_raw_matches_legacy_k120": (
            exact_signatures["raw"] == source_exact_signatures["raw"]
        ),
        "variant_k71_physical_matches_legacy_k120": (
            exact_signatures["physical"] == source_exact_signatures["physical"]
        ),
        "variant_k71_fifo_matches_legacy_k120": (
            exact_signatures["fifo_execution"]
            == source_exact_signatures["fifo_execution"]
        ),
        "selection_bound_and_work_checks_pass": execution_ok,
        "signature_completion_offset_reads": (
            legacy_reads + small_reads + exact_reads + source_exact_reads
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
        "physical_signature_changed_frame_fraction": frames_where(
            lambda _state, evaluation: evaluation[
                "physical_signature_changed_from_legacy_k26"
            ]
        )
        / frame_count,
        "physical_signature_changed_time_fraction": time_where(
            lambda _state, evaluation: evaluation[
                "physical_signature_changed_from_legacy_k26"
            ]
        )
        / visited_time,
        "fifo_execution_signature_changed_frame_fraction": frames_where(
            lambda _state, evaluation: evaluation[
                "fifo_execution_signature_changed_from_legacy_k26"
            ]
        )
        / frame_count,
        "fifo_execution_signature_changed_time_fraction": time_where(
            lambda _state, evaluation: evaluation[
                "fifo_execution_signature_changed_from_legacy_k26"
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
        "known_protocol_fifo_pair_frame_fraction": frames_where(
            lambda _state, evaluation: evaluation[
                "known_protocol_fifo_pair_match"
            ]
        )
        / frame_count,
        "evidence_partition_frame_count": {
            category: frames_where(
                lambda _state, evaluation, category=category: evaluation[
                    "evidence_partition"
                ]
                == category
            )
            for category in EVIDENCE_PARTITION
        },
        "evidence_partition_visited_time_s": {
            category: time_where(
                lambda _state, evaluation, category=category: evaluation[
                    "evidence_partition"
                ]
                == category
            )
            for category in EVIDENCE_PARTITION
        },
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


def _decision(checks, evaluations):
    if not checks["execution_passed"]:
        return "INVALID_EXECUTION"
    fifo_effective_change = any(
        evaluation["fifo_execution_signature_changed_from_legacy_k26"]
        for evaluation in evaluations
    )
    if all(checks.values()) and fifo_effective_change:
        return "ADVANCE_TO_EVIDENCE_REUSE_OR_AUTHORED_TRANSITION"
    non_improvement_keys = {
        "at_least_one_primary_load_strictly_improves_support",
        "at_least_one_primary_load_strictly_improves_certificate_or_attempts",
    }
    nonworse = all(
        value for key, value in checks.items() if key not in non_improvement_keys
    )
    if (
        nonworse
        and not checks["at_least_one_primary_load_strictly_improves_support"]
        and not fifo_effective_change
        and checks[
            "at_least_one_primary_load_strictly_improves_certificate_or_attempts"
        ]
    ):
        return "EFFICIENCY_ONLY_STOP"
    return "CLOSE_WITHOUT_TRANSITION_SCREEN"


def _transition_record(state, evaluation):
    return {
        "state_index": evaluation["state_index"],
        "queue": state["queue"],
        "primary_population_visits": [
            visit for visit in state["population_visits"] if visit[2] == SMALL_BUDGET
        ],
        "support_outcome": evaluation["support_outcome"],
        "support_score_delta_over_legacy_k26": evaluation[
            "support_score_delta_over_legacy_k26"
        ],
        "upper_bound_delta_over_legacy_k26": evaluation[
            "upper_bound_delta_over_legacy_k26"
        ],
        "gap_delta_over_legacy_k26": evaluation["gap_delta_over_legacy_k26"],
        "evidence_partition": evaluation["evidence_partition"],
        "known_protocol_raw_pair_ids": evaluation[
            "known_protocol_raw_pair_ids"
        ],
        "known_protocol_physical_pair_ids": evaluation[
            "known_protocol_physical_pair_ids"
        ],
        "known_protocol_fifo_pair_ids": evaluation[
            "known_protocol_fifo_pair_ids"
        ],
        "legacy_signatures": evaluation["legacy_signatures"],
        "variant_k26_signatures": evaluation["variant_k26_signatures"],
        "variant_attempts": evaluation["variant_k26"]["attempts"],
        "variant_cache_hits": evaluation["variant_k26"]["cache_hits"],
    }


def register(output=DEFAULT_OUTPUT):
    output, protocol_path, sources_path = _paths(output)
    if any(path.exists() for path in (output, protocol_path, sources_path)):
        raise ValueError("FIFO-execution-quotient shadow artifacts already exist")

    # Source, roster, structure, and known action protocols are frozen without
    # calling the selector or reading any prior transition outcome.
    sources = _source_texts()
    diagnostic, model = _validate_source_diagnostic()
    roster = _validated_roster(diagnostic)
    structure = _structure_snapshot(model)
    if not _structure_is_expected(structure):
        raise ValueError("registered 100-to-71 FIFO execution structure changed")
    known_pairs = _known_protocol_action_pairs(model, roster)
    fifo_fixture = _known_protocol_fifo_fixture(model, roster)
    protocol = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "aligned_hardware_fifo_execution_quotient_state_shadow_v1",
        "role": ROLE,
        "source_experiment": SOURCE_DIAGNOSTIC.name,
        "registered_state_roster": REGISTERED_STATES.name,
        "known_transition_protocol": KNOWN_TRANSITION_PROTOCOL.name,
        "loads": list(LOADS),
        "seeds": list(SEEDS),
        "source_policy_budgets": list(SOURCE_POLICY_BUDGETS),
        "small_checkpoint_budget": SMALL_BUDGET,
        "legacy_exact_reference_budget": LEGACY_EXACT_BUDGET,
        "exact_fifo_budget": EXACT_FIFO_BUDGET,
        "state_count": STATE_COUNT,
        "ordered_action_count": ORDERED_ACTION_COUNT,
        "fifo_signature_count": FIFO_SIGNATURE_COUNT,
        "model": model.snapshot(),
        "registered_state_schema": REGISTERED_STATE_SCHEMA,
        "rule": RULE,
        "signature_schema": SIGNATURE_SCHEMA,
        "fifo_execution_structure": structure,
        "evidence_partition": list(EVIDENCE_PARTITION),
        "known_protocol_use": KNOWN_PROTOCOL_USE,
        "known_protocol_action_pairs": known_pairs,
        "known_fifo_pairing_fixture": fifo_fixture,
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
        raise ValueError("FIFO-execution-quotient shadow result already exists")
    protocol = json.loads(protocol_path.read_text())
    sources = json.loads(sources_path.read_text())
    expected = {
        "experiment": "aligned_hardware_fifo_execution_quotient_state_shadow_v1",
        "source_experiment": SOURCE_DIAGNOSTIC.name,
        "registered_state_roster": REGISTERED_STATES.name,
        "known_transition_protocol": KNOWN_TRANSITION_PROTOCOL.name,
        "loads": list(LOADS),
        "seeds": list(SEEDS),
        "source_policy_budgets": list(SOURCE_POLICY_BUDGETS),
        "small_checkpoint_budget": SMALL_BUDGET,
        "legacy_exact_reference_budget": LEGACY_EXACT_BUDGET,
        "exact_fifo_budget": EXACT_FIFO_BUDGET,
        "state_count": STATE_COUNT,
        "ordered_action_count": ORDERED_ACTION_COUNT,
        "fifo_signature_count": FIFO_SIGNATURE_COUNT,
    }
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise ValueError("registered FIFO-execution roster or constants changed")
    frozen = {
        "role": ROLE,
        "registered_state_schema": REGISTERED_STATE_SCHEMA,
        "rule": RULE,
        "signature_schema": SIGNATURE_SCHEMA,
        "evidence_partition": list(EVIDENCE_PARTITION),
        "known_protocol_use": KNOWN_PROTOCOL_USE,
        "advance_gate": ADVANCE_GATE,
        "stopping_rule": STOPPING_RULE,
        "prohibited_in_this_stage": PROHIBITED,
        "support_reference": SUPPORT_REFERENCE,
    }
    if any(protocol.get(key) != value for key, value in frozen.items()):
        raise ValueError("registered FIFO-execution protocol rule changed")

    # The full source/predecessor/roster preflight finishes before the first
    # selector call. No prior transition outcome is opened here.
    if sources != _source_texts():
        raise ValueError("registered FIFO-execution sources changed")
    diagnostic, model = _validate_source_diagnostic()
    roster = _validated_roster(diagnostic)
    if protocol.get("model") != _json(model.snapshot()):
        raise ValueError("registered FIFO-execution model changed")
    structure = _structure_snapshot(model)
    if (
        structure != protocol.get("fifo_execution_structure")
        or not _structure_is_expected(structure)
    ):
        raise ValueError("registered FIFO-execution structure changed")
    known_pairs = _known_protocol_action_pairs(model, roster)
    if known_pairs != protocol.get("known_protocol_action_pairs"):
        raise ValueError("registered known action-pair protocol changed")
    fifo_fixture = _known_protocol_fifo_fixture(model, roster)
    if fifo_fixture != protocol.get("known_fifo_pairing_fixture"):
        raise ValueError("registered FIFO-pairing fixture structure changed")
    if (
        protocol.get("source_population_summaries")
        != diagnostic["summary_by_load_and_source_policy"]
    ):
        raise ValueError("registered FIFO-execution source summaries changed")
    if protocol.get("source_population_cells") != _source_population_cells(
        diagnostic
    ):
        raise ValueError("registered FIFO-execution source cells changed")

    states = roster["states"]
    selector_started = time.perf_counter()
    evaluations = [
        _evaluate_state(
            model,
            known_pairs,
            index,
            state,
            diagnostic["state_diagnostics"][index]["exact"],
        )
        for index, state in enumerate(states)
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
    partition_state_counts = {
        category: sum(
            row["evidence_partition"] == category for row in evaluations
        )
        for category in EVIDENCE_PARTITION
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
        "cartesian_100_raw_to_71_fifo_signatures_holds": (
            _structure_is_expected(structure)
        ),
        "known_protocol_16_completed_case_pairs_reconstruct": (
            len(known_pairs) == 16
            and known_pairs == protocol["known_protocol_action_pairs"]
        ),
        "fifo_pairing_fixture_is_separate_and_structurally_reconstructs": (
            fifo_fixture == protocol["known_fifo_pairing_fixture"]
            and fifo_fixture["raw_tuples_are_reversals"]
            and fifo_fixture["physical_signatures_equal"]
            and fifo_fixture["fifo_execution_signatures_differ"]
        ),
        "candidate_cache_and_frontier_work_identities_hold": all(
            row["selection_bound_and_work_checks_pass"] for row in evaluations
        ),
        "k71_matches_registered_legacy_k120_support_on_every_state": all(
            row["variant_k71"]["exact"]
            and row["variant_k71_support_matches_registered_reference"]
            for row in evaluations
        ),
        "k71_replays_registered_k120_raw_physical_and_fifo_witnesses": all(
            row["variant_k71_raw_matches_legacy_k120"]
            and row["variant_k71_physical_matches_legacy_k120"]
            and row["variant_k71_fifo_matches_legacy_k120"]
            for row in evaluations
        ),
        "k26_support_bound_gap_and_epsilon_hold": all(
            abs(row["variant_gap_identity_error"]) <= SCORE_TOLERANCE
            and abs(row["variant_epsilon_error"]) <= SCORE_TOLERANCE
            for row in evaluations
        ),
        "candidate_attempts_respect_k26_and_k71_budgets": all(
            row["variant_k26"]["attempts"] <= SMALL_BUDGET
            and row["variant_k71"]["attempts"] <= EXACT_FIFO_BUDGET
            for row in evaluations
        ),
        "raw_fifo_physical_signature_implications_hold": all(
            (
                row["raw_lane_witness_changed_from_legacy_k26"]
                or not row["fifo_execution_signature_changed_from_legacy_k26"]
            )
            and (
                row["raw_lane_witness_changed_from_legacy_k26"]
                or not row["physical_signature_changed_from_legacy_k26"]
            )
            and (
                row["fifo_execution_signature_changed_from_legacy_k26"]
                or not row["physical_signature_changed_from_legacy_k26"]
            )
            for row in evaluations
        ),
        "evidence_partition_is_complete_and_exclusive": (
            set(row["evidence_partition"] for row in evaluations)
            <= set(EVIDENCE_PARTITION)
            and sum(partition_state_counts.values()) == STATE_COUNT
            and all(
                sum(row["evidence_partition_frame_count"].values())
                == row["frame_count"]
                and math.isclose(
                    sum(row["evidence_partition_visited_time_s"].values()),
                    row["legacy_visited_time_s"],
                    rel_tol=0.0,
                    abs_tol=1e-8,
                )
                for row in (*population, *cells)
            )
        ),
        "known_transition_outcomes_not_read": True,
        "no_arrivals_trajectories_or_randomness_used": True,
        "registered_sources_unchanged": all(
            (ROOT / name).read_text() == text for name, text in sources.items()
        ),
    }
    passed = all(execution_checks.values())
    primary = [
        row for row in population if row["source_policy_budget"] == SMALL_BUDGET
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
    changed = [
        row
        for row in evaluations
        if row["raw_lane_witness_changed_from_legacy_k26"]
        or row["support_outcome"] != "unchanged"
    ]
    fifo_effective = [
        row
        for row in evaluations
        if row["fifo_execution_signature_changed_from_legacy_k26"]
    ]
    known_fifo = [
        row for row in fifo_effective if row["known_protocol_fifo_pair_match"]
    ]
    novel_fifo = [
        row
        for row in fifo_effective
        if not row["known_protocol_fifo_pair_match"]
    ]
    report = {
        "experiment": "aligned_hardware_fifo_execution_quotient_state_shadow_v1",
        "status": "PASS" if passed else "FAIL",
        "checks": execution_checks,
        "protocol": protocol,
        "registered_state_roster": REGISTERED_STATES.name,
        "state_evaluation_count": len(evaluations),
        "changed_state_count": len(changed),
        "raw_change_state_count": sum(
            row["raw_lane_witness_changed_from_legacy_k26"]
            for row in evaluations
        ),
        "physical_change_state_count": sum(
            row["physical_signature_changed_from_legacy_k26"]
            for row in evaluations
        ),
        "fifo_effective_change_state_count": len(fifo_effective),
        "known_fifo_pairing_fixture": fifo_fixture,
        "evidence_partition": {
            "categories": list(EVIDENCE_PARTITION),
            "unique_state_counts": partition_state_counts,
            "primary_k26_policy_occurrence_counts": {
                category: sum(
                    visit[3]
                    for state, evaluation in zip(states, evaluations, strict=True)
                    if evaluation["evidence_partition"] == category
                    for visit in state["population_visits"]
                    if visit[2] == SMALL_BUDGET
                )
                for category in EVIDENCE_PARTITION
            },
            "sensitivity_k120_policy_occurrence_counts": {
                category: sum(
                    visit[3]
                    for state, evaluation in zip(states, evaluations, strict=True)
                    if evaluation["evidence_partition"] == category
                    for visit in state["population_visits"]
                    if visit[2] == LEGACY_EXACT_BUDGET
                )
                for category in EVIDENCE_PARTITION
            },
            "known_protocol_matches": [
                _transition_record(states[row["state_index"]], row)
                for row in known_fifo
            ],
            "novel_fifo_effective_roster": [
                _transition_record(states[row["state_index"]], row)
                for row in novel_fifo
            ],
        },
        "state_changes": [
            _transition_record(states[row["state_index"]], row) for row in changed
        ],
        "summary_by_load_and_source_policy": population,
        "summary_by_load_seed_and_source_policy": cells,
        "work_accounting": {
            "selector_calls": 2 * len(evaluations),
            "selector_evaluation_wall_s": selector_wall_s,
            "k26_paid_candidate_attempts": sum(
                row["variant_k26"]["attempts"] for row in evaluations
            ),
            "k71_paid_candidate_attempts": sum(
                row["variant_k71"]["attempts"] for row in evaluations
            ),
            "k71_raw_match_legacy_k120_state_count": sum(
                row["variant_k71_raw_matches_legacy_k120"]
                for row in evaluations
            ),
            "k71_physical_match_legacy_k120_state_count": sum(
                row["variant_k71_physical_matches_legacy_k120"]
                for row in evaluations
            ),
            "k71_fifo_match_legacy_k120_state_count": sum(
                row["variant_k71_fifo_matches_legacy_k120"]
                for row in evaluations
            ),
            "free_cache_hit_pops": sum(
                row[checkpoint]["free_cache_hit_pops"]
                for row in evaluations
                for checkpoint in ("variant_k26", "variant_k71")
            ),
            "seed_signature_collisions": sum(
                row[checkpoint]["seed_signature_collisions"]
                for row in evaluations
                for checkpoint in ("variant_k26", "variant_k71")
            ),
            "cache_key_evaluations": sum(
                row[checkpoint]["cache_key_evaluations"]
                for row in evaluations
                for checkpoint in ("variant_k26", "variant_k71")
            ),
            "cache_key_requirement_coordinate_reads": sum(
                row[checkpoint]["cache_key_requirement_coordinate_reads"]
                for row in evaluations
                for checkpoint in ("variant_k26", "variant_k71")
            ),
            "cache_key_duration_reads": sum(
                row[checkpoint]["cache_key_duration_reads"]
                for row in evaluations
                for checkpoint in ("variant_k26", "variant_k71")
            ),
            "cache_key_completion_offset_reads": sum(
                row[checkpoint]["cache_key_completion_offset_reads"]
                for row in evaluations
                for checkpoint in ("variant_k26", "variant_k71")
            ),
            "signature_completion_offset_reads": sum(
                row["signature_completion_offset_reads"] for row in evaluations
            ),
            "arrival_tapes": 0,
            "trajectory_frames": 0,
            "rng_initializations": 0,
            "random_draws": 0,
            "prior_transition_outcome_reads": 0,
        },
        "shadow_advance_checks": advance_checks,
        "scientific_decision": decision,
        "development_advance": (
            decision
            == "ADVANCE_TO_EVIDENCE_REUSE_OR_AUTHORED_TRANSITION"
        ),
        "next_step": (
            "reuse registered known action pairs first, then preregister authored transitions only for the novel FIFO-effective roster"
            if decision
            == "ADVANCE_TO_EVIDENCE_REUSE_OR_AUTHORED_TRANSITION"
            else "close this FIFO-execution quotient branch without tuning or trajectories"
        ),
        "limitations": [
            "The fixed states and mechanism rationale use consumed development evidence; this is not independent confirmation.",
            "K71 certifies maximum one-frame support only; raw, physical, and FIFO witness matches remain diagnostics.",
            "Known-pair labels come only from a prior registered protocol; no prior transition outcome is read in this screen.",
            "Legacy visited-time weights are not variant trajectory time.",
        ],
    }
    if sources != _source_texts():
        raise ValueError("registered FIFO-execution sources changed during execution")
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
