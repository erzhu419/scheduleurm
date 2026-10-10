"""Balanced four-arm timing, borrowed references and unchanged scientific gates."""
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from algorithm.experiments import branch_reserved_timing_summary as summary
from algorithm.experiments.aligned_hardware_shared_threshold_scaling_screen import _metrics, _record, reference_key
from algorithm.experiments.aligned_hardware_integer_capacity_development import _combine_metrics
from algorithm.experiments.aligned_hardware_queue_aware_canonical_development import selection_from_record
from algorithm.experiments.score_face_completion_confirmation_summary import _reference
from algorithm.score_face_canonical_completion_frame_candidates import ScoreFaceCompletionWork
from algorithm.strict_score_proof_canonical_completion_frame_candidates import StrictScoreProofWork
from skill.tests.test_root_pruning_confirmation_summary import stage, routed, selection

LABELS = summary.LABELS


def refresh(protocol, rows):
    model = protocol["models"][0]
    for row in rows:
        for state in row["states"]:
            expected, _ = _reference(state, protocol, 1)
            state["metrics"] = {label: _combine_metrics(summary._audit(state["first_results"][label], state, protocol,
                model, row["lanes"], label, expected)[0] for _ in range(8)) for label in LABELS}
        for item in row["warmups"]:
            state = row["states"][item["state_index"]]
            expected, _ = _reference(state, protocol, 1)
            item["metrics"] = summary._audit(item["result"], state, protocol, model, row["lanes"], item["label"], expected)[0]


@pytest.fixture
def case():
    designs = ((0, 1, 3, 2), (1, 2, 0, 3), (2, 3, 1, 0), (3, 0, 2, 1))
    orders = [[LABELS[index] for index in order] for order in designs] * 2
    model = {"node": "toy", "lanes": 2, "options": [{"requirements": [1], "duration": 1.0}]}
    states = []
    for index, (name, cohort) in enumerate((("fresh0", "fresh_confirmation"), ("fresh1", "fresh_confirmation"),
        ("score", "historical_trigger"), ("other", "historical_trigger"), ("untimed", "historical_trigger"), ("tail", "ordinary_tail_regression"))):
        queue = 4 + index
        ref, old, new = selection(queue, 4096, 1), stage(queue, 144), stage(queue, 144, new=True)
        selected = selection_from_record(ref)
        reference_metrics = _metrics()
        _record(reference_metrics, selected, SimpleNamespace(options=(SimpleNamespace(requirements=(1,), duration=1.),)),
                (queue,), 2, 4096, reference_key(selected, (queue,), 1))
        candidate, predecessor = routed(new), routed(old)
        probe = deepcopy(predecessor)
        probe.pop("probe_stage")
        long_stage = stage(queue, 4096)
        long = {"selection": deepcopy(long_stage["selection"]), "baseline_selection": deepcopy(long_stage["selection"]),
            "score_certified_selection": deepcopy(long_stage["selection"]), "preprocessing": long_stage["preprocessing"],
            "capacity_propagation": long_stage["capacity_propagation"], "integer_state_budget": 100000,
            "score_proof": asdict(StrictScoreProofWork("SKIPPED_CANONICAL_COMPLETE", False, True, 100000)),
            "completion": asdict(ScoreFaceCompletionWork("SKIPPED_CANONICAL_COMPLETE", False, True, 100000))}
        metadata = {"state_id": name, "queue": [queue], "panel": cohort, "cohort": cohort,
            "seed": 123 if cohort == "fresh_confirmation" else None, "source_index": index, "source_state_id": name,
            "previous_integer_trigger": cohort == "historical_trigger", "score_unproved_regression": name == "score",
            "previous_untimed_probe_trigger": name == "untimed"}
        states.append({**metadata, "reference": ref, "reference_metrics": reference_metrics,
            "reference_complete": True, "oracle": None, "effective_reference_complete": True, "effective_reference_source": "full_K4096",
            "expected_initial_lp": {key: deepcopy(old[key]) for key in ("selection", "preprocessing", "capacity_propagation")},
            "expected_reserved_initial_lp": {key: deepcopy(new[key]) for key in ("selection", "preprocessing", "capacity_propagation")},
            "expected_results": {"candidate": deepcopy(candidate), "routed": deepcopy(predecessor)},
            "first_results": {"candidate": candidate, "routed": predecessor, "probe": probe, "long": long},
            "repeat_mismatches": {label: [] for label in LABELS}, "metrics": {},
            "blocks": [{"repeat": repeat, "order": list(order)} for repeat, order in enumerate(orders)],
            "wall_ns": {label: [90 if name == "score" and label == "candidate" else 100] * 8 for label in LABELS},
            "cpu_ns": {label: [80] * 8 for label in LABELS}})
    identities = ({key: deepcopy(value) for key, value in state.items() if key in (
        "state_id", "queue", "panel", "cohort", "seed", "source_index", "source_state_id", "previous_integer_trigger",
        "score_unproved_regression", "previous_untimed_probe_trigger")} for state in states)
    design = {"label_orders": [deepcopy(orders) for _ in states], "state_orders": [list(range(6)) for _ in range(8)]}
    def keys(indices):
        return [{"node": "toy", "lanes": 2, "queue": states[index]["queue"]} for index in indices]
    protocol = {"experiment": "mock_four_arm_timing", "registered_at_utc": "fixed", "models": [model],
        "cells": [{"node": "toy", "lanes": 2, "states": list(identities), "orders": design}],
        "execution_order": [["toy", 2]], "labels": list(LABELS), "cpu": 3, "repeats": 8, "warmups_per_label": 2,
        "initial_lp_budget": 144, "lp_node_budget": 256, "integer_state_budget": 8192, "reference_budget": 4096,
        "oracle_state_budget": 5000000, "long_lp_budget": 4096, "control_integer_state_budget": 100000,
        "fresh_nonregression_limit": 1.05, "fresh_speedup_limit": .95, "score_gain_limit": .95,
        "state_ratio_limit": 1.05, "nonregression_limit": 1.05, "trigger_long_ratio_limit": 1., "tail_state_limit": 1.05,
        "direct_predecessor_limit": 1.05, "historical_trigger_keys": keys((2, 3, 4)), "score_unproved_keys": keys((2,)),
        "previous_untimed_probe_keys": keys((4,)), "ordinary_tail_keys": keys((5,)),
        "expected_timed_calls": 192, "expected_warmup_calls": 8, "expected_selector_calls": 200,
        "prior_decisions": {"confirmation": "CERTIFICATE_ROUTED_CONFIRMATION_TIMING_HOLD"}, "limitations": "Mock states, no significance or independent timing claim."}
    row = {"node": "toy", "lanes": 2, "physical_lanes": 2, "option_count": 1, "registered_at_utc": "fixed",
        "states": states, "warmups": [{"label": label, "state_index": 0, "result": deepcopy(states[0]["first_results"][label])}
                                     for _ in range(2) for label in LABELS], "state_orders": deepcopy(design["state_orders"]),
        "new_reference_calls": 0, "new_oracle_calls": 0, "screen_selector_calls": 0, "borrowed_reference_count": 6,
        **{key: {"cpu_affinity": [3], "gc_enabled": True} for key in summary.RUNTIME_KEYS}}
    refresh(protocol, [row])
    return protocol, [row]


def test_all_calls_and_borrowed_costs_are_charged_once_without_new_references(case):
    result = summary.summarize(*case)
    assert result["decision"] == "BRANCH_RESERVED_TIMING_ADVANCE" and not result["fresh_speedup_established"]
    assert result["timed_selector_calls"] == 192 and result["warmup_calls"] == 8 and result["executed_selector_calls"] == 200
    assert result["executed_reference_calls"] == result["executed_oracle_calls"] == result["screen_selector_calls"] == 0
    assert result["borrowed_reference_count"] == result["borrowed_reference_work"]["calls"] == 6
    assert result["executed_method_search_work"]["lp_calls"] == 150
    assert result["timed"]["work"]["routed"]["probe_stages"] == {"none": 48}
    assert result["warmup_work"]["routed"]["calls"] == 2 and result["executed_method_integer_states"] == 0
    assert result["prior_hold_retained"] and len(result["ordinary_tail_states"][0]["wall_ns"]["candidate"]) == 8


@pytest.mark.parametrize("mutation", ["positions", "adjacency", "missing_state", "actual_state_order", "block_order",
    "warmup_count", "duration_count", "budget", "frozen_result", "quality", "new_reference"])
def test_supported_design_accounting_and_quality_failures_are_invalid(case, mutation):
    protocol, rows = case
    orders, row = protocol["cells"][0]["orders"], rows[0]
    state = row["states"][0]
    if mutation == "positions":
        orders["label_orders"][0] = [list(LABELS)] * 8
    elif mutation == "adjacency":
        orders["label_orders"][0] = [list(LABELS[index:] + LABELS[:index]) for index in range(4)] * 2
    elif mutation == "missing_state":
        orders["state_orders"][0].pop()
    elif mutation == "actual_state_order":
        row["state_orders"][0].reverse()
    elif mutation == "block_order":
        state["blocks"][0]["order"].reverse()
    elif mutation == "warmup_count":
        row["warmups"].pop()
    elif mutation == "duration_count":
        state["wall_ns"]["candidate"].pop()
    elif mutation == "budget":
        state["first_results"]["candidate"]["integer_state_budget"] = 0
    elif mutation == "frozen_result":
        state["expected_results"]["routed"]["status"] = "different"
    elif mutation == "quality":
        state["metrics"]["long"]["canonical_action_mismatch_count"] = 1
    elif mutation == "new_reference":
        row["new_reference_calls"] = 1
    result = summary.summarize(protocol, rows)
    assert result["decision"] == "INVALID_EXECUTION" and not result["confirmation_gate_passed"]
    assert result["accounting_failures"] and result["prior_hold_retained"]


@pytest.mark.parametrize("index,label,value,gate", [
    (2, "candidate", 96, "score_unproved_probe_gain"),
    (5, "candidate", 106, "tail_no_state_regression_vs_probe"),
    (3, "routed", 90, "max_paired_state_slowdown"),
])
def test_old_score_and_tail_gates_and_new_direct_predecessor_gate_are_preserved(case, index, label, value, gate):
    protocol, rows = case
    rows[0]["states"][index]["wall_ns"][label] = [value] * 8
    result = summary.summarize(protocol, rows)
    assert result["execution_checks_passed"] and result["comparison_quality_gate_passed"]
    assert result["decision"] == "BRANCH_RESERVED_TIMING_HOLD"
    gates = result["timing_gates"]["direct_routed_historical" if label == "routed" else "regression"]
    assert not gates[gate]


def test_direct_routed_has_no_score_subset_gain_requirement(case):
    protocol, rows = case
    rows[0]["states"][2]["wall_ns"]["routed"] = [90] * 8
    result = summary.summarize(protocol, rows)
    assert result["decision"] == "BRANCH_RESERVED_TIMING_ADVANCE"
    assert result["score_unproved_timing"]["comparisons"]["routed"]["wall_ratio"] == 1.
    assert all(result["timing_gates"]["direct_routed_historical"].values())


def test_fresh_speedup_requires_all_three_controls(case):
    protocol, rows = case
    for state in rows[0]["states"][:2]:
        state["wall_ns"]["candidate"] = [90] * 8
    result = summary.summarize(protocol, rows)
    assert result["fresh_speedup_established"]
    for state in rows[0]["states"][:2]:
        state["wall_ns"]["routed"] = [90] * 8
    result = summary.summarize(protocol, rows)
    assert result["confirmation_gate_passed"] and not result["fresh_speedup_established"]


def test_honest_unresolved_borrowed_reference_retains_incomplete_decision(case):
    protocol, rows = case
    state = rows[0]["states"][0]
    state["reference"].update(exact=False, score_exact=False, canonical_exact=False, frontier_regions_remaining=1)
    state.update(reference_complete=False, effective_reference_complete=False, effective_reference_source=None)
    state["reference_metrics"].update(score_exact_count=0, canonical_exact_count=0, reference_comparable_count=0)
    refresh(protocol, rows)
    result = summary.summarize(protocol, rows)
    assert result["decision"] == "INCOMPLETE_REFERENCE" and result["execution_checks_passed"]
    assert not result["confirmation_gate_passed"] and not result["accounting_failures"]
