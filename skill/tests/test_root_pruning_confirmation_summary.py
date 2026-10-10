"""Fresh/regression gates on exact mocked records; no LP selector invocation."""
from copy import deepcopy
from dataclasses import fields

import pytest

from algorithm.canonical_count_space_bnb_frame_candidates import CanonicalCountSpaceBnbSelection
from algorithm.experiments.aligned_hardware_shared_threshold_scaling_screen import _metrics, _record, reference_key
from algorithm.experiments.aligned_hardware_queue_aware_canonical_development import selection_from_record
from algorithm.experiments.root_pruning_confirmation_summary import summarize
from types import SimpleNamespace


def selection(queue, budget, lp):
    count = min(queue, 2)
    record = {field.name: 0 for field in fields(CanonicalCountSpaceBnbSelection)}
    record.update(option_counts=[count], option_indices=[0] * count + [None] * (2 - count), service=[count],
        duration=1.0, score=float(queue * count), upper_bound=float(queue * count), gap=0., epsilon=0.,
        exact=True, score_exact=True, canonical_exact=True, lp_node_budget=budget, lp_calls=lp,
        successful_lp_calls=lp, dual_certified_lp_calls=lp, duration_root_count=1,
        lp_matrix_coefficients=lp * 2, maximum_dual_repair=0.)
    return record


def stage(queue, budget, new=False, lp=None):
    lp = int(not new) if lp is None else min(lp, budget)
    item = {"selection": selection(queue, budget, lp), "thresholds": [1.0],
        "preprocessing": {"original_option_count": 1, "requirement_group_count": 1,
                          "duration_threshold_count": 1, "lift_option_probes": 1},
        "capacity_propagation": {"region_visits": 0, "requirement_coordinate_reads": 0,
                                 "tightened_bounds": 0, "removed_count_values": 0}}
    if new:
        item["root_pruning"] = {"roots_seen": 1, "roots_solved": int(lp > 0),
            "roots_skipped_empty": int(lp == 0 and queue == 0),
            "roots_skipped_bound": int(lp == 0 and queue > 0), "roots_skipped_canonical_tie": 0}
    return item


def routed(initial):
    return {"selection": deepcopy(initial["selection"]), "initial_lp": deepcopy(initial),
        "final_lp": deepcopy(initial), "probe": None, "resumed": False,
        "initial_lp_budget": 144, "lp_node_budget": 256, "integer_state_budget": 8192,
        "status": "LP_INITIAL_COMPLETE", "probe_stage": "none"}


@pytest.fixture
def records():
    options = (SimpleNamespace(requirements=(1,), duration=1.0),)
    model = {"node": "toy", "lanes": 2, "options": [{"requirements": [1], "duration": 1.0}]}
    states = []
    for source, queue in (("fresh_confirmation", 4), ("historical_regression", 5), ("new_zero", 0)):
        metadata = {"state_id": source, "source": source, "queue": [queue], "cohort": source,
                    "panel": source, "seed": 123 if source == "fresh_confirmation" else None}
        if source == "historical_regression":
            metadata.update(source_index=7, source_state_id="old7", historical_trigger=True, ordinary_tail=False)
        ref = selection(queue, 4096, 1)
        selected = selection_from_record(ref)
        metric = _metrics()
        _record(metric, selected, SimpleNamespace(options=options), (queue,), 2, 4096,
                reference_key(selected, (queue,), 1))
        traces = {label: {str(cap): stage(queue, cap, new=label == "new") for cap in (1, 2, 144, 256)}
                  for label in ("old", "new")}
        states.append({**metadata, "caps": [1, 2, 144, 256], "reference": ref,
            "reference_metrics": metric, "reference_complete": True, "oracle": None,
            "effective_reference_complete": True, "effective_reference_source": "full_K4096",
            "traces": traces, "old_routed": routed(traces["old"]["144"]), "routed": routed(traces["new"]["144"])})
    protocol = {"models": [model], "cells": [{"node": "toy", "lanes": 2,
        "states": [{key: deepcopy(value) for key, value in state.items() if key in (
            "state_id", "source", "queue", "cohort", "panel", "seed", "source_index", "source_state_id", "historical_trigger", "ordinary_tail")}
                   for state in states]}], "registered_at_utc": "fixed", "reference_budget": 4096, "cpu": 3,
        "initial_lp_budget": 144, "lp_node_budget": 256, "integer_state_budget": 8192,
        "oracle_state_budget": 5000000, "prior_decisions": {"confirmation": "CERTIFICATE_ROUTED_CONFIRMATION_TIMING_HOLD"},
        "limitations": "Work-count confirmation; earlier timing HOLD is retained."}
    row = {"node": "toy", "lanes": 2, "physical_lanes": 2, "option_count": 1,
        "registered_at_utc": "fixed", "states": states, "new_reference_calls": 2,
        "new_reference_oracle_calls": 0, "new_old_routed_calls": 3,
        "runtime_start": {"cpu_affinity": [3], "gc_enabled": True},
        "runtime_end": {"cpu_affinity": [3], "gc_enabled": True}}
    return protocol, [row]


def replace_new_lp(state, lp):
    queue = state["queue"][0]
    state["traces"]["new"] = {str(cap): stage(queue, cap, new=True, lp=lp) for cap in state["caps"]}
    state["routed"] = routed(state["traces"]["new"]["144"])


def test_fresh_confirmation_advances_work_and_quality_but_retains_timing_hold(records):
    summary = summarize(*records)
    assert summary["decision"] == "ROOT_PRUNING_FRESH_CONFIRMATION_ADVANCE"
    assert summary["confirmation_gate_passed"] and summary["prior_hold_retained"]
    assert summary["fresh_state_count"] == summary["regression_state_count"] == summary["zero_state_count"] == 1
    assert summary["executed_interface_calls"] == 30 and summary["native_search_constructions"] == 12
    assert summary["reference_native_search_constructions"] == summary["new_reference_calls"] == 2
    assert summary["borrowed_reference_count"] == 1 and summary["new_old_routed_control_calls"] == 3
    assert summary["reference_work"]["new"]["calls"] == 2 and summary["reference_work"]["borrowed"]["calls"] == 1
    assert summary["all_executed_trace_and_routed_work"]["routed_old"]["calls"] == 3
    assert all(item["states"] == 3 and item["score_regression"] == 0 for item in summary["per_budget"])
    assert all(group["quality_gate_passed"] and group["finite_budget_gate_passed"] for group in summary["groups"].values())


def test_zero_and_regression_savings_cannot_replace_strict_fresh_gain(records):
    protocol, rows = records
    replace_new_lp(rows[0]["states"][0], 1)
    summary = summarize(protocol, rows)
    assert summary["decision"] == "ROOT_PRUNING_FRESH_CONFIRMATION_HOLD"
    assert not summary["work_gate_passed"]
    assert summary["groups"]["fresh_confirmation"]["routed_lp_ratio"] == 1.0
    all_work = summary["all_executed_trace_and_routed_work"]
    assert all_work["routed_new"]["search_work"]["lp_calls"] < all_work["routed_old"]["search_work"]["lp_calls"]


def test_regression_cost_increase_blocks_advance_even_with_fresh_gain(records):
    protocol, rows = records
    replace_new_lp(rows[0]["states"][1], 2)
    summary = summarize(protocol, rows)
    assert summary["execution_checks_passed"] and summary["groups"]["fresh_confirmation"]["strict_pooled_lp_reduction"]
    assert not summary["groups"]["historical_regression"]["cost_nonincrease_in_every_cell"]
    assert summary["decision"] == "ROOT_PRUNING_FRESH_CONFIRMATION_HOLD"


@pytest.mark.parametrize("index", [0, 1])
def test_valid_low_budget_loss_is_recorded_in_its_cohort_and_blocks_finite_gate(records, index):
    protocol, rows = records
    state = rows[0]["states"][index]
    queue = state["queue"][0]
    selected = state["traces"]["new"]["1"]["selection"]
    selected.update(option_counts=[0], option_indices=[None, None], service=[0], score=0.,
        exact=False, score_exact=False, canonical_exact=False, upper_bound=float(queue * 2),
        gap=float(queue * 2), epsilon=2., frontier_regions_remaining=1)
    summary = summarize(protocol, rows)
    assert summary["execution_checks_passed"] and summary["routed_quality_gate_passed"]
    assert not summary["finite_budget_gate_passed"] and summary["decision"] == "ROOT_PRUNING_FRESH_CONFIRMATION_HOLD"
    assert len(summary["finite_budget_score_regressions"]) == 1
    loss = summary["finite_budget_score_regressions"][0]
    assert loss["old_score"] == str(queue * 2) and loss["new_score"] == "0" and loss["lp_budget"] == 1
    assert not summary["groups"][state["source"]]["finite_budget_gate_passed"]


def test_honest_incomplete_reference_has_distinct_decision(records):
    protocol, rows = records
    state = rows[0]["states"][0]
    state["reference"].update(exact=False, score_exact=False, canonical_exact=False, frontier_regions_remaining=1)
    state.update(reference_complete=False, effective_reference_complete=False, effective_reference_source=None)
    state["reference_metrics"].update(score_exact_count=0, canonical_exact_count=0, reference_comparable_count=0)
    summary = summarize(protocol, rows)
    assert summary["execution_checks_passed"] and not summary["effective_references_complete"]
    assert summary["decision"] == "INCOMPLETE_REFERENCE" and not summary["confirmation_gate_passed"]
    assert not summary["failures"]


@pytest.mark.parametrize("mutation", ["missing_cell", "missing_state", "missing_budget", "reference_from_candidate",
    "false_reference_complete", "root_partition", "root_revisited", "routed_initial", "metadata", "reference_work",
    "reference_budget", "old_routed_call_count", "new_reference_call_count", "physical_catalogue"])
def test_actual_integrity_failures_are_invalid(records, mutation):
    protocol, rows = records
    state = rows[0]["states"][0]
    if mutation == "missing_cell":
        rows.clear()
    elif mutation == "missing_state":
        rows[0]["states"].pop()
    elif mutation == "missing_budget":
        state["traces"]["new"].pop("2")
    elif mutation == "reference_from_candidate":
        state["effective_reference_source"] = "candidate"
    elif mutation == "false_reference_complete":
        state["reference_complete"] = False
    elif mutation == "root_partition":
        state["traces"]["new"]["1"]["root_pruning"]["roots_skipped_bound"] = 2
    elif mutation == "root_revisited":
        state["traces"]["new"]["2"]["root_pruning"]["roots_skipped_canonical_tie"] = 1
    elif mutation == "routed_initial":
        state["routed"]["initial_lp"]["capacity_propagation"]["region_visits"] = 1
    elif mutation == "metadata":
        state["seed"] = 999
    elif mutation == "reference_work":
        state["reference_metrics"]["search_work"]["lp_calls"] += 1
    elif mutation == "reference_budget":
        state["reference"]["lp_node_budget"] = 100000
    elif mutation == "old_routed_call_count":
        rows[0]["new_old_routed_calls"] = 0
    elif mutation == "new_reference_call_count":
        rows[0]["new_reference_calls"] = 3
    elif mutation == "physical_catalogue":
        rows[0]["physical_lanes"] = 4
    summary = summarize(protocol, rows)
    assert summary["decision"] == "INVALID_EXECUTION" and summary["failures"]
    assert not summary["confirmation_gate_passed"] and summary["prior_hold_retained"]
