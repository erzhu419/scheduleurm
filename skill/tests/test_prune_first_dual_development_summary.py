"""Pure records for preserved paths, paid dual work and frozen failure targets."""
from copy import deepcopy

import pytest

from algorithm.experiments import prune_first_dual_development_summary as summary
from algorithm.experiments.dual_reuse_development_summary import DUAL_COUNTERS
from skill.tests.test_residual_capacity_trigger_confirmation_summary import residual_state


def _counter_work(record, evaluations):
    if isinstance(record, dict):
        if "dual_reuse" in record:
            work = record["dual_reuse"]
            count = 0 if record["selection"]["lp_node_budget"] == 1 else evaluations
            work.update(child_bound_evaluations=count, child_bound_improvements=0, child_bound_prunes=0,
                child_rhs_entries=4 * count, child_requirement_reads=2 * count, child_weight_reads=2 * count)
        if "option_indices" in record:
            count = 0 if record["lp_node_budget"] == 1 else evaluations
            record.update(canonical_key_bound_evaluations=count, canonical_tie_pruned_regions=count)
        for value in record.values():
            _counter_work(value, evaluations)
    elif isinstance(record, list):
        for value in record:
            _counter_work(value, evaluations)


@pytest.fixture
def records():
    model = {"node": "toy", "lanes": 2, "options": [{"requirements": [1], "duration": 1.}]}
    metadata, states = [], []
    roster = [("historical_regression", queue) for queue in range(4, 8)] + [
        ("ordinary_development", 8), ("zero_queue", 0), ("trigger_control", 9), ("new_trigger", 10)]
    for index, (group, queue) in enumerate(roster):
        original = {"state_id": f"q{queue}", "queue": [queue], "source": group, "cohort": group,
            "panel": group, "caps": [1, 2, 144, 256], "historical_regression": group == "historical_regression",
            "timing_group": group, "timing_source": "previous", "timing_source_index": index,
            "timing_source_state_id": f"q{queue}", "development_source_index": index,
            "development_source_state_id": f"q{queue}"}
        state = residual_state(original, proved=group not in ("trigger_control", "new_trigger"))
        state.update(deepcopy(original))
        state["previous_dual_traces"] = deepcopy(state["traces"]["new"])
        state["previous_dual_routed"] = deepcopy(state["routed"])
        for field in ("previous_dual_traces", "previous_dual_routed"):
            _counter_work(state[field], 2)
        for record in (state["traces"]["new"], state["routed"]):
            _counter_work(record, 1)
        metadata.append(original)
        states.append(state)
    keys = lambda group: [{"node": "toy", "lanes": 2, "queue": state["queue"]}
                          for state in states if state["timing_group"] == group]
    protocol = {"experiment": "pure_mock_prune_first", "registered_at_utc": "frozen", "models": [model],
        "cells": [{"node": "toy", "lanes": 2, "states": metadata}], "cpu": 3,
        "initial_lp_budget": 144, "lp_node_budget": 256, "integer_state_budget": 8192,
        "reference_budget": 4096, "oracle_state_budget": 5000000,
        "historical_regression_keys": keys("historical_regression"), "ordinary_tail_keys": keys("ordinary_development"),
        "trigger_control_keys": keys("trigger_control"), "new_trigger_keys": keys("new_trigger"),
        "previous_timing_failure_keys": keys("historical_regression"), "expected_state_count": 8,
        "expected_selector_calls": 8, "expected_native_advance_calls": 32, "expected_native_constructions": 16,
        "expected_interface_calls": 40,
        "expected_group_counts": {group: sum(state["timing_group"] == group for state in states) for group in summary.GROUPS},
        "prior_decisions": {"residual_timing": "RESIDUAL_CAPACITY_LOCAL_TIMING_HOLD"},
        "limitations": "Pure mocked paid records; no selectors or clocks executed."}
    runtime = {"cpu_affinity": [3], "gc_enabled": True}
    row = {"node": "toy", "lanes": 2, "physical_lanes": 2, "option_count": 1, "registered_at_utc": "frozen",
        "runtime_start": deepcopy(runtime), "runtime_end": deepcopy(runtime), "states": states,
        "new_reference_calls": 0, "new_oracle_calls": 0, "candidate_calls": 8,
        "native_advance_calls": 32, "native_search_constructions": 16}
    return protocol, [row]


def test_complete_preserved_paths_advance_with_all_actual_costs_separate(records):
    result = summary.summarize(*records)
    assert result["decision"] == "PRUNE_FIRST_DUAL_DEVELOPMENT_ADVANCE"
    assert result["development_gate_passed"] and result["prior_hold_retained"]
    assert result["state_count"] == result["executed_selector_calls"] == result["borrowed_reference_count"] == 8
    assert result["executed_native_advance_calls"] == 32 and result["executed_interface_calls"] == 40
    assert result["native_search_constructions"] == 16 and result["new_reference_calls"] == result["new_oracle_calls"] == 0
    assert result["before_work"]["whole_selector"]["dual_reuse"]["child_bound_evaluations"] == 16
    assert result["after_work"]["whole_selector"]["dual_reuse"]["child_bound_evaluations"] == 8
    assert result["executed_dual_reuse_work"]["child_bound_evaluations"] == 16
    assert result["after_work"]["whole_selector"]["axis_capacity"]["merged_values_copied"] == 6
    assert result["after_work"]["whole_selector"]["residual_capacity_completion"]["prefix_reads"] == 6
    assert result["after_work"]["whole_selector"]["shared_integer_states"] == 26
    assert len(result["previous_timing_failures"]) == 4 and result["previous_timing_failures_improved"]
    for key in DUAL_COUNTERS:
        assert result["executed_dual_reuse_work"][key] == sum(result["after_work"][field]["dual_reuse"][key]
                                                            for field in ("native", "whole_selector"))
    assert not result["failures"] and not result["behavior_mismatches"] and not result["dual_work_regressions"]


def test_normalization_ignores_only_permitted_counter_values(records):
    state = records[1][0]["states"][-1]
    assert summary.normalized_result(state["routed"]) == summary.normalized_result(state["previous_dual_routed"])
    changed = deepcopy(state["routed"])
    changed["final_lp"]["dual_reuse"]["certificates_saved"] += 1
    assert summary.normalized_result(changed) != summary.normalized_result(state["routed"])


@pytest.mark.parametrize("field", ("action", "upper", "lp", "integer"))
def test_normalization_preserves_actions_bounds_lp_and_integer_work(records, field):
    record = records[1][0]["states"][-1]["routed"]
    changed = deepcopy(record)
    if field == "action":
        changed["selection"]["option_counts"] = [1]
    elif field == "upper":
        changed["final_lp"]["selection"]["upper_bound"] += 1
    elif field == "lp":
        changed["final_lp"]["selection"]["lp_calls"] += 1
    else:
        changed["probe"]["completion"]["prefix_reads"] += 1
    assert summary.normalized_result(changed) != summary.normalized_result(record)


def test_one_native_cap_growth_cannot_hide_inside_pooled_selector_savings(records):
    protocol, rows = records
    state = rows[0]["states"][4]
    _counter_work(state["traces"]["new"]["2"], 3)
    # Preserve cumulative counters, while cap2 alone exceeds its old snapshot.
    _counter_work(state["traces"]["new"]["144"], 3)
    _counter_work(state["traces"]["new"]["256"], 3)
    _counter_work(state["routed"], 3)
    result = summary.summarize(protocol, rows)
    assert result["execution_checks_passed"] and result["behavior_preserved"]
    assert result["strict_pooled_dual_reduction"] and not result["dual_work_nonincrease_passed"]
    assert result["decision"] == "PRUNE_FIRST_DUAL_DEVELOPMENT_HOLD"
    assert any(item["phase"] == "native" and item["lp_budget"] == 2 for item in result["dual_work_regressions"])


def test_each_fixed_timing_failure_must_improve_not_only_the_pooled_roster(records):
    protocol, rows = records
    state = rows[0]["states"][0]
    _counter_work(state["traces"]["new"], 2)
    _counter_work(state["routed"], 2)
    result = summary.summarize(protocol, rows)
    assert result["strict_pooled_dual_reduction"] and result["dual_work_nonincrease_passed"]
    assert not result["previous_timing_failures_improved"] and result["decision"] == "PRUNE_FIRST_DUAL_DEVELOPMENT_HOLD"


def test_no_paid_dual_reduction_is_an_honest_development_hold(records):
    protocol, rows = records
    for state in rows[0]["states"]:
        _counter_work(state["traces"]["new"], 2)
        _counter_work(state["routed"], 2)
    result = summary.summarize(protocol, rows)
    assert result["execution_checks_passed"] and result["quality_gate_passed"] and result["behavior_preserved"]
    assert not result["strict_pooled_dual_reduction"] and result["decision"] == "PRUNE_FIRST_DUAL_DEVELOPMENT_HOLD"


def test_child_rhs_fees_are_recomputed_from_actual_evaluations(records):
    protocol, rows = records
    state = rows[0]["states"][0]
    state["routed"]["final_lp"]["dual_reuse"]["child_rhs_entries"] -= 1
    result = summary.summarize(protocol, rows)
    assert result["decision"] == "INVALID_EXECUTION" and result["failures"]


def test_incomplete_reference_has_priority_over_the_development_gate(records):
    protocol, rows = records
    state = rows[0]["states"][0]
    state["reference"].update(exact=False, score_exact=False, canonical_exact=False, frontier_regions_remaining=1)
    state.update(reference_complete=False, effective_reference_complete=False, effective_reference_source=None)
    state["reference_metrics"].update(score_exact_count=0, canonical_exact_count=0, reference_comparable_count=0)
    result = summary.summarize(protocol, rows)
    assert result["execution_checks_passed"] and result["decision"] == "INCOMPLETE_REFERENCE"


def test_honest_residual_nonclosure_is_preserved_as_quality_hold(records):
    protocol, rows = records
    result = rows[0]["states"][-1]["routed"]
    probe = result["probe"]
    probe["completion"].update(status="STATE_BUDGET_EXHAUSTED", complete=False,
        states_visited=8184, dfs_states_visited=8181)
    probe["selection"] = deepcopy(probe["score_certified_selection"])
    result.update(selection=deepcopy(probe["selection"]), status="FRONTIER_PROBE_INCOMPLETE")
    actual = summary.summarize(protocol, rows)
    assert actual["execution_checks_passed"] and not actual["quality_gate_passed"]
    assert actual["decision"] == "PRUNE_FIRST_DUAL_DEVELOPMENT_HOLD" and actual["nonclosures"]


@pytest.mark.parametrize("mutation", ("new_reference", "missing_call", "group_count"))
def test_registered_execution_and_group_coverage_cannot_change(records, mutation):
    protocol, rows = records
    if mutation == "new_reference":
        rows[0]["new_reference_calls"] = 1
    elif mutation == "missing_call":
        rows[0]["candidate_calls"] -= 1
    else:
        protocol["expected_group_counts"]["historical_regression"] -= 1
    actual = summary.summarize(protocol, rows)
    assert actual["decision"] == "INVALID_EXECUTION" and not actual["development_gate_passed"]


def test_full_original_state_roster_is_required(records):
    protocol, rows = records
    rows[0]["states"].pop()
    with pytest.raises(ValueError, match="complete frozen timing roster"):
        summary.summarize(protocol, rows)
