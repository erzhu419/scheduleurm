"""Frozen predecessor costs, cumulative dual charges, and finite-cap regressions."""
from copy import deepcopy

import pytest

from algorithm.experiments import dual_reuse_development_summary as summary_module
from algorithm.experiments.dual_reuse_development_summary import DUAL_COUNTERS, summarize
from skill.tests.test_branch_reserved_development_summary import records as reserved_records
from skill.tests.test_root_pruning_confirmation_summary import records as reference_records, routed, stage


def new_trace(state, lp=1):
    trace = {str(cap): stage(state["queue"][0], cap, new=True, lp=lp) for cap in state["caps"]}
    for snapshot in trace.values():
        snapshot["dual_reuse"] = {key: snapshot["selection"]["dual_certified_lp_calls"] if key == "certificates_saved" else 0 for key in DUAL_COUNTERS}
    return trace


@pytest.fixture
def records():
    protocol, rows = deepcopy(reserved_records.__wrapped__(reference_records.__wrapped__()))
    for state in rows[0]["states"]:
        state["source_state_id"] = state["state_id"]
        state["old_traces"] = {str(cap): stage(state["queue"][0], cap, new=True, lp=2) for cap in state["caps"]}
        state["traces"] = new_trace(state)
        state["old_routed"] = routed(state["old_traces"]["144"])
        state["expected_old_routed"] = deepcopy(state["old_routed"])
        state["routed"] = routed(state["traces"]["144"])
    protocol["cells"][0]["states"] = [{key: deepcopy(value) for key, value in state.items() if key in (
        "state_id", "queue", "source", "cohort", "panel", "seed", "source_index", "source_state_id",
        "prior_index", "historical_regression", "caps")} for state in rows[0]["states"]]
    protocol.update(expected_selector_calls=6, expected_native_advance_calls=12,
                    expected_interface_calls=18, expected_native_constructions=9)
    return protocol, rows


def test_complete_work_run_charges_each_actual_search_once_and_keeps_borrowed_work_separate(records):
    summary = summarize(*records)
    assert summary["decision"] == "DUAL_REUSE_DEVELOPMENT_ADVANCE" and summary["prior_hold_retained"]
    assert summary["executed_selector_calls"] == 6 and summary["executed_native_advance_calls"] == 12
    assert summary["executed_interface_calls"] == 18 and summary["native_search_constructions"] == 9
    assert summary["new_reference_calls"] == summary["new_reference_oracle_calls"] == 0
    assert summary["borrowed_reference_count"] == 3
    work = summary["all_executed_trace_and_routed_work"]
    assert work["kernel_old_borrowed"]["search_work"]["lp_calls"] == 6
    assert work["kernel_new"]["search_work"]["lp_calls"] == work["routed_new"]["search_work"]["lp_calls"] == 3
    assert summary["executed_search_work"]["lp_calls"] == 12
    assert summary["executed_dual_reuse_work"]["certificates_saved"] == 6
    assert summary["nonzero_work_gate_passed"] and summary["historical_work_gate_passed"] and summary["native_work_gate_passed"]


def test_integer_gain_alone_can_supply_strict_pooled_gain_but_not_offset_other_costs(records, monkeypatch):
    protocol, rows = records
    for state in rows[0]["states"]:
        state["traces"] = new_trace(state, lp=2)
        state["routed"] = routed(state["traces"]["144"])
    original = summary_module._candidate_work

    def mock_integer_work(results):
        value = original(results)
        value["shared_integer_states"] = sum(5 for result in results if "dual_reuse" not in result["final_lp"])
        return value

    monkeypatch.setattr(summary_module, "_candidate_work", mock_integer_work)
    summary = summarize(protocol, rows)
    assert summary["decision"] == "DUAL_REUSE_DEVELOPMENT_ADVANCE"
    a, b = (summary["nonzero_work"][key] for key in ("routed_old", "routed_new"))
    assert a["search_work"]["lp_calls"] == b["search_work"]["lp_calls"]
    assert b["shared_integer_states"] < a["shared_integer_states"]


def test_native_cap_matrix_growth_blocks_routed_savings(records):
    protocol, rows = records
    state = rows[0]["states"][0]
    for cap in (144, 256):
        state["old_traces"][str(cap)]["selection"]["lp_matrix_coefficients"] = 8
    state["old_routed"] = routed(state["old_traces"]["144"])
    state["expected_old_routed"] = deepcopy(state["old_routed"])
    for cap in (2, 144, 256):
        state["traces"][str(cap)]["selection"]["lp_matrix_coefficients"] = 7
    state["routed"] = routed(state["traces"]["144"])
    summary = summarize(protocol, rows)
    assert summary["execution_checks_passed"] and summary["nonzero_work_gate_passed"]
    assert not summary["native_work_gate_passed"] and summary["decision"] == "DUAL_REUSE_DEVELOPMENT_HOLD"
    assert summary["work_increases"][0]["native"][0]["position"] == 1


def test_fixed_historical_increase_cannot_hide_inside_nonzero_pooled_savings(records):
    protocol, rows = records
    state = rows[0]["states"][1]
    state["traces"] = new_trace(state, lp=2)
    for cap in (144, 256):
        state["traces"][str(cap)]["selection"]["lp_matrix_coefficients"] = 5
    state["routed"] = routed(state["traces"]["144"])
    summary = summarize(protocol, rows)
    assert summary["execution_checks_passed"] and summary["nonzero_work_gate_passed"]
    assert not summary["historical_work_gate_passed"] and not summary["native_work_gate_passed"]
    assert summary["decision"] == "DUAL_REUSE_DEVELOPMENT_HOLD"
    assert summary["work_increases"][0]["historical_regression"]


def test_valid_low_budget_service_loss_is_retained(records):
    protocol, rows = records
    selected = rows[0]["states"][0]["traces"]["1"]["selection"]
    selected.update(option_counts=[0], option_indices=[None, None], service=[0], score=0., exact=False,
                    score_exact=False, canonical_exact=False, upper_bound=8., gap=8., epsilon=2., frontier_regions_remaining=1)
    summary = summarize(protocol, rows)
    assert summary["execution_checks_passed"] and summary["routed_quality_gate_passed"]
    assert not summary["finite_budget_gate_passed"] and summary["decision"] == "DUAL_REUSE_DEVELOPMENT_HOLD"
    assert summary["finite_budget_score_regressions"][0]["new_score"] == "0"


@pytest.mark.parametrize("mutation", ["negative_dual_work", "lost_cumulative_dual_work", "rhs_accounting", "routed_dual_snapshot", "old_literal", "missing_cap"])
def test_actual_dual_accounting_and_independent_snapshot_failures_are_invalid(records, mutation):
    protocol, rows = records
    state = rows[0]["states"][0]
    if mutation == "negative_dual_work":
        state["traces"]["2"]["dual_reuse"]["child_rhs_entries"] = -1
    elif mutation == "lost_cumulative_dual_work":
        state["traces"]["256"]["dual_reuse"]["certificates_saved"] = 0
    elif mutation == "rhs_accounting":
        state["traces"]["256"]["dual_reuse"]["child_bound_evaluations"] = 1
    elif mutation == "routed_dual_snapshot":
        state["routed"]["initial_lp"]["dual_reuse"]["certificates_saved"] = 2
    elif mutation == "old_literal":
        state["expected_old_routed"]["status"] = "INTEGER_PROBE_COMPLETE"
    else:
        state["traces"].pop("2")
    summary = summarize(protocol, rows)
    assert summary["decision"] == "INVALID_EXECUTION" and summary["failures"]


def test_reference_nonclosure_is_distinct_and_zero_gain_cannot_advance(records):
    protocol, rows = records
    for state in rows[0]["states"][:2]:
        state["traces"] = new_trace(state, lp=2)
        state["routed"] = routed(state["traces"]["144"])
    summary = summarize(protocol, rows)
    assert summary["execution_checks_passed"] and not summary["nonzero_work_gate_passed"]
    assert summary["decision"] == "DUAL_REUSE_DEVELOPMENT_HOLD"
    state = rows[0]["states"][0]
    state["reference"].update(exact=False, score_exact=False, canonical_exact=False, frontier_regions_remaining=1)
    state.update(reference_complete=False, effective_reference_complete=False, effective_reference_source=None)
    state["reference_metrics"].update(score_exact_count=0, canonical_exact_count=0, reference_comparable_count=0)
    summary = summarize(protocol, rows)
    assert summary["execution_checks_passed"] and not summary["effective_references_complete"]
    assert summary["decision"] == "INCOMPLETE_REFERENCE" and not summary["failures"]
