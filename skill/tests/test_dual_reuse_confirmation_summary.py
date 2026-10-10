"""Fresh dual-reuse gates reuse mocked independent reference and LP records."""
from copy import deepcopy

import pytest

from algorithm.experiments import dual_reuse_confirmation_summary as summary_module
from algorithm.experiments.dual_reuse_confirmation_summary import summarize
from algorithm.experiments.dual_reuse_development_summary import DUAL_COUNTERS
from skill.tests.test_root_pruning_confirmation_summary import records as base_records, routed, stage


def new_trace(state, lp=1):
    trace = {str(cap): stage(state["queue"][0], cap, new=True, lp=lp) for cap in state["caps"]}
    for snapshot in trace.values():
        snapshot["dual_reuse"] = {key: snapshot["selection"]["dual_certified_lp_calls"] if key == "certificates_saved" else 0 for key in DUAL_COUNTERS}
    return trace


@pytest.fixture
def records():
    protocol, rows = deepcopy(base_records.__wrapped__())
    for state in rows[0]["states"]:
        state["historical_regression"] = state["source"] == "historical_regression"
        state["traces"]["old"] = {str(cap): stage(state["queue"][0], cap, new=True, lp=2) for cap in state["caps"]}
        state["traces"]["new"] = new_trace(state)
        state["old_routed"] = routed(state["traces"]["old"]["144"])
        state["routed"] = routed(state["traces"]["new"]["144"])
    protocol["cells"][0]["states"] = [{key: deepcopy(value) for key, value in state.items() if key in (
        "state_id", "queue", "source", "cohort", "panel", "seed", "source_index", "source_state_id",
        "historical_trigger", "ordinary_tail", "historical_regression", "caps")} for state in rows[0]["states"]]
    protocol.update(historical_regression_keys=[{"node": "toy", "lanes": 2, "queue": [5]}],
                    expected_selector_calls=6, expected_native_advance_calls=24,
                    expected_interface_calls=30, expected_native_constructions=12, expected_new_reference_calls=2)
    return protocol, rows


def test_independent_confirmation_charges_both_native_ladders_and_routes_once(records):
    result = summarize(*records)
    assert result["decision"] == "DUAL_REUSE_FRESH_CONFIRMATION_ADVANCE" and result["confirmation_gate_passed"]
    assert result["prior_hold_retained"] and not result["failures"]
    assert result["fresh_state_count"] == result["regression_state_count"] == result["zero_state_count"] == 1
    assert result["executed_selector_calls"] == 6 and result["executed_native_advance_calls"] == 24
    assert result["executed_interface_calls"] == 30 and result["native_search_constructions"] == 12
    assert result["new_reference_calls"] == result["reference_native_search_constructions"] == 2
    assert result["borrowed_reference_count"] == result["reference_work"]["borrowed"]["calls"] == 1
    assert result["reference_work"]["new"]["calls"] == 2 and result["new_reference_oracle_calls"] == 0
    assert result["executed_search_work"]["lp_calls"] == 18
    assert result["executed_dual_reuse_work"]["certificates_saved"] == 6
    assert all(group["native_cost_nonincrease_in_every_cell"] for group in result["groups"].values())
    assert all("expected_old_routed" not in state for state in records[1][0]["states"])


def test_regression_and_zero_savings_cannot_replace_strict_fresh_gain(records):
    protocol, rows = records
    state = rows[0]["states"][0]
    state["traces"]["new"] = new_trace(state, lp=2)
    state["routed"] = routed(state["traces"]["new"]["144"])
    result = summarize(protocol, rows)
    assert result["execution_checks_passed"] and result["routed_quality_gate_passed"]
    assert not result["fresh_work_gate_passed"] and result["decision"] == "DUAL_REUSE_FRESH_CONFIRMATION_HOLD"
    assert result["groups"]["historical_regression"]["strict_pooled_lp_reduction"]
    assert result["groups"]["new_zero"]["strict_pooled_lp_reduction"]


def test_fresh_integer_reduction_can_supply_gain_without_lp_reduction(records, monkeypatch):
    protocol, rows = records
    for state in rows[0]["states"]:
        state["traces"]["new"] = new_trace(state, lp=2)
        state["routed"] = routed(state["traces"]["new"]["144"])
    original = summary_module._work

    def mock_integer_work(states):
        work = original(states)
        work["routed_old"]["shared_integer_states"] = 5 * len(states)
        return work

    monkeypatch.setattr(summary_module, "_work", mock_integer_work)
    result = summarize(protocol, rows)
    fresh = result["groups"]["fresh_confirmation"]
    assert result["decision"] == "DUAL_REUSE_FRESH_CONFIRMATION_ADVANCE"
    assert not fresh["strict_pooled_lp_reduction"] and fresh["strict_pooled_integer_reduction"]


def test_native_cap_growth_blocks_fresh_route_savings(records):
    protocol, rows = records
    state = rows[0]["states"][0]
    for cap in (144, 256):
        state["traces"]["old"][str(cap)]["selection"]["lp_matrix_coefficients"] = 8
    state["old_routed"] = routed(state["traces"]["old"]["144"])
    for cap in (2, 144, 256):
        state["traces"]["new"][str(cap)]["selection"]["lp_matrix_coefficients"] = 5
    state["routed"] = routed(state["traces"]["new"]["144"])
    result = summarize(protocol, rows)
    assert result["execution_checks_passed"] and result["fresh_work_gate_passed"]
    assert not result["native_work_gate_passed"] and result["decision"] == "DUAL_REUSE_FRESH_CONFIRMATION_HOLD"
    assert result["work_increases"][0]["native"][0]["position"] == 1


def test_historical_increase_is_gated_separately_from_fresh_savings(records):
    protocol, rows = records
    state = rows[0]["states"][1]
    state["traces"]["new"] = new_trace(state, lp=2)
    for cap in (144, 256):
        state["traces"]["new"][str(cap)]["selection"]["lp_matrix_coefficients"] = 5
    state["routed"] = routed(state["traces"]["new"]["144"])
    result = summarize(protocol, rows)
    assert result["execution_checks_passed"] and result["fresh_work_gate_passed"]
    assert not result["historical_work_gate_passed"] and not result["native_work_gate_passed"]
    assert result["decision"] == "DUAL_REUSE_FRESH_CONFIRMATION_HOLD"
    assert result["work_increases"][0]["historical_regression"]


def test_valid_finite_score_loss_and_honest_reference_nonclosure_are_preserved(records):
    protocol, rows = records
    state = rows[0]["states"][0]
    state["traces"]["new"]["1"]["selection"].update(option_counts=[0], option_indices=[None, None], service=[0],
        score=0., exact=False, score_exact=False, canonical_exact=False, upper_bound=8., gap=8., epsilon=2., frontier_regions_remaining=1)
    result = summarize(protocol, rows)
    assert result["execution_checks_passed"] and not result["finite_budget_gate_passed"]
    assert result["decision"] == "DUAL_REUSE_FRESH_CONFIRMATION_HOLD"
    assert result["finite_budget_score_regressions"][0]["new_score"] == "0"
    state["reference"].update(exact=False, score_exact=False, canonical_exact=False, frontier_regions_remaining=1)
    state.update(reference_complete=False, effective_reference_complete=False, effective_reference_source=None)
    state["reference_metrics"].update(score_exact_count=0, canonical_exact_count=0, reference_comparable_count=0)
    result = summarize(protocol, rows)
    assert result["decision"] == "INCOMPLETE_REFERENCE" and result["execution_checks_passed"] and not result["failures"]


def test_honest_old_control_nonclosure_also_blocks_quality(records):
    protocol, rows = records
    state = rows[0]["states"][0]
    for snapshot in state["traces"]["old"].values():
        snapshot["selection"].update(option_counts=[0], option_indices=[None, None], service=[0], score=0.,
            exact=False, score_exact=False, canonical_exact=False, upper_bound=8., gap=8., epsilon=2., frontier_regions_remaining=1)
    state["old_routed"] = routed(state["traces"]["old"]["144"])
    state["old_routed"].update(resumed=True, final_lp=deepcopy(state["traces"]["old"]["256"]),
        selection=deepcopy(state["traces"]["old"]["256"]["selection"]), status="LP_RESUME_INCOMPLETE")
    result = summarize(protocol, rows)
    assert result["execution_checks_passed"] and not result["routed_quality_gate_passed"]
    assert result["decision"] == "DUAL_REUSE_FRESH_CONFIRMATION_HOLD"
    assert result["routed_nonclosures"][0]["label"] == "old"


@pytest.mark.parametrize("mutation", ["dual_rhs", "dual_cumulative", "old_roots", "routed_snapshot", "historical_flag", "missing_cap", "candidate_reference"])
def test_actual_accounting_independence_and_roster_failures_are_invalid(records, mutation):
    protocol, rows = records
    state = rows[0]["states"][0]
    if mutation == "dual_rhs":
        state["traces"]["new"]["256"]["dual_reuse"]["child_bound_evaluations"] = 1
    elif mutation == "dual_cumulative":
        state["traces"]["new"]["256"]["dual_reuse"]["certificates_saved"] = 0
    elif mutation == "old_roots":
        state["traces"]["old"]["2"]["root_pruning"]["roots_seen"] = 2
    elif mutation == "routed_snapshot":
        state["routed"]["initial_lp"]["dual_reuse"]["certificates_saved"] = 2
    elif mutation == "historical_flag":
        rows[0]["states"][1]["historical_regression"] = False
    elif mutation == "missing_cap":
        state["traces"]["new"].pop("2")
    else:
        state["effective_reference_source"] = "candidate"
    result = summarize(protocol, rows)
    assert result["decision"] == "INVALID_EXECUTION" and result["failures"]
    assert not result["confirmation_gate_passed"] and result["prior_hold_retained"]
