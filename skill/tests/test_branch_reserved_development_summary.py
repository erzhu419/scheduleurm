"""Reserved slots, honest nonclosure, and independently gated historical work."""
from copy import deepcopy

import pytest

from algorithm.experiments.branch_reserved_development_summary import audit_reserved_stage, summarize
from skill.tests.test_root_pruning_confirmation_summary import (
    records as confirmation_records, routed, stage,
)


@pytest.fixture
def records(confirmation_records):
    protocol, rows = deepcopy(confirmation_records)
    for index, state in enumerate(rows[0]["states"]):
        source = ("development", "confirmation", "new_zero")[index]
        state.update(source=source, cohort=source, prior_index=index, historical_regression=index == 1)
    protocol["cells"][0]["states"] = [{key: deepcopy(value) for key, value in state.items() if key in (
        "state_id", "queue", "source", "cohort", "panel", "seed", "source_index", "prior_index", "historical_regression")}
        for state in rows[0]["states"]]
    protocol["historical_regression_keys"] = [{"node": "toy", "lanes": 2, "queue": [5]}]
    rows[0]["new_reference_calls"] = 0
    return protocol, rows


def test_complete_development_charges_no_reference_solves_and_retains_hold(records):
    summary = summarize(*records)
    assert summary["decision"] == "BRANCH_RESERVED_DEVELOPMENT_ADVANCE"
    assert summary["development_gate_passed"] and summary["prior_hold_retained"]
    assert summary["state_count"] == summary["borrowed_reference_count"] == 3
    assert summary["new_reference_calls"] == summary["new_reference_oracle_calls"] == summary["reference_native_search_constructions"] == 0
    assert summary["executed_interface_calls"] == 30 and summary["native_search_constructions"] == 12
    assert summary["new_old_routed_control_calls"] == 3
    assert summary["nonzero_work_gate_passed"] and summary["historical_work_gate_passed"]
    assert all(item["states"] == 3 and item["score_regression"] == 0 for item in summary["per_budget"])


@pytest.mark.parametrize("cap,lp,splits,needle", [
    (1, 1, 0, "actual LP calls plus skipped-root slots exceed the cap"),
    (1, 0, 1, "minimum root cap must not split or solve a branch"),
    (2, 2, 0, "branch LP calls exceed cap minus reserved roots"),
])
def test_skipped_roots_cannot_buy_branch_slots_at_initial_or_resume_cap(cap, lp, splits, needle):
    snapshot = stage(4, cap, new=True)
    snapshot["selection"].update(lp_calls=lp, branch_splits=splits)
    assert needle in audit_reserved_stage(snapshot, cap, 1)


def test_exact_branch_allowance_is_valid_and_root_metadata_stays_fixed():
    snapshot = stage(4, 2, new=True)
    snapshot["selection"].update(lp_calls=1, branch_splits=1)
    assert audit_reserved_stage(snapshot, 2, 1) == []
    assert audit_reserved_stage(stage(4, 1, new=True), 1, 1) == []


def test_historical_matrix_increase_is_not_hidden_by_nonzero_pooled_savings(records):
    protocol, rows = records
    state = rows[0]["states"][1]
    state["traces"]["new"] = {str(cap): stage(5, cap, new=True, lp=1) for cap in state["caps"]}
    for snapshot in state["traces"]["new"].values():
        snapshot["selection"]["lp_matrix_coefficients"] = 3
    state["routed"] = routed(state["traces"]["new"]["144"])
    summary = summarize(protocol, rows)
    assert summary["execution_checks_passed"] and summary["nonzero_work_gate_passed"]
    assert not summary["historical_work_gate_passed"]
    assert summary["decision"] == "BRANCH_RESERVED_DEVELOPMENT_HOLD"
    assert summary["work_increases"][0]["increases"]["lp_matrix_coefficients"] == {"old": 2, "new": 3, "increase": 1}
    assert summary["cells"][0]["nonzero_cost_nonincrease"]
    assert not summary["cells"][0]["historical_cost_nonincrease"]


def test_low_budget_score_loss_is_preserved_and_blocks_finite_gate(records):
    protocol, rows = records
    selected = rows[0]["states"][0]["traces"]["new"]["1"]["selection"]
    selected.update(option_counts=[0], option_indices=[None, None], service=[0], score=0.,
        exact=False, score_exact=False, canonical_exact=False, upper_bound=8., gap=8., epsilon=2., frontier_regions_remaining=1)
    summary = summarize(protocol, rows)
    assert summary["execution_checks_passed"] and summary["routed_quality_gate_passed"]
    assert not summary["finite_budget_gate_passed"]
    assert summary["decision"] == "BRANCH_RESERVED_DEVELOPMENT_HOLD"
    assert summary["finite_budget_score_regressions"][0]["old_score"] == "8"
    assert summary["finite_budget_score_regressions"][0]["new_score"] == "0"


def test_incomplete_reference_is_not_invalid_execution(records):
    protocol, rows = records
    state = rows[0]["states"][0]
    state["reference"].update(exact=False, score_exact=False, canonical_exact=False, frontier_regions_remaining=1)
    state.update(reference_complete=False, effective_reference_complete=False, effective_reference_source=None)
    state["reference_metrics"].update(score_exact_count=0, canonical_exact_count=0, reference_comparable_count=0)
    summary = summarize(protocol, rows)
    assert summary["execution_checks_passed"] and not summary["effective_references_complete"]
    assert summary["decision"] == "INCOMPLETE_REFERENCE" and not summary["development_gate_passed"]
    assert not summary["failures"]


def test_zero_savings_cannot_supply_the_nonzero_strict_reduction(records):
    protocol, rows = records
    for state in rows[0]["states"][:2]:
        state["traces"]["new"] = {str(cap): stage(state["queue"][0], cap, new=True, lp=1) for cap in state["caps"]}
        state["routed"] = routed(state["traces"]["new"]["144"])
    summary = summarize(protocol, rows)
    assert summary["execution_checks_passed"] and not summary["nonzero_work_gate_passed"]
    assert summary["decision"] == "BRANCH_RESERVED_DEVELOPMENT_HOLD"
    all_work = summary["all_executed_trace_and_routed_work"]
    assert all_work["routed_new"]["search_work"]["lp_calls"] < all_work["routed_old"]["search_work"]["lp_calls"]


def test_frozen_subgroup_and_full_roster_are_required(records):
    protocol, rows = records
    rows[0]["states"][1]["historical_regression"] = False
    rows[0]["states"].pop()
    summary = summarize(protocol, rows)
    assert summary["decision"] == "INVALID_EXECUTION"
    messages = [message for failure in summary["failures"] for message in failure["errors"]]
    assert "state roster differs from the frozen development roster" in messages
    assert "historical subgroup flag differs from its independently frozen keys" in messages
