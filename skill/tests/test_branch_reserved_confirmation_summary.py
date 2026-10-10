"""Incremental reserved-slot checks reuse the frozen confirmation gate fixture."""
from copy import deepcopy

from algorithm.experiments.branch_reserved_confirmation_summary import summarize
from algorithm.experiments.root_pruning_confirmation_summary import summarize as base_summary
from skill.tests.test_root_pruning_confirmation_summary import records, routed, replace_new_lp


def test_confirmation_advance_keeps_reference_accounting_and_prior_hold(records):
    result = summarize(*records)
    assert result["decision"] == "BRANCH_RESERVED_FRESH_CONFIRMATION_ADVANCE"
    assert result["confirmation_gate_passed"] and result["reserved_budget_checks_passed"]
    assert result["prior_hold_retained"] and not result["reserved_budget_failures"]
    assert result["executed_interface_calls"] == 30 and result["native_search_constructions"] == 12
    assert result["new_reference_calls"] == 2 and result["borrowed_reference_count"] == 1
    assert result["reference_work"]["new"]["calls"] == 2 and result["reference_work"]["borrowed"]["calls"] == 1


def test_skipped_root_slot_is_charged_even_when_actual_lp_count_fits_cap(records):
    protocol, rows = records
    state = rows[0]["states"][0]
    for snapshot in state["traces"]["new"].values():
        snapshot["selection"].update(lp_calls=1, successful_lp_calls=1, dual_certified_lp_calls=1, lp_matrix_coefficients=2)
    state["routed"] = routed(state["traces"]["new"]["144"])
    assert base_summary(protocol, rows)["execution_checks_passed"]
    result = summarize(protocol, rows)
    assert result["decision"] == "INVALID_EXECUTION" and not result["confirmation_gate_passed"]
    errors = result["reserved_budget_failures"][0]["errors"]
    assert "new cap1: actual LP calls plus skipped-root slots exceed the cap" in errors
    assert "new cap1: branch LP calls exceed cap minus reserved roots" in errors


def test_minimum_root_cap_cannot_split_before_branch_allowance(records):
    protocol, rows = records
    state = rows[0]["states"][0]
    for snapshot in state["traces"]["new"].values():
        snapshot["selection"]["branch_splits"] = 1
    state["routed"] = routed(state["traces"]["new"]["144"])
    assert base_summary(protocol, rows)["execution_checks_passed"]
    result = summarize(protocol, rows)
    assert result["decision"] == "INVALID_EXECUTION"
    assert "new cap1: minimum root cap must not split or solve a branch" in result["reserved_budget_failures"][0]["errors"]


def test_continuation_cannot_spend_the_reserved_root_slot(records):
    protocol, rows = records
    state = rows[0]["states"][0]
    for cap in (1, 2, 144):
        state["traces"]["new"][str(cap)]["selection"].update(
            option_counts=[0], option_indices=[None, None], service=[0], score=0., upper_bound=8., gap=8.,
            epsilon=2., exact=False, score_exact=False, canonical_exact=False, frontier_regions_remaining=1)
    final = state["traces"]["new"]["256"]
    final["selection"].update(lp_calls=256, successful_lp_calls=256, dual_certified_lp_calls=256, lp_matrix_coefficients=512)
    state["routed"] = routed(state["traces"]["new"]["144"])
    state["routed"].update(final_lp=deepcopy(final), selection=deepcopy(final["selection"]), resumed=True, status="LP_RESUME_COMPLETE")
    assert base_summary(protocol, rows)["execution_checks_passed"]
    result = summarize(protocol, rows)
    assert result["decision"] == "INVALID_EXECUTION"
    messages = result["reserved_budget_failures"][0]["errors"]
    assert "new cap256: branch LP calls exceed cap minus reserved roots" in messages
    assert "new routed final_lp: branch LP calls exceed cap minus reserved roots" in messages
    assert result["finite_budget_score_regressions"]


def test_incomplete_reference_keeps_its_original_decision(records):
    protocol, rows = records
    state = rows[0]["states"][0]
    state["reference"].update(exact=False, score_exact=False, canonical_exact=False, frontier_regions_remaining=1)
    state.update(reference_complete=False, effective_reference_complete=False, effective_reference_source=None)
    state["reference_metrics"].update(score_exact_count=0, canonical_exact_count=0, reference_comparable_count=0)
    result = summarize(protocol, rows)
    assert result["decision"] == "INCOMPLETE_REFERENCE" and result["execution_checks_passed"]
    assert result["reserved_budget_checks_passed"] and not result["confirmation_gate_passed"]


def test_existing_fresh_strict_gain_gate_is_preserved(records):
    protocol, rows = records
    replace_new_lp(rows[0]["states"][0], 1)
    result = summarize(protocol, rows)
    assert result["decision"] == "BRANCH_RESERVED_FRESH_CONFIRMATION_HOLD"
    assert result["reserved_budget_checks_passed"] and result["execution_checks_passed"]
    assert not result["work_gate_passed"] and result["prior_hold_retained"]
