"""Paid strict boxes, separate export fees, and same-arm predecessor gates."""
from copy import deepcopy
from dataclasses import asdict

import pytest

from algorithm.experiments.frontier_score_proof_development_summary import (
    audit_frontier_evidence, summarize,
)
from algorithm.score_face_canonical_completion_frame_candidates import ScoreFaceCompletionWork
from skill.tests.test_late_score_proof_development_summary import records as late_records, late
from skill.tests.test_root_pruning_confirmation_summary import selection


def evidence():
    return {"incumbent_score_fraction": [12, 1], "root_count": 1, "raw_region_count": 1,
        "live_region_count": 1, "live_tie_region_count": 0,
        "regions": [{"threshold_fraction": [1, 1], "bound_fraction": [13, 1], "eligible": [0],
            "requirements": [[1]], "lower": [1], "upper": [2], "exact_duration_positions": [0], "representative_options": [0]}],
        "export_work": {"heap_entries_read": 1, "group_ids_copied": 1, "requirement_rows_copied": 1,
            "requirement_coordinates_copied": 1, "box_count_values_copied": 2, "exact_duration_positions_copied": 1,
            "representative_options_copied": 1, "fraction_pairs_copied": 3}}


@pytest.fixture
def records():
    protocol, rows = deepcopy(late_records.__wrapped__())
    for index, state in enumerate(rows[0]["states"]):
        state.update(borrowed_from="late_development", borrowed_index=index,
            source_state_id=state["state_id"], source_record_source=state["source"])
        state["prior_routed"] = {"old": deepcopy(state["old_routed"]), "new": deepcopy(state["routed"])}
        for label, field in (("old", "old_routed"), ("new", "routed")):
            if state["cohort"] == "failure_development":
                result = late(state["original_routed"][label], 1)
                result.update(probe_stage="frontier_late", status="FRONTIER_PROBE_COMPLETE", frontier_evidence=evidence())
                result["probe"]["score_proof"]["thresholds_checked"] = 1
                state[field] = result
            else:
                state[field]["frontier_evidence"] = None
    protocol["cells"][0]["states"] = [{key: deepcopy(value) for key, value in state.items() if key in (
        "state_id", "queue", "source", "cohort", "panel", "seed", "source_index", "source_state_id", "prior_index",
        "historical_regression", "caps", "borrowed_from", "borrowed_index", "source_record_source")} for state in rows[0]["states"]]
    protocol["borrowed_sources"] = {"late_development": {"path": "previous_late", "registered_at_utc": "previous"}}
    return protocol, rows


def test_complete_frontier_proof_preserves_paid_lp_and_charges_only_actual_exports(records):
    result = summarize(*records)
    assert result["decision"] == "FRONTIER_SCORE_PROOF_DEVELOPMENT_ADVANCE" and result["prior_hold_retained"]
    assert result["development_gate_passed"] and result["failure_development_gate_passed"] and result["predecessor_work_gate_passed"]
    assert result["executed_interface_calls"] == result["native_search_constructions"] == 8
    assert result["executed_native_advance_calls"] == result["new_reference_calls"] == result["new_reference_oracle_calls"] == 0
    assert result["borrowed_native_snapshots"] == 32 and result["executed_search_work"]["lp_calls"] == 12
    assert result["all_executed_routed_work"]["routed_old"]["shared_integer_states"] == 6
    assert result["all_executed_routed_work"]["routed_new"]["shared_integer_states"] == 6
    assert result["borrowed_prior_routed_work"]["old"]["shared_integer_states"] == 35
    assert result["borrowed_prior_routed_work"]["new"]["shared_integer_states"] == 15
    assert result["executed_frontier_export_work"]["old"]["box_count_values_copied"] == 2
    assert result["executed_frontier_export_work"]["new"]["fraction_pairs_copied"] == 3
    assert result["all_executed_routed_work"]["probe_stage_counts"] == {
        "old": {"none": 3, "frontier_late": 1}, "new": {"none": 3, "frontier_late": 1}}


@pytest.mark.parametrize("mutation", ["score_target", "non_strict_bound", "missing_box", "wrong_requirements", "invalid_lower", "representative", "export_fee", "proof_skipped_box"])
def test_actual_exported_geometry_and_proof_coverage_must_match_paid_frontier(records, mutation):
    protocol, rows = records
    new = rows[0]["states"][-1]["routed"]
    exported = new["frontier_evidence"]
    if mutation == "score_target":
        exported["incumbent_score_fraction"] = [13, 1]
    elif mutation == "non_strict_bound":
        exported["regions"][0]["bound_fraction"] = [12, 1]
    elif mutation == "missing_box":
        exported["regions"].clear()
    elif mutation == "wrong_requirements":
        exported["regions"][0]["requirements"] = [[2]]
    elif mutation == "invalid_lower":
        exported["regions"][0]["lower"] = [3]
    elif mutation == "representative":
        exported["regions"][0]["representative_options"] = [1]
    elif mutation == "export_fee":
        exported["export_work"]["fraction_pairs_copied"] = 0
    else:
        new["probe"]["score_proof"]["thresholds_checked"] = 0
    result = summarize(protocol, rows)
    assert result["decision"] == "INVALID_EXECUTION" and result["failures"]


def test_representative_order_follows_original_ids_at_each_duration_stratum():
    model = {"options": [{"requirements": [1], "duration": 2.}, {"requirements": [2], "duration": 1.},
                          {"requirements": [1], "duration": 1.}]}
    exported = evidence()
    exported["regions"][0].update(eligible=[1, 0], requirements=[[2], [1]], lower=[0, 1], upper=[1, 2],
                                  exact_duration_positions=[0, 1], representative_options=[1, 2])
    exported["export_work"].update(group_ids_copied=2, requirement_rows_copied=2, requirement_coordinates_copied=2,
        box_count_values_copied=4, exact_duration_positions_copied=2, representative_options_copied=2)
    paid = selection(6, 256, 1)
    paid.update(option_counts=[0, 0, 2], option_indices=[2, 2], exact=False, score_exact=False, canonical_exact=False,
        upper_bound=13., gap=1., epsilon=1 / 6, frontier_regions_remaining=1)
    stage = {"selection": paid, "thresholds": [1.]}
    assert audit_frontier_evidence(exported, stage, model, [6], 2) == []


def test_same_arm_integer_growth_cannot_be_paid_by_between_arm_gain(records):
    protocol, rows = records
    old = rows[0]["states"][-1]["old_routed"]
    old["probe"]["score_proof"]["states_visited"] = 40
    old["probe"]["completion"]["state_budget"] = 8192 - 40
    result = summarize(protocol, rows)
    assert result["execution_checks_passed"] and result["routed_quality_gate_passed"] and result["nonzero_work_gate_passed"]
    assert not result["predecessor_work_gate_passed"] and result["decision"] == "FRONTIER_SCORE_PROOF_DEVELOPMENT_HOLD"
    change = result["predecessor_work_regressions"][0]
    assert change["label"] == "old" and change["prior_integer_states"] == 35 and change["actual_integer_states"] == 45


def test_ordinary_projection_cannot_export_or_repeat_a_probe(records):
    protocol, rows = records
    rows[0]["states"][0]["routed"]["frontier_evidence"] = evidence()
    result = summarize(protocol, rows)
    assert result["decision"] == "INVALID_EXECUTION" and result["failures"]


def test_honest_frontier_proof_exhaustion_stays_incomplete(records):
    protocol, rows = records
    new = rows[0]["states"][-1]["routed"]
    probe = new["probe"]
    baseline = deepcopy(probe["baseline_selection"])
    probe["score_proof"].update(status="STATE_BUDGET_EXHAUSTED", complete=False, states_visited=8192)
    probe.update(selection=deepcopy(baseline), score_certified_selection=deepcopy(baseline),
        completion=asdict(ScoreFaceCompletionWork("SKIPPED_SCORE_UNPROVEN", False, False, 0)))
    new.update(selection=deepcopy(baseline), status="FRONTIER_PROBE_INCOMPLETE")
    result = summarize(protocol, rows)
    assert result["execution_checks_passed"] and not result["routed_quality_gate_passed"]
    assert not result["failure_development_gate_passed"] and result["decision"] == "FRONTIER_SCORE_PROOF_DEVELOPMENT_HOLD"
    assert result["all_executed_routed_work"]["routed_new"]["shared_integer_states"] == 8192


def test_incomplete_reference_remains_honest_and_cannot_advance(records):
    protocol, rows = records
    state = rows[0]["states"][0]
    state["reference"].update(exact=False, score_exact=False, canonical_exact=False, frontier_regions_remaining=1)
    state.update(reference_complete=False, effective_reference_complete=False, effective_reference_source=None)
    state["reference_metrics"].update(score_exact_count=0, canonical_exact_count=0, reference_comparable_count=0)
    result = summarize(protocol, rows)
    assert result["decision"] == "INCOMPLETE_REFERENCE" and result["execution_checks_passed"] and not result["failures"]
