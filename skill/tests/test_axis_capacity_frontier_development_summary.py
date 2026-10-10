"""Borrowed frontier controls, actual DP fees, and per-arm shared budgets."""
from copy import deepcopy
from dataclasses import asdict

import pytest

from algorithm.axis_capacity_frontier_score_proof_frame_candidates import AxisCapacityBound, AxisCapacityWork
from algorithm.experiments.axis_capacity_frontier_development_summary import summarize
from algorithm.score_face_canonical_completion_frame_candidates import ScoreFaceCompletionWork
from skill.tests.test_frontier_score_proof_development_summary import records as frontier_records
from skill.tests.test_frontier_trigger_discovery_summary import validation_state
from skill.tests.test_late_score_proof_development_summary import late


def capacity_work(queue):
    return AxisCapacityWork("COMPLETE", True, states_visited=8, axis_states=4, merge_states=4,
        threshold_tables=1, cache_hits=1, bound_checks=2, boxes_pruned=2, bitset_shifts=3,
        bitset_ors=3, bitset_masks=3, service_reads=3, merge_candidates=6, requirement_coordinate_reads=1,
        bounds=(AxisCapacityBound((1, 1), (0,), ((1,),), queue * 2, queue * 2 + 1,
                                  (0, 1), ((0, 1, 2),)),))


def axis_route(prior, *, dp=False):
    result = late(prior, 8) if dp else deepcopy(prior)
    if dp:
        result.update(probe_stage="frontier_late", status="FRONTIER_PROBE_COMPLETE")
    if result["probe_stage"] == "frontier_late":
        proof = result["probe"]["score_proof"]
        if dp:
            queue = int(result["selection"]["score"]) // 2
            proof.update(capacity=asdict(capacity_work(queue)), dfs_states_visited=0,
                         bound_prunes=2, thresholds_checked=2)
        else:
            proof.update(capacity=asdict(AxisCapacityWork()), dfs_states_visited=proof["states_visited"])
    return result


def exhaust_frontier(result):
    probe = result["probe"]
    baseline = deepcopy(probe["baseline_selection"])
    probe["score_proof"].update(status="STATE_BUDGET_EXHAUSTED", complete=False,
                              states_visited=8192, bound_prunes=0, thresholds_checked=1)
    probe.update(selection=deepcopy(baseline), score_certified_selection=deepcopy(baseline),
        completion=asdict(ScoreFaceCompletionWork("SKIPPED_SCORE_UNPROVEN", False, False, 0)))
    result.update(selection=deepcopy(baseline), status="FRONTIER_PROBE_INCOMPLETE")


@pytest.fixture
def records():
    protocol, rows = deepcopy(frontier_records.__wrapped__())
    row = rows[0]
    for index, state in enumerate(row["states"]):
        state.update(borrowed_from="frontier_development", borrowed_index=index,
                     source_state_id=state["state_id"], source_record_source=state["source"])
        state.pop("original_routed")
        state["prior_routed"] = {"old": deepcopy(state["old_routed"]), "new": deepcopy(state["routed"])}
        state["old_routed"] = axis_route(state["prior_routed"]["old"])
        state["routed"] = axis_route(state["prior_routed"]["new"])
    failure = validation_state({"state_id": "trigger:q7", "queue": [7], "seed": 42, "index": 0,
        "panel": "trigger_42", "cohort": "conditional_trigger_validation", "trigger_nodes": ["toy"]})
    failure.update(state_id="axis_failure:trigger:q7", source="axis_failure_development",
        cohort="axis_failure_development", panel="axis_failure_development", borrowed_from="trigger_validation",
        borrowed_index=0, source_state_id="trigger:q7", source_record_source="fresh_confirmation")
    for trace in failure["traces"].values():
        for snapshot in trace.values():
            snapshot["selection"]["frontier_regions_remaining"] = 2
    previous = {}
    for label, field in (("old", "old_routed"), ("new", "routed")):
        result = failure[field]
        result["initial_lp"] = deepcopy(failure["traces"][label]["144"])
        result["final_lp"] = deepcopy(failure["traces"][label]["256"])
        result["probe"]["baseline_selection"] = deepcopy(result["final_lp"]["selection"])
        exported = result["frontier_evidence"]
        exported.update(raw_region_count=2, live_region_count=2)
        second = deepcopy(exported["regions"][0])
        second["lower"] = [0]
        exported["regions"].append(second)
        exported["export_work"].update(heap_entries_read=2, group_ids_copied=2, requirement_rows_copied=2,
            requirement_coordinates_copied=2, box_count_values_copied=4, exact_duration_positions_copied=2,
            representative_options_copied=2, fraction_pairs_copied=5)
        exhaust_frontier(result)
        previous[label] = deepcopy(result)
    failure.update(prior_routed=previous, old_routed=axis_route(previous["old"], dp=True),
                   routed=axis_route(previous["new"], dp=True))
    row["states"].append(failure)
    metadata_fields = ("state_id", "queue", "source", "cohort", "panel", "seed", "index", "source_index",
        "source_state_id", "source_record_source", "prior_index", "historical_regression", "caps",
        "borrowed_from", "borrowed_index", "trigger_nodes")
    protocol["cells"][0]["states"] = [{key: deepcopy(value) for key, value in state.items() if key in metadata_fields}
                                     for state in row["states"]]
    protocol.update(expected_state_count=5, expected_nonzero_state_count=4, expected_failure_development_count=1,
        expected_axis_failure_development_count=1, expected_selector_calls=10, expected_interface_calls=10,
        expected_native_constructions=10, expected_native_advance_calls=0, expected_borrowed_native_snapshots=40,
        axis_failure_development_keys=[{"node": "toy", "lanes": 2, "queue": [7]}],
        borrowed_sources={"frontier_development": {"path": "frozen_frontier", "registered_at_utc": "previous"},
                          "trigger_validation": {"path": "frozen_trigger", "registered_at_utc": "previous_trigger"}})
    protocol["prior_decisions"]["frontier_trigger_discovery"] = "FRONTIER_TRIGGER_VALIDATION_HOLD"
    row["new_old_routed_calls"] = 5
    return protocol, rows


def test_axis_closure_charges_only_new_routes_and_actual_shared_dp_work(records):
    result = summarize(*records)
    assert result["decision"] == "AXIS_CAPACITY_FRONTIER_DEVELOPMENT_ADVANCE" and result["prior_hold_retained"]
    assert result["development_gate_passed"] and result["predecessor_work_gate_passed"]
    assert result["state_count"] == result["borrowed_reference_count"] == 5
    assert result["borrowed_native_snapshots"] == 40
    assert result["executed_interface_calls"] == result["native_search_constructions"] == 10
    assert result["executed_native_advance_calls"] == result["new_reference_calls"] == result["new_reference_oracle_calls"] == 0
    assert result["executed_search_work"]["lp_calls"] == 15
    assert result["borrowed_native_work"]["old"]["search_work"]["lp_calls"] == 10
    assert result["borrowed_native_work"]["new"]["search_work"]["lp_calls"] == 5
    assert result["all_executed_routed_work"]["routed_new"]["shared_integer_states"] == 19
    assert result["borrowed_prior_routed_work"]["new"]["shared_integer_states"] == 8198
    assert result["executed_axis_capacity_work"]["old"]["states_visited"] == 8
    assert result["executed_axis_capacity_work"]["new"]["merge_candidates"] == 6
    assert result["all_executed_routed_work"]["routed_new"]["fallback_dfs_states"] == 1
    assert result["executed_axis_capacity_work"]["new"]["statuses"] == {"NOT_ATTEMPTED": 1, "COMPLETE": 1}
    assert result["executed_frontier_export_work"]["new"]["fraction_pairs_copied"] == 8
    assert all("original_routed" not in state for state in records[1][0]["states"])
    assert not result["failures"]


def test_single_box_uses_original_dfs_and_ordinary_routes_remain_complete_copies(records):
    protocol, rows = records
    ordinary, single = rows[0]["states"][0], rows[0]["states"][-2]
    assert ordinary["routed"] == ordinary["prior_routed"]["new"]
    proof = single["routed"]["probe"]["score_proof"]
    assert proof["capacity"] == asdict(AxisCapacityWork())
    assert proof["dfs_states_visited"] == proof["states_visited"] == 1
    assert summarize(protocol, rows)["decision"] == "AXIS_CAPACITY_FRONTIER_DEVELOPMENT_ADVANCE"


def test_between_arm_strict_gain_is_not_required_for_same_arm_development(records):
    protocol, rows = records
    for state in rows[0]["states"]:
        state["traces"]["old"] = deepcopy(state["traces"]["new"])
        for snapshot in state["traces"]["old"].values():
            snapshot.pop("dual_reuse")
        state["prior_routed"]["old"] = deepcopy(state["prior_routed"]["new"])
        state["old_routed"] = deepcopy(state["routed"])
        for result in (state["prior_routed"]["old"], state["old_routed"]):
            for name in ("initial_lp", "final_lp"):
                result[name].pop("dual_reuse")
    result = summarize(protocol, rows)
    assert result["decision"] == "AXIS_CAPACITY_FRONTIER_DEVELOPMENT_ADVANCE"
    work = result["all_executed_routed_work"]
    assert work["routed_old"]["search_work"] == work["routed_new"]["search_work"]
    assert work["routed_old"]["shared_integer_states"] == work["routed_new"]["shared_integer_states"]
    assert result["predecessor_work_gate_passed"] and not result["failures"]


@pytest.mark.parametrize("mutation", ["ordinary_changed", "paid_lp", "shared_remainder", "dp_state_identity",
    "unsafe_bound", "bad_profile", "missing_dp_fee", "source_pointer", "candidate_reference"])
def test_axis_proof_and_actual_paid_evidence_must_match(records, mutation):
    protocol, rows = records
    state = rows[0]["states"][-1]
    proof = state["routed"]["probe"]["score_proof"]
    if mutation == "ordinary_changed":
        rows[0]["states"][0]["routed"]["status"] = "FRONTIER_PROBE_COMPLETE"
    elif mutation == "paid_lp":
        state["routed"]["final_lp"]["selection"]["lp_calls"] += 1
    elif mutation == "shared_remainder":
        state["routed"]["probe"]["completion"]["state_budget"] = 8192
    elif mutation == "dp_state_identity":
        proof["capacity"]["states_visited"] += 1
    elif mutation == "unsafe_bound":
        proof["capacity"]["bounds"][0]["upper_numerator"] = 15
    elif mutation == "bad_profile":
        proof["capacity"]["bounds"][0]["axis_service_profiles"] = [[0, 1, 3]]
    elif mutation == "missing_dp_fee":
        proof["capacity"]["bitset_masks"] = 0
    elif mutation == "source_pointer":
        state["borrowed_index"] += 1
    else:
        state["effective_reference_source"] = "candidate"
    result = summarize(protocol, rows)
    assert result["decision"] == "INVALID_EXECUTION" and result["failures"]


def test_each_arm_integer_growth_cannot_be_paid_by_other_arm_savings(records):
    protocol, rows = records
    state = rows[0]["states"][-2]
    old = state["old_routed"]["probe"]
    old["score_proof"].update(states_visited=2, dfs_states_visited=2)
    old["completion"]["state_budget"] = 8192 - 2
    result = summarize(protocol, rows)
    assert result["execution_checks_passed"] and result["routed_quality_gate_passed"]
    assert not result["predecessor_work_gate_passed"]
    assert result["decision"] == "AXIS_CAPACITY_FRONTIER_DEVELOPMENT_HOLD"
    assert result["predecessor_work_regressions"][0]["label"] == "old"


def test_honest_fallback_exhaustion_keeps_failure_and_spent_states(records):
    protocol, rows = records
    state = rows[0]["states"][-2]
    exhaust_frontier(state["routed"])
    state["routed"]["probe"]["score_proof"]["dfs_states_visited"] = 8192
    result = summarize(protocol, rows)
    assert result["decision"] == "AXIS_CAPACITY_FRONTIER_DEVELOPMENT_HOLD"
    assert result["execution_checks_passed"] and not result["routed_quality_gate_passed"]
    assert result["all_executed_routed_work"]["routed_new"]["shared_integer_states"] == 8205
    assert not result["failures"]


def test_borrowed_reference_nonclosure_stays_incomplete(records):
    protocol, rows = records
    state = rows[0]["states"][0]
    state["reference"].update(exact=False, score_exact=False, canonical_exact=False, frontier_regions_remaining=1)
    state.update(reference_complete=False, effective_reference_complete=False, effective_reference_source=None)
    state["reference_metrics"].update(score_exact_count=0, canonical_exact_count=0, reference_comparable_count=0)
    result = summarize(protocol, rows)
    assert result["decision"] == "INCOMPLETE_REFERENCE" and result["execution_checks_passed"]
    assert not result["failures"]
