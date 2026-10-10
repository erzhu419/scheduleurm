"""Retained tables, truthful canonical prefix fees, and same-arm cost gates."""
from copy import deepcopy
from dataclasses import asdict

import pytest

from algorithm.experiments.residual_capacity_frontier_development_summary import (
    RESIDUAL_COUNTERS, audit_residual_completion, summarize,
)
from algorithm.residual_capacity_frontier_frame_candidates import ResidualCapacityCompletionWork
from algorithm.score_face_canonical_completion_frame_candidates import ScoreFaceCompletionWork
from skill.tests.test_axis_capacity_frontier_development_summary import records as axis_records


@pytest.fixture
def records():
    protocol, rows = deepcopy(axis_records.__wrapped__())
    for index, state in enumerate(rows[0]["states"]):
        state.update(borrowed_from="axis_capacity_development", borrowed_index=index,
                     source_state_id=state["state_id"], source_record_source=state["source"])
        state["prior_routed"] = {"old": deepcopy(state["old_routed"]), "new": deepcopy(state["routed"])}
        for label, field in (("old", "old_routed"), ("new", "routed")):
            result = state[field]
            if result["probe_stage"] != "frontier_late":
                continue
            capacity = result["probe"]["score_proof"]["capacity"]
            capacity["merged_values_copied"] = 3 * len(capacity["bounds"])
            for bound in capacity["bounds"]:
                bound["exact_slot_numerators"] = [0, state["queue"][0], 2 * state["queue"][0]]
            if capacity["bounds"]:
                result["probe"]["completion"] = asdict(ResidualCapacityCompletionWork(
                    "COMPLETE", True, True, 8184, states_visited=5, thresholds_checked=1,
                    score_fraction=(14, 1), prefix_states=3, dfs_states_visited=2, prefix_reads=3,
                    capacity_queries=1, tables_reused=1, coverage_coordinate_reads=2))
                previous = state["prior_routed"][label]
                previous["probe"]["completion"] = asdict(ScoreFaceCompletionWork(
                    "STATE_BUDGET_EXHAUSTED", True, False, 8184, states_visited=8184,
                    thresholds_checked=1, score_fraction=(14, 1)))
                previous["probe"]["selection"] = deepcopy(previous["probe"]["score_certified_selection"])
                previous.update(selection=deepcopy(previous["probe"]["selection"]), status="FRONTIER_PROBE_INCOMPLETE")
    for registered, state in zip(protocol["cells"][0]["states"], rows[0]["states"]):
        registered.update({key: state[key] for key in (
            "borrowed_from", "borrowed_index", "source_state_id", "source_record_source")})
    protocol["borrowed_sources"] = {"axis_capacity_development": {"path": "frozen_axis_hold", "registered_at_utc": "previous_axis"}}
    protocol["prior_decisions"]["axis_capacity_development"] = "AXIS_CAPACITY_FRONTIER_DEVELOPMENT_HOLD"
    return protocol, rows


def test_retained_tables_and_prefixes_have_separate_fees_without_new_native_calls(records):
    result = summarize(*records)
    assert result["decision"] == "RESIDUAL_CAPACITY_FRONTIER_DEVELOPMENT_ADVANCE"
    assert result["retained_capacity_checks_passed"] and result["prior_hold_retained"]
    assert result["state_count"] == result["borrowed_reference_count"] == 5
    assert result["borrowed_native_snapshots"] == 40
    assert result["executed_interface_calls"] == result["native_search_constructions"] == 10
    assert result["executed_native_advance_calls"] == result["new_reference_calls"] == result["new_reference_oracle_calls"] == 0
    assert result["executed_search_work"]["lp_calls"] == 15
    assert result["all_executed_routed_work"]["routed_new"]["shared_integer_states"] == 19
    assert result["borrowed_prior_routed_work"]["new"]["shared_integer_states"] == 8198
    assert result["executed_axis_capacity_work"]["new"]["merged_values_copied"] == 3
    fee = result["executed_residual_completion_work"]["new"]
    assert fee["extended_calls"] == fee["tables_reused"] == 1
    assert fee["prefix_states"] == fee["prefix_reads"] == 3 and fee["dfs_states_visited"] == 2
    assert result["all_executed_routed_work"]["routed_new"]["canonical_dfs_states"] == 7
    assert result["groups"]["axis_failure_development"]["work"]["routed_new"]["axis_capacity"]["merged_values_copied"] == 3
    assert not result["failures"]


def test_ordinary_and_single_box_completion_keep_previous_semantics(records):
    protocol, rows = records
    ordinary, single = rows[0]["states"][0], rows[0]["states"][-2]
    assert ordinary["routed"] == ordinary["prior_routed"]["new"]
    current, previous = single["routed"]["probe"], single["prior_routed"]["new"]["probe"]
    assert current["completion"] == previous["completion"]
    assert current["score_proof"]["capacity"]["merged_values_copied"] == 0
    assert summarize(protocol, rows)["decision"] == "RESIDUAL_CAPACITY_FRONTIER_DEVELOPMENT_ADVANCE"


@pytest.mark.parametrize("mutation", ["merged_vector", "copy_fee", "old_proof_counter", "prefix_identity",
    "prefix_reads", "coverage_reads", "table_count", "query_fee", "source_pointer"])
def test_retained_arithmetic_and_actual_prefix_work_must_match(records, mutation):
    protocol, rows = records
    state = rows[0]["states"][-1]
    probe = state["routed"]["probe"]
    proof, completion = probe["score_proof"], probe["completion"]
    if mutation == "merged_vector":
        proof["capacity"]["bounds"][0]["exact_slot_numerators"][1] = 8
    elif mutation == "copy_fee":
        proof["capacity"]["merged_values_copied"] = 0
    elif mutation == "old_proof_counter":
        proof["memo_hits"] += 1
    elif mutation == "prefix_identity":
        completion["dfs_states_visited"] += 1
    elif mutation == "prefix_reads":
        completion["prefix_reads"] = 0
    elif mutation == "coverage_reads":
        completion["coverage_coordinate_reads"] = 1
    elif mutation == "table_count":
        completion["tables_reused"] = 0
    elif mutation == "query_fee":
        completion["capacity_queries"] = 3
    else:
        state["borrowed_index"] += 1
    result = summarize(protocol, rows)
    assert result["decision"] == "INVALID_EXECUTION" and result["failures"]


def test_same_arm_integer_growth_cannot_use_other_arm_savings(records):
    protocol, rows = records
    state = rows[0]["states"][-1]
    prior = state["prior_routed"]["new"]
    prior["probe"]["completion"] = asdict(ScoreFaceCompletionWork(
        "COMPLETE", True, True, 8184, states_visited=4, score_fraction=(14, 1)))
    prior["probe"]["selection"] = deepcopy(state["routed"]["selection"])
    prior.update(selection=deepcopy(prior["probe"]["selection"]), status="FRONTIER_PROBE_COMPLETE")
    result = summarize(protocol, rows)
    assert result["execution_checks_passed"] and result["routed_quality_gate_passed"]
    assert result["decision"] == "RESIDUAL_CAPACITY_FRONTIER_DEVELOPMENT_HOLD"
    assert result["predecessor_work_regressions"][0]["label"] == "new"


def test_honest_canonical_exhaustion_keeps_quality_hold_and_all_spent_states(records):
    protocol, rows = records
    result_record = rows[0]["states"][-1]["routed"]
    probe = result_record["probe"]
    probe["completion"].update(status="STATE_BUDGET_EXHAUSTED", complete=False,
        states_visited=8184, dfs_states_visited=8181)
    probe["selection"] = deepcopy(probe["score_certified_selection"])
    result_record.update(selection=deepcopy(probe["selection"]), status="FRONTIER_PROBE_INCOMPLETE")
    result = summarize(protocol, rows)
    assert result["execution_checks_passed"] and not result["routed_quality_gate_passed"]
    assert result["predecessor_work_gate_passed"] and not result["failures"]
    assert result["decision"] == "RESIDUAL_CAPACITY_FRONTIER_DEVELOPMENT_HOLD"
    assert result["all_executed_routed_work"]["routed_new"]["shared_integer_states"] == 8198


def test_borrowed_reference_nonclosure_stays_incomplete(records):
    protocol, rows = records
    state = rows[0]["states"][0]
    state["reference"].update(exact=False, score_exact=False, canonical_exact=False, frontier_regions_remaining=1)
    state.update(reference_complete=False, effective_reference_complete=False, effective_reference_source=None)
    state["reference_metrics"].update(score_exact_count=0, canonical_exact_count=0, reference_comparable_count=0)
    result = summarize(protocol, rows)
    assert result["decision"] == "INCOMPLETE_REFERENCE" and result["execution_checks_passed"]
    assert not result["failures"]


def test_coverage_rejection_preserves_dfs_but_keeps_actual_scan_fees(records):
    protocol, rows = records
    state = rows[0]["states"][-1]
    prior = deepcopy(state["prior_routed"]["new"]["probe"])
    probe = deepcopy(state["routed"]["probe"])
    probe.update(selection=deepcopy(prior["selection"]), completion=deepcopy(prior["completion"]))
    probe["completion"].update({key: 0 for key in RESIDUAL_COUNTERS})
    probe["completion"].update(dfs_states_visited=8184, coverage_coordinate_reads=3)
    model = deepcopy(protocol["models"][0])
    model["options"].append({**model["options"][0], "requirements": [2]})
    assert audit_residual_completion(probe, prior, model, [7], 2) == []
    probe["completion"].update(tables_reused=1, prefix_states=3, prefix_reads=3, dfs_states_visited=8181)
    assert audit_residual_completion(probe, prior, model, [7], 2)


def test_partial_prefix_consumes_only_remaining_shared_states(records):
    protocol, rows = records
    state = rows[0]["states"][-1]
    probe = deepcopy(state["routed"]["probe"])
    probe["integer_state_budget"] = 10
    probe["completion"].update(status="STATE_BUDGET_EXHAUSTED", complete=False, state_budget=2,
        states_visited=2, prefix_states=2, prefix_reads=2, dfs_states_visited=0, capacity_queries=0)
    probe["selection"] = deepcopy(probe["score_certified_selection"])
    assert audit_residual_completion(probe, state["prior_routed"]["new"]["probe"], protocol["models"][0], [7], 2) == []
    probe["completion"]["state_budget"] = 10
    assert audit_residual_completion(probe, state["prior_routed"]["new"]["probe"], protocol["models"][0], [7], 2)
