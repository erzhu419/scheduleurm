"""Sparse work accounting cannot erase changes to search or proof paths."""
from copy import deepcopy

import pytest

from algorithm.experiments import sparse_zero_delta_affine_development_summary as summary
from skill.tests.test_zero_delta_affine_development_summary import records as zero_records


def add_sparse(stage):
    stage["zero_delta"]["box_coordinate_reads"] = 0
    stage["zero_delta"]["box_coordinate_comparisons"] = 0
    stage["sparse_delta"] = dict.fromkeys(summary.SPARSE_COUNTERS, 0)
    visits = stage["zero_delta"]["variable_visits"]
    stage["sparse_delta"].update(branch_changes=visits, pending_maps_created=visits,
        pending_entries_copied=visits, pending_merge_reads=2 * visits, pending_merge_writes=visits,
        pending_integer_operations=3 * visits, pending_sort_calls=visits, pending_sort_entries=visits,
        guard_delta_reads=3 * visits, guard_delta_comparisons=2 * visits,
        anchor_resets=stage["dual_reuse"]["certificates_saved"])


@pytest.fixture
def records():
    protocol, rows = deepcopy(zero_records.__wrapped__())
    for state in rows[0]["states"]:
        state["original_affine_traces"] = state.pop("previous_dual_traces")
        state["original_affine_routed"] = state.pop("previous_dual_routed")
        state["previous_zero_traces"] = deepcopy(state["traces"]["new"])
        state["previous_zero_routed"] = deepcopy(state["routed"])
        for stage in [*state["traces"]["new"].values(), state["routed"]["initial_lp"], state["routed"]["final_lp"]]:
            add_sparse(stage)
    protocol["experiment"] = "pure_mock_sparse_delta"
    return protocol, rows


def test_complete_paths_truthful_work_advance_as_development(records):
    result = summary.summarize(*records)
    assert result["decision"] == "SPARSE_ZERO_DELTA_AFFINE_DEVELOPMENT_ADVANCE"
    assert not result["independent_confirmation"]
    assert result["state_count"] == 12 and result["executed_interface_calls"] == 60
    assert result["new_reference_calls"] == result["new_oracle_calls"] == 0
    assert not result["failures"] and not result["behavior_mismatches"]
    assert result["executed"]["whole_selector"]["sparse_delta"]["pending_entries_copied"] == 12


@pytest.mark.parametrize("fee", ("pending_merge_reads", "pending_merge_writes", "pending_integer_operations",
    "pending_sort_calls", "guard_delta_reads", "guard_delta_comparisons", "anchor_resets"))
def test_missing_collection_and_guard_fees_hold(records, fee):
    records[1][0]["states"][0]["routed"]["final_lp"]["sparse_delta"][fee] += 1
    result = summary.summarize(*records)
    assert result["decision"] == "SPARSE_ZERO_DELTA_AFFINE_DEVELOPMENT_HOLD" and result["failures"]


@pytest.mark.parametrize("kind", ("canonical", "dual_evaluations", "action", "LP", "proof"))
def test_only_arithmetic_fees_are_normalized(records, kind):
    original = records[1][0]["states"][0]["routed"]
    changed = deepcopy(original)
    if kind == "canonical":
        changed["final_lp"]["selection"]["canonical_key_bound_evaluations"] += 1
    elif kind == "dual_evaluations":
        changed["final_lp"]["dual_reuse"]["child_bound_evaluations"] += 1
    elif kind == "action":
        changed["selection"]["option_indices"] = [1]
    elif kind == "LP":
        changed["final_lp"]["selection"]["lp_calls"] += 1
    else:
        changed["status"] = "different"
    assert summary.normalized_result(changed) != summary.normalized_result(original)


@pytest.mark.parametrize("label,field", (("zero", "previous_zero_routed"), ("affine", "original_affine_routed")))
def test_control_audits_its_real_native_ladder(records, label, field):
    protocol, rows = records
    state, model = rows[0]["states"][0], protocol["models"][0]
    expected, _ = summary._reference(state, protocol, len(model["options"]))
    _, errors, quality = summary._audit_sparse_result(state[field], state,
        protocol, model, rows[0]["lanes"], label, expected)
    assert not errors and quality


def test_cumulative_support_work_cannot_decrease(records):
    state = records[1][0]["states"][0]
    previous = deepcopy(state["traces"]["new"]["1"])
    previous["sparse_delta"]["pending_entries_copied"] = state["traces"]["new"]["256"]["sparse_delta"]["pending_entries_copied"] + 1
    errors = summary.audit_sparse_stage(state["traces"]["new"]["256"], len(state["queue"]), previous)
    assert "cumulative sparse, zero, affine or dual fees decreased" in errors
