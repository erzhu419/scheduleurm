"""Pure three-arm records distinguish new queues from historically fresh controls."""
from collections import Counter
from copy import deepcopy

import pytest

from algorithm.experiments import zero_delta_affine_fresh_confirmation_summary as summary
from skill.tests.test_affine_dual_fresh_confirmation_summary import _open_reference
from skill.tests.test_zero_delta_affine_development_summary import records as old_records


@pytest.fixture
def records():
    protocol, rows = deepcopy(old_records.__wrapped__())
    cell, row = protocol["cells"][0], rows[0]
    fresh_states, fresh_metadata = [], []
    for index in range(4):
        metadata, state = deepcopy(cell["states"][index]), deepcopy(row["states"][index])
        for key in tuple(metadata):
            if "source_index" in key or "source_state_id" in key or key.startswith("timing_source"):
                metadata.pop(key)
                state.pop(key, None)
        metadata.update(state_id=f"independent:{index}", independent_source="fresh",
            timing_group="independent_fresh", confirmation_source="fresh", seed=2893 + index,
            source="independent_confirmation", cohort="independent_confirmation", panel="independent_confirmation")
        state.update(deepcopy(metadata))
        fresh_metadata.append(metadata)
        fresh_states.append(state)
    for index, (metadata, state) in enumerate(zip(cell["states"], row["states"], strict=True)):
        pointer = {"independent_source": "borrowed_control", "independent_source_index": index,
                   "independent_source_state_id": state["state_id"]}
        metadata.update(pointer)
        state.update(pointer)
    cell["states"] = fresh_metadata + cell["states"]
    row["states"] = fresh_states + row["states"]
    protocol.update(experiment="pure_mock_zero_delta_independent", expected_state_count=16,
        expected_independent_state_count=4, expected_borrowed_state_count=12,
        expected_selector_calls=12, expected_native_advance_calls=48,
        expected_native_constructions=24, expected_interface_calls=60, expected_new_reference_calls=4,
        expected_group_counts=dict(Counter(state["timing_group"] for state in row["states"])))
    row.update(new_reference_calls=4, new_oracle_calls=0, candidate_calls=4, affine_calls=4,
        branch_calls=4, native_advance_calls=48, native_search_constructions=24)
    return protocol, rows


def test_complete_independent_confirmation_advances_and_keeps_old_fresh_controls_borrowed(records):
    result = summary.summarize(*records)
    assert result["decision"] == "ZERO_DELTA_AFFINE_FRESH_CONFIRMATION_ADVANCE"
    assert all(result[key] for key in ("execution_checks_passed", "references_complete", "quality_gate_passed",
        "behavior_preserved", "work_gate_passed", "confirmation_gate_passed", "prior_hold_retained"))
    assert result["state_count"] == 16 and result["independent_state_count"] == 4 and result["borrowed_state_count"] == 12
    assert result["new_reference_calls"] == 4 and result["borrowed_reference_count"] == 12
    assert result["new_oracle_calls"] == 0
    assert result["executed_selector_calls"] == 12 and result["executed_native_advance_calls"] == 48
    assert result["native_search_constructions"] == 24 and result["executed_interface_calls"] == 60
    assert result["reference_work"]["new"]["search_work"]["lp_calls"] == 4
    assert result["reference_work"]["borrowed"]["search_work"]["lp_calls"] == 12
    assert result["groups"]["fresh_confirmation"]["independent_sources"] == {"borrowed_control": 4}
    assert result["groups"]["independent_fresh"]["independent_sources"] == {"fresh": 4}
    assert set(result["executed_work"]["whole_selector"]) == {"branch", "affine", "candidate"}
    for name in summary.ZERO_COUNTERS:
        assert result["executed_zero_delta_work"][name] == sum(result["executed_work"][field]["candidate"]["zero_delta"][name]
            for field in ("native", "whole_selector"))
    assert not result["failures"] and not result["nonclosures"] and not result["behavior_mismatches"]


def test_no_fresh_zero_hits_still_pass_the_quality_gate(records):
    for state in records[1][0]["states"][:4]:
        state["traces"]["new"] = deepcopy(state["previous_dual_traces"])
        state["routed"] = deepcopy(state["previous_dual_routed"])
        for stage in [*state["traces"]["new"].values(), state["routed"]["initial_lp"], state["routed"]["final_lp"]]:
            calls, flags = stage["dual_reuse"]["child_bound_evaluations"], stage["selection"]["dual_column_checks"]
            stage["zero_delta"] = dict.fromkeys(summary.ZERO_COUNTERS, 0)
            stage["zero_delta"].update(lower_flags_saved=flags, lower_flag_comparisons=flags,
                guard_calls=calls, variable_visits=calls, box_coordinate_reads=4 * calls,
                box_coordinate_comparisons=2 * calls, upper_price_reads=calls, upper_price_zero_comparisons=calls)
    result = summary.summarize(*records)
    assert result["decision"] == "ZERO_DELTA_AFFINE_FRESH_CONFIRMATION_ADVANCE"
    assert result["executed_zero_delta_work"]["zero_delta_skips"] == 0
    assert result["work_gate_passed"] and result["confirmation_gate_passed"]


@pytest.mark.parametrize("field", ("candidate_calls", "affine_calls", "branch_calls", "native_advance_calls",
    "native_search_constructions", "new_reference_calls"))
def test_any_borrowed_execution_count_is_rejected(records, field):
    records[1][0][field] += 1
    result = summary.summarize(*records)
    assert result["decision"] == "ZERO_DELTA_AFFINE_FRESH_CONFIRMATION_HOLD"
    assert not result["execution_checks_passed"] and result["failures"]


@pytest.mark.parametrize("arm,field", (("candidate", "zero_delta"), ("candidate", "affine_dual"), ("affine", "affine_dual")))
def test_all_actual_guard_and_affine_fee_vectors_are_audited(records, arm, field):
    result = records[1][0]["states"][0]["routed" if arm == "candidate" else "previous_dual_routed"]
    key = "box_coordinate_reads" if field == "zero_delta" else "certificate_numerator_products"
    result["final_lp"][field][key] += 1
    actual = summary.summarize(*records)
    assert actual["decision"] == "ZERO_DELTA_AFFINE_FRESH_CONFIRMATION_HOLD"
    assert not actual["work_gate_passed"] and actual["failures"]


def test_native_affine_coverage_is_compared_with_actual_zero_guard_calls(records):
    state = records[1][0]["states"][0]
    state["previous_dual_traces"]["256"]["dual_reuse"]["child_bound_evaluations"] += 1
    result = summary.summarize(*records)
    assert not result["execution_checks_passed"]
    assert any("guard coverage" in message for item in result["failures"] for message in item["errors"])


def test_changed_canonical_work_is_reported_separately_as_a_behavior_mismatch(records):
    state = records[1][0]["states"][0]
    state["traces"]["new"]["256"]["selection"]["canonical_key_bound_evaluations"] += 1
    result = summary.summarize(*records)
    assert not result["behavior_preserved"] and result["behavior_mismatches"]
    assert result["behavior_mismatches"][0]["independent_source"] == "fresh"


def test_paired_honest_nonclosure_retains_every_state_and_is_a_quality_hold(records):
    state = records[1][0]["states"][2]
    for field in ("routed", "previous_dual_routed"):
        result, probe = state[field], state[field]["probe"]
        probe["completion"].update(status="STATE_BUDGET_EXHAUSTED", complete=False,
            states_visited=8184, dfs_states_visited=8181)
        probe["selection"] = deepcopy(probe["score_certified_selection"])
        result.update(selection=deepcopy(probe["selection"]), status="FRONTIER_PROBE_INCOMPLETE")
    result = summary.summarize(*records)
    assert result["execution_checks_passed"] and result["behavior_preserved"]
    assert not result["quality_gate_passed"] and not result["confirmation_gate_passed"]
    assert result["decision"] == "ZERO_DELTA_AFFINE_FRESH_CONFIRMATION_HOLD" and result["state_count"] == 16
    assert {item["label"] for item in result["nonclosures"]} == {"candidate", "affine"}
    assert all(item["independent_source"] == "fresh" for item in result["nonclosures"])


def test_unclosed_independent_reference_keeps_all_states_without_method_rescue(records):
    _open_reference(records[1][0]["states"][0], score_exact=False)
    result = summary.summarize(*records)
    assert result["execution_checks_passed"] and not result["references_complete"]
    assert result["decision"] == "ZERO_DELTA_AFFINE_FRESH_CONFIRMATION_HOLD" and result["state_count"] == 16
    assert result["new_reference_calls"] == 4 and result["new_oracle_calls"] == 0


def test_conditional_oracle_depends_only_on_independent_reference_score_certificate(records):
    state = records[1][0]["states"][0]
    _open_reference(state, score_exact=True)
    reference = state["reference"]
    oracle = {**{name: deepcopy(reference[name]) for name in ("option_counts", "option_indices", "service", "duration", "score")},
        "conditional_on_external_score_certificate": True, "state_visit_budget": 5000000,
        "score_fraction": [int(reference["score"]), 1], "duration_fraction": [1, 1],
        "complete": True, "canonical_exact": True, "status": "COMPLETE", "work": dict.fromkeys(summary.ORACLE_COUNTERS, 0)}
    oracle["work"]["states_visited"] = 2
    state.update(oracle=oracle, effective_reference_complete=True, effective_reference_source="independent_score_face_oracle")
    records[1][0]["new_oracle_calls"] = 1
    result = summary.summarize(*records)
    assert result["decision"] == "ZERO_DELTA_AFFINE_FRESH_CONFIRMATION_ADVANCE"
    assert result["oracle_work"]["new"]["work"]["states_visited"] == 2
    state["reference"]["score_exact"] = False
    state["reference_metrics"]["score_exact_count"] = 0
    result = summary.summarize(*records)
    assert not result["execution_checks_passed"] and result["failures"]


def test_independent_fresh_records_cannot_invent_an_axis_predecessor(records):
    records[1][0]["states"][0]["prior_routed"] = deepcopy(records[1][0]["states"][-1]["prior_routed"])
    result = summary.summarize(*records)
    assert not result["execution_checks_passed"] and result["failures"]


def test_borrowed_historically_fresh_pointer_cannot_be_replaced(records):
    records[1][0]["states"][4]["independent_source_index"] += 1
    with pytest.raises(ValueError, match="source pointers"):
        summary.summarize(*records)


def test_missing_registered_independent_queue_cannot_be_dropped(records):
    records[1][0]["states"].pop(0)
    with pytest.raises(ValueError, match="independent queues"):
        summary.summarize(*records)


def test_borrowed_previous_fresh_group_cannot_replace_new_independent_coverage(records):
    records[0]["expected_group_counts"]["independent_fresh"] -= 1
    result = summary.summarize(*records)
    assert not result["execution_checks_passed"] and not result["confirmation_gate_passed"]


def test_missing_one_of_three_native_ladders_is_not_accepted(records):
    records[1][0]["states"][0]["previous_dual_traces"].pop("256")
    with pytest.raises(ValueError, match="three actual"):
        summary.summarize(*records)
