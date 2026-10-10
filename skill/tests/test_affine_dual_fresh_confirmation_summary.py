"""Pure fresh and borrowed records; no kernel, reference or timing execution."""
from collections import Counter
from copy import deepcopy

import pytest

from algorithm.experiments import affine_dual_fresh_confirmation_summary as summary
from skill.tests.test_affine_dual_development_summary import records as development_records


@pytest.fixture
def records():
    protocol, rows = deepcopy(development_records.__wrapped__())
    cell, row = protocol["cells"][0], rows[0]
    fresh_states, fresh_metadata = [], []
    for index, source_index in enumerate((0, 1, 6, 7)):
        metadata, state = deepcopy(cell["states"][source_index]), deepcopy(row["states"][source_index])
        for key in tuple(metadata):
            if "source_index" in key or "source_state_id" in key or key.startswith("timing_source"):
                metadata.pop(key)
                state.pop(key, None)
        metadata.update(state_id=f"fresh:{index}", source="fresh_confirmation", cohort="fresh_confirmation",
            panel="fresh_confirmation", confirmation_source="fresh", timing_group="fresh_confirmation",
            historical_regression=False, seed=2026092881 + index, index=0)
        state.update(deepcopy(metadata))
        for field in ("prior_routed", "previous_dual_traces", "previous_dual_routed"):
            state.pop(field)
        fresh_metadata.append(metadata)
        fresh_states.append(state)
    for index, (metadata, state) in enumerate(zip(cell["states"], row["states"], strict=True)):
        pointer = {"confirmation_source": "borrowed_control", "confirmation_source_index": index,
                   "confirmation_source_state_id": state["state_id"]}
        metadata.update(pointer)
        state.update(pointer)
    # A borrowed old source can itself be fresh; only confirmation_source dispatches.
    cell["states"][-1]["source"] = row["states"][-1]["source"] = "fresh_confirmation"
    cell["states"] = fresh_metadata + cell["states"]
    row["states"] = fresh_states + row["states"]
    protocol.update(experiment="pure_mock_affine_fresh", expected_state_count=12,
        expected_fresh_state_count=4, expected_control_state_count=8, expected_selector_calls=8,
        expected_native_advance_calls=32, expected_native_constructions=16,
        expected_interface_calls=40, expected_new_reference_calls=4,
        expected_group_counts=dict(Counter(state["timing_group"] for state in row["states"])))
    row.update(new_reference_calls=4, new_oracle_calls=0, candidate_calls=4, branch_calls=4,
        native_advance_calls=32, native_search_constructions=16)
    return protocol, rows


def test_independent_fresh_quality_advances_without_fake_predecessors(records):
    result = summary.summarize(*records)
    assert result["decision"] == "AFFINE_DUAL_FRESH_CONFIRMATION_ADVANCE"
    assert result["execution_checks_passed"] and result["references_complete"] and result["quality_gate_passed"]
    assert result["affine_work_validated"] and result["confirmation_gate_passed"] and result["prior_hold_retained"]
    assert result["state_count"] == 12 and result["fresh_state_count"] == 4 and result["control_state_count"] == 8
    assert result["new_reference_calls"] == 4 and result["new_oracle_calls"] == 0
    assert result["executed_selector_calls"] == 8 and result["executed_native_advance_calls"] == 32
    assert result["native_search_constructions"] == 16 and result["executed_interface_calls"] == 40
    assert result["borrowed_reference_count"] == 8
    assert result["reference_work"]["new"]["search_work"]["lp_calls"] == 4
    assert result["reference_work"]["borrowed"]["search_work"]["lp_calls"] == 8
    assert result["executed_work"]["whole_selector"]["candidate"]["residual_capacity_completion"]["prefix_reads"] == 6
    assert result["borrowed_work"]["whole_selector"]["candidate"]["residual_capacity_completion"]["prefix_reads"] == 6
    for name in summary.AFFINE_COUNTERS:
        assert result["executed_affine_dual_work"][name] == sum(
            result["executed_work"][field]["candidate"]["affine_dual"][name]
            for field in ("native", "whole_selector"))
    assert not result["failures"] and not result["nonclosures"]


def test_fresh_quality_does_not_claim_unexecuted_prune_first_path_equivalence(records):
    state = records[1][0]["states"][0]
    assert "previous_dual_routed" not in state and "prior_routed" not in state
    for item in [state["traces"]["new"]["144"], state["traces"]["new"]["256"],
                 state["routed"]["initial_lp"], state["routed"]["final_lp"]]:
        item["selection"]["canonical_key_bound_evaluations"] += 1
    state["routed"]["selection"]["canonical_key_bound_evaluations"] += 1
    result = summary.summarize(*records)
    assert result["decision"] == "AFFINE_DUAL_FRESH_CONFIRMATION_ADVANCE"


@pytest.mark.parametrize("fee", ("price_reads", "box_coordinate_reads", "delta_integer_operations"))
def test_omitted_affine_operations_invalidate_confirmation(records, fee):
    state = records[1][0]["states"][0]
    state["routed"]["final_lp"]["affine_dual"][fee] -= 1
    result = summary.summarize(*records)
    assert result["decision"] == "INVALID_EXECUTION" and result["failures"]


@pytest.mark.parametrize("field", ("merged_values_copied", "exact_slot_numerators", "prefix_reads", "coverage_coordinate_reads"))
def test_fresh_retained_proof_and_prefix_fees_are_directly_audited(records, field):
    probe = records[1][0]["states"][2]["routed"]["probe"]
    if field == "merged_values_copied":
        probe["score_proof"]["capacity"][field] -= 1
    elif field == "exact_slot_numerators":
        probe["score_proof"]["capacity"]["bounds"][0][field][1] += 1
    else:
        probe["completion"][field] -= 1
    result = summary.summarize(*records)
    assert result["decision"] == "INVALID_EXECUTION" and result["failures"]


def test_honest_fresh_nonclosure_remains_a_quality_hold(records):
    result = records[1][0]["states"][2]["routed"]
    probe = result["probe"]
    probe["completion"].update(status="STATE_BUDGET_EXHAUSTED", complete=False,
        states_visited=8184, dfs_states_visited=8181)
    probe["selection"] = deepcopy(probe["score_certified_selection"])
    result.update(selection=deepcopy(probe["selection"]), status="FRONTIER_PROBE_INCOMPLETE")
    actual = summary.summarize(*records)
    assert actual["execution_checks_passed"] and not actual["quality_gate_passed"]
    assert actual["decision"] == "AFFINE_DUAL_FRESH_CONFIRMATION_HOLD"
    assert actual["nonclosures"][0]["confirmation_source"] == "fresh"
    assert actual["executed_work"]["whole_selector"]["candidate"]["shared_integer_states"] >= 8192


def _open_reference(state, *, score_exact):
    state["reference"].update(exact=False, canonical_exact=False, score_exact=score_exact,
        frontier_regions_remaining=1, canonical_tie_regions_remaining=1)
    state.update(reference_complete=False, effective_reference_complete=False, effective_reference_source=None)
    state["reference_metrics"].update(score_exact_count=int(score_exact), canonical_exact_count=0,
        reference_comparable_count=0)


def test_unproved_independent_reference_keeps_all_states_and_incomplete_priority(records):
    state = records[1][0]["states"][0]
    _open_reference(state, score_exact=False)
    result = summary.summarize(*records)
    assert result["execution_checks_passed"] and not result["references_complete"]
    assert result["decision"] == "INCOMPLETE_REFERENCE" and result["state_count"] == 12
    assert result["new_reference_calls"] == 4


def test_oracle_is_only_independent_full_reference_score_triggered(records):
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
    assert result["decision"] == "AFFINE_DUAL_FRESH_CONFIRMATION_ADVANCE"
    assert result["new_oracle_calls"] == 1 and result["oracle_work"]["new"]["work"]["states_visited"] == 2
    state["reference"]["score_exact"] = False
    state["reference_metrics"]["score_exact_count"] = 0
    result = summary.summarize(*records)
    assert result["decision"] == "INVALID_EXECUTION" and result["failures"]


@pytest.mark.parametrize("field", ("candidate_calls", "branch_calls", "native_advance_calls", "native_search_constructions", "new_reference_calls"))
def test_borrowed_controls_cannot_be_counted_as_new_execution(records, field):
    records[1][0][field] += 1
    result = summary.summarize(*records)
    assert result["decision"] == "INVALID_EXECUTION" and not result["confirmation_gate_passed"]


def test_missing_or_replaced_fresh_queue_is_not_dropped(records):
    records[1][0]["states"].pop(0)
    with pytest.raises(ValueError, match="frozen fresh queues"):
        summary.summarize(*records)


def test_borrowed_control_source_pointer_is_frozen(records):
    records[1][0]["states"][-1]["confirmation_source_index"] += 1
    with pytest.raises(ValueError, match="borrowed control pointers"):
        summary.summarize(*records)


def test_ordinary_controls_cannot_fill_missing_fresh_group(records):
    records[0]["expected_group_counts"]["fresh_confirmation"] -= 1
    result = summary.summarize(*records)
    assert result["decision"] == "INVALID_EXECUTION"


def test_native_reserved_root_cap_must_remain_actual_catalogue_bound(records):
    state = records[1][0]["states"][0]
    state["traces"]["new"]["1"]["selection"]["branch_splits"] = 1
    result = summary.summarize(*records)
    assert result["decision"] == "INVALID_EXECUTION"
    assert any("minimum root cap" in message for item in result["failures"] for message in item["errors"])
