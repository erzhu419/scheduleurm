"""Pure frozen records: exact paths, lazy preparation and charged child updates."""
from copy import deepcopy

import pytest

from algorithm.experiments import affine_dual_development_summary as summary
from algorithm.experiments.dual_reuse_development_summary import DUAL_COUNTERS
from skill.tests.test_prune_first_dual_development_summary import records as prune_records


def add_affine(record):
    if isinstance(record, dict):
        if "dual_reuse" in record:
            dual = record["dual_reuse"]
            count, saved = dual["child_bound_evaluations"], dual["certificates_saved"]
            prepared = int(count > 0)
            # One-variable, one-resource toy certificates; changed lower only.
            record["affine_dual"] = dict.fromkeys(summary.AFFINE_COUNTERS, 0)
            record["affine_dual"].update(
                certificates_prepared=prepared, certificate_numerator_products=saved,
                price_reads=6 * prepared, denominator_lcm_steps=4 * prepared,
                integer_scalings=5 * prepared, coefficient_requirement_reads=prepared,
                coefficient_weight_reads=prepared, coefficient_integer_operations=5 * prepared,
                bound_variable_visits=count, box_coordinate_reads=4 * count,
                box_coordinate_comparisons=2 * count, changed_lower_coordinates=count,
                delta_coefficient_reads=count, delta_integer_operations=3 * count,
                bound_integer_products=2 * count, bound_rational_conversions=count)
            for name in summary.FEE_COUNTERS:
                dual[name] = 0
        for value in list(record.values()):
            add_affine(value)
    elif isinstance(record, list):
        for value in record:
            add_affine(value)


@pytest.fixture
def records():
    protocol, rows = deepcopy(prune_records.__wrapped__())
    protocol["experiment"] = "pure_mock_affine_dual"
    protocol["affine_focus_keys"] = deepcopy(protocol["previous_timing_failure_keys"])
    protocol["affine_focus_keys"].extend(protocol["ordinary_tail_keys"] + protocol["new_trigger_keys"])
    protocol["prior_decisions"]["prune_first_timing"] = "PRUNE_FIRST_DUAL_LOCAL_TIMING_HOLD"
    for metadata, state in zip(protocol["cells"][0]["states"], rows[0]["states"], strict=True):
        pointer = {"affine_source_index": metadata["development_source_index"],
                   "affine_source_state_id": metadata["state_id"]}
        metadata.update(pointer)
        state.update(pointer)
        state["previous_dual_traces"] = deepcopy(state["traces"]["new"])
        state["previous_dual_routed"] = deepcopy(state["routed"])
        add_affine(state["traces"]["new"])
        add_affine(state["routed"])
    return protocol, rows


def test_exact_paths_advance_without_converting_different_fees_into_speed(records):
    result = summary.summarize(*records)
    assert result["decision"] == "AFFINE_DUAL_DEVELOPMENT_ADVANCE" and result["prior_hold_retained"]
    assert result["behavior_preserved"] and result["affine_work_validated"] and result["work_gate_passed"]
    assert result["affine_updates_exercised"] and len(result["affine_focus_states"]) == 6
    assert result["state_count"] == result["executed_selector_calls"] == 8
    assert result["executed_native_advance_calls"] == 32 and result["executed_interface_calls"] == 40
    assert result["new_reference_calls"] == result["new_oracle_calls"] == 0
    assert result["before_work"]["whole_selector"]["dual_reuse"]["child_rhs_entries"] == 32
    assert result["after_work"]["whole_selector"]["dual_reuse"]["child_rhs_entries"] == 0
    assert result["after_work"]["whole_selector"]["affine_dual"]["price_reads"] == 48
    assert result["executed_affine_dual_work"]["bound_rational_conversions"] == 16
    assert result["after_work"]["whole_selector"]["axis_capacity"]["merged_values_copied"] == 6
    assert result["after_work"]["whole_selector"]["residual_capacity_completion"]["prefix_reads"] == 6
    for name in summary.AFFINE_COUNTERS:
        assert result["executed_affine_dual_work"][name] == sum(result["after_work"][field]["affine_dual"][name]
                                                             for field in ("native", "whole_selector"))
    for name in DUAL_COUNTERS[:4]:
        assert result["before_work"]["whole_selector"]["dual_reuse"][name] == result["after_work"]["whole_selector"]["dual_reuse"][name]
    assert not result["failures"] and not result["behavior_mismatches"]


@pytest.mark.parametrize("field", ("canonical", "child_evaluations", "saved_certificates", "action", "upper", "lp", "integer"))
def test_normalization_keeps_every_semantic_and_search_counter(records, field):
    record = records[1][0]["states"][-1]["routed"]
    changed = deepcopy(record)
    if field == "canonical":
        changed["final_lp"]["selection"]["canonical_key_bound_evaluations"] += 1
    elif field == "child_evaluations":
        changed["final_lp"]["dual_reuse"]["child_bound_evaluations"] += 1
    elif field == "saved_certificates":
        changed["final_lp"]["dual_reuse"]["certificates_saved"] += 1
    elif field == "action":
        changed["selection"]["option_counts"] = [1]
    elif field == "upper":
        changed["final_lp"]["selection"]["upper_bound"] += 1
    elif field == "lp":
        changed["final_lp"]["selection"]["lp_calls"] += 1
    else:
        changed["probe"]["completion"]["prefix_reads"] += 1
    assert summary.normalized_result(changed) != summary.normalized_result(record)


@pytest.mark.parametrize("fee", ("certificate_numerator_products", "price_reads", "denominator_lcm_steps",
    "integer_scalings", "coefficient_requirement_reads", "coefficient_integer_operations",
    "box_coordinate_reads", "box_coordinate_comparisons", "delta_coefficient_reads",
    "delta_integer_operations", "bound_integer_products", "bound_rational_conversions"))
def test_one_omitted_actual_operation_invalidates_the_run(records, fee):
    stage = records[1][0]["states"][0]["routed"]["final_lp"]
    stage["affine_dual"][fee] -= 1
    result = summary.summarize(*records)
    assert result["decision"] == "INVALID_EXECUTION" and result["failures"]


def test_lazy_root_keeps_numerator_fee_without_preparing_unused_coefficients(records):
    state = records[1][0]["states"][0]
    stage = state["traces"]["new"]["1"]
    assert stage["affine_dual"]["certificates_prepared"] == 0
    assert stage["affine_dual"]["certificate_numerator_products"] == stage["dual_reuse"]["certificates_saved"] == 1
    assert not summary.audit_affine_stage(stage, 1)
    broken = deepcopy(stage)
    broken["affine_dual"]["certificates_prepared"] = 2
    assert summary.audit_affine_stage(broken, 1)


def test_cumulative_snapshot_cannot_forget_already_paid_preparation(records):
    state = records[1][0]["states"][0]
    assert not summary.audit_affine_stage(state["traces"]["new"]["1"], 1)
    errors = summary.audit_affine_stage(state["traces"]["new"]["1"], 1, state["traces"]["new"]["2"])
    assert "cumulative affine or dual-reuse work decreased" in errors


def test_each_of_six_fixed_cases_must_actually_use_affine_updates(records):
    state = records[1][0]["states"][0]
    for value in [*state["traces"]["new"].values(), state["routed"]["initial_lp"], state["routed"]["final_lp"],
                  *state["previous_dual_traces"].values(), state["previous_dual_routed"]["initial_lp"], state["previous_dual_routed"]["final_lp"]]:
        for name in DUAL_COUNTERS[1:]:
            value["dual_reuse"][name] = 0
    add_affine(state["traces"]["new"])
    add_affine(state["routed"])
    result = summary.summarize(*records)
    assert result["execution_checks_passed"] and result["behavior_preserved"]
    assert not result["affine_updates_exercised"] and result["decision"] == "AFFINE_DUAL_DEVELOPMENT_HOLD"


def test_incomplete_reference_takes_priority_over_development_advance(records):
    state = records[1][0]["states"][0]
    state["reference"].update(exact=False, score_exact=False, canonical_exact=False, frontier_regions_remaining=1)
    state.update(reference_complete=False, effective_reference_complete=False, effective_reference_source=None)
    state["reference_metrics"].update(score_exact_count=0, canonical_exact_count=0, reference_comparable_count=0)
    result = summary.summarize(*records)
    assert result["execution_checks_passed"] and result["decision"] == "INCOMPLETE_REFERENCE"


@pytest.mark.parametrize("mutation", ("new_reference", "missing_call", "group_count", "old_rhs_fee"))
def test_registered_execution_and_actual_fee_scope_cannot_change(records, mutation):
    protocol, rows = records
    if mutation == "new_reference":
        rows[0]["new_reference_calls"] = 1
    elif mutation == "missing_call":
        rows[0]["candidate_calls"] -= 1
    elif mutation == "group_count":
        protocol["expected_group_counts"]["historical_regression"] -= 1
    else:
        rows[0]["states"][0]["routed"]["final_lp"]["dual_reuse"]["child_rhs_entries"] = 1
    result = summary.summarize(protocol, rows)
    assert result["decision"] == "INVALID_EXECUTION" and not result["development_gate_passed"]


def test_full_original_roster_and_new_affine_source_pointers_are_required(records):
    records[1][0]["states"][0]["affine_source_index"] += 1
    with pytest.raises(ValueError, match="complete frozen timing roster"):
        summary.summarize(*records)
