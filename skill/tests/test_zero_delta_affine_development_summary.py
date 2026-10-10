"""Actual skip fees cannot hide changed LP, proof or canonical paths."""
from copy import deepcopy

import pytest

from algorithm.experiments import zero_delta_affine_development_summary as summary
from skill.tests.test_affine_dual_fresh_confirmation_summary import records as parent_records


def add_zero(stage):
    dual = stage["dual_reuse"]
    evaluations, flags = dual["child_bound_evaluations"], stage["selection"]["dual_column_checks"]
    assert not dual["child_bound_improvements"] and not dual["child_bound_prunes"]
    stage["zero_delta"] = dict.fromkeys(summary.ZERO_COUNTERS, 0)
    stage["zero_delta"].update(lower_flags_saved=flags, lower_flag_comparisons=flags,
        guard_calls=evaluations, zero_delta_skips=evaluations, variable_visits=evaluations,
        box_coordinate_reads=4 * evaluations, box_coordinate_comparisons=2 * evaluations,
        lower_flag_reads=evaluations)
    dual["child_bound_evaluations"] = 0
    stage["affine_dual"] = dict.fromkeys(summary.AFFINE_COUNTERS, 0)
    stage["affine_dual"]["certificate_numerator_products"] = dual["certificates_saved"]


@pytest.fixture
def records():
    protocol, rows = deepcopy(parent_records.__wrapped__())
    for metadata, state in zip(protocol["cells"][0]["states"], rows[0]["states"], strict=True):
        state["previous_dual_traces"] = deepcopy(state["traces"]["new"])
        state["previous_dual_routed"] = deepcopy(state["routed"])
        for stage in [*state["traces"]["new"].values(), state["routed"]["initial_lp"], state["routed"]["final_lp"]]:
            add_zero(stage)
    count = len(rows[0]["states"])
    protocol.update(experiment="pure_mock_zero_delta", expected_state_count=count,
        expected_selector_calls=count, expected_native_advance_calls=count * 4,
        expected_native_constructions=count * 2, expected_interface_calls=count * 5, focus_keys=[])
    rows[0].update(new_reference_calls=0, new_oracle_calls=0, candidate_calls=count,
        native_advance_calls=count * 4, native_search_constructions=count * 2)
    return protocol, rows


def test_complete_paths_and_paid_skips_advance(records):
    result = summary.summarize(*records)
    assert result["decision"] == "ZERO_DELTA_AFFINE_DEVELOPMENT_ADVANCE"
    assert all(result[key] for key in ("execution_checks_passed", "references_complete", "quality_gate_passed",
        "behavior_preserved", "work_gate_passed", "development_gate_passed", "prior_hold_retained"))
    assert result["state_count"] == 12 and result["executed_interface_calls"] == 60
    assert result["new_reference_calls"] == result["new_oracle_calls"] == 0
    assert result["executed"]["whole_selector"]["zero_delta"]["zero_delta_skips"] == 12
    assert result["borrowed"]["whole_selector"]["dual_reuse"]["child_bound_evaluations"] == 12
    assert not result["failures"] and not result["behavior_mismatches"]


@pytest.mark.parametrize("fee", ("lower_flags_saved", "lower_flag_comparisons", "guard_calls",
    "zero_delta_skips", "box_coordinate_reads", "box_coordinate_comparisons", "upper_price_zero_comparisons"))
def test_incorrect_guard_or_flag_fee_invalidates_execution(records, fee):
    stage = records[1][0]["states"][0]["routed"]["final_lp"]
    stage["zero_delta"][fee] += 1
    result = summary.summarize(*records)
    assert result["decision"] == "INVALID_EXECUTION" and result["failures"]


@pytest.mark.parametrize("kind", ("canonical", "child_improvements", "legacy_rhs", "action", "LP", "proof"))
def test_normalization_preserves_semantics_and_unchanged_work(records, kind):
    original = records[1][0]["states"][0]["routed"]
    changed = deepcopy(original)
    stage = changed["final_lp"]
    if kind == "canonical":
        stage["selection"]["canonical_key_bound_evaluations"] += 1
    elif kind == "child_improvements":
        stage["dual_reuse"]["child_bound_improvements"] += 1
    elif kind == "legacy_rhs":
        stage["dual_reuse"]["child_rhs_entries"] += 1
    elif kind == "action":
        changed["selection"]["option_indices"] = [1]
    elif kind == "LP":
        stage["selection"]["lp_calls"] += 1
    else:
        changed["status"] = "different"
    assert summary.normalized_result(changed) != summary.normalized_result(original)


def test_frozen_affine_audit_uses_its_real_native_ladder(records):
    protocol, rows = records
    state, model = rows[0]["states"][0], protocol["models"][0]
    from algorithm.experiments.score_face_completion_confirmation_summary import _reference
    expected, _ = _reference(state, protocol, len(model["options"]))
    _, errors, quality = summary._audit_zero_result(state["previous_dual_routed"], state,
        protocol, model, rows[0]["lanes"], "affine", expected)
    assert not errors and quality


def test_guard_coverage_must_match_frozen_actual_evaluations(records):
    state = records[1][0]["states"][0]
    frozen = deepcopy(state["previous_dual_traces"]["256"])
    frozen["dual_reuse"]["child_bound_evaluations"] += 1
    assert summary.audit_zero_stage(state["traces"]["new"]["256"], frozen=frozen)


def test_paid_guard_work_cannot_decrease(records):
    state = records[1][0]["states"][0]
    errors = summary.audit_zero_stage(state["traces"]["new"]["1"], state["traces"]["new"]["256"])
    assert "cumulative zero-delta fees decreased" in errors
