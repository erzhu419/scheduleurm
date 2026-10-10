"""Fresh frontier closure and separately frozen regression costs on mocked records."""
from copy import deepcopy
from dataclasses import asdict

import pytest

from algorithm.experiments import frontier_score_proof_confirmation_summary as summary_module
from algorithm.experiments.frontier_score_proof_confirmation_summary import summarize
from algorithm.score_face_canonical_completion_frame_candidates import ScoreFaceCompletionWork
from skill.tests.test_frontier_score_proof_development_summary import records as development_records
from skill.tests.test_dual_reuse_development_summary import new_trace
from skill.tests.test_root_pruning_confirmation_summary import routed


@pytest.fixture
def records():
    protocol, rows = deepcopy(development_records.__wrapped__())
    for index, state in enumerate(rows[0]["states"]):
        source = ("fresh_confirmation", "historical_regression", "new_zero", "failure_regression")[index]
        state.update(source=source, cohort="zero_queue" if source == "new_zero" else source,
            panel=source, historical_regression=source == "historical_regression")
        state.pop("original_routed")
        state.pop("prior_routed")
        for key in ("borrowed_from", "borrowed_index"):
            state.pop(key)
    protocol["cells"][0]["states"] = [{key: deepcopy(value) for key, value in state.items() if key in (
        "state_id", "queue", "source", "cohort", "panel", "seed", "source_index", "source_state_id", "prior_index",
        "historical_regression", "caps", "source_record_source")} for state in rows[0]["states"]]
    protocol.update(expected_state_count=4, expected_fresh_state_count=1, expected_failure_regression_count=1,
        expected_selector_calls=8, expected_native_advance_calls=32, expected_interface_calls=40,
        expected_native_constructions=16, expected_new_reference_calls=2,
        failure_regression_keys=[{"node": "toy", "lanes": 2, "queue": [6]}])
    rows[0]["new_reference_calls"] = 2
    return protocol, rows


def replace_fresh_with_equal_lp(state):
    state["traces"]["new"] = new_trace(state, lp=2)
    state["routed"] = {**routed(state["traces"]["new"]["144"]), "frontier_evidence": None}


def test_fresh_run_executes_both_native_ladders_and_cold_frontier_routes(records):
    result = summarize(*records)
    assert result["decision"] == "FRONTIER_SCORE_PROOF_FRESH_CONFIRMATION_ADVANCE" and result["prior_hold_retained"]
    assert result["confirmation_gate_passed"] and not result["failures"]
    assert result["fresh_state_count"] == result["regression_state_count"] == result["failure_regression_count"] == result["zero_state_count"] == 1
    assert result["executed_interface_calls"] == 40 and result["native_search_constructions"] == 16
    assert result["executed_native_advance_calls"] == 32 and result["executed_selector_calls"] == 8
    assert result["new_reference_calls"] == result["reference_work"]["new"]["calls"] == 2
    assert result["borrowed_reference_count"] == result["reference_work"]["borrowed"]["calls"] == 2
    assert result["executed_search_work"]["lp_calls"] == 24
    assert result["executed_dual_reuse_work"]["certificates_saved"] == 8
    assert result["executed_frontier_export_work"]["new"]["fraction_pairs_copied"] == 3
    assert all(group["quality_gate_passed"] for group in result["groups"].values())
    assert all("original_routed" not in state and "prior_routed" not in state for state in records[1][0]["states"])


def test_borrowed_historical_record_without_seed_keeps_confirmation_gates(records):
    protocol, rows = records
    rows[0]["states"][1].pop("seed")
    protocol["cells"][0]["states"][1].pop("seed")
    result = summarize(protocol, rows)
    assert result["decision"] == "FRONTIER_SCORE_PROOF_FRESH_CONFIRMATION_ADVANCE"
    assert result["execution_checks_passed"] and result["confirmation_gate_passed"] and not result["failures"]
    assert result["groups"]["historical_regression"]["quality_gate_passed"] and result["historical_work_gate_passed"]


def test_regression_and_zero_gains_cannot_supply_fresh_strict_gain(records):
    protocol, rows = records
    replace_fresh_with_equal_lp(rows[0]["states"][0])
    result = summarize(protocol, rows)
    assert result["execution_checks_passed"] and result["routed_quality_gate_passed"]
    assert not result["fresh_work_gate_passed"] and result["decision"] == "FRONTIER_SCORE_PROOF_FRESH_CONFIRMATION_HOLD"
    assert result["groups"]["failure_regression"]["strict_pooled_lp_or_integer_reduction"]


def test_integer_reduction_alone_can_supply_fresh_strict_gain(records, monkeypatch):
    protocol, rows = records
    replace_fresh_with_equal_lp(rows[0]["states"][0])
    original = summary_module._work

    def mock_fresh_integer_work(states):
        work = original(states)
        fresh = sum(state["source"] == "fresh_confirmation" for state in states)
        work["routed_old"]["shared_integer_states"] += 10 * fresh
        work["routed_new"]["shared_integer_states"] += 5 * fresh
        return work

    monkeypatch.setattr(summary_module, "_work", mock_fresh_integer_work)
    result = summarize(protocol, rows)
    assert result["decision"] == "FRONTIER_SCORE_PROOF_FRESH_CONFIRMATION_ADVANCE"
    assert result["groups"]["fresh_confirmation"]["strict_pooled_lp_or_integer_reduction"]


@pytest.mark.parametrize("index,gate", [(1, "historical_work_gate_passed"), (3, "failure_regression_work_gate_passed")])
def test_each_fixed_regression_group_has_its_own_native_and_routed_cost_gate(records, index, gate):
    protocol, rows = records
    state = rows[0]["states"][index]
    for cap in (144, 256):
        state["traces"]["new"][str(cap)]["selection"]["lp_matrix_coefficients"] = 5
    result = state["routed"]
    for stage in (result["initial_lp"], result["final_lp"]):
        stage["selection"]["lp_matrix_coefficients"] = 5
    result["selection"]["lp_matrix_coefficients"] = 5
    if result["probe"] is not None:
        for field in ("selection", "baseline_selection", "score_certified_selection"):
            result["probe"][field]["lp_matrix_coefficients"] = 5
    result = summarize(protocol, rows)
    assert result["execution_checks_passed"] and result["fresh_work_gate_passed"] and result["routed_quality_gate_passed"]
    assert not result[gate] and result["decision"] == "FRONTIER_SCORE_PROOF_FRESH_CONFIRMATION_HOLD"


def test_valid_low_budget_service_loss_is_preserved(records):
    protocol, rows = records
    selected = rows[0]["states"][0]["traces"]["new"]["1"]["selection"]
    selected.update(option_counts=[0], option_indices=[None, None], service=[0], score=0., exact=False,
        score_exact=False, canonical_exact=False, upper_bound=8., gap=8., epsilon=2., frontier_regions_remaining=1)
    result = summarize(protocol, rows)
    assert result["execution_checks_passed"] and not result["finite_budget_gate_passed"]
    assert result["decision"] == "FRONTIER_SCORE_PROOF_FRESH_CONFIRMATION_HOLD"
    assert result["finite_budget_score_regressions"][0]["new_score"] == "0"


@pytest.mark.parametrize("mutation", ["native_cold_mismatch", "export_geometry", "shared_remainder", "missing_cap", "oracle_from_candidate", "failure_label"])
def test_actual_execution_and_reference_failures_are_invalid(records, mutation):
    protocol, rows = records
    state = rows[0]["states"][-1]
    if mutation == "native_cold_mismatch":
        state["routed"]["initial_lp"]["dual_reuse"]["certificates_saved"] += 1
    elif mutation == "export_geometry":
        state["routed"]["frontier_evidence"]["regions"][0]["bound_fraction"] = [12, 1]
    elif mutation == "shared_remainder":
        state["routed"]["probe"]["completion"]["state_budget"] = 8192
    elif mutation == "missing_cap":
        state["traces"]["new"].pop("2")
    elif mutation == "oracle_from_candidate":
        rows[0]["states"][0]["effective_reference_source"] = "candidate"
    else:
        state["cohort"] = "fresh_confirmation"
    result = summarize(protocol, rows)
    assert result["decision"] == "INVALID_EXECUTION" and result["failures"] and not result["confirmation_gate_passed"]


def test_known_failure_exhaustion_and_incomplete_reference_are_not_rescued(records):
    protocol, rows = records
    new = rows[0]["states"][-1]["routed"]
    probe = new["probe"]
    baseline = deepcopy(probe["baseline_selection"])
    probe["score_proof"].update(status="STATE_BUDGET_EXHAUSTED", complete=False, states_visited=8192)
    probe.update(selection=deepcopy(baseline), score_certified_selection=deepcopy(baseline),
        completion=asdict(ScoreFaceCompletionWork("SKIPPED_SCORE_UNPROVEN", False, False, 0)))
    new.update(selection=deepcopy(baseline), status="FRONTIER_PROBE_INCOMPLETE")
    result = summarize(protocol, rows)
    assert result["execution_checks_passed"] and not result["groups"]["failure_regression"]["quality_gate_passed"]
    assert result["decision"] == "FRONTIER_SCORE_PROOF_FRESH_CONFIRMATION_HOLD"
    state = rows[0]["states"][0]
    state["reference"].update(exact=False, score_exact=False, canonical_exact=False, frontier_regions_remaining=1)
    state.update(reference_complete=False, effective_reference_complete=False, effective_reference_source=None)
    state["reference_metrics"].update(score_exact_count=0, canonical_exact_count=0, reference_comparable_count=0)
    result = summarize(protocol, rows)
    assert result["decision"] == "INCOMPLETE_REFERENCE" and result["execution_checks_passed"] and not result["failures"]
