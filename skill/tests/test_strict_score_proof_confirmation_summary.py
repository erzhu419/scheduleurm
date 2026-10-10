"""Confirmation gates retain budget failures and separate fresh evidence."""
from copy import deepcopy
from dataclasses import replace

import pytest

import algorithm.strict_score_proof_canonical_completion_frame_candidates as candidate
from algorithm.experiments import aligned_hardware_integer_capacity_development as runtime
from algorithm.experiments import aligned_hardware_strict_score_proof_confirmation as screen
from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption
from algorithm.experiments.strict_score_proof_confirmation_summary import summarize


@pytest.fixture
def run_case(monkeypatch):
    monkeypatch.setattr(screen, "runtime_provenance", lambda: {})
    model = FrameModel("summary_test", "synthetic", 1, ("a", "b", "c"), (1.0, 1.0, 1.0),
                       (FrameOption("a", (1, 0, 0), 1.0, ((1.0,), (), ())),))
    original = candidate.select_integer_capacity_shared_threshold_frame
    original_reference = runtime.select_canonical_count_space_bnb_frame

    def run(*, budget=3, baseline="unproved", reference="complete", with_regression=False):
        def stage(options, queue, lanes, lp_budget, idle=1.0):
            result = original(options, queue, lanes, lp_budget, idle)
            if baseline == "closed" or queue[0] == 2:
                return result
            changes = dict(exact=False, score_exact=False, canonical_exact=False,
                           upper_bound=result.selection.score + 1, gap=1.0, epsilon=1.0,
                           frontier_regions_remaining=1, inherited_unsolved_regions=1)
            if baseline == "inferior":
                changes.update(option_counts=(0,), option_indices=(None,), service=(0, 0, 0),
                               score=0.0, upper_bound=1.0)
            return replace(result, selection=replace(result.selection, **changes))

        def full_reference(*args):
            selected = original_reference(*args)
            if reference == "complete":
                return selected
            if reference == "score_only":
                return replace(selected, exact=False, canonical_exact=False)
            return replace(selected, exact=False, score_exact=False, canonical_exact=False,
                           upper_bound=selected.score + 1, gap=1.0, epsilon=1.0)

        monkeypatch.setattr(candidate, "select_integer_capacity_shared_threshold_frame", stage)
        monkeypatch.setattr(runtime, "select_canonical_count_space_bnb_frame", full_reference)
        panels = [{"panel": "fresh", "cohort": "fresh_confirmation",
                   "states": [{"state_id": "fresh:0", "queue": [1, 0, 0]}]}]
        if with_regression:
            panels.append({"panel": "old", "cohort": "old_regression",
                           "states": [{"state_id": "old:0", "queue": [2, 0, 0]}]})
        protocol = {"registered_at_utc": "test", "experiment": "summary_test", "models": [model.snapshot()],
                    "lanes": [1], "budget": 144, "integer_state_budget": budget, "reference_budget": 4096,
                    "oracle_state_budget": 5000000, "limitations": "synthetic",
                    "queue_panels": {"1": panels}}
        row = screen.screen_cell(model, 1, protocol, runtime, progress=False)
        return protocol, row
    return run


def test_confirmation_advances_with_two_cold_calls_and_embedded_views(run_case):
    protocol, row = run_case(with_regression=True)
    result = summarize(protocol, [row])
    assert result["decision"] == "STRICT_SCORE_PROOF_CONFIRMATION_ADVANCE"
    assert result["certificate_checks_passed"] and result["confirmation_action_gate_passed"]
    assert result["executed_selector_calls"] == 4
    assert result["executed_reference_calls"] == result["executed_candidate_calls"] == 2
    assert result["embedded_baseline_count"] == result["intermediate_certificate_count"] == 2
    assert result["executed_oracle_calls"] == 0
    assert result["fresh_confirmation"]["queue_count"] == result["regression"]["queue_count"] == 1
    assert result["score_proof"]["completed_count"] == result["completion"]["completed_count"] == 1
    assert result["shared_integer_states"] == result["max_shared_integer_states"] == 3
    assert result["fresh_confirmation"]["trigger_points"][0]["final_canonical_exact"]
    assert not result["regression"]["trigger_points"]
    assert not any("reused" in key or "reproduction" in key for key in result)
    assert result["executed_search_work"] == {
        key: result["reference_metrics"]["search_work"][key] + result["metrics"]["completed"]["search_work"][key]
        for key in runtime.COUNTERS}


@pytest.mark.parametrize("budget,proof_complete", ((0, False), (1, True), (2, True)))
def test_honest_proof_or_completion_exhaustion_holds(run_case, budget, proof_complete):
    protocol, row = run_case(budget=budget, with_regression=True)
    result = summarize(protocol, [row])
    assert result["decision"] == "STRICT_SCORE_PROOF_CONFIRMATION_HOLD"
    assert result["certificate_checks_passed"] and result["effective_references_complete"]
    assert result["score_proof"]["completed_count"] == int(proof_complete)
    assert result["shared_integer_states"] == budget
    assert not result["fresh_confirmation_gate_passed"]
    assert result["regression_gate_passed"]
    assert result["first_unresolved"]["queue"] == [1, 0, 0]


def test_honest_better_action_witness_is_hold(run_case):
    protocol, row = run_case(baseline="inferior")
    result = summarize(protocol, [row])
    assert result["decision"] == "STRICT_SCORE_PROOF_CONFIRMATION_HOLD"
    assert result["certificate_checks_passed"]
    assert result["score_proof"]["better_action_found_count"] == 1
    assert result["metrics"]["completed"]["score_mismatch_count"] == 1
    assert row["states"][0]["completed"]["selection"] == row["states"][0]["baseline"]["selection"]


def test_unproved_reference_retains_incomplete_status(run_case):
    protocol, row = run_case(reference="unproved")
    result = summarize(protocol, [row])
    assert result["decision"] == "INCOMPLETE_REFERENCE"
    assert result["certificate_checks_passed"]
    assert not result["effective_references_complete"]
    assert row["states"][0]["completed"]["selection"]["canonical_exact"]


def test_reference_oracle_closes_only_its_own_score_certified_reference(run_case):
    protocol, row = run_case(reference="score_only")
    result = summarize(protocol, [row])
    assert result["decision"] == "STRICT_SCORE_PROOF_CONFIRMATION_ADVANCE"
    assert not result["original_references_complete"]
    assert result["effective_references_complete"]
    assert result["executed_oracle_calls"] == 1
    assert result["reference_oracle"]["completed_count"] == 1
    assert result["reference_metrics"]["canonical_exact_count"] == 0


@pytest.mark.parametrize("mutation", ("target", "intermediate", "lp_counter", "false_skip", "baseline_source", "reference_lp"))
def test_corrupted_proof_or_accounting_is_invalid(run_case, mutation):
    protocol, row = run_case()
    state = row["states"][0]
    if mutation == "target":
        state["completed"]["score_proof"]["score_fraction"] = (2, 1)
    elif mutation == "intermediate":
        state["completed"]["score_certified_selection"]["frontier_regions_remaining"] = 0
    elif mutation == "lp_counter":
        state["completed"]["selection"]["lp_calls"] += 1
    elif mutation == "false_skip":
        state["completed"]["score_proof"]["attempted"] = False
    elif mutation == "reference_lp":
        state["reference_metrics"]["search_work"]["lp_calls"] += 1
    else:
        state["baseline_source"] = "independent_call"
    result = summarize(protocol, [row])
    assert result["decision"] == "INVALID_EXECUTION"
    assert not result["certificate_checks_passed"]
    assert result["proof_accounting_failures"]


def test_false_mismatch_metrics_are_invalid_instead_of_scientific_hold(run_case):
    protocol, row = run_case(baseline="inferior")
    row["states"][0]["completed"]["metrics"]["score_mismatch_count"] = 0
    assert summarize(protocol, [row])["decision"] == "INVALID_EXECUTION"


def test_reference_oracle_cannot_be_dropped_after_trigger(run_case):
    protocol, row = run_case(reference="score_only")
    state = row["states"][0]
    state["oracle"] = None
    state["effective_reference_complete"] = False
    state["effective_reference_source"] = None
    for metrics in (state["baseline"]["metrics"], state["score_certified_metrics"], state["completed"]["metrics"]):
        metrics["reference_comparable_count"] = 0
    result = summarize(protocol, [row])
    assert result["decision"] == "INVALID_EXECUTION"


@pytest.mark.parametrize("mutation", ("cell", "state", "queue", "registration", "cohort", "shared_cap"))
def test_registered_coverage_and_shared_budget_cannot_change(run_case, mutation):
    protocol, row = run_case()
    rows = [row]
    if mutation == "cell":
        rows.append(deepcopy(row))
    elif mutation == "state":
        row["states"].clear()
    elif mutation == "queue":
        row["states"][0]["queue"] = [2, 0, 0]
    elif mutation == "registration":
        row["registered_at_utc"] = "different"
    elif mutation == "cohort":
        row["states"][0]["cohort"] = "old_regression"
    else:
        row["states"][0]["completed"]["score_proof"]["states_visited"] += 1
    with pytest.raises(ValueError):
        summarize(protocol, rows)


def test_regression_success_alone_is_not_fresh_confirmation(run_case):
    protocol, row = run_case(baseline="closed")
    protocol["queue_panels"]["1"][0]["cohort"] = "old_regression"
    row["states"][0]["cohort"] = "old_regression"
    result = summarize(protocol, [row])
    assert result["decision"] == "STRICT_SCORE_PROOF_CONFIRMATION_HOLD"
    assert result["regression_gate_passed"]
    assert not result["fresh_confirmation_gate_passed"]
