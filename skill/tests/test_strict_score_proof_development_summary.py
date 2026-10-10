"""Summary gates distinguish shared-budget failure, bad evidence and reuse."""
from copy import deepcopy
from dataclasses import replace

import pytest

import algorithm.strict_score_proof_canonical_completion_frame_candidates as candidate
from algorithm.experiments import aligned_hardware_integer_capacity_development as runtime
from algorithm.experiments import aligned_hardware_strict_score_proof_development as screen
from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption
from algorithm.experiments.strict_score_proof_development_summary import summarize


@pytest.fixture
def run_case(monkeypatch):
    monkeypatch.setattr(screen, "runtime_provenance", lambda: {})
    model = FrameModel("summary_test", "synthetic", 1, ("a", "b", "c"), (1.0, 1.0, 1.0),
                       (FrameOption("a", (1, 0, 0), 1.0, ((1.0,), (), ())),))
    original = candidate.select_integer_capacity_shared_threshold_frame

    def run(*, budget=3, baseline="unproved", reference_complete=True, reused=False):
        def stage(options, queue, lanes, lp_budget, idle=1.0):
            result = original(options, queue, lanes, lp_budget, idle)
            if baseline == "closed":
                return result
            changes = dict(exact=False, score_exact=False, canonical_exact=False,
                           upper_bound=result.selection.score + 1, gap=1.0, epsilon=1.0,
                           frontier_regions_remaining=1, inherited_unsolved_regions=1)
            if baseline == "inferior":
                changes.update(option_counts=(0,), option_indices=(None,), service=(0, 0, 0),
                               score=0.0, upper_bound=1.0)
            return replace(result, selection=replace(result.selection, **changes))
        monkeypatch.setattr(candidate, "select_integer_capacity_shared_threshold_frame", stage)
        protocol = {"registered_at_utc": "test", "experiment": "summary_test", "models": [model.snapshot()],
                    "lanes": [1], "budget": 144, "integer_state_budget": budget, "reference_budget": 4096,
                    "oracle_state_budget": 5000000, "prior_sources": {"current": {}}, "limitations": "synthetic",
                    "queue_panels": {"1": [{"panel": "test", "cohort": "new_neighborhood", "states": [
                        {"state_id": "new:0", "queue": [1, 0, 0], "source": "new_neighborhood", "source_state_id": None}]}]}}
        reference = runtime.select_canonical_count_space_bnb_frame
        if not reference_complete:
            monkeypatch.setattr(runtime, "select_canonical_count_space_bnb_frame", lambda *args:
                                replace(reference(*args), exact=False, score_exact=False, canonical_exact=False,
                                        upper_bound=2.0, gap=1.0, epsilon=1.0))
        row = screen.screen_cell(model, 1, protocol, {}, runtime, progress=False)
        if reused:
            old = row["states"][0]
            item = protocol["queue_panels"]["1"][0]
            item["cohort"] = "reused_confirmation"
            item["states"][0].update(state_id="current:new:0", source="current", source_state_id="new:0")
            row = screen.screen_cell(model, 1, protocol, {"current": {"new:0": old}}, runtime, progress=False)
        return protocol, row
    return run


def test_complete_proof_and_completion_advance_with_single_cold_candidate(run_case):
    protocol, row = run_case()
    result = summarize(protocol, [row])
    assert result["decision"] == "STRICT_SCORE_PROOF_DEVELOPMENT_ADVANCE"
    assert result["certificate_checks_passed"] and result["development_action_gate_passed"]
    assert result["executed_selector_calls"] == 2
    assert result["executed_reference_calls"] == result["executed_candidate_calls"] == 1
    assert result["reused_reference_calls"] == result["executed_oracle_calls"] == 0
    assert result["score_proof"]["completed_count"] == result["completion"]["completed_count"] == 1
    assert result["shared_integer_states"] == result["max_shared_integer_states"] == 3
    assert result["proof_trigger_points"][0]["final_canonical_exact"]


@pytest.mark.parametrize("budget,proof_complete", ((0, False), (1, True), (2, True)))
def test_honest_budget_nonclosure_holds_and_retains_intermediate_score(run_case, budget, proof_complete):
    protocol, row = run_case(budget=budget)
    result = summarize(protocol, [row])
    assert result["decision"] == "STRICT_SCORE_PROOF_DEVELOPMENT_HOLD"
    assert result["certificate_checks_passed"] and result["effective_references_complete"]
    assert not result["development_action_gate_passed"]
    assert result["score_proof"]["completed_count"] == int(proof_complete)
    assert row["states"][0]["completed"]["score_certified_selection"]["score_exact"] == proof_complete
    assert result["shared_integer_states"] == budget
    assert result["first_unresolved"]["queue"] == [1, 0, 0]


def test_strict_improvement_witness_remains_hold_not_false_optimization(run_case):
    protocol, row = run_case(baseline="inferior")
    result = summarize(protocol, [row])
    assert result["decision"] == "STRICT_SCORE_PROOF_DEVELOPMENT_HOLD"
    assert result["certificate_checks_passed"]
    assert result["score_proof"]["better_action_found_count"] == 1
    assert row["states"][0]["completed"]["selection"] == row["states"][0]["baseline"]["selection"]


def test_unproved_reference_is_incomplete_even_when_candidate_completes(run_case):
    protocol, row = run_case(reference_complete=False)
    result = summarize(protocol, [row])
    assert result["decision"] == "INCOMPLETE_REFERENCE"
    assert result["certificate_checks_passed"]
    assert not result["effective_references_complete"]
    assert row["states"][0]["completed"]["selection"]["canonical_exact"]


def test_retained_history_is_not_charged_as_an_executed_reference(run_case):
    protocol, row = run_case(baseline="closed", reused=True)
    result = summarize(protocol, [row])
    assert result["decision"] == "STRICT_SCORE_PROOF_DEVELOPMENT_ADVANCE"
    assert result["executed_selector_calls"] == result["executed_candidate_calls"] == 1
    assert result["reused_reference_calls"] == 1 and result["executed_reference_calls"] == 0
    assert result["baseline_reproduction_required_count"] == result["baseline_reproduced_count"] == 1
    assert result["completion_reproduction_required_count"] == result["completion_reproduced_count"] == 1
    assert result["executed_search_work"] == result["metrics"]["completed"]["search_work"]


@pytest.mark.parametrize("mutation", ("target", "intermediate", "lp_counter", "false_skip"))
def test_corrupted_proof_or_accounting_is_invalid_not_hold(run_case, mutation):
    protocol, row = run_case()
    state = row["states"][0]
    if mutation == "target":
        state["completed"]["score_proof"]["score_fraction"] = (2, 1)
    elif mutation == "intermediate":
        state["completed"]["score_certified_selection"]["frontier_regions_remaining"] = 0
    elif mutation == "lp_counter":
        state["completed"]["selection"]["lp_calls"] += 1
    else:
        state["completed"]["score_proof"]["attempted"] = False
    result = summarize(protocol, [row])
    assert result["decision"] == "INVALID_EXECUTION"
    assert not result["certificate_checks_passed"]
    assert result["proof_accounting_failures"]


def test_infeasible_witness_is_invalid(run_case):
    protocol, row = run_case(baseline="inferior")
    row["states"][0]["completed"]["score_proof"]["witness"]["option_counts"] = (2,)
    assert summarize(protocol, [row])["decision"] == "INVALID_EXECUTION"


def test_historical_completion_reproduction_is_recomputed(run_case):
    protocol, row = run_case(baseline="closed", reused=True)
    state = row["states"][0]
    assert state["completion_reproduced"]
    state["previous"]["completion"]["thresholds_checked"] += 1
    result = summarize(protocol, [row])
    assert result["decision"] == "INVALID_EXECUTION"
    assert "did not reproduce" in " ".join(result["proof_accounting_failures"][0]["errors"])


@pytest.mark.parametrize("mutation", ("cell", "state", "source", "reuse", "shared_cap"))
def test_registered_coverage_sources_and_caps_cannot_change(run_case, mutation):
    protocol, row = run_case()
    rows = [row]
    if mutation == "cell":
        rows.append(deepcopy(row))
    elif mutation == "state":
        row["states"].clear()
    elif mutation == "source":
        row["states"][0]["source"] = "current"
    elif mutation == "reuse":
        row["states"][0]["reference_reused"] = True
    else:
        row["states"][0]["completed"]["score_proof"]["states_visited"] += 1
    with pytest.raises(ValueError):
        summarize(protocol, rows)
