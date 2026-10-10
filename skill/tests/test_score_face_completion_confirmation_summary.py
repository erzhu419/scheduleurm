from copy import deepcopy
from dataclasses import replace

import pytest

from algorithm.experiments import aligned_hardware_integer_capacity_development as runtime
from algorithm.experiments import aligned_hardware_score_face_completion_confirmation as screen
from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption
from algorithm.experiments.score_face_completion_confirmation_summary import summarize
import algorithm.score_face_canonical_completion_frame_candidates as completion


@pytest.fixture
def case(monkeypatch):
    model = FrameModel(
        node="confirmation_test", scenario_id="synthetic", lanes=2,
        workloads=("a", "b", "c"), scales=(1.0, 1.0, 1.0),
        options=(FrameOption("a_slow", (1, 0, 0), 3.0, ((3.0,), (), ())),
                 FrameOption("a_fast", (1, 0, 0), 1.0, ((1.0,), (), ())),
                 FrameOption("b", (0, 1, 0), 2.0, ((), (2.0,), ()))),
    )
    queue = (3, 3, 0)
    reference = runtime.select_canonical_count_space_bnb_frame(model.options, queue, 2, 4096)
    baseline = runtime.select_integer_capacity_shared_threshold_frame(model.options, queue, 2, 144)
    protocol = {
        "experiment": "synthetic_confirmation", "registered_at_utc": "frozen", "lanes": [2],
        "models": [model.snapshot()], "budget": 144, "reference_budget": 4096,
        "canonical_state_budget": 100000, "oracle_state_budget": 5000000,
        "limitations": "synthetic test", "queue_panels": {"2": [
            {"panel": "fresh", "cohort": "fresh_confirmation", "states": [
                {"state_id": "fresh:0", "queue": list(queue)}]}]},
    }
    monkeypatch.setattr(screen, "runtime_provenance", lambda: {})
    return model, protocol, baseline, reference


def rows_for(case, monkeypatch, *, open_baseline=False, missing_reference=False,
             unproved_reference=False, state_budget=None, oracle_budget=None):
    model, protocol, baseline, reference = deepcopy(case)
    if open_baseline:
        baseline = replace(baseline, selection=replace(baseline.selection, exact=False,
            canonical_exact=False, canonical_tie_regions_remaining=2,
            frontier_regions_remaining=2, inherited_unsolved_regions=1))
    if missing_reference or unproved_reference:
        reference = replace(reference, exact=False, canonical_exact=False,
                            score_exact=not unproved_reference)
    if state_budget is not None:
        protocol["canonical_state_budget"] = state_budget
    if oracle_budget is not None:
        protocol["oracle_state_budget"] = oracle_budget
    monkeypatch.setattr(completion, "select_integer_capacity_shared_threshold_frame", lambda *args: baseline)
    monkeypatch.setattr(runtime, "select_canonical_count_space_bnb_frame", lambda *args: reference)
    row = screen.screen_cell(model, 2, protocol, runtime, progress=False)
    return protocol, [row]


def test_complete_fresh_confirmation_counts_embedded_baseline_once(case, monkeypatch):
    result = summarize(*rows_for(case, monkeypatch))
    assert result["decision"] == "SCORE_FACE_COMPLETION_CONFIRMATION_ADVANCE"
    assert result["fresh_confirmation_gate_passed"]
    assert result["executed_selector_calls"] == 2
    assert result["executed_reference_calls"] == result["executed_candidate_calls"] == result["embedded_baseline_count"] == 1
    assert result["executed_oracle_calls"] == 0
    assert result["completion"]["skipped_canonical_complete_count"] == 1
    assert result["executed_search_work"]["lp_calls"] == (
        result["reference_metrics"]["search_work"]["lp_calls"] + result["metrics"]["completed"]["search_work"]["lp_calls"])


def test_completed_integer_work_is_separate_from_identical_lp_work(case, monkeypatch):
    result = summarize(*rows_for(case, monkeypatch, open_baseline=True))
    assert result["confirmation_action_gate_passed"]
    assert result["completion"]["attempted_count"] == result["completion"]["completed_count"] == 1
    assert result["completion"]["work"]["states_visited"] > 0
    assert result["metrics"]["baseline"]["search_work"] == result["metrics"]["completed"]["search_work"]


def test_honest_completion_exhaustion_is_hold_not_invalid(case, monkeypatch):
    result = summarize(*rows_for(case, monkeypatch, open_baseline=True, state_budget=0))
    assert result["decision"] == "SCORE_FACE_COMPLETION_CONFIRMATION_HOLD"
    assert result["certificate_checks_passed"]
    assert result["completion"]["exhausted_count"] == 1
    assert not result["fresh_confirmation_gate_passed"]


def test_reference_oracle_retains_original_nonclosure(case, monkeypatch):
    result = summarize(*rows_for(case, monkeypatch, missing_reference=True))
    assert result["confirmation_action_gate_passed"]
    assert not result["original_references_complete"]
    assert result["effective_references_complete"]
    assert result["executed_oracle_calls"] == result["reference_oracle"]["completed_count"] == 1


def test_exhausted_reference_oracle_stays_incomplete(case, monkeypatch):
    result = summarize(*rows_for(case, monkeypatch, missing_reference=True, oracle_budget=0))
    assert result["decision"] == "INCOMPLETE_REFERENCE"
    assert result["certificate_checks_passed"]
    assert not result["effective_references_complete"]


def test_candidate_score_proof_cannot_supply_reference(case, monkeypatch):
    args = rows_for(case, monkeypatch, unproved_reference=True)
    result = summarize(*args)
    assert result["decision"] == "INCOMPLETE_REFERENCE"
    assert result["executed_oracle_calls"] == 0
    assert result["metrics"]["completed"]["score_exact_count"] == 1
    args[1][0]["states"][0]["effective_reference_complete"] = True
    with pytest.raises(ValueError, match="effective reference"):
        summarize(*args)


def test_old_regression_cannot_be_presented_as_fresh_confirmation(case, monkeypatch):
    args = rows_for(case, monkeypatch)
    args[0]["queue_panels"]["2"][0]["cohort"] = "old_regression"
    args[1][0]["states"][0]["cohort"] = "old_regression"
    result = summarize(*args)
    assert result["decision"] == "SCORE_FACE_COMPLETION_CONFIRMATION_HOLD"
    assert result["fresh_confirmation"]["queue_count"] == 0
    assert result["regression"]["queue_count"] == 1
    assert result["regression_gate_passed"]


@pytest.mark.parametrize("mutation", ("missing_cell", "missing_state", "duplicate_state", "changed_queue", "changed_cohort", "registration"))
def test_coverage_changes_are_rejected(case, monkeypatch, mutation):
    args = rows_for(case, monkeypatch)
    row = args[1][0]
    if mutation == "missing_cell":
        args[1].clear()
    elif mutation == "missing_state":
        row["states"].clear()
    elif mutation == "duplicate_state":
        row["states"].append(deepcopy(row["states"][0]))
    elif mutation == "changed_queue":
        row["states"][0]["queue"] = [4, 3, 0]
    elif mutation == "changed_cohort":
        row["states"][0]["cohort"] = "old_regression"
    else:
        row["registered_at_utc"] = "changed"
    with pytest.raises(ValueError):
        summarize(*args)


@pytest.mark.parametrize("mutation", ("lp_budget", "state_budget", "oracle_budget", "reference_budget", "calls"))
def test_changed_caps_or_duplicated_calls_are_rejected(case, monkeypatch, mutation):
    args = rows_for(case, monkeypatch, missing_reference=mutation == "oracle_budget")
    state = args[1][0]["states"][0]
    if mutation == "lp_budget":
        state["completed"]["selection"]["lp_node_budget"] += 1
    elif mutation == "state_budget":
        state["completed"]["completion"]["state_budget"] += 1
    elif mutation == "oracle_budget":
        state["oracle"]["state_visit_budget"] += 1
    elif mutation == "reference_budget":
        state["reference"]["lp_node_budget"] += 1
    else:
        state["completed"]["metrics"]["calls"] = 2
    with pytest.raises(ValueError):
        summarize(*args)


@pytest.mark.parametrize("mutation", ("baseline", "counter_overrun", "trigger", "skip_work", "lp_work", "score_target", "live_frontier", "change_flag", "comparison_score", "comparison_action"))
def test_execution_or_accounting_errors_are_invalid(case, monkeypatch, mutation):
    args = rows_for(case, monkeypatch, open_baseline=mutation in ("score_target", "live_frontier"))
    state = args[1][0]["states"][0]
    work = state["completed"]["completion"]
    if mutation == "baseline":
        state["completed"]["baseline_selection"]["lp_calls"] += 1
    elif mutation == "counter_overrun":
        work["states_visited"] = work["state_budget"] + 1
    elif mutation == "trigger":
        work["attempted"] = not work["attempted"]
    elif mutation == "skip_work":
        work["count_branches"] = 1
    elif mutation == "lp_work":
        state["completed"]["selection"]["lp_calls"] += 1
    elif mutation == "score_target":
        work["score_fraction"] = [1, 1]
    elif mutation == "live_frontier":
        state["completed"]["selection"]["frontier_regions_remaining"] = 1
    elif mutation == "change_flag":
        work["action_changed"] = True
    elif mutation == "comparison_score":
        state["completed"]["metrics"]["score_mismatch_count"] = 1
    else:
        state["completed"]["metrics"]["canonical_action_mismatch_count"] = 1
    result = summarize(*args)
    assert result["decision"] == "INVALID_EXECUTION"
    assert result["completion_accounting_failures"]


@pytest.mark.parametrize("mutation", ("target", "missing", "contradiction"))
def test_reference_oracle_cannot_change_its_trigger_or_external_target(case, monkeypatch, mutation):
    args = rows_for(case, monkeypatch, missing_reference=True)
    state = args[1][0]["states"][0]
    if mutation == "target":
        state["oracle"]["score_fraction"][0] += 1
    elif mutation == "missing":
        state["oracle"] = None
        state["effective_reference_complete"] = False
        state["effective_reference_source"] = None
        for label in ("baseline", "completed"):
            state[label]["metrics"]["reference_comparable_count"] = 0
    else:
        state["oracle"].update(complete=True, canonical_exact=False, status="NO_ACTION_AT_CERTIFIED_SCORE")
        state["effective_reference_complete"] = False
        state["effective_reference_source"] = None
        for label in ("baseline", "completed"):
            state[label]["metrics"]["reference_comparable_count"] = 0
    assert summarize(*args)["decision"] == "INVALID_EXECUTION"
