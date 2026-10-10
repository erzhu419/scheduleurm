from copy import deepcopy
from dataclasses import replace
import gzip
import json

import pytest

from algorithm.experiments import aligned_hardware_queue_aware_canonical_development as screen
from algorithm.experiments import aligned_hardware_integer_capacity_large_r as prior_screen
from algorithm.experiments import aligned_hardware_integer_capacity_development as runtime
from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption
from algorithm.experiments.integer_capacity_canonical_oracle import (
    canonical_at_certified_score, certified_score_from_selection,
)
from algorithm.queue_aware_canonical_integer_capacity_frame_candidates import (
    select_queue_aware_canonical_integer_capacity_frame,
)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    model = FrameModel(
        node="queue_bound_test", scenario_id="synthetic", lanes=2,
        workloads=("a", "b", "c"), scales=(1.0, 1.0, 1.0),
        options=(FrameOption("a_slow", (1, 0, 0), 3.0, ((3.0,), (), ())),
                 FrameOption("a_fast", (1, 0, 0), 1.0, ((1.0,), (), ())),
                 FrameOption("b", (0, 1, 0), 2.0, ((), (2.0,), ()))),
    )
    witness = {"name": "synthetic", "node": model.node, "lanes": 2,
               "queue": [3, 3, 0], "state_id": "state:0"}
    old_protocol = {
        "models": [model.snapshot()], "lanes": [2], "registered_at_utc": "old",
        "execution_sources": {"selector.py": "frozen source\n"},
        "search_counters": runtime.COUNTERS, "capacity_counters": runtime.CAPACITY_COUNTERS,
        "queue_panels": {"2": [{"panel": "tiny", "cohort": "fresh", "states": [
            {"state_id": "state:0", "queue": [3, 3, 0], "cohort": "fresh"}]}]},
        "unique_budgets": {"2": [144]}, "reference_budget": 4096,
        "execution_order": [[model.node, 2]],
    }
    (tmp_path / "selector.py").write_text("frozen source\n")
    prior_protocol = tmp_path / "prior_protocol.json"
    prior_protocol.write_text(json.dumps(old_protocol))
    prior_output = tmp_path / "prior.json"
    prior_output.write_text(json.dumps({"decision": "INCOMPLETE_REFERENCE", "certificate_checks_passed": True}))
    monkeypatch.setattr(screen, "ROOT", tmp_path)
    monkeypatch.setattr(screen, "PRIOR_PROTOCOL", prior_protocol)
    monkeypatch.setattr(screen, "PRIOR_OUTPUT", prior_output)
    monkeypatch.setattr(screen, "EXTRA_SOURCES", ())
    monkeypatch.setattr(screen, "ORACLE_WITNESSES", (witness,))
    monkeypatch.setattr(screen, "runtime_provenance", lambda: {})
    monkeypatch.setattr(prior_screen, "runtime_provenance", lambda: {})
    old_row = prior_screen.screen_cell(model, 2, old_protocol, runtime=runtime, progress=False)
    with gzip.open(prior_screen._cell_path(prior_output, model.node, 2), "wt") as target:
        json.dump({"registered_at_utc": "old", "cell": old_row}, target)
    protocol = screen.build_protocol((model,))
    protocol["registered_at_utc"] = "new"
    reference = old_row["states"][0]["reference"]
    score = certified_score_from_selection(model.options, witness["queue"], 2, reference)
    oracle = canonical_at_certified_score(model.options, witness["queue"], 2, score,
        state_visit_budget=protocol["oracle_state_budget"], maximum_duration=reference["duration"])
    return model, protocol, old_row, [{**witness, "result": oracle, "operational_elapsed_s": 0.0}]


def run(fixture, *, selector=None):
    model, protocol, previous, oracles = fixture
    return screen.screen_cell(model, 2, protocol, previous, oracles, runtime,
                              selector=selector, progress=False)


def test_only_new_selector_is_executed_on_reused_history(fixture, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("a historical selector was rerun")

    for name in ("select_canonical_count_space_bnb_frame",
                 "select_shared_threshold_canonical_requirement_dominance_frame",
                 "select_fractional_shared_threshold_canonical_requirement_dominance_frame",
                 "select_integer_capacity_shared_threshold_frame"):
        monkeypatch.setattr(runtime, name, forbidden)
    calls = []

    def observed(options, queue, lanes, budget):
        calls.append((queue, lanes, budget))
        return select_queue_aware_canonical_integer_capacity_frame(options, queue, lanes, budget)

    row = run(fixture, selector=observed)
    assert calls == [((3, 3, 0), 2, 144)]
    assert row["states"][0]["legacy"]["selection"] == fixture[2]["states"][0]["budgets"]["144"]["integer_capacity"]["selection"]
    assert row["states"][0]["queue_aware"]["metrics"]["calls"] == 1
    assert row["states"][0]["cohort"] == "reused_confirmation"


def incomplete_reference(fixture):
    model, protocol, previous, oracles = deepcopy(fixture)
    state = previous["states"][0]
    state["reference"].update(exact=False, canonical_exact=False)
    state["reference_complete"] = False
    state["reference_metrics"]["canonical_exact_count"] = 0
    return model, protocol, previous, oracles


def test_completed_independent_oracle_supplements_without_rewriting_history(fixture):
    model, protocol, previous, oracles = changed = incomplete_reference(fixture)
    row = run(changed)
    state = row["states"][0]
    assert not state["reference_complete"]
    assert not state["reference"]["canonical_exact"]
    assert state["effective_reference_complete"]
    assert state["effective_reference_source"] == "independent_score_face_oracle"
    assert state["queue_aware"]["metrics"]["reference_comparable_count"] == 1
    assert state["queue_aware"]["metrics"]["canonical_action_mismatch_count"] == 0
    assert protocol["prior_decision"] == "INCOMPLETE_REFERENCE"


def test_candidate_certificate_never_supplies_its_own_reference(fixture):
    model, protocol, previous, _ = incomplete_reference(fixture)
    row = run((model, protocol, previous, []))
    state = row["states"][0]
    assert state["queue_aware"]["selection"]["canonical_exact"]
    assert not state["effective_reference_complete"]
    assert state["effective_reference_source"] is None
    assert state["queue_aware"]["metrics"]["reference_comparable_count"] == 0


@pytest.mark.parametrize("complete,canonical", ((False, False), (True, False)))
def test_incomplete_or_infeasible_oracle_is_not_a_reference(complete, canonical):
    assert screen.oracle_key({"complete": complete, "canonical_exact": canonical}, 3) is None


def test_oracle_key_preserves_exact_fraction_and_idle_order(fixture):
    model, _, previous, oracles = fixture
    expected = runtime.reference_key(screen.selection_from_record(previous["states"][0]["reference"]),
                                     (3, 3, 0), len(model.options))
    assert screen.oracle_key(oracles[0]["result"], len(model.options)) == expected


def test_independent_oracle_runner_uses_external_score_record(fixture):
    model, protocol, previous, expected = fixture
    actual = screen.run_oracles((model,), protocol, [previous])
    assert actual[0]["result"]["canonical_exact"]
    assert actual[0]["result"]["option_indices"] == expected[0]["result"]["option_indices"]


def test_protocol_rejects_changed_frozen_source_and_budget(fixture):
    model, protocol, _, _ = fixture
    screen.verify_protocol(protocol, (model,))
    changed = deepcopy(protocol)
    changed["budget"] = 145
    with pytest.raises(ValueError, match="changed"):
        screen.verify_protocol(changed, (model,))
    (screen.ROOT / "selector.py").write_text("changed source\n")
    with pytest.raises(ValueError, match="predecessor source changed"):
        screen.build_protocol((model,))


def test_historical_cell_requires_same_roster_and_registration(fixture):
    model, protocol, _, _ = fixture
    restored = screen.load_prior_cell(protocol, model.node, 2)
    assert restored["states"][0]["queue"] == [3, 3, 0]
    changed = deepcopy(protocol)
    changed["prior_registered_at_utc"] = "other"
    with pytest.raises(ValueError, match="registration"):
        screen.load_prior_cell(changed, model.node, 2)
    changed = deepcopy(protocol)
    changed["queue_panels"]["2"][0]["states"][0]["queue"] = [4, 3, 0]
    with pytest.raises(ValueError, match="roster"):
        screen.load_prior_cell(changed, model.node, 2)


def summarize_fixture(fixture, row=None):
    from algorithm.experiments.queue_aware_canonical_development_summary import summarize
    model, protocol, _, oracles = fixture
    return summarize((model,), protocol, [run(fixture) if row is None else row], oracles)


def test_summary_separates_reused_calls_from_new_execution(fixture):
    result = summarize_fixture(fixture)
    assert result["queue_count"] == result["executed_selector_calls"] == 1
    assert result["reused_legacy_calls"] == result["reused_reference_calls"] == 1
    assert result["executed_oracle_calls"] == 1
    assert result["certificate_checks_passed"]
    assert result["original_references_complete"]
    assert result["effective_references_complete"]
    assert result["independent_oracles_complete"]
    assert result["development_action_gate_passed"]


def test_summary_accepts_independent_reference_and_keeps_old_incompleteness(fixture):
    result = summarize_fixture(incomplete_reference(fixture))
    assert not result["original_references_complete"]
    assert result["effective_references_complete"]
    assert result["independent_oracles_complete"]
    assert result["development_action_gate_passed"]
    assert result["certificate_checks_passed"]


def test_contradictory_independent_reference_invalidates_execution(fixture):
    changed = deepcopy(fixture)
    changed[3][0]["result"]["score_fraction"][0] += 1
    row = run(changed)
    assert row["states"][0]["oracle_crosscheck_failed"]
    result = summarize_fixture(changed, row)
    assert result["decision"] == "INVALID_EXECUTION"
    assert not result["certificate_checks_passed"]


@pytest.mark.parametrize("completed", (False, True))
def test_unfinished_or_impossible_oracle_blocks_development(fixture, completed):
    changed = incomplete_reference(fixture)
    result = changed[3][0]["result"]
    result.update(complete=completed, canonical_exact=False,
                  status="NO_ACTION_AT_CERTIFIED_SCORE" if completed else "STATE_BUDGET_EXHAUSTED")
    summary = summarize_fixture(changed)
    assert not summary["independent_oracles_complete"]
    assert not summary["development_action_gate_passed"]
    assert summary["decision"] == ("INVALID_EXECUTION" if completed else "INCOMPLETE_INDEPENDENT_REFERENCE")


def test_canonical_candidate_cannot_close_missing_reference_in_summary(fixture):
    model, protocol, previous, _ = incomplete_reference(fixture)
    protocol["oracle_witnesses"] = []
    result = summarize_fixture((model, protocol, previous, []))
    assert not result["effective_references_complete"]
    assert not result["development_action_gate_passed"]
    assert result["decision"] == "INCOMPLETE_INDEPENDENT_REFERENCE"


@pytest.mark.parametrize("mismatch", (False, True))
def test_honest_nonclosure_or_wrong_action_remains_action_hold(fixture, mismatch):
    def unresolved(options, queue, lanes, budget):
        result = select_queue_aware_canonical_integer_capacity_frame(options, queue, lanes, budget)
        selected = replace(result.selection, exact=False, canonical_exact=False)
        if mismatch:
            selected = replace(selected, option_counts=(0,) * len(options),
                option_indices=(None,) * lanes, service=(0,) * len(queue), duration=1.0,
                score=0.0, score_exact=False, gap=selected.upper_bound,
                epsilon=selected.upper_bound / sum(queue))
        return replace(result, selection=selected)

    row = run(fixture, selector=unresolved)
    result = summarize_fixture(fixture, row)
    assert result["certificate_checks_passed"]
    assert result["effective_references_complete"]
    assert not result["development_action_gate_passed"]
    assert result["decision"].endswith("ACTION_HOLD")


@pytest.mark.parametrize("mutation", ("missing_cell", "missing_state", "duplicate_state", "wrong_queue"))
def test_summary_rejects_incomplete_or_changed_coverage(fixture, mutation):
    from algorithm.experiments.queue_aware_canonical_development_summary import summarize
    model, protocol, _, oracles = fixture
    rows = [run(fixture)]
    if mutation == "missing_cell":
        rows.clear()
    elif mutation == "missing_state":
        rows[0]["states"].clear()
    elif mutation == "duplicate_state":
        rows[0]["states"].append(deepcopy(rows[0]["states"][0]))
    else:
        rows[0]["states"][0]["queue"] = [4, 3, 0]
    with pytest.raises(ValueError):
        summarize((model,), protocol, rows, oracles)


@pytest.mark.parametrize("mutation", ("budget", "call_count"))
def test_summary_rejects_changed_budget_or_duplicate_execution(fixture, mutation):
    row = run(fixture)
    if mutation == "budget":
        row["states"][0]["queue_aware"]["selection"]["lp_node_budget"] = 145
    else:
        row["states"][0]["queue_aware"]["metrics"]["calls"] = 2
    with pytest.raises(ValueError):
        summarize_fixture(fixture, row)
