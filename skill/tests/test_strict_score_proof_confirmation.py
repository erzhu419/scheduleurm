from dataclasses import replace
from fractions import Fraction
from itertools import product
import json

import pytest

from algorithm.experiments import aligned_hardware_integer_capacity_development as runtime
from algorithm.experiments import aligned_hardware_strict_score_proof_confirmation as screen
from algorithm.experiments import integer_capacity_canonical_oracle as oracle
from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption
from algorithm.experiments.aligned_hardware_shared_threshold_scaling_screen import QUEUE_LEVELS, scaled_queues
from algorithm.strict_score_proof_canonical_completion_frame_candidates import (
    select_strict_score_proof_canonical_completion_frame,
)


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setattr(screen, "runtime_provenance", lambda: {})
    return FrameModel(
        node="strict_confirmation_test", scenario_id="synthetic", lanes=2,
        workloads=("a", "b", "c"), scales=(1.0, 1.0, 1.0),
        options=(FrameOption("a_slow", (1, 0, 0), 3.0, ((3.0,), (), ())),
                 FrameOption("a_fast", (1, 0, 0), 1.0, ((1.0,), (), ())),
                 FrameOption("b", (0, 1, 0), 2.0, ((), (2.0,), ()))),
    )


def tiny_protocol():
    return {
        "registered_at_utc": "synthetic", "budget": 144,
        "integer_state_budget": 100000, "reference_budget": 4096,
        "oracle_state_budget": 5000000,
        "queue_panels": {"2": [{"panel": "fresh_test", "cohort": "fresh_confirmation", "states": [
            {"state_id": "fresh:0", "queue": [3, 3, 0]},
        ]}]},
    }


def observed_run(model, monkeypatch, reference_status, *, oracle_budget=None, two_states=False):
    protocol = tiny_protocol()
    if oracle_budget is not None:
        protocol["oracle_state_budget"] = oracle_budget
    if two_states:
        protocol["queue_panels"]["2"][0]["states"].append(
            {"state_id": "fresh:1", "queue": [2, 2, 0]})
    calls = []
    original = runtime.select_canonical_count_space_bnb_frame

    def reference(options, queue, lanes, budget):
        calls.append(("reference", queue, lanes, budget))
        selected = original(options, queue, lanes, budget)
        if reference_status == "score_only":
            return replace(selected, exact=False, canonical_exact=False)
        if reference_status == "unproved":
            return replace(selected, exact=False, score_exact=False, canonical_exact=False,
                           upper_bound=selected.score + 1, gap=1.0,
                           epsilon=1.0 / sum(queue))
        return selected

    def independent(options, queue, lanes, proven_score, **kwargs):
        calls.append(("oracle", queue, lanes, proven_score, kwargs))
        return oracle.canonical_at_certified_score(options, queue, lanes, proven_score, **kwargs)

    def candidate(options, queue, lanes, budget, state_budget):
        calls.append(("candidate", queue, lanes, budget, state_budget))
        return select_strict_score_proof_canonical_completion_frame(
            options, queue, lanes, budget, state_budget)

    monkeypatch.setattr(runtime, "select_canonical_count_space_bnb_frame", reference)
    row = screen.screen_cell(model, 2, protocol, runtime, selector=candidate,
                             reference_oracle=independent, progress=False)
    return protocol, row, calls


def test_one_embedded_baseline_and_shared_budget_are_passed_once(model, monkeypatch):
    _, row, calls = observed_run(model, monkeypatch, "complete")
    state = row["states"][0]
    assert calls == [("reference", (3, 3, 0), 2, 4096),
                     ("candidate", (3, 3, 0), 2, 144, 100000)]
    assert state["reference_complete"]
    assert state["effective_reference_source"] == "full_K4096"
    assert state["oracle"] is None
    assert state["baseline_source"] == "embedded_stage_one"
    assert state["baseline"]["selection"] == state["completed"]["baseline_selection"]
    assert state["completed"]["integer_state_budget"] == 100000
    assert state["baseline"]["metrics"]["calls"] == 1
    assert state["score_certified_metrics"]["calls"] == 1
    assert state["completed"]["metrics"]["calls"] == 1
    assert state["completed"]["score_proof"]["states_visited"] + state["completed"]["completion"]["states_visited"] <= 100000
    assert "previous" not in state
    assert "baseline_reproduced" not in state


def test_reference_oracle_precedes_candidate_and_uses_only_reference_score(model, monkeypatch):
    protocol, row, calls = observed_run(model, monkeypatch, "score_only")
    state = row["states"][0]
    assert [call[0] for call in calls] == ["reference", "oracle", "candidate"]
    reference = state["reference"]
    expected = oracle.certified_score_from_selection(model.options, (3, 3, 0), 2, reference)
    assert calls[1][3] == expected
    assert isinstance(calls[1][3], Fraction)
    assert calls[1][4] == {"state_visit_budget": protocol["oracle_state_budget"],
                           "maximum_duration": reference["duration"]}
    assert not state["reference_complete"]
    assert not reference["canonical_exact"]
    assert state["effective_reference_complete"]
    assert state["effective_reference_source"] == "independent_score_face_oracle"
    assert state["completed"]["metrics"]["reference_comparable_count"] == 1


def test_candidate_proof_cannot_rescue_unproved_reference_and_run_continues(model, monkeypatch):
    _, row, calls = observed_run(model, monkeypatch, "unproved", two_states=True)
    assert [call[0] for call in calls] == ["reference", "candidate", "reference", "candidate"]
    assert [state["state_id"] for state in row["states"]] == ["fresh:0", "fresh:1"]
    for state in row["states"]:
        assert state["completed"]["selection"]["canonical_exact"]
        assert not state["reference_complete"]
        assert not state["effective_reference_complete"]
        assert state["effective_reference_source"] is None
        assert state["oracle"] is None
        assert state["completed"]["metrics"]["reference_comparable_count"] == 0


def test_oracle_exhaustion_stays_incomplete_despite_exact_candidate(model, monkeypatch):
    _, row, calls = observed_run(model, monkeypatch, "score_only", oracle_budget=0)
    state = row["states"][0]
    assert [call[0] for call in calls] == ["reference", "oracle", "candidate"]
    assert state["completed"]["selection"]["canonical_exact"]
    assert not state["effective_reference_complete"]
    assert state["effective_reference_source"] is None
    assert state["oracle"]["status"] == "STATE_BUDGET_EXHAUSTED"
    assert state["oracle"]["state_visit_budget"] == 0
    assert state["oracle"]["work"]["states_visited"] == 0
    assert state["completed"]["metrics"]["reference_comparable_count"] == 0
    assert state["completed"]["metrics"]["score_reference_comparable_count"] == 1


def test_fresh_roster_excludes_development_and_preserves_all_regression_centers():
    previous = json.loads(screen.PRIOR_PROTOCOL.read_text(encoding="utf-8"))
    excluded = screen.static_exclusions(previous)
    panels = screen.queue_design(previous["queue_panels"], excluded)
    assert panels == screen.queue_design(previous["queue_panels"], excluded)
    assert screen.PANEL_SEEDS == (2026092771, 2026092772, 2026092773, 2026092774)
    fresh_count = regression_count = 0
    for lanes in screen.LANES:
        fresh_panels = [panel for panel in panels[str(lanes)] if panel["cohort"] == "fresh_confirmation"]
        assert [panel["seed"] for panel in fresh_panels] == list(screen.PANEL_SEEDS)
        assert all(len(panel["states"]) == 64 for panel in fresh_panels)
        fresh = [state for panel in fresh_panels for state in panel["states"]]
        regression = [state for panel in panels[str(lanes)] if panel["cohort"] != "fresh_confirmation"
                      for state in panel["states"]]
        queues = {tuple(state["queue"]) for state in fresh}
        old_queues = {tuple(state["queue"]) for panel in previous["queue_panels"][str(lanes)]
                      for state in panel["states"]}
        assert old_queues <= excluded[str(lanes)]
        assert len(fresh) == len(queues) == 256
        assert queues.isdisjoint(excluded[str(lanes)])
        assert queues.isdisjoint(product(QUEUE_LEVELS, repeat=3))
        assert queues.isdisjoint(scaled_queues(lanes))
        regression_queues = {tuple(state["queue"]) for state in regression}
        assert queues.isdisjoint(regression_queues)
        assert len(regression_queues) == len(regression)
        assert len({state["state_id"] for state in fresh + regression}) == len(fresh + regression)
        expected_old = {tuple(state["queue"]) for panel in previous["queue_panels"][str(lanes)]
                        if panel["cohort"] == "old_regression" for state in panel["states"]}
        expected_extra = set()
        if lanes == 32:
            expected_extra = {tuple(q + d for q, d in zip((86, 87, 146), offsets))
                              for offsets in product((-1, 0, 1), repeat=3)}
        if lanes == 64:
            expected_extra = {(153, 147, 290), (81, 31, 48), (101, 9, 59),
                              (39, 3, 313), (145, 1, 67), (85, 34, 58)}
        assert regression_queues == expected_old | expected_extra
        fresh_count += len(fresh)
        regression_count += len(regression)
    assert fresh_count * 4 == 4096
    assert regression_count * 4 == 456


@pytest.fixture
def saved_protocol(model, monkeypatch, tmp_path):
    previous = json.loads(screen.PRIOR_PROTOCOL.read_text(encoding="utf-8"))
    previous.update(models=[model.snapshot()], execution_sources={"selector.py": "frozen source\n"})
    (tmp_path / "selector.py").write_text("frozen source\n")
    prior_protocol = tmp_path / "previous_protocol.json"
    prior_protocol.write_text(json.dumps(previous))
    prior_output = tmp_path / "previous.json"
    prior_output.write_text(json.dumps({
        "decision": "STRICT_SCORE_PROOF_DEVELOPMENT_ADVANCE", "certificate_checks_passed": True,
        "effective_references_complete": True, "development_action_gate_passed": True,
    }))
    frontier = tmp_path / "frontier.json"
    frontier.write_text(json.dumps({"queue_panels": previous["queue_panels"]}))
    for name, value in (("ROOT", tmp_path), ("PRIOR_PROTOCOL", prior_protocol),
                        ("PRIOR_OUTPUT", prior_output), ("FRONTIER_PROTOCOL", frontier),
                        ("STATIC_PROTOCOLS", ()), ("EXTRA_SOURCES", ())):
        monkeypatch.setattr(screen, name, value)
    monkeypatch.setattr(screen, "static_exclusions", lambda old: {
        str(lanes): {tuple(state["queue"]) for panel in old["queue_panels"][str(lanes)]
                     for state in panel["states"]} for lanes in screen.LANES})
    monkeypatch.setattr(screen, "runtime_setup", lambda: runtime)
    monkeypatch.setattr(runtime, "load_catalogues", lambda: (model,))
    output = tmp_path / "confirmation.json"
    monkeypatch.setattr("sys.argv", ["confirmation", "--output", str(output), "--register"])
    screen.main()
    return output, output.with_name(output.stem + "_protocol.json")


@pytest.mark.parametrize("change", ("source", "model", "queue", "lp_budget", "shared_budget", "oracle_budget", "reference_rule"))
def test_changed_frozen_protocol_stops_before_experiment_calls(model, saved_protocol, monkeypatch, tmp_path, change):
    output, path = saved_protocol
    protocol = json.loads(path.read_text())
    if change == "source":
        (tmp_path / "selector.py").write_text("changed source\n")
    elif change == "model":
        monkeypatch.setattr(runtime, "load_catalogues", lambda: (replace(model, lanes=3),))
    elif change == "queue":
        protocol["queue_panels"]["8"][0]["states"][0]["queue"][0] += 1
    elif change == "lp_budget":
        protocol["budget"] += 1
    elif change == "shared_budget":
        protocol["integer_state_budget"] += 1
    elif change == "oracle_budget":
        protocol["oracle_state_budget"] += 1
    else:
        protocol["reference_policy"] = "Candidate supplies its own reference"
    path.write_text(json.dumps(protocol))

    def forbidden(*args, **kwargs):
        raise AssertionError("changed protocol reached experiment calls")

    monkeypatch.setattr(screen, "screen_cell", forbidden)
    monkeypatch.setattr("sys.argv", ["confirmation", "--output", str(output), "--run"])
    with pytest.raises(ValueError):
        screen.main()
    assert not output.exists()
