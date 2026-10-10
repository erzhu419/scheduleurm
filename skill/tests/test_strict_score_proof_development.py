"""Independent small-action checks and frozen-development accounting checks."""
from copy import deepcopy
from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import combinations_with_replacement
import json
import random

import pytest

import algorithm.integer_capacity_shared_threshold_frame_candidates as baseline_module
import algorithm.strict_score_proof_canonical_completion_frame_candidates as strict
from algorithm.experiments import aligned_hardware_integer_capacity_development as runtime
from algorithm.experiments import aligned_hardware_strict_score_proof_development as screen
from algorithm.experiments import aligned_hardware_score_face_completion_confirmation as previous_screen
from algorithm.experiments import integer_capacity_canonical_oracle as oracle
from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption


@dataclass(frozen=True)
class Option:
    requirements: tuple[int, ...]
    duration: float


def exhaustive_actions(options, queue, lanes, idle=1.0):
    """Enumerate lane tuples independently of the production count DFS."""
    actions = []
    for indices in combinations_with_replacement(range(len(options) + 1), lanes):
        service = tuple(sum(options[i].requirements[j] for i in indices if i < len(options))
                        for j in range(len(queue)))
        if any(s > q for s, q in zip(service, queue, strict=True)):
            continue
        duration = max((Fraction.from_float(idle), *(Fraction.from_float(options[i].duration)
                                                     for i in indices if i < len(options))))
        score = Fraction(sum(q * s for q, s in zip(queue, service, strict=True))) / duration
        counts = tuple(indices.count(i) for i in range(len(options)))
        actions.append(((-score, duration, indices), counts, service))
    return sorted(actions)


def own_incumbent(monkeypatch, module, options, queue, lanes, action, idle=1.0):
    original = baseline_module.select_integer_capacity_shared_threshold_frame(options, queue, lanes, 4096, idle)
    key, counts, service = action
    score, duration, indices = -key[0], key[1], key[2]
    selected = replace(original.selection, option_counts=counts,
                       option_indices=tuple(None if i == len(options) else i for i in indices),
                       service=service, duration=float(duration), score=float(score),
                       exact=False, score_exact=False, canonical_exact=False,
                       upper_bound=float(score + 10), gap=10.0,
                       epsilon=10.0 / max(1, sum(queue)),
                       canonical_tie_regions_remaining=2, frontier_regions_remaining=2,
                       inherited_unsolved_regions=1)
    injected = replace(original, selection=selected)
    monkeypatch.setattr(module, "select_integer_capacity_shared_threshold_frame", lambda *args: injected)
    return selected


def selected_key(selected, queue, size):
    return (-Fraction(sum(q * s for q, s in zip(queue, selected.service, strict=True))) /
            Fraction.from_float(selected.duration), Fraction.from_float(selected.duration),
            tuple(size if i is None else i for i in selected.option_indices))


@pytest.mark.parametrize("case", range(16))
def test_strict_absence_proof_agrees_with_independent_lane_enumeration(monkeypatch, case):
    rng = random.Random(78330 + case)
    options = tuple(Option(required, rng.choice((0.5, 1.0, 1.25, 1.75, 2.5)))
                    for required in ((1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (1, 0)))
    queue = tuple(rng.randrange(5) for _ in range(2))
    lanes, idle = 2 + case % 2, (1.0, 1.5)[case % 2]
    actions = exhaustive_actions(options, queue, lanes, idle)
    incumbent = actions[0] if case % 3 else actions[-1]
    before = own_incumbent(monkeypatch, strict, options, queue, lanes, incumbent, idle)
    result = strict.select_strict_score_proof_canonical_completion_frame(
        options, queue, lanes, 4096, 10000, idle)
    assert result.baseline_selection == before
    assert result.score_proof.states_visited + result.completion.states_visited <= 10000
    assert result.selection.lp_calls == before.lp_calls
    if incumbent[0][0] == actions[0][0][0]:
        assert result.score_proof.status == "COMPLETE"
        assert result.score_proof.complete
        assert result.score_proof.witness is None
        assert result.selection.score_exact and result.selection.canonical_exact
        assert selected_key(result.selection, queue, len(options)) == actions[0][0]
    else:
        assert result.score_proof.status == "BETTER_ACTION_FOUND"
        assert not result.score_proof.complete
        assert result.selection == before
        assert result.completion.status == "SKIPPED_SCORE_UNPROVEN"
        witness = result.score_proof.witness
        matching = [action for action in actions if action[1] == tuple(witness["option_counts"])]
        assert len(matching) == 1
        assert Fraction(*witness["score_fraction"]) == -matching[0][0][0] > -incumbent[0][0]
        assert tuple(witness["service"]) == matching[0][2]


def test_improvement_at_slower_duration_may_overshoot_nondivisible_target(monkeypatch):
    # At T=2, the strict target is 7 but achievable weights are multiples of 3.
    # Weight 9 is a valid improvement; equality-style gcd pruning would miss it.
    options, queue = (Option((1,), 1.0), Option((3,), 2.0)), (3,)
    actions = exhaustive_actions(options, queue, 1)
    incumbent = next(action for action in actions if action[1] == (1, 0))
    before = own_incumbent(monkeypatch, strict, options, queue, 1, incumbent)
    result = strict.select_strict_score_proof_canonical_completion_frame(options, queue, 1, 4096, 100)
    assert result.score_proof.status == "BETTER_ACTION_FOUND"
    assert result.score_proof.thresholds_checked == 2
    assert result.score_proof.witness["duration"] == 2.0 > before.duration
    assert tuple(result.score_proof.witness["option_counts"]) == (0, 1)
    assert Fraction(*result.score_proof.witness["score_fraction"]) == Fraction(9, 2)
    assert result.selection == before


def test_equal_score_action_does_not_disprove_score_and_can_improve_tie(monkeypatch):
    options, queue = (Option((1,), 1.0), Option((2,), 2.0)), (4,)
    actions = exhaustive_actions(options, queue, 1)
    incumbent = next(action for action in actions if action[1] == (0, 1))
    before = own_incumbent(monkeypatch, strict, options, queue, 1, incumbent)
    result = strict.select_strict_score_proof_canonical_completion_frame(options, queue, 1, 4096, 100)
    assert result.score_proof.status == "COMPLETE"
    assert result.score_proof.witness is None
    assert result.score_certified_selection.option_counts == before.option_counts
    assert result.score_certified_selection.score_exact
    assert result.selection.option_counts == (1, 0)
    assert result.selection.canonical_exact
    assert result.completion.action_changed


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setattr(screen, "runtime_provenance", lambda: {})
    monkeypatch.setattr(previous_screen, "runtime_provenance", lambda: {})
    return FrameModel(
        node="strict_test", scenario_id="synthetic", lanes=2,
        workloads=("a", "b", "c"), scales=(1.0, 1.0, 1.0),
        options=(FrameOption("a_slow", (1, 0, 0), 3.0, ((3.0,), (), ())),
                 FrameOption("a_fast", (1, 0, 0), 1.0, ((1.0,), (), ())),
                 FrameOption("b", (0, 1, 0), 2.0, ((), (2.0,), ()))),
    )


def tiny_protocol(source="new_neighborhood"):
    return {"registered_at_utc": "synthetic", "budget": 144,
            "integer_state_budget": 100000, "reference_budget": 4096,
            "oracle_state_budget": 5000000,
            "queue_panels": {"2": [{"panel": "test", "cohort": source,
                "states": [{"state_id": "new:0", "queue": [3, 3, 0], "source": source,
                            "source_state_id": None if source == "new_neighborhood" else "old:0"}]}]}}


@pytest.mark.parametrize("reference_status", ("complete", "score_only", "unproved"))
def test_new_reference_is_independent_and_precedes_candidate(model, monkeypatch, reference_status):
    calls = []
    original = runtime.select_canonical_count_space_bnb_frame

    def reference(options, queue, lanes, budget):
        calls.append(("reference", budget))
        selected = original(options, queue, lanes, budget)
        if reference_status != "complete":
            selected = replace(selected, exact=False, canonical_exact=False,
                               score_exact=reference_status == "score_only")
        return selected

    def independent(options, queue, lanes, score, **kwargs):
        calls.append(("oracle", score, kwargs))
        return oracle.canonical_at_certified_score(options, queue, lanes, score, **kwargs)

    def candidate(options, queue, lanes, budget, state_budget):
        calls.append(("candidate", budget, state_budget))
        return strict.select_strict_score_proof_canonical_completion_frame(
            options, queue, lanes, budget, state_budget)

    monkeypatch.setattr(runtime, "select_canonical_count_space_bnb_frame", reference)
    row = screen.screen_cell(model, 2, tiny_protocol(), {}, runtime,
                             selector=candidate, reference_oracle=independent, progress=False)
    state = row["states"][0]
    expected_calls = ["reference", "oracle", "candidate"] if reference_status == "score_only" else ["reference", "candidate"]
    assert [call[0] for call in calls] == expected_calls
    assert calls[-1] == ("candidate", 144, 100000)
    assert not state["reference_reused"]
    assert state["previous"] is None
    assert state["baseline_reproduced"] is None
    assert state["completion_reproduced"] is None
    assert state["effective_reference_complete"] == (reference_status != "unproved")
    if reference_status == "score_only":
        reference_score = oracle.certified_score_from_selection(model.options, (3, 3, 0), 2, state["reference"])
        assert calls[1][1] == reference_score
        assert calls[1][2] == {"state_visit_budget": 5000000,
                                "maximum_duration": state["reference"]["duration"]}
    elif reference_status == "unproved":
        assert state["completed"]["selection"]["canonical_exact"]
        assert state["oracle"] is None
        assert state["completed"]["metrics"]["reference_comparable_count"] == 0


@pytest.mark.parametrize("source,open_reference", (("current", False), ("earlier", False), ("current", True)))
def test_existing_references_are_reused_and_old_completion_reproduces(model, monkeypatch, source, open_reference):
    old_protocol = {"registered_at_utc": "old", "budget": 144,
                    "canonical_state_budget": 100000, "reference_budget": 4096,
                    "oracle_state_budget": 5000000,
                    "queue_panels": {"2": [{"panel": "old", "cohort": "fresh_confirmation",
                        "states": [{"state_id": "old:0", "queue": [3, 3, 0]}]}]}}
    if open_reference:
        original = runtime.select_canonical_count_space_bnb_frame
        monkeypatch.setattr(runtime, "select_canonical_count_space_bnb_frame", lambda *args:
                            replace(original(*args), exact=False, canonical_exact=False))
    old = previous_screen.screen_cell(model, 2, old_protocol, runtime, progress=False)["states"][0]

    def unexpected(*args, **kwargs):
        pytest.fail("a reused state must not rerun its full reference or independent oracle")

    monkeypatch.setattr(runtime, "select_canonical_count_space_bnb_frame", unexpected)
    row = screen.screen_cell(model, 2, tiny_protocol(source), {source: {"old:0": old}}, runtime,
                             reference_oracle=unexpected, progress=False)
    state = row["states"][0]
    assert state["reference_reused"]
    assert state["reference"] == old["reference"]
    assert state["reference_metrics"] == old["reference_metrics"]
    assert state["oracle"] == old["oracle"]
    assert state["effective_reference_complete"]
    if open_reference:
        assert not state["reference_complete"]
        assert state["effective_reference_source"] == "independent_score_face_oracle"
    assert state["baseline_reproduced"] is True
    assert state["completion_reproduced"] is True


def test_development_union_keeps_known_failure_and_only_adds_declared_neighbors():
    current = json.loads(screen.PRIOR_PROTOCOL.read_text())
    earlier = json.loads(screen.EARLIER_PROTOCOL.read_text())
    panels = screen.queue_design(current, earlier)
    counts = {"current": 0, "earlier": 0, "new_neighborhood": 0}
    for lanes in current["lanes"]:
        states = [state for panel in panels[str(lanes)] for state in panel["states"]]
        queues = {tuple(state["queue"]) for state in states}
        assert len(queues) == len(states)
        old_queues = {tuple(state["queue"]) for previous in (current, earlier)
                      for panel in previous["queue_panels"][str(lanes)] for state in panel["states"]}
        assert old_queues <= queues
        added = [state for state in states if state["source"] == "new_neighborhood"]
        if lanes != 32:
            assert not added
        else:
            assert screen.WITNESS_QUEUE in old_queues
            assert len(added) == 26
            assert all(tuple(state["queue"]) not in old_queues for state in added)
            assert all(max(abs(q - center) for q, center in zip(state["queue"], screen.WITNESS_QUEUE)) == 1
                       for state in added)
        for state in states:
            counts[state["source"]] += 4
    assert counts == {"current": 4432, "earlier": 4084, "new_neighborhood": 104}
