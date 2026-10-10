"""Independent references and frozen sampling for dual-reuse confirmation."""
from dataclasses import asdict, dataclass, fields
from fractions import Fraction
import json
from types import SimpleNamespace

import pytest

from algorithm.canonical_count_space_bnb_frame_candidates import CanonicalCountSpaceBnbSelection
from algorithm.experiments import aligned_hardware_dual_reuse_confirmation as runner
from algorithm.experiments.aligned_hardware_shared_threshold_queue_confirmation import sample_queues


def test_new_panels_keep_the_sampler_law_and_exclude_history_and_prior_draws():
    excluded = {str(lanes): {(lanes + 1, lanes + 2, lanes + 3)} for lanes in (8, 16, 32, 64)}
    before = {key: values.copy() for key, values in excluded.items()}
    panels = runner.fresh_design(excluded)
    for lanes in (8, 16, 32, 64):
        seen = excluded[str(lanes)].copy()
        assert len(panels[str(lanes)]) == 4
        for panel, seed in zip(panels[str(lanes)], (2026092831, 2026092832, 2026092833, 2026092834)):
            queues = tuple(tuple(state["queue"]) for state in panel["states"])
            assert panel["seed"] == seed and panel["panel"] == f"fresh_{seed}"
            assert queues == sample_queues(lanes, seed, 64, excluded=seen)
            assert len(set(queues)) == 64 and set(queues).isdisjoint(seen)
            seen.update(queues)
    assert excluded == before


@dataclass(frozen=True)
class Option:
    requirements: tuple[int, ...]
    duration: float


@dataclass(frozen=True)
class MethodResult:
    label: str
    queue: tuple[int, ...]
    budget: int
    canonical_exact: bool
    score: float = 999.0


def full_reference(queue, *, score_exact=True):
    zero = not any(queue)
    values = {field.name: 0 for field in fields(CanonicalCountSpaceBnbSelection)}
    values.update(option_counts=(0, 0) if zero else (1, 0),
        option_indices=(None, None) if zero else (0, None), service=queue,
        duration=1.0, score=0.0 if zero else 1.0,
        upper_bound=0.0 if zero else 1.0 if score_exact else 2.0,
        gap=0.0 if zero or score_exact else 1.0, epsilon=0.0 if zero or score_exact else 1.0,
        exact=zero, score_exact=zero or score_exact, canonical_exact=zero, lp_node_budget=4096,
        frontier_regions_remaining=0 if zero else 1,
        canonical_tie_regions_remaining=int(not zero and score_exact))
    return CanonicalCountSpaceBnbSelection(**values)


def call_harness(monkeypatch, *, incomplete=False):
    model = SimpleNamespace(node="toy", lanes=1,
        options=(Option((1, 0, 0), 1.0), Option((0, 1, 0), 2.0)))
    states = [
        {"state_id": "fresh", "queue": [1, 0, 0], "source": "fresh_confirmation", "panel": "fresh", "cohort": "fresh_confirmation"},
        {"state_id": "historical", "source_state_id": "old historical", "source_index": 0,
         "queue": [0, 1, 0], "source": "historical_regression", "panel": "historical", "cohort": "historical_regression"},
        {"state_id": "zero", "queue": [0, 0, 0], "source": "new_zero", "panel": "zero", "cohort": "zero_queue"}]
    for state in states:
        state["caps"] = [2, 3, 144, 256]
        state["historical_regression"] = state["source"] == "historical_regression"
    borrowed = {"state_id": "old historical", "queue": [0, 1, 0],
        "source": "historical_regression",
        "reference": {"label": "borrowed full reference"}, "reference_metrics": {"borrowed": True},
        "reference_complete": False, "oracle": {"label": "borrowed oracle"},
        "effective_reference_complete": True, "effective_reference_source": "borrowed source",
        "routed": {"label": "stale routed output"}}
    protocol = {"registered_at_utc": "new", "initial_lp_budget": 144, "lp_node_budget": 256,
        "integer_state_budget": 8192, "reference_budget": 4096, "oracle_state_budget": 5000000,
        "reference_source": {"path": "source.json", "registered_at_utc": "old"}}
    events, searches, reference_calls, oracle_calls = [], {"old": [], "new": []}, [], []

    def search_class(label):
        class Search:
            def __init__(self, options, queue, lanes):
                self.queue, self.caps = queue, []
                searches[label].append(self)
                events.append((label + " search", queue))

            def advance_to(self, cap):
                self.caps.append(cap)
                events.append((label + " advance", self.queue))
                return MethodResult(label + " trace", self.queue, cap, not incomplete)
        return Search

    def selector(label):
        def run(options, queue, lanes, initial, total, integer):
            assert options is model.options and (lanes, initial, total, integer) == (2, 144, 256, 8192)
            events.append((label + " routed", queue))
            return MethodResult(label + " routed", queue, total, not incomplete)
        return run

    def original_catalogue_reference(options, queue, lanes, budget):
        assert options is model.options and (lanes, budget) == (2, 4096)
        events.append(("reference", queue))
        reference_calls.append(queue)
        return full_reference(queue, score_exact=not incomplete)

    def oracle(options, queue, lanes, proven_score, *, state_visit_budget, maximum_duration):
        assert options is model.options and queue == (1, 0, 0) and lanes == 2
        assert (proven_score, state_visit_budget, maximum_duration) == (Fraction(1), 5000000, 1.0)
        assert not any(event[1] == queue and event[0].split()[0] in ("old", "new") for event in events)
        events.append(("oracle", queue))
        oracle_calls.append(proven_score)
        return {"complete": True, "canonical_exact": True, "score_fraction": [1, 1],
                "duration_fraction": [1, 1], "option_indices": [0, None]}

    def reference_key(reference, queue, count):
        assert count == 2
        return ("full",) if reference.canonical_exact else None

    def record(metrics, *args, **kwargs):
        metrics["calls"] += 1

    from algorithm.experiments.aligned_hardware_strict_score_proof_development import _new_reference
    runtime = SimpleNamespace(select_canonical_count_space_bnb_frame=original_catalogue_reference,
        reference_key=reference_key, _metrics=lambda: {"calls": 0}, _record=record)
    dependencies = {"searches": {label: search_class(label) for label in ("old", "new")},
        "selectors": {label: selector(label) for label in ("old", "new")},
        "new_reference": _new_reference, "oracle": oracle,
        "oracle_key": lambda result, count: ("oracle",) if result["complete"] else None}

    def load(path, node, lanes, registration):
        assert (path, node, lanes, registration) == (runner.ROOT / "source.json", "toy", 2, "old")
        return {"states": [borrowed]}

    monkeypatch.setattr(runner, "load_cell", load)
    monkeypatch.setattr(runner, "runtime_provenance", lambda: {"cpu_affinity": [0]})
    return model, {"node": "toy", "lanes": 2, "states": states}, protocol, runtime, dependencies, events, searches, reference_calls, oracle_calls, borrowed


def test_full_references_and_their_oracle_precede_both_continuous_and_cold_arms(monkeypatch):
    model, cell, protocol, runtime, dependencies, events, searches, references, oracles, borrowed = call_harness(monkeypatch)
    row = runner.run_cell(model, cell, protocol, runtime, dependencies, progress=False)
    assert references == [(1, 0, 0), (0, 0, 0)] and oracles == [Fraction(1)]
    assert row["new_reference_calls"] == 2 and row["new_reference_oracle_calls"] == 1
    assert row["new_old_routed_calls"] == 3
    for label in ("old", "new"):
        assert len(searches[label]) == 3 and all(search.caps == [2, 3, 144, 256] for search in searches[label])
    for state in row["states"]:
        queue = tuple(state["queue"])
        selected = [event[0] for event in events if event[1] == queue]
        if state["source"] != "historical_regression":
            assert selected[0] == "reference"
        assert selected.count("old routed") == selected.count("new routed") == 1
        assert selected.index("old routed") < selected.index("new routed")
        assert state["caps"] == [2, 3, 144, 256]
        assert list(state["traces"]["old"]) == list(state["traces"]["new"]) == ["2", "3", "144", "256"]
        assert state["old_routed"]["label"] == "old routed" and state["routed"]["label"] == "new routed"
    for key in ("reference", "reference_metrics", "reference_complete", "oracle",
                "effective_reference_complete", "effective_reference_source"):
        assert row["states"][1][key] == borrowed[key]
    assert row["states"][0]["reference"]["score"] == 1.0
    assert row["states"][0]["routed"]["score"] == 999.0
    assert row["states"][0]["effective_reference_source"] == "independent_score_face_oracle"
    assert row["states"][2]["effective_reference_source"] == "full_K4096"


def test_unproved_reference_and_method_nonclosure_are_retained_without_replacement(monkeypatch):
    model, cell, protocol, runtime, dependencies, _, searches, references, oracles, _ = call_harness(monkeypatch, incomplete=True)
    row = runner.run_cell(model, cell, protocol, runtime, dependencies, progress=False)
    assert len(row["states"]) == 3 and references == [(1, 0, 0), (0, 0, 0)] and not oracles
    first = row["states"][0]
    assert first["reference_complete"] is False and first["effective_reference_complete"] is False
    assert first["oracle"] is None and first["reference"]["score_exact"] is False
    assert all(state["routed"]["canonical_exact"] is False for state in row["states"])
    assert row["new_old_routed_calls"] == 3 and len(searches["old"]) == len(searches["new"]) == 3


def test_protocol_excludes_prior_panels_development_and_other_registered_work(monkeypatch, tmp_path):
    from skill.tests.test_branch_reserved_confirmation import protocol_fixture
    from algorithm.experiments import aligned_hardware_branch_reserved_confirmation as branch_confirmation
    models, _, branch_development = protocol_fixture(monkeypatch, tmp_path)
    previous = branch_confirmation.build_protocol(models)
    previous["registered_at_utc"] = "branch confirmation"
    development = {**branch_development, "reference_budget": 4096, "oracle_state_budget": 5000000,
        "rule": "Only reuse an already certified parent dual; reserve every root position."}
    extra = {key: previous[key] for key in ("models", "execution_sources", "initial_lp_budget", "lp_node_budget", "integer_state_budget")}
    extra["cells"] = [{"node": model.node, "lanes": lanes, "states": [{"queue": [lanes, 9, 10]}]}
                      for model in models for lanes in (8, 16, 32, 64)]
    prior_output, previous_output, extra_output = tmp_path / "dual_development.json", tmp_path / "branch_confirmation.json", tmp_path / "extra.json"
    prior_output.write_text(json.dumps({"decision": "DUAL_REUSE_DEVELOPMENT_ADVANCE", "development_gate_passed": True}))
    previous_output.write_text(json.dumps({"decision": "BRANCH_RESERVED_FRESH_CONFIRMATION_ADVANCE",
        "confirmation_gate_passed": True, "execution_checks_passed": True, "effective_references_complete": True}))
    for path, protocol in ((prior_output, development), (previous_output, previous), (extra_output, extra)):
        path.with_name(path.stem + "_protocol.json").write_text(json.dumps(protocol))
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "PRIOR_OUTPUT", prior_output)
    monkeypatch.setattr(runner, "PREVIOUS_OUTPUT", previous_output)
    monkeypatch.setattr(runner, "EXCLUSION_OUTPUTS", (extra_output,))
    monkeypatch.setattr(runner, "SOURCES", ())
    protocol = runner.build_protocol(models)
    assert len(protocol["cells"]) == 16
    states = [state for cell in protocol["cells"] for state in cell["states"]]
    assert len(states) == 4146
    assert sum(state["source"] == "fresh_confirmation" for state in states) == 4096
    assert sum(state["source"] == "historical_regression" for state in states) == 34
    assert sum(state["source"] == "new_zero" for state in states) == 16
    assert protocol["expected_new_reference_calls"] == 4112
    assert protocol["expected_interface_calls"] == 41460
    assert protocol["expected_native_advance_calls"] == 33168
    for lanes in (8, 16, 32, 64):
        excluded = set(map(tuple, protocol["excluded_static_queues"][str(lanes)]))
        assert (lanes, 9, 10) in excluded
        assert {tuple(state["queue"]) for old in (previous, development, extra) for cell in old["cells"]
                if cell["lanes"] == lanes for state in cell["states"]} <= excluded
        fresh = {tuple(state["queue"]) for panel in protocol["queue_panels"][str(lanes)] for state in panel["states"]}
        assert len(fresh) == 256 and fresh.isdisjoint(excluded)
    old_cells = {(cell["node"], cell["lanes"]): cell for cell in previous["cells"]}
    expected_history = {(cell["node"], cell["lanes"], tuple(state["queue"]))
        for cell in previous["cells"] for state in cell["states"] if state["source"] == "historical_regression"}
    actual_history = {(cell["node"], cell["lanes"], tuple(state["queue"]))
        for cell in protocol["cells"] for state in cell["states"] if state["historical_regression"]}
    assert actual_history == expected_history
    for cell in protocol["cells"]:
        for state in cell["states"]:
            assert state["caps"] == [2, 3, 144, 256]
            if state["source"] == "historical_regression":
                old = old_cells[cell["node"], cell["lanes"]]["states"][state["source_index"]]
                assert (state["source_state_id"], state["queue"]) == (old["state_id"], old["queue"])
    registered = {**protocol, "registered_at_utc": "new confirmation"}
    runner.verify_protocol(registered, models)
    with pytest.raises(ValueError):
        runner.verify_protocol({**registered, "lp_node_budget": 512}, models)
