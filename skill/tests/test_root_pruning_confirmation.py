"""Frozen fresh sampling and new-versus-borrowed confirmation calls."""
from dataclasses import dataclass
import json
from types import SimpleNamespace

import pytest

import algorithm.certificate_routed_frame_candidates as old_routed_module
import algorithm.resumable_integer_capacity_frame_candidates as old_engine
import algorithm.root_pruned_certificate_routed_frame_candidates as new_routed_module
import algorithm.root_pruned_integer_capacity_frame_candidates as new_engine
from algorithm.experiments import aligned_hardware_root_pruning_confirmation as runner
from algorithm.experiments import aligned_hardware_strict_score_proof_development as reference_module
from algorithm.experiments.aligned_hardware_shared_threshold_queue_confirmation import sample_queues


def test_fresh_panels_retain_sampler_seeds_and_exclude_accepted_prior_queues():
    seeds = (2026092811, 2026092812, 2026092813, 2026092814)
    excluded = {str(lanes): {(lanes + 1, lanes + 2, lanes + 3)} for lanes in (8, 16, 32, 64)}
    before = {key: queues.copy() for key, queues in excluded.items()}
    actual = runner.fresh_design(excluded)
    assert set(actual) == {"8", "16", "32", "64"}
    for lanes in (8, 16, 32, 64):
        seen = excluded[str(lanes)].copy()
        assert len(actual[str(lanes)]) == 4
        for panel, seed in zip(actual[str(lanes)], seeds):
            queues = tuple(tuple(state["queue"]) for state in panel["states"])
            assert panel["seed"] == seed and panel["panel"] == f"fresh_{seed}"
            assert panel["cohort"] == "fresh_confirmation"
            assert queues == sample_queues(lanes, seed, 64, excluded=seen)
            assert len(queues) == len(set(queues)) == 64
            assert set(queues).isdisjoint(seen)
            assert all(0 <= value <= 5 * lanes for queue in queues for value in queue)
            assert [state["state_id"] for state in panel["states"]] == [
                f"fresh:{seed}:{index}" for index in range(64)]
            seen.update(queues)
    assert excluded == before


@dataclass(frozen=True)
class Option:
    requirements: tuple[int, ...]
    duration: float


@dataclass(frozen=True)
class Result:
    label: str
    queue: tuple[int, ...]
    budget: int
    canonical_exact: bool = True


def registered(source, queue, state_id):
    extra = {"source_index": 0, "source_state_id": "original regression"} if source == "historical_regression" else {}
    return {"source": source, "queue": queue, "state_id": state_id,
            "panel": source, "cohort": source, "seed": None, **extra}


def call_harness(monkeypatch, *, incomplete=False):
    events, searches, references = [], {"old": [], "new": []}, []
    frozen = {"state_id": "original regression", "queue": [2, 1, 3],
        "reference": {"label": "borrowed reference"}, "reference_metrics": {"label": "borrowed metrics"},
        "reference_complete": False, "oracle": {"label": "borrowed oracle"},
        "effective_reference_complete": True, "effective_reference_source": "borrowed source",
        "screen_results": {"candidate": {"label": "stale routed result"}}}

    def search_class(label):
        class Search:
            def __init__(self, options, queue, lanes):
                events.append((label + " construction", queue))
                self.queue = queue
                self.caps = []
                searches[label].append(self)

            def advance_to(self, cap):
                self.caps.append(cap)
                events.append((label + " advance", self.queue, cap))
                return Result(label + " trace", self.queue, cap)
        return Search

    def selector(label):
        def select(options, queue, lanes, initial, total, integer):
            events.append((label + " routed", queue, lanes, initial, total, integer))
            return Result(label + " routed", queue, total, not (incomplete and label == "new"))
        return select

    def new_reference(model, queue, lanes, protocol, runtime, oracle):
        events.append(("reference", queue))
        references.append((model, queue, lanes, protocol, runtime, oracle))
        canonical = queue == (0, 0, 0)
        independent = None if canonical or incomplete else {
            "complete": True, "canonical_exact": True, "score_fraction": [0, 1],
            "duration_fraction": [1, 1], "option_indices": [None] * lanes}
        return Result("new reference", queue, 4096, canonical), {"label": "new metrics"}, canonical, independent

    def reference_key(result, queue, option_count):
        events.append(("reference key", queue))
        assert result.queue == queue and option_count == 2
        return ("complete",) if result.canonical_exact else None

    def load(path, node, lanes, registration):
        assert path == runner.ROOT / "historical.json"
        assert (node, lanes, registration) == ("node007", 16, "historical registration")
        return {"states": [frozen]}

    monkeypatch.setattr(old_engine, "IntegerCapacitySearch", search_class("old"))
    monkeypatch.setattr(new_engine, "RootPrunedIntegerCapacitySearch", search_class("new"))
    monkeypatch.setattr(old_routed_module, "select_certificate_routed_frame", selector("old"))
    monkeypatch.setattr(new_routed_module, "select_root_pruned_certificate_routed_frame", selector("new"))
    monkeypatch.setattr(reference_module, "_new_reference", new_reference)
    monkeypatch.setattr(runner, "load_cell", load)
    monkeypatch.setattr(runner, "runtime_provenance", lambda: {"affinity": [2], "gc_enabled": True})
    model = SimpleNamespace(node="node007", lanes=2,
                            options=(Option((1, 0, 0), 1.0), Option((0, 1, 0), 2.0)))
    protocol = {"registered_at_utc": "new registration", "initial_lp_budget": 144,
        "lp_node_budget": 256, "integer_state_budget": 8192, "reference_budget": 4096,
        "oracle_state_budget": 5000000,
        "reference_source": {"path": "historical.json", "registered_at_utc": "historical registration"}}
    runtime = SimpleNamespace(reference_key=reference_key)
    cell = {"lanes": 16, "states": [registered("fresh_confirmation", [9, 8, 7], "fresh"),
        registered("historical_regression", [2, 1, 3], "regression:original regression"),
        registered("new_zero", [0, 0, 0], "zero_queue")]}
    return model, cell, protocol, runtime, events, searches, references, frozen


def test_each_state_has_one_continuous_trace_per_arm_and_new_routed_calls(monkeypatch):
    model, cell, protocol, runtime, events, searches, references, frozen = call_harness(monkeypatch)
    row = runner.run_cell(model, cell, protocol, runtime, progress=False)
    assert len(searches["old"]) == len(searches["new"]) == 3
    assert all(search.caps == [2, 3, 144, 256] for arm in searches.values() for search in arm)
    for state in row["states"]:
        assert state["caps"] == [2, 3, 144, 256]
        for label in ("old", "new"):
            assert list(state["traces"][label]) == ["2", "3", "144", "256"]
            assert state["traces"][label]["256"]["queue"] == state["queue"]
        assert state["old_routed"]["label"] == "old routed"
        assert state["routed"]["label"] == "new routed"
        queue = tuple(state["queue"])
        selected = [(index, event) for index, event in enumerate(events) if event[1] == queue]
        old = next(index for index, event in selected if event[0] == "old routed")
        new = next(index for index, event in selected if event[0] == "new routed")
        assert old < new
        if state["source"] != "historical_regression":
            reference = next(index for index, event in selected if event[0] == "reference")
            trace = next(index for index, event in selected if event[0] == "old construction")
            assert reference < trace < old
    assert len(references) == row["new_reference_calls"] == 2
    assert row["new_reference_oracle_calls"] == 1
    assert row["new_old_routed_calls"] == 3
    assert all(call[0] is model and call[3] is protocol and call[4] is runtime for call in references)
    assert {call[1] for call in references} == {(9, 8, 7), (0, 0, 0)}
    assert all(event[3:] == (144, 256, 8192) for event in events if event[0].endswith("routed"))
    borrowed = row["states"][1]
    for key in ("reference", "reference_metrics", "reference_complete", "oracle",
                "effective_reference_complete", "effective_reference_source"):
        assert borrowed[key] == frozen[key]
    assert row["states"][0]["effective_reference_source"] == "independent_score_face_oracle"
    assert row["states"][2]["effective_reference_source"] == "full_K4096"
    assert row["runtime_start"] == row["runtime_end"] == {"affinity": [2], "gc_enabled": True}


def test_nonclosure_is_retained_without_replacement_and_later_states_still_execute(monkeypatch):
    model, cell, protocol, runtime, events, searches, _, _ = call_harness(monkeypatch, incomplete=True)
    row = runner.run_cell(model, cell, protocol, runtime, progress=False)
    assert len(row["states"]) == 3
    assert row["states"][0]["reference_complete"] is False
    assert row["states"][0]["effective_reference_complete"] is False
    assert row["states"][0]["effective_reference_source"] is None
    assert row["states"][0]["oracle"] is None
    assert all(state["routed"]["canonical_exact"] is False for state in row["states"])
    assert row["new_reference_calls"] == 2 and row["new_reference_oracle_calls"] == 0
    assert row["new_old_routed_calls"] == 3
    assert len(searches["old"]) == len(searches["new"]) == 3
    assert sum(event[0] == "new routed" for event in events) == 3


def protocol_fixture(monkeypatch, tmp_path):
    options = (Option((1, 0, 0), 1.0), Option((0, 1, 0), 2.0))
    models = [SimpleNamespace(node=f"node{index}", options=options,
        snapshot=lambda index=index: {"node": f"node{index}", "physical_lanes": 2}) for index in range(4)]
    snapshots = [model.snapshot() for model in models]
    kernel = tmp_path / "kernel.py"
    kernel.write_text("frozen kernel")
    (tmp_path / "new_runner.py").write_text("new runner")
    budgets = {"initial_lp_budget": 144, "lp_node_budget": 256, "integer_state_budget": 8192}
    development = {**budgets, "models": snapshots, "registered_at_utc": "development registration",
        "execution_sources": {"kernel.py": "frozen kernel"}, "cells": [
            {"node": model.node, "lanes": lanes, "states": [{"queue": [7, 8, 9]}]}
            for model in models for lanes in (8, 16, 32, 64)]}
    tail_queues = {8: ((27, 15, 34),), 16: (), 32: ((96, 15, 13),),
                   64: ((88, 19, 118), (184, 78, 235))}
    panels = {str(lanes): [
        {"panel": "old", "cohort": "old", "states": [
            {"state_id": f"trigger:{lanes}:0", "queue": [lanes, 2, 3]},
            {"state_id": f"trigger:{lanes}:1", "queue": [lanes, 4, 5]},
            {"state_id": "ignored", "queue": [6, 7, 8]}]},
        {"panel": "tails", "cohort": "ordinary_tail_regression", "states": [
            {"state_id": "tail:" + ":".join(map(str, queue)), "queue": list(queue)}
            for queue in tail_queues[lanes]]}]
        for lanes in (8, 16, 32, 64)}
    triggers = [{"node": model.node, "lanes": lanes, "queue": [lanes, 2, 3]}
        for model in models for lanes in (8, 16, 32, 64)]
    triggers += [{"node": "node0", "lanes": 8, "queue": [8, 4, 5]},
                 {"node": "node1", "lanes": 16, "queue": [16, 4, 5]}]
    previous = {**budgets, "models": snapshots, "registered_at_utc": "previous registration",
        "execution_sources": {"kernel.py": "frozen kernel"}, "queue_panels": panels,
        "historical_trigger_keys": triggers,
        "excluded_static_queues": {str(lanes): [[1, 2, 3]] for lanes in (8, 16, 32, 64)}}
    prior_output, previous_output = tmp_path / "development.json", tmp_path / "previous.json"
    prior_output.write_text(json.dumps({"decision": "ROOT_PRUNING_DEVELOPMENT_ADVANCE", "development_gate_passed": True}))
    previous_output.write_text(json.dumps({"decision": "PREVIOUS_TIMING_HOLD", "certificate_checks_passed": True,
        "effective_references_complete": True, "execution_checks_passed": True}))
    prior_output.with_name("development_protocol.json").write_text(json.dumps(development))
    previous_output.with_name("previous_protocol.json").write_text(json.dumps(previous))
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "PRIOR_OUTPUT", prior_output)
    monkeypatch.setattr(runner, "PREVIOUS_OUTPUT", previous_output)
    monkeypatch.setattr(runner, "SOURCES", ("new_runner.py",))
    return models, previous


def test_protocol_excludes_every_prior_source_and_keeps_node_specific_regressions(monkeypatch, tmp_path):
    models, previous = protocol_fixture(monkeypatch, tmp_path)
    protocol = runner.build_protocol(models)
    assert len(protocol["cells"]) == 16
    states = [state for cell in protocol["cells"] for state in cell["states"]]
    assert sum(state["source"] == "fresh_confirmation" for state in states) == 4096
    assert sum(state["source"] == "historical_regression" for state in states) == 34
    assert sum(state["source"] == "new_zero" for state in states) == 16
    assert sum(state.get("historical_trigger", False) for state in states) == 18
    assert sum(state.get("ordinary_tail", False) for state in states) == 16
    for lanes in (8, 16, 32, 64):
        excluded = set(map(tuple, protocol["excluded_static_queues"][str(lanes)]))
        assert {(1, 2, 3), (7, 8, 9), (6, 7, 8), (lanes, 2, 3), (lanes, 4, 5)} <= excluded
        assert set(tuple(state["queue"]) for panel in protocol["queue_panels"][str(lanes)]
                   for state in panel["states"]).isdisjoint(excluded)
    actual_triggers = {(cell["node"], cell["lanes"], tuple(state["queue"]))
        for cell in protocol["cells"] for state in cell["states"] if state.get("historical_trigger")}
    assert actual_triggers == {(item["node"], item["lanes"], tuple(item["queue"]))
                               for item in previous["historical_trigger_keys"]}
    for cell in protocol["cells"]:
        assert len(cell["states"]) == len({tuple(state["queue"]) for state in cell["states"]})
        old_states = [state for panel in previous["queue_panels"][str(cell["lanes"])] for state in panel["states"]]
        for state in cell["states"]:
            if state["source"] == "historical_regression":
                old = old_states[state["source_index"]]
                assert (state["source_state_id"], state["queue"]) == (old["state_id"], old["queue"])
    registered_protocol = {**protocol, "registered_at_utc": "new registration"}
    runner.verify_protocol(registered_protocol, models)
    with pytest.raises(ValueError, match="budgets changed"):
        runner.verify_protocol({**registered_protocol, "lp_node_budget": 512}, models)

