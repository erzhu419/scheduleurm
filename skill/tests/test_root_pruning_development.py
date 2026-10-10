"""Development runner calls and frozen-source selection, without real selectors."""
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

import algorithm.certificate_routed_frame_candidates as old_routed_module
import algorithm.resumable_integer_capacity_frame_candidates as old_engine
import algorithm.root_pruned_certificate_routed_frame_candidates as new_routed_module
import algorithm.root_pruned_integer_capacity_frame_candidates as new_engine
from algorithm.experiments import aligned_hardware_root_pruning_development as runner


@dataclass(frozen=True)
class Option:
    requirements: tuple[int, ...]
    duration: float


@dataclass(frozen=True)
class Result:
    label: str
    queue: tuple[int, ...]
    budget: int


def saved_state(state_id, queue, label, source):
    reference = {"label": "borrowed reference", "queue": queue}
    common = {"state_id": state_id, "queue": queue, "reference": reference,
              "oracle": {"source": source}, "effective_reference_complete": True}
    result = {"label": label}
    return {**common, **({"screen_result": result} if source == "development" else
                         {"screen_results": {"candidate": result, "probe": {"label": "wrong arm"}}})}


def harness(monkeypatch, *, reference_complete=True):
    calls = {"old_search": [], "new_search": [], "old_routed": [],
             "new_routed": [], "reference": [], "reference_key": [], "loads": []}
    sources = {"development": {"states": [saved_state("d0", [2, 1], "old development", "development")]},
               "confirmation": {"states": [saved_state("c0", [1, 3], "old confirmation", "confirmation")]}}

    def search_type(label):
        class Search:
            def __init__(self, options, queue, lanes):
                self.queue = queue
                self.record = {"options": options, "queue": queue, "lanes": lanes, "caps": []}
                calls[label + "_search"].append(self.record)

            def advance_to(self, cap):
                self.record["caps"].append(cap)
                return Result(label, self.queue, cap)
        return Search

    def routed(label):
        def select(options, queue, lanes, initial, total, integer):
            calls[label + "_routed"].append((options, queue, lanes, initial, total, integer))
            return Result(label + " routed", queue, total)
        return select

    def reference(options, queue, lanes, budget):
        calls["reference"].append((options, queue, lanes, budget))
        return Result("new reference", queue, budget)

    def reference_key(result, queue, option_count):
        calls["reference_key"].append((result, queue, option_count))
        return ("canonical",) if reference_complete else None

    def load(path, node, lanes, registration):
        source = path.stem
        calls["loads"].append((source, node, lanes, registration))
        return sources[source]

    monkeypatch.setattr(old_engine, "IntegerCapacitySearch", search_type("old"))
    monkeypatch.setattr(new_engine, "RootPrunedIntegerCapacitySearch", search_type("new"))
    monkeypatch.setattr(old_routed_module, "select_certificate_routed_frame", routed("old"))
    monkeypatch.setattr(new_routed_module, "select_root_pruned_certificate_routed_frame", routed("new"))
    monkeypatch.setattr(runner, "load_cell", load)
    model = SimpleNamespace(node="node007", options=(Option((1, 0), 1.0), Option((0, 1), 2.0)))
    protocol = {"registered_at_utc": "new registration", "sources": {
        "development": {"path": "development.json", "registered_at_utc": "development registration"},
        "confirmation": {"path": "confirmation.json", "registered_at_utc": "confirmation registration"}}}
    runtime = SimpleNamespace(select_canonical_count_space_bnb_frame=reference,
                              reference_key=reference_key)
    return model, protocol, runtime, calls, sources


def registered(state_id, queue, source, index=None):
    return {"state_id": state_id, "queue": queue, "source": source,
            **({"source_index": index} if index is not None else {}), "cohort": source}


def test_one_search_per_state_advances_the_registered_cumulative_ladder(monkeypatch):
    model, protocol, runtime, calls, _ = harness(monkeypatch)
    cell = {"lanes": 32, "states": [registered("d0", [2, 1], "development", 0),
                                     registered("c0", [1, 3], "confirmation", 0)]}
    row = runner.run_cell(model, cell, protocol, runtime)
    assert len(calls["old_search"]) == len(calls["new_search"]) == 2
    for label in ("old", "new"):
        for search in calls[label + "_search"]:
            assert search["caps"] == [2, 3, 144, 256]
            assert search["lanes"] == 32
            assert search["options"] is model.options
    assert len(calls["new_routed"]) == 2
    assert not calls["old_routed"] and not calls["reference"] and not calls["reference_key"]
    assert row["new_reference_calls"] == row["new_old_routed_calls"] == 0
    for state in row["states"]:
        assert state["caps"] == [2, 3, 144, 256]
        assert list(state["traces"]["old"]) == ["2", "3", "144", "256"]
        assert list(state["traces"]["new"]) == ["2", "3", "144", "256"]
        assert state["routed"]["budget"] == 256
    assert all(call[3:] == (144, 256, 8192) for call in calls["new_routed"])


def test_development_and_confirmation_borrow_the_correct_routed_arm_and_reference(monkeypatch):
    model, protocol, runtime, calls, sources = harness(monkeypatch)
    cell = {"lanes": 8, "states": [registered("d0", [2, 1], "development", 0),
                                    registered("c0", [1, 3], "confirmation", 0)]}
    row = runner.run_cell(model, cell, protocol, runtime)
    for state, source, label in zip(row["states"], ("development", "confirmation"),
                                    ("old development", "old confirmation")):
        saved = sources[source]["states"][0]
        assert state["old_routed"] == {"label": label}
        assert state["reference"] == saved["reference"]
        assert state["oracle"] == saved["oracle"]
        assert state["effective_reference_complete"] is True
    assert calls["loads"] == [
        ("development", "node007", 8, "development registration"),
        ("confirmation", "node007", 8, "confirmation registration")]


def test_sixteen_zero_controls_alone_add_new_references_and_old_routed_calls(monkeypatch):
    model, protocol, runtime, calls, _ = harness(monkeypatch)
    rows = []
    for node in ("cpu_a", "node007", "gpu_a", "gpu_b"):
        model.node = node
        for lanes in (8, 16, 32, 64):
            cell = {"lanes": lanes, "states": [registered("d0", [2, 1], "development", 0),
                                                registered("zero_queue", [0, 0], "new_zero")]}
            rows.append(runner.run_cell(model, cell, protocol, runtime))
    assert sum(row["new_reference_calls"] for row in rows) == 16
    assert sum(row["new_old_routed_calls"] for row in rows) == 16
    assert len(calls["reference"]) == len(calls["reference_key"]) == len(calls["old_routed"]) == 16
    assert len(calls["new_routed"]) == len(calls["old_search"]) == len(calls["new_search"]) == 32
    assert all(call[1] == (0, 0) and call[3] == 4096 for call in calls["reference"])
    assert all(call[1] == (0, 0) and call[3:] == (144, 256, 8192) for call in calls["old_routed"])
    for row in rows:
        assert row["states"][1]["oracle"] is None
        assert row["states"][1]["reference"]["label"] == "new reference"
        assert row["states"][1]["effective_reference_complete"] is True


@pytest.mark.parametrize("complete", (False, True))
def test_runtime_reference_key_is_given_the_actual_new_reference_and_retained(monkeypatch, complete):
    model, protocol, runtime, calls, _ = harness(monkeypatch, reference_complete=complete)
    row = runner.run_cell(model, {"lanes": 16, "states": [registered("zero_queue", [0, 0], "new_zero")]},
                          protocol, runtime)
    assert calls["reference_key"] == [(Result("new reference", (0, 0), 4096), (0, 0), 2)]
    assert row["states"][0]["effective_reference_complete"] is complete
    assert row["new_reference_calls"] == row["new_old_routed_calls"] == 1


def test_changed_borrowed_queue_stops_before_invoking_new_searches(monkeypatch):
    model, protocol, runtime, calls, _ = harness(monkeypatch)
    with pytest.raises(ValueError, match="borrowed state changed"):
        runner.run_cell(model, {"lanes": 8, "states": [registered("d0", [9, 9], "development", 0)]},
                        protocol, runtime)
    assert not calls["old_search"] and not calls["new_search"] and not calls["new_routed"]
