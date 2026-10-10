"""Freeze trigger discovery before independent reference and proof validation."""
from copy import deepcopy
from dataclasses import dataclass
import json
from types import SimpleNamespace

import pytest

from algorithm.experiments import aligned_hardware_frontier_trigger_discovery as runner
from algorithm.experiments.aligned_hardware_shared_threshold_queue_confirmation import sample_queues


@dataclass(frozen=True)
class Option:
    requirements: tuple[int, ...]
    duration: float


@dataclass(frozen=True)
class Selection:
    score_exact: bool
    canonical_exact: bool
    lp_calls: int = 3
    lp_node_budget: int = 256


@dataclass(frozen=True)
class Stage:
    selection: Selection


def models_fixture():
    return [SimpleNamespace(node=f"node{index}", lanes=2,
        options=(Option((1, 0, 0), 1.0), Option((0, 1, 0), 2.0)),
        snapshot=lambda index=index: {"node": f"node{index}", "physical_lanes": 2}) for index in range(4)]


def panels_fixture():
    return [{"seed": seed, "panel": f"trigger_{seed}", "states": [
        {"state_id": f"trigger:{seed}:{index}", "queue": [100 + index % 200, 100 + index // 200, 200 + offset],
         "seed": seed, "index": index, "panel": f"trigger_{seed}", "cohort": "conditional_trigger_validation"}
        for index in range(1024)]} for offset, seed in enumerate((2026092851, 2026092852, 2026092853, 2026092854))]


def screen_factory(models, triggers):
    nodes = {id(model.options): model.node for model in models}
    instances = []

    class Search:
        def __init__(self, options, queue, lanes):
            assert lanes == 64
            self.node, self.queue, self.caps = nodes[id(options)], tuple(queue), []
            instances.append(self)

        def advance_to(self, cap):
            assert cap == 256 and not self.caps
            self.caps.append(cap)
            return Stage(Selection((self.node, self.queue) not in triggers, False))

    return Search, instances


def test_discovery_sampler_keeps_seeds_law_and_all_prior_exclusions():
    excluded = {(54, 58, 290), (100, 100, 200), (7, 8, 9)}
    original, seen = excluded.copy(), excluded.copy()
    panels = runner.fresh_design(excluded)
    assert len(panels) == 4
    for panel, seed in zip(panels, (2026092851, 2026092852, 2026092853, 2026092854)):
        queues = tuple(tuple(state["queue"]) for state in panel["states"])
        assert (panel["seed"], panel["panel"]) == (seed, f"trigger_{seed}")
        assert queues == sample_queues(64, seed, 1024, excluded=seen)
        assert len(set(queues)) == 1024 and set(queues).isdisjoint(seen)
        for index, state in enumerate(panel["states"]):
            assert (state["state_id"], state["index"], state["seed"], state["panel"], state["cohort"]) == (
                f"trigger:{seed}:{index}", index, seed, f"trigger_{seed}", "conditional_trigger_validation")
        seen.update(queues)
    assert excluded == original


def screen_protocol(panels):
    return {"registered_at_utc": "screen registered", "lanes": 64, "queue_panels": panels,
        "initial_lp_budget": 144, "lp_node_budget": 256, "integer_state_budget": 8192,
        "block_size": 64, "max_rounds": 16, "min_trigger_queues": 2, "min_trigger_seeds": 2,
        "validation_template": {"historical_regression_keys": [], "failure_regression_keys": [],
            "initial_lp_budget": 144, "lp_node_budget": 256, "integer_state_budget": 8192,
            "reference_budget": 4096, "oracle_state_budget": 5000000,
            "reference_source": {"path": "independent.json", "registered_at_utc": "prior"}}}


def test_stopping_completes_the_round_and_retains_every_trigger_before_validation(monkeypatch):
    models, panels = models_fixture(), panels_fixture()
    queue = lambda seed, index: tuple(panels[seed]["states"][index]["queue"])
    triggers = {("node0", queue(0, 0)), ("node2", queue(0, 0)), ("node1", queue(0, 1)),
                ("node2", queue(1, 64)), ("node3", queue(3, 127))}
    factory, instances = screen_factory(models, triggers)
    completed = []

    def on_round(record):
        assert len(record["states"]) == 4 * 64 * 4
        assert len(instances) == (record["round_index"] + 1) * 4 * 64 * 4
        completed.append(record["round_index"])

    def no_validation(*args, **kwargs):
        pytest.fail("references, proof results and validation cannot participate in discovery")

    monkeypatch.setattr(runner, "run_cell", no_validation)
    monkeypatch.setattr(runner, "runtime_provenance", lambda: {"cpu_affinity": [0]})
    protocol = screen_protocol(panels)
    discovery = runner.run_discovery(models, protocol, factory, progress=False, on_round=on_round)
    assert completed == [0, 1] and discovery["stop_reason"] == "two_seed_trigger_threshold"
    assert len(instances) == 2048 and all(instance.caps == [256] for instance in instances)
    assert sum(state["stage"]["selection"]["lp_calls"] for block in discovery["rounds"]
               for state in block["states"]) == 3 * 2048
    expected = {queue(0, 0): ["node0", "node2"], queue(0, 1): ["node1"],
                queue(1, 64): ["node2"], queue(3, 127): ["node3"]}
    assert {tuple(state["queue"]): state["trigger_nodes"] for state in discovery["trigger_states"]} == expected
    assert discovery["runtime_start"] == discovery["runtime_end"] == {"cpu_affinity": [0]}
    for index, block in enumerate(discovery["rounds"]):
        assert [(state["seed"], state["index"], state["node"]) for state in block["states"]] == [
            (panel["seed"], position, model.node) for panel in panels
            for position in range(index * 64, (index + 1) * 64) for model in models]


@pytest.mark.parametrize("trigger_case", ("none", "one_queue_four_catalogues", "two_queues_one_seed"))
def test_insufficient_distinct_queues_or_seeds_use_the_full_registered_limit(monkeypatch, trigger_case):
    models, panels = models_fixture(), panels_fixture()
    first = tuple(panels[0]["states"][0]["queue"])
    second = tuple(panels[0]["states"][1]["queue"])
    triggers = set()
    if trigger_case == "one_queue_four_catalogues":
        triggers = {(model.node, first) for model in models}
    elif trigger_case == "two_queues_one_seed":
        triggers = {("node0", first), ("node1", second)}
    factory, instances = screen_factory(models, triggers)
    monkeypatch.setattr(runner, "runtime_provenance", lambda: {})
    discovery = runner.run_discovery(models, screen_protocol(panels), factory, progress=False)
    assert discovery["stop_reason"] == "screen_limit_reached"
    assert len(discovery["rounds"]) == 16 and len(instances) == 4 * 4 * 1024
    assert [block["round_index"] for block in discovery["rounds"]] == list(range(16))
    assert all(len(block["states"]) == 1024 for block in discovery["rounds"])
    assert len(discovery["trigger_states"]) == {"none": 0, "one_queue_four_catalogues": 1, "two_queues_one_seed": 2}[trigger_case]
    assert all(instance.caps == [256] for instance in instances)


def test_locked_validation_roster_pairs_every_trigger_with_all_catalogues_and_new_references():
    models, panels = models_fixture(), panels_fixture()
    protocol = screen_protocol(panels)
    trigger_states = [{**panels[seed]["states"][index], "trigger_nodes": [node]}
                      for seed, index, node in ((0, 0, "node0"), (1, 64, "node2"), (3, 127, "node3"))]
    discovery = {"trigger_states": trigger_states, "stop_reason": "two_seed_trigger_threshold"}
    before, original_discovery = deepcopy(protocol), deepcopy(discovery)
    validation = runner.validation_protocol(protocol, discovery, models)
    assert protocol == before and validation["registered_at_utc"] != protocol["registered_at_utc"]
    assert validation["discovery_registered_at_utc"] == "screen registered"
    assert validation["trigger_states"] == trigger_states and validation["screen_stop_reason"] == discovery["stop_reason"]
    assert validation["historical_regression_keys"] == validation["failure_regression_keys"] == []
    assert validation["reference_source"] == protocol["validation_template"]["reference_source"]
    assert validation["expected_state_count"] == validation["expected_new_reference_calls"] == 12
    assert validation["expected_interface_calls"] == 120 and validation["expected_native_constructions"] == 48
    assert validation["expected_selector_calls"] == 24 and validation["expected_native_advance_calls"] == 96
    assert [cell["node"] for cell in validation["cells"]] == [model.node for model in models]
    for cell in validation["cells"]:
        assert cell["lanes"] == 64 and len(cell["states"]) == 3
        for state, trigger in zip(cell["states"], trigger_states):
            assert {key: state[key] for key in trigger} == trigger
            assert state["source"] == "fresh_confirmation" and state["historical_regression"] is False
            assert state["cohort"] == "conditional_trigger_validation" and state["caps"] == [2, 3, 144, 256]
    validation["cells"][0]["states"][0]["queue"][0] += 1
    assert discovery == original_discovery and protocol == before


def test_registration_records_the_prior_summary_repair_and_freezes_algorithm_sources(monkeypatch, tmp_path):
    models = models_fixture()
    repaired_source = "summary pointers use an optional seed\n"
    repair = {"file": "prior_summary.py", "reason": "historical seed metadata was absent",
        "source_diff": "optional seed pointer only", "source": repaired_source,
        "new_selector_calls": 0, "new_reference_calls": 0}
    previous_result = {"decision": "FRONTIER_SCORE_PROOF_FRESH_CONFIRMATION_ADVANCE",
        "execution_checks_passed": True, "effective_references_complete": True,
        "confirmation_gate_passed": True, "summary_repair": repair,
        "prior_decisions": {"late_score_proof_development": "LATE_SCORE_PROOF_DEVELOPMENT_HOLD",
                            "branch_reserved_timing": "BRANCH_RESERVED_TIMING_HOLD"}}
    old = {"registered_at_utc": "previous confirmation", "models": [model.snapshot() for model in models],
        "initial_lp_budget": 144, "lp_node_budget": 256, "integer_state_budget": 8192,
        "reference_budget": 4096, "oracle_state_budget": 5000000,
        "reference_source": {"path": "frontier_development.json", "registered_at_utc": "development"},
        "rule": "Only paid frontier boxes share the unchanged 8192 integer state budget.",
        "execution_sources": {"kernel.py": "immutable algorithm\n", "prior_summary.py": "registered summary\n"},
        "excluded_static_queues": {"64": [[1, 2, 3], [7, 8, 9]]},
        "cells": [{"node": model.node, "lanes": 64, "states": [
            {"queue": [100 + index, 29, 280], "source": "fresh_confirmation"},
            {"queue": [54, 58, 290], "source": "failure_regression"},
            {"queue": [27, 15, 34], "source": "historical_regression"},
            {"queue": [0, 0, 0], "source": "new_zero"}]} for index, model in enumerate(models)]}
    previous_output = tmp_path / "previous.json"
    previous_output.write_text(json.dumps(previous_result))
    previous_output.with_name("previous_protocol.json").write_text(json.dumps(old))
    (tmp_path / "kernel.py").write_text("immutable algorithm\n")
    (tmp_path / "prior_summary.py").write_text(repaired_source)
    (tmp_path / "new_runner.py").write_text("new discovery runner\n")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "PREVIOUS_OUTPUT", previous_output)
    monkeypatch.setattr(runner, "SOURCES", ("new_runner.py",))

    protocol = runner.build_protocol(models)
    assert protocol["prior_summary_repair"] == {key: repair[key] for key in ("file", "reason", "source_diff")}
    assert protocol["execution_sources"] == {"kernel.py": "immutable algorithm\n",
        "prior_summary.py": repaired_source, "new_runner.py": "new discovery runner\n"}
    assert json.loads(previous_output.with_name("previous_protocol.json").read_text())["execution_sources"] == old["execution_sources"]
    assert protocol["prior_decisions"] == {**previous_result["prior_decisions"],
        "frontier_score_proof_confirmation": previous_result["decision"]}
    assert protocol["expected_max_screen_calls"] == 4 * 4 * 1024
    assert (protocol["lanes"], protocol["queues_per_panel"], protocol["block_size"], protocol["max_rounds"],
            protocol["min_trigger_queues"], protocol["min_trigger_seeds"]) == (64, 1024, 64, 16, 2, 2)
    excluded = set(map(tuple, protocol["excluded_static_queues"]["64"]))
    assert {(1, 2, 3), (7, 8, 9)} <= excluded
    assert {tuple(state["queue"]) for cell in old["cells"] for state in cell["states"]} <= excluded
    queues = [tuple(state["queue"]) for panel in protocol["queue_panels"] for state in panel["states"]]
    assert len(queues) == len(set(queues)) == 4096 and set(queues).isdisjoint(excluded)
    template = protocol["validation_template"]
    assert template["reference_source"] == old["reference_source"]
    assert template["execution_sources"] == protocol["execution_sources"]
    assert template["historical_regression_keys"] == template["failure_regression_keys"] == []
    assert (template["initial_lp_budget"], template["lp_node_budget"], template["integer_state_budget"],
            template["reference_budget"], template["oracle_state_budget"]) == (144, 256, 8192, 4096, 5000000)
    registered = {**protocol, "registered_at_utc": "new discovery"}
    runner.verify_protocol(registered, models)
    with pytest.raises(ValueError, match="stopping rule changed"):
        runner.verify_protocol({**registered, "block_size": 32}, models)
    (tmp_path / "kernel.py").write_text("changed algorithm\n")
    with pytest.raises(ValueError, match="frozen source changed: kernel.py"):
        runner.build_protocol(models)
