"""Keep native trigger discovery separate from residual proof confirmation."""
from copy import deepcopy
from dataclasses import dataclass, fields
from fractions import Fraction
import json
from types import SimpleNamespace

import pytest

from algorithm.canonical_count_space_bnb_frame_candidates import CanonicalCountSpaceBnbSelection
from algorithm.experiments import aligned_hardware_frontier_trigger_discovery as discovery_runner
from algorithm.experiments import aligned_hardware_residual_capacity_trigger_confirmation as runner
from algorithm.experiments.aligned_hardware_shared_threshold_queue_confirmation import sample_queues


@dataclass(frozen=True)
class Option:
    requirements: tuple[int, ...]
    duration: float


@dataclass(frozen=True)
class Selection:
    score_exact: bool
    canonical_exact: bool = False
    lp_calls: int = 3
    lp_node_budget: int = 256


@dataclass(frozen=True)
class Stage:
    selection: Selection


@dataclass(frozen=True)
class MethodResult:
    label: str
    queue: tuple[int, ...]
    budget: int
    canonical_exact: bool
    score: float = 999.0
    status: str = "COMPLETE"


def models_fixture():
    return [SimpleNamespace(node=f"node{index}", lanes=2,
        options=(Option((1, 0, 0), 1.0), Option((0, 1, 0), 2.0)),
        snapshot=lambda index=index: {"node": f"node{index}", "physical_lanes": 2}) for index in range(4)]


def panels_fixture():
    return [{"seed": seed, "panel": f"trigger_{seed}", "states": [
        {"state_id": f"trigger:{seed}:{index}", "queue": [100 + index, 100, 200 + offset],
         "seed": seed, "index": index, "panel": f"trigger_{seed}", "cohort": "conditional_trigger_validation"}
        for index in range(128)]} for offset, seed in enumerate((2026092861, 2026092862, 2026092863, 2026092864))]


def test_sampling_preserves_frozen_law_and_excludes_every_earlier_panel():
    excluded = {(54, 58, 290), (120, 123, 258), (7, 8, 9)}
    original, seen = excluded.copy(), excluded.copy()
    panels = runner.fresh_design(excluded)
    assert len(panels) == 4
    for panel, seed in zip(panels, (2026092861, 2026092862, 2026092863, 2026092864)):
        queues = tuple(tuple(state["queue"]) for state in panel["states"])
        assert (panel["seed"], panel["panel"]) == (seed, f"trigger_{seed}")
        assert queues == sample_queues(64, seed, 1024, excluded=seen)
        assert len(set(queues)) == 1024 and set(queues).isdisjoint(seen)
        assert [state["state_id"] for state in panel["states"]] == [f"trigger:{seed}:{index}" for index in range(1024)]
        assert all(state["cohort"] == "conditional_trigger_validation" and state["seed"] == seed for state in panel["states"])
        seen.update(queues)
    assert excluded == original


def control_cells(models):
    return [{"node": model.node, "lanes": 64, "states": [
        {"state_id": f"{model.node}:control:{index}", "queue": list(queue), "source": source,
         "panel": source, "cohort": source, "seed": None, "historical_regression": False,
         "caps": [2, 3, 144, 256], "source_index": 99,
         "borrowed_from": "residual_capacity_development", "borrowed_index": index,
         "source_state_id": f"{model.node}:control:{index}", "source_record_source": source}
        for index, (queue, source) in enumerate((((54, 58, 290), "failure_development"),
            ((120, 123, 258), "axis_failure_development")))]} for model in models]


def screen_protocol(models, panels):
    return {"registered_at_utc": "screen registered", "lanes": 64, "queue_panels": panels,
        "initial_lp_budget": 144, "lp_node_budget": 256, "integer_state_budget": 8192,
        "block_size": 64, "max_rounds": 2, "min_trigger_queues": 2, "min_trigger_seeds": 2,
        "validation_template": {"initial_lp_budget": 144, "lp_node_budget": 256,
            "integer_state_budget": 8192, "reference_budget": 4096, "oracle_state_budget": 5000000,
            "control_cells": control_cells(models),
            "reference_source": {"path": "residual_development.json", "registered_at_utc": "developed"}}}


def test_native_discovery_finishes_the_stopping_round_without_reference_or_proof_access(monkeypatch):
    assert runner.run_discovery is discovery_runner.run_discovery
    models, panels = models_fixture(), panels_fixture()
    queue = lambda seed, index: tuple(panels[seed]["states"][index]["queue"])
    triggers = {("node0", queue(0, 0)), ("node2", queue(0, 0)), ("node1", queue(0, 1)),
                ("node2", queue(1, 64)), ("node3", queue(3, 127))}
    nodes, instances, completed = {id(model.options): model.node for model in models}, [], []

    class Search:
        def __init__(self, options, q, lanes):
            assert lanes == 64
            self.node, self.queue, self.caps = nodes[id(options)], tuple(q), []
            instances.append(self)

        def advance_to(self, cap):
            assert cap == 256 and not self.caps
            self.caps.append(cap)
            return Stage(Selection((self.node, self.queue) not in triggers))

    def forbidden(*args, **kwargs):
        pytest.fail("references, residual proofs and validation must not influence discovery")

    def on_round(record):
        assert len(record["states"]) == 4 * 64 * 4
        assert len(instances) == (record["round_index"] + 1) * 4 * 64 * 4
        completed.append(record["round_index"])

    monkeypatch.setattr(runner, "run_cell", forbidden)
    monkeypatch.setattr(runner, "_dependencies", forbidden)
    monkeypatch.setattr(discovery_runner, "runtime_provenance", lambda: {"cpu_affinity": [0]})
    discovery = runner.run_discovery(models, screen_protocol(models, panels), Search, progress=False, on_round=on_round)
    assert completed == [0, 1] and discovery["stop_reason"] == "two_seed_trigger_threshold"
    assert len(instances) == 2048 and all(instance.caps == [256] for instance in instances)
    assert sum(state["stage"]["selection"]["lp_calls"] for record in discovery["rounds"]
               for state in record["states"]) == 3 * 2048
    assert {tuple(state["queue"]): state["trigger_nodes"] for state in discovery["trigger_states"]} == {
        queue(0, 0): ["node0", "node2"], queue(0, 1): ["node1"],
        queue(1, 64): ["node2"], queue(3, 127): ["node3"]}
    for index, record in enumerate(discovery["rounds"]):
        assert [(state["seed"], state["index"], state["node"]) for state in record["states"]] == [
            (panel["seed"], position, model.node) for panel in panels
            for position in range(index * 64, (index + 1) * 64) for model in models]


def test_validation_locks_all_paired_triggers_and_eight_controls_before_execution():
    models, panels = models_fixture(), panels_fixture()
    protocol = screen_protocol(models, panels)
    triggers = [{**panels[seed]["states"][index], "trigger_nodes": [node]}
                for seed, index, node in ((0, 0, "node0"), (1, 64, "node2"), (3, 127, "node3"))]
    discovery = {"trigger_states": triggers, "stop_reason": "two_seed_trigger_threshold"}
    original_protocol, original_discovery = deepcopy(protocol), deepcopy(discovery)
    validation = runner.validation_protocol(protocol, discovery, models)
    assert validation["registered_at_utc"] != protocol["registered_at_utc"]
    assert validation["discovery_registered_at_utc"] == "screen registered"
    assert validation["trigger_states"] == triggers and validation["screen_stop_reason"] == discovery["stop_reason"]
    assert (validation["expected_state_count"], validation["expected_fresh_state_count"],
        validation["expected_control_state_count"], validation["expected_new_reference_calls"]) == (20, 12, 8, 12)
    assert (validation["expected_borrowed_reference_count"], validation["expected_borrowed_native_snapshots"],
        validation["expected_borrowed_prior_axis_count"]) == (8, 64, 16)
    assert (validation["expected_prior_axis_selector_calls"], validation["expected_residual_selector_calls"],
        validation["expected_native_advance_calls"], validation["expected_interface_calls"],
        validation["expected_native_constructions"]) == (24, 40, 96, 160, 88)
    for cell, controls in zip(validation["cells"], protocol["validation_template"]["control_cells"]):
        assert (cell["node"], cell["lanes"]) == (controls["node"], 64)
        assert cell["states"][3:] == controls["states"]
        for state, trigger in zip(cell["states"], triggers):
            assert {key: state[key] for key in trigger} == trigger
            assert state["source"] == "fresh_confirmation" and state["historical_regression"] is False
            assert state["caps"] == [2, 3, 144, 256]
    validation["cells"][0]["states"][0]["queue"][0] += 1
    assert protocol == original_protocol and discovery == original_discovery


def full_reference(queue, *, incomplete=False):
    first = queue == (1, 0, 0)
    duration, score = (1.0, 1.0) if first else (2.0, 0.5)
    score_exact, canonical_exact = not (incomplete and first), not first
    values = {field.name: 0 for field in fields(CanonicalCountSpaceBnbSelection)}
    values.update(option_counts=(1, 0) if first else (0, 1),
        option_indices=((0 if first else 1),) + (None,) * 63, service=queue,
        duration=duration, score=score, upper_bound=score if score_exact else 2.0,
        gap=0.0 if score_exact else 1.0, epsilon=0.0 if score_exact else 1.0,
        exact=canonical_exact, score_exact=score_exact, canonical_exact=canonical_exact, lp_node_budget=4096,
        frontier_regions_remaining=0 if canonical_exact else 1,
        canonical_tie_regions_remaining=int(score_exact and not canonical_exact))
    return CanonicalCountSpaceBnbSelection(**values)


def call_harness(monkeypatch, *, incomplete=False):
    model = models_fixture()[0]
    controls = control_cells([model])[0]["states"]
    borrowed = []
    for index, control in enumerate(controls):
        borrowed.append({**deepcopy(control),
            "reference": {"independent": index}, "reference_metrics": {"borrowed": index},
            "reference_complete": False, "oracle": {"borrowed_oracle": index},
            "effective_reference_complete": True, "effective_reference_source": f"borrowed source {index}",
            "traces": {arm: {str(cap): {"label": f"saved native {arm}:{index}:{cap}"}
                for cap in (2, 3, 144, 256)} for arm in ("old", "new")},
            "prior_routed": {arm: {"label": f"saved axis {arm}:{index}"} for arm in ("old", "new")},
            "old_routed": {"label": f"previous residual old:{index}"},
            "routed": {"label": f"previous residual new:{index}"}})
    fresh = [{"state_id": f"fresh:{index}", "queue": list(q), "source": "fresh_confirmation",
        "caps": [2, 3, 144, 256], "panel": f"trigger_{2026092861 + index}",
        "cohort": "conditional_trigger_validation", "seed": 2026092861 + index, "historical_regression": False}
        for index, q in enumerate(((1, 0, 0), (0, 1, 0)))]
    states = [fresh[0], controls[0], fresh[1], controls[1]]
    protocol = {"registered_at_utc": "locked validation", "initial_lp_budget": 144, "lp_node_budget": 256,
        "integer_state_budget": 8192, "reference_budget": 4096, "oracle_state_budget": 5000000,
        "reference_source": {"path": "residual_development.json", "registered_at_utc": "developed"}}
    events, searches, cold, references, oracles = [], {"old": [], "new": []}, [], [], []

    def search_class(label):
        class Search:
            def __init__(self, options, queue, lanes):
                assert options is model.options and lanes == 64
                self.queue, self.caps = queue, []
                searches[label].append(self)
                events.append((label + " search", queue, None))

            def advance_to(self, cap):
                assert cap == (2, 3, 144, 256)[len(self.caps)]
                self.caps.append(cap)
                events.append((label + " advance", self.queue, cap))
                return MethodResult(label + " trace", self.queue, cap, not incomplete,
                    status="STATE_BUDGET_EXHAUSTED" if incomplete else "COMPLETE")
        return Search

    def selector(label):
        def run(options, queue, lanes, initial, total, integer):
            assert options is model.options and (lanes, initial, total, integer) == (64, 144, 256, 8192)
            cold.append(SimpleNamespace(label=label, queue=queue))
            events.append((label + " routed", queue, total))
            return MethodResult(label + " routed", queue, total, not incomplete,
                status="STATE_BUDGET_EXHAUSTED" if incomplete else "COMPLETE")
        return run

    def original_catalogue_reference(options, queue, lanes, budget):
        assert options is model.options and (lanes, budget) == (64, 4096)
        assert not any(event[1] == queue for event in events)
        events.append(("reference", queue, budget))
        references.append(queue)
        return full_reference(queue, incomplete=incomplete)

    def oracle(options, queue, lanes, proven_score, *, state_visit_budget, maximum_duration):
        assert options is model.options and (queue, lanes) == ((1, 0, 0), 64)
        assert (proven_score, state_visit_budget, maximum_duration) == (Fraction(1), 5000000, 1.0)
        assert [event[0] for event in events if event[1] == queue] == ["reference"]
        events.append(("oracle", queue, state_visit_budget))
        oracles.append(proven_score)
        return {"complete": True, "canonical_exact": True, "score_fraction": [1, 1],
                "duration_fraction": [1, 1], "option_indices": [0] + [None] * 63}

    def record(metrics, *args, **kwargs):
        metrics["calls"] += 1

    from algorithm.experiments.aligned_hardware_strict_score_proof_development import _new_reference
    runtime = SimpleNamespace(select_canonical_count_space_bnb_frame=original_catalogue_reference,
        reference_key=lambda ref, q, count: ("full",) if ref.canonical_exact else None,
        _metrics=lambda: {"calls": 0}, _record=record)
    dependencies = {"searches": {arm: search_class(arm) for arm in ("old", "new")},
        "prior_selectors": {arm: selector("axis " + arm) for arm in ("old", "new")},
        "selectors": {arm: selector("residual " + arm) for arm in ("old", "new")},
        "new_reference": _new_reference, "oracle": oracle,
        "oracle_key": lambda result, count: ("oracle",) if result["complete"] else None}

    def load(path, node, lanes, registration):
        assert (path, node, lanes, registration) == (runner.ROOT / "residual_development.json", model.node, 64, "developed")
        return {"states": borrowed}

    monkeypatch.setattr(runner, "load_cell", load)
    monkeypatch.setattr(runner, "runtime_provenance", lambda: {"cpu_affinity": [0]})
    return model, {"node": model.node, "lanes": 64, "states": states}, protocol, runtime, dependencies, events, searches, cold, references, oracles, borrowed


def test_fresh_references_precede_two_ladders_and_four_cold_routers_while_controls_borrow(monkeypatch):
    model, cell, protocol, runtime, deps, events, searches, cold, references, oracles, borrowed = call_harness(monkeypatch)
    row = runner.run_cell(model, cell, protocol, runtime, deps, progress=False)
    assert references == [(1, 0, 0), (0, 1, 0)] and oracles == [Fraction(1)]
    assert {key: row[key] for key in ("new_reference_calls", "new_reference_oracle_calls", "borrowed_reference_count",
        "borrowed_native_snapshots", "borrowed_prior_axis_count", "new_prior_axis_selector_calls",
        "new_residual_selector_calls", "executed_native_advance_calls", "fresh_native_search_constructions")} == {
        "new_reference_calls": 2, "new_reference_oracle_calls": 1, "borrowed_reference_count": 2,
        "borrowed_native_snapshots": 16, "borrowed_prior_axis_count": 4, "new_prior_axis_selector_calls": 4,
        "new_residual_selector_calls": 8, "executed_native_advance_calls": 16, "fresh_native_search_constructions": 4}
    assert len(cold) == 12 and len({id(search) for search in cold}) == 12
    assert sum(map(len, searches.values())) + len(cold) == 16
    for arm in ("old", "new"):
        assert len(searches[arm]) == 2 and all(search.caps == [2, 3, 144, 256] for search in searches[arm])
    for state in row["states"]:
        labels = [event[0] for event in events if event[1] == tuple(state["queue"])]
        if state["source"] == "fresh_confirmation":
            assert labels[0] == "reference"
            assert labels.count("old advance") == labels.count("new advance") == 4
            assert labels[-4:] == ["axis old routed", "axis new routed", "residual old routed", "residual new routed"]
            assert list(state["traces"]["old"]) == list(state["traces"]["new"]) == ["2", "3", "144", "256"]
            assert state["previous_residual_routed"] is None
        else:
            assert labels == ["residual old routed", "residual new routed"]
            saved = borrowed[state["borrowed_index"]]
            for key in ("reference", "reference_metrics", "reference_complete", "oracle",
                        "effective_reference_complete", "effective_reference_source", "traces", "prior_routed"):
                assert state[key] == saved[key]
            assert state["previous_residual_routed"] == {"old": saved["old_routed"], "new": saved["routed"]}
            assert state["prior_routed"] != state["previous_residual_routed"]
        assert state["old_routed"]["label"] == "residual old routed" and state["routed"]["label"] == "residual new routed"
    assert row["states"][0]["reference"]["score"] == 1.0 and row["states"][0]["routed"]["score"] == 999.0
    assert row["states"][0]["effective_reference_source"] == "independent_score_face_oracle"
    assert row["states"][2]["effective_reference_source"] == "full_K4096"
    assert row["runtime_start"] == row["runtime_end"] == {"cpu_affinity": [0]}


def test_incomplete_reference_and_exhausted_methods_remain_visible_without_candidate_rescue(monkeypatch):
    model, cell, protocol, runtime, deps, _, searches, cold, references, oracles, _ = call_harness(monkeypatch, incomplete=True)
    row = runner.run_cell(model, cell, protocol, runtime, deps, progress=False)
    assert len(row["states"]) == 4 and references == [(1, 0, 0), (0, 1, 0)] and not oracles
    first, later = row["states"][0], row["states"][2]
    assert first["reference"]["score_exact"] is False and first["oracle"] is None
    assert first["reference_complete"] is False and first["effective_reference_complete"] is False
    assert first["effective_reference_source"] is None
    assert later["reference_complete"] is True and later["effective_reference_source"] == "full_K4096"
    assert all(state["routed"]["canonical_exact"] is False and state["routed"]["status"] == "STATE_BUDGET_EXHAUSTED"
               for state in row["states"])
    assert len(cold) == 12 and len(searches["old"]) == len(searches["new"]) == 2
    assert row["new_reference_calls"] == 2 and row["new_reference_oracle_calls"] == 0


@pytest.mark.parametrize("field,value", (("state_id", "wrong saved id"), ("queue", [9, 9, 9]),
    ("source", "confirmation"), ("caps", [2, 4, 144, 256]), ("borrowed_index", 1)))
def test_control_source_mismatch_blocks_every_reference_and_method(monkeypatch, field, value):
    model, cell, protocol, runtime, deps, events, _, cold, references, oracles, borrowed = call_harness(monkeypatch)
    registered = cell["states"][1]
    cell["states"] = [registered]
    if field == "borrowed_index":
        registered[field] = value
    else:
        borrowed[0][field] = value
    with pytest.raises(ValueError, match="borrowed control state, source or native ladder changed"):
        runner.run_cell(model, cell, protocol, runtime, deps, progress=False)
    assert not events and not cold and not references and not oracles


def test_registration_excludes_all_observed_queues_and_retains_eight_paired_control_sources(monkeypatch, tmp_path):
    models = models_fixture()
    common = {"models": [model.snapshot() for model in models],
        "initial_lp_budget": 144, "lp_node_budget": 256, "integer_state_budget": 8192,
        "execution_sources": {"kernel.py": "frozen native kernel\n", "residual.py": "frozen residual proof\n"}}
    cells = control_cells(models)
    for cell in cells:
        for state in cell["states"]:
            state.update(borrowed_from="earlier development", borrowed_index=99,
                source_state_id="stale pointer", source_record_source="earlier source")
        cell["states"].insert(0, {"state_id": "ordinary", "queue": [17, 18, 19],
            "source": "development", "caps": [2, 3, 144, 256]})
    fixed = [{"node": model.node, "lanes": 64, "queue": [54, 58, 290]} for model in models]
    axis_fixed = [{"node": model.node, "lanes": 64, "queue": [120, 123, 258]} for model in models]
    development = {**common, "registered_at_utc": "residual developed", "reference_budget": 4096,
        "oracle_state_budget": 5000000, "cells": cells, "failure_development_keys": fixed,
        "axis_failure_development_keys": axis_fixed}
    previous = {"decision": "RESIDUAL_CAPACITY_FRONTIER_DEVELOPMENT_ADVANCE",
        "execution_checks_passed": True, "effective_references_complete": True, "development_gate_passed": True,
        "prior_decisions": {"frontier_trigger_validation": "FRONTIER_TRIGGER_VALIDATION_HOLD",
            "branch_reserved_timing": "BRANCH_RESERVED_TIMING_HOLD"}}
    earlier = {"decision": "FRONTIER_TRIGGER_VALIDATION_HOLD", "execution_checks_passed": True}
    screened = {**common, "registered_at_utc": "earlier native discovery",
        "excluded_static_queues": {"8": [[1, 2, 3]], "64": [[4, 5, 6]]},
        "queue_panels": [{"seed": 2026092851, "states": [{"queue": [91, 2, 3]}, {"queue": [94, 4, 6]}]}]}
    confirmed = {"excluded_static_queues": {"8": [[11, 12, 13]], "64": [[14, 15, 16]]},
        "cells": [{"node": model.node, "lanes": lanes, "states": [{"queue": [25 + index, lanes, 3]}]}
                  for index, model in enumerate(models) for lanes in (8, 64)]}
    previous_path, discovery_path, confirmation_path = (tmp_path / name for name in
        ("residual_development.json", "earlier_discovery.json", "confirmation_protocol.json"))
    for path, result, protocol in ((previous_path, previous, development), (discovery_path, earlier, screened)):
        path.write_text(json.dumps(result))
        path.with_name(path.stem + "_protocol.json").write_text(json.dumps(protocol))
    confirmation_path.write_text(json.dumps(confirmed))
    for path, literal in common["execution_sources"].items():
        (tmp_path / path).write_text(literal)
    (tmp_path / "new_runner.py").write_text("new trigger confirmation runner\n")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "PREVIOUS_OUTPUT", previous_path)
    monkeypatch.setattr(runner, "DISCOVERY_OUTPUT", discovery_path)
    monkeypatch.setattr(runner, "CONFIRMATION_PROTOCOL", confirmation_path)
    monkeypatch.setattr(runner, "SOURCES", ("new_runner.py",))

    protocol = runner.build_protocol(models)
    excluded = set(map(tuple, protocol["excluded_static_queues"]["64"]))
    assert {(1, 2, 3), (4, 5, 6), (11, 12, 13), (14, 15, 16), (91, 2, 3), (94, 4, 6)} <= excluded
    assert {tuple(state["queue"]) for source in (development, confirmed) for cell in source["cells"]
            for state in cell["states"]} <= excluded
    fresh = [tuple(state["queue"]) for panel in protocol["queue_panels"] for state in panel["states"]]
    assert len(fresh) == len(set(fresh)) == 4096 and set(fresh).isdisjoint(excluded)
    assert protocol["panel_seeds"] == [2026092861, 2026092862, 2026092863, 2026092864]
    assert (protocol["lanes"], protocol["queues_per_panel"], protocol["block_size"], protocol["max_rounds"],
        protocol["min_trigger_queues"], protocol["min_trigger_seeds"], protocol["expected_max_screen_calls"]) == (64, 1024, 64, 16, 2, 2, 16384)
    assert protocol["prior_decisions"] == {**previous["prior_decisions"],
        "residual_capacity_development": previous["decision"]}
    assert protocol["execution_sources"] == {**common["execution_sources"], "new_runner.py": "new trigger confirmation runner\n"}
    template = protocol["validation_template"]
    assert template["reference_source"] == {"path": "residual_development.json", "registered_at_utc": "residual developed"}
    assert template["failure_development_keys"] == fixed and template["axis_failure_development_keys"] == axis_fixed
    assert (template["initial_lp_budget"], template["lp_node_budget"], template["integer_state_budget"],
        template["reference_budget"], template["oracle_state_budget"]) == (144, 256, 8192, 4096, 5000000)
    assert len(template["control_cells"]) == 4 and sum(len(cell["states"]) for cell in template["control_cells"]) == 8
    for cell, original in zip(template["control_cells"], cells):
        assert (cell["node"], cell["lanes"]) == (original["node"], 64)
        for state, index in zip(cell["states"], (1, 2)):
            saved = original["states"][index]
            assert (state["borrowed_from"], state["borrowed_index"], state["source_state_id"], state["source_record_source"]) == (
                "residual_capacity_development", index, saved["state_id"], saved["source"])
            assert state["source_index"] == 99 and state["queue"] == saved["queue"] and state["caps"] == saved["caps"]

    (tmp_path / "residual.py").write_text("changed residual proof\n")
    with pytest.raises(ValueError, match="frozen source changed: residual.py"):
        runner.build_protocol(models)
    (tmp_path / "residual.py").write_text(common["execution_sources"]["residual.py"])
    development["cells"][-1]["states"].pop()
    previous_path.with_name(previous_path.stem + "_protocol.json").write_text(json.dumps(development))
    with pytest.raises(ValueError, match="all previous paired failure controls must be retained"):
        runner.build_protocol(models)
