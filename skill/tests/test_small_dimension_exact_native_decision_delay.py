"""Freeze the main-scenario tapes and keep policy timing ahead of all audits."""
from collections import Counter
from dataclasses import dataclass, replace
import json

import pytest

from algorithm.experiments import aligned_hardware_small_dimension_exact_native_decision_delay as experiment


@dataclass(frozen=True)
class Model:
    node: str
    lanes: int = 2

    def snapshot(self):
        return {"node": self.node, "lanes": self.lanes}


@pytest.fixture
def registered_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "ROOT", tmp_path)
    outputs = {name: tmp_path / f"{name}.json" for name in (
        "QUALITY_OUTPUT", "TIMING_OUTPUT", "TAPE_OUTPUT")}
    for name, path in outputs.items():
        monkeypatch.setattr(experiment, name, path)
    models = [Model(f"catalogue{i}") for i in range(4)]
    source_path = "whole_selector.py"
    (tmp_path / source_path).write_text("native source\n")
    monkeypatch.setattr(experiment, "SOURCES", (source_path,))
    snapshots = [model.snapshot() for model in models]
    budgets = dict(initial_lp_budget=144, lp_node_budget=256,
                   integer_state_budget=8192, reference_budget=4096, oracle_state_budget=5_000_000)
    quality = {key: True for key in ("execution_checks_passed", "references_complete",
        "quality_gate_passed", "work_gate_passed", "confirmation_gate_passed")}
    quality["decision"] = "SMALL_DIMENSION_EXACT_NATIVE_CONFIRMATION_ADVANCE"
    timing = {key: True for key in ("execution_checks_passed", "certificate_checks_passed",
        "references_complete", "quality_gate_passed", "repeat_paths_stable",
        "cpu_affinity_preserved", "local_timing_gate_passed", "new_independent_timing_gate_passed",
        "independent_gain_gate_passed")}
    timing.update(decision="SMALL_DIMENSION_EXACT_NATIVE_INDEPENDENT_TIMING_ADVANCE",
                  prior_decisions={"old_utility": "UTILITY_HOLD"})
    tapes = [{"tape_id": f"{model.node}:R8:load{load:g}:seed{seed}", "node": model.node,
        "lanes": 8, "load": load, "seed": seed,
        "arrivals": [[0.0, 0, 0], [0.5, 1, 0]]}
        for model in models for load in experiment.LOADS for seed in experiment.ARRIVAL_SEEDS]
    old = dict(models=snapshots, lanes=[8], loads=list(experiment.LOADS),
        arrival_seeds=list(experiment.ARRIVAL_SEEDS), arrival_horizon=600., observation_end=1200.,
        tapes=tapes, capacities=[{"certificate": "fixed service rates"}],
        execution_sources={source_path: "native source\n"})
    inputs = {outputs["QUALITY_OUTPUT"]: quality, outputs["TIMING_OUTPUT"]: timing,
        outputs["TAPE_OUTPUT"]: {"decision": "INTEGER_CAPACITY_DECISION_DELAY_UTILITY_HOLD"},
        experiment._protocol_path(outputs["QUALITY_OUTPUT"]): {**budgets, "models": snapshots,
            "execution_sources": {source_path: "native source\n"}},
        experiment._protocol_path(outputs["TAPE_OUTPUT"]): old}
    for path, content in inputs.items():
        path.write_text(json.dumps(content))
    return models, inputs


def test_main_protocol_reuses_all_old_tapes_and_whole_budgets(registered_inputs):
    models, inputs = registered_inputs
    protocol = experiment.build_protocol(models)
    previous = inputs[experiment._protocol_path(experiment.TAPE_OUTPUT)]
    assert protocol["tapes"] == previous["tapes"]
    assert protocol["capacities"] == previous["capacities"]
    assert len(protocol["tapes"]) == 24
    assert (protocol["expected_zero_trajectory_count"], protocol["expected_measured_trajectory_count"],
            protocol["expected_trajectory_count"]) == (72, 288, 360)
    assert [protocol[key] for key in ("initial_lp_budget", "lp_node_budget", "integer_state_budget")] == [144, 256, 8192]
    assert protocol["minimum_pooled_flow_improvement"] == .01
    assert protocol["prior_decisions"]["old_utility"] == "UTILITY_HOLD"
    assert protocol["prior_decisions"]["integer_capacity_decision_delay"].endswith("UTILITY_HOLD")


def test_each_tape_has_exactly_three_of_each_whole_call_order(registered_inputs):
    models, _ = registered_inputs
    protocol = experiment.build_protocol(models)
    for tape, orders in protocol["timing_orders"].items():
        assert orders == experiment.timing_orders(tape)
        assert Counter(map(tuple, orders)) == {("candidate", "direct"): 3, ("direct", "candidate"): 3}


@pytest.mark.parametrize("change", ("source", "models", "tape", "budget", "prior_quality", "prior_timing"))
def test_source_roster_and_prerequisite_changes_stop_registration(registered_inputs, change):
    models, inputs = registered_inputs
    if change == "source":
        (experiment.ROOT / "whole_selector.py").write_text("changed native source\n")
    elif change == "models":
        models[0] = replace(models[0], lanes=3)
    elif change == "tape":
        path = experiment._protocol_path(experiment.TAPE_OUTPUT)
        inputs[path]["tapes"].pop()
        path.write_text(json.dumps(inputs[path]))
    elif change == "budget":
        path = experiment._protocol_path(experiment.QUALITY_OUTPUT)
        inputs[path]["integer_state_budget"] += 1
        path.write_text(json.dumps(inputs[path]))
    else:
        path = experiment.QUALITY_OUTPUT if change == "prior_quality" else experiment.TIMING_OUTPUT
        inputs[path]["quality_gate_passed"] = False
        path.write_text(json.dumps(inputs[path]))
    with pytest.raises(ValueError):
        experiment.build_protocol(models)


def test_registered_order_is_checked_before_execution(registered_inputs):
    models, _ = registered_inputs
    protocol = experiment.build_protocol(models)
    protocol["registered_at_utc"] = "before policies"
    experiment.verify_protocol(protocol, models)
    key = next(iter(protocol["timing_orders"]))
    protocol["timing_orders"][key][0] = list(reversed(protocol["timing_orders"][key][0]))
    with pytest.raises(ValueError, match="registered"):
        experiment.verify_protocol(protocol, models)


def test_audit_is_deferred_until_all_six_measured_rounds(registered_inputs):
    models, _ = registered_inputs
    protocol = experiment.build_protocol(models)
    protocol["tapes"] = [protocol["tapes"][0]]
    timeline = []

    def simulate(model, tape, horizon, cutoff, budget, selector, **kwargs):
        assert model.lanes == 8
        assert tape == protocol["tapes"][0]["arrivals"]
        assert (horizon, cutoff) == (600., 1200.)
        label = kwargs["policy"]
        assert budget == (4096 if label == "full_reference" else 256)
        if "clock" in kwargs:
            assert kwargs["clock"]() == kwargs["clock"]() == 0
        timeline.append(("simulate", label, "clock" in kwargs))
        return {"policy": label}, [object()]

    class Reference:
        def __init__(self, model, registered):
            assert len(timeline) == 15
            timeline.append(("reference",))

        def summary(self):
            return {"calls": 0}

    def audit(model, result, original, reference, registered):
        assert len(original) == 1
        assert len([entry for entry in timeline if entry[0] == "simulate"]) == 15
        timeline.append(("audit", result["policy"]))
        result["selector_metrics"] = {"calls": 1}

    dependencies = dict(simulate=simulate, reference=Reference, audit=audit,
        selectors=lambda protocol: dict.fromkeys(experiment.ZERO_LABELS, object()), provenance=lambda: {})
    row = experiment.run_cell(models[0], 8, protocol, dependencies=dependencies, progress=False)
    assert [entry[1] for entry in timeline[:3]] == list(experiment.ZERO_LABELS)
    assert all(entry[2] for entry in timeline[:3])
    assert all(not entry[2] for entry in timeline[3:15])
    assert timeline[15] == ("reference",)
    assert len([entry for entry in timeline if entry[0] == "audit"]) == 15
    run = row["runs"][0]
    assert len(run["measured"]) == 6
    assert run["reference_audit"] == {"calls": 0}
    assert all("selector_metrics" in result for result in run["zero"].values())


def test_run_cell_uses_only_its_registered_node_and_lanes(registered_inputs):
    models, _ = registered_inputs
    protocol = experiment.build_protocol(models)
    protocol["tapes"] = [protocol["tapes"][-1]]
    def unused(*args, **kwargs):
        raise AssertionError("a different node's tape was executed")
    dependencies = dict(simulate=unused, reference=unused, audit=unused,
        selectors=lambda protocol: {}, provenance=lambda: {})
    row = experiment.run_cell(models[0], 8, protocol, dependencies=dependencies, progress=False)
    assert row["runs"] == []
