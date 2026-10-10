"""Fresh references precede methods; old controls never become fresh calls."""
from dataclasses import dataclass
from copy import deepcopy
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from algorithm.experiments import aligned_hardware_affine_dual_fresh_confirmation as runner


@dataclass(frozen=True)
class Record:
    label: str
    cap: int


def harness(monkeypatch, *, complete=True, independent=None):
    model = SimpleNamespace(node="toy", lanes=4, options=())
    fresh = {"state_id": "fresh", "queue": [3, 1], "caps": [2, 3, 144, 256], "panel": "fresh",
             "confirmation_source": "fresh", "timing_group": "fresh_confirmation"}
    old = {"state_id": "control", "queue": [2, 1], "caps": [2, 3, 144, 256],
           "source": "fresh_confirmation", "timing_group": "new_trigger"}
    saved = {**old, **{key: {"saved": key} for key in runner.REFERENCE_FIELDS},
             "traces": {"old": {"branch": True}, "new": {"affine": True}},
             "prior_routed": {"real_axis": True},
             "first_results": {"branch": {"old": True}, "candidate": {"new": True}}}
    control = {**old, "confirmation_source": "borrowed_control", "confirmation_source_index": 0,
               "confirmation_source_state_id": "control"}
    protocol = {"registered_at_utc": "new", "initial_lp_budget": 144, "lp_node_budget": 256,
                "integer_state_budget": 8192, "reference_source": {"path": "old.json", "registered_at_utc": "old"}}
    events = []

    def reference(*args):
        assert args[0] is model and args[1:3] == ((3, 1), 8)
        events.append("reference")
        return Record("reference", 4096), {"independent": True}, complete, independent

    def search(label):
        class Search:
            def __init__(self, options, queue, lanes):
                assert options is model.options and (queue, lanes) == ((3, 1), 8)
                events.append((label, "construct"))

            def advance_to(self, cap):
                events.append((label, cap))
                return Record(label, cap)
        return Search

    def selector(label):
        def call(options, queue, lanes, *budgets):
            assert options is model.options and (queue, lanes, budgets) == ((3, 1), 8, (144, 256, 8192))
            events.append((label, "selector"))
            return Record(label, 256)
        return call

    def load(path, node, lanes, stamp):
        assert path == runner.ROOT / "old.json" and (node, lanes, stamp) == ("toy", 8, "old")
        return {"states": [saved]}

    monkeypatch.setattr(runner, "load_cell", load)
    monkeypatch.setattr(runner, "runtime_provenance", lambda: {"cpu_affinity": [0], "gc_enabled": True})
    runtime = SimpleNamespace(reference_key=lambda *args: "ref-key" if complete else None)
    dependencies = {"searches": {label: search(label) for label in ("old", "new")},
                    "selectors": {label: selector(label) for label in ("old", "new")},
                    "new_reference": reference, "oracle": object(), "oracle_key": lambda record, n: "oracle-key" if record["complete"] else None}
    return model, {"lanes": 8, "states": [fresh, control]}, protocol, runtime, dependencies, saved, events


def test_reference_first_then_two_cold_ladders_and_routers_borrow_old_trigger(monkeypatch):
    model, cell, protocol, runtime, dependencies, saved, events = harness(monkeypatch)
    row = runner.run_cell(model, cell, protocol, runtime, dependencies, progress=True)
    assert events == ["reference", *[(label, phase) for label in ("old", "new") for phase in ("construct", 2, 3, 144, 256)],
                      ("old", "selector"), ("new", "selector")]
    assert (row["new_reference_calls"], row["new_oracle_calls"], row["candidate_calls"], row["branch_calls"],
            row["native_advance_calls"], row["native_search_constructions"]) == (1, 0, 1, 1, 8, 4)
    fresh, control = row["states"]
    assert "prior_routed" not in fresh and fresh["effective_reference_source"] == "full_K4096"
    assert control["traces"] == saved["traces"] and control["prior_routed"] == saved["prior_routed"]
    assert control["routed"] == saved["first_results"]["candidate"]
    assert control["old_routed"] == saved["first_results"]["branch"]


@pytest.mark.parametrize("independent,expected,kind,calls", (
    (None, False, None, 0),
    ({"complete": False}, False, None, 1),
    ({"complete": True}, True, "independent_score_face_oracle", 1),
))
def test_incomplete_references_are_retained_and_all_methods_still_run(monkeypatch, independent, expected, kind, calls):
    model, cell, protocol, runtime, dependencies, saved, events = harness(monkeypatch, complete=False, independent=independent)
    row = runner.run_cell(model, cell, protocol, runtime, dependencies, progress=False)
    assert row["states"][0]["effective_reference_complete"] is expected
    assert row["states"][0]["effective_reference_source"] == kind
    assert row["states"][0]["oracle"] == independent and row["new_oracle_calls"] == calls
    assert row["candidate_calls"] == row["branch_calls"] == 1 and len(row["states"]) == 2


@pytest.mark.parametrize("field,value", (("state_id", "wrong"), ("queue", [9, 9]), ("caps", [2, 4, 144, 256])))
def test_changed_control_blocks_borrowing_without_running_it(monkeypatch, field, value):
    model, cell, protocol, runtime, dependencies, saved, events = harness(monkeypatch)
    cell["states"] = cell["states"][1:]
    saved[field] = value
    with pytest.raises(ValueError, match="exact frozen state"):
        runner.run_cell(model, cell, protocol, runtime, dependencies, progress=False)
    assert not events


def test_fresh_design_passes_accumulating_exclusions_to_each_new_seed(monkeypatch):
    draws = []

    def sampler(lanes, seed, count, *, excluded):
        queue = (seed, lanes, 1)
        assert count == runner.QUEUES_PER_PANEL
        earlier = [previous for r, previous in draws if r == lanes]
        assert (9, 9, 9) in excluded and all(previous in excluded for previous in earlier)
        draws.append((lanes, queue))
        return (queue,)

    from algorithm.experiments import aligned_hardware_shared_threshold_queue_confirmation as sampling
    monkeypatch.setattr(sampling, "sample_queues", sampler)
    panels = runner.fresh_design({str(lanes): {(9, 9, 9)} for lanes in runner.LANES})
    assert len(draws) == len(runner.LANES) * len(runner.PANEL_SEEDS)
    assert all([panel["seed"] for panel in panels[str(lanes)]] == list(runner.PANEL_SEEDS) for lanes in runner.LANES)


def test_import_keeps_numerical_libraries_deferred_until_runtime_setup():
    result = subprocess.run([sys.executable, "-c",
        "from algorithm.experiments.aligned_hardware_affine_dual_fresh_confirmation import runtime_setup; runtime_setup()"],
        cwd=runner.ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_historical_queue_metadata_keeps_latest_repaired_execution_source(monkeypatch, tmp_path):
    path = runner.PREVIOUS_OUTPUT
    previous = json.loads(path.with_name(path.stem + "_protocol.json").read_text())
    exclusions = [json.loads((runner.ROOT / p).read_text()) for p in runner.EXCLUSION_PROTOCOLS]
    (tmp_path / "shared.py").write_text("repaired\n")
    for old in exclusions:
        old["execution_sources"] = {"shared.py": "original\n"}
    previous["execution_sources"] = {"shared.py": "repaired\n"}
    output = tmp_path / "old.json"
    output.with_name("old_protocol.json").write_text(json.dumps(previous))
    output.write_text(json.dumps({"decision": "AFFINE_DUAL_LOCAL_TIMING_ADVANCE", "prior_decisions": {},
        **{key: True for key in ("execution_checks_passed", "references_complete", "quality_gate_passed",
                                "repeat_paths_stable", "frozen_residual_results_reproduced", "local_timing_gate_passed")}}))
    for index, old in enumerate(exclusions):
        (tmp_path / f"exclude{index}.json").write_text(json.dumps(old))
    panels = {str(lanes): [{"seed": seed, "panel": f"fresh_{seed}", "states": [
        {"state_id": f"ordinary:{seed}:{i}", "queue": [lanes, seed_index + 1, i + 1]}
        for i in range(runner.QUEUES_PER_PANEL)]} for seed_index, seed in enumerate(runner.PANEL_SEEDS)]
        for lanes in runner.LANES}
    models = [SimpleNamespace(node=model["node"], snapshot=lambda saved=deepcopy(model): saved)
              for model in previous["models"]]
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "PREVIOUS_OUTPUT", output)
    monkeypatch.setattr(runner, "EXCLUSION_PROTOCOLS", ("exclude0.json", "exclude1.json"))
    monkeypatch.setattr(runner, "SOURCES", ())
    monkeypatch.setattr(runner, "_caps", lambda model: [1, 2, 144, 256])
    monkeypatch.setattr(runner, "fresh_design", lambda excluded: deepcopy(panels))
    protocol = runner.build_protocol(models)
    assert protocol["execution_sources"]["shared.py"] == "repaired\n"
    assert (protocol["expected_fresh_state_count"], protocol["expected_control_state_count"]) == (1024, 754)
    (tmp_path / "shared.py").write_text("unexpected\n")
    with pytest.raises(ValueError, match="frozen source changed"):
        runner.build_protocol(models)
