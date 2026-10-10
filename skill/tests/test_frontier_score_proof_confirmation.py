"""Fresh and borrowed references for the frozen frontier-proof confirmation."""
from dataclasses import dataclass, fields
from fractions import Fraction
import json
from types import SimpleNamespace

import pytest

from algorithm.canonical_count_space_bnb_frame_candidates import CanonicalCountSpaceBnbSelection
from algorithm.experiments import aligned_hardware_frontier_score_proof_confirmation as runner
from algorithm.experiments.aligned_hardware_shared_threshold_queue_confirmation import sample_queues


def test_fresh_panels_preserve_sampling_law_seeds_and_disjointness():
    excluded = {str(lanes): {(lanes + 1, lanes + 2, lanes + 3)} for lanes in (8, 16, 32, 64)}
    before = {key: values.copy() for key, values in excluded.items()}
    panels = runner.fresh_design(excluded)
    for lanes in (8, 16, 32, 64):
        seen = excluded[str(lanes)].copy()
        assert len(panels[str(lanes)]) == 4
        for panel, seed in zip(panels[str(lanes)], (2026092841, 2026092842, 2026092843, 2026092844)):
            queues = tuple(tuple(state["queue"]) for state in panel["states"])
            assert (panel["seed"], panel["panel"], panel["cohort"]) == (seed, f"fresh_{seed}", "fresh_confirmation")
            assert queues == sample_queues(lanes, seed, 64, excluded=seen)
            assert len(set(queues)) == 64 and set(queues).isdisjoint(seen)
            assert [state["state_id"] for state in panel["states"]] == [f"fresh:{seed}:{index}" for index in range(64)]
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
    borrowed = [{"state_id": f"saved{index}", "queue": queue, "source": source,
        "historical_regression": index == 0,
        "reference": {"independent": index}, "reference_metrics": {"borrowed": index},
        "reference_complete": False, "oracle": {"borrowed_oracle": index},
        "effective_reference_complete": True, "effective_reference_source": f"borrowed source {index}",
        "old_routed": {"label": "stale old routed"}, "routed": {"label": "stale new routed"}}
        for index, (queue, source) in enumerate((([0, 1, 0], "confirmation"),
            ([54, 58, 290], "failure_development")))]
    states = [{"state_id": "fresh", "queue": [1, 0, 0], "source": "fresh_confirmation"},
        *[{"state_id": f"regression:{saved['state_id']}", "source_state_id": saved["state_id"],
           "source_index": index, "source_record_source": saved["source"], "queue": saved["queue"],
           "source": "historical_regression" if index == 0 else "failure_regression"}
          for index, saved in enumerate(borrowed)],
        {"state_id": "zero", "queue": [0, 0, 0], "source": "new_zero"}]
    for state in states:
        state.update(caps=[2, 3, 144, 256], historical_regression=state["source"] == "historical_regression",
            panel=state["source"], cohort=state["source"], seed=None)
    protocol = {"registered_at_utc": "new", "initial_lp_budget": 144, "lp_node_budget": 256,
        "integer_state_budget": 8192, "reference_budget": 4096, "oracle_state_budget": 5000000,
        "reference_source": {"path": "frontier_development.json", "registered_at_utc": "developed"}}
    events, searches, references, oracles = [], {"old": [], "new": []}, [], []

    def search_class(label):
        class Search:
            def __init__(self, options, queue, lanes):
                assert options is model.options and lanes == 2
                self.queue, self.caps = queue, []
                searches[label].append(self)
                events.append((label + " search", queue, None))

            def advance_to(self, cap):
                assert cap == (2, 3, 144, 256)[len(self.caps)]
                self.caps.append(cap)
                events.append((label + " advance", self.queue, cap))
                return MethodResult(label + " trace", self.queue, cap, not incomplete or not any(self.queue))
        return Search

    def selector(label):
        def run(options, queue, lanes, initial, total, integer):
            assert options is model.options and (lanes, initial, total, integer) == (2, 144, 256, 8192)
            events.append((label + " routed", queue, total))
            return MethodResult(label + " routed", queue, total, not incomplete or not any(queue))
        return run

    def original_catalogue_reference(options, queue, lanes, budget):
        assert options is model.options and (lanes, budget) == (2, 4096)
        events.append(("reference", queue, budget))
        references.append(queue)
        return full_reference(queue, score_exact=not incomplete)

    def oracle(options, queue, lanes, proven_score, *, state_visit_budget, maximum_duration):
        assert options is model.options and queue == (1, 0, 0) and lanes == 2
        assert (proven_score, state_visit_budget, maximum_duration) == (Fraction(1), 5000000, 1.0)
        assert not any(event[1] == queue and event[0].split()[0] in ("old", "new") for event in events)
        events.append(("oracle", queue, state_visit_budget))
        oracles.append(proven_score)
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
        assert (path, node, lanes, registration) == (runner.ROOT / "frontier_development.json", "toy", 2, "developed")
        return {"states": borrowed}

    monkeypatch.setattr(runner, "load_cell", load)
    monkeypatch.setattr(runner, "runtime_provenance", lambda: {"cpu_affinity": [0]})
    return model, {"node": "toy", "lanes": 2, "states": states}, protocol, runtime, dependencies, events, searches, references, oracles, borrowed


def test_independent_references_precede_both_native_ladders_and_cold_routers(monkeypatch):
    model, cell, protocol, runtime, dependencies, events, searches, references, oracles, borrowed = call_harness(monkeypatch)
    row = runner.run_cell(model, cell, protocol, runtime, dependencies, progress=False)
    assert references == [(1, 0, 0), (0, 0, 0)] and oracles == [Fraction(1)]
    assert row["new_reference_calls"] == 2 and row["new_reference_oracle_calls"] == 1
    assert row["new_old_routed_calls"] == 4
    for label in ("old", "new"):
        assert len(searches[label]) == 4 and all(search.caps == [2, 3, 144, 256] for search in searches[label])
    for state in row["states"]:
        selected = [event[0] for event in events if event[1] == tuple(state["queue"])]
        if state["source"] not in ("historical_regression", "failure_regression"):
            assert selected[0] == "reference"
        assert selected.count("old search") == selected.count("new search") == 1
        assert selected.count("old advance") == selected.count("new advance") == 4
        assert selected.count("old routed") == selected.count("new routed") == 1
        assert selected.index("new advance") < selected.index("old routed") < selected.index("new routed")
        assert list(state["traces"]["old"]) == list(state["traces"]["new"]) == ["2", "3", "144", "256"]
        assert state["old_routed"]["label"] == "old routed" and state["routed"]["label"] == "new routed"
    for state, saved in zip(row["states"][1:3], borrowed):
        for key in ("reference", "reference_metrics", "reference_complete", "oracle",
                    "effective_reference_complete", "effective_reference_source"):
            assert state[key] == saved[key]
    assert row["states"][1]["historical_regression"] is True and row["states"][2]["historical_regression"] is False
    assert row["states"][0]["reference"]["score"] == 1.0 and row["states"][0]["routed"]["score"] == 999.0
    assert row["states"][0]["effective_reference_source"] == "independent_score_face_oracle"
    assert row["states"][3]["effective_reference_source"] == "full_K4096"
    assert row["runtime_start"] == row["runtime_end"] == {"cpu_affinity": [0]}


def test_incomplete_reference_and_outcomes_are_preserved_while_later_states_execute(monkeypatch):
    model, cell, protocol, runtime, dependencies, _, searches, references, oracles, _ = call_harness(monkeypatch, incomplete=True)
    row = runner.run_cell(model, cell, protocol, runtime, dependencies, progress=False)
    assert len(row["states"]) == 4 and references == [(1, 0, 0), (0, 0, 0)] and not oracles
    first = row["states"][0]
    assert first["reference_complete"] is False and first["effective_reference_complete"] is False
    assert first["oracle"] is None and first["reference"]["score_exact"] is False
    assert first["effective_reference_source"] is None
    assert [state["routed"]["canonical_exact"] for state in row["states"]] == [False, False, False, True]
    assert row["new_old_routed_calls"] == 4 and len(searches["old"]) == len(searches["new"]) == 4


@pytest.mark.parametrize("index,field,value", ((0, "state_id", "wrong saved id"),
    (0, "queue", [9, 9, 9]), (1, "source", "confirmation"), (1, "source_index", 0)))
def test_borrowed_reference_identity_mismatch_precedes_every_method(monkeypatch, index, field, value):
    model, cell, protocol, runtime, dependencies, events, _, references, oracles, borrowed = call_harness(monkeypatch)
    registered = cell["states"][index + 1]
    cell["states"] = [registered]
    if field == "source_index":
        registered[field] = value
    else:
        borrowed[index][field] = value
    with pytest.raises(ValueError, match="borrowed historical or failure reference changed"):
        runner.run_cell(model, cell, protocol, runtime, dependencies, progress=False)
    assert not events and not references and not oracles


def test_registration_excludes_all_prior_queues_and_binds_both_borrowed_cohorts(monkeypatch, tmp_path):
    options = (Option((1, 0, 0), 1.0), Option((0, 1, 0), 2.0))
    models = [SimpleNamespace(node=f"node{index}", options=options,
        snapshot=lambda index=index: {"node": f"node{index}", "physical_lanes": 2}) for index in range(4)]
    budgets = {"initial_lp_budget": 144, "lp_node_budget": 256, "integer_state_budget": 8192}
    tails = {8: ((27, 15, 34),), 16: (), 32: ((96, 15, 13),),
             64: ((88, 19, 118), (184, 78, 235))}
    development_cells, previous_cells, historical, failures = [], [], [], []
    for model in models:
        for lanes in (8, 16, 32, 64):
            states = [{"state_id": "ordinary", "queue": [7, 8, 9], "source": "development",
                "cohort": "development", "seed": 27, "caps": [2, 3, 144, 256], "historical_regression": False}]
            regressions = [("trigger", (lanes, 2, 3), True, False)]
            if (model.node, lanes) in (("node0", 8), ("node1", 16)):
                regressions.append(("extra_trigger", (lanes, 4, 5), True, False))
            regressions.extend((f"tail{index}", queue, False, True) for index, queue in enumerate(tails[lanes]))
            for index, (name, queue, trigger, tail) in enumerate(regressions):
                historical.append({"node": model.node, "lanes": lanes, "queue": list(queue)})
                states.append({"state_id": f"{model.node}:{lanes}:{name}", "queue": list(queue),
                    "source": "confirmation" if index % 2 else "development", "cohort": "reused_confirmation",
                    "seed": None, "caps": [2, 3, 144, 256], "historical_regression": True,
                    "historical_trigger": trigger, "ordinary_tail": tail,
                    "source_index": 99, "source_state_id": "older source id"})
            if lanes == 64:
                queue = [54, 58, 290]
                failures.append({"node": model.node, "lanes": lanes, "queue": queue})
                states.append({"state_id": f"failure_development:{model.node}", "queue": queue,
                    "source": "failure_development", "cohort": "failure_development", "seed": 2026092831,
                    "caps": [2, 3, 144, 256], "historical_regression": False})
            states.append({"state_id": "zero", "queue": [0, 0, 0], "source": "new_zero",
                "cohort": "zero_queue", "seed": None, "caps": [2, 3, 144, 256], "historical_regression": False})
            development_cells.append({"node": model.node, "lanes": lanes, "states": states})
            previous_cells.append({"node": model.node, "lanes": lanes, "states": [
                {"state_id": "fresh:2026092831:0", "queue": [lanes, 9, 10], "source": "fresh_confirmation"},
                {"state_id": "zero", "queue": [0, 0, 0], "source": "new_zero"}]})
    common = {**budgets, "models": [model.snapshot() for model in models],
        "execution_sources": {"kernel.py": "frozen kernel\n"}}
    development = {**common, "registered_at_utc": "frontier development", "cells": development_cells,
        "reference_budget": 4096, "oracle_state_budget": 5000000,
        "historical_regression_keys": historical, "failure_development_keys": failures,
        "prior_decisions": {"late_score_proof_development": "LATE_SCORE_PROOF_DEVELOPMENT_HOLD",
                            "branch_reserved_timing": "BRANCH_RESERVED_TIMING_HOLD"},
        "rule": "Only exported paid frontier boxes share the fixed 8192 integer budget."}
    previous = {**common, "registered_at_utc": "dual confirmation", "cells": previous_cells,
        "excluded_static_queues": {str(lanes): [[1, 2, 3], [5, 6, 7]] for lanes in (8, 16, 32, 64)}}
    prior_result = {"decision": "FRONTIER_SCORE_PROOF_DEVELOPMENT_ADVANCE", "development_gate_passed": True,
        "execution_checks_passed": True, "effective_references_complete": True, "failure_development_gate_passed": True}
    previous_result = {"decision": "DUAL_REUSE_FRESH_CONFIRMATION_HOLD", "execution_checks_passed": True,
        "effective_references_complete": True, "final_score_gate_passed": True, "work_gate_passed": True}
    prior_output, previous_output = tmp_path / "development.json", tmp_path / "previous.json"
    for path, result, protocol in ((prior_output, prior_result, development), (previous_output, previous_result, previous)):
        path.write_text(json.dumps(result))
        path.with_name(path.stem + "_protocol.json").write_text(json.dumps(protocol))
    (tmp_path / "kernel.py").write_text("frozen kernel\n")
    (tmp_path / "new_runner.py").write_text("new runner\n")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "PRIOR_OUTPUT", prior_output)
    monkeypatch.setattr(runner, "PREVIOUS_OUTPUT", previous_output)
    monkeypatch.setattr(runner, "SOURCES", ("new_runner.py",))

    protocol = runner.build_protocol(models)
    assert len(protocol["cells"]) == 16
    states = [state for cell in protocol["cells"] for state in cell["states"]]
    assert len(states) == protocol["expected_state_count"] == 4150
    assert sum(state["source"] == "fresh_confirmation" for state in states) == 4096
    assert sum(state["source"] == "historical_regression" for state in states) == 34
    assert sum(state["source"] == "failure_regression" for state in states) == 4
    assert sum(state["source"] == "new_zero" for state in states) == 16
    assert protocol["expected_new_reference_calls"] == 4112
    assert protocol["expected_interface_calls"] == 41500 and protocol["expected_native_constructions"] == 16600
    assert protocol["expected_selector_calls"] == 8300 and protocol["expected_native_advance_calls"] == 33200
    assert protocol["reference_source"] == {"path": "development.json", "registered_at_utc": "frontier development"}
    assert protocol["historical_regression_keys"] == historical and protocol["failure_regression_keys"] == failures
    assert protocol["prior_decisions"] == {**development["prior_decisions"],
        "frontier_score_proof_development": "FRONTIER_SCORE_PROOF_DEVELOPMENT_ADVANCE",
        "dual_reuse_confirmation": "DUAL_REUSE_FRESH_CONFIRMATION_HOLD"}
    assert protocol["execution_sources"] == {"kernel.py": "frozen kernel\n", "new_runner.py": "new runner\n"}
    old_cells = {(cell["node"], cell["lanes"]): cell for cell in development_cells}
    for lanes in (8, 16, 32, 64):
        excluded = set(map(tuple, protocol["excluded_static_queues"][str(lanes)]))
        assert {(1, 2, 3), (5, 6, 7)} <= excluded
        assert {tuple(state["queue"]) for cell in [*previous_cells, *development_cells] if cell["lanes"] == lanes
                for state in cell["states"]} <= excluded
        fresh = [tuple(state["queue"]) for panel in protocol["queue_panels"][str(lanes)] for state in panel["states"]]
        assert len(set(fresh)) == 256 and set(fresh).isdisjoint(excluded)
        for cell in (cell for cell in protocol["cells"] if cell["lanes"] == lanes):
            assert [tuple(state["queue"]) for state in cell["states"] if state["source"] == "fresh_confirmation"] == fresh
    for cell in protocol["cells"]:
        for state in cell["states"]:
            assert state["caps"] == [2, 3, 144, 256]
            if state["source"] in ("historical_regression", "failure_regression"):
                old = old_cells[cell["node"], cell["lanes"]]["states"][state["source_index"]]
                assert (state["source_state_id"], state["queue"], state["source_record_source"]) == (
                    old["state_id"], old["queue"], old["source"])
                assert state["historical_regression"] is (state["source"] == "historical_regression")
                assert (state["seed"], state["historical_regression"]) == (old["seed"], old["historical_regression"])
    runner.verify_protocol({**protocol, "registered_at_utc": "new confirmation"}, models)
    previous_output.write_text(json.dumps({**previous_result, "decision": "DUAL_REUSE_FRESH_CONFIRMATION_ADVANCE"}))
    with pytest.raises(ValueError, match="preceding independent confirmation HOLD"):
        runner.build_protocol(models)
    previous_output.write_text(json.dumps(previous_result))
    (tmp_path / "kernel.py").write_text("changed kernel\n")
    with pytest.raises(ValueError, match="frozen source changed"):
        runner.build_protocol(models)
