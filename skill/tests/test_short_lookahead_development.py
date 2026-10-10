"""Independent routing calls with frozen references and native snapshots."""
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from algorithm.experiments import aligned_hardware_short_lookahead_development as runner


@dataclass(frozen=True)
class Result:
    label: str
    canonical_exact: bool


def harness(monkeypatch):
    events = []
    saved = [{"state_id": f"dev:{index}", "queue": [index + 1, 2], "source": "development",
        "reference": {"saved": index}, "reference_metrics": {"calls": 1}, "reference_complete": index == 0,
        "oracle": None if index == 0 else {"saved": "oracle"}, "effective_reference_complete": True,
        "effective_reference_source": "full_K4096" if index == 0 else "independent_score_face_oracle",
        "traces": {"new": {"144": {"saved": "initial"}, "256": {"saved": "final"}}},
        "routed": {"saved": "old routing"}} for index in range(2)]
    states = [{"state_id": state["state_id"], "source_state_id": state["state_id"], "queue": state["queue"],
        "source": state["source"], "cohort": "development", "prior_index": index,
        "historical_regression": index == 1} for index, state in enumerate(saved)]
    cell = {"node": "toy", "lanes": 8, "states": states}
    protocol = {"registered_at_utc": "new registration", "initial_lp_budget": 144, "lp_node_budget": 256,
        "integer_state_budget": 8192, "prior_source": {"path": "prior.json", "registered_at_utc": "old registration"}}
    model = SimpleNamespace(node="toy", lanes=2, options=(object(), object()))

    def load(path, node, lanes, registration):
        assert (path, node, lanes, registration) == (runner.ROOT / "prior.json", "toy", 8, "old registration")
        events.append(("borrow",))
        return {"states": saved}

    def selector(label):
        def invoke(options, queue, lanes, *budgets):
            assert options is model.options and lanes == 8 and budgets == (144, 256, 8192)
            events.append((label, queue))
            return Result(label, not (label == "new" and queue == (1, 2)))
        return invoke

    def unexpected(*args, **kwargs):
        pytest.fail("development must not solve new references, oracles or budget-ladder kernels")

    from algorithm.experiments import aligned_hardware_strict_score_proof_development as reference
    from algorithm.experiments import integer_capacity_canonical_oracle as oracle
    from algorithm.branch_reserved_integer_capacity_frame_candidates import BranchReservedIntegerCapacitySearch
    monkeypatch.setattr(reference, "_new_reference", unexpected)
    monkeypatch.setattr(oracle, "canonical_at_certified_score", unexpected)
    monkeypatch.setattr(BranchReservedIntegerCapacitySearch, "__init__", unexpected)
    monkeypatch.setattr(runner, "load_cell", load)
    monkeypatch.setattr(runner, "runtime_provenance", lambda: {"cpu_affinity": [0], "gc_enabled": True})
    return model, cell, protocol, {label: selector(label) for label in ("old", "new")}, saved, events


def test_cold_arms_borrow_only_reference_and_independent_native_views(monkeypatch):
    model, cell, protocol, selectors, saved, events = harness(monkeypatch)
    row = runner.run_cell(model, cell, protocol, None, selectors, progress=False)
    assert events == [("borrow",), ("old", (1, 2)), ("new", (1, 2)), ("old", (2, 2)), ("new", (2, 2))]
    assert row["new_reference_calls"] == row["new_reference_oracle_calls"] == 0
    assert row["new_old_routed_calls"] == 2
    assert len(row["states"]) == 2
    for state, original in zip(row["states"], saved):
        for key in ("reference", "reference_metrics", "reference_complete", "oracle",
                    "effective_reference_complete", "effective_reference_source"):
            assert state[key] == original[key]
        assert state["expected_initial_lp"] == original["traces"]["new"]["144"]
        assert state["expected_native_final"] == original["traces"]["new"]["256"]
        assert state["expected_old_routed"] == original["routed"]
        assert state["old_routed"]["label"] == "old" and state["routed"]["label"] == "new"
        assert "traces" not in state
    assert row["states"][0]["routed"]["canonical_exact"] is False
    assert row["states"][1]["routed"]["canonical_exact"] is True


@pytest.mark.parametrize("field", ("prior_index", "queue"))
def test_wrong_source_binding_stops_before_routing_calls(monkeypatch, field):
    model, cell, protocol, selectors, saved, events = harness(monkeypatch)
    if field == "prior_index":
        cell["states"][0][field] = 1
    else:
        saved[0][field] = [9, 9]
    with pytest.raises(ValueError, match="borrowed development state"):
        runner.run_cell(model, cell, protocol, None, selectors, progress=False)
    assert events == [("borrow",)]
