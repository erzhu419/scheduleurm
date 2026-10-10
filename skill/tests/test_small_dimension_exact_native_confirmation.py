"""Independent references, fresh cold calls and unchanged borrowed controls."""
from copy import deepcopy
from dataclasses import dataclass
import json
from types import SimpleNamespace

import pytest

from algorithm.experiments import aligned_hardware_small_dimension_exact_native_confirmation as runner


@dataclass
class Stage:
    label: str
    cap: int
    kind: str = "native"


@dataclass
class Whole:
    label: str
    budgets: tuple
    root_settle_work: Stage | None = None


@dataclass
class Reference:
    exact: bool = True


def harness(monkeypatch, *, settled=True, complete=True, oracle=None):
    calls = []
    fresh = dict(state_id="ordinary:new:0", queue=[2, 3, 4], caps=[2, 3, 144, 256],
        seed=runner.PANEL_SEEDS[0], panel="independent:new", timing_group=runner.NEW_GROUP,
        source="small_dimension_exact_native_confirmation", cohort=runner.NEW_GROUP,
        confirmation_source="fresh", small_dimension_confirmation_source="fresh", historical_regression=False)
    saved = dict(state_id="ordinary:old:0", queue=[8, 9, 10], caps=[2, 3, 144, 256],
        source="independent_confirmation", cohort="independent_confirmation", timing_group="independent_fresh",
        independent_source="fresh", confirmation_source="fresh", panel="previous independent",
        routed={"actual_old_candidate": True}, candidate_native_root={"actual_old_root": True},
        candidate_native_traces={"actual_old_ladder": True}, reference={"actual_old_reference": True},
        prior_routed={"actual_axis": True})
    old = {key: deepcopy(value) for key, value in saved.items()
        if key not in ("routed", "candidate_native_root", "candidate_native_traces", "reference", "prior_routed")}
    old.update(small_dimension_confirmation_source="borrowed_control",
        small_dimension_confirmation_source_index=0, small_dimension_confirmation_source_state_id=saved["state_id"])
    model = SimpleNamespace(node="mocknode", lanes=2, options=("option",))
    cell = dict(node=model.node, lanes=8, states=[fresh, old])
    protocol = dict(registered_at_utc="new registration", reference_source={"path": "old.json", "registered_at_utc": "old"},
        initial_lp_budget=144, lp_node_budget=256, integer_state_budget=8192,
        long_lp_budget=4096, control_integer_state_budget=100_000)

    def reference(*args):
        calls.append("reference")
        return Reference(complete), {"actual_reference_metrics": True}, complete, oracle

    def search_type(label):
        class Search:
            def __init__(self, *_):
                calls.append((label, "construct"))

            def advance_to(self, cap):
                calls.append((label, "advance", cap))
                return Stage(label, cap)

            def advance_roots_and_settle(self, cap):
                calls.append((label, "root", cap))
                return Stage(label, cap, "raw"), Stage(label, cap, "settled") if settled else None, Stage(label, cap, "phase") if settled else None

        return Search

    def selector(label):
        def whole(*args):
            calls.append((label, "whole", args[3:]))
            return Whole(label, args[3:], Stage(label, 2, "phase") if settled and label in runner.ROOT_LABELS else None)
        return whole

    deps = {"new_reference": reference, "oracle": "independent oracle",
        "oracle_key": lambda result, _: "actual oracle key" if result.get("complete") else None,
        "searches": {label: search_type(label) for label in runner.NATIVE_LABELS},
        "selectors": {label: selector(label) for label in runner.LABELS}}
    runtime = SimpleNamespace(reference_key=lambda result, *_: "actual reference key" if result.exact else None)
    monkeypatch.setattr(runner, "load_cell", lambda *_: {"states": [saved]})
    monkeypatch.setattr(runner, "runtime_provenance", lambda: {"cpu_affinity": [31], "gc_enabled": True})
    return SimpleNamespace(model=model, cell=cell, protocol=protocol, runtime=runtime, deps=deps, saved=saved, calls=calls)


@pytest.mark.parametrize("settled", (False, True))
def test_every_fresh_reference_precedes_six_native_ladders_and_twelve_wholes(monkeypatch, settled):
    h = harness(monkeypatch, settled=settled)
    row = runner.run_cell(h.model, h.cell, h.protocol, h.runtime, h.deps, progress=False)
    assert h.calls[0] == "reference"
    assert row["new_reference_calls"] == 1 and row["new_oracle_calls"] == 0
    assert row["selector_calls"] == dict.fromkeys(runner.LABELS, 1)
    assert row["native_advance_calls"] == 24 and row["native_search_constructions"] == 18
    assert row["root_settle_calls"] == dict.fromkeys(runner.ROOT_LABELS, int(settled))
    for label in runner.NATIVE_LABELS:
        expected = [(label, "construct")]
        expected.append((label, "root" if label in runner.ROOT_LABELS else "advance", 2))
        expected.extend((label, "advance", cap) for cap in (3, 144, 256))
        assert [call for call in h.calls if isinstance(call, tuple) and call[0] == label and call[1] != "whole"] == expected
    for label in runner.LABELS:
        budgets = (4096, 100_000) if label == "long" else (144, 256, 8192)
        assert (label, "whole", budgets) in h.calls
    fresh = row["states"][0]
    assert "prior_routed" not in fresh
    assert fresh["effective_reference_source"] == "full_K4096"
    assert fresh["long_routed"]["label"] == "long"
    for field, label in (("candidate_native_root", "candidate"), ("previous_direct_probe_native_root", "direct"),
                         ("previous_own_tight_native_root", "own")):
        assert fresh[field]["root_lp"] == dict(label=label, cap=2, kind="raw")
        assert (fresh[field]["settled_lp"] is not None) == settled
    for field in ("previous_zero_traces", "previous_dual_face_traces", "previous_native_root_traces",
                  "previous_settled_root_traces", "previous_fused_root_traces"):
        assert fresh[field] == fresh["traces"]["new"]
    assert fresh["traces"]["old"]["256"]["label"] == "branch"
    assert fresh["original_affine_traces"]["256"]["label"] == "affine"
    assert fresh["candidate_native_traces"]["2"]["kind"] == ("settled" if settled else "raw")


def test_borrowed_state_retains_every_original_field_without_execution(monkeypatch):
    h = harness(monkeypatch)
    h.cell["states"] = [h.cell["states"][1]]
    row = runner.run_cell(h.model, h.cell, h.protocol, h.runtime, h.deps, progress=False)
    saved = row["states"][0]
    assert {key: saved[key] for key in h.saved} == h.saved
    assert saved["independent_source"] == "fresh" and saved["confirmation_source"] == "fresh"
    assert "long_routed" not in saved and h.calls == []
    assert row["new_reference_calls"] == row["new_oracle_calls"] == row["native_advance_calls"] == row["native_search_constructions"] == 0
    assert not any(row["selector_calls"].values())
    saved["prior_routed"]["actual_axis"] = False
    assert h.saved["prior_routed"]["actual_axis"] is True


@pytest.mark.parametrize("field,value", (("queue", [1, 1, 1]), ("independent_source", "borrowed_control"),
    ("small_dimension_confirmation_source_state_id", "wrong old state")))
def test_borrowed_metadata_drift_stops_before_any_method_calls(monkeypatch, field, value):
    h = harness(monkeypatch)
    old = h.cell["states"][1]
    old[field] = value
    h.cell["states"] = [old]
    with pytest.raises(ValueError, match="complete frozen state"):
        runner.run_cell(h.model, h.cell, h.protocol, h.runtime, h.deps, progress=False)
    assert h.calls == []


@pytest.mark.parametrize("oracle,kind", ((None, None), ({"complete": True}, "independent_score_face_oracle"),
    ({"complete": False}, None)))
def test_reference_incompleteness_and_conditional_oracle_are_retained(monkeypatch, oracle, kind):
    h = harness(monkeypatch, complete=False, oracle=oracle)
    row = runner.run_cell(h.model, h.cell, h.protocol, h.runtime, h.deps, progress=False)
    state = row["states"][0]
    assert state["reference_complete"] is False
    assert state["oracle"] == oracle
    assert state["effective_reference_source"] == kind
    assert state["effective_reference_complete"] == (kind is not None)
    assert row["new_oracle_calls"] == int(oracle is not None)
    assert row["selector_calls"] == dict.fromkeys(runner.LABELS, 1)


@pytest.fixture
def frozen_sources(monkeypatch, tmp_path):
    previous = json.loads(runner._protocol_path(runner.PREVIOUS_OUTPUT).read_text())
    timing = json.loads(runner._protocol_path(runner.PREVIOUS_TIMING).read_text())
    exclusion = json.loads((runner.ROOT / runner.EXCLUSION_PROTOCOLS[0]).read_text())
    for item in (previous, timing):
        item["execution_sources"] = {"old.py": "unchanged real algorithm\n"}
    dev_flags = dict.fromkeys(("execution_checks_passed", "references_complete", "quality_gate_passed",
        "work_gate_passed", "development_gate_passed"), True)
    timing_flags = dict.fromkeys(("execution_checks_passed", "certificate_checks_passed", "references_complete",
        "quality_gate_passed", "repeat_paths_stable", "cpu_affinity_preserved", "frozen_residual_results_reproduced",
        "local_timing_gate_passed"), True)
    result = dict(decision="SMALL_DIMENSION_EXACT_NATIVE_DEVELOPMENT_ADVANCE", **dev_flags)
    clocks = dict(decision="SMALL_DIMENSION_EXACT_NATIVE_LOCAL_TIMING_ADVANCE",
        prior_decisions=timing["prior_decisions"], **timing_flags)
    dev_path, timing_path = tmp_path / "previous.json", tmp_path / "timing.json"
    for path, protocol, value in ((dev_path, previous, result), (timing_path, timing, clocks)):
        runner._protocol_path(path).write_text(json.dumps(protocol))
        path.write_text(json.dumps(value))
    (tmp_path / "exclusion.json").write_text(json.dumps(exclusion))
    (tmp_path / "old.py").write_text("unchanged real algorithm\n")
    (tmp_path / "new.py").write_text("new frozen confirmation source\n")
    for name, value in (("ROOT", tmp_path), ("PREVIOUS_OUTPUT", dev_path), ("PREVIOUS_TIMING", timing_path),
                       ("SOURCES", ("new.py",)), ("EXCLUSION_PROTOCOLS", ("exclusion.json",))):
        monkeypatch.setattr(runner, name, value)
    models = [SimpleNamespace(node=snapshot["node"], lanes=snapshot["lanes"],
        options=[SimpleNamespace(**option) for option in snapshot["options"]],
        snapshot=lambda snapshot=deepcopy(snapshot): deepcopy(snapshot)) for snapshot in previous["models"]]
    return SimpleNamespace(models=models, previous=previous, timing=timing, exclusion=exclusion,
        dev_path=dev_path, result=result, timing_path=timing_path, clocks=clocks, root=tmp_path)


def test_registration_preserves_roster_law_sources_budgets_and_new_separate_group(frozen_sources):
    source = frozen_sources
    protocol = runner.build_protocol(source.models)
    assert (protocol["expected_state_count"], protocol["expected_independent_state_count"], protocol["expected_borrowed_state_count"]) == (3826, 1024, 2802)
    assert (protocol["expected_selector_calls"], protocol["expected_native_advance_calls"],
            protocol["expected_native_constructions"], protocol["expected_interface_calls"]) == (12288, 24576, 18432, 36864)
    assert protocol["expected_group_counts"] == {**source.previous["expected_group_counts"], runner.NEW_GROUP: 1024}
    assert protocol["subsequent_timing_plan"] == runner.TIMING_PLAN
    assert protocol["prior_decisions"]["small_dimension_exact_native_timing"] == source.clocks["decision"]
    for cell, original in zip(protocol["cells"], source.previous["cells"], strict=True):
        old = [state for state in cell["states"] if state["small_dimension_confirmation_source"] == "borrowed_control"]
        assert [{key: value for key, value in state.items() if not key.startswith("small_dimension_confirmation_source")}
                for state in old] == original["states"]
        fresh = [state for state in cell["states"] if state["small_dimension_confirmation_source"] == "fresh"]
        assert len(fresh) == 64
        assert all(state["timing_group"] == runner.NEW_GROUP and state["confirmation_source"] == "fresh" for state in fresh)
        assert {tuple(state["queue"]) for state in fresh}.isdisjoint(map(tuple, protocol["excluded_static_queues"][str(cell["lanes"])]))
        assert len({tuple(state["queue"]) for state in fresh}) == 64
    first_node = protocol["cells"][0]["node"]
    for lanes in runner.LANES:
        paired = [[state["queue"] for state in cell["states"][:64]] for cell in protocol["cells"] if cell["lanes"] == lanes]
        assert len(paired) == 4 and all(panel == paired[0] for panel in paired)
    assert first_node == source.models[0].node
    assert protocol["panel_seeds"] == list(runner.PANEL_SEEDS)


def test_exclusions_retain_registered_discovery_tails_and_all_previous_draws(frozen_sources):
    from algorithm.experiments.aligned_hardware_zero_delta_affine_fresh_confirmation import excluded_queues

    protocol = runner.build_protocol(frozen_sources.models)
    excluded = excluded_queues((frozen_sources.exclusion, frozen_sources.previous, frozen_sources.timing))
    assert protocol["excluded_static_queues"] == {key: [list(queue) for queue in sorted(values)] for key, values in excluded.items()}
    for lanes, panels in protocol["queue_panels"].items():
        accepted = [tuple(state["queue"]) for panel in panels for state in panel["states"]]
        assert len(accepted) == len(set(accepted)) == 64
        assert set(accepted).isdisjoint(excluded[lanes])
    assert runner.fresh_design(excluded) == runner.fresh_design(excluded)


def test_previous_failed_timing_or_changed_algorithm_blocks_registration(frozen_sources):
    source = frozen_sources
    source.timing_path.write_text(json.dumps({**source.clocks, "local_timing_gate_passed": False}))
    with pytest.raises(ValueError, match="local timing advance"):
        runner.build_protocol(source.models)
    source.timing_path.write_text(json.dumps(source.clocks))
    (source.root / "old.py").write_text("changed real algorithm\n")
    with pytest.raises(ValueError, match="frozen source changed"):
        runner.build_protocol(source.models)


def test_new_margin_and_all_timing_calls_are_frozen_before_results():
    plan = runner.TIMING_PLAN
    assert plan["expected_timed_calls"] == 3826 * 6 * 12 == 275472
    assert plan["expected_selector_calls"] == plan["expected_timed_calls"] + plan["expected_warmup_calls"] == 275856
    assert plan["expected_warmup_calls"] == 16 * 2 * 12 == 384
    assert plan["nonregression_limit"] == 1.05 and plan["trigger_speedup_limit"] == .95
    assert plan["independent_speedup_limit"] == .95 and plan["expected_timing_gate_count"] == 189
    assert plan["order_seed"] == 2026092915
    assert len(set(plan["labels"])) == 12
