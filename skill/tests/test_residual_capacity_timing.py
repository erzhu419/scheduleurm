"""Freeze retained cohorts and time three whole selectors before their audits."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from itertools import permutations
import json
from types import SimpleNamespace

import pytest

from algorithm.experiments import aligned_hardware_residual_capacity_timing as timing
from algorithm.experiments import aligned_hardware_queue_aware_canonical_development as references


def test_orders_use_all_six_permutations_once_at_each_state():
    orders = timing.timing_orders("timing_test", 64, 37)
    assert orders == timing.timing_orders("timing_test", 64, 37)
    assert timing.LABELS == ("candidate", "branch", "long")
    assert (timing.REPEATS, timing.WARMUPS_PER_LABEL, timing.ORDER_SEED) == (6, 2, 2026092871)
    assert len(orders["state_orders"]) == 6
    assert all(sorted(order) == list(range(37)) for order in orders["state_orders"])
    for state_orders in orders["label_orders"]:
        assert Counter(map(tuple, state_orders)) == Counter(permutations(timing.LABELS))
        assert all(Counter(order[position] for order in state_orders) == dict.fromkeys(timing.LABELS, 2)
                   for position in range(3))
        assert Counter((order[index], order[index + 1]) for order in state_orders for index in (0, 1)) == {
            pair: 2 for pair in permutations(timing.LABELS, 2)}


@pytest.fixture
def frozen_sources(monkeypatch, tmp_path):
    # These are literal roster data; no catalogue construction or selector runs.
    development = json.loads(timing.DEVELOPMENT_OUTPUT.with_name(
        timing.DEVELOPMENT_OUTPUT.stem + "_protocol.json").read_text())
    confirmation = json.loads(timing.CONFIRMATION_OUTPUT.with_name(
        timing.CONFIRMATION_OUTPUT.stem + "_validation_protocol.json").read_text())
    tails = json.loads(timing.TAIL_PROTOCOL.read_text())
    models = [SimpleNamespace(snapshot=lambda snapshot=deepcopy(snapshot): deepcopy(snapshot))
              for snapshot in development["models"]]
    for protocol in (development, confirmation):
        protocol["execution_sources"] = {"selector.py": "frozen selector\n"}
    paths = {name: tmp_path / name for name in ("development.json", "confirmation.json", "tails.json")}
    paths["development.json"].with_name("development_protocol.json").write_text(json.dumps(development))
    paths["confirmation.json"].with_name("confirmation_validation_protocol.json").write_text(json.dumps(confirmation))
    paths["development.json"].write_text(json.dumps({
        "decision": "RESIDUAL_CAPACITY_FRONTIER_DEVELOPMENT_ADVANCE", "development_gate_passed": True}))
    paths["confirmation.json"].write_text(json.dumps({
        "decision": "RESIDUAL_CAPACITY_TRIGGER_CONFIRMATION_ADVANCE", "conditional_confirmation_gate_passed": True,
        "prior_decisions": {"older_quality": "HOLD"}}))
    paths["tails.json"].write_text(json.dumps(tails))
    for name, literal in (("selector.py", "frozen selector\n"), ("timing.py", "new timing runner\n"),
                          ("summary.py", "new timing summary\n")):
        (tmp_path / name).write_text(literal)
    for name, value in (("ROOT", tmp_path), ("DEVELOPMENT_OUTPUT", paths["development.json"]),
                        ("CONFIRMATION_OUTPUT", paths["confirmation.json"]), ("TAIL_PROTOCOL", paths["tails.json"]),
                        ("SOURCES", ("timing.py", "summary.py"))):
        monkeypatch.setattr(timing, name, value)
    return models, development, confirmation, tmp_path


def test_registration_retains_every_control_tail_and_paired_new_trigger(frozen_sources):
    models, development, confirmation, _ = frozen_sources
    protocol = timing.build_protocol(models)
    assert protocol["expected_group_counts"] == dict(ordinary_development=688,
        historical_regression=34, zero_queue=16, trigger_control=8, new_trigger=8)
    assert len(protocol["cells"]) == 16 and protocol["expected_state_count"] == 754
    assert (protocol["expected_timed_calls"], protocol["expected_warmup_calls"],
            protocol["expected_selector_calls"]) == (13572, 96, 13668)
    assert tuple(protocol[key] for key in ("initial_lp_budget", "lp_node_budget", "integer_state_budget",
        "long_lp_budget", "control_integer_state_budget", "nonregression_limit", "trigger_speedup_limit")) == (
            144, 256, 8192, 4096, 100000, 1.05, .95)
    assert len(protocol["historical_regression_keys"]) == 34
    assert len(protocol["ordinary_tail_keys"]) == 16
    assert len(protocol["trigger_control_keys"]) == len(protocol["new_trigger_keys"]) == 8
    assert protocol["reference_sources"] == {
        "residual_development": {"path": "development.json", "registered_at_utc": development["registered_at_utc"]},
        "trigger_confirmation": {"path": "confirmation.json", "registered_at_utc": confirmation["registered_at_utc"]}}
    sources = {"residual_development": development, "trigger_confirmation": confirmation}
    for cell in protocol["cells"]:
        assert len({tuple(state["queue"]) for state in cell["states"]}) == len(cell["states"])
        assert cell["orders"] == timing.timing_orders(cell["node"], cell["lanes"], len(cell["states"]))
        for state in cell["states"]:
            old_cell = next(source for source in sources[state["timing_source"]]["cells"]
                            if (source["node"], source["lanes"]) == (cell["node"], cell["lanes"]))
            original = old_cell["states"][state["timing_source_index"]]
            assert {key: value for key, value in state.items() if not key.startswith("timing_")} == original
            assert state["timing_source_state_id"] == original["state_id"]
            if state["timing_group"] == "new_trigger":
                assert original["source"] == "fresh_confirmation"
    assert protocol["prior_decisions"]["older_quality"] == "HOLD"


def test_source_drift_dropped_controls_and_changed_design_fail_before_timing(frozen_sources):
    models, development, _, root = frozen_sources
    protocol = timing.build_protocol(models)
    protocol["registered_at_utc"] = "timing registered"
    timing.verify_protocol(protocol, models)
    for change in ("budget", "gate", "control", "order"):
        altered = deepcopy(protocol)
        if change == "budget":
            altered["integer_state_budget"] += 1
        elif change == "gate":
            altered["trigger_speedup_limit"] = 1.0
        elif change == "control":
            cell = next(cell for cell in altered["cells"]
                        if any(state["timing_group"] == "trigger_control" for state in cell["states"]))
            cell["states"].pop(next(index for index, state in enumerate(cell["states"])
                                    if state["timing_group"] == "trigger_control"))
        else:
            altered["cells"][0]["orders"]["label_orders"][0][0].reverse()
        with pytest.raises(ValueError, match="registered sources, roster, budgets or timing design changed"):
            timing.verify_protocol(altered, models)
    (root / "selector.py").write_text("changed selector\n")
    with pytest.raises(ValueError, match="frozen source changed"):
        timing.build_protocol(models)
    (root / "selector.py").write_text("frozen selector\n")
    altered = deepcopy(development)
    cell = next(cell for cell in altered["cells"]
                if any(state["cohort"] == "failure_development" for state in cell["states"]))
    cell["states"] = [state for state in cell["states"] if state["cohort"] != "failure_development"]
    (root / "development_protocol.json").write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="complete 746-state roster"):
        timing.build_protocol(models)


@dataclass(frozen=True)
class Selection:
    lp_node_budget: int
    canonical_exact: bool = True
    score_exact: bool = True


@dataclass(frozen=True)
class Routed:
    selection: Selection
    final_lp: dict
    label: str
    work: int = 1
    status: str = "COMPLETE"


@dataclass(frozen=True)
class Long:
    selection: Selection
    preprocessing: dict
    label: str = "long"
    work: int = 1
    status: str = "COMPLETE"


def result(label, *, complete=True, work=1):
    selection = Selection(4096 if label == "long" else 256, complete, complete)
    if label == "long":
        return Long(selection, {"label": label}, work=work,
                    status="COMPLETE" if complete else "STATE_BUDGET_EXHAUSTED")
    return Routed(selection, {"preprocessing": {"label": label}}, label, work,
                  "COMPLETE" if complete else "STATE_BUDGET_EXHAUSTED")


@pytest.fixture
def harness(monkeypatch):
    model = SimpleNamespace(node="mock_node", lanes=4, options=(object(), object()))
    saved, registered = {"residual_development": [], "trigger_confirmation": []}, []
    for index, (source, queue, group) in enumerate((
            ("residual_development", [4, 0, 0], "ordinary_development"),
            ("residual_development", [54, 58, 290], "trigger_control"),
            ("trigger_confirmation", [192, 192, 317], "new_trigger"))):
        old = {"state_id": f"state:{index}", "queue": queue,
            "source": "fresh_confirmation" if index == 2 else "failure_development" if index == 1 else "development",
            "cohort": "conditional_trigger_validation" if index == 2 else "failure_development" if index == 1 else "old_regression",
            "panel": f"panel:{index}", "seed": 2026092862 if index == 2 else None,
            "source_index": 99 + index, "caps": [2, 3, 144, 256], "historical_regression": False}
        registered.append({**deepcopy(old), "timing_source": source, "timing_source_index": len(saved[source]),
            "timing_source_state_id": old["state_id"], "timing_group": group})
        saved[source].append({**old,
            "reference": {"stub": index}, "reference_metrics": {"saved": index},
            "reference_complete": index != 2, "oracle": {"stub": index} if index == 2 else None,
            "effective_reference_complete": True,
            "effective_reference_source": "oracle" if index == 2 else "full_K4096",
            "traces": {"old": {"256": {"saved": index}}, "new": {"256": {"saved": index}}},
            "prior_routed": {"old": {"axis": index}, "new": {"axis": index}},
            "routed": asdict(result("candidate")), "old_routed": asdict(result("branch"))})
    protocol = {"registered_at_utc": "timing registered", "initial_lp_budget": 144, "lp_node_budget": 256,
        "integer_state_budget": 8192, "long_lp_budget": 4096, "control_integer_state_budget": 100000,
        "reference_sources": {name: {"path": name + ".json", "registered_at_utc": name + " registered"}
                              for name in saved}}
    cell = {"node": model.node, "lanes": 64, "states": registered,
            "orders": timing.timing_orders(model.node, 64, len(registered))}
    events, calls, loaded, recorded = [], [], [], []

    def load(path, node, lanes, utc):
        name = path.stem
        assert (node, lanes, utc) == (model.node, 64, name + " registered")
        loaded.append(name)
        return {"node": node, "lanes": lanes, "states": saved[name]}

    def record(metrics, selection, actual_model, queue, lanes, budget, key, preprocessing, reference):
        assert actual_model is model and lanes == 64 and budget == selection.lp_node_budget
        assert preprocessing["label"] in timing.LABELS
        assert key == (f"oracle:{reference.stub}" if reference.stub == 2 else f"reference:{reference.stub}")
        events.append("record")
        recorded.append((queue, budget, reference.stub))
        metrics["calls"] += 1
        metrics["canonical_exact_count"] += int(selection.canonical_exact)

    runtime = SimpleNamespace(
        _metrics=lambda: {"calls": 0, "canonical_exact_count": 0}, _record=record,
        reference_key=lambda reference, queue, count: None if reference.stub == 2 else f"reference:{reference.stub}")
    monkeypatch.setattr(references, "selection_from_record", lambda payload: SimpleNamespace(**payload))
    monkeypatch.setattr(references, "oracle_key", lambda payload, count: f"oracle:{payload['stub']}")
    monkeypatch.setattr(timing, "load_cell", load)
    monkeypatch.setattr(timing, "runtime_provenance", lambda: {"cpu_affinity": [0]})
    from algorithm.experiments import aligned_hardware_strict_score_proof_development as independent
    from algorithm.experiments import integer_capacity_canonical_oracle as oracle

    def forbidden(*args, **kwargs):
        pytest.fail("timing must reuse its independently frozen references and oracle records")

    monkeypatch.setattr(independent, "_new_reference", forbidden)
    monkeypatch.setattr(oracle, "canonical_at_certified_score", forbidden)

    def selector(label):
        def call(options, queue, lanes, *budgets):
            assert options is model.options and lanes == 64
            assert budgets == ((4096, 100000) if label == "long" else (144, 256, 8192))
            calls.append((label, tuple(queue), budgets))
            events.append("selector")
            return result(label)
        return call

    selectors = {label: selector(label) for label in timing.LABELS}
    return SimpleNamespace(model=model, protocol=protocol, cell=cell, saved=saved,
        events=events, calls=calls, loaded=loaded, recorded=recorded, runtime=runtime, selectors=selectors)


def execute(harness):
    return timing.run_cell(harness.model, harness.cell, harness.protocol, harness.runtime,
                           selectors=harness.selectors, progress=False)


def test_calls_are_cold_adjacent_and_borrow_all_reference_and_method_fields(harness):
    untouched = deepcopy(harness.saved)
    row = execute(harness)
    assert len(harness.calls) == len(harness.recorded) == 60
    assert Counter(label for label, _, _ in harness.calls) == dict.fromkeys(timing.LABELS, 20)
    assert {tuple(queue) for _, queue, _ in harness.calls[:6]} == {(4, 0, 0)}
    expected = [(label, tuple(harness.cell["states"][index]["queue"]),
                 (4096, 100000) if label == "long" else (144, 256, 8192))
                for repeat, order in enumerate(harness.cell["orders"]["state_orders"])
                for index in order for label in harness.cell["orders"]["label_orders"][index][repeat]]
    assert harness.calls[6:] == expected
    assert set(harness.loaded) == {"residual_development", "trigger_confirmation"}
    assert len(row["warmups"]) == 6 and row["state_orders"] == harness.cell["orders"]["state_orders"]
    assert (row["new_reference_calls"], row["new_oracle_calls"], row["screen_selector_calls"],
            row["borrowed_reference_count"]) == (0, 0, 0, 3)
    assert all(row[key] == {"cpu_affinity": [0]} for key in (
        "runtime_start", "runtime_before_warmup", "runtime_after_warmup", "runtime_end"))
    for index, state in enumerate(row["states"]):
        old = harness.saved[state["timing_source"]][state["timing_source_index"]]
        for key in (*timing.REFERENCE_FIELDS, "traces", "prior_routed"):
            assert state[key] == old[key]
        assert state["expected_results"] == {"candidate": old["routed"], "branch": old["old_routed"]}
        assert state["blocks"] == [{"repeat": repeat, "order": orders}
                                   for repeat, orders in enumerate(harness.cell["orders"]["label_orders"][index])]
        assert all(state["metrics"][label]["calls"] == 6 for label in timing.LABELS)
        assert all(len(state["wall_ns"][label]) == len(state["cpu_ns"][label]) == 6 for label in timing.LABELS)
    assert harness.saved == untouched


def test_whole_selector_clocks_exclude_serialization_and_three_arm_audits(harness, monkeypatch):
    values = {"wall": 0, "cpu": 0}

    def clock(name):
        values[name] += 100
        harness.events.append(name)
        return values[name]

    def serialize(value):
        harness.events.append("asdict")
        return asdict(value)

    monkeypatch.setattr(timing, "perf_counter_ns", lambda: clock("wall"))
    monkeypatch.setattr(timing, "process_time_ns", lambda: clock("cpu"))
    monkeypatch.setattr(timing, "asdict", serialize)
    row = execute(harness)
    measured = harness.events[next(index for index, event in enumerate(harness.events) if event == "cpu"):]
    count = 0
    for index, event in enumerate(measured):
        if event == "selector":
            count += 1
            assert measured[index - 2:index + 3] == ["cpu", "wall", "selector", "wall", "cpu"]
        elif event in ("record", "asdict"):
            assert count % 3 == 0
    assert count == 54
    for state in row["states"]:
        assert all(state["wall_ns"][label] == state["cpu_ns"][label] == [100] * 6 for label in timing.LABELS)


def test_repeat_work_changes_are_retained_even_when_action_and_quality_stay_equal(harness):
    original = harness.selectors["candidate"]
    counts = Counter()

    def changed(*args):
        outcome = original(*args)
        counts[args[1]] += 1
        return replace(outcome, work=2) if args[1] == (4, 0, 0) and counts[args[1]] == 8 else outcome

    harness.selectors["candidate"] = changed
    row = execute(harness)
    first = row["states"][0]
    mismatch, = first["repeat_mismatches"]["candidate"]
    assert mismatch["repeat_index"] == 5
    assert mismatch["result"]["selection"] == first["first_results"]["candidate"]["selection"]
    assert mismatch["result"]["work"] == 2 and first["first_results"]["candidate"]["work"] == 1
    assert all(not state["repeat_mismatches"][label] for state in row["states"] for label in ("branch", "long"))


def test_honest_nonclosure_keeps_all_calls_and_later_states(harness):
    original = harness.selectors["long"]

    def incomplete(*args):
        original(*args)
        return result("long", complete=False)

    harness.selectors["long"] = incomplete
    row = execute(harness)
    assert len(harness.calls) == 60 and len(row["states"]) == 3
    for state in row["states"]:
        assert state["metrics"]["long"] == {"calls": 6, "canonical_exact_count": 0}
        assert all(value is not None for value in state["wall_ns"]["long"])
        assert state["first_results"]["long"]["status"] == "STATE_BUDGET_EXHAUSTED"
        assert all(state["metrics"][label]["canonical_exact_count"] == 6 for label in ("candidate", "branch"))
    assert all(warmup["metrics"]["canonical_exact_count"] == (warmup["label"] != "long")
               for warmup in row["warmups"])


@pytest.mark.parametrize("change", ("source_index", "state_id", "queue", "control_caps"))
def test_source_mismatch_stops_before_warmups_or_any_timed_call(harness, change):
    state = harness.cell["states"][1]
    if change == "source_index":
        state["timing_source_index"] = 0
    elif change == "state_id":
        state["timing_source_state_id"] = "wrong state"
    elif change == "queue":
        state["queue"][0] += 1
    else:
        state["caps"][-1] += 1
    with pytest.raises(ValueError, match="borrowed timing state differs from its frozen source"):
        execute(harness)
    assert harness.calls == harness.recorded == []
