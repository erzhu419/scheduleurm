"""Six-arm timing retains the failed early comparator and pays every new call."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from itertools import permutations
import json
from types import SimpleNamespace

import pytest

from algorithm.experiments import aligned_hardware_dual_face_root_canonical_timing as timing
from algorithm.experiments import dual_face_root_canonical_timing_summary as summary
from algorithm.experiments.dual_face_root_canonical_development_summary import DUAL_FACE_COUNTERS
from skill.tests.test_early_root_canonical_timing import (
    affine_records, fresh_harness, fresh_records, independent_harness, independent_records,
    old_harness, previous_records, zero_harness, zero_records,
    harness as parent_harness, records as parent_records,
)


@pytest.fixture
def frozen_sources(monkeypatch, tmp_path):
    old = json.loads(timing._protocol_path(timing.PREVIOUS_OUTPUT).read_text())
    old["execution_sources"] = {"old.py": "frozen early source\n"}
    development = deepcopy(old)
    development["registered_at_utc"] = "new quality"
    development["focus_keys"] = [{"node": cell["node"], "lanes": cell["lanes"],
        **{key: state[key] for key in ("state_id", "queue", "timing_group")}}
        for cell in development["cells"] for state in cell["states"]
        if cell["node"] == "gpu_b" and cell["lanes"] == 64 and state["queue"] == [122, 235, 280]]
    development["subsequent_timing_plan"] = dict(order_seed=2026092900, labels=list(timing.LABELS),
        repeats=6, warmups_per_label=2, expected_timed_calls=100872, expected_warmup_calls=192,
        nonregression_limit=1.05, trigger_speedup_limit=.95)
    for cell in development["cells"]:
        cell.pop("orders")
        for index, state in enumerate(cell["states"]):
            state.update(dual_face_source_index=index, dual_face_source_state_id=state["state_id"])
    prior = dict(decision="EARLY_ROOT_CANONICAL_LOCAL_TIMING_HOLD", local_timing_gate_passed=False,
        **dict.fromkeys(("execution_checks_passed", "references_complete", "quality_gate_passed",
                        "repeat_paths_stable", "frozen_residual_results_reproduced"), True))
    quality = dict(decision="DUAL_FACE_ROOT_CANONICAL_DEVELOPMENT_ADVANCE", **dict.fromkeys((
        "execution_checks_passed", "references_complete", "quality_gate_passed", "work_gate_passed",
        "development_gate_passed"), True))
    old_path, dev_path = tmp_path / "old.json", tmp_path / "quality.json"
    for path, protocol, result in ((old_path, old, prior), (dev_path, development, quality)):
        timing._protocol_path(path).write_text(json.dumps(protocol))
        path.write_text(json.dumps(result))
    (tmp_path / "old.py").write_text("frozen early source\n")
    (tmp_path / "timer.py").write_text("new six-arm source\n")
    for name, value in (("ROOT", tmp_path), ("PREVIOUS_OUTPUT", old_path), ("DEVELOPMENT_OUTPUT", dev_path),
                        ("SOURCES", ("timer.py",))):
        monkeypatch.setattr(timing, name, value)
    models = [SimpleNamespace(snapshot=lambda model=deepcopy(model): deepcopy(model)) for model in old["models"]]
    return SimpleNamespace(old=old, development=development, models=models, dev_path=dev_path, quality=quality)


def test_registers_all2802_old_states_and_fully_new_six_arm_clocks(frozen_sources):
    source = frozen_sources
    protocol = timing.build_protocol(source.models)
    assert (protocol["expected_state_count"], protocol["expected_timed_calls"], protocol["expected_warmup_calls"],
            protocol["expected_selector_calls"]) == (2802, 100872, 192, 101064)
    assert protocol["expected_group_counts"] == source.old["expected_group_counts"]
    assert protocol["focus_keys"] == source.development["focus_keys"]
    assert protocol["focus_keys"][0]["queue"] == [122, 235, 280]
    assert protocol["order_seed"] == 2026092900
    assert timing.LABELS == summary.LABELS == ("candidate", "early", "zero", "affine", "branch", "long")
    assert protocol["prior_decisions"]["early_root_canonical_timing"] == "EARLY_ROOT_CANONICAL_LOCAL_TIMING_HOLD"
    for cell, old_cell in zip(protocol["cells"], source.development["cells"], strict=True):
        for index, (state, old) in enumerate(zip(cell["states"], old_cell["states"], strict=True)):
            assert state == {**old, "timing_source": "dual_face_development", "timing_source_index": index,
                             "timing_source_state_id": old["state_id"]}
        for blocks in cell["orders"]["label_orders"]:
            assert Counter(block.index("early") for block in blocks) == dict.fromkeys(range(6), 1)
            assert Counter(tuple(label for label in block if label not in ("early", "zero", "affine"))
                           for block in blocks) == Counter(permutations(timing.ORIGINAL_LABELS))
            assert sorted(Counter([label for label in block if label not in ("early", "zero")].index("affine")
                                  for block in blocks).values()) == [1, 1, 2, 2]
            assert sorted(Counter([label for label in block if label != "early"].index("zero")
                                  for block in blocks).values()) == [1, 1, 1, 1, 2]


@pytest.mark.parametrize("change", ("plan", "roster", "quality"))
def test_six_arm_plan_and_all_sources_are_frozen_before_new_outcomes(frozen_sources, change):
    source = frozen_sources
    protocol = timing.build_protocol(source.models)
    protocol["registered_at_utc"] = "registered"
    if change == "quality":
        source.dev_path.write_text(json.dumps({**source.quality, "work_gate_passed": False}))
    else:
        changed = deepcopy(source.development)
        if change == "plan":
            changed["subsequent_timing_plan"]["expected_timed_calls"] -= 1
        else:
            changed["cells"][0]["states"].pop()
        timing._protocol_path(source.dev_path).write_text(json.dumps(changed))
    with pytest.raises(ValueError):
        timing.verify_protocol(protocol, source.models)


@pytest.fixture
def harness(parent_harness, monkeypatch):
    h = parent_harness
    for index, (state, saved) in enumerate(zip(h.cell["states"], h.saved, strict=True)):
        metadata = dict(dual_face_source_index=index, dual_face_source_state_id=state["state_id"])
        saved.update(**deepcopy(state), **metadata, previous_early_routed=deepcopy(saved["routed"]))
        state.update(**metadata, timing_source="dual_face_development", timing_source_index=index,
                     timing_source_state_id=state["state_id"])
    h.cell["orders"] = timing.timing_orders(h.model.node, 64, len(h.saved))
    h.protocol["reference_sources"] = {"dual_face_development": {"path": "quality.json", "registered_at_utc": "quality"}}
    parent_selector = h.selectors["candidate"]

    def early(*args):
        result = parent_selector(*args)
        _, queue, budgets = h.calls[-1]
        h.calls[-1] = ("early", queue, budgets)
        return result

    h.selectors["early"] = early

    def load(path, node, lanes, utc):
        assert (path.stem, node, lanes, utc) == ("quality", h.model.node, 64, "quality")
        return {"states": h.saved}

    monkeypatch.setattr(timing, "load_cell", load)
    monkeypatch.setattr(timing, "runtime_provenance", lambda: {"cpu_affinity": [0]})
    return h


def test_every_early_control_call_is_new_and_inside_its_own_clock(harness, monkeypatch):
    h, clocks = harness, Counter()

    def clock(name):
        clocks[name] += 100
        h.events.append(name)
        return clocks[name]

    def serialize(value):
        h.events.append("asdict")
        return asdict(value)

    monkeypatch.setattr(timing, "perf_counter_ns", lambda: clock("wall"))
    monkeypatch.setattr(timing, "process_time_ns", lambda: clock("cpu"))
    monkeypatch.setattr(timing, "asdict", serialize)
    row = timing.run_cell(h.model, h.cell, h.protocol, h.runtime, selectors=h.selectors, progress=False)
    assert len(h.calls) == len(h.recorded) == 120
    assert Counter(label for label, _, _ in h.calls) == dict.fromkeys(timing.LABELS, 20)
    for state, saved in zip(row["states"], h.saved, strict=True):
        assert state["expected_results"]["early"] == state["previous_early_routed"] == saved["previous_early_routed"]
        assert ("prior_routed" in state) == ("prior_routed" in saved)
    count = 0
    events = h.events[h.events.index("cpu"):]
    for index, event in enumerate(events):
        if event == "selector":
            count += 1
            assert events[index - 2:index + 3] == ["cpu", "wall", "selector", "wall", "cpu"]
        elif event in ("record", "asdict"):
            assert count % 6 == 0
    assert count == 108


@pytest.fixture
def records(parent_records):
    protocol, rows = parent_records
    row, cell = rows[0], protocol["cells"][0]
    for state in row["states"]:
        state["previous_early_routed"] = deepcopy(state["first_results"]["candidate"])
        for field in ("expected_results", "first_results", "repeat_mismatches", "metrics", "wall_ns", "cpu_ns"):
            state[field]["early"] = deepcopy(state[field]["candidate"])
        state["wall_ns"]["early"], state["cpu_ns"]["early"] = [100] * 6, [90] * 6
        for result in (state["first_results"]["candidate"], state["expected_results"]["candidate"]):
            result.update(dual_face_evidence=None, dual_face_work=dict.fromkeys(DUAL_FACE_COUNTERS, 0))
    warmups = row["warmups"]
    row["warmups"] = []
    for repeat in range(2):
        block = {item["label"]: deepcopy(item) for item in warmups[repeat * 5:(repeat + 1) * 5]}
        block["early"] = {**deepcopy(block["candidate"]), "label": "early"}
        block["candidate"]["result"].update(dual_face_evidence=None, dual_face_work=dict.fromkeys(DUAL_FACE_COUNTERS, 0))
        row["warmups"].extend(block[label] for label in timing.LABELS)
    cell["orders"] = timing.timing_orders(row["node"], row["lanes"], len(row["states"]))
    row["state_orders"] = deepcopy(cell["orders"]["state_orders"])
    for index, state in enumerate(row["states"]):
        state["blocks"] = [dict(repeat=repeat, order=block)
                           for repeat, block in enumerate(cell["orders"]["label_orders"][index])]
    protocol.update(labels=list(timing.LABELS), expected_timed_calls=360, expected_warmup_calls=12, expected_selector_calls=372)
    protocol["prior_decisions"]["early_root_canonical_timing"] = "EARLY_ROOT_CANONICAL_LOCAL_TIMING_HOLD"
    return protocol, rows


def test_six_arm_summary_retains_all_costs_and_prior_hold(records):
    result = summary.summarize(*records)
    assert result["decision"] == "DUAL_FACE_ROOT_CANONICAL_LOCAL_TIMING_ADVANCE"
    assert result["early_timing_gate_passed"] and result["primary_timing_gate_passed"]
    assert (result["timed_selector_calls"], result["warmup_calls"], result["executed_selector_calls"]) == (360, 12, 372)
    assert result["borrowed_native_snapshots"] == 120 and result["borrowed_early_router_count"] == 10
    assert result["executed_early_routing_preparation_work"]["root_snapshot_calls"] == 62
    assert result["executed_routing_preparation_work"]["root_snapshot_calls"] == 62
    assert set(result["executed_dual_face_work"]) == set(DUAL_FACE_COUNTERS)
    assert not any(result["executed_dual_face_work"].values())
    assert result["prior_decisions"]["early_root_canonical_timing"] == "EARLY_ROOT_CANONICAL_LOCAL_TIMING_HOLD"


@pytest.mark.parametrize("group", ("independent_fresh", "fresh_confirmation", "ordinary_development", "historical_regression", "new_trigger"))
def test_all_previous_gains_cannot_hide_regression_against_the_real_early_control(records, group):
    protocol, rows = records
    for state in rows[0]["states"]:
        if state["timing_group"] == group:
            state["wall_ns"]["early"] = [60] * 6
    result = summary.summarize(protocol, rows)
    assert result["decision"] == "DUAL_FACE_ROOT_CANONICAL_LOCAL_TIMING_HOLD"
    assert not result["early_timing_gate_passed"]
    assert result["primary_timing_gate_passed"] and result["affine_timing_gate_passed"] and result["zero_timing_gate_passed"]


@pytest.mark.parametrize("mutation", ("lost_early", "early_positions", "early_output", "face_fee"))
def test_the_added_control_observations_and_new_face_fees_cannot_be_omitted(records, mutation):
    protocol, rows = records
    if mutation in ("lost_early", "early_positions"):
        blocks = protocol["cells"][0]["orders"]["label_orders"][0]
        if mutation == "lost_early":
            blocks[0].remove("early")
        else:
            for block in blocks:
                block.remove("early")
                block.insert(0, "early")
        with pytest.raises(ValueError):
            summary.summarize(protocol, rows)
    else:
        state = rows[0]["states"][-1]
        if mutation == "early_output":
            state["expected_results"]["early"]["status"] = "CHANGED"
        else:
            state["first_results"]["candidate"]["dual_face_work"]["capture_calls"] = 1
            state["expected_results"]["candidate"] = deepcopy(state["first_results"]["candidate"])
        assert summary.summarize(protocol, rows)["decision"] == "INVALID_TIMING_EXECUTION"
