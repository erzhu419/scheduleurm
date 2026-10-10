"""Nine-arm timing measures fused and genuine settled controls without dropping gates."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from itertools import permutations
import json
from types import SimpleNamespace

import pytest

from algorithm.experiments import aligned_hardware_fused_root_dual_face_timing as timing
from algorithm.experiments import fused_root_dual_face_timing_summary as summary
from skill.tests.test_settled_root_dual_face_timing import (
    affine_records, fresh_harness, fresh_records, independent_harness, independent_records,
    old_harness, previous_records, zero_harness, zero_records,
    parent_harness, parent_records, face_parent_harness, face_parent_records,
    native_parent_harness, native_parent_records,
    harness as settled_parent_harness, records as settled_parent_records,
)


@pytest.fixture
def frozen_sources(monkeypatch, tmp_path):
    old = json.loads(timing._protocol_path(timing.PREVIOUS_OUTPUT).read_text())
    parent = json.loads(timing._protocol_path(timing.PREVIOUS_DEVELOPMENT_OUTPUT).read_text())
    for protocol in (old, parent):
        protocol["execution_sources"] = {"old.py": "frozen settled source\n"}
    development = deepcopy(parent)
    development["registered_at_utc"] = "new fused development"
    development["prior_decisions"]["settled_root_dual_face_timing"] = "SETTLED_ROOT_DUAL_FACE_LOCAL_TIMING_HOLD"
    development["subsequent_timing_plan"] = dict(order_seed=2026092903, labels=list(timing.LABELS),
        repeats=6, warmups_per_label=2, expected_timed_calls=151308, expected_warmup_calls=288,
        nonregression_limit=1.05, trigger_speedup_limit=.95)
    for cell in development["cells"]:
        for index, state in enumerate(cell["states"]):
            state.update(fused_root_source_index=index, fused_root_source_state_id=state["state_id"])
    flags = dict.fromkeys(("execution_checks_passed", "references_complete", "quality_gate_passed",
                          "development_gate_passed", "work_gate_passed"), True)
    quality = dict(decision="FUSED_ROOT_DUAL_FACE_DEVELOPMENT_ADVANCE", **flags)
    parent_result = dict(decision="SETTLED_ROOT_DUAL_FACE_DEVELOPMENT_ADVANCE", **flags)
    prior = dict(decision="SETTLED_ROOT_DUAL_FACE_LOCAL_TIMING_HOLD", local_timing_gate_passed=False,
        **dict.fromkeys(("execution_checks_passed", "references_complete", "quality_gate_passed",
                        "repeat_paths_stable", "frozen_residual_results_reproduced"), True))
    old_path, parent_path, dev_path, raw_path = (tmp_path / name for name in (
        "timing_checked.json", "parent_checked.json", "new_development.json", "parent_v1.json"))
    for path, protocol, result in ((old_path, old, prior), (parent_path, parent, parent_result),
                                  (dev_path, development, quality), (raw_path, parent, parent_result)):
        timing._protocol_path(path).write_text(json.dumps(protocol))
        path.write_text(json.dumps(result))
    (tmp_path / "old.py").write_text("frozen settled source\n")
    (tmp_path / "timer.py").write_text("new nine-arm source\n")
    for name, value in (("ROOT", tmp_path), ("PREVIOUS_OUTPUT", old_path),
                       ("PREVIOUS_DEVELOPMENT_OUTPUT", parent_path), ("DEVELOPMENT_OUTPUT", dev_path),
                       ("RAW_PREVIOUS_DEVELOPMENT_OUTPUT", raw_path), ("SOURCES", ("timer.py",))):
        monkeypatch.setattr(timing, name, value)
    models = [SimpleNamespace(snapshot=lambda model=deepcopy(model): deepcopy(model)) for model in old["models"]]
    return SimpleNamespace(old=old, parent=parent, parent_path=parent_path,
                           development=development, models=models, dev_path=dev_path, quality=quality)


def test_registers_all2802_states_nine_arms_and_every_previous_insertion_law(frozen_sources):
    s = frozen_sources
    protocol = timing.build_protocol(s.models)
    assert (protocol["expected_state_count"], protocol["expected_timed_calls"], protocol["expected_warmup_calls"],
            protocol["expected_selector_calls"]) == (2802, 151308, 288, 151596)
    assert timing.LABELS == summary.LABELS == ("candidate", "settled", "native", "face", "early", "zero", "affine", "branch", "long")
    assert protocol["order_seed"] == 2026092903
    assert protocol["expected_group_counts"] == s.old["expected_group_counts"]
    assert protocol["focus_keys"] == s.development["focus_keys"]
    assert protocol["prior_decisions"]["settled_root_dual_face_timing"] == "SETTLED_ROOT_DUAL_FACE_LOCAL_TIMING_HOLD"
    assert protocol["reference_sources"]["fused_root_development"] == {
        "path": s.dev_path.name, "registered_at_utc": s.development["registered_at_utc"]}
    for cell in protocol["cells"]:
        for blocks in cell["orders"]["label_orders"]:
            assert len({block.index("settled") for block in blocks}) == 6
            strip = lambda ignored: [[label for label in block if label not in ignored] for block in blocks]
            assert len({block.index("native") for block in strip(("settled",))}) == 6
            assert len({block.index("face") for block in strip(("settled", "native"))}) == 6
            assert Counter(block.index("early") for block in strip(("settled", "native", "face"))) == dict.fromkeys(range(6), 1)
            assert sorted(Counter(block.index("zero") for block in strip(("settled", "native", "face", "early"))).values()) == [1, 1, 1, 1, 2]
            assert sorted(Counter(block.index("affine") for block in strip(("settled", "native", "face", "early", "zero"))).values()) == [1, 1, 2, 2]
            assert Counter(tuple(block) for block in strip(("settled", "native", "face", "early", "zero", "affine"))) == Counter(permutations(timing.ORIGINAL_LABELS))


@pytest.mark.parametrize("change", ("plan", "roster", "quality", "parent_registration"))
def test_frozen_plan_roster_quality_and_original_parent_registration(frozen_sources, change):
    s = frozen_sources
    protocol = {**timing.build_protocol(s.models), "registered_at_utc": "registered"}
    if change == "quality":
        s.dev_path.write_text(json.dumps({**s.quality, "work_gate_passed": False}))
    elif change == "parent_registration":
        changed = {**s.parent, "registered_at_utc": "unrelated checked registration"}
        timing._protocol_path(s.parent_path).write_text(json.dumps(changed))
    else:
        changed = deepcopy(s.development)
        if change == "plan":
            changed["subsequent_timing_plan"]["expected_warmup_calls"] -= 1
        else:
            changed["cells"][0]["states"].pop()
        timing._protocol_path(s.dev_path).write_text(json.dumps(changed))
    with pytest.raises(ValueError):
        timing.verify_protocol(protocol, s.models)


@pytest.fixture
def harness(settled_parent_harness, monkeypatch):
    h = settled_parent_harness
    for index, (state, saved) in enumerate(zip(h.cell["states"], h.saved, strict=True)):
        metadata = dict(fused_root_source_index=index, fused_root_source_state_id=state["state_id"])
        saved.update(**deepcopy(state), **metadata, previous_settled_root_routed=deepcopy(saved["routed"]),
                     previous_settled_root_traces=deepcopy(saved["traces"]["new"]))
        state.update(**metadata, timing_source="fused_root_development", timing_source_index=index,
                     timing_source_state_id=state["state_id"])
    h.cell["orders"] = timing.timing_orders(h.model.node, 64, len(h.saved))
    h.protocol["reference_sources"] = {"fused_root_development": {"path": "new_quality.json", "registered_at_utc": "fused quality"}}
    parent_selector = h.selectors["candidate"]

    def settled(*args):
        result = parent_selector(*args)
        _, queue, budgets = h.calls[-1]
        h.calls[-1] = ("settled", queue, budgets)
        return result

    h.selectors["settled"] = settled

    def load(path, node, lanes, utc):
        assert (path.stem, node, lanes, utc) == ("new_quality", h.model.node, 64, "fused quality")
        return {"states": h.saved}

    monkeypatch.setattr(timing, "load_cell", load)
    monkeypatch.setattr(timing, "runtime_provenance", lambda: {"cpu_affinity": [0]})
    return h


def test_each_of_nine_actual_mock_calls_has_its_own_clock_before_serialization(harness, monkeypatch):
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
    assert len(h.calls) == len(h.recorded) == 180
    assert Counter(label for label, _, _ in h.calls) == dict.fromkeys(timing.LABELS, 20)
    for state, saved in zip(row["states"], h.saved, strict=True):
        assert state["expected_results"]["settled"] == saved["previous_settled_root_routed"]
        assert state["previous_settled_root_traces"] == saved["traces"]["new"]
        assert ("prior_routed" in state) == ("prior_routed" in saved)
    count = 0
    events = h.events[h.events.index("cpu"):]
    for index, event in enumerate(events):
        if event == "selector":
            count += 1
            assert events[index - 2:index + 3] == ["cpu", "wall", "selector", "wall", "cpu"]
        elif event in ("record", "asdict"):
            assert count % 9 == 0
    assert count == 162


def _fuse_mock_fee(result):
    result["routing_preparation"]["root_snapshot_heap_entries_read"] //= 2
    result["routing_preparation"]["root_snapshot_retained_entries_read"] = 0


@pytest.fixture
def records(settled_parent_records):
    protocol, rows = settled_parent_records
    row, cell = rows[0], protocol["cells"][0]
    for state in row["states"]:
        state["previous_settled_root_routed"] = deepcopy(state["first_results"]["candidate"])
        state["previous_settled_root_traces"] = deepcopy(state["traces"]["new"])
        for field in ("expected_results", "first_results", "repeat_mismatches", "metrics", "wall_ns", "cpu_ns"):
            state[field]["settled"] = deepcopy(state[field]["candidate"])
        state["wall_ns"]["settled"], state["cpu_ns"]["settled"] = [100] * 6, [90] * 6
        _fuse_mock_fee(state["first_results"]["candidate"])
        _fuse_mock_fee(state["expected_results"]["candidate"])
    warmups, row["warmups"] = row["warmups"], []
    for repeat in range(2):
        block = {item["label"]: deepcopy(item) for item in warmups[repeat * 8:(repeat + 1) * 8]}
        block["settled"] = {**deepcopy(block["candidate"]), "label": "settled"}
        _fuse_mock_fee(block["candidate"]["result"])
        row["warmups"].extend(block[label] for label in timing.LABELS)
    cell["orders"] = timing.timing_orders(row["node"], row["lanes"], len(row["states"]))
    row["state_orders"] = deepcopy(cell["orders"]["state_orders"])
    for index, state in enumerate(row["states"]):
        state["blocks"] = [dict(repeat=repeat, order=block) for repeat, block in enumerate(cell["orders"]["label_orders"][index])]
    protocol.update(labels=list(timing.LABELS), expected_timed_calls=540, expected_warmup_calls=18, expected_selector_calls=558)
    protocol["prior_decisions"]["settled_root_dual_face_timing"] = "SETTLED_ROOT_DUAL_FACE_LOCAL_TIMING_HOLD"
    return protocol, rows


def test_summary_pays_new_single_pass_and_genuine_settled_double_pass_separately(records):
    r = summary.summarize(*records)
    assert r["decision"] == "FUSED_ROOT_DUAL_FACE_LOCAL_TIMING_ADVANCE", r["accounting_failures"][:1]
    assert r["settled_timing_gate_passed"] and r["native_timing_gate_passed"] and r["primary_timing_gate_passed"]
    assert (r["timed_selector_calls"], r["warmup_calls"], r["executed_selector_calls"]) == (540, 18, 558)
    assert r["borrowed_native_snapshots"] == 120 and r["borrowed_settled_root_router_count"] == 10
    assert r["executed_routing_preparation_work"]["root_snapshot_calls"] == 62
    assert r["executed_settled_routing_preparation_work"]["root_snapshot_calls"] == 62
    assert r["executed_routing_preparation_work"]["root_snapshot_heap_entries_read"] == 24
    assert r["executed_settled_routing_preparation_work"]["root_snapshot_heap_entries_read"] == 48
    assert r["work"]["candidate"]["search_work"] == r["work"]["settled"]["search_work"]
    assert r["prior_decisions"]["settled_root_dual_face_timing"] == "SETTLED_ROOT_DUAL_FACE_LOCAL_TIMING_HOLD"


@pytest.mark.parametrize("group", ("independent_fresh", "historical_regression", "new_trigger"))
def test_gains_against_old_controls_cannot_hide_settled_regression(records, group):
    protocol, rows = records
    for state in rows[0]["states"]:
        if state["timing_group"] == group:
            state["wall_ns"]["settled"] = [60] * 6
    r = summary.summarize(protocol, rows)
    assert r["decision"] == "FUSED_ROOT_DUAL_FACE_LOCAL_TIMING_HOLD"
    assert not r["settled_timing_gate_passed"] and r["native_timing_gate_passed"] and r["primary_timing_gate_passed"]


@pytest.mark.parametrize("label", ("native", "zero"))
def test_added_settled_control_never_replaces_old_failed_comparators(records, label):
    protocol, rows = records
    for state in rows[0]["states"]:
        if state["timing_group"] == "ordinary_development":
            state["wall_ns"][label] = [60] * 6
    r = summary.summarize(protocol, rows)
    assert r["decision"] == "FUSED_ROOT_DUAL_FACE_LOCAL_TIMING_HOLD"
    assert r["settled_timing_gate_passed"] and not r[label + "_timing_gate_passed"]


@pytest.mark.parametrize("mutation", ("lost_settled", "settled_positions", "discounted_settled"))
def test_settled_clock_order_and_actual_old_snapshot_cost_cannot_be_replaced(records, mutation):
    protocol, rows = records
    if mutation == "discounted_settled":
        state = rows[0]["states"][-1]
        _fuse_mock_fee(state["first_results"]["settled"])
        state["expected_results"]["settled"] = deepcopy(state["first_results"]["settled"])
        assert summary.summarize(protocol, rows)["decision"] == "INVALID_TIMING_EXECUTION"
    else:
        blocks = protocol["cells"][0]["orders"]["label_orders"][0]
        if mutation == "lost_settled":
            blocks[0].remove("settled")
        else:
            for block in blocks:
                block.remove("settled")
                block.insert(0, "settled")
        with pytest.raises(ValueError):
            summary.summarize(protocol, rows)
