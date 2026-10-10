"""Frozen independent clocks cannot hide new or retained-control regressions."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest

from algorithm.experiments import aligned_hardware_small_dimension_exact_native_confirmation_timing as timing
from algorithm.experiments import small_dimension_exact_native_confirmation_timing_summary as summary
from algorithm.experiments import small_dimension_exact_native_timing_summary as previous_summary
from algorithm.experiments import small_dimension_exact_native_development_summary as auditor
from skill.tests.test_small_dimension_exact_native_timing import (
    affine_records, fresh_harness, fresh_records, independent_harness, independent_records,
    old_harness, previous_records, zero_harness, zero_records,
    parent_harness, parent_records, face_parent_harness, face_parent_records,
    native_parent_harness, native_parent_records, settled_parent_harness, settled_parent_records,
    fused_parent_harness, fused_parent_records, own_parent_harness, own_parent_records,
    direct_parent_harness, direct_parent_records,
    harness as small_harness, records as small_records,
)
from skill.tests.test_residual_capacity_timing_summary import _second_cell


@pytest.fixture
def frozen_sources(monkeypatch, tmp_path):
    old = json.loads(timing._protocol_path(timing.PREVIOUS_OUTPUT).read_text())
    old["execution_sources"] = {"old.py": "frozen exact-LP selectors\n"}
    quality = deepcopy(old)
    quality.update(registered_at_utc="new independent quality", expected_state_count=3826,
                   expected_fresh_state_count=1024, expected_control_state_count=2802,
                   expected_group_counts={**timing.CONTROL_COUNTS, "new_independent_fresh": 1024})
    quality["prior_decisions"]["small_dimension_exact_native_timing"] = "SMALL_DIMENSION_EXACT_NATIVE_LOCAL_TIMING_ADVANCE"
    quality["subsequent_timing_plan"] = dict(order_seed=2026092915, labels=list(timing.LABELS),
        repeats=6, warmups_per_label=2, expected_timed_calls=275472, expected_warmup_calls=384,
        expected_selector_calls=275856, nonregression_limit=1.05, trigger_speedup_limit=.95,
        independent_speedup_limit=.95, expected_timing_gate_count=189)
    for cell in quality["cells"]:
        controls = [{**deepcopy(state), "small_dimension_confirmation_source": "borrowed_control",
            "small_dimension_confirmation_source_index": index,
            "small_dimension_confirmation_source_state_id": state["state_id"]}
            for index, state in enumerate(cell["states"])]
        # Deliberately artificial metadata; no new sampler, catalogue or selector runs.
        fresh = [dict(state_id=f"new:{index}", queue=[1000 + index, 2000 + index, 3000 + index],
            source="fresh_confirmation", cohort="new_independent_fresh", panel=f"panel:{index // 16}",
            seed=2026092911 + index // 16, index=index % 16, historical_regression=False,
            timing_group="new_independent_fresh", small_dimension_confirmation_source="fresh",
            caps=deepcopy(controls[0]["caps"])) for index in range(64)]
        # The earlier development and timing stages retain different genuine source pointers.
        for state in controls:
            state.update(timing_source="older_quality", timing_source_index=state["small_dimension_confirmation_source_index"],
                         timing_source_state_id=state["state_id"])
        cell["states"] = fresh + controls
        cell.pop("orders")
    prior = dict(decision="SMALL_DIMENSION_EXACT_NATIVE_LOCAL_TIMING_ADVANCE", **dict.fromkeys((
        "execution_checks_passed", "certificate_checks_passed", "references_complete", "quality_gate_passed",
        "repeat_paths_stable", "cpu_affinity_preserved", "frozen_residual_results_reproduced", "local_timing_gate_passed"), True))
    result = dict(decision="SMALL_DIMENSION_EXACT_NATIVE_CONFIRMATION_ADVANCE", **dict.fromkeys((
        "execution_checks_passed", "references_complete", "quality_gate_passed", "confirmation_gate_passed", "work_gate_passed"), True))
    old_path, quality_path = tmp_path / "old.json", tmp_path / "quality.json"
    for path, protocol, value in ((old_path, old, prior), (quality_path, quality, result)):
        timing._protocol_path(path).write_text(json.dumps(protocol))
        path.write_text(json.dumps(value))
    (tmp_path / "old.py").write_text("frozen exact-LP selectors\n")
    (tmp_path / "timer.py").write_text("new independent clock runner\n")
    for name, value in (("ROOT", tmp_path), ("PREVIOUS_OUTPUT", old_path),
                       ("CONFIRMATION_OUTPUT", quality_path), ("SOURCES", ("timer.py",))):
        monkeypatch.setattr(timing, name, value)
    models = [SimpleNamespace(snapshot=lambda model=deepcopy(model): deepcopy(model)) for model in old["models"]]
    return SimpleNamespace(old=old, quality=quality, quality_path=quality_path, root=tmp_path,
                           result=result, models=models)


def test_registration_retains3826_states_and_every_original_gate(frozen_sources):
    s = frozen_sources
    p = timing.build_protocol(s.models)
    assert (p["expected_state_count"], p["expected_fresh_state_count"], p["expected_control_state_count"],
            p["expected_timed_calls"], p["expected_warmup_calls"], p["expected_selector_calls"]) == (
                3826, 1024, 2802, 275472, 384, 275856)
    assert p["expected_timing_gate_count"] == 189 and p["independent_speedup_limit"] == .95
    assert p["order_seed"] == 2026092915
    assert p["expected_group_counts"] == {**timing.CONTROL_COUNTS, "new_independent_fresh": 1024}
    assert p["prior_decisions"]["direct_probe_own_tight_native_timing"] == "DIRECT_PROBE_OWN_TIGHT_NATIVE_LOCAL_TIMING_HOLD"
    assert p["reference_sources"] == {"small_dimension_confirmation": {"path": "quality.json", "registered_at_utc": "new independent quality"}}
    assert timing.LABELS == summary.LABELS == previous_summary.LABELS
    for cell, source in zip(p["cells"], s.quality["cells"], strict=True):
        assert cell["orders"] == timing.timing_orders(cell["node"], cell["lanes"], len(cell["states"]))
        for index, (state, saved) in enumerate(zip(cell["states"], source["states"], strict=True)):
            assert state == {**saved, "timing_source": "small_dimension_confirmation", "timing_source_index": index,
                             "timing_source_state_id": saved["state_id"]}
        blocks = cell["orders"]["label_orders"][0]
        row = {"state_orders": deepcopy(cell["orders"]["state_orders"]), "states": [
            {"blocks": [{"repeat": repeat, "order": block} for repeat, block in enumerate(orders)]}
            for orders in cell["orders"]["label_orders"]]}
        summary._orders(cell, row, 6)
        assert len({block.index("candidate") for block in blocks}) == 6


@pytest.mark.parametrize("mutation", ("quality", "plan", "source", "lost_fresh", "lost_control", "lineage", "order", "gain"))
def test_incomplete_quality_and_post_registration_changes_block_timing(frozen_sources, mutation):
    s = frozen_sources
    p = {**timing.build_protocol(s.models), "registered_at_utc": "frozen timing"}
    changed = deepcopy(s.quality)
    if mutation == "quality":
        s.quality_path.write_text(json.dumps({**s.result, "work_gate_passed": False}))
    elif mutation == "plan":
        changed["subsequent_timing_plan"]["independent_speedup_limit"] = 1.
    elif mutation == "source":
        (s.root / "old.py").write_text("changed algorithm\n")
    elif mutation == "lost_fresh":
        changed["cells"][0]["states"].pop(0)
    elif mutation == "lost_control":
        changed["cells"][0]["states"].pop()
    elif mutation == "lineage":
        changed["cells"][0]["states"][-1]["state_id"] += ":changed"
    elif mutation == "order":
        p["cells"][0]["orders"]["label_orders"][0][0].reverse()
    else:
        p["independent_speedup_limit"] = 1.
    timing._protocol_path(s.quality_path).write_text(json.dumps(changed))
    with pytest.raises(ValueError):
        timing.verify_protocol(p, s.models)


@pytest.fixture
def harness(small_harness, monkeypatch):
    h = small_harness
    for index, (state, saved) in enumerate(zip(h.cell["states"], h.saved, strict=True)):
        added = {"small_dimension_confirmation_source": "fresh" if index == 0 else "borrowed_control"}
        saved.update(deepcopy(state), **added)
        if index == 0:
            saved.pop("prior_routed", None)
        state.update(**added, timing_source="small_dimension_confirmation", timing_source_index=index,
                     timing_source_state_id=state["state_id"])
    h.cell["orders"] = timing.timing_orders(h.model.node, 64, len(h.saved))
    h.protocol["reference_sources"] = {"small_dimension_confirmation": {"path": "quality.json", "registered_at_utc": "new quality"}}
    monkeypatch.setattr(timing, "load_cell", lambda *args: {"states": h.saved})
    monkeypatch.setattr(timing, "runtime_provenance", lambda: {"cpu_affinity": [0]})
    return h


def test_all_paid_cold_calls_end_before_serialization_and_borrow_genuine_prefixes(harness, monkeypatch):
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
    r = timing.run_cell(h.model, h.cell, h.protocol, h.runtime, selectors=h.selectors, progress=False)
    assert len(h.calls) == len(h.recorded) == 240
    assert Counter(label for label, _, _ in h.calls) == dict.fromkeys(timing.LABELS, 20)
    assert r["new_reference_calls"] == r["new_oracle_calls"] == r["screen_selector_calls"] == 0
    assert "prior_routed" not in r["states"][0]
    for state, saved in zip(r["states"], h.saved, strict=True):
        assert state["expected_results"]["candidate"] == saved["routed"]
        assert state["expected_results"]["direct"] == saved["previous_direct_probe_routed"]
        for field in ("candidate_native_root", "candidate_native_traces", "previous_direct_probe_native_root",
                      "previous_direct_probe_native_traces", "previous_own_tight_native_root", "previous_own_tight_native_traces"):
            assert state[field] == saved[field]
    count = 0
    events = h.events[h.events.index("cpu"):]
    for index, event in enumerate(events):
        if event == "selector":
            count += 1
            assert events[index - 2:index + 3] == ["cpu", "wall", "selector", "wall", "cpu"]
        elif event in ("record", "asdict"):
            assert count % 12 == 0
    assert count == 216


@pytest.fixture
def records(small_records, monkeypatch):
    p, rows = small_records
    row, cell = rows[0], p["cells"][0]
    for state, metadata in zip(row["states"], cell["states"], strict=True):
        state["small_dimension_confirmation_source"] = metadata["small_dimension_confirmation_source"] = "borrowed_control"
    for index in (0, 1):
        state, metadata = deepcopy(row["states"][index]), deepcopy(cell["states"][index])
        added = dict(state_id=f"new:{index}", timing_group="new_independent_fresh",
                     small_dimension_confirmation_source="fresh", historical_regression=False,
                     confirmation_source="fresh", source="fresh_confirmation", cohort="new_independent_fresh")
        state.update(added)
        metadata.update(added)
        state.pop("prior_routed", None)
        row["states"].append(state)
        cell["states"].append(metadata)
    cell["orders"] = timing.timing_orders(row["node"], row["lanes"], len(row["states"]))
    row["state_orders"] = deepcopy(cell["orders"]["state_orders"])
    for index, state in enumerate(row["states"]):
        state["blocks"] = [{"repeat": repeat, "order": block}
                           for repeat, block in enumerate(cell["orders"]["label_orders"][index])]
    row["borrowed_reference_count"] = 12
    p.update(expected_state_count=12, expected_fresh_state_count=2, expected_control_state_count=10,
             expected_timed_calls=864, expected_selector_calls=888,
             independent_speedup_limit=.95, expected_timing_gate_count=189)
    p["expected_group_counts"]["new_independent_fresh"] = 2
    monkeypatch.setattr(summary, "_audit_result", previous_summary._audit_result)  # Explicit inherited toy-native hook.
    return p, rows


def test_separate_new_endpoint_keeps_all155_old_and34_new_gates(records):
    r = summary.summarize(*records)
    assert r["decision"] == "SMALL_DIMENSION_EXACT_NATIVE_INDEPENDENT_TIMING_ADVANCE", r["accounting_failures"][:1]
    assert r["timing_gate_count"] == 189
    assert (r["state_count"], r["new_independent_state_count"], r["control_state_count"], r["paired_queue_draw_count"],
            r["timed_selector_calls"], r["warmup_calls"], r["executed_selector_calls"]) == (12, 2, 10, 2, 864, 24, 888)
    assert r["new_independent_timing_gate_passed"] and r["retained_control_timing_gate_passed"] and r["independent_gain_gate_passed"]
    assert r["groups"]["fresh_confirmation"]["queue_count"] == r["groups"]["independent_fresh"]["queue_count"] == 2
    assert r["groups"]["new_independent_fresh"]["comparisons"]["direct"]["wall_ratio"] == .8
    assert r["groups"]["new_independent_fresh"]["comparisons"]["direct"]["cpu_ratio"] is not None
    assert r["borrowed_native_snapshots"] == 288
    assert r["executed_reference_calls"] == r["executed_oracle_calls"] == r["screen_selector_calls"] == 0
    assert all(sum(key.startswith("new_independent_") for key in gates) == (4 if label == "direct" else 3)
               for label, gates in r["timing_gates"].items())


@pytest.mark.parametrize("ratio,passed", ((.95, True), (.96, False)))
def test_new_positive_gain_is_required_even_with_every_nonregression_gate(records, ratio, passed):
    p, rows = records
    for state in rows[0]["states"][-2:]:
        state["wall_ns"]["candidate"] = [int(100 * ratio)] * 6
    r = summary.summarize(p, rows)
    assert r["independent_gain_gate_passed"] is passed
    assert r["retained_control_timing_gate_passed"]
    assert all(value for gates in r["timing_gates"].values() for key, value in gates.items()
               if key != "new_independent_pooled_direct_gain")
    assert r["decision"].endswith("ADVANCE" if passed else "HOLD")


@pytest.mark.parametrize("label", timing.LABELS[1:])
def test_each_genuine_new_comparator_can_block_confirmation_without_changing_old_flags(records, label):
    p, rows = records
    for state in rows[0]["states"][-2:]:
        state["wall_ns"][label] = [60] * 6
    r = summary.summarize(p, rows)
    assert r["decision"] == "SMALL_DIMENSION_EXACT_NATIVE_INDEPENDENT_TIMING_HOLD"
    assert r["retained_control_timing_gate_passed"] and not r["new_independent_timing_gate_passed"]
    assert r["older_origin_timing_gate_passed"] and r["origin_fresh_timing_gate_passed"] and r["origin_independent_timing_gate_passed"]


@pytest.mark.parametrize("group", ("ordinary_development", "historical_regression", "new_trigger", "fresh_confirmation", "independent_fresh"))
def test_new_gain_cannot_hide_any_retained_cohort_regression(records, group):
    p, rows = records
    for state in rows[0]["states"]:
        if state["timing_group"] == group:
            state["wall_ns"]["direct"] = [60] * 6
    r = summary.summarize(p, rows)
    assert r["decision"] == "SMALL_DIMENSION_EXACT_NATIVE_INDEPENDENT_TIMING_HOLD"
    assert r["new_independent_timing_gate_passed"] and not r["retained_control_timing_gate_passed"]


def test_new_p99_regression_is_preserved_despite_five_percent_pooled_gain(records):
    p, rows = records
    for state, candidate, direct in zip(rows[0]["states"][-2:], (200, 10), (185, 40), strict=True):
        for label in timing.LABELS:
            state["wall_ns"][label] = [200] * 6
        state["wall_ns"]["candidate"], state["wall_ns"]["direct"] = [candidate] * 6, [direct] * 6
    r = summary.summarize(p, rows)
    assert r["independent_gain_gate_passed"] and r["retained_control_timing_gate_passed"]
    assert r["timing_gates"]["direct"]["new_independent_pooled_nonregression"]
    assert not r["timing_gates"]["direct"]["new_independent_p99_nonregression"]
    assert r["decision"].endswith("HOLD")


def test_new_cell_regression_is_preserved_despite_pooled_gain(records):
    p, rows = records
    other = _second_cell(p, rows)
    p["expected_fresh_state_count"] *= 2
    p["expected_control_state_count"] *= 2
    for state in rows[0]["states"][-2:]:
        state["wall_ns"]["direct"] = [60] * 6
    for state in other["states"][-2:]:
        state["wall_ns"]["direct"] = [1000] * 6
    r = summary.summarize(p, rows)
    assert r["independent_gain_gate_passed"] and r["retained_control_timing_gate_passed"]
    assert r["timing_gates"]["direct"]["new_independent_pooled_nonregression"]
    assert r["timing_gates"]["direct"]["new_independent_p99_nonregression"]
    assert not r["timing_gates"]["direct"]["new_independent_cell_nonregression"]
    assert r["decision"].endswith("HOLD")


def test_production_algorithms_and_independent_auditor_remain_frozen():
    from algorithm.experiments.aligned_hardware_small_dimension_exact_native_timing import _selectors
    assert timing._selectors() == _selectors()
    assert summary._audit_result is auditor._audit_small_dimension_result
    assert summary._work is auditor._work
