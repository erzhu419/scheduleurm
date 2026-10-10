"""Pure mocked confirmation timing: fresh endpoints cannot hide in old controls."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from itertools import permutations
import json
from types import SimpleNamespace

import pytest

from algorithm.experiments import aligned_hardware_affine_dual_fresh_timing as timing
from algorithm.experiments import affine_dual_fresh_timing_summary as summary
from skill.tests.test_residual_capacity_timing import harness as old_harness
from skill.tests.test_residual_capacity_timing_summary import (
    records as previous_records, _nonclosure, _refresh, _second_cell,
)
from skill.tests.test_affine_dual_timing import affine_records


@pytest.fixture
def frozen_sources(monkeypatch, tmp_path):
    # Only frozen metadata and artificial fixtures; no actual queue sampler runs.
    old = json.loads(timing._protocol_path(timing.PREVIOUS_OUTPUT).read_text())
    old["execution_sources"] = {"old.py": "old frozen source\n"}
    models = [SimpleNamespace(snapshot=lambda data=deepcopy(model): deepcopy(data)) for model in old["models"]]
    quality = deepcopy(old)
    quality.update(registered_at_utc="quality registered", expected_state_count=1778,
                   expected_fresh_state_count=1024, expected_control_state_count=754)
    quality["execution_sources"]["quality.py"] = "quality frozen source\n"
    for cell in quality["cells"]:
        controls = [{**deepcopy(state), "confirmation_source": "borrowed_control",
            "confirmation_source_index": index, "confirmation_source_state_id": state["state_id"]}
            for index, state in enumerate(cell["states"])]
        fresh = [{"state_id": f"fresh:{index}", "queue": [1000 + index, 2000 + index, 3000 + index],
            "source": "fresh_confirmation", "cohort": "fresh_confirmation", "panel": f"panel:{index // 16}",
            "seed": 2026092881 + index // 16, "index": index % 16,
            "confirmation_source": "fresh", "historical_regression": False,
            "timing_group": "fresh_confirmation", "caps": controls[0]["caps"]} for index in range(64)]
        cell.pop("orders")
        cell["states"] = fresh + controls
    prior = {"decision": "AFFINE_DUAL_LOCAL_TIMING_ADVANCE", **dict.fromkeys((
        "execution_checks_passed", "references_complete", "quality_gate_passed", "repeat_paths_stable",
        "frozen_residual_results_reproduced", "local_timing_gate_passed"), True)}
    result = {"decision": "AFFINE_DUAL_FRESH_CONFIRMATION_ADVANCE", **dict.fromkeys((
        "execution_checks_passed", "references_complete", "quality_gate_passed", "confirmation_gate_passed"), True)}
    old_path, quality_path = tmp_path / "old.json", tmp_path / "quality.json"
    for path, protocol, output in ((old_path, old, prior), (quality_path, quality, result)):
        timing._protocol_path(path).write_text(json.dumps(protocol))
        path.write_text(json.dumps(output))
    for path, literal in (("old.py", "old frozen source\n"), ("quality.py", "quality frozen source\n"),
                           ("timing.py", "new timing source\n")):
        (tmp_path / path).write_text(literal)
    for name, value in (("ROOT", tmp_path), ("PREVIOUS_OUTPUT", old_path),
            ("CONFIRMATION_OUTPUT", quality_path), ("SOURCES", ("timing.py",))):
        monkeypatch.setattr(timing, name, value)
    return SimpleNamespace(models=models, old=old, quality=quality, prior=prior, result=result,
                           old_path=old_path, quality_path=quality_path, root=tmp_path)


def test_registration_keeps_all_fresh_and_control_states_with_separate_gates(frozen_sources):
    source = frozen_sources
    protocol = timing.build_protocol(source.models)
    assert (protocol["expected_state_count"], protocol["expected_fresh_state_count"],
            protocol["expected_control_state_count"], protocol["expected_timed_calls"],
            protocol["expected_warmup_calls"], protocol["expected_selector_calls"]) == (1778, 1024, 754, 32004, 96, 32100)
    assert protocol["expected_group_counts"] == {**timing.CONTROL_COUNTS, "fresh_confirmation": 1024}
    assert tuple(protocol[key] for key in ("initial_lp_budget", "lp_node_budget", "integer_state_budget",
        "long_lp_budget", "control_integer_state_budget", "nonregression_limit", "trigger_speedup_limit")) == (
            144, 256, 8192, 4096, 100000, 1.05, .95)
    assert protocol["order_seed"] == 2026092891
    assert protocol["reference_sources"] == {"fresh_confirmation_quality": {
        "path": "quality.json", "registered_at_utc": "quality registered"}}
    for cell, source_cell in zip(protocol["cells"], source.quality["cells"], strict=True):
        assert cell["orders"] == timing.timing_orders(cell["node"], cell["lanes"], len(cell["states"]))
        assert all(Counter(map(tuple, orders)) == Counter(permutations(timing.LABELS))
                   for orders in cell["orders"]["label_orders"])
        for index, (state, saved) in enumerate(zip(cell["states"], source_cell["states"], strict=True)):
            assert state == {**saved, "timing_source": "fresh_confirmation_quality",
                "timing_source_index": index, "timing_source_state_id": saved["state_id"]}


@pytest.mark.parametrize("field", ("decision", "execution_checks_passed", "references_complete",
                                  "quality_gate_passed", "confirmation_gate_passed"))
def test_held_or_incomplete_quality_blocks_timing(frozen_sources, field):
    source = frozen_sources
    altered = {**source.result, field: "AFFINE_DUAL_FRESH_CONFIRMATION_HOLD" if field == "decision" else False}
    source.quality_path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="complete independent fresh confirmation"):
        timing.build_protocol(source.models)


@pytest.mark.parametrize("change", ("fresh", "control", "metadata", "source", "budget"))
def test_dropped_or_changed_frozen_evidence_stops_registration(frozen_sources, change):
    source = frozen_sources
    quality = deepcopy(source.quality)
    if change == "fresh":
        quality["cells"][0]["states"].pop(0)
    elif change == "control":
        quality["cells"][0]["states"].pop()
    elif change == "metadata":
        quality["cells"][0]["states"][-1]["queue"][0] += 1
    elif change == "source":
        (source.root / "old.py").write_text("changed source\n")
    else:
        quality["integer_state_budget"] += 1
    timing._protocol_path(source.quality_path).write_text(json.dumps(quality))
    with pytest.raises(ValueError):
        timing.build_protocol(source.models)


@pytest.mark.parametrize("change", ("gate", "order", "pointer"))
def test_registration_keeps_new_orders_and_independent_source_pointers(frozen_sources, change):
    source = frozen_sources
    protocol = timing.build_protocol(source.models)
    protocol["registered_at_utc"] = "timing registered"
    timing.verify_protocol(protocol, source.models)
    changed = deepcopy(protocol)
    if change == "gate":
        changed["nonregression_limit"] = 1.1
    elif change == "order":
        changed["cells"][0]["orders"]["label_orders"][0][0].reverse()
    else:
        changed["cells"][0]["states"][0]["timing_source_index"] += 1
    with pytest.raises(ValueError, match="registered sources, roster, budgets or timing design changed"):
        timing.verify_protocol(changed, source.models)


@pytest.fixture
def harness(old_harness, monkeypatch):
    harness = old_harness
    saved, states = [], []
    for index, state in enumerate(harness.cell["states"]):
        original = harness.saved[state["timing_source"]][state["timing_source_index"]]
        added = {"confirmation_source": "fresh" if index == 0 else "borrowed_control",
                 "confirmation_source_index": index, "confirmation_source_state_id": state["state_id"]}
        record = {**deepcopy(original), **deepcopy(state), **added}
        if index == 0:
            record.pop("prior_routed")
        saved.append(record)
        states.append({**deepcopy(state), **added, "timing_group": "fresh_confirmation" if index == 0 else state["timing_group"],
            "timing_source": "fresh_confirmation_quality", "timing_source_index": index,
            "timing_source_state_id": state["state_id"]})
    harness.saved, harness.cell["states"] = saved, states
    harness.cell["orders"] = timing.timing_orders(harness.model.node, 64, len(states))
    harness.protocol["reference_sources"] = {"fresh_confirmation_quality": {
        "path": "quality.json", "registered_at_utc": "quality registered"}}

    def load(path, node, lanes, utc):
        assert (path.stem, node, lanes, utc) == ("quality", harness.model.node, 64, "quality registered")
        harness.loaded.append(path.stem)
        return {"node": node, "lanes": lanes, "states": saved}

    monkeypatch.setattr(timing, "load_cell", load)
    monkeypatch.setattr(timing, "runtime_provenance", lambda: {"cpu_affinity": [0]})
    return harness


def execute(harness):
    return timing.run_cell(harness.model, harness.cell, harness.protocol, harness.runtime,
                           selectors=harness.selectors, progress=False)


def test_fresh_borrows_real_quality_outputs_without_fabricating_an_axis_prior(harness):
    row = execute(harness)
    assert harness.loaded == ["quality"] and len(harness.calls) == len(harness.recorded) == 60
    assert Counter(label for label, _, _ in harness.calls) == dict.fromkeys(timing.LABELS, 20)
    assert "prior_routed" not in row["states"][0]
    assert row["states"][2]["source"] == "fresh_confirmation"
    assert row["states"][2]["confirmation_source"] == "borrowed_control"
    assert row["states"][2]["prior_routed"] == harness.saved[2]["prior_routed"]
    for state, saved in zip(row["states"], harness.saved, strict=True):
        assert state["traces"] == saved["traces"]
        assert state["expected_results"] == {"candidate": saved["routed"], "branch": saved["old_routed"]}
    assert row["new_reference_calls"] == row["new_oracle_calls"] == row["screen_selector_calls"] == 0


def test_clocks_enclose_whole_selectors_and_leave_serialization_after_three_calls(harness, monkeypatch):
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
    measured = harness.events[harness.events.index("cpu"):]
    count = 0
    for index, event in enumerate(measured):
        if event == "selector":
            count += 1
            assert measured[index - 2:index + 3] == ["cpu", "wall", "selector", "wall", "cpu"]
        elif event in ("record", "asdict"):
            assert count % 3 == 0
    assert count == 54
    assert all(state["wall_ns"][label] == state["cpu_ns"][label] == [100] * 6
               for state in row["states"] for label in timing.LABELS)


def test_changed_quality_pointer_stops_before_warmups(harness):
    harness.cell["states"][0]["confirmation_source_index"] += 1
    with pytest.raises(ValueError, match="borrowed timing state differs"):
        execute(harness)
    assert not harness.calls and not harness.recorded


@pytest.fixture
def records(affine_records):
    protocol, rows = affine_records
    cell, row = protocol["cells"][0], rows[0]
    for state, metadata in zip(row["states"], cell["states"], strict=True):
        state["confirmation_source"] = metadata["confirmation_source"] = "borrowed_control"
    for index in (0, 5):
        state = deepcopy(row["states"][index])
        metadata = deepcopy(cell["states"][index])
        added = {"state_id": f"fresh:{index}", "confirmation_source": "fresh", "timing_group": "fresh_confirmation",
                 "historical_regression": False, "cohort": "fresh_confirmation", "source": "fresh_confirmation"}
        state.update(added)
        metadata.update(added)
        state.pop("prior_routed")
        row["states"].append(state)
        cell["states"].append(metadata)
        cell["orders"]["label_orders"].append(deepcopy(cell["orders"]["label_orders"][index]))
    for order in cell["orders"]["state_orders"]:
        order.extend((6, 7))
    row["state_orders"] = deepcopy(cell["orders"]["state_orders"])
    row["borrowed_reference_count"] = 8
    protocol.update(expected_state_count=8, expected_fresh_state_count=2, expected_control_state_count=6,
                    expected_timed_calls=144, expected_selector_calls=150)
    protocol["expected_group_counts"]["fresh_confirmation"] = 2
    return protocol, rows


def test_fresh_and_all_prior_groups_pass_with_separate_work_accounting(records):
    result = summary.summarize(*records)
    assert result["decision"] == "AFFINE_DUAL_FRESH_TIMING_ADVANCE"
    assert result["fresh_timing_gate_passed"] and result["control_timing_gate_passed"]
    assert (result["state_count"], result["fresh_state_count"], result["control_state_count"],
            result["executed_selector_calls"]) == (8, 2, 6, 150)
    assert result["groups"]["fresh_confirmation"]["queue_count"] == 2
    assert result["groups"]["ordinary_development"]["queue_count"] == 2
    assert result["borrowed_axis_router_count"] == 12
    assert result["executed_reference_calls"] == result["executed_oracle_calls"] == 0
    assert result["executed_affine_dual_work"]["certificate_numerator_products"] == 50
    assert result["work"]["candidate"]["axis_capacity"]["merged_values_copied"] == 54
    assert result["prior_hold_retained"]


def test_old_pooled_gain_cannot_hide_a_fresh_slowdown(records):
    protocol, rows = records
    for state in rows[0]["states"]:
        if state["confirmation_source"] == "fresh":
            state["wall_ns"]["candidate"] = [120] * 6
        else:
            state["wall_ns"]["branch"] = state["wall_ns"]["long"] = [10000] * 6
    result = summary.summarize(protocol, rows)
    assert result["comparisons"]["branch"]["wall_ratio"] < 1.05
    assert result["decision"] == "AFFINE_DUAL_FRESH_TIMING_HOLD"
    assert not result["fresh_timing_gate_passed"] and result["control_timing_gate_passed"]


def test_fresh_pooled_gain_cannot_hide_a_retained_historical_regression(records):
    protocol, rows = records
    for state in rows[0]["states"]:
        if state["timing_group"] == "historical_regression":
            state["wall_ns"]["candidate"] = [120] * 6
        else:
            state["wall_ns"]["branch"] = state["wall_ns"]["long"] = [10000] * 6
    result = summary.summarize(protocol, rows)
    assert result["comparisons"]["branch"]["wall_ratio"] < 1.05
    assert result["decision"] == "AFFINE_DUAL_FRESH_TIMING_HOLD"
    assert result["fresh_timing_gate_passed"] and not result["control_timing_gate_passed"]


def test_fresh_p99_is_independent_of_pooled_and_retained_controls(records):
    protocol, rows = records
    rows[0]["states"][6]["wall_ns"]["candidate"] = [120] * 6
    rows[0]["states"][7]["wall_ns"]["candidate"] = [10] * 6
    result = summary.summarize(protocol, rows)
    assert result["decision"] == "AFFINE_DUAL_FRESH_TIMING_HOLD"
    for label in timing.LABELS[1:]:
        assert result["timing_gates"][label]["fresh_pooled_nonregression"]
        assert result["timing_gates"][label]["fresh_cell_nonregression"]
        assert not result["timing_gates"][label]["fresh_p99_nonregression"]


def test_fresh_cell_regression_blocks_despite_pooled_and_p99_gain(records):
    protocol, rows = records
    other = _second_cell(protocol, rows)
    protocol["expected_fresh_state_count"] *= 2
    protocol["expected_control_state_count"] *= 2
    for state in rows[0]["states"][-2:]:
        state["wall_ns"]["candidate"] = [120] * 6
    for state in other["states"][-2:]:
        state["wall_ns"]["branch"] = state["wall_ns"]["long"] = [10000] * 6
    result = summary.summarize(protocol, rows)
    assert result["decision"] == "AFFINE_DUAL_FRESH_TIMING_HOLD"
    for label in timing.LABELS[1:]:
        assert result["timing_gates"][label]["fresh_pooled_nonregression"]
        assert result["timing_gates"][label]["fresh_p99_nonregression"]
        assert not result["timing_gates"][label]["fresh_cell_nonregression"]


def test_honest_fresh_long_nonclosure_is_retained_with_all_spent_work(records):
    protocol, rows = records
    state = rows[0]["states"][-1]
    state["repeat_mismatches"]["long"] = [{"repeat_index": 2,
        "result": _nonclosure(state["first_results"]["long"], state)}]
    _refresh(state, protocol["models"][0])
    result = summary.summarize(protocol, rows)
    assert result["decision"] == "AFFINE_DUAL_FRESH_TIMING_QUALITY_HOLD"
    assert result["work"]["long"]["shared_integer_states"] == 100000
    assert result["quality_failures"] and not result["repeat_paths_stable"]


@pytest.mark.parametrize("mutation", ("fake_prior", "prefix_fee", "affine_fee", "source_count"))
def test_actual_source_and_fresh_proof_cost_errors_block_timing(records, mutation):
    protocol, rows = records
    state = rows[0]["states"][-1]
    if mutation == "fake_prior":
        state["prior_routed"] = deepcopy(rows[0]["states"][5]["prior_routed"])
    elif mutation == "prefix_fee":
        state["first_results"]["candidate"]["probe"]["completion"]["prefix_reads"] -= 1
        state["expected_results"]["candidate"] = deepcopy(state["first_results"]["candidate"])
    elif mutation == "affine_fee":
        state["first_results"]["candidate"]["final_lp"]["affine_dual"]["certificate_numerator_products"] -= 1
        state["expected_results"]["candidate"] = deepcopy(state["first_results"]["candidate"])
    else:
        protocol["expected_fresh_state_count"] += 1
    result = summary.summarize(protocol, rows)
    assert result["decision"] == "INVALID_TIMING_EXECUTION" and result["accounting_failures"]
