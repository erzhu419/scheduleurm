"""Sparse development timing keeps every cohort, real control and paid fee."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from itertools import permutations
import json
from types import SimpleNamespace

import pytest

from algorithm.experiments import aligned_hardware_sparse_zero_delta_affine_timing as timing
from algorithm.experiments import sparse_zero_delta_affine_timing_summary as summary
from algorithm.experiments.sparse_zero_delta_affine_development_summary import SPARSE_COUNTERS
from skill.tests.test_zero_delta_affine_fresh_timing import harness as independent_harness, records as independent_records
from skill.tests.test_zero_delta_affine_timing import harness as zero_harness, records as zero_records
from skill.tests.test_affine_dual_fresh_timing import harness as fresh_harness, records as fresh_records
from skill.tests.test_affine_dual_timing import affine_records
from skill.tests.test_residual_capacity_timing import harness as old_harness
from skill.tests.test_residual_capacity_timing_summary import records as previous_records, _nonclosure, _refresh


@pytest.fixture
def frozen_sources(monkeypatch, tmp_path):
    old = json.loads(timing._protocol_path(timing.PREVIOUS_OUTPUT).read_text())
    old["execution_sources"] = {"old.py": "frozen zero selector\n"}
    models = [SimpleNamespace(snapshot=lambda data=deepcopy(model): deepcopy(data)) for model in old["models"]]
    development = deepcopy(old)
    development["registered_at_utc"] = "sparse development registered"
    development["execution_sources"]["sparse.py"] = "frozen sparse selector\n"
    development["subsequent_timing_plan"] = {"order_seed": 2026092898, "labels": list(timing.LABELS),
        "repeats": 6, "warmups_per_label": 2, "expected_timed_calls": 84060, "expected_warmup_calls": 160,
        "nonregression_limit": 1.05, "trigger_speedup_limit": .95}
    for cell in development["cells"]:
        cell.pop("orders")
        for index, state in enumerate(cell["states"]):
            state.update(sparse_delta_source_index=index, sparse_delta_source_state_id=state["state_id"])
    prior = {"decision": "ZERO_DELTA_AFFINE_FRESH_TIMING_HOLD", **dict.fromkeys((
        "execution_checks_passed", "references_complete", "quality_gate_passed",
        "repeat_paths_stable", "frozen_residual_results_reproduced"), True)}
    result = {"decision": "SPARSE_ZERO_DELTA_AFFINE_DEVELOPMENT_ADVANCE", **dict.fromkeys((
        "execution_checks_passed", "references_complete", "quality_gate_passed", "development_gate_passed",
        "behavior_preserved", "work_gate_passed"), True)}
    old_path, dev_path = tmp_path / "old.json", tmp_path / "development.json"
    for path, protocol, output in ((old_path, old, prior), (dev_path, development, result)):
        timing._protocol_path(path).write_text(json.dumps(protocol))
        path.write_text(json.dumps(output))
    for path, literal in (("old.py", "frozen zero selector\n"), ("sparse.py", "frozen sparse selector\n"),
                           ("timing.py", "new timing runner\n")):
        (tmp_path / path).write_text(literal)
    for name, value in (("ROOT", tmp_path), ("PREVIOUS_OUTPUT", old_path),
            ("DEVELOPMENT_OUTPUT", dev_path), ("SOURCES", ("timing.py",))):
        monkeypatch.setattr(timing, name, value)
    return SimpleNamespace(models=models, old=old, development=development, prior=prior, result=result,
                           old_path=old_path, dev_path=dev_path, root=tmp_path)


def test_all2802_states_are_development_controls_with_original_gates(frozen_sources):
    source = frozen_sources
    protocol = timing.build_protocol(source.models)
    assert (protocol["expected_state_count"], protocol["expected_timed_calls"],
            protocol["expected_warmup_calls"], protocol["expected_selector_calls"]) == (2802, 84060, 160, 84220)
    assert protocol["expected_group_counts"] == source.old["expected_group_counts"]
    assert "expected_independent_state_count" not in protocol and "expected_borrowed_state_count" not in protocol
    assert protocol["reference_sources"] == {"sparse_delta_development": {
        "path": "development.json", "registered_at_utc": "sparse development registered"}}
    assert timing.LABELS == summary.LABELS == ("candidate", "zero", "affine", "branch", "long")
    assert protocol["order_seed"] == 2026092898
    assert (protocol["nonregression_limit"], protocol["trigger_speedup_limit"]) == (1.05, .95)
    for cell, dev_cell in zip(protocol["cells"], source.development["cells"], strict=True):
        assert cell["orders"] == timing.timing_orders(cell["node"], cell["lanes"], len(cell["states"]))
        for blocks in cell["orders"]["label_orders"]:
            assert Counter(tuple(label for label in block if label not in ("zero", "affine")) for block in blocks) == Counter(
                permutations(timing.ORIGINAL_LABELS))
            assert sorted(Counter([label for label in block if label != "zero"].index("affine")
                                  for block in blocks).values()) == [1, 1, 2, 2]
            assert sorted(Counter(block.index("zero") for block in blocks).values()) == [1, 1, 1, 1, 2]
        for index, (state, original) in enumerate(zip(cell["states"], dev_cell["states"], strict=True)):
            assert state == {**original, "timing_source": "sparse_delta_development",
                "timing_source_index": index, "timing_source_state_id": original["state_id"]}


@pytest.mark.parametrize("field", ("decision", "development_gate_passed", "execution_checks_passed",
                                  "references_complete", "quality_gate_passed", "behavior_preserved", "work_gate_passed"))
def test_incomplete_or_held_development_prevents_timing(frozen_sources, field):
    source = frozen_sources
    changed = {**source.result, field: "SPARSE_ZERO_DELTA_AFFINE_DEVELOPMENT_HOLD" if field == "decision" else False}
    source.dev_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="complete sparse-delta development"):
        timing.build_protocol(source.models)


@pytest.mark.parametrize("change", ("state", "metadata", "plan", "source", "order", "gate"))
def test_roster_sources_metadata_and_pre_outcome_timing_plan_remain_frozen(frozen_sources, change):
    source = frozen_sources
    protocol = timing.build_protocol(source.models)
    protocol["registered_at_utc"] = "timing registered"
    changed = deepcopy(source.development)
    if change == "state":
        changed["cells"][0]["states"].pop()
    elif change == "metadata":
        changed["cells"][0]["states"][0]["queue"][0] += 1
    elif change == "plan":
        changed["subsequent_timing_plan"]["order_seed"] += 1
    elif change == "source":
        (source.root / "old.py").write_text("changed selector\n")
    elif change == "order":
        protocol["cells"][0]["orders"]["label_orders"][0][0].reverse()
    else:
        protocol["nonregression_limit"] = 1.1
    timing._protocol_path(source.dev_path).write_text(json.dumps(changed))
    with pytest.raises(ValueError):
        timing.verify_protocol(protocol, source.models)


@pytest.fixture
def harness(independent_harness, monkeypatch):
    harness = independent_harness
    saved, states = [], []
    for index, (state, record) in enumerate(zip(harness.cell["states"], harness.saved, strict=True)):
        added = {"sparse_delta_source_index": index, "sparse_delta_source_state_id": state["state_id"]}
        saved.append({**deepcopy(record), **deepcopy(state), **added,
            "previous_zero_traces": deepcopy(record["traces"]["new"]),
            "previous_zero_routed": deepcopy(record["routed"]),
            "original_affine_traces": deepcopy(record["previous_dual_traces"]),
            "original_affine_routed": deepcopy(record["previous_dual_routed"])})
        states.append({**deepcopy(state), **added, "timing_source": "sparse_delta_development",
            "timing_source_index": index, "timing_source_state_id": state["state_id"]})
    harness.saved, harness.cell["states"] = saved, states
    harness.cell["orders"] = timing.timing_orders(harness.model.node, 64, len(states))
    harness.protocol["reference_sources"] = {"sparse_delta_development": {
        "path": "development.json", "registered_at_utc": "sparse development registered"}}
    wrapped = harness.selectors["candidate"]

    def zero(*args):
        result = wrapped(*args)
        _, queue, budgets = harness.calls[-1]
        harness.calls[-1] = ("zero", queue, budgets)
        return result

    harness.selectors["zero"] = zero

    def load(path, node, lanes, utc):
        assert (path.stem, node, lanes, utc) == ("development", harness.model.node, 64, "sparse development registered")
        harness.loaded.append(path.stem)
        return {"node": node, "lanes": lanes, "states": saved}

    monkeypatch.setattr(timing, "load_cell", load)
    monkeypatch.setattr(timing, "runtime_provenance", lambda: {"cpu_affinity": [0]})
    return harness


def execute(harness):
    return timing.run_cell(harness.model, harness.cell, harness.protocol, harness.runtime,
                           selectors=harness.selectors, progress=False)


def test_each_arm_borrows_real_outputs_and_prior_presence_is_not_inferred_from_origin(harness):
    row = execute(harness)
    assert harness.loaded == ["development"] and len(harness.calls) == len(harness.recorded) == 100
    assert Counter(label for label, _, _ in harness.calls) == dict.fromkeys(timing.LABELS, 20)
    assert "prior_routed" not in row["states"][0]
    for state, saved in zip(row["states"], harness.saved, strict=True):
        assert state["previous_zero_routed"] == state["expected_results"]["zero"] == saved["previous_zero_routed"]
        assert state["original_affine_routed"] == state["expected_results"]["affine"] == saved["original_affine_routed"]
        assert ("prior_routed" in state) == ("prior_routed" in saved)
    assert row["new_reference_calls"] == row["new_oracle_calls"] == row["screen_selector_calls"] == 0


def test_all_five_whole_call_clocks_finish_before_serialization(harness, monkeypatch):
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
    events = harness.events[harness.events.index("cpu"):]
    count = 0
    for index, event in enumerate(events):
        if event == "selector":
            count += 1
            assert events[index - 2:index + 3] == ["cpu", "wall", "selector", "wall", "cpu"]
        elif event in ("record", "asdict"):
            assert count % 5 == 0
    assert count == 90
    assert all(state["wall_ns"][label] == state["cpu_ns"][label] == [100] * 6
               for state in row["states"] for label in timing.LABELS)


@pytest.fixture
def records(independent_records):
    protocol, rows = independent_records
    row, cell = rows[0], protocol["cells"][0]

    def sparse(value):
        if isinstance(value, dict):
            if "dual_reuse" in value:
                value["sparse_delta"] = dict.fromkeys(SPARSE_COUNTERS, 0)
                # The inherited mock saves one certificate and has no child updates.
                value["sparse_delta"]["anchor_resets"] = 1
            for nested in value.values():
                sparse(nested)
        elif isinstance(value, list):
            for nested in value:
                sparse(nested)

    for state in row["states"]:
        state["previous_zero_traces"] = deepcopy(state["traces"]["new"])
        state["previous_zero_routed"] = deepcopy(state["first_results"]["candidate"])
        state["original_affine_traces"] = deepcopy(state["previous_dual_traces"])
        state["original_affine_routed"] = deepcopy(state["first_results"]["affine"])
        for field in ("expected_results", "first_results", "repeat_mismatches", "metrics", "wall_ns", "cpu_ns"):
            state[field]["zero"] = deepcopy(state[field]["candidate"])
        state["wall_ns"]["zero"] = [100] * 6
        state["cpu_ns"]["zero"] = [90] * 6
        sparse(state["traces"]["new"])
        sparse(state["expected_results"]["candidate"])
        sparse(state["first_results"]["candidate"])
    original = row["warmups"]
    row["warmups"] = []
    for repeat in range(2):
        block = {item["label"]: deepcopy(item) for item in original[repeat * 4:(repeat + 1) * 4]}
        block["zero"] = {**deepcopy(block["candidate"]), "label": "zero"}
        sparse(block["candidate"]["result"])
        row["warmups"].extend(block[label] for label in timing.LABELS)
    orders = timing.timing_orders(row["node"], row["lanes"], len(row["states"]))
    cell["orders"] = orders
    row["state_orders"] = deepcopy(orders["state_orders"])
    for index, state in enumerate(row["states"]):
        state["blocks"] = [{"repeat": repeat, "order": block} for repeat, block in enumerate(orders["label_orders"][index])]
    protocol.update(labels=list(timing.LABELS), expected_timed_calls=300,
                    expected_warmup_calls=10, expected_selector_calls=310)
    protocol.pop("expected_independent_state_count")
    protocol.pop("expected_borrowed_state_count")
    return protocol, rows


def test_all_cohorts_pass_as_development_with_sparse_and_zero_fees_separate(records):
    result = summary.summarize(*records)
    assert result["decision"] == "SPARSE_ZERO_DELTA_AFFINE_LOCAL_TIMING_ADVANCE"
    assert result["primary_timing_gate_passed"] and result["affine_timing_gate_passed"] and result["zero_timing_gate_passed"]
    assert result["state_count"] == result["development_state_count"] == 10
    assert result["executed_selector_calls"] == 310 and result["warmup_calls"] == 10
    assert result["borrowed_native_snapshots"] == 160
    assert result["borrowed_affine_audit_native_snapshots"] == 40
    assert result["borrowed_residual_router_count"] == 40
    assert result["borrowed_axis_router_count"] == 12
    assert set(result["executed_sparse_delta_work"]) == set(SPARSE_COUNTERS)
    assert result["executed_sparse_delta_work"]["anchor_resets"] == 62
    assert not any(value for key, value in result["executed_sparse_delta_work"].items() if key != "anchor_resets")
    assert "sparse_delta" not in result["work"]["zero"]
    assert result["executed_affine_dual_work"]["certificate_numerator_products"] == 62
    assert result["executed_zero_control_affine_work"]["certificate_numerator_products"] == 62
    assert result["executed_affine_control_work"]["certificate_numerator_products"] == 62
    assert not result["independent_confirmation"] and result["prior_hold_retained"]


@pytest.mark.parametrize("group", ("independent_fresh", "fresh_confirmation", "ordinary_development", "historical_regression", "new_trigger"))
@pytest.mark.parametrize("label", ("zero", "affine"))
def test_primary_gain_cannot_hide_a_real_control_regression_in_any_origin(records, group, label):
    protocol, rows = records
    for state in rows[0]["states"]:
        if state["timing_group"] == group:
            state["wall_ns"][label] = [60] * 6
    result = summary.summarize(protocol, rows)
    assert result["decision"] == "SPARSE_ZERO_DELTA_AFFINE_LOCAL_TIMING_HOLD"
    assert result["primary_timing_gate_passed"] and not result[label + "_timing_gate_passed"]
    assert result[("affine" if label == "zero" else "zero") + "_timing_gate_passed"]


def test_zero_gain_cannot_hide_the_retained_long_trigger_gain_requirement(records):
    protocol, rows = records
    for state in rows[0]["states"]:
        if state["timing_group"] in ("trigger_control", "new_trigger"):
            state["wall_ns"]["long"] = [82] * 6
    result = summary.summarize(protocol, rows)
    assert result["decision"] == "SPARSE_ZERO_DELTA_AFFINE_LOCAL_TIMING_HOLD"
    assert result["zero_timing_gate_passed"] and not result["primary_timing_gate_passed"]
    assert not result["timing_gates"]["long"]["trigger_pooled_five_percent_gain"]


@pytest.mark.parametrize("mutation", ("lost_affine", "lost_zero", "original_relative_order",
                                      "affine_positions", "zero_positions"))
def test_five_arm_registration_keeps_both_controls_and_the_original_relative_orders(records, mutation):
    protocol, rows = records
    blocks = protocol["cells"][0]["orders"]["label_orders"][0]
    if mutation in ("lost_affine", "lost_zero"):
        blocks[0].remove(mutation.removeprefix("lost_"))
    elif mutation == "original_relative_order":
        blocks[1] = deepcopy(blocks[0])
    else:
        label = mutation.removesuffix("_positions")
        for block in blocks:
            block.remove(label)
            block.insert(0, label)
    with pytest.raises(ValueError):
        summary.summarize(protocol, rows)


@pytest.mark.parametrize("mutation", ("sparse_fee", "zero_output", "affine_output", "fake_prior", "lost_repeat"))
def test_actual_sparse_fees_and_frozen_zero_outputs_cannot_be_hidden(records, mutation):
    protocol, rows = records
    state = rows[0]["states"][-1]
    if mutation == "sparse_fee":
        state["first_results"]["candidate"]["final_lp"]["sparse_delta"]["guard_delta_reads"] = 3
        state["expected_results"]["candidate"] = deepcopy(state["first_results"]["candidate"])
    elif mutation in ("zero_output", "affine_output"):
        state["expected_results"][mutation.removesuffix("_output")]["status"] = "CHANGED"
    elif mutation == "fake_prior":
        state["prior_routed"] = deepcopy(rows[0]["states"][5]["prior_routed"])
    else:
        state["wall_ns"]["zero"].pop()
        with pytest.raises(ValueError):
            summary.summarize(protocol, rows)
        return
    result = summary.summarize(protocol, rows)
    assert result["decision"] == "INVALID_TIMING_EXECUTION" and not result["local_timing_gate_passed"]


def test_honest_long_nonclosure_keeps_its_quality_priority_and_all_spent_states(records):
    protocol, rows = records
    state = rows[0]["states"][-1]
    state["repeat_mismatches"]["long"] = [{"repeat_index": 2,
        "result": _nonclosure(state["first_results"]["long"], state)}]
    _refresh(state, protocol["models"][0])
    result = summary.summarize(protocol, rows)
    assert result["decision"] == "SPARSE_ZERO_DELTA_AFFINE_TIMING_QUALITY_HOLD"
    assert result["work"]["long"]["shared_integer_states"] == 100000
    assert result["quality_failures"] and not result["repeat_paths_stable"]
