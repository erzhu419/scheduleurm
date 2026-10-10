"""Four-arm orders, honest work and the retained development timing gates."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from itertools import permutations
import json
from types import SimpleNamespace

import pytest

from algorithm.experiments import aligned_hardware_zero_delta_affine_timing as timing
from algorithm.experiments import zero_delta_affine_timing_summary as summary
from algorithm.experiments.zero_delta_affine_development_summary import ZERO_COUNTERS
from skill.tests.test_affine_dual_fresh_timing import harness as fresh_harness, records as fresh_records
from skill.tests.test_affine_dual_timing import affine_records
from skill.tests.test_residual_capacity_timing import harness as old_harness
from skill.tests.test_residual_capacity_timing_summary import records as previous_records, _nonclosure, _refresh


def test_four_arm_orders_retain_six_original_relative_permutations():
    orders = timing.timing_orders("mock", 64, 17)
    assert timing.LABELS == summary.LABELS == ("candidate", "affine", "branch", "long")
    assert (timing.REPEATS, timing.WARMUPS_PER_LABEL, timing.ORDER_SEED) == (6, 2, 2026092892)
    assert orders == timing.timing_orders("mock", 64, 17)
    assert all(sorted(order) == list(range(17)) for order in orders["state_orders"])
    for blocks in orders["label_orders"]:
        assert Counter(tuple(label for label in block if label != "affine") for block in blocks) == Counter(
            permutations(("candidate", "branch", "long")))
        assert all(sorted(block) == sorted(timing.LABELS) for block in blocks)
        assert sorted(Counter(block.index("affine") for block in blocks).values()) == [1, 1, 2, 2]
    from algorithm.experiments.aligned_hardware_affine_dual_fresh_timing import LABELS
    assert LABELS == ("candidate", "branch", "long")


@pytest.fixture
def frozen_sources(monkeypatch, tmp_path):
    old = json.loads(timing._protocol_path(timing.PREVIOUS_OUTPUT).read_text())
    old["execution_sources"] = {"old.py": "old frozen source\n"}
    models = [SimpleNamespace(snapshot=lambda data=deepcopy(model): deepcopy(data)) for model in old["models"]]
    development = deepcopy(old)
    development.update(registered_at_utc="zero development registered")
    development["execution_sources"]["zero.py"] = "zero frozen source\n"
    for cell in development["cells"]:
        cell.pop("orders")
        for index, state in enumerate(cell["states"]):
            state.update(zero_delta_source_index=index, zero_delta_source_state_id=state["state_id"])
    prior = {"decision": "AFFINE_DUAL_FRESH_TIMING_HOLD", **dict.fromkeys((
        "execution_checks_passed", "references_complete", "quality_gate_passed",
        "repeat_paths_stable", "frozen_residual_results_reproduced"), True)}
    result = {"decision": "ZERO_DELTA_AFFINE_DEVELOPMENT_ADVANCE", **dict.fromkeys((
        "development_gate_passed", "execution_checks_passed", "references_complete", "quality_gate_passed",
        "behavior_preserved", "work_gate_passed"), True)}
    old_path, dev_path = tmp_path / "old.json", tmp_path / "development.json"
    for path, protocol, output in ((old_path, old, prior), (dev_path, development, result)):
        timing._protocol_path(path).write_text(json.dumps(protocol))
        path.write_text(json.dumps(output))
    for path, literal in (("old.py", "old frozen source\n"), ("zero.py", "zero frozen source\n"),
                           ("timing.py", "new timing source\n")):
        (tmp_path / path).write_text(literal)
    for name, value in (("ROOT", tmp_path), ("PREVIOUS_OUTPUT", old_path),
            ("DEVELOPMENT_OUTPUT", dev_path), ("SOURCES", ("timing.py",))):
        monkeypatch.setattr(timing, name, value)
    return SimpleNamespace(models=models, old=old, development=development, prior=prior, result=result,
                           old_path=old_path, dev_path=dev_path, root=tmp_path)


def test_registration_preserves_1778_states_and_adds_only_new_clocks(frozen_sources):
    source = frozen_sources
    protocol = timing.build_protocol(source.models)
    assert (protocol["expected_state_count"], protocol["expected_timed_calls"],
            protocol["expected_warmup_calls"], protocol["expected_selector_calls"]) == (1778, 42672, 128, 42800)
    assert protocol["expected_group_counts"] == source.old["expected_group_counts"]
    assert protocol["reference_sources"] == {"zero_delta_development": {
        "path": "development.json", "registered_at_utc": "zero development registered"}}
    assert tuple(protocol[key] for key in ("initial_lp_budget", "lp_node_budget", "integer_state_budget",
        "long_lp_budget", "control_integer_state_budget", "nonregression_limit", "trigger_speedup_limit")) == (
            144, 256, 8192, 4096, 100000, 1.05, .95)
    for cell, dev in zip(protocol["cells"], source.development["cells"], strict=True):
        assert cell["orders"] == timing.timing_orders(cell["node"], cell["lanes"], len(cell["states"]))
        for index, (state, saved) in enumerate(zip(cell["states"], dev["states"], strict=True)):
            assert state == {**saved, "timing_source": "zero_delta_development",
                "timing_source_index": index, "timing_source_state_id": saved["state_id"]}


@pytest.mark.parametrize("field", ("decision", "development_gate_passed", "execution_checks_passed",
                                  "references_complete", "quality_gate_passed", "behavior_preserved", "work_gate_passed"))
def test_incomplete_development_stops_before_timing(frozen_sources, field):
    source = frozen_sources
    altered = {**source.result, field: "ZERO_DELTA_AFFINE_DEVELOPMENT_HOLD" if field == "decision" else False}
    source.dev_path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="complete zero-delta affine development"):
        timing.build_protocol(source.models)


@pytest.mark.parametrize("change", ("state", "metadata", "source", "gate", "order"))
def test_frozen_sources_roster_gates_and_orders_cannot_change(frozen_sources, change):
    source = frozen_sources
    protocol = timing.build_protocol(source.models)
    protocol["registered_at_utc"] = "timing registered"
    if change in ("state", "metadata"):
        altered = deepcopy(source.development)
        if change == "state":
            altered["cells"][0]["states"].pop()
        else:
            altered["cells"][0]["states"][0]["queue"][0] += 1
        timing._protocol_path(source.dev_path).write_text(json.dumps(altered))
    elif change == "source":
        (source.root / "old.py").write_text("changed source\n")
    elif change == "gate":
        protocol["nonregression_limit"] = 1.1
    else:
        protocol["cells"][0]["orders"]["label_orders"][0][0].reverse()
    with pytest.raises(ValueError):
        timing.verify_protocol(protocol, source.models)


@pytest.fixture
def harness(fresh_harness, monkeypatch):
    harness = fresh_harness
    saved, states = [], []
    for index, (state, record) in enumerate(zip(harness.cell["states"], harness.saved, strict=True)):
        added = {"zero_delta_source_index": index, "zero_delta_source_state_id": state["state_id"]}
        saved.append({**deepcopy(record), **deepcopy(state), **added,
            "previous_dual_traces": deepcopy(record["traces"]["new"]),
            "previous_dual_routed": deepcopy(record["routed"])})
        states.append({**deepcopy(state), **added, "timing_source": "zero_delta_development",
            "timing_source_index": index, "timing_source_state_id": state["state_id"]})
    harness.saved, harness.cell["states"] = saved, states
    harness.cell["orders"] = timing.timing_orders(harness.model.node, 64, len(states))
    harness.protocol["reference_sources"] = {"zero_delta_development": {
        "path": "development.json", "registered_at_utc": "zero development registered"}}
    old_candidate = harness.selectors["candidate"]

    def affine(*args):
        result = old_candidate(*args)
        _, queue, budgets = harness.calls[-1]
        harness.calls[-1] = ("affine", queue, budgets)
        return result

    harness.selectors["affine"] = affine

    def load(path, node, lanes, utc):
        assert (path.stem, node, lanes, utc) == ("development", harness.model.node, 64, "zero development registered")
        harness.loaded.append(path.stem)
        return {"node": node, "lanes": lanes, "states": saved}

    monkeypatch.setattr(timing, "load_cell", load)
    monkeypatch.setattr(timing, "runtime_provenance", lambda: {"cpu_affinity": [0]})
    return harness


def execute(harness):
    return timing.run_cell(harness.model, harness.cell, harness.protocol, harness.runtime,
                           selectors=harness.selectors, progress=False)


def test_each_arm_runs_fresh_whole_calls_and_borrows_the_real_affine_output(harness):
    row = execute(harness)
    assert len(harness.calls) == len(harness.recorded) == 80
    assert Counter(label for label, _, _ in harness.calls) == dict.fromkeys(timing.LABELS, 20)
    assert harness.loaded == ["development"] and len(row["warmups"]) == 8
    assert "prior_routed" not in row["states"][0]
    assert row["states"][2]["prior_routed"] == harness.saved[2]["prior_routed"]
    for state, saved in zip(row["states"], harness.saved, strict=True):
        assert state["previous_dual_traces"] == saved["previous_dual_traces"]
        assert state["previous_dual_routed"] == state["expected_results"]["affine"] == saved["previous_dual_routed"]
    assert row["new_reference_calls"] == row["new_oracle_calls"] == row["screen_selector_calls"] == 0


def test_serialization_waits_until_all_four_whole_call_clocks_finish(harness, monkeypatch):
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
            assert count % 4 == 0
    assert count == 72
    assert all(state["wall_ns"][label] == state["cpu_ns"][label] == [100] * 6
               for state in row["states"] for label in timing.LABELS)


@pytest.fixture
def records(fresh_records):
    protocol, rows = fresh_records
    row, cell = rows[0], protocol["cells"][0]

    def zero(value):
        if isinstance(value, dict):
            if "dual_reuse" in value:
                work = dict.fromkeys(ZERO_COUNTERS, 0)
                work["lower_flags_saved"] = work["lower_flag_comparisons"] = value["selection"]["dual_column_checks"]
                value["zero_delta"] = work
            for nested in value.values():
                zero(nested)
        elif isinstance(value, list):
            for nested in value:
                zero(nested)

    for state in row["states"]:
        state["previous_dual_traces"] = deepcopy(state["traces"]["new"])
        state["previous_dual_routed"] = deepcopy(state["first_results"]["candidate"])
        for field in ("expected_results", "first_results", "repeat_mismatches", "metrics", "wall_ns", "cpu_ns"):
            state[field]["affine"] = deepcopy(state[field]["candidate"])
        state["wall_ns"]["affine"] = [100] * 6
        state["cpu_ns"]["affine"] = [90] * 6
        zero(state["traces"]["new"])
        zero(state["expected_results"]["candidate"])
        zero(state["first_results"]["candidate"])
    orders = timing.timing_orders(row["node"], row["lanes"], len(row["states"]))
    cell["orders"] = orders
    row["state_orders"] = deepcopy(orders["state_orders"])
    for index, state in enumerate(row["states"]):
        state["blocks"] = [{"repeat": repeat, "order": block} for repeat, block in enumerate(orders["label_orders"][index])]
    original = row["warmups"]
    row["warmups"] = []
    for repeat in range(2):
        block = {item["label"]: deepcopy(item) for item in original[repeat * 3:(repeat + 1) * 3]}
        block["affine"] = {**deepcopy(block["candidate"]), "label": "affine"}
        zero(block["candidate"]["result"])
        row["warmups"].extend(block[label] for label in timing.LABELS)
    protocol.update(labels=list(timing.LABELS), expected_timed_calls=len(row["states"]) * 24,
                    expected_warmup_calls=8, expected_selector_calls=len(row["states"]) * 24 + 8)
    return protocol, rows


def test_all_four_arm_results_fees_and_borrowed_evidence_are_kept_separate(records):
    result = summary.summarize(*records)
    assert result["decision"] == "ZERO_DELTA_AFFINE_LOCAL_TIMING_ADVANCE"
    assert result["primary_timing_gate_passed"] and result["affine_timing_gate_passed"]
    assert result["executed_selector_calls"] == 200 and result["warmup_calls"] == 8
    assert result["borrowed_native_snapshots"] == 96 and result["borrowed_residual_router_count"] == 24
    assert result["borrowed_axis_router_count"] == 12
    assert set(result["work"]) == set(timing.LABELS)
    assert result["executed_zero_delta_work"]["lower_flags_saved"] == 0
    assert result["executed_zero_delta_work"]["guard_calls"] == 0
    assert "zero_delta" not in result["work"]["affine"]
    assert result["executed_affine_dual_work"]["certificate_numerator_products"] == 50
    assert result["executed_affine_control_work"]["certificate_numerator_products"] == 50
    assert not result["independent_confirmation"] and result["prior_hold_retained"]


@pytest.mark.parametrize("group", ("fresh_confirmation", "ordinary_development", "historical_regression", "new_trigger"))
def test_primary_gain_cannot_hide_regression_against_real_affine_control(records, group):
    protocol, rows = records
    for state in rows[0]["states"]:
        if state["timing_group"] == group:
            state["wall_ns"]["affine"] = [60] * 6
    result = summary.summarize(protocol, rows)
    assert result["decision"] == "ZERO_DELTA_AFFINE_LOCAL_TIMING_HOLD"
    assert result["primary_timing_gate_passed"] and not result["affine_timing_gate_passed"]


def test_affine_gain_cannot_hide_a_primary_trigger_failure(records):
    protocol, rows = records
    rows[0]["states"][5]["wall_ns"]["branch"] = [60] * 6
    result = summary.summarize(protocol, rows)
    assert result["decision"] == "ZERO_DELTA_AFFINE_LOCAL_TIMING_HOLD"
    assert result["affine_timing_gate_passed"] and not result["primary_timing_gate_passed"]


@pytest.mark.parametrize("mutation", ("lost_affine", "wrong_relative_permutation", "wrong_insertions"))
def test_registered_four_arm_blocks_cannot_drop_or_replace_observations(records, mutation):
    protocol, rows = records
    blocks = protocol["cells"][0]["orders"]["label_orders"][0]
    if mutation == "lost_affine":
        blocks[0].remove("affine")
    elif mutation == "wrong_relative_permutation":
        blocks[1] = deepcopy(blocks[0])
    else:
        for block in blocks:
            block.remove("affine")
            block.insert(0, "affine")
    with pytest.raises(ValueError):
        summary.summarize(protocol, rows)


def test_actual_long_nonclosure_is_saved_and_charged(records):
    protocol, rows = records
    state = rows[0]["states"][-1]
    state["repeat_mismatches"]["long"] = [{"repeat_index": 2,
        "result": _nonclosure(state["first_results"]["long"], state)}]
    _refresh(state, protocol["models"][0])
    result = summary.summarize(protocol, rows)
    assert result["decision"] == "ZERO_DELTA_AFFINE_TIMING_QUALITY_HOLD"
    assert result["work"]["long"]["shared_integer_states"] == 100000
    assert result["quality_failures"] and not result["repeat_paths_stable"]


@pytest.mark.parametrize("mutation", ("saved_flag_fee", "affine_output", "fake_prior"))
def test_actual_fees_and_frozen_control_outputs_must_be_reproduced(records, mutation):
    protocol, rows = records
    state = rows[0]["states"][-1]
    if mutation == "saved_flag_fee":
        state["first_results"]["candidate"]["final_lp"]["zero_delta"]["lower_flags_saved"] -= 1
        state["expected_results"]["candidate"] = deepcopy(state["first_results"]["candidate"])
    elif mutation == "affine_output":
        state["expected_results"]["affine"]["status"] = "CHANGED"
    else:
        state["prior_routed"] = deepcopy(rows[0]["states"][5]["prior_routed"])
    result = summary.summarize(protocol, rows)
    assert result["decision"] == "INVALID_TIMING_EXECUTION" and not result["local_timing_gate_passed"]
