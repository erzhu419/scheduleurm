"""Ten-arm timing pays changed native work while retaining every genuine control."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from itertools import permutations
import json
from types import SimpleNamespace

import pytest

from algorithm.experiments import aligned_hardware_own_tight_dual_native_timing as timing
from algorithm.experiments import own_tight_dual_native_timing_summary as summary
from algorithm.experiments import own_tight_dual_native_development_summary as own_auditor
from algorithm.experiments.fused_root_dual_face_development_summary import _audit_fused_root_result
from skill.tests.test_fused_root_dual_face_timing import (
    affine_records, fresh_harness, fresh_records, independent_harness, independent_records,
    old_harness, previous_records, zero_harness, zero_records,
    parent_harness, parent_records, face_parent_harness, face_parent_records,
    native_parent_harness, native_parent_records,
    settled_parent_harness, settled_parent_records,
    harness as fused_parent_harness, records as fused_parent_records,
)


@pytest.fixture
def frozen_sources(monkeypatch, tmp_path):
    old = json.loads(timing._protocol_path(timing.PREVIOUS_OUTPUT).read_text())
    parent = json.loads(timing._protocol_path(timing.PREVIOUS_DEVELOPMENT_OUTPUT).read_text())
    for protocol in (old, parent):
        protocol["execution_sources"] = {"old.py": "frozen settled source\n"}
    development = deepcopy(parent)
    development["registered_at_utc"] = "new own-tight development"
    development["prior_decisions"]["fused_root_dual_face_timing"] = "FUSED_ROOT_DUAL_FACE_LOCAL_TIMING_HOLD"
    development["subsequent_timing_plan"] = dict(order_seed=2026092904, labels=list(timing.LABELS),
        repeats=6, warmups_per_label=2, expected_timed_calls=168120, expected_warmup_calls=320,
        nonregression_limit=1.05, trigger_speedup_limit=.95)
    for cell in development["cells"]:
        for index, state in enumerate(cell["states"]):
            state.update(own_tight_source_index=index, own_tight_source_state_id=state["state_id"])
    flags = dict.fromkeys(("execution_checks_passed", "references_complete", "quality_gate_passed",
                          "development_gate_passed", "work_gate_passed"), True)
    quality = dict(decision="OWN_TIGHT_DUAL_NATIVE_DEVELOPMENT_ADVANCE", **flags)
    parent_result = dict(decision="FUSED_ROOT_DUAL_FACE_DEVELOPMENT_ADVANCE", **flags)
    prior = dict(decision="FUSED_ROOT_DUAL_FACE_LOCAL_TIMING_HOLD", local_timing_gate_passed=False,
        **dict.fromkeys(("execution_checks_passed", "references_complete", "quality_gate_passed",
                        "repeat_paths_stable", "frozen_residual_results_reproduced"), True))
    old_path, parent_path, dev_path = (tmp_path / name for name in (
        "old_timing.json", "old_development.json", "new_development.json"))
    for path, protocol, result in ((old_path, old, prior), (parent_path, parent, parent_result),
                                  (dev_path, development, quality)):
        timing._protocol_path(path).write_text(json.dumps(protocol))
        path.write_text(json.dumps(result))
    (tmp_path / "old.py").write_text("frozen settled source\n")
    (tmp_path / "timer.py").write_text("new ten-arm source\n")
    for name, value in (("ROOT", tmp_path), ("PREVIOUS_OUTPUT", old_path),
                       ("PREVIOUS_DEVELOPMENT_OUTPUT", parent_path), ("DEVELOPMENT_OUTPUT", dev_path),
                       ("SOURCES", ("timer.py",))):
        monkeypatch.setattr(timing, name, value)
    models = [SimpleNamespace(snapshot=lambda model=deepcopy(model): deepcopy(model)) for model in old["models"]]
    return SimpleNamespace(old=old, parent=parent, parent_path=parent_path,
                           development=development, models=models, dev_path=dev_path, quality=quality)


def test_registers_all2802_states_ten_arms_and_every_previous_insertion_law(frozen_sources):
    s = frozen_sources
    protocol = timing.build_protocol(s.models)
    assert (protocol["expected_state_count"], protocol["expected_timed_calls"], protocol["expected_warmup_calls"],
            protocol["expected_selector_calls"]) == (2802, 168120, 320, 168440)
    assert timing.LABELS == summary.LABELS == ("candidate", "fused", "settled", "native", "face", "early", "zero", "affine", "branch", "long")
    assert protocol["order_seed"] == 2026092904
    assert protocol["expected_group_counts"] == s.old["expected_group_counts"]
    assert protocol["focus_keys"] == s.development["focus_keys"]
    assert protocol["prior_decisions"]["fused_root_dual_face_timing"] == "FUSED_ROOT_DUAL_FACE_LOCAL_TIMING_HOLD"
    assert protocol["reference_sources"]["own_tight_development"] == {
        "path": s.dev_path.name, "registered_at_utc": s.development["registered_at_utc"]}
    for cell in protocol["cells"]:
        for blocks in cell["orders"]["label_orders"]:
            assert len({block.index("fused") for block in blocks}) == 6
            strip = lambda ignored: [[label for label in block if label not in ignored] for block in blocks]
            assert len({block.index("settled") for block in strip(("fused",))}) == 6
            assert len({block.index("native") for block in strip(("fused", "settled"))}) == 6
            assert len({block.index("face") for block in strip(("fused", "settled", "native"))}) == 6
            assert Counter(block.index("early") for block in strip(("fused", "settled", "native", "face"))) == dict.fromkeys(range(6), 1)
            assert sorted(Counter(block.index("zero") for block in strip(("fused", "settled", "native", "face", "early"))).values()) == [1, 1, 1, 1, 2]
            assert sorted(Counter(block.index("affine") for block in strip(("fused", "settled", "native", "face", "early", "zero"))).values()) == [1, 1, 2, 2]
            assert Counter(tuple(block) for block in strip(("fused", "settled", "native", "face", "early", "zero", "affine"))) == Counter(permutations(timing.ORIGINAL_LABELS))


@pytest.mark.parametrize("change", ("plan", "roster", "quality", "parent_lineage"))
def test_frozen_plan_roster_quality_and_parent_lineage(frozen_sources, change):
    s = frozen_sources
    protocol = {**timing.build_protocol(s.models), "registered_at_utc": "registered"}
    if change == "quality":
        s.dev_path.write_text(json.dumps({**s.quality, "work_gate_passed": False}))
    elif change == "parent_lineage":
        changed = deepcopy(s.parent)
        changed["cells"][0]["states"][0]["seed"] += 1
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
def harness(fused_parent_harness, monkeypatch):
    h = fused_parent_harness
    for index, (state, saved) in enumerate(zip(h.cell["states"], h.saved, strict=True)):
        metadata = dict(own_tight_source_index=index, own_tight_source_state_id=state["state_id"])
        saved.update(**deepcopy(state), **metadata, previous_fused_root_routed=deepcopy(saved["routed"]),
                     previous_fused_root_traces=deepcopy(saved["traces"]["new"]),
                     candidate_native_root={"root_lp": deepcopy(next(iter(saved["traces"]["new"].values()))),
                                            "settled_lp": None, "root_settle_work": None},
                     candidate_native_traces=deepcopy(saved["traces"]["new"]))
        state.update(**metadata, timing_source="own_tight_development", timing_source_index=index,
                     timing_source_state_id=state["state_id"])
    h.cell["orders"] = timing.timing_orders(h.model.node, 64, len(h.saved))
    h.protocol["reference_sources"] = {"own_tight_development": {"path": "new_quality.json", "registered_at_utc": "own-tight quality"}}
    parent_selector = h.selectors["candidate"]

    def fused(*args):
        result = parent_selector(*args)
        _, queue, budgets = h.calls[-1]
        h.calls[-1] = ("fused", queue, budgets)
        return result

    h.selectors["fused"] = fused

    def load(path, node, lanes, utc):
        assert (path.stem, node, lanes, utc) == ("new_quality", h.model.node, 64, "own-tight quality")
        return {"states": h.saved}

    monkeypatch.setattr(timing, "load_cell", load)
    monkeypatch.setattr(timing, "runtime_provenance", lambda: {"cpu_affinity": [0]})
    return h


def test_each_of_ten_actual_mock_calls_has_its_own_clock_before_serialization(harness, monkeypatch):
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
    assert len(h.calls) == len(h.recorded) == 200
    assert Counter(label for label, _, _ in h.calls) == dict.fromkeys(timing.LABELS, 20)
    for state, saved in zip(row["states"], h.saved, strict=True):
        assert state["expected_results"]["fused"] == saved["previous_fused_root_routed"]
        assert state["previous_fused_root_traces"] == saved["traces"]["new"]
        assert state["candidate_native_root"] == saved["candidate_native_root"]
        assert state["candidate_native_traces"] == saved["candidate_native_traces"]
        assert ("prior_routed" in state) == ("prior_routed" in saved)
    count = 0
    events = h.events[h.events.index("cpu"):]
    for index, event in enumerate(events):
        if event == "selector":
            count += 1
            assert events[index - 2:index + 3] == ["cpu", "wall", "selector", "wall", "cpu"]
        elif event in ("record", "asdict"):
            assert count % 10 == 0
    assert count == 180


def test_production_timing_uses_the_independent_own_tight_auditor():
    assert summary._audit_result is own_auditor._audit_own_tight_result
    assert summary._work is own_auditor._work


def _mock_native_work(stage):
    """Declare cumulative mock cache writes; native certificate math has its own tests."""
    stage["native_own_tight_work"] = dict.fromkeys(own_auditor.OWN_TIGHT_COUNTERS, 0)
    stage["native_own_tight_work"]["own_bound_cache_writes"] = stage["selection"]["lp_calls"]
    stage["native_own_tight_evidence"] = []


def _strip_mock_own(result):
    result = deepcopy(result)
    for name in ("root_lp", "settled_lp", "initial_lp", "final_lp"):
        if result[name] is not None:
            result[name].pop("native_own_tight_work")
            result[name].pop("native_own_tight_evidence")
    return result


@pytest.fixture
def records(fused_parent_records, monkeypatch):
    # Isolate the timer's order, gates and cumulative-fee contract. The old
    # no-elimination paths still receive their genuine independent audit;
    # own certificate/prefix mathematics belongs to the development auditor.
    protocol, rows = fused_parent_records
    row, cell = rows[0], protocol["cells"][0]
    for state in row["states"]:
        state["previous_fused_root_routed"] = deepcopy(state["first_results"]["candidate"])
        state["previous_fused_root_traces"] = deepcopy(state["traces"]["new"])
        for field in ("expected_results", "first_results", "repeat_mismatches", "metrics", "wall_ns", "cpu_ns"):
            state[field]["fused"] = deepcopy(state[field]["candidate"])
        state["wall_ns"]["fused"], state["cpu_ns"]["fused"] = [100] * 6, [90] * 6
        for result in (state["first_results"]["candidate"], state["expected_results"]["candidate"]):
            for name in ("root_lp", "settled_lp", "initial_lp", "final_lp"):
                if result[name] is not None:
                    _mock_native_work(result[name])
        state["candidate_native_root"] = {name: deepcopy(state["first_results"]["candidate"][name])
            for name in ("root_lp", "settled_lp", "root_settle_work")}
        state["candidate_native_traces"] = deepcopy(state["previous_fused_root_traces"])
        for stage in state["candidate_native_traces"].values():
            _mock_native_work(stage)
    warmups, row["warmups"] = row["warmups"], []
    for repeat in range(2):
        block = {item["label"]: deepcopy(item) for item in warmups[repeat * 9:(repeat + 1) * 9]}
        block["fused"] = {**deepcopy(block["candidate"]), "label": "fused"}
        for name in ("root_lp", "settled_lp", "initial_lp", "final_lp"):
            if block["candidate"]["result"][name] is not None:
                _mock_native_work(block["candidate"]["result"][name])
        row["warmups"].extend(block[label] for label in timing.LABELS)
    cell["orders"] = timing.timing_orders(row["node"], row["lanes"], len(row["states"]))
    row["state_orders"] = deepcopy(cell["orders"]["state_orders"])
    for index, state in enumerate(row["states"]):
        state["blocks"] = [dict(repeat=repeat, order=block) for repeat, block in enumerate(cell["orders"]["label_orders"][index])]
    protocol.update(labels=list(timing.LABELS), expected_timed_calls=600, expected_warmup_calls=20, expected_selector_calls=620)
    protocol["prior_decisions"]["fused_root_dual_face_timing"] = "FUSED_ROOT_DUAL_FACE_LOCAL_TIMING_HOLD"

    def audit(result, state, protocol, model, lanes, label, expected):
        metric, errors, quality = _audit_fused_root_result(
            _strip_mock_own(result) if label == "candidate" else result,
            state, protocol, model, lanes, "candidate" if label == "fused" else label, expected)
        if label == "fused" and result != state["previous_fused_root_routed"]:
            errors.append("mock timer replaced the genuine fused control")
        return metric, errors, quality

    monkeypatch.setattr(summary, "_audit_result", audit)
    return protocol, rows


def test_summary_pays_only_final_own_work_and_keeps_fused_work_genuine(records):
    protocol, rows = records
    r = summary.summarize(protocol, rows)
    assert r["decision"] == "OWN_TIGHT_DUAL_NATIVE_LOCAL_TIMING_ADVANCE", r["accounting_failures"][:1]
    assert r["fused_timing_gate_passed"] and r["settled_timing_gate_passed"] and r["primary_timing_gate_passed"]
    assert (r["timed_selector_calls"], r["warmup_calls"], r["executed_selector_calls"]) == (600, 20, 620)
    assert (r["borrowed_native_snapshots"], r["borrowed_candidate_native_snapshots"],
            r["borrowed_candidate_native_root_records"], r["borrowed_fused_root_router_count"]) == (160, 40, 10, 10)
    assert set(r["executed_native_own_tight_work"]) == set(own_auditor.OWN_TIGHT_FEE_COUNTERS)
    assert set(r["executed_native_own_tight_diagnostics"]) == set(own_auditor.OWN_TIGHT_COUNTERS) - set(own_auditor.OWN_TIGHT_FEE_COUNTERS)
    expected = 6 * sum(state["first_results"]["candidate"]["final_lp"]["native_own_tight_work"]["own_bound_cache_writes"]
                       for state in rows[0]["states"])
    expected += sum(item["result"]["final_lp"]["native_own_tight_work"]["own_bound_cache_writes"]
                    for item in rows[0]["warmups"] if item["label"] == "candidate")
    assert expected > 0 and r["executed_native_own_tight_work"]["own_bound_cache_writes"] == expected
    assert "native_own_tight_work" not in r["work"]["fused"]
    assert r["executed_fused_routing_preparation_work"]["root_snapshot_heap_entries_read"] == 24
    assert r["executed_settled_routing_preparation_work"]["root_snapshot_heap_entries_read"] == 48
    assert r["prior_decisions"]["fused_root_dual_face_timing"] == "FUSED_ROOT_DUAL_FACE_LOCAL_TIMING_HOLD"


@pytest.mark.parametrize("group", ("independent_fresh", "historical_regression", "new_trigger"))
def test_other_comparator_gains_cannot_hide_a_real_fused_regression(records, group):
    protocol, rows = records
    for state in rows[0]["states"]:
        if state["timing_group"] == group:
            state["wall_ns"]["fused"] = [60] * 6
    r = summary.summarize(protocol, rows)
    assert r["decision"] == "OWN_TIGHT_DUAL_NATIVE_LOCAL_TIMING_HOLD"
    assert not r["fused_timing_gate_passed"] and r["settled_timing_gate_passed"] and r["primary_timing_gate_passed"]


@pytest.mark.parametrize("label", ("settled", "native", "zero"))
def test_new_fused_control_never_replaces_old_failed_comparators(records, label):
    protocol, rows = records
    for state in rows[0]["states"]:
        if state["timing_group"] == "ordinary_development":
            state["wall_ns"][label] = [60] * 6
    r = summary.summarize(protocol, rows)
    assert r["decision"] == "OWN_TIGHT_DUAL_NATIVE_LOCAL_TIMING_HOLD"
    assert r["fused_timing_gate_passed"] and not r[label + "_timing_gate_passed"]


@pytest.mark.parametrize("mutation", ("lost_fused", "fused_positions", "invented_old_own_work"))
def test_fused_observations_orders_and_actual_old_fees_cannot_be_replaced(records, mutation):
    protocol, rows = records
    if mutation == "invented_old_own_work":
        state = rows[0]["states"][-1]
        _mock_native_work(state["first_results"]["fused"]["root_lp"])
        state["expected_results"]["fused"] = deepcopy(state["first_results"]["fused"])
        assert summary.summarize(protocol, rows)["decision"] == "INVALID_TIMING_EXECUTION"
    else:
        blocks = protocol["cells"][0]["orders"]["label_orders"][0]
        if mutation == "lost_fused":
            blocks[0].remove("fused")
        else:
            for block in blocks:
                block.remove("fused")
                block.insert(0, "fused")
        with pytest.raises(ValueError):
            summary.summarize(protocol, rows)
