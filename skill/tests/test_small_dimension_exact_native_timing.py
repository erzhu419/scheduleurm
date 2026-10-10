"""Twelve new clocks retain direct prefixes, exact-LP fees and every prior gate."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from itertools import permutations
import json
from types import SimpleNamespace

import pytest

from algorithm.experiments import aligned_hardware_small_dimension_exact_native_timing as timing
from algorithm.experiments import small_dimension_exact_native_timing_summary as summary
from algorithm.experiments import small_dimension_exact_native_development_summary as auditor
from algorithm.experiments import direct_probe_own_tight_native_timing_summary as parent_summary
from skill.tests.test_direct_probe_own_tight_native_timing import (
    affine_records, fresh_harness, fresh_records, independent_harness, independent_records,
    old_harness, previous_records, zero_harness, zero_records,
    parent_harness, parent_records, face_parent_harness, face_parent_records,
    native_parent_harness, native_parent_records, settled_parent_harness, settled_parent_records,
    fused_parent_harness, fused_parent_records,
    own_parent_harness, own_parent_records,
    harness as direct_parent_harness, records as direct_parent_records,
)


@pytest.fixture
def frozen_sources(monkeypatch, tmp_path):
    old = json.loads(timing._protocol_path(timing.PREVIOUS_OUTPUT).read_text())
    parent = json.loads(timing._protocol_path(timing.PREVIOUS_DEVELOPMENT_OUTPUT).read_text())
    for protocol in (old, parent):
        protocol["execution_sources"] = {"old.py": "frozen direct source\n"}
    development = deepcopy(parent)
    development["registered_at_utc"] = "new small-dimension development"
    development["prior_decisions"]["direct_probe_own_tight_native_timing"] = "DIRECT_PROBE_OWN_TIGHT_NATIVE_LOCAL_TIMING_HOLD"
    development["subsequent_timing_plan"] = dict(order_seed=2026092906, labels=list(timing.LABELS),
        repeats=6, warmups_per_label=2, expected_timed_calls=201744, expected_warmup_calls=384,
        nonregression_limit=1.05, trigger_speedup_limit=.95)
    for cell in development["cells"]:
        for index, state in enumerate(cell["states"]):
            state.update(small_dimension_source_index=index, small_dimension_source_state_id=state["state_id"])
    flags = dict.fromkeys(("execution_checks_passed", "references_complete", "quality_gate_passed",
                          "development_gate_passed", "work_gate_passed"), True)
    quality = dict(decision="SMALL_DIMENSION_EXACT_NATIVE_DEVELOPMENT_ADVANCE", **flags)
    parent_result = dict(decision="DIRECT_PROBE_OWN_TIGHT_NATIVE_DEVELOPMENT_ADVANCE", **flags)
    prior = dict(decision="DIRECT_PROBE_OWN_TIGHT_NATIVE_LOCAL_TIMING_HOLD", local_timing_gate_passed=False,
        **dict.fromkeys(("execution_checks_passed", "references_complete", "quality_gate_passed",
                        "repeat_paths_stable", "frozen_residual_results_reproduced"), True))
    old_path, parent_path, dev_path = (tmp_path / name for name in (
        "old_timing.json", "old_development.json", "new_development.json"))
    for path, protocol, result in ((old_path, old, prior), (parent_path, parent, parent_result),
                                  (dev_path, development, quality)):
        timing._protocol_path(path).write_text(json.dumps(protocol))
        path.write_text(json.dumps(result))
    (tmp_path / "old.py").write_text("frozen direct source\n")
    (tmp_path / "timer.py").write_text("new twelve-arm source\n")
    for name, value in (("ROOT", tmp_path), ("PREVIOUS_OUTPUT", old_path),
                       ("PREVIOUS_DEVELOPMENT_OUTPUT", parent_path), ("DEVELOPMENT_OUTPUT", dev_path),
                       ("SOURCES", ("timer.py",))):
        monkeypatch.setattr(timing, name, value)
    models = [SimpleNamespace(snapshot=lambda model=deepcopy(model): deepcopy(model)) for model in old["models"]]
    return SimpleNamespace(old=old, parent=parent, parent_path=parent_path,
                           development=development, models=models, dev_path=dev_path, quality=quality)


def test_registers_all2802_twelve_arms_and_relabels_the_parent_candidate_as_direct(frozen_sources):
    s = frozen_sources
    protocol = timing.build_protocol(s.models)
    assert (protocol["expected_state_count"], protocol["expected_timed_calls"], protocol["expected_warmup_calls"],
            protocol["expected_selector_calls"]) == (2802, 201744, 384, 202128)
    assert timing.LABELS == summary.LABELS == ("candidate", "direct", "own", "fused", "settled", "native", "face", "early", "zero", "affine", "branch", "long")
    assert protocol["order_seed"] == 2026092906
    assert protocol["expected_group_counts"] == s.old["expected_group_counts"]
    assert protocol["focus_keys"] == s.development["focus_keys"]
    assert protocol["prior_decisions"]["direct_probe_own_tight_native_timing"] == "DIRECT_PROBE_OWN_TIGHT_NATIVE_LOCAL_TIMING_HOLD"
    assert protocol["reference_sources"]["small_dimension_development"] == {
        "path": s.dev_path.name, "registered_at_utc": s.development["registered_at_utc"]}
    for cell in protocol["cells"]:
        for blocks in cell["orders"]["label_orders"]:
            assert len({block.index("candidate") for block in blocks}) == 6
            strip = lambda ignored: [[label for label in block if label not in ignored] for block in blocks]
            assert len({block.index("own") for block in strip(("candidate",))}) == 6
            assert len({block.index("fused") for block in strip(("candidate", "own"))}) == 6
            assert len({block.index("settled") for block in strip(("candidate", "own", "fused"))}) == 6
            assert len({block.index("native") for block in strip(("candidate", "own", "fused", "settled"))}) == 6
            assert len({block.index("face") for block in strip(("candidate", "own", "fused", "settled", "native"))}) == 6
            assert Counter(block.index("early") for block in strip(("candidate", "own", "fused", "settled", "native", "face"))) == dict.fromkeys(range(6), 1)
            assert sorted(Counter(block.index("zero") for block in strip(("candidate", "own", "fused", "settled", "native", "face", "early"))).values()) == [1, 1, 1, 1, 2]
            assert sorted(Counter(block.index("affine") for block in strip(("candidate", "own", "fused", "settled", "native", "face", "early", "zero"))).values()) == [1, 1, 2, 2]
            assert Counter(tuple(block) for block in strip(("candidate", "own", "fused", "settled", "native", "face", "early", "zero", "affine"))) == Counter(permutations(timing.ORIGINAL_LABELS))


@pytest.mark.parametrize("change", ("plan", "quality", "parent_lineage"))
def test_plan_quality_and_genuine_parent_lineage_block_invalid_registration(frozen_sources, change):
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
        changed["subsequent_timing_plan"]["expected_warmup_calls"] -= 1
        timing._protocol_path(s.dev_path).write_text(json.dumps(changed))
    with pytest.raises(ValueError):
        timing.verify_protocol(protocol, s.models)


@pytest.fixture
def harness(direct_parent_harness, monkeypatch):
    h = direct_parent_harness
    for index, (state, saved) in enumerate(zip(h.cell["states"], h.saved, strict=True)):
        metadata = dict(small_dimension_source_index=index, small_dimension_source_state_id=state["state_id"])
        saved.update(**deepcopy(state), **metadata, previous_direct_probe_routed=deepcopy(saved["routed"]),
                     previous_direct_probe_native_root=deepcopy(saved["candidate_native_root"]),
                     previous_direct_probe_native_traces=deepcopy(saved["candidate_native_traces"]))
        state.update(**metadata, timing_source="small_dimension_development", timing_source_index=index,
                     timing_source_state_id=state["state_id"])
    h.cell["orders"] = timing.timing_orders(h.model.node, 64, len(h.saved))
    h.protocol["reference_sources"] = {"small_dimension_development": {"path": "new_quality.json", "registered_at_utc": "small-dimension quality"}}
    parent_selector = h.selectors["candidate"]

    def direct(*args):
        result = parent_selector(*args)
        _, queue, budgets = h.calls[-1]
        h.calls[-1] = ("direct", queue, budgets)
        return result

    h.selectors["direct"] = direct

    def load(path, node, lanes, utc):
        assert (path.stem, node, lanes, utc) == ("new_quality", h.model.node, 64, "small-dimension quality")
        return {"states": h.saved}

    monkeypatch.setattr(timing, "load_cell", load)
    monkeypatch.setattr(timing, "runtime_provenance", lambda: {"cpu_affinity": [0]})
    return h


def test_all_twelve_calls_include_cold_work_before_serialization_and_keep_real_direct_prefixes(harness, monkeypatch):
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
    assert len(h.calls) == len(h.recorded) == 240
    assert Counter(label for label, _, _ in h.calls) == dict.fromkeys(timing.LABELS, 20)
    for state, saved in zip(row["states"], h.saved, strict=True):
        assert state["expected_results"]["direct"] == saved["previous_direct_probe_routed"]
        for name in ("previous_direct_probe_native_root", "previous_direct_probe_native_traces", "candidate_native_root", "candidate_native_traces"):
            assert state[name] == saved[name]
        assert ("prior_routed" in state) == ("prior_routed" in saved)
    count = 0
    events = h.events[h.events.index("cpu"):]
    for index, event in enumerate(events):
        if event == "selector":
            count += 1
            assert events[index - 2:index + 3] == ["cpu", "wall", "selector", "wall", "cpu"]
        elif event in ("record", "asdict"):
            assert count % 12 == 0
    assert count == 216


def test_production_bindings_and_actual_auditor_are_independent():
    from algorithm.small_dimension_exact_native_frame_candidates import select_small_dimension_exact_native_frame
    from algorithm.direct_probe_own_tight_native_frame_candidates import select_direct_probe_own_tight_native_frame
    from algorithm.own_tight_dual_native_frame_candidates import select_own_tight_dual_native_frame
    from algorithm.fused_root_dual_face_frame_candidates import select_fused_root_dual_face_frame
    selectors = timing._selectors()
    assert selectors["candidate"] is select_small_dimension_exact_native_frame
    assert selectors["direct"] is select_direct_probe_own_tight_native_frame
    assert selectors["own"] is select_own_tight_dual_native_frame
    assert selectors["fused"] is select_fused_root_dual_face_frame
    assert summary._audit_result is auditor._audit_small_dimension_result
    assert summary._work is auditor._work


def _mock_small_work(result):
    for name in ("root_lp", "settled_lp", "initial_lp", "final_lp"):
        stage = result[name]
        if stage is not None:
            stage["small_dimension_lp_work"] = dict.fromkeys(auditor.SMALL_DIMENSION_LP_COUNTERS, 0)
            stage["small_dimension_lp_work"]["generic_lp_calls"] = stage["selection"]["lp_calls"]
            stage["small_dimension_lp_work"]["active_bound_reads"] = 2 * stage["selection"]["lp_calls"]
            stage["small_dimension_lp_work"]["active_bound_comparisons"] = stage["selection"]["lp_calls"]
            stage["small_dimension_lp_evidence"] = []


def _strip_small(result):
    result = deepcopy(result)
    for name in ("root_lp", "settled_lp", "initial_lp", "final_lp"):
        if result[name] is not None:
            result[name].pop("small_dimension_lp_work")
            result[name].pop("small_dimension_lp_evidence")
    return result


@pytest.fixture
def records(direct_parent_records, monkeypatch):
    protocol, rows = direct_parent_records
    row, cell = rows[0], protocol["cells"][0]
    parent_audit = parent_summary._audit_result  # Parent fixture's explicit mock-native hook.
    for state in row["states"]:
        state["previous_direct_probe_routed"] = deepcopy(state["first_results"]["candidate"])
        state["previous_direct_probe_native_root"] = deepcopy(state["candidate_native_root"])
        state["previous_direct_probe_native_traces"] = deepcopy(state["candidate_native_traces"])
        for field in ("expected_results", "first_results", "repeat_mismatches", "metrics", "wall_ns", "cpu_ns"):
            state[field]["direct"] = deepcopy(state[field]["candidate"])
        state["wall_ns"]["direct"], state["cpu_ns"]["direct"] = [100] * 6, [90] * 6
        for result in (state["first_results"]["candidate"], state["expected_results"]["candidate"]):
            _mock_small_work(result)
    warmups, row["warmups"] = row["warmups"], []
    for repeat in range(2):
        block = {item["label"]: deepcopy(item) for item in warmups[repeat * 11:(repeat + 1) * 11]}
        block["direct"] = {**deepcopy(block["candidate"]), "label": "direct"}
        _mock_small_work(block["candidate"]["result"])
        row["warmups"].extend(block[label] for label in timing.LABELS)
    cell["orders"] = timing.timing_orders(row["node"], row["lanes"], len(row["states"]))
    row["state_orders"] = deepcopy(cell["orders"]["state_orders"])
    for index, state in enumerate(row["states"]):
        state["blocks"] = [dict(repeat=repeat, order=block) for repeat, block in enumerate(cell["orders"]["label_orders"][index])]
    protocol.update(labels=list(timing.LABELS), expected_timed_calls=720, expected_warmup_calls=24, expected_selector_calls=744)
    protocol["prior_decisions"]["direct_probe_own_tight_native_timing"] = "DIRECT_PROBE_OWN_TIGHT_NATIVE_LOCAL_TIMING_HOLD"

    def audit(result, state, protocol, model, lanes, label, expected):
        metric, errors, quality = parent_audit(_strip_small(result) if label == "candidate" else result,
            state, protocol, model, lanes, "candidate" if label == "direct" else label, expected)
        if label == "direct" and result != state["previous_direct_probe_routed"]:
            errors.append("mock timer replaced the genuine direct control")
        return metric, errors, quality

    monkeypatch.setattr(summary, "_audit_result", audit)
    return protocol, rows


def test_summary_counts24_prefixes_and_pays_only_final_small_lp_work(records):
    protocol, rows = records
    r = summary.summarize(*records)
    assert r["decision"] == "SMALL_DIMENSION_EXACT_NATIVE_LOCAL_TIMING_ADVANCE", r["accounting_failures"][:1]
    assert all(r[label + "_timing_gate_passed"] for label in timing.LABELS[1:] if label not in ("branch", "long"))
    assert r["primary_timing_gate_passed"]
    assert (r["timed_selector_calls"], r["warmup_calls"], r["executed_selector_calls"]) == (720, 24, 744)
    assert (r["borrowed_native_snapshots"], r["borrowed_old_native_snapshots"], r["borrowed_own_native_snapshots"],
            r["borrowed_direct_native_snapshots"], r["borrowed_candidate_native_snapshots"],
            r["borrowed_own_native_root_records"], r["borrowed_direct_native_root_records"],
            r["borrowed_candidate_native_root_records"]) == (240, 120, 40, 40, 40, 10, 10, 10)
    assert set(r["executed_root_settle_diagnostics"]) == set(auditor.ROOT_SETTLE_COUNTERS) - set(auditor.ROOT_SETTLE_FEE_COUNTERS)
    assert "stopped_for_branch" in r["executed_direct_root_settle_diagnostics"]
    assert "stopped_for_branch" not in r["executed_own_root_settle_diagnostics"]
    assert "stopped_for_branch" not in r["executed_fused_root_settle_diagnostics"]
    assert "stopped_for_branch" not in r["executed_settled_root_settle_diagnostics"]
    assert set(r["executed_native_own_tight_work"]) == set(auditor.OWN_TIGHT_FEE_COUNTERS)
    assert r["executed_native_own_tight_work"] == r["executed_own_control_native_work"]
    assert r["executed_own_control_native_work"]["own_bound_cache_writes"] > 0
    assert "native_own_tight_work" not in r["work"]["fused"]
    assert set(r["executed_small_dimension_lp_work"]) == set(auditor.SMALL_DIMENSION_LP_FEE_COUNTERS)
    assert set(r["executed_small_dimension_lp_diagnostics"]) == set(auditor.SMALL_DIMENSION_LP_DIAGNOSTIC_COUNTERS)
    expected = 6 * sum(state["first_results"]["candidate"]["final_lp"]["small_dimension_lp_work"]["active_bound_reads"]
        for state in rows[0]["states"])
    expected += sum(item["result"]["final_lp"]["small_dimension_lp_work"]["active_bound_reads"]
        for item in rows[0]["warmups"] if item["label"] == "candidate")
    assert r["executed_small_dimension_lp_work"]["active_bound_reads"] == expected > 0
    assert "small_dimension_lp_work" not in r["work"]["direct"]
    assert "small_dimension_lp_work" not in r["work"]["own"]
    assert r["prior_decisions"]["direct_probe_own_tight_native_timing"] == "DIRECT_PROBE_OWN_TIGHT_NATIVE_LOCAL_TIMING_HOLD"


@pytest.mark.parametrize("group", ("independent_fresh", "fresh_confirmation", "ordinary_development", "historical_regression", "new_trigger"))
def test_gains_against_earlier_controls_do_not_hide_a_real_direct_regression(records, group):
    protocol, rows = records
    for state in rows[0]["states"]:
        if state["timing_group"] == group:
            state["wall_ns"]["direct"] = [60] * 6
    r = summary.summarize(protocol, rows)
    assert r["decision"] == "SMALL_DIMENSION_EXACT_NATIVE_LOCAL_TIMING_HOLD"
    assert not r["direct_timing_gate_passed"] and r["fused_timing_gate_passed"] and r["primary_timing_gate_passed"]


@pytest.mark.parametrize("label", ("own", "fused", "settled", "zero", "affine"))
def test_added_direct_control_preserves_all_previous_failed_comparators(records, label):
    protocol, rows = records
    for state in rows[0]["states"]:
        if state["timing_group"] == "ordinary_development":
            state["wall_ns"][label] = [60] * 6
    r = summary.summarize(protocol, rows)
    assert r["decision"] == "SMALL_DIMENSION_EXACT_NATIVE_LOCAL_TIMING_HOLD"
    assert r["direct_timing_gate_passed"] and not r[label + "_timing_gate_passed"]


@pytest.mark.parametrize("mutation", ("lost_candidate", "candidate_positions"))
def test_new_candidate_clocks_and_six_different_slots_cannot_be_removed(records, mutation):
    protocol, rows = records
    blocks = protocol["cells"][0]["orders"]["label_orders"][0]
    if mutation == "lost_candidate":
        blocks[0].remove("candidate")
    else:
        for block in blocks:
            block.remove("candidate")
            block.insert(0, "candidate")
    with pytest.raises(ValueError):
        summary.summarize(protocol, rows)
