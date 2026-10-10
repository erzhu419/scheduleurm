"""Pay both new searches while retaining genuine direct and older sources."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from algorithm.experiments import aligned_hardware_small_dimension_exact_native_development as runner
from algorithm.experiments import aligned_hardware_direct_probe_own_tight_native_development as parent_runner
from skill.tests.test_direct_probe_own_tight_native_development import harness as parent_harness


def harness(monkeypatch, group="independent_fresh", settled=True):
    model, cell, protocol, selector, audit, saved, calls = parent_harness(monkeypatch, group, settled)
    saved.update(previous_own_tight_routed=deepcopy(saved["routed"]),
        previous_own_tight_native_root=deepcopy(saved["candidate_native_root"]),
        previous_own_tight_native_traces=deepcopy(saved["candidate_native_traces"]),
        routed={"actual_direct": True}, candidate_native_root={"parent_direct_root": True},
        candidate_native_traces={"parent_direct_ladder": True},
        direct_probe_source_index=0, direct_probe_source_state_id="original own")
    state = cell["states"][0]
    state.update(direct_probe_source_index=0, direct_probe_source_state_id="original own",
        small_dimension_source_index=0, small_dimension_source_state_id=state["state_id"])
    monkeypatch.setattr(runner, "load_cell", parent_runner.load_cell)
    monkeypatch.setattr(runner, "runtime_provenance", parent_runner.runtime_provenance)
    return model, cell, protocol, selector, audit, saved, calls


@pytest.mark.parametrize("group", ("independent_fresh", "fresh_confirmation", "trigger_control"))
@pytest.mark.parametrize("settled", (True, False))
def test_new_native_prefixes_pay_two_searches_and_keep_all_real_controls(monkeypatch, group, settled):
    model, cell, protocol, selector, audit, saved, calls = harness(monkeypatch, group, settled)
    row = runner.run_cell(model, cell, protocol, None, selector, audit, progress=False)
    assert calls == ["audit_construct", ("root", 2), ("advance", 3), ("advance", 144), ("advance", 256), "whole"]
    assert (row["native_advance_calls"], row["candidate_calls"], row["native_search_constructions"]) == (4, 1, 2)
    assert row["root_settle_calls"] == int(settled)
    assert row["new_reference_calls"] == row["new_oracle_calls"] == 0
    state = row["states"][0]
    assert state["previous_direct_probe_routed"] == saved["routed"]
    assert state["previous_direct_probe_native_root"] == saved["candidate_native_root"]
    assert state["previous_direct_probe_native_traces"] == saved["candidate_native_traces"]
    assert state["previous_own_tight_routed"] == saved["previous_own_tight_routed"] != saved["routed"]
    assert state["candidate_native_root"]["root_lp"] == {"budget": 2, "source": "raw"}
    assert state["candidate_native_traces"]["2"] == {"budget": 2, "source": "settled" if settled else "raw"}
    for key in ("traces", "old_routed", "original_affine_traces", "original_affine_routed",
                "previous_zero_traces", "previous_zero_routed", "previous_early_routed",
                "previous_dual_face_routed", "previous_dual_face_traces", "previous_native_root_routed",
                "previous_native_root_traces", "previous_settled_root_routed", "previous_settled_root_traces",
                "previous_fused_root_routed", "previous_fused_root_traces",
                "previous_own_tight_native_root", "previous_own_tight_native_traces"):
        assert state[key] == saved[key]
    assert ("prior_routed" in state) == (group == "trigger_control")


@pytest.mark.parametrize("field,value", (("queue", [9, 9]), ("caps", [2, 4, 144, 256]),
    ("direct_probe_source_state_id", "wrong direct predecessor")))
def test_source_metadata_drift_stops_before_new_native_calls(monkeypatch, field, value):
    model, cell, protocol, selector, audit, saved, calls = harness(monkeypatch)
    saved[field] = value
    with pytest.raises(ValueError, match="frozen direct-probe development source"):
        runner.run_cell(model, cell, protocol, None, selector, audit, progress=False)
    assert calls == []


@pytest.fixture
def frozen_sources(monkeypatch, tmp_path):
    parent = json.loads(runner._protocol_path(runner.PREVIOUS_OUTPUT).read_text())
    timing = json.loads(runner._protocol_path(runner.PREVIOUS_TIMING).read_text())
    for protocol in (parent, timing):
        protocol["execution_sources"] = {"old.py": "unchanged direct source\n"}
    flags = dict.fromkeys(("execution_checks_passed", "certificate_checks_passed", "references_complete",
                          "quality_gate_passed", "work_gate_passed", "development_gate_passed",
                          "repeat_paths_stable", "cpu_affinity_preserved", "frozen_residual_results_reproduced"), True)
    result = dict(decision="DIRECT_PROBE_OWN_TIGHT_NATIVE_DEVELOPMENT_ADVANCE", executed_root_settle_calls=2265, **flags)
    clocks = dict(decision="DIRECT_PROBE_OWN_TIGHT_NATIVE_LOCAL_TIMING_HOLD", prior_decisions=timing["prior_decisions"], **flags)
    dev_path, timing_path = tmp_path / "parent.json", tmp_path / "parent_timing.json"
    for path, protocol, value in ((dev_path, parent, result), (timing_path, timing, clocks)):
        runner._protocol_path(path).write_text(json.dumps(protocol))
        path.write_text(json.dumps(value))
    (tmp_path / "old.py").write_text("unchanged direct source\n")
    (tmp_path / "new.py").write_text("new exact primitive source\n")
    for name, value in (("ROOT", tmp_path), ("PREVIOUS_OUTPUT", dev_path), ("PREVIOUS_TIMING", timing_path),
                       ("SOURCES", ("new.py",))):
        monkeypatch.setattr(runner, name, value)
    models = [SimpleNamespace(snapshot=lambda model=deepcopy(model): deepcopy(model)) for model in parent["models"]]
    return SimpleNamespace(parent=parent, models=models, dev_path=dev_path, result=result)


def test_registration_keeps_roster_budgets_focuses_and_descriptive_parent_root_count(frozen_sources):
    source = frozen_sources
    protocol = runner.build_protocol(source.models)
    assert (protocol["expected_state_count"], protocol["expected_selector_calls"], protocol["expected_native_advance_calls"],
            protocol["expected_native_constructions"], protocol["expected_interface_calls"]) == (2802, 2802, 11208, 5604, 14010)
    assert "expected_root_settle_calls" not in protocol and protocol["parent_root_settle_calls"] == 2265
    assert protocol["expected_group_counts"] == source.parent["expected_group_counts"]
    assert [(cell["node"], cell["lanes"]) for cell in protocol["cells"]] == [(cell["node"], cell["lanes"]) for cell in source.parent["cells"]]
    for cell, original in zip(protocol["cells"], source.parent["cells"], strict=True):
        assert [{key: value for key, value in state.items() if not key.startswith("small_dimension_source_")}
                for state in cell["states"]] == original["states"]
    focuses = {(point["node"], point["lanes"], point["state_id"]) for point in protocol["focus_keys"]}
    assert len(focuses) == len(protocol["focus_keys"])
    assert {(point["node"], point["lanes"], point["state_id"]) for point in source.parent["focus_keys"]} <= focuses
    for node, lanes, queue in (("cpu_a", 64, [184, 78, 235]), ("cpu_a", 64, [192, 192, 317]),
            ("gpu_b", 64, [88, 19, 118]), ("gpu_b", 32, [61, 0, 35]), ("cpu_a", 8, [27, 15, 34])):
        assert any((point["node"], point["lanes"], point["queue"]) == (node, lanes, queue) for point in protocol["focus_keys"])
    assert protocol["subsequent_timing_plan"] == runner.TIMING_PLAN
    assert protocol["prior_decisions"]["direct_probe_own_tight_native_timing"] == "DIRECT_PROBE_OWN_TIGHT_NATIVE_LOCAL_TIMING_HOLD"


def test_quality_failure_blocks_registration(frozen_sources):
    source = frozen_sources
    source.dev_path.write_text(json.dumps({**source.result, "work_gate_passed": False}))
    with pytest.raises(ValueError, match="quality-valid direct-probe"):
        runner.build_protocol(source.models)


def test_twelve_arm_plan_is_frozen_before_outcomes():
    assert runner.TIMING_PLAN["expected_timed_calls"] == 2802 * 6 * 12
    assert runner.TIMING_PLAN["expected_warmup_calls"] == 16 * 2 * 12
    assert len(set(runner.TIMING_PLAN["labels"])) == 12
    assert runner.TIMING_PLAN["order_seed"] == 2026092906
    assert runner.TIMING_PLAN["labels"][:3] == ["candidate", "direct", "own"]
