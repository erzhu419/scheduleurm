"""Affine timing retains all frozen clocks, gates and actual preparation/update costs."""
from collections import Counter
from copy import deepcopy
from itertools import permutations
import json
from types import SimpleNamespace

import pytest

from algorithm.experiments import aligned_hardware_affine_dual_timing as timing
from algorithm.experiments import affine_dual_timing_summary as affine_summary
from algorithm.experiments import aligned_hardware_residual_capacity_timing as previous
from algorithm.experiments.affine_dual_development_summary import AFFINE_COUNTERS
from skill.tests.test_residual_capacity_timing import harness as old_harness
from skill.tests.test_residual_capacity_timing_summary import records as previous_records


@pytest.fixture
def frozen_sources(monkeypatch, tmp_path):
    # Only literal existing roster data: no numerical setup or selector execution.
    old = json.loads(timing.PREVIOUS_OUTPUT.with_name(
        timing.PREVIOUS_OUTPUT.stem + "_protocol.json").read_text())
    models = [SimpleNamespace(snapshot=lambda data=deepcopy(model): deepcopy(data)) for model in old["models"]]
    old["execution_sources"] = {"old.py": "old frozen source\n"}
    dev = deepcopy(old)
    dev["registered_at_utc"] = "affine development registered"
    dev["execution_sources"] = {"old.py": "old frozen source\n", "new.py": "new frozen source\n"}
    failures = []
    for cell in dev["cells"]:
        for index, state in enumerate(cell["states"]):
            state.update(affine_source_index=index, affine_source_state_id=state["state_id"])
            if state["historical_regression"] and len(failures) < 4:
                failures.append({"node": cell["node"], "lanes": cell["lanes"],
                                 "queue": state["queue"], "state_id": state["state_id"]})
    dev["previous_timing_failure_keys"] = failures
    dev["affine_focus_keys"] = failures + [{"node": cell["node"], "lanes": cell["lanes"],
        "queue": state["queue"], "state_id": state["state_id"]}
        for cell in old["cells"] for state in cell["states"] if state["timing_group"] == "new_trigger"][:2]
    old_result = {"decision": "PRUNE_FIRST_DUAL_LOCAL_TIMING_HOLD", **dict.fromkeys((
        "execution_checks_passed", "references_complete", "quality_gate_passed", "repeat_paths_stable", "frozen_residual_results_reproduced"), True)}
    dev_result = {"decision": "AFFINE_DUAL_DEVELOPMENT_ADVANCE", **dict.fromkeys((
        "development_gate_passed", "behavior_preserved", "quality_gate_passed", "work_gate_passed",
        "references_complete", "execution_checks_passed"), True)}
    old_path, dev_path = tmp_path / "old.json", tmp_path / "development.json"
    for path, protocol, result in ((old_path, old, old_result), (dev_path, dev, dev_result)):
        timing._protocol_path(path).write_text(json.dumps(protocol))
        path.write_text(json.dumps(result))
    for path, literal in (("old.py", "old frozen source\n"), ("new.py", "new frozen source\n"),
                          ("timing.py", "timing frozen source\n")):
        (tmp_path / path).write_text(literal)
    for name, value in (("ROOT", tmp_path), ("PREVIOUS_OUTPUT", old_path),
                        ("DEVELOPMENT_OUTPUT", dev_path), ("SOURCES", ("timing.py",))):
        monkeypatch.setattr(timing, name, value)
    return SimpleNamespace(models=models, old=old, dev=dev, old_result=old_result,
                           dev_result=dev_result, old_path=old_path, dev_path=dev_path, root=tmp_path)


def test_registration_keeps_the_exact_754_roster_orders_budgets_and_gates(frozen_sources):
    source = frozen_sources
    protocol = timing.build_protocol(source.models)
    assert "registered_at_utc" not in protocol
    assert (protocol["expected_state_count"], protocol["expected_timed_calls"],
            protocol["expected_warmup_calls"], protocol["expected_selector_calls"]) == (754, 13572, 96, 13668)
    assert tuple(protocol[key] for key in ("initial_lp_budget", "lp_node_budget", "integer_state_budget",
        "long_lp_budget", "control_integer_state_budget", "nonregression_limit", "trigger_speedup_limit")) == (
            144, 256, 8192, 4096, 100000, 1.05, .95)
    assert protocol["expected_group_counts"] == source.old["expected_group_counts"]
    assert protocol["reference_sources"] == {"affine_development": {
        "path": "development.json", "registered_at_utc": source.dev["registered_at_utc"]}}
    assert protocol["reference_sources"]["affine_development"]["registered_at_utc"] != source.old["registered_at_utc"]
    assert protocol["prior_decisions"]["prune_first_dual_timing"] == "PRUNE_FIRST_DUAL_LOCAL_TIMING_HOLD"
    assert protocol["execution_sources"] == {"old.py": "old frozen source\n",
        "new.py": "new frozen source\n", "timing.py": "timing frozen source\n"}
    for cell, old, dev in zip(protocol["cells"], source.old["cells"], source.dev["cells"], strict=True):
        assert cell["orders"] == old["orders"] == timing.timing_orders(cell["node"], cell["lanes"], len(cell["states"]))
        for index, (state, original) in enumerate(zip(cell["states"], dev["states"], strict=True)):
            assert state == {**original, "timing_source": "affine_development",
                             "timing_source_index": index, "timing_source_state_id": original["state_id"]}
        assert all(set(map(tuple, orders)) == set(permutations(timing.LABELS))
                   for orders in cell["orders"]["label_orders"])


def test_incomplete_or_held_development_and_invalid_previous_hold_block_timing(frozen_sources):
    source = frozen_sources
    for field in ("decision", "development_gate_passed", "behavior_preserved", "quality_gate_passed",
                  "work_gate_passed", "references_complete", "execution_checks_passed"):
        changed = {**source.dev_result, field: "AFFINE_DUAL_DEVELOPMENT_HOLD" if field == "decision" else False}
        source.dev_path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="complete affine dual development"):
            timing.build_protocol(source.models)
    source.dev_path.write_text(json.dumps(source.dev_result))
    for field in ("decision", "execution_checks_passed", "references_complete", "quality_gate_passed", "repeat_paths_stable", "frozen_residual_results_reproduced"):
        changed = {**source.old_result, field: "PRUNE_FIRST_DUAL_LOCAL_TIMING_ADVANCE" if field == "decision" else False}
        source.old_path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="complete prior prune-first timing HOLD"):
            timing.build_protocol(source.models)


def test_changed_sources_dropped_states_and_changed_metadata_stop_registration(frozen_sources):
    source = frozen_sources
    (source.root / "old.py").write_text("changed source\n")
    with pytest.raises(ValueError, match="frozen source changed"):
        timing.build_protocol(source.models)
    (source.root / "old.py").write_text("old frozen source\n")
    for change in ("removed", "metadata", "models", "budget"):
        dev = deepcopy(source.dev)
        if change == "removed":
            dev["cells"][0]["states"].pop()
        elif change == "metadata":
            dev["cells"][0]["states"][0]["queue"][0] += 1
        elif change == "models":
            dev["models"][0]["lanes"] += 1
        else:
            dev["lp_node_budget"] += 1
        timing._protocol_path(source.dev_path).write_text(json.dumps(dev))
        with pytest.raises(ValueError):
            timing.build_protocol(source.models)


def test_registration_locks_budgets_gates_source_pointers_and_original_orders(frozen_sources):
    protocol = timing.build_protocol(frozen_sources.models)
    protocol["registered_at_utc"] = "timing registered"
    timing.verify_protocol(protocol, frozen_sources.models)
    for change in ("budget", "gate", "source", "order"):
        altered = deepcopy(protocol)
        if change == "budget":
            altered["integer_state_budget"] += 1
        elif change == "gate":
            altered["nonregression_limit"] = 1.1
        elif change == "source":
            altered["cells"][0]["states"][0]["timing_source_index"] += 1
        else:
            altered["cells"][0]["orders"]["label_orders"][0][0].reverse()
        with pytest.raises(ValueError, match="registered sources, roster, budgets or timing design changed"):
            timing.verify_protocol(altered, frozen_sources.models)


@pytest.fixture
def replay_harness(old_harness, monkeypatch):
    harness = old_harness
    saved, states = [], []
    for index, state in enumerate(harness.cell["states"]):
        original = harness.saved[state["timing_source"]][state["timing_source_index"]]
        added = {"affine_source_index": state["timing_source_index"],
                 "affine_source_state_id": state["timing_source_state_id"]}
        saved.append({**deepcopy(original), **deepcopy(state), **added})
        states.append({**deepcopy(state), **added, "timing_source": "affine_development",
                       "timing_source_index": index, "timing_source_state_id": original["state_id"]})
    harness.saved = saved
    harness.cell["states"] = states
    harness.protocol["reference_sources"] = {"affine_development": {
        "path": "affine_development.json", "registered_at_utc": "new development registered"}}

    def load(path, node, lanes, utc):
        assert (path.stem, node, lanes, utc) == (
            "affine_development", harness.model.node, 64, "new development registered")
        harness.loaded.append(path.stem)
        return {"node": node, "lanes": lanes, "states": saved}

    monkeypatch.setattr(previous, "load_cell", load)
    return harness


def test_custom_selectors_keep_whole_cold_calls_and_borrow_new_development(replay_harness):
    harness = replay_harness
    row = timing.run_cell(harness.model, harness.cell, harness.protocol, harness.runtime,
                          selectors=harness.selectors, progress=False)
    assert harness.loaded == ["affine_development"]
    assert len(harness.calls) == len(harness.recorded) == 60
    assert Counter(label for label, _, _ in harness.calls) == dict.fromkeys(timing.LABELS, 20)
    expected = [(label, tuple(harness.cell["states"][index]["queue"]),
                 (4096, 100000) if label == "long" else (144, 256, 8192))
                for repeat, order in enumerate(harness.cell["orders"]["state_orders"])
                for index in order for label in harness.cell["orders"]["label_orders"][index][repeat]]
    assert harness.calls[6:] == expected
    for state, saved in zip(row["states"], harness.saved, strict=True):
        for field in (*previous.REFERENCE_FIELDS, "traces", "prior_routed"):
            assert state[field] == saved[field]
        assert state["expected_results"] == {"candidate": saved["routed"], "branch": saved["old_routed"]}
    assert (row["new_reference_calls"], row["new_oracle_calls"], row["screen_selector_calls"],
            row["borrowed_reference_count"]) == (0, 0, 0, 3)


def test_changed_development_pointer_fails_before_warmup(replay_harness):
    harness = replay_harness
    harness.cell["states"][0]["affine_source_index"] += 1
    with pytest.raises(ValueError, match="borrowed timing state differs"):
        timing.run_cell(harness.model, harness.cell, harness.protocol, harness.runtime,
                        selectors=harness.selectors, progress=False)
    assert not harness.calls and not harness.recorded


def test_summary_reuses_paid_work_and_gates_and_reports_current_failure_ratios(monkeypatch):
    from algorithm.experiments import affine_dual_timing_summary as frozen_summary
    protocol = {"previous_timing_failure_keys": [{"node": "n", "lanes": 64, "queue": [88, 8, 23]}],
                "affine_focus_keys": [{"node": "n", "lanes": 64, "queue": [88, 8, 23]}]}
    rows = [{"node": "n", "lanes": 64, "states": [{"state_id": "old failure", "queue": [88, 8, 23], "timing_group": "historical_regression",
        "wall_ns": {"candidate": [90, 90, 90, 90, 90, 90], "branch": [100] * 6, "long": [120] * 6}}]}]
    original = {"decision": "AFFINE_DUAL_LOCAL_TIMING_HOLD", "work": {"paid": 3377},
                "timing_gates": {"branch": {"historical_every_state_nonregression": False}},
                "prior_hold_retained": True, "references_complete": True,
                "executed_affine_dual_work": {"coefficient_requirement_reads": 330, "delta_integer_operations": 21}}
    calls = []

    def summarize(actual_protocol, actual_rows):
        assert actual_protocol is protocol and actual_rows is rows
        calls.append((actual_protocol, actual_rows))
        return deepcopy(original)

    monkeypatch.setattr(frozen_summary, "summarize", summarize)
    result = timing.summarize(protocol, rows)
    assert len(calls) == 1 and result["decision"] == "AFFINE_DUAL_LOCAL_TIMING_HOLD"
    assert result["work"] == original["work"] and result["timing_gates"] == original["timing_gates"]
    point, = result["previous_timing_failure_states"]
    assert point["comparisons"] == {"branch": {"wall_ratio": .9}, "long": {"wall_ratio": .75}}
    assert point["candidate_median_wall_ms"] == .00009
    assert result["prior_hold_retained"]
    assert result["executed_affine_dual_work"] == original["executed_affine_dual_work"]
    assert result["affine_focus_states"][0]["comparisons"] == point["comparisons"]


def test_existing_outputs_stop_main_before_selector_creation(frozen_sources, monkeypatch):
    source = frozen_sources
    output = source.root / "timing_output.json"
    protocol = timing.build_protocol(source.models)
    protocol["registered_at_utc"] = "timing registered"
    timing._protocol_path(output).write_text(json.dumps(protocol))
    output.write_text("previous results preserved\n")
    monkeypatch.setattr(timing, "runtime_setup", lambda: SimpleNamespace(load_catalogues=lambda: source.models))
    monkeypatch.setattr(timing, "_selectors", lambda: pytest.fail("existing outputs must not execute selectors"))
    monkeypatch.setattr("sys.argv", ["timing", "--output", str(output), "--run"])
    with pytest.raises(FileExistsError, match="affine dual timing outputs already exist"):
        timing.main()


@pytest.fixture
def affine_records(previous_records):
    """No children need lazy preparation; each saved LP numerator is still charged."""
    def convert(value):
        if isinstance(value, dict):
            if "dual_reuse" in value:
                for key in ("child_rhs_entries", "child_requirement_reads", "child_weight_reads"):
                    value["dual_reuse"][key] = 0
                value["affine_dual"] = dict.fromkeys(AFFINE_COUNTERS, 0)
                value["affine_dual"]["certificate_numerator_products"] = value["dual_reuse"]["certificates_saved"]
            for nested in value.values():
                convert(nested)
        elif isinstance(value, list):
            for nested in value:
                convert(nested)
    for value in previous_records:
        convert(value)
    return previous_records


def test_timed_and_warmup_affine_preparation_fees_are_charged_separately(affine_records):
    result = affine_summary.summarize(*affine_records)
    assert result["decision"] == "AFFINE_DUAL_LOCAL_TIMING_ADVANCE"
    assert result["certificate_checks_passed"] and result["frozen_residual_results_reproduced"]
    assert result["work"]["candidate"]["affine_dual"]["certificate_numerator_products"] == 36
    assert result["warmup_work"]["candidate"]["affine_dual"]["certificate_numerator_products"] == 2
    assert result["executed_affine_dual_work"]["certificate_numerator_products"] == 38
    assert set(result["executed_affine_dual_work"]) == set(AFFINE_COUNTERS)
    assert result["work"]["candidate"]["dual_reuse"]["child_rhs_entries"] == 0
    assert result["work"]["candidate"]["axis_capacity"]["merged_values_copied"] == 36
    assert result["work"]["candidate"]["residual_capacity_completion"]["prefix_reads"] == 36
    assert "affine_dual" not in result["work"]["branch"]
    assert result["executed_selector_calls"] == 114 and result["prior_hold_retained"]


@pytest.mark.parametrize("phase", ("timed", "warmup"))
def test_missing_affine_preparation_fee_invalidates_actual_result(affine_records, phase):
    protocol, rows = affine_records
    if phase == "timed":
        state = rows[0]["states"][-1]
        result = state["first_results"]["candidate"]
        result["final_lp"]["affine_dual"]["certificate_numerator_products"] = 0
        state["expected_results"]["candidate"] = deepcopy(result)
    else:
        result = rows[0]["warmups"][0]["result"]
        result["final_lp"]["affine_dual"]["certificate_numerator_products"] = 0
    actual = affine_summary.summarize(protocol, rows)
    assert actual["decision"] == "INVALID_TIMING_EXECUTION"
    assert not actual["local_timing_gate_passed"] and actual["accounting_failures"]


@pytest.mark.parametrize("group", ("historical_regression", "new_trigger"))
def test_affine_cost_validity_cannot_hide_slow_fixed_states(affine_records, group):
    protocol, rows = affine_records
    for state in rows[0]["states"]:
        if state["timing_group"] == group:
            state["wall_ns"]["candidate"] = [120] * 6
        else:
            state["wall_ns"]["branch"] = state["wall_ns"]["long"] = [10000] * 6
    actual = affine_summary.summarize(protocol, rows)
    assert actual["decision"] == "AFFINE_DUAL_LOCAL_TIMING_HOLD"
    assert actual["execution_checks_passed"] and actual["quality_gate_passed"]


def test_changed_candidate_repeat_retains_and_charges_its_actual_affine_fee(affine_records):
    protocol, rows = affine_records
    state = rows[0]["states"][0]
    changed = deepcopy(state["first_results"]["candidate"])
    changed["final_lp"]["affine_dual"]["box_coordinate_reads"] = 4
    state["repeat_mismatches"]["candidate"] = [{"repeat_index": 3, "result": changed}]
    result = affine_summary.summarize(protocol, rows)
    assert result["decision"] == "INVALID_TIMING_EXECUTION"
    assert result["work"]["candidate"]["affine_dual"]["box_coordinate_reads"] == 4
    assert not result["repeat_paths_stable"] and result["accounting_failures"]
