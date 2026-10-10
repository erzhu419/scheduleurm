"""Independent endpoints stay separate from all prior four-arm regression cohorts."""
from collections import Counter
from copy import deepcopy
from itertools import permutations
import json
from types import SimpleNamespace

import pytest

from algorithm.experiments import aligned_hardware_zero_delta_affine_fresh_timing as timing
from algorithm.experiments import zero_delta_affine_fresh_timing_summary as summary
from skill.tests.test_zero_delta_affine_timing import harness as zero_harness, records as zero_records
from skill.tests.test_affine_dual_fresh_timing import harness as fresh_harness, records as fresh_records
from skill.tests.test_affine_dual_timing import affine_records
from skill.tests.test_residual_capacity_timing import harness as old_harness
from skill.tests.test_residual_capacity_timing_summary import records as previous_records, _second_cell


@pytest.fixture
def frozen_sources(monkeypatch, tmp_path):
    old = json.loads(timing._protocol_path(timing.PREVIOUS_OUTPUT).read_text())
    old["execution_sources"] = {"old.py": "frozen zero selector\n"}
    models = [SimpleNamespace(snapshot=lambda data=deepcopy(model): deepcopy(data)) for model in old["models"]]
    quality = deepcopy(old)
    quality.update(registered_at_utc="independent quality registered", expected_state_count=2802,
                   expected_independent_state_count=1024, expected_borrowed_state_count=1778)
    quality["subsequent_timing_plan"] = {"order_seed": 2026092897, "labels": list(timing.LABELS),
        "repeats": 6, "warmups_per_label": 2, "expected_timed_calls": 67248, "expected_warmup_calls": 128,
        "nonregression_limit": 1.05, "trigger_speedup_limit": .95}
    quality["execution_sources"]["quality.py"] = "frozen quality runner\n"
    for cell in quality["cells"]:
        cell.pop("orders")
        controls = [{**deepcopy(state), "independent_source": "borrowed_control",
            "independent_source_index": index, "independent_source_state_id": state["state_id"]}
            for index, state in enumerate(cell["states"])]
        # Artificial metadata fixture; never invoke the new sampler or a selector.
        fresh = [{"state_id": f"independent:{index}", "queue": [1000 + index, 2000 + index, 3000 + index],
            "source": "fresh_confirmation", "cohort": "independent_fresh", "panel": f"panel:{index // 16}",
            "seed": 2026092893 + index // 16, "confirmation_source": "fresh", "independent_source": "fresh",
            "historical_regression": False, "timing_group": "independent_fresh", "caps": controls[0]["caps"]}
            for index in range(64)]
        cell["states"] = fresh + controls
    prior = {"decision": "ZERO_DELTA_AFFINE_LOCAL_TIMING_ADVANCE", **dict.fromkeys((
        "execution_checks_passed", "references_complete", "quality_gate_passed", "repeat_paths_stable",
        "frozen_residual_results_reproduced", "local_timing_gate_passed"), True)}
    result = {"decision": "ZERO_DELTA_AFFINE_FRESH_CONFIRMATION_ADVANCE", **dict.fromkeys((
        "execution_checks_passed", "references_complete", "quality_gate_passed", "confirmation_gate_passed",
        "behavior_preserved", "work_gate_passed"), True)}
    old_path, quality_path = tmp_path / "old.json", tmp_path / "quality.json"
    for path, protocol, output in ((old_path, old, prior), (quality_path, quality, result)):
        timing._protocol_path(path).write_text(json.dumps(protocol))
        path.write_text(json.dumps(output))
    for path, literal in (("old.py", "frozen zero selector\n"), ("quality.py", "frozen quality runner\n"),
                           ("timing.py", "new timing runner\n")):
        (tmp_path / path).write_text(literal)
    for name, value in (("ROOT", tmp_path), ("PREVIOUS_OUTPUT", old_path),
            ("CONFIRMATION_OUTPUT", quality_path), ("SOURCES", ("timing.py",))):
        monkeypatch.setattr(timing, name, value)
    return SimpleNamespace(models=models, old=old, quality=quality, prior=prior, result=result,
                           old_path=old_path, quality_path=quality_path, root=tmp_path)


def test_all_new_states_and_prior_lineage_are_frozen_with_separate_endpoints(frozen_sources):
    source = frozen_sources
    protocol = timing.build_protocol(source.models)
    assert (protocol["expected_state_count"], protocol["expected_independent_state_count"],
            protocol["expected_borrowed_state_count"], protocol["expected_timed_calls"],
            protocol["expected_warmup_calls"], protocol["expected_selector_calls"]) == (2802, 1024, 1778, 67248, 128, 67376)
    assert protocol["expected_group_counts"] == {**source.old["expected_group_counts"], "independent_fresh": 1024}
    assert "expected_fresh_state_count" not in protocol and "expected_control_state_count" not in protocol
    assert protocol["reference_sources"] == {"zero_delta_fresh_quality": {
        "path": "quality.json", "registered_at_utc": "independent quality registered"}}
    assert tuple(protocol[key] for key in ("initial_lp_budget", "lp_node_budget", "integer_state_budget",
        "long_lp_budget", "control_integer_state_budget", "nonregression_limit", "trigger_speedup_limit")) == (
            144, 256, 8192, 4096, 100000, 1.05, .95)
    assert protocol["order_seed"] == 2026092897
    for cell, source_cell in zip(protocol["cells"], source.quality["cells"], strict=True):
        assert cell["orders"] == timing.timing_orders(cell["node"], cell["lanes"], len(cell["states"]))
        for index, (state, saved) in enumerate(zip(cell["states"], source_cell["states"], strict=True)):
            assert state == {**saved, "timing_source": "zero_delta_fresh_quality",
                "timing_source_index": index, "timing_source_state_id": saved["state_id"]}
        for blocks in cell["orders"]["label_orders"]:
            assert Counter(tuple(label for label in block if label != "affine") for block in blocks) == Counter(
                permutations(timing.ORIGINAL_LABELS))
            assert sorted(Counter(block.index("affine") for block in blocks).values()) == [1, 1, 2, 2]


@pytest.mark.parametrize("field", ("decision", "confirmation_gate_passed", "execution_checks_passed",
                                  "references_complete", "quality_gate_passed", "behavior_preserved", "work_gate_passed"))
def test_incomplete_quality_prevents_any_timing(frozen_sources, field):
    source = frozen_sources
    changed = {**source.result, field: "ZERO_DELTA_AFFINE_FRESH_CONFIRMATION_HOLD" if field == "decision" else False}
    source.quality_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="complete independent zero-delta confirmation"):
        timing.build_protocol(source.models)


def test_timing_keeps_the_plan_registered_before_quality_outcomes(frozen_sources):
    source = frozen_sources
    changed = deepcopy(source.quality)
    changed["subsequent_timing_plan"]["order_seed"] += 1
    timing._protocol_path(source.quality_path).write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="timing plan frozen before independent quality outcomes"):
        timing.build_protocol(source.models)


@pytest.mark.parametrize("change", ("independent", "prior", "lineage", "source", "order", "gate"))
def test_rosters_sources_original_lineage_and_gates_cannot_change(frozen_sources, change):
    source = frozen_sources
    protocol = timing.build_protocol(source.models)
    protocol["registered_at_utc"] = "timing registered"
    changed = deepcopy(source.quality)
    if change == "independent":
        changed["cells"][0]["states"].pop(0)
    elif change == "prior":
        changed["cells"][0]["states"].pop()
    elif change == "lineage":
        changed["cells"][0]["states"][-1]["confirmation_source"] = "fresh"
    elif change == "source":
        (source.root / "old.py").write_text("changed selector\n")
    elif change == "order":
        protocol["cells"][0]["orders"]["label_orders"][0][0].reverse()
    else:
        protocol["nonregression_limit"] = 1.1
    timing._protocol_path(source.quality_path).write_text(json.dumps(changed))
    with pytest.raises(ValueError):
        timing.verify_protocol(protocol, source.models)


@pytest.fixture
def harness(zero_harness, monkeypatch):
    harness = zero_harness
    saved, states = [], []
    for index, (state, record) in enumerate(zip(harness.cell["states"], harness.saved, strict=True)):
        added = {"independent_source": "fresh" if index == 0 else "borrowed_control",
                 "independent_source_index": index, "independent_source_state_id": state["state_id"]}
        saved.append({**deepcopy(record), **deepcopy(state), **added})
        states.append({**deepcopy(state), **added, "timing_group": "independent_fresh" if index == 0 else state["timing_group"],
            "timing_source": "zero_delta_fresh_quality", "timing_source_index": index,
            "timing_source_state_id": state["state_id"]})
    harness.saved, harness.cell["states"] = saved, states
    harness.cell["orders"] = timing.timing_orders(harness.model.node, 64, len(states))
    harness.protocol["reference_sources"] = {"zero_delta_fresh_quality": {
        "path": "quality.json", "registered_at_utc": "independent quality registered"}}

    def load(path, node, lanes, utc):
        assert (path.stem, node, lanes, utc) == ("quality", harness.model.node, 64, "independent quality registered")
        harness.loaded.append(path.stem)
        return {"node": node, "lanes": lanes, "states": saved}

    monkeypatch.setattr(timing, "load_cell", load)
    monkeypatch.setattr(timing, "runtime_provenance", lambda: {"cpu_affinity": [0]})
    return harness


def test_new_clock_calls_reuse_only_true_outputs_and_true_axis_priors(harness):
    row = timing.run_cell(harness.model, harness.cell, harness.protocol, harness.runtime,
                          selectors=harness.selectors, progress=False)
    assert harness.loaded == ["quality"] and len(harness.calls) == len(harness.recorded) == 80
    assert Counter(label for label, _, _ in harness.calls) == dict.fromkeys(timing.LABELS, 20)
    assert "prior_routed" not in row["states"][0]
    assert row["states"][2]["independent_source"] == "borrowed_control"
    assert row["states"][2]["confirmation_source"] == "borrowed_control"
    assert row["states"][2]["prior_routed"] == harness.saved[2]["prior_routed"]
    for state, saved in zip(row["states"], harness.saved, strict=True):
        assert state["previous_dual_routed"] == state["expected_results"]["affine"] == saved["previous_dual_routed"]
    assert row["new_reference_calls"] == row["new_oracle_calls"] == row["screen_selector_calls"] == 0


@pytest.fixture
def records(zero_records):
    protocol, rows = zero_records
    row, cell = rows[0], protocol["cells"][0]
    for state, metadata in zip(row["states"], cell["states"], strict=True):
        state["independent_source"] = metadata["independent_source"] = "borrowed_control"
    for index in (0, 7):
        state, metadata = deepcopy(row["states"][index]), deepcopy(cell["states"][index])
        added = {"state_id": f"independent:{index}", "independent_source": "fresh", "confirmation_source": "fresh",
                 "timing_group": "independent_fresh", "historical_regression": False,
                 "cohort": "independent_fresh", "source": "fresh_confirmation"}
        state.update(added)
        metadata.update(added)
        state.pop("prior_routed", None)
        row["states"].append(state)
        cell["states"].append(metadata)
    orders = timing.timing_orders(row["node"], row["lanes"], len(row["states"]))
    cell["orders"] = orders
    row["state_orders"] = deepcopy(orders["state_orders"])
    for index, state in enumerate(row["states"]):
        state["blocks"] = [{"repeat": repeat, "order": block} for repeat, block in enumerate(orders["label_orders"][index])]
    row["borrowed_reference_count"] = 10
    protocol.pop("expected_fresh_state_count")
    protocol.pop("expected_control_state_count")
    protocol.update(expected_state_count=10, expected_independent_state_count=2, expected_borrowed_state_count=8,
                    expected_timed_calls=240, expected_selector_calls=248)
    protocol["expected_group_counts"]["independent_fresh"] = 2
    return protocol, rows


def test_independent_and_all_prior_cohorts_pass_with_all_costs(records):
    result = summary.summarize(*records)
    assert result["decision"] == "ZERO_DELTA_AFFINE_FRESH_TIMING_ADVANCE"
    assert result["independent_timing_gate_passed"] and result["borrowed_timing_gate_passed"]
    assert (result["state_count"], result["independent_state_count"], result["borrowed_state_count"],
            result["executed_selector_calls"]) == (10, 2, 8, 248)
    assert result["groups"]["fresh_confirmation"]["queue_count"] == result["groups"]["independent_fresh"]["queue_count"] == 2
    assert result["borrowed_axis_router_count"] == 12
    assert result["borrowed_native_snapshots"] == 120 and result["borrowed_residual_router_count"] == 30
    assert result["executed_affine_dual_work"]["certificate_numerator_products"] == 62
    assert result["executed_affine_control_work"]["certificate_numerator_products"] == 62
    assert result["independent_confirmation"] and result["prior_hold_retained"]


@pytest.mark.parametrize("label", ("affine", "branch", "long"))
def test_prior_pooled_gain_cannot_hide_independent_regression(records, label):
    protocol, rows = records
    for state in rows[0]["states"]:
        if state["independent_source"] == "fresh":
            state["wall_ns"][label] = [60] * 6
        else:
            for control in timing.LABELS[1:]:
                state["wall_ns"][control] = [10000] * 6
    result = summary.summarize(protocol, rows)
    assert result["comparisons"][label]["wall_ratio"] < 1.05
    assert result["decision"] == "ZERO_DELTA_AFFINE_FRESH_TIMING_HOLD"
    assert not result["independent_timing_gate_passed"] and result["borrowed_timing_gate_passed"]


@pytest.mark.parametrize("group", ("fresh_confirmation", "historical_regression", "new_trigger"))
def test_independent_gain_cannot_hide_prior_regression(records, group):
    protocol, rows = records
    for state in rows[0]["states"]:
        if state["timing_group"] == group:
            state["wall_ns"]["branch"] = [60] * 6
    result = summary.summarize(protocol, rows)
    assert result["decision"] == "ZERO_DELTA_AFFINE_FRESH_TIMING_HOLD"
    assert result["independent_timing_gate_passed"] and not result["borrowed_timing_gate_passed"]


def test_independent_p99_blocks_despite_good_pooled_ratios(records):
    protocol, rows = records
    rows[0]["states"][-2]["wall_ns"]["candidate"] = [120] * 6
    rows[0]["states"][-1]["wall_ns"]["candidate"] = [10] * 6
    result = summary.summarize(protocol, rows)
    assert result["decision"] == "ZERO_DELTA_AFFINE_FRESH_TIMING_HOLD"
    for label in timing.LABELS[1:]:
        assert result["timing_gates"][label]["independent_pooled_nonregression"]
        assert result["timing_gates"][label]["independent_cell_nonregression"]
        assert not result["timing_gates"][label]["independent_p99_nonregression"]


def test_independent_cell_gate_cannot_hide_in_another_cell(records):
    protocol, rows = records
    other = _second_cell(protocol, rows)
    protocol["expected_independent_state_count"] *= 2
    protocol["expected_borrowed_state_count"] *= 2
    for state in rows[0]["states"][-2:]:
        state["wall_ns"]["candidate"] = [120] * 6
    for state in other["states"][-2:]:
        for label in timing.LABELS[1:]:
            state["wall_ns"][label] = [10000] * 6
    result = summary.summarize(protocol, rows)
    assert result["decision"] == "ZERO_DELTA_AFFINE_FRESH_TIMING_HOLD"
    for label in timing.LABELS[1:]:
        assert result["timing_gates"][label]["independent_pooled_nonregression"]
        assert result["timing_gates"][label]["independent_p99_nonregression"]
        assert not result["timing_gates"][label]["independent_cell_nonregression"]


@pytest.mark.parametrize("mutation", ("fake_prior", "count", "output", "lost_repeat"))
def test_independent_integrity_failures_keep_their_original_priority(records, mutation):
    protocol, rows = records
    state = rows[0]["states"][-1]
    if mutation == "fake_prior":
        state["prior_routed"] = deepcopy(rows[0]["states"][5]["prior_routed"])
    elif mutation == "count":
        protocol["expected_independent_state_count"] += 1
    elif mutation == "output":
        state["expected_results"]["affine"]["status"] = "CHANGED"
    else:
        state["wall_ns"]["affine"].pop()
        with pytest.raises(ValueError):
            summary.summarize(protocol, rows)
        return
    result = summary.summarize(protocol, rows)
    assert result["decision"] == "INVALID_TIMING_EXECUTION" and not result["local_timing_gate_passed"]
