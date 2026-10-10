"""Independent quality, timing-cohort and paid-work gates on tiny real calls."""
from copy import deepcopy
from dataclasses import asdict, replace

import pytest

from algorithm.experiments import aligned_hardware_integer_capacity_development as runtime
from algorithm.experiments import aligned_hardware_resumable_probe_confirmation as runner
from algorithm.experiments import resumable_probe_confirmation_summary as summary
from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption
from algorithm.experiments.aligned_hardware_queue_aware_canonical_development import selection_from_record
from algorithm.experiments.integer_capacity_canonical_oracle import canonical_at_certified_score
from algorithm.experiments.score_face_completion_confirmation_summary import _reference, _score
from algorithm.experiments.strict_score_proof_timing_summary import _repeat_results
from algorithm.resumable_probe_frame_candidates import select_resumable_probe_frame
from algorithm.strict_score_proof_canonical_completion_frame_candidates import select_strict_score_proof_canonical_completion_frame


LABELS = ("candidate", "legacy", "long")


def refresh(protocol, rows, models):
    by_node = {model.node: model for model in models}
    for row in rows:
        model = by_node[row["node"]]
        for state in row["states"]:
            expected, _ = _reference(state, protocol, len(model.options))
            state["screen_metrics"] = {
                label: summary.audit_result(result, state, protocol, model.snapshot(), row["lanes"], label, expected)[0]
                for label, result in state["screen_results"].items()
            }
            if "first_results" in state:
                state["metrics"] = {
                    label: runtime._combine_metrics(
                        summary.audit_result(result, state, protocol, model.snapshot(), row["lanes"], label, expected)[0]
                        for result in _repeat_results(state, label, protocol["repeats"]))
                    for label in LABELS
                }
        for warmup in row["warmups"]:
            state = row["states"][warmup["state_index"]]
            expected, _ = _reference(state, protocol, len(model.options))
            warmup["metrics"] = summary.audit_result(
                warmup["result"], state, protocol, model.snapshot(), row["lanes"], warmup["label"], expected)[0]


@pytest.fixture
def case(monkeypatch):
    monkeypatch.setattr(runner, "runtime_provenance", lambda: {"cpu_affinity": [0]})
    pairs = (((1, 0), 5.), ((0, 1), 2.), ((2, 1), 5.),
             ((1, 2), 3.), ((3, 0), 2.), ((0, 3), 3.))
    first = FrameModel(node="tiny_a", scenario_id="tiny", lanes=4,
        workloads=("a", "b"), scales=(1., 1.),
        options=tuple(FrameOption(str(i), req, duration, ((), ())) for i, (req, duration) in enumerate(pairs)))
    models = (first, replace(first, node="tiny_b"))
    protocol = {
        "experiment": "synthetic_confirmation", "registered_at_utc": "test", "models": [model.snapshot() for model in models],
        "lanes": [4], "labels": list(LABELS), "repeats": 6, "warmups_per_label": 2,
        "initial_lp_budget": 3, "lp_node_budget": 64, "integer_state_budget": 0,
        "control_integer_state_budget": 100000, "long_lp_budget": 64,
        "reference_budget": 4096, "oracle_state_budget": 5000000, "cpu": 0,
        "timing_fresh_per_panel": 2, "fresh_nonregression_limit": 1.05,
        "trigger_legacy_ratio_limit": .90, "trigger_legacy_state_limit": 1.05,
        "trigger_long_ratio_limit": 1., "trigger_long_p99_limit": 1.05,
        "tail_state_limit": 1.05, "fresh_speedup_limit": .95,
        "historical_trigger_keys": [{"node": model.node, "lanes": 4, "queue": [1, 6]} for model in models],
        "queue_panels": {"4": [
            {"panel": "fresh_17", "cohort": "fresh_confirmation", "seed": 17, "states": [
                {"state_id": "fresh:17:0", "queue": [0, 0]}, {"state_id": "fresh:17:1", "queue": [2, 2]}]},
            {"panel": "known", "cohort": "prior_fresh_trigger_regression", "seed": None, "states": [
                {"state_id": "known:0", "queue": [1, 6]}]},
            {"panel": "tail", "cohort": "ordinary_tail_regression", "seed": None, "states": [
                {"state_id": "tail:0", "queue": [1, 1]}]},
        ]},
        "timing_indices": {f"{model.node}:4": [0, 1, 2, 3] for model in models},
        "timing_orders": {f"{model.node}:4": runner.timing_orders(model.node, 4, 4) for model in models},
        "limitations": "Tiny synthetic models only.",
    }
    rows = [runner.screen_and_time_cell(model, 4, protocol, runtime, progress=False) for model in models]
    for row in rows:
        for state in row["states"]:
            value = 80 if state["previous_integer_trigger"] else 100
            state["wall_ns"] = {"candidate": [value] * 6, "legacy": [100] * 6, "long": [100] * 6}
            state["cpu_ns"] = deepcopy(state["wall_ns"])
    return protocol, rows, models


def test_all_actual_calls_charged_once_and_fresh_speed_claim_is_separate(case):
    protocol, rows, _ = case
    result = summary.summarize(protocol, rows)
    assert result["decision"] == "RESUMABLE_PROBE_CONFIRMATION_ADVANCE"
    assert result["confirmation_gate_passed"] and not result["fresh_speedup_established"]
    assert result["candidate_quality_gate_passed"] and result["comparison_quality_gate_passed"]
    assert result["screen_selector_calls"] == 16
    assert result["timed_selector_calls"] == 8 * 6 * 3
    assert result["warmup_calls"] == 2 * 2 * 3
    assert result["executed_selector_calls"] == 172
    assert result["executed_reference_calls"] == 8 and result["executed_oracle_calls"] == 0
    lp_calls = 0
    for row in rows:
        for state in row["states"]:
            lp_calls += sum(record["selection"]["lp_calls"] for record in state["screen_results"].values())
            lp_calls += sum(record["selection"]["lp_calls"] * 6 for record in state["first_results"].values())
        lp_calls += sum(item["result"]["selection"]["lp_calls"] for item in row["warmups"])
    assert result["executed_method_search_work"]["lp_calls"] == lp_calls


def test_fresh_five_percent_speedup_is_reported_only_when_fresh_states_supply_it(case):
    protocol, rows, _ = case
    for row in rows:
        for state in row["states"]:
            if state["cohort"] == "fresh_confirmation":
                state["wall_ns"]["candidate"] = [90] * 6
    result = summary.summarize(protocol, rows)
    assert result["confirmation_gate_passed"] and result["fresh_speedup_established"]


@pytest.mark.parametrize("failure", ("fresh_pooled", "fresh_p99", "fresh_cell", "legacy_gain", "long_regression", "long_p99", "tail_state"))
def test_known_hard_gain_cannot_mask_each_registered_other_timing_gate(case, failure):
    protocol, rows, _ = case
    if failure == "fresh_pooled":
        for row in rows:
            for state in row["states"][:2]:
                state["wall_ns"]["candidate"] = [120] * 6
    elif failure == "fresh_p99":
        for row in rows:
            row["states"][0]["wall_ns"]["candidate"] = [106] * 6
            row["states"][1]["wall_ns"]["candidate"] = [1] * 6
    elif failure == "fresh_cell":
        for state in rows[0]["states"][:2]:
            state["wall_ns"]["candidate"] = [110] * 6
        for state in rows[1]["states"][:2]:
            state["wall_ns"] = {"candidate": [180] * 6, "legacy": [200] * 6, "long": [200] * 6}
    elif failure == "legacy_gain":
        for row in rows:
            row["states"][2]["wall_ns"]["candidate"] = [95] * 6
    elif failure == "long_regression":
        for row in rows:
            row["states"][2]["wall_ns"]["long"] = [70] * 6
    elif failure == "long_p99":
        rows[0]["states"][2]["wall_ns"]["candidate"] = [106] * 6
        rows[0]["states"][2]["wall_ns"]["legacy"] = [200] * 6
        rows[1]["states"][2]["wall_ns"]["candidate"] = [1] * 6
    else:
        rows[0]["states"][3]["wall_ns"]["candidate"] = [106] * 6
        rows[1]["states"][3]["wall_ns"]["candidate"] = [1] * 6
    result = summary.summarize(protocol, rows)
    assert result["decision"] == "RESUMABLE_PROBE_CONFIRMATION_TIMING_HOLD"
    assert result["candidate_quality_gate_passed"] and result["comparison_quality_gate_passed"]
    assert result["certificate_checks_passed"] and result["execution_checks_passed"]
    assert not result["confirmation_gate_passed"]
    if failure == "fresh_p99":
        assert result["timing_gates"]["fresh"]["legacy"]["pooled_no_material_regression"]
        assert not result["timing_gates"]["fresh"]["legacy"]["p99_no_material_regression"]
    elif failure == "fresh_cell":
        assert result["timing_gates"]["fresh"]["legacy"]["pooled_no_material_regression"]
        assert result["timing_gates"]["fresh"]["legacy"]["p99_no_material_regression"]
        assert not result["timing_gates"]["fresh"]["legacy"]["no_cell_material_regression"]
    elif failure == "long_p99":
        assert result["timing_gates"]["regression"]["trigger_long_pooled_no_regression"]
        assert result["timing_gates"]["regression"]["trigger_legacy_no_state_regression"]
        assert not result["timing_gates"]["regression"]["trigger_long_p99_no_material_regression"]


def test_honest_candidate_nonclosure_blocks_confirmation_without_deleting_timing(case):
    protocol, rows, models = case
    protocol["lp_node_budget"] = 3
    for row, model in zip(rows, models):
        for state in row["states"]:
            selected = asdict(select_resumable_probe_frame(model.options, state["queue"], 4, 3, 3, 0))
            state["screen_results"]["candidate"] = deepcopy(selected)
            state["first_results"]["candidate"] = selected
        for item in row["warmups"]:
            if item["label"] == "candidate":
                item["result"] = deepcopy(row["states"][item["state_index"]]["screen_results"]["candidate"])
    refresh(protocol, rows, models)
    result = summary.summarize(protocol, rows)
    assert result["decision"] == "RESUMABLE_PROBE_CONFIRMATION_QUALITY_HOLD"
    assert result["certificate_checks_passed"] and not result["candidate_quality_gate_passed"]
    assert result["timed_selector_calls"] == 144 and result["executed_selector_calls"] == 172


def test_honest_legacy_nonclosure_blocks_matched_quality_timing_claim(case):
    protocol, rows, models = case
    protocol["control_integer_state_budget"] = 0
    for row, model in zip(rows, models):
        for state in row["states"]:
            for label, cap in (("legacy", 3), ("long", 64)):
                selected = asdict(select_strict_score_proof_canonical_completion_frame(model.options, state["queue"], 4, cap, 0))
                state["first_results"][label] = selected
                if label == "legacy":
                    state["screen_results"][label] = deepcopy(selected)
        for item in row["warmups"]:
            if item["label"] in ("legacy", "long"):
                item["result"] = deepcopy(row["states"][item["state_index"]]["first_results"][item["label"]])
    refresh(protocol, rows, models)
    result = summary.summarize(protocol, rows)
    assert result["decision"] == "RESUMABLE_PROBE_CONFIRMATION_QUALITY_HOLD"
    assert result["certificate_checks_passed"] and result["candidate_quality_gate_passed"]
    assert not result["comparison_quality_gate_passed"] and not result["confirmation_gate_passed"]


@pytest.mark.parametrize("phase,label", (("screen", "candidate"), ("timing", "long"), ("warmup", "legacy")))
def test_any_actual_arm_accounting_failure_blocks_advance(case, phase, label):
    protocol, rows, models = case
    state = rows[0]["states"][2]
    if phase == "screen":
        state["screen_results"][label]["final_lp"]["selection"]["lp_calls"] = 1
    elif phase == "timing":
        state["first_results"][label]["selection"]["lp_calls"] += 1
    else:
        item = next(item for item in rows[0]["warmups"] if item["label"] == label)
        item["result"]["selection"]["lp_calls"] += 1
    refresh(protocol, rows, models)
    result = summary.summarize(protocol, rows)
    assert result["decision"] == "INVALID_EXECUTION"
    assert not result["certificate_checks_passed"] and not result["confirmation_gate_passed"]


def test_different_timed_repeat_is_charged_and_invalidates_timing(case):
    protocol, rows, models = case
    state = rows[0]["states"][2]
    changed = deepcopy(state["first_results"]["candidate"])
    for selected in (changed["selection"], changed["final_lp"]["selection"]):
        selected["candidate_evaluations"] += 1
    state["repeat_mismatches"]["candidate"] = [{"repeat_index": 2, "result": changed}]
    refresh(protocol, rows, models)
    result = summary.summarize(protocol, rows)
    assert result["decision"] == "INVALID_TIMING_EXECUTION"
    assert result["candidate_quality_gate_passed"] and not result["execution_checks_passed"]
    initial_work = sum(
        record["selection"]["candidate_evaluations"]
        for row in rows for state in row["states"] for record in state["screen_results"].values())
    repeated_work = sum(
        record["selection"]["candidate_evaluations"] * 6
        for row in rows for state in row["states"] for record in state["first_results"].values())
    warmup_work = sum(item["result"]["selection"]["candidate_evaluations"] for row in rows for item in row["warmups"])
    assert result["executed_method_search_work"]["candidate_evaluations"] == initial_work + repeated_work + warmup_work + 1


@pytest.mark.parametrize("oracle_finishes", (False, True))
def test_reference_incompleteness_is_retained_and_only_independent_oracle_can_complete_it(case, oracle_finishes):
    protocol, rows, models = case
    state, model = rows[0]["states"][2], models[0]
    reference = state["reference"]
    reference.update(exact=False, canonical_exact=False, score_exact=oracle_finishes)
    if not oracle_finishes:
        reference.update(upper_bound=reference["score"] + 1., gap=1., epsilon=1.)
    state["reference_complete"] = False
    state["reference_metrics"] = runtime._metrics()
    runtime._record(state["reference_metrics"], selection_from_record(reference), model,
                    state["queue"], 4, protocol["reference_budget"], None)
    state["oracle"] = (canonical_at_certified_score(
        model.options, state["queue"], 4, _score(reference, state["queue"]),
        state_visit_budget=protocol["oracle_state_budget"], maximum_duration=reference["duration"])
        if oracle_finishes else None)
    state["effective_reference_complete"] = oracle_finishes
    state["effective_reference_source"] = "independent_score_face_oracle" if oracle_finishes else None
    refresh(protocol, rows, models)
    result = summary.summarize(protocol, rows)
    assert not state["reference_complete"] and not state["reference"]["canonical_exact"]
    assert result["original_incomplete_reference_count"] == 1
    assert result["decision"] == ("RESUMABLE_PROBE_CONFIRMATION_ADVANCE" if oracle_finishes else "INCOMPLETE_REFERENCE")
    assert result["executed_oracle_calls"] == int(oracle_finishes)


@pytest.mark.parametrize("missing", ("long_repeat", "long_warmup", "screen_legacy", "registered_state"))
def test_missing_registered_comparison_or_measurement_is_rejected(case, missing):
    protocol, rows, _ = case
    if missing == "long_repeat":
        rows[0]["states"][0]["wall_ns"]["long"].pop()
    elif missing == "long_warmup":
        rows[0]["warmups"].remove(next(item for item in rows[0]["warmups"] if item["label"] == "long"))
    elif missing == "screen_legacy":
        del rows[0]["states"][0]["screen_results"]["legacy"]
    else:
        rows[0]["states"].pop()
    with pytest.raises((ValueError, KeyError)):
        summary.summarize(protocol, rows)
