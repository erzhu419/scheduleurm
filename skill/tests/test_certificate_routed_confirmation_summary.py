"""Independent reference, frozen subgroups and paid-work confirmation gates."""
from copy import deepcopy
from dataclasses import asdict, replace

import pytest

from algorithm.certificate_routed_frame_candidates import select_certificate_routed_frame
from algorithm.experiments import aligned_hardware_certificate_routed_confirmation as runner
from algorithm.experiments import aligned_hardware_integer_capacity_development as runtime
from algorithm.experiments import certificate_routed_confirmation_summary as summary
from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption
from algorithm.experiments.aligned_hardware_queue_aware_canonical_development import selection_from_record
from algorithm.experiments.integer_capacity_canonical_oracle import canonical_at_certified_score
from algorithm.experiments.score_face_completion_confirmation_summary import _reference, _score
from algorithm.experiments.strict_score_proof_timing_summary import _repeat_results
from algorithm.resumable_probe_frame_candidates import select_resumable_probe_frame
from algorithm.strict_score_proof_canonical_completion_frame_candidates import select_strict_score_proof_canonical_completion_frame


LABELS = ("candidate", "probe", "long")


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
        for item in row["warmups"]:
            state = row["states"][item["state_index"]]
            expected, _ = _reference(state, protocol, len(model.options))
            item["metrics"] = summary.audit_result(
                item["result"], state, protocol, model.snapshot(), row["lanes"], item["label"], expected)[0]


@pytest.fixture
def case(monkeypatch):
    monkeypatch.setattr(runner, "runtime_provenance", lambda: {"cpu_affinity": [0]})
    pairs = (((1, 0), 5.), ((0, 1), 2.), ((2, 1), 5.),
             ((1, 2), 3.), ((3, 0), 2.), ((0, 3), 3.))
    first = FrameModel(node="tiny_a", scenario_id="tiny", lanes=4,
        workloads=("a", "b"), scales=(1., 1.),
        options=tuple(FrameOption(str(i), req, duration, ((), ())) for i, (req, duration) in enumerate(pairs)))
    models = (first, replace(first, node="tiny_b"))
    def keys(queues):
        return [{"node": model.node, "lanes": 4, "queue": list(queue)} for model in models for queue in queues]
    protocol = {
        "experiment": "synthetic_routed_confirmation", "registered_at_utc": "test",
        "models": [model.snapshot() for model in models], "lanes": [4], "labels": list(LABELS),
        "repeats": 6, "warmups_per_label": 2, "initial_lp_budget": 3, "lp_node_budget": 64,
        "integer_state_budget": 8192, "control_integer_state_budget": 100000, "long_lp_budget": 64,
        "reference_budget": 4096, "oracle_state_budget": 5000000, "cpu": 0,
        "timing_fresh_per_panel": 2, "fresh_nonregression_limit": 1.05, "fresh_speedup_limit": .95,
        "score_gain_limit": .95, "state_ratio_limit": 1.05, "nonregression_limit": 1.05,
        "trigger_long_ratio_limit": 1., "tail_state_limit": 1.05,
        "historical_trigger_keys": keys(((4, 6), (1, 6), (6, 4))),
        "score_unproved_keys": keys(((4, 6),)), "previous_untimed_probe_keys": keys(((6, 4),)),
        "queue_panels": {"4": [
            {"panel": "fresh_17", "cohort": "fresh_confirmation", "seed": 17, "states": [
                {"state_id": "fresh:17:0", "queue": [0, 0]}, {"state_id": "fresh:17:1", "queue": [2, 2]}]},
            {"panel": "known", "cohort": "prior_trigger_regression", "seed": None, "states": [
                {"state_id": "score", "queue": [4, 6]}, {"state_id": "other", "queue": [1, 6]},
                {"state_id": "untimed", "queue": [6, 4]}]},
            {"panel": "tail", "cohort": "ordinary_tail_regression", "seed": None, "states": [
                {"state_id": "tail", "queue": [1, 1]}]},
        ]},
        "timing_indices": {f"{model.node}:4": list(range(6)) for model in models},
        "timing_orders": {f"{model.node}:4": runner.timing_orders(model.node, 4, 6) for model in models},
        "limitations": "Tiny synthetic models only.",
    }
    rows = [runner.screen_and_time_cell(model, 4, protocol, runtime, progress=False) for model in models]
    for row in rows:
        for state in row["states"]:
            value = 80 if state["score_unproved_regression"] else 100
            state["wall_ns"] = {"candidate": [value] * 6, "probe": [100] * 6, "long": [100] * 6}
            state["cpu_ns"] = deepcopy(state["wall_ns"])
    return protocol, rows, models


def test_actual_calls_charged_once_and_fixed_subgroups_do_not_claim_fresh_speedup(case):
    protocol, rows, _ = case
    result = summary.summarize(protocol, rows)
    assert result["decision"] == "CERTIFICATE_ROUTED_CONFIRMATION_ADVANCE"
    assert result["confirmation_gate_passed"] and not result["fresh_speedup_established"]
    assert result["candidate_quality_gate_passed"] and result["comparison_quality_gate_passed"]
    assert result["screen_selector_calls"] == 24
    assert result["timed_selector_calls"] == 216 and result["warmup_calls"] == 12
    assert result["executed_selector_calls"] == 252
    assert result["executed_reference_calls"] == 12 and result["executed_oracle_calls"] == 0
    assert result["total_executed_lp_selector_calls"] == 264
    assert result["score_unproved_timing"]["queue_count"] == 2
    assert result["other_trigger_timing"]["queue_count"] == 4
    assert result["historical_trigger_timing"]["queue_count"] == 6
    assert result["previous_untimed_probe_timing"]["queue_count"] == 2
    assert all(row["states"][4]["score_unproved"] and not row["states"][4]["score_unproved_regression"] for row in rows)
    lp_calls = 0
    for row in rows:
        for state in row["states"]:
            lp_calls += sum(record["selection"]["lp_calls"] for record in state["screen_results"].values())
            lp_calls += sum(record["selection"]["lp_calls"] * 6 for record in state["first_results"].values())
        lp_calls += sum(item["result"]["selection"]["lp_calls"] for item in row["warmups"])
    assert result["executed_method_search_work"]["lp_calls"] == lp_calls


def test_fresh_speedup_requires_fresh_performance_against_both_controls(case):
    protocol, rows, _ = case
    for row in rows:
        for state in row["states"][:2]:
            state["wall_ns"]["candidate"] = [90] * 6
    result = summary.summarize(protocol, rows)
    assert result["confirmation_gate_passed"] and result["fresh_speedup_established"]
    for row in rows:
        for state in row["states"][:2]:
            state["wall_ns"]["long"] = [90] * 6
    result = summary.summarize(protocol, rows)
    assert result["confirmation_gate_passed"] and not result["fresh_speedup_established"]


@pytest.mark.parametrize("failure", (
    "fresh_pooled", "fresh_p99", "fresh_cell", "score_gain", "score_state", "score_long",
    "other_pooled", "other_state", "trigger_long", "trigger_p99", "tail_probe", "tail_long"))
def test_targeted_gain_cannot_mask_each_registered_timing_gate(case, failure):
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
            state["wall_ns"] = {"candidate": [180] * 6, "probe": [200] * 6, "long": [200] * 6}
    elif failure == "score_gain":
        for row in rows:
            row["states"][2]["wall_ns"]["candidate"] = [96] * 6
    elif failure == "score_state":
        rows[0]["states"][2]["wall_ns"]["candidate"] = [106] * 6
    elif failure == "score_long":
        for row in rows:
            row["states"][2]["wall_ns"]["long"] = [50] * 6
    elif failure in ("other_pooled", "other_state"):
        for row in rows:
            row["states"][3]["wall_ns"]["candidate"] = [112 if failure == "other_pooled" else 106] * 6
            if failure == "other_state":
                row["states"][4]["wall_ns"]["candidate"] = [1] * 6
    elif failure == "trigger_long":
        for row in rows:
            for state in row["states"][2:5]:
                state["wall_ns"]["long"] = [70] * 6
    elif failure == "trigger_p99":
        rows[0]["states"][2]["wall_ns"]["candidate"] = [106] * 6
        rows[0]["states"][2]["wall_ns"]["probe"] = [200] * 6
    else:
        control = failure.removeprefix("tail_")
        rows[0]["states"][5]["wall_ns"][control] = [90] * 6
        rows[1]["states"][5]["wall_ns"]["candidate"] = [1] * 6
    result = summary.summarize(protocol, rows)
    assert result["decision"] == "CERTIFICATE_ROUTED_CONFIRMATION_TIMING_HOLD"
    assert result["candidate_quality_gate_passed"] and result["comparison_quality_gate_passed"]
    assert result["certificate_checks_passed"] and result["execution_checks_passed"]
    assert not result["confirmation_gate_passed"]
    gates = result["timing_gates"]
    if failure.startswith("fresh_"):
        gate = {"fresh_pooled": "pooled_no_material_regression", "fresh_p99": "p99_no_material_regression",
                "fresh_cell": "no_cell_material_regression"}[failure]
        assert not gates["fresh"]["probe"][gate]
        if failure != "fresh_pooled":
            assert gates["fresh"]["probe"]["pooled_no_material_regression"]
        if failure == "fresh_cell":
            assert gates["fresh"]["probe"]["p99_no_material_regression"]
    else:
        gate = {"score_gain": "score_unproved_probe_gain", "score_state": "score_unproved_probe_no_state_regression",
                "score_long": "score_unproved_long_no_regression", "other_pooled": "other_trigger_probe_no_regression",
                "other_state": "other_trigger_probe_no_state_regression", "trigger_long": "trigger_long_no_regression",
                "trigger_p99": "trigger_long_p99_no_regression", "tail_probe": "tail_no_state_regression_vs_probe",
                "tail_long": "tail_no_state_regression_vs_long"}[failure]
        assert not gates["regression"][gate]
        if failure == "score_state":
            assert gates["regression"]["score_unproved_probe_gain"]
        if failure == "other_state":
            assert gates["regression"]["other_trigger_probe_no_regression"]
        if failure == "trigger_p99":
            assert gates["regression"]["trigger_long_no_regression"]


def test_honest_candidate_nonclosure_retains_all_timing_and_paid_calls(case):
    protocol, rows, models = case
    protocol["lp_node_budget"] = 3
    for row, model in zip(rows, models):
        for state in row["states"]:
            candidate = asdict(select_certificate_routed_frame(model.options, state["queue"], 4, 3, 3, 8192))
            probe = asdict(select_resumable_probe_frame(model.options, state["queue"], 4, 3, 3, 8192))
            state["screen_results"].update(candidate=deepcopy(candidate), probe=deepcopy(probe))
            state["first_results"].update(candidate=candidate, probe=probe)
            state["expected_probe_result"] = deepcopy(probe)
        for item in row["warmups"]:
            item["result"] = deepcopy(row["states"][item["state_index"]]["first_results"][item["label"]])
    refresh(protocol, rows, models)
    result = summary.summarize(protocol, rows)
    assert result["decision"] == "CERTIFICATE_ROUTED_CONFIRMATION_QUALITY_HOLD"
    assert result["certificate_checks_passed"] and not result["candidate_quality_gate_passed"]
    assert result["timed_selector_calls"] == 216 and result["executed_selector_calls"] == 252


def test_honest_long_nonclosure_blocks_equal_quality_claim(case):
    protocol, rows, models = case
    protocol.update(long_lp_budget=3, control_integer_state_budget=0)
    for row, model in zip(rows, models):
        for state in row["states"]:
            state["first_results"]["long"] = asdict(select_strict_score_proof_canonical_completion_frame(
                model.options, state["queue"], 4, 3, 0))
        for item in row["warmups"]:
            if item["label"] == "long":
                item["result"] = deepcopy(row["states"][item["state_index"]]["first_results"]["long"])
    refresh(protocol, rows, models)
    result = summary.summarize(protocol, rows)
    assert result["decision"] == "CERTIFICATE_ROUTED_CONFIRMATION_QUALITY_HOLD"
    assert result["certificate_checks_passed"] and result["candidate_quality_gate_passed"]
    assert not result["comparison_quality_gate_passed"]


@pytest.mark.parametrize("fault", ("screen_candidate", "independent_probe", "warmup_long", "frozen_subgroup"))
def test_paid_work_and_independently_executed_control_cannot_be_replaced(case, fault):
    protocol, rows, models = case
    state = rows[0]["states"][2]
    if fault == "screen_candidate":
        state["screen_results"]["candidate"]["final_lp"]["selection"]["lp_calls"] = 1
    elif fault == "independent_probe":
        state["expected_probe_result"]["final_lp"]["capacity_propagation"]["region_visits"] += 1
    elif fault == "warmup_long":
        item = next(item for item in rows[0]["warmups"] if item["label"] == "long")
        item["result"]["selection"]["lp_calls"] += 1
    else:
        state["score_unproved_regression"] = False
    refresh(protocol, rows, models)
    result = summary.summarize(protocol, rows)
    assert result["decision"] == "INVALID_EXECUTION"
    assert not result["certificate_checks_passed"] and not result["confirmation_gate_passed"]


def test_changed_repeat_retains_its_extra_paid_work_and_blocks_timing(case):
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
    expected = sum(record["selection"]["candidate_evaluations"] for row in rows for state in row["states"]
                   for record in state["screen_results"].values())
    expected += sum(record["selection"]["candidate_evaluations"] * 6 for row in rows for state in row["states"]
                    for record in state["first_results"].values())
    expected += sum(item["result"]["selection"]["candidate_evaluations"] for row in rows for item in row["warmups"])
    assert result["executed_method_search_work"]["candidate_evaluations"] == expected + 1


@pytest.mark.parametrize("oracle_finishes", (False, True))
def test_incomplete_reference_is_retained_and_only_independent_oracle_completes_it(case, oracle_finishes):
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
    assert result["decision"] == ("CERTIFICATE_ROUTED_CONFIRMATION_ADVANCE" if oracle_finishes else "INCOMPLETE_REFERENCE")
    assert result["executed_oracle_calls"] == int(oracle_finishes)


@pytest.mark.parametrize("missing", ("long_repeat", "screen_probe", "registered_state"))
def test_registered_comparison_or_measurement_cannot_disappear(case, missing):
    protocol, rows, _ = case
    if missing == "long_repeat":
        rows[0]["states"][0]["wall_ns"]["long"].pop()
    elif missing == "screen_probe":
        del rows[0]["states"][0]["screen_results"]["probe"]
    else:
        rows[0]["states"].pop()
    with pytest.raises((ValueError, KeyError)):
        summary.summarize(protocol, rows)
