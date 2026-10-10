"""Decisive timing gates and accounting, using only tiny synthetic models."""
from copy import deepcopy
from dataclasses import asdict
from fractions import Fraction

import pytest

from algorithm.experiments import aligned_hardware_integer_capacity_development as runtime
from algorithm.experiments import strict_score_proof_timing_summary as summary
from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption
from algorithm.experiments.aligned_hardware_queue_aware_canonical_development import selection_from_record
from algorithm.strict_score_proof_canonical_completion_frame_candidates import (
    select_strict_score_proof_canonical_completion_frame,
)


@pytest.fixture
def case():
    model = FrameModel(node="synthetic", scenario_id="tiny", lanes=2,
                       workloads=("a", "b", "c"), scales=(1., 1., 1.),
                       options=(FrameOption("a", (1, 0, 0), 1., ((1.,), (), ())),))
    protocol = {
        "experiment": "synthetic_timing", "registered_at_utc": "test",
        "models": [model.snapshot()], "lanes": [2], "repeats": 6, "warmups_per_label": 2,
        "lp_budgets": {"candidate": 144, "control": 4096}, "integer_state_budget": 100000,
        "reference_budget": 4096, "oracle_state_budget": 5000000, "cpu": 0,
        "pooled_ratio_limit": .95, "cell_ratio_limit": 1., "limitations": "Synthetic only.",
        "queue_panels": {"2": [
            {"panel": "fresh", "seed": 123, "cohort": "fresh_confirmation",
             "states": [{"state_id": "fresh:0", "queue": [2, 0, 0]}]},
            {"panel": "regression", "seed": None, "cohort": "old_regression",
             "states": [{"state_id": "old:0", "queue": [1, 0, 0]}]},
        ]},
    }
    states = []
    for panel in protocol["queue_panels"]["2"]:
        registered = panel["states"][0]
        queue = tuple(registered["queue"])
        reference = runtime.select_canonical_count_space_bnb_frame(model.options, queue, 2, 4096)
        expected = runtime.reference_key(reference, queue, 1)
        reference_metrics = runtime._metrics()
        runtime._record(reference_metrics, reference, model, queue, 2, 4096, expected)
        results = {label: asdict(select_strict_score_proof_canonical_completion_frame(
            model.options, queue, 2, cap, 100000)) for label, cap in protocol["lp_budgets"].items()}
        if panel["seed"] is None:
            original = deepcopy(results["candidate"]["selection"])
            results["candidate"] = _honest_nonclosure(results["candidate"], queue)
            results["candidate"]["selection"] = original
            results["candidate"]["completion"].update(status="COMPLETE", complete=True, states_visited=7)
        state = {**registered, **{key: panel[key] for key in ("panel", "cohort", "seed")},
                 "reference": asdict(reference), "reference_metrics": reference_metrics,
                 "reference_complete": True, "oracle": None, "effective_reference_complete": True,
                 "effective_reference_source": "full_K4096", "previous_integer_trigger": panel["seed"] is None,
                 "expected_candidate_result": deepcopy(results["candidate"]),
                 "first_results": results, "repeat_mismatches": {label: [] for label in summary.LABELS},
                 "wall_ns": {"candidate": [80] * 6, "control": [100] * 6},
                 "cpu_ns": {"candidate": [70] * 6, "control": [90] * 6}, "metrics": {}}
        _refresh_metrics(state, model, protocol)
        states.append(state)
    row = {"node": model.node, "lanes": 2, "physical_lanes": 2, "option_count": 1,
           "registered_at_utc": "test", "states": states,
           **{key: {"cpu_affinity": [0]} for key in
              ("runtime_start", "runtime_before_warmup", "runtime_after_warmup", "runtime_end")},
           "warmups": []}
    for _ in range(protocol["warmups_per_label"]):
        index, state = 0, states[0]
        for label in summary.LABELS:
            result = deepcopy(state["first_results"][label])
            metric = _single_metric(result, state, model, protocol, label)
            row["warmups"].append({"label": label, "state_index": index, "result": result, "metrics": metric})
    return protocol, row, model


def _single_metric(result, state, model, protocol, label):
    metric = runtime._metrics()
    reference = selection_from_record(state["reference"])
    expected = runtime.reference_key(reference, state["queue"], len(model.options))
    runtime._record(metric, selection_from_record(result["selection"]), model, tuple(state["queue"]), 2,
                    protocol["lp_budgets"][label], expected, result["preprocessing"], reference)
    return metric


def _refresh_metrics(state, model, protocol):
    for label in summary.LABELS:
        state["metrics"][label] = runtime._combine_metrics(
            _single_metric(result, state, model, protocol, label)
            for result in summary._repeat_results(state, label, protocol["repeats"]))


def _honest_nonclosure(result, queue):
    result = deepcopy(result)
    for name in ("selection", "baseline_selection", "score_certified_selection"):
        result[name].update(exact=False, canonical_exact=False, canonical_tie_regions_remaining=1,
                            frontier_regions_remaining=1)
    result["score_proof"]["status"] = "SKIPPED_SCORE_CERTIFIED"
    score = Fraction(sum(q * done for q, done in zip(queue, result["selection"]["service"]))) / Fraction.from_float(result["selection"]["duration"])
    result["completion"].update(status="STATE_BUDGET_EXHAUSTED", attempted=True, complete=False,
                                 states_visited=100000, score_fraction=[score.numerator, score.denominator])
    return result


def test_complete_pair_advances_and_only_actual_calls_are_charged(case):
    protocol, row, _ = case
    result = summary.summarize(protocol, [row])
    assert result["decision"] == "STRICT_SCORE_PROOF_LOCAL_TIMING_ADVANCE"
    assert result["timed_selector_calls"] == 24
    assert result["warmup_calls"] == 4
    assert result["executed_selector_calls"] == result["embedded_baseline_count"] == 28
    assert result["executed_reference_calls"] == result["executed_oracle_calls"] == 0
    assert result["reused_reference_count"] == 2
    assert result["fresh_confirmation"]["queue_count"] == result["regression"]["queue_count"] == 1
    assert result["previous_integer_trigger"]["queue_count"] == 1
    expected_lp = sum(result["work"][label]["search_work"]["lp_calls"] +
                      result["warmup_work"][label]["search_work"]["lp_calls"] for label in summary.LABELS)
    assert result["executed_search_work"]["lp_calls"] == expected_lp
    assert all(result["metrics"][label]["calls"] == 12 for label in summary.LABELS)


@pytest.mark.parametrize("label", summary.LABELS)
def test_honest_nonclosure_in_either_arm_retained_and_blocks_speed_claim(case, label):
    protocol, row, model = case
    state = row["states"][0]
    changed = _honest_nonclosure(state["first_results"][label], state["queue"])
    state["repeat_mismatches"][label] = [{"repeat_index": 4, "result": changed}]
    _refresh_metrics(state, model, protocol)
    result = summary.summarize(protocol, [row])
    assert result["decision"] == "STRICT_SCORE_PROOF_TIMING_QUALITY_HOLD"
    assert result["certificate_checks_passed"]
    assert not result["quality_gate_passed"] and not result["local_timing_gate_passed"]
    assert result["timed_selector_calls"] == 24
    assert result["metrics"][label]["canonical_exact_count"] == 11
    assert result["work"][label]["shared_integer_states"] == 100000 + (42 if label == "candidate" else 0)
    assert result["quality_failures"][0]["repeat_index"] == 4


def test_warmup_nonclosure_is_quality_hold_and_work_is_charged(case):
    protocol, row, model = case
    warmup = row["warmups"][0]
    state = row["states"][warmup["state_index"]]
    warmup["result"] = _honest_nonclosure(warmup["result"], state["queue"])
    warmup["metrics"] = _single_metric(warmup["result"], state, model, protocol, warmup["label"])
    result = summary.summarize(protocol, [row])
    assert result["decision"] == "STRICT_SCORE_PROOF_TIMING_QUALITY_HOLD"
    assert result["warmup_work"]["candidate"]["shared_integer_states"] == 100000
    assert result["quality_failures"][0]["phase"] == "warmup"


def test_regression_cohort_cannot_be_hidden_by_fresh_speedup(case):
    protocol, row, _ = case
    row["states"][0]["wall_ns"]["control"] = [10000] * 6
    row["states"][1]["wall_ns"]["candidate"] = [120] * 6
    result = summary.summarize(protocol, [row])
    assert result["comparison"]["wall_ratio"] < .95
    assert result["decision"] == "STRICT_SCORE_PROOF_LOCAL_TIMING_HOLD"
    assert result["timing_gates_by_cohort"]["fresh_confirmation"]["pooled_at_least_five_percent_faster"]
    assert not result["timing_gates_by_cohort"]["regression"]["pooled_at_least_five_percent_faster"]
    assert result["previous_integer_trigger"]["comparison"]["wall_ratio"] == 1.2


def test_nonempty_cohort_cell_regression_blocks_despite_pooled_gain(case):
    protocol, row, _ = case
    other = deepcopy(row)
    other["node"] = "other"
    other_model = deepcopy(protocol["models"][0])
    other_model["node"] = "other"
    protocol["models"].append(other_model)
    row["states"][0]["wall_ns"]["candidate"] = [120] * 6
    other["states"][0]["wall_ns"]["control"] = [10000] * 6
    result = summary.summarize(protocol, [row, other])
    assert result["fresh_confirmation"]["comparison"]["wall_ratio"] < .95
    assert not result["timing_gates_by_cohort"]["fresh_confirmation"]["no_node_lane_cell_regression"]
    assert result["decision"] == "STRICT_SCORE_PROOF_LOCAL_TIMING_HOLD"


def test_latency_uses_each_state_median_and_preserves_tails(case):
    protocol, row, _ = case
    row["states"][0]["wall_ns"]["candidate"] = [1, 1, 50, 70, 300, 1000]
    result = summary.summarize(protocol, [row])
    assert result["latency"]["candidate"]["sum_state_median_wall_ns"] == 60 + 80
    assert result["latency"]["candidate"]["p99_state_median_wall_ms"] == 80 / 1e6
    assert result["latency"]["candidate"]["p95_raw_call_wall_ms"] == 1000 / 1e6
    assert result["comparison"]["max_paired_state_slowdown"] == .8


@pytest.mark.parametrize("failure", ("missing_repeat", "missing_state", "duplicate_state", "missing_warmup", "duplicate_override"))
def test_incomplete_measurement_rosters_are_rejected(case, failure):
    protocol, row, _ = case
    if failure == "missing_repeat":
        row["states"][0]["wall_ns"]["candidate"].pop()
    elif failure == "missing_state":
        row["states"].pop()
    elif failure == "duplicate_state":
        row["states"].append(deepcopy(row["states"][0]))
    elif failure == "missing_warmup":
        row["warmups"].pop()
    else:
        state = row["states"][0]
        changed = _honest_nonclosure(state["first_results"]["candidate"], state["queue"])
        state["repeat_mismatches"]["candidate"] = [{"repeat_index": 4, "result": changed}] * 2
    with pytest.raises(ValueError):
        summary.summarize(protocol, [row])


@pytest.mark.parametrize("failure", ("lost_call", "counter", "shared_cap", "affinity", "changed_candidate", "changed_subgroup"))
def test_accounting_or_execution_failures_never_advance(case, failure):
    protocol, row, model = case
    state = row["states"][0]
    if failure == "lost_call":
        state["metrics"]["control"]["calls"] -= 1
    elif failure == "counter":
        state["metrics"]["candidate"]["search_work"]["lp_calls"] += 1
    elif failure == "shared_cap":
        changed = _honest_nonclosure(state["first_results"]["candidate"], state["queue"])
        changed["completion"]["states_visited"] += 1
        state["repeat_mismatches"]["candidate"] = [{"repeat_index": 1, "result": changed}]
        _refresh_metrics(state, model, protocol)
    elif failure == "affinity":
        row["runtime_end"]["cpu_affinity"] = [0, 1]
    elif failure == "changed_candidate":
        state["expected_candidate_result"]["capacity_propagation"]["region_visits"] += 1
    else:
        state["previous_integer_trigger"] = True
    result = summary.summarize(protocol, [row])
    assert result["decision"] == "INVALID_TIMING_EXECUTION"
    assert not result["local_timing_gate_passed"]


def test_stable_quality_but_changed_work_invalidates_repeat_timing(case):
    protocol, row, model = case
    state = row["states"][0]
    changed = deepcopy(state["first_results"]["control"])
    changed["capacity_propagation"]["region_visits"] += 1
    state["repeat_mismatches"]["control"] = [{"repeat_index": 2, "result": changed}]
    _refresh_metrics(state, model, protocol)
    result = summary.summarize(protocol, [row])
    assert result["quality_gate_passed"]
    assert not result["repeat_paths_stable"]
    assert result["decision"] == "INVALID_TIMING_EXECUTION"
