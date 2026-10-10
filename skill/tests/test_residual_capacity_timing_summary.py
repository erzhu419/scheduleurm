"""Fixed three-arm timing gates and fees from pure mocked selector records."""
from copy import deepcopy
from dataclasses import asdict
from itertools import permutations

import pytest

from algorithm.experiments import residual_capacity_timing_summary as summary
from algorithm.experiments.aligned_hardware_integer_capacity_development import _combine_metrics
from algorithm.experiments.resumable_probe_development_summary import _metric
from algorithm.experiments.score_face_completion_confirmation_summary import _key
from algorithm.score_face_canonical_completion_frame_candidates import ScoreFaceCompletionWork
from algorithm.strict_score_proof_canonical_completion_frame_candidates import StrictScoreProofWork
from skill.tests.test_residual_capacity_trigger_confirmation_summary import residual_state


def _long(state):
    selected = deepcopy(state["reference"])
    return {"selection": deepcopy(selected), "baseline_selection": deepcopy(selected),
        "score_certified_selection": deepcopy(selected),
        "preprocessing": deepcopy(state["traces"]["new"]["144"]["preprocessing"]),
        "capacity_propagation": deepcopy(state["traces"]["new"]["144"]["capacity_propagation"]),
        "integer_state_budget": 100000,
        "score_proof": asdict(StrictScoreProofWork("SKIPPED_CANONICAL_COMPLETE", False, True, 100000)),
        "completion": asdict(ScoreFaceCompletionWork("SKIPPED_CANONICAL_COMPLETE", False, True, 100000))}


def _single(state, result, model, label):
    selected = result["selection"]
    preprocessing = result["preprocessing"] if label == "long" else result["final_lp"]["preprocessing"]
    return _metric(selected, state, model, 2, selected["lp_node_budget"], _key(state["reference"], state["queue"], 1), preprocessing)


def _refresh(state, model):
    for label in summary.LABELS:
        state["metrics"][label] = _combine_metrics(_single(state, result, model, label)
            for result in summary._repeat_results(state, label, 6))


@pytest.fixture
def records():
    model = {"node": "toy", "lanes": 2, "options": [{"requirements": [1], "duration": 1.}]}
    states, metadata = [], []
    for index, (group, queue) in enumerate((("ordinary_development", 4), ("ordinary_development", 5),
            ("historical_regression", 6), ("zero_queue", 0), ("trigger_control", 7), ("new_trigger", 8))):
        source = "fresh_confirmation" if group == "new_trigger" else group
        registered = {"state_id": f"q{queue}", "queue": [queue], "source": source, "cohort": group,
            "panel": group, "seed": index + 10, "index": index, "caps": [1, 2, 144, 256],
            "historical_regression": group == "historical_regression", "timing_group": group,
            "timing_source": "trigger_confirmation" if group == "new_trigger" else "residual_development",
            "timing_source_index": index, "timing_source_state_id": f"q{queue}"}
        state = residual_state(registered, proved=group not in ("trigger_control", "new_trigger"))
        state.update(deepcopy(registered))
        state["expected_results"] = {"candidate": state.pop("routed"), "branch": state.pop("old_routed")}
        state.update(first_results=deepcopy(state["expected_results"]),
            repeat_mismatches={label: [] for label in summary.LABELS}, metrics={},
            wall_ns={"candidate": [80] * 6, "branch": [100] * 6, "long": [100] * 6},
            cpu_ns={"candidate": [70] * 6, "branch": [90] * 6, "long": [90] * 6})
        state["first_results"]["long"] = _long(state)
        _refresh(state, model)
        states.append(state)
        metadata.append(registered)
    orders = {"state_orders": [list(range(len(states))) for _ in range(6)],
        "label_orders": [[list(order) for order in permutations(summary.LABELS)] for _ in states]}
    for index, state in enumerate(states):
        state["blocks"] = [{"repeat": repeat, "order": orders["label_orders"][index][repeat]} for repeat in range(6)]
    keys = lambda group: [{"node": "toy", "lanes": 2, "queue": state["queue"]} for state in states if state["timing_group"] == group]
    protocol = {"experiment": "pure_mock_timing", "registered_at_utc": "frozen", "models": [model],
        "cells": [{"node": "toy", "lanes": 2, "states": metadata, "orders": orders}],
        "labels": list(summary.LABELS), "repeats": 6, "warmups_per_label": 2, "cpu": 3,
        "initial_lp_budget": 144, "lp_node_budget": 256, "integer_state_budget": 8192,
        "long_lp_budget": 4096, "control_integer_state_budget": 100000, "reference_budget": 4096,
        "oracle_state_budget": 5000000, "historical_regression_keys": keys("historical_regression"),
        "ordinary_tail_keys": keys("historical_regression"), "trigger_control_keys": keys("trigger_control"),
        "new_trigger_keys": keys("new_trigger"), "expected_state_count": 6, "expected_timed_calls": 108,
        "expected_warmup_calls": 6, "expected_selector_calls": 114,
        "expected_group_counts": {group: sum(state["timing_group"] == group for state in states) for group in summary.GROUPS},
        "nonregression_limit": 1.05, "trigger_speedup_limit": .95,
        "prior_decisions": {"axis_capacity_development": "AXIS_CAPACITY_FRONTIER_DEVELOPMENT_HOLD"},
        "limitations": "Pure mock local timing; no selectors executed."}
    runtime = {"cpu_affinity": [3], "gc_enabled": True}
    row = {"node": "toy", "lanes": 2, "physical_lanes": 2, "option_count": 1, "registered_at_utc": "frozen",
        "states": states, "state_orders": deepcopy(orders["state_orders"]), "warmups": [],
        "new_reference_calls": 0, "new_oracle_calls": 0, "screen_selector_calls": 0, "borrowed_reference_count": 6,
        **{key: deepcopy(runtime) for key in summary.RUNTIME_FIELDS}}
    for _ in range(2):
        for label in summary.LABELS:
            result = deepcopy(states[0]["first_results"][label])
            row["warmups"].append({"label": label, "state_index": 0, "result": result, "metrics": _single(states[0], result, model, label)})
    return protocol, [row]


def test_complete_balanced_roster_advances_and_keeps_all_costs_separate(records):
    result = summary.summarize(*records)
    assert result["decision"] == "RESIDUAL_CAPACITY_LOCAL_TIMING_ADVANCE" and result["prior_hold_retained"]
    assert result["state_count"] == result["queue_count"] == result["borrowed_reference_count"] == 6
    assert result["timed_selector_calls"] == 108 and result["warmup_calls"] == 6 and result["executed_selector_calls"] == 114
    assert result["executed_reference_calls"] == result["executed_oracle_calls"] == result["screen_selector_calls"] == 0
    assert result["work"]["candidate"]["axis_capacity"]["merged_values_copied"] == 36
    assert result["work"]["candidate"]["residual_capacity_completion"]["prefix_reads"] == 36
    assert result["work"]["candidate"]["shared_integer_states"] == 156
    assert result["work"]["candidate"]["frontier_export"]["fraction_pairs_copied"] == 60
    assert result["warmup_work"]["candidate"]["shared_integer_states"] == 0
    assert result["borrowed_reference_work"]["search_work"]["lp_calls"] == 6
    assert result["executed_search_work"]["lp_calls"] == sum(result["work"][label]["search_work"]["lp_calls"] +
        result["warmup_work"][label]["search_work"]["lp_calls"] for label in summary.LABELS)
    assert result["retained_table_active"]["queue_count"] == result["trigger_combined"]["queue_count"] == 2
    assert not result["accounting_failures"] and not result["quality_failures"]


@pytest.mark.parametrize("mutation", ("missing_repeat", "missing_state", "missing_warmup", "block_order", "sweep_order", "unbalanced_orders"))
def test_incomplete_or_unbalanced_measurements_are_rejected(records, mutation):
    protocol, rows = records
    row = rows[0]
    if mutation == "missing_repeat":
        row["states"][0]["wall_ns"]["candidate"].pop()
    elif mutation == "missing_state":
        row["states"].pop()
    elif mutation == "missing_warmup":
        row["warmups"].pop()
    elif mutation == "block_order":
        row["states"][0]["blocks"][0]["order"] = list(reversed(row["states"][0]["blocks"][0]["order"]))
    elif mutation == "sweep_order":
        row["state_orders"][0].reverse()
    else:
        protocol["cells"][0]["orders"]["label_orders"][0][1] = deepcopy(protocol["cells"][0]["orders"]["label_orders"][0][0])
    with pytest.raises(ValueError):
        summary.summarize(protocol, rows)


@pytest.mark.parametrize("mutation", ("new_reference", "lost_call", "wrong_group_count", "affinity", "frozen_output", "prefix_fee", "copy_fee"))
def test_registration_work_and_frozen_outputs_must_match(records, mutation):
    protocol, rows = records
    row, state = rows[0], rows[0]["states"][-1]
    if mutation == "new_reference":
        row["new_reference_calls"] = 1
    elif mutation == "lost_call":
        state["metrics"]["long"]["calls"] -= 1
    elif mutation == "wrong_group_count":
        protocol["expected_group_counts"]["new_trigger"] -= 1
    elif mutation == "affinity":
        row["runtime_end"]["cpu_affinity"] = [2, 3]
    elif mutation == "frozen_output":
        state["expected_results"]["branch"]["status"] = "CHANGED"
    elif mutation == "prefix_fee":
        state["first_results"]["candidate"]["probe"]["completion"]["prefix_reads"] -= 1
        state["expected_results"]["candidate"] = deepcopy(state["first_results"]["candidate"])
    else:
        state["first_results"]["candidate"]["probe"]["score_proof"]["capacity"]["merged_values_copied"] -= 1
        state["expected_results"]["candidate"] = deepcopy(state["first_results"]["candidate"])
    result = summary.summarize(protocol, rows)
    assert result["decision"] == "INVALID_TIMING_EXECUTION" and not result["local_timing_gate_passed"]


def _nonclosure(result, state):
    changed = deepcopy(result)
    for field in ("selection", "baseline_selection", "score_certified_selection"):
        changed[field].update(exact=False, canonical_exact=False, frontier_regions_remaining=1, canonical_tie_regions_remaining=1)
    changed["score_proof"]["status"] = "SKIPPED_SCORE_CERTIFIED"
    changed["completion"] = asdict(ScoreFaceCompletionWork("STATE_BUDGET_EXHAUSTED", True, False, 100000,
        states_visited=100000, score_fraction=(int(state["reference"]["score"]), 1)))
    return changed


def test_honest_long_nonclosure_has_quality_priority_over_repeat_instability(records):
    protocol, rows = records
    state = rows[0]["states"][-1]
    state["repeat_mismatches"]["long"] = [{"repeat_index": 2, "result": _nonclosure(state["first_results"]["long"], state)}]
    _refresh(state, protocol["models"][0])
    result = summary.summarize(protocol, rows)
    assert result["decision"] == "RESIDUAL_CAPACITY_TIMING_QUALITY_HOLD"
    assert result["certificate_checks_passed"] and not result["repeat_paths_stable"]
    assert result["work"]["long"]["shared_integer_states"] == 100000
    assert result["quality_failures"][0]["repeat_index"] == 2


def test_warmup_nonclosure_is_retained_and_charged(records):
    protocol, rows = records
    row, item = rows[0], rows[0]["warmups"][2]
    item["result"] = _nonclosure(item["result"], row["states"][0])
    item["metrics"] = _single(row["states"][0], item["result"], protocol["models"][0], "long")
    result = summary.summarize(protocol, rows)
    assert result["decision"] == "RESIDUAL_CAPACITY_TIMING_QUALITY_HOLD"
    assert result["warmup_work"]["long"]["shared_integer_states"] == 100000
    assert result["quality_failures"][0]["phase"] == "warmup"


def test_stable_quality_but_changed_repeat_output_invalidates_timing(records):
    protocol, rows = records
    state = rows[0]["states"][0]
    changed = deepcopy(state["first_results"]["long"])
    changed["capacity_propagation"]["region_visits"] += 1
    state["repeat_mismatches"]["long"] = [{"repeat_index": 4, "result": changed}]
    _refresh(state, protocol["models"][0])
    result = summary.summarize(protocol, rows)
    assert result["decision"] == "INVALID_TIMING_EXECUTION" and result["quality_gate_passed"]
    assert not result["repeat_paths_stable"]


@pytest.mark.parametrize("group", ("ordinary_development", "historical_regression", "trigger_control", "new_trigger"))
def test_slow_fixed_groups_cannot_hide_in_pooled_gain(records, group):
    protocol, rows = records
    for state in rows[0]["states"]:
        if state["timing_group"] == group:
            state["wall_ns"]["candidate"] = [120] * 6
        else:
            state["wall_ns"]["branch"] = state["wall_ns"]["long"] = [10000] * 6
    result = summary.summarize(protocol, rows)
    assert result["comparisons"]["branch"]["wall_ratio"] < 1.05
    assert result["decision"] == "RESIDUAL_CAPACITY_LOCAL_TIMING_HOLD"


def test_zero_time_is_descriptive_and_trigger_gain_is_mandatory(records):
    protocol, rows = records
    rows[0]["states"][3]["wall_ns"]["candidate"] = [100000] * 6
    result = summary.summarize(protocol, rows)
    assert result["decision"] == "RESIDUAL_CAPACITY_LOCAL_TIMING_ADVANCE"
    for state in rows[0]["states"][-2:]:
        state["wall_ns"]["candidate"] = [98] * 6
    result = summary.summarize(protocol, rows)
    assert result["decision"] == "RESIDUAL_CAPACITY_LOCAL_TIMING_HOLD"
    assert not result["timing_gates"]["long"]["trigger_pooled_five_percent_gain"]


def test_state_medians_keep_outliers_and_p99_observations(records):
    protocol, rows = records
    rows[0]["states"][0]["wall_ns"]["candidate"] = [1, 1, 50, 70, 300, 1000]
    result = summary.summarize(protocol, rows)
    ordinary = result["groups"]["ordinary_development"]
    assert ordinary["latency"]["candidate"]["sum_state_median_wall_ns"] == 140
    assert ordinary["latency"]["candidate"]["p99_state_median_wall_ms"] == 80 / 1e6
    assert ordinary["latency"]["candidate"]["sum_raw_wall_ns"] == 1902


def _second_cell(protocol, rows):
    cell, row, model = deepcopy(protocol["cells"][0]), deepcopy(rows[0]), deepcopy(protocol["models"][0])
    cell["node"] = row["node"] = model["node"] = "other"
    protocol["cells"].append(cell)
    protocol["models"].append(model)
    rows.append(row)
    for field in ("historical_regression_keys", "ordinary_tail_keys", "trigger_control_keys", "new_trigger_keys"):
        protocol[field].extend([{**item, "node": "other"} for item in deepcopy(protocol[field])])
    for field in ("expected_state_count", "expected_timed_calls", "expected_warmup_calls", "expected_selector_calls"):
        protocol[field] *= 2
    protocol["expected_group_counts"] = {key: 2 * value for key, value in protocol["expected_group_counts"].items()}
    return row


def test_ordinary_cell_regression_blocks_even_when_pooled_and_p99_pass(records):
    protocol, rows = records
    other = _second_cell(protocol, rows)
    for state in rows[0]["states"][:2]:
        state["wall_ns"]["candidate"] = [120] * 6
    for state in other["states"][:2]:
        state["wall_ns"]["branch"] = state["wall_ns"]["long"] = [10000] * 6
    result = summary.summarize(protocol, rows)
    assert result["decision"] == "RESIDUAL_CAPACITY_LOCAL_TIMING_HOLD"
    for label in summary.LABELS[1:]:
        assert result["timing_gates"][label]["ordinary_pooled_nonregression"]
        assert result["timing_gates"][label]["ordinary_p99_nonregression"]
        assert not result["timing_gates"][label]["ordinary_cell_nonregression"]


def test_every_fixed_historical_tail_blocks_despite_good_pooled_and_marginal_p99(records):
    protocol, rows = records
    other = _second_cell(protocol, rows)
    rows[0]["states"][2]["wall_ns"] = {"candidate": [8000] * 6, "branch": [10000] * 6, "long": [10000] * 6}
    other["states"][2]["wall_ns"] = {"candidate": [80] * 6, "branch": [60] * 6, "long": [60] * 6}
    result = summary.summarize(protocol, rows)
    assert result["decision"] == "RESIDUAL_CAPACITY_LOCAL_TIMING_HOLD"
    for label in summary.LABELS[1:]:
        assert result["timing_gates"][label]["historical_pooled_nonregression"]
        assert result["timing_gates"][label]["historical_p99_nonregression"]
        assert not result["timing_gates"][label]["historical_every_state_nonregression"]
    assert result["ordinary_tail"]["queue_count"] == 2


def test_honest_incomplete_borrowed_reference_has_priority_over_timing(records):
    protocol, rows = records
    state, model = rows[0]["states"][-1], protocol["models"][0]
    state["reference"].update(exact=False, score_exact=False, canonical_exact=False,
        upper_bound=state["reference"]["score"] + 1., gap=1., epsilon=1 / 8, frontier_regions_remaining=1)
    state.update(reference_complete=False, effective_reference_complete=False, effective_reference_source=None)
    state["reference_metrics"] = _metric(state["reference"], state, model, 2, 4096, None,
        state["traces"]["new"]["144"]["preprocessing"])
    for label in summary.LABELS:
        selected = state["first_results"][label]["selection"]
        preprocessing = (state["first_results"][label]["preprocessing"] if label == "long" else
                         state["first_results"][label]["final_lp"]["preprocessing"])
        state["metrics"][label] = _combine_metrics([_metric(selected, state, model, 2,
            selected["lp_node_budget"], None, preprocessing)] * 6)
    result = summary.summarize(protocol, rows)
    assert result["decision"] == "INCOMPLETE_REFERENCE" and result["certificate_checks_passed"]
    assert not result["references_complete"] and result["borrowed_reference_count"] == 6
