"""Mocked replay coverage, actual implementation costs and A/A directionality."""
from collections import Counter
from copy import deepcopy
from itertools import permutations

import pytest

from algorithm.experiments.certificate_routed_cadence_diagnosis_summary import summarize


@pytest.fixture
def case():
    labels = ["candidate", "probe", "long"]
    orders = [list(order) for order in permutations(labels)]
    states = [{"state_id": name, "queue": queue, "target": target, "effective_reference_complete": True,
               "expected_results": {role: {"implementation": role,
                   "selection": {"exact": True, "score_exact": True, "canonical_exact": True}} for role in labels}}
              for name, queue, target in (("ordinary", [1, 2, 3], False), ("tail88", [88, 19, 118], True),
                                           ("tail184", [184, 78, 235], True))]
    cell = {"node": "node007", "lanes": 64, "states": states,
            "orders": {"state_orders": [[0, 1, 2], [2, 0, 1]] * 3,
                       "label_orders": [deepcopy(orders) for _ in states]}}
    protocol = {"cells": [cell], "labels": labels, "repeats": 6, "warmups_per_label": 2, "cpu": 3,
                "prior_quality_passed": True, "prior_decision": "CERTIFICATE_ROUTED_CONFIRMATION_TIMING_HOLD",
                "limitations": "Two selected targets, not an independent error rate; old screens are omitted.",
                "expected_selector_calls": 60, "expected_timed_calls": 54, "expected_warmup_calls": 6,
                "expected_implementation_calls": {"candidate": 44, "probe": 8, "long": 8}}
    provenance = {"cpu_affinity": [3], "gc_enabled": True, "thread_environment": {"OMP_NUM_THREADS": "1"},
                  "hostname": "host", "platform": "linux", "python": "3", "scipy": "1", "cpu_model": "cpu",
                  "gc_threshold": [700, 10, 10], "native_thread_count": 2}
    row = {"node": "node007", "lanes": 64, "states": [], "work": {}, "elapsed_s": 10.,
           **{key: deepcopy(provenance) for key in ("runtime_start", "runtime_before_warmup", "runtime_after_warmup", "runtime_end")}}
    for state in states:
        row["states"].append({"state_id": state["state_id"], "queue": state["queue"], "target": state["target"],
            "first_results": {role: deepcopy(state["expected_results"]["candidate" if state["target"] else role]) for role in labels},
            "calls": [], "blocks": [], "metrics": {}, "mismatches": []})
    timeline = [("warmup", repeat, 0, [role]) for repeat in range(2) for role in labels]
    timeline += [("timed", repeat, index, cell["orders"]["label_orders"][index][repeat])
                 for repeat, indices in enumerate(cell["orders"]["state_orders"]) for index in indices]
    for tick, (kind, repeat, index, order) in enumerate(timeline):
        actual = row["states"][index]
        actual["blocks"].append({"kind": kind, "repeat": repeat, "order": order, "start_ns": tick * 1_000_000_000})
        actual["calls"].extend({"kind": kind, "repeat": repeat, "role": role,
            "implementation": "candidate" if actual["target"] else role,
            "wall_ns": 100, "cpu_ns": 80, "matches_frozen": True} for role in order)
    counts = Counter()
    for state in row["states"]:
        counts.update(call["implementation"] for call in state["calls"])
        for role in labels:
            count = sum(call["role"] == role for call in state["calls"])
            state["metrics"][role] = {**dict.fromkeys(("calls", "score_exact_count", "canonical_exact_count", "reference_comparable_count"), count),
                **dict.fromkeys(("failures", "score_mismatch_count", "canonical_action_mismatch_count"), 0)}
    row["work"] = {role: {"calls": counts[role], "search_work": {"lp_calls": counts[role] * 21},
                          "shared_integer_states": counts[role] * 2, "max_shared_integer_states": 2} for role in labels}
    return protocol, [row]


def test_complete_cadence_preserves_prior_hold_and_charges_actual_implementation(case):
    summary = summarize(*case)
    assert summary["decision"] == "CADENCE_DIAGNOSIS_COMPLETE"
    assert summary["prior_hold_retained"] and summary["prior_decision"].endswith("HOLD")
    assert summary["executed_selector_calls"] == 60
    assert summary["call_counts"] == {"warmup": 6, "timed": 54}
    assert summary["selector_calls_by_implementation"] == {"candidate": 44, "probe": 8, "long": 8}
    assert summary["borrowed_reference_count"] == 3
    assert summary["executed_reference_calls"] == summary["executed_oracle_calls"] == 0
    assert summary["work"]["candidate"]["search_work"]["lp_calls"] == 924
    assert summary["aa_readout"]["states_with_apparent_slowdown"] == 0
    for target in summary["targets"]:
        assert len(target["consecutive_block_gaps_s"]) == 5
        assert all(gap > 0 for gap in target["consecutive_block_gaps_s"])
        assert target["arms"]["candidate"]["wall_ns"] == [100] * 6


def test_fixed_candidate_role_and_symmetric_spread_have_distinct_meanings(case):
    protocol, rows = case
    for call in rows[0]["states"][1]["calls"]:
        if call["role"] == "probe":
            call["wall_ns"] = 120
    result = summarize(protocol, rows)
    target = result["targets"][0]
    assert target["candidate_over_control_wall_ratio"] == {"probe": 100 / 120, "long": 1.}
    assert target["symmetric_wall_max_min_ratio"] == 1.2
    assert not target["any_apparent_slowdown_over_5pct"]
    assert result["aa_readout"]["states_with_symmetric_spread_over_5pct"] == 1


def test_exact_five_percent_is_not_failure_and_cpu_is_separate(case):
    protocol, rows = case
    for index, state in enumerate(rows[0]["states"][1:]):
        for call in state["calls"]:
            if call["role"] == "candidate":
                call.update(wall_ns=105 + index, cpu_ns=88)
    result = summarize(protocol, rows)
    assert result["aa_readout"]["states_with_apparent_slowdown"] == 1
    assert result["targets"][0]["candidate_over_control_cpu_ratio"] == {"probe": 1.1, "long": 1.1}


@pytest.mark.parametrize("mutation", ["cell_roster", "state_roster", "missing_call", "duplicate_call", "call_order",
    "missing_block", "label_order", "state_cadence", "wrong_target_implementation", "wrong_background_implementation",
    "first_result", "incomplete_reference", "quality", "metric_count", "mismatch", "warmup_mismatch", "mismatch_record",
    "affinity", "disabled_gc", "thread_environment", "runtime_drift", "work_count", "prior_quality", "invalid_duration"])
def test_reachable_execution_failures_are_not_reported_as_aa_results(case, mutation):
    protocol, rows = case
    cell, actual = protocol["cells"][0], rows[0]
    target = actual["states"][1]
    if mutation == "cell_roster":
        rows.append(deepcopy(actual))
    elif mutation == "state_roster":
        actual["states"].reverse()
    elif mutation == "missing_call":
        target["calls"].pop()
    elif mutation == "duplicate_call":
        target["calls"].append(deepcopy(target["calls"][-1]))
    elif mutation == "call_order":
        target["calls"][0], target["calls"][1] = target["calls"][1], target["calls"][0]
    elif mutation == "missing_block":
        target["blocks"].pop()
    elif mutation == "label_order":
        target["blocks"][0]["order"] = list(reversed(target["blocks"][0]["order"]))
    elif mutation == "state_cadence":
        for index, state in enumerate(actual["states"]):
            for block in state["blocks"]:
                if block["kind"] == "timed":
                    block["start_ns"] = (100 + index * 10 + block["repeat"]) * 1_000_000_000
    elif mutation == "wrong_target_implementation":
        target["calls"][1]["implementation"] = "probe"
    elif mutation == "wrong_background_implementation":
        actual["states"][0]["calls"][1]["implementation"] = "candidate"
    elif mutation == "first_result":
        target["first_results"]["probe"] = deepcopy(cell["states"][1]["expected_results"]["probe"])
    elif mutation == "incomplete_reference":
        cell["states"][1]["effective_reference_complete"] = False
    elif mutation == "quality":
        target["metrics"]["candidate"]["canonical_action_mismatch_count"] = 1
    elif mutation == "metric_count":
        target["metrics"]["candidate"]["calls"] = 7
    elif mutation == "mismatch":
        target["calls"][0]["matches_frozen"] = False
    elif mutation == "warmup_mismatch":
        actual["states"][0]["calls"][0]["matches_frozen"] = False
    elif mutation == "mismatch_record":
        target["mismatches"].append({"result": "wrong"})
    elif mutation == "affinity":
        actual["runtime_after_warmup"]["cpu_affinity"] = [4]
    elif mutation == "disabled_gc":
        actual["runtime_before_warmup"]["gc_enabled"] = False
    elif mutation == "thread_environment":
        actual["runtime_start"]["thread_environment"]["OMP_NUM_THREADS"] = "2"
    elif mutation == "runtime_drift":
        actual["runtime_end"]["scipy"] = "different"
    elif mutation == "work_count":
        actual["work"]["candidate"]["calls"] -= 1
    elif mutation == "prior_quality":
        protocol["prior_quality_passed"] = False
    elif mutation == "invalid_duration":
        target["calls"][0]["wall_ns"] = 0
    result = summarize(protocol, rows)
    assert result["decision"] == "INVALID_EXECUTION" and result["execution_failures"]
    assert not result["targets"] and "aa_readout" not in result
    assert result["prior_hold_retained"]


def test_combined_work_sums_charges_but_not_per_call_maximum(case):
    protocol, rows = case
    protocol["cells"].append(deepcopy(protocol["cells"][0]))
    protocol["cells"][1]["node"] = "second"
    rows.append(deepcopy(rows[0]))
    rows[1]["node"] = "second"
    for key in ("expected_selector_calls", "expected_timed_calls", "expected_warmup_calls"):
        protocol[key] *= 2
    protocol["expected_implementation_calls"] = {key: value * 2 for key, value in protocol["expected_implementation_calls"].items()}
    result = summarize(protocol, rows)
    assert result["decision"] == "CADENCE_DIAGNOSIS_COMPLETE"
    assert result["work"]["candidate"]["search_work"]["lp_calls"] == 1848
    assert result["work"]["candidate"]["shared_integer_states"] == 176
    assert result["work"]["candidate"]["max_shared_integer_states"] == 2


def test_once_per_cell_json_normalization_preserves_native_selector_tuples(case):
    import json

    protocol, rows = case
    for state in protocol["cells"][0]["states"]:
        for result in state["expected_results"].values():
            result["selection"]["option_indices"] = [0, None, 2]
    for state in rows[0]["states"]:
        for result in state["first_results"].values():
            result["selection"]["option_indices"] = (0, None, 2)
    saved_rows = json.loads(json.dumps(rows))
    assert summarize(protocol, saved_rows)["decision"] == "CADENCE_DIAGNOSIS_COMPLETE"
