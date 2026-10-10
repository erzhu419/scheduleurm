"""A/A fixed roles, full call coverage and nested-phase accounting without selectors."""
from copy import deepcopy
from itertools import permutations

import pytest

from algorithm.experiments.certificate_routed_aa_diagnosis_summary import summarize


@pytest.fixture
def case():
    labels, phase_labels = ["a", "b", "c"], ["candidate", "probe"]
    quality = {"failures": [], "score_mismatch_count": 0, "canonical_action_mismatch_count": 0}
    expected = {"selection": {"exact": True, "score_exact": True, "canonical_exact": True}}
    states = [{"name": name, "node": "node007", "lanes": 64, "queue": queue,
        "effective_reference_complete": True,
        "expected_results": {label: deepcopy(expected) for label in phase_labels},
        "screen_metrics": {label: deepcopy(quality) for label in phase_labels}}
        for name, queue in (("tail_88", [88, 19, 118]), ("tail_184", [184, 78, 235]))]
    aa_orders = [list(order) for order in permutations(labels)]
    def design(orders):
        return {"state_orders": [[0, 1] for _ in orders],
                "label_orders": [deepcopy(orders) for _ in states]}
    protocol = {"states": states, "labels": labels, "phase_labels": phase_labels,
        "windows": 12, "repeats": 6, "warmups": 2, "phase_repeats": 6, "phase_warmups": 2,
        "cpu": 3, "prior_quality_passed": True, "prior_decision": "CERTIFICATE_ROUTED_CONFIRMATION_TIMING_HOLD",
        "limitations": "Two selected queues in one process.",
        "aa_orders": {"state_orders": [[[0, 1] for _ in range(6)] for _ in range(12)],
                      "label_orders": [[deepcopy(aa_orders) for _ in range(12)] for _ in states]},
        "aa_warmup_orders": design([labels, list(reversed(labels))]),
        "phase_orders": design([phase_labels, list(reversed(phase_labels))] * 3),
        "phase_warmup_orders": design([phase_labels, list(reversed(phase_labels))])}
    rows = []
    for index, state in enumerate(states):
        row = {"name": state["name"], "calls": [], "blocks": [],
               "first_results": deepcopy(state["expected_results"]), "mismatches": []}
        for stage, kind, windows, orders in (
            ("aa", "warmup", [-1], protocol["aa_warmup_orders"]["label_orders"][index]),
            ("aa", "timed", range(12), aa_orders),
            ("phase", "warmup", [-1], protocol["phase_warmup_orders"]["label_orders"][index]),
            ("phase", "timed", [-1], protocol["phase_orders"]["label_orders"][index]),
        ):
            for window in windows:
                for repeat, order in enumerate(orders):
                    pointer = {"stage": stage, "kind": kind, "window": window, "repeat": repeat}
                    row["blocks"].append({**pointer, "order": list(order)})
                    for label in order:
                        call = {**pointer, "label": label, "wall_ns": 100, "cpu_ns": 80, "matches_frozen": True}
                        if stage == "phase":
                            call.update(phases={
                                "initial_lp": {"wall_ns": 90, "cpu_ns": 70, "calls": 1},
                                "initial_lp_linprog": {"wall_ns": 60, "cpu_ns": 45, "calls": 21},
                                "linprog": {"wall_ns": 60, "cpu_ns": 45, "calls": 21}},
                                primary_phase_wall_ns=90, primary_phase_cpu_ns=70,
                                residual_wall_ns=10, residual_cpu_ns=10,
                                gc_events=[{"wall_ns": 5, "cpu_ns": 4}])
                        row["calls"].append(call)
        rows.append(row)
    record = {"runtime_start": {"cpu_affinity": [3]}, "runtime_end": {"cpu_affinity": [3]},
              "states": rows, "work": {"candidate": {"lp_calls": 9200}, "probe": {"lp_calls": 320}}, "elapsed_s": 1.}
    return protocol, record


def test_complete_accounting_keeps_hold_and_no_independent_error_rate(case):
    protocol, record = case
    result = summarize(protocol, record)
    assert result["decision"] == "AA_DIAGNOSIS_COMPLETE"
    assert result["prior_hold_retained"] and result["prior_decision"].endswith("HOLD")
    assert result["executed_selector_calls"] == 476
    assert result["call_counts"] == {"aa": {"warmup": 12, "timed": 432}, "phase": {"warmup": 8, "timed": 24}}
    assert result["selector_calls_by_implementation"] == {"candidate": 460, "probe": 16}
    assert result["executed_reference_calls"] == result["executed_oracle_calls"] == 0
    assert result["borrowed_reference_count"] == 2 and result["work"] == record["work"]
    assert result["aa_readout"]["state_windows"] == 24
    assert result["aa_readout"]["windows_with_apparent_slowdown"] == 0
    assert "not independent" in result["aa_readout"]["interpretation"]


def test_fixed_candidate_role_is_not_replaced_by_symmetric_extremes(case):
    protocol, record = case
    for call in record["states"][0]["calls"]:
        if call["stage"] == "aa" and call["kind"] == "timed" and call["window"] == 0 and call["label"] == "b":
            call["wall_ns"] = 110
    result = summarize(protocol, record)
    first = result["states"][0]["aa"]["windows"][0]
    assert first["a_over_control_wall_ratio"] == {"b": 100 / 110, "c": 1.}
    assert not first["any_apparent_slowdown_over_5pct"]
    assert first["symmetric_wall_max_min_ratio"] == 1.1
    assert result["aa_readout"]["windows_with_apparent_slowdown"] == 0
    assert result["aa_readout"]["windows_with_symmetric_spread_over_5pct"] == 1


def test_strict_five_percent_gate_and_cpu_are_separate(case):
    protocol, record = case
    for call in record["states"][0]["calls"]:
        if call["stage"] == "aa" and call["kind"] == "timed" and call["label"] == "a":
            call["wall_ns"] = 106 if call["window"] == 0 else 105
            call["cpu_ns"] = 88
    result = summarize(protocol, record)
    assert result["aa_readout"]["windows_with_apparent_slowdown"] == 1
    assert result["aa_readout"]["apparent_slowdown_counts_by_control"] == {"b": 1, "c": 1}
    assert result["states"][0]["aa"]["windows"][1]["a_over_control_cpu_ratio"] == {"b": 1.1, "c": 1.1}


@pytest.mark.parametrize("mutation", ["missing_call", "duplicate_call", "missing_block", "duplicate_block",
    "block_order", "call_order", "mismatch", "warmup_mismatch", "mismatch_record", "first_result",
    "incomplete_reference", "screen_quality", "affinity", "prior_quality", "state_roster"])
def test_real_execution_failures_invalidate_diagnosis(case, mutation):
    protocol, record = case
    row = record["states"][0]
    if mutation == "missing_call":
        row["calls"].pop()
    elif mutation == "duplicate_call":
        row["calls"].append(deepcopy(row["calls"][0]))
    elif mutation == "missing_block":
        row["blocks"].pop()
    elif mutation == "duplicate_block":
        row["blocks"].append(deepcopy(row["blocks"][0]))
    elif mutation == "block_order":
        row["blocks"][0]["order"].reverse()
    elif mutation == "call_order":
        row["calls"][0], row["calls"][1] = row["calls"][1], row["calls"][0]
    elif mutation in ("mismatch", "warmup_mismatch"):
        row["calls"][-1 if mutation == "mismatch" else 0]["matches_frozen"] = False
    elif mutation == "mismatch_record":
        row["mismatches"].append({"result": "wrong"})
    elif mutation == "first_result":
        row["first_results"]["probe"]["selection"]["canonical_exact"] = False
    elif mutation == "incomplete_reference":
        protocol["states"][0]["effective_reference_complete"] = False
    elif mutation == "screen_quality":
        protocol["states"][0]["screen_metrics"]["candidate"]["failures"].append("wrong")
    elif mutation == "affinity":
        record["runtime_end"]["cpu_affinity"] = [4]
    elif mutation == "prior_quality":
        protocol["prior_quality_passed"] = False
    elif mutation == "state_roster":
        record["states"].reverse()
    result = summarize(protocol, record)
    assert result["decision"] == "INVALID_EXECUTION"
    assert result["execution_failures"] and not result["states"]
    assert result["prior_hold_retained"]


def test_nested_linprog_is_subtracted_once_and_gc_fraction_uses_full_calls(case):
    result = summarize(*case)
    phase = result["states"][0]["phase"]["candidate"]
    assert phase["wall_ns"] == 100 and phase["primary_phase_wall_ns"] == 90
    assert phase["initial_lp_less_linprog_wall_ns"] == 30
    assert phase["initial_lp_less_linprog_cpu_ns"] == 25
    assert phase["residual_wall_ns"] == 10
    assert phase["phases"]["initial_lp_linprog"]["total_calls"] == 126
    assert phase["gc_events"] == 6 and phase["gc_pause_wall_ns"] == 30
    assert phase["gc_pause_fraction_of_whole"] == .05
