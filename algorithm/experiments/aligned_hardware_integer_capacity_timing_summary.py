"""Host-local timing summaries; repetitions are not independent queue samples."""
from __future__ import annotations

import math
from statistics import median


CONTROLS = ("old_shared", "fractional_shared")
CANDIDATE = "integer_capacity"
LABELS = (*CONTROLS, CANDIDATE)


def _p95(values):
    """Empirical nearest-rank percentile, with no interpolation or trimming."""
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def _latency(states, label):
    wall = [median(state["wall_ns"][label]) for state in states]
    cpu = [median(state["cpu_ns"][label]) for state in states]
    raw_wall = [value for state in states for value in state["wall_ns"][label]]
    return {
        "sum_state_median_wall_ns": sum(wall),
        "mean_state_median_wall_ms": sum(wall) / len(wall) / 1e6,
        "median_state_median_wall_ms": median(wall) / 1e6,
        "p95_state_median_wall_ms": _p95(wall) / 1e6,
        "p95_raw_call_wall_ms": _p95(raw_wall) / 1e6,
        "sum_raw_wall_ns": sum(raw_wall),
        "sum_state_median_cpu_ns": sum(cpu),
        "mean_state_median_cpu_ms": sum(cpu) / len(cpu) / 1e6,
    }


def _comparisons(latencies):
    new = latencies[CANDIDATE]
    return {control: {
        "wall_ratio": new["sum_state_median_wall_ns"] / latencies[control]["sum_state_median_wall_ns"],
        "cpu_ratio": (new["sum_state_median_cpu_ns"] / latencies[control]["sum_state_median_cpu_ns"]
                      if latencies[control]["sum_state_median_cpu_ns"] else None),
        "p95_state_median_ratio": new["p95_state_median_wall_ms"] / latencies[control]["p95_state_median_wall_ms"],
        "p95_raw_call_ratio": new["p95_raw_call_wall_ms"] / latencies[control]["p95_raw_call_wall_ms"],
    } for control in CONTROLS}


def summarize(models, protocol, rows):
    """Require complete paired measurements before interpreting a timing gate."""
    expected = {(model.node, lanes) for model in models for lanes in protocol["lanes"]}
    actual = [(row["node"], row["lanes"]) for row in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("all registered node/R cells must appear exactly once")
    repeats = protocol["repeats"]
    valid = complete = exact = stable = pinned = True
    cells = []
    for row in rows:
        roster = [(panel["panel"], panel["seed"], tuple(queue))
                  for panel in protocol["queue_panels"][str(row["lanes"]) ]
                  for queue in panel["queues"]]
        measured = [(state["panel"], state["seed"], tuple(state["queues"])) for state in row["states"]]
        if measured != roster or row["queue_count"] != len(roster):
            raise ValueError("measured states differ from registered queue roster")
        for state in row["states"]:
            for clock in ("wall_ns", "cpu_ns"):
                for label in LABELS:
                    values = state[clock][label]
                    if len(values) != repeats or any(not isinstance(value, int) or
                            value < (1 if clock == "wall_ns" else 0) for value in values):
                        raise ValueError("every state needs all positive wall/nonnegative CPU measurements")
        for label in ("reference", *LABELS):
            metrics = row["correctness"][label]
            calls = len(roster) * (1 if label == "reference" else repeats)
            valid &= metrics["calls"] == calls and not metrics["failures"]
            if label != "reference":
                stable &= not metrics["deterministic_repeat_mismatch_count"]
        complete &= (not row["uncertified_reference_count"] and
                     row["correctness"]["reference"]["canonical_exact_count"] == len(roster))
        candidate = row["correctness"][CANDIDATE]
        exact &= (candidate["canonical_exact_count"] == repeats * len(roster)
                  and not candidate["score_mismatch_count"]
                  and not candidate["canonical_action_mismatch_count"])
        pinned &= all(row[key]["cpu_affinity"] == [protocol["cpu"]]
                      for key in ("runtime_before_warmup", "runtime_after_warmup", "runtime_end"))
        latencies = {label: _latency(row["states"], label) for label in LABELS}
        cells.append({
            "node": row["node"], "lanes": row["lanes"], "queue_count": len(roster),
            "latency": latencies, "comparison": _comparisons(latencies),
            "correctness": row["correctness"],
            "uncertified_reference_count": row["uncertified_reference_count"],
            "runtime_before_warmup": row["runtime_before_warmup"],
            "runtime_after_warmup": row["runtime_after_warmup"],
            "runtime_end": row["runtime_end"],
        })
    all_states = [state for row in rows for state in row["states"]]
    pooled = {label: _latency(all_states, label) for label in LABELS}
    comparison = _comparisons(pooled)
    gates = {control: {
        "pooled_at_least_five_percent_faster": comparison[control]["wall_ratio"] <= protocol["pooled_ratio_limit"],
        "no_node_lane_cell_regression": all(cell["comparison"][control]["wall_ratio"] <= protocol["cell_ratio_limit"] for cell in cells),
    } for control in CONTROLS}
    faster = all(all(gate.values()) for gate in gates.values())
    execution = bool(valid and stable and pinned)
    action = bool(execution and complete and exact)
    decision = ("INVALID_TIMING_EXECUTION" if not execution else
                "INCOMPLETE_REFERENCE" if not complete else
                "TIMING_ACTION_REVIEW_REQUIRED" if not exact else
                "INTEGER_CAPACITY_LOCAL_TIMING_ADVANCE" if faster else
                "LOCAL_TIMING_REVIEW_REQUIRED")
    return {
        "schema_version": 1, "experiment": protocol["experiment"],
        "registered_at_utc": protocol["registered_at_utc"], "decision": decision,
        "execution_checks_passed": execution, "certificate_checks_passed": bool(valid),
        "repeat_paths_stable": bool(stable), "cpu_affinity_preserved": bool(pinned),
        "references_complete": bool(complete), "action_gate_passed": action,
        "local_timing_gate_passed": bool(action and faster), "timing_gates_by_control": gates,
        "queue_count": len(all_states), "distinct_lane_queue_count": sum(
            len(panel["queues"]) for panels in protocol["queue_panels"].values() for panel in panels),
        "timed_selector_calls": len(all_states) * repeats * len(LABELS),
        "untimed_reference_calls": len(all_states),
        "warmup_calls": len(rows) * 2 * len(LABELS),
        "latency": pooled, "comparison": comparison, "cells": cells,
        "limitations": protocol["limitations"],
    }
