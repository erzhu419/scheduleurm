"""Descriptive same-implementation timing windows; previous confirmation stays HOLD."""
from __future__ import annotations

from collections import Counter
from statistics import median


def _pointer(item, label=False):
    keys = ("stage", "kind", "window", "repeat", "label") if label else (
        "stage", "kind", "window", "repeat")
    return tuple(item[key] for key in keys)


def _expected_blocks(protocol, index):
    blocks = []
    for stage, kind, count, design in (
        ("aa", "warmup", protocol["warmups"], protocol["aa_warmup_orders"]),
        ("aa", "timed", protocol["repeats"], protocol["aa_orders"]),
        ("phase", "warmup", protocol["phase_warmups"], protocol["phase_warmup_orders"]),
        ("phase", "timed", protocol["phase_repeats"], protocol["phase_orders"]),
    ):
        windows = range(protocol["windows"]) if (stage, kind) == ("aa", "timed") else (-1,)
        for window in windows:
            orders = design["label_orders"][index]
            if window != -1:
                orders = orders[window]
            for repeat in range(count):
                blocks.append({"stage": stage, "kind": kind, "window": window,
                    "repeat": repeat, "order": orders[repeat]})
    return blocks


def _validate(protocol, record):
    errors = []
    states = protocol["states"]
    if [row["name"] for row in record["states"]] != [state["name"] for state in states]:
        errors.append("state roster differs from registration")
        return errors
    if not protocol["prior_quality_passed"]:
        errors.append("prior quality checks did not pass")
    if any(record[key]["cpu_affinity"] != [protocol["cpu"]]
           for key in ("runtime_start", "runtime_end")):
        errors.append("CPU affinity differs from registration")
    for index, (state, row) in enumerate(zip(states, record["states"], strict=True)):
        prefix = state["name"] + ": "
        expected = _expected_blocks(protocol, index)
        if row["blocks"] != expected:
            errors.append(prefix + "missing, duplicate or reordered registered blocks")
        expected_ids = [(*_pointer(block), label) for block in expected for label in block["order"]]
        actual_ids = [_pointer(call, label=True) for call in row["calls"]]
        if Counter(actual_ids) != Counter(expected_ids):
            errors.append(prefix + "missing, duplicate or unregistered calls")
        if actual_ids != [(*_pointer(block), label) for block in row["blocks"] for label in block["order"]]:
            errors.append(prefix + "call order differs from recorded blocks")
        if row["mismatches"] or any(not call["matches_frozen"] for call in row["calls"]):
            errors.append(prefix + "a selector output differs from its frozen reference")
        if not state["effective_reference_complete"]:
            errors.append(prefix + "independent reference is incomplete")
        for label in protocol["phase_labels"]:
            result, metric = state["expected_results"][label], state["screen_metrics"][label]
            if (row["first_results"].get(label) != result
                    or not all(result["selection"][key] for key in ("exact", "score_exact", "canonical_exact"))
                    or metric["failures"] or metric["score_mismatch_count"] or metric["canonical_action_mismatch_count"]):
                errors.append(prefix + label + " frozen result or quality is invalid")
        for call in row["calls"]:
            if any(type(call[key]) is not int or call[key] < minimum
                   for key, minimum in (("wall_ns", 1), ("cpu_ns", 0))):
                errors.append(prefix + "invalid selector duration")
                break
    return errors


def _medians(calls):
    result = {clock: median(call[clock] for call in calls) for clock in ("wall_ns", "cpu_ns")}
    return {"calls": len(calls), **result, "wall_ms": result["wall_ns"] / 1e6,
            "cpu_ms": result["cpu_ns"] / 1e6}


def _ratios(arms, clock):
    return {label: arms["a"][clock] / arms[label][clock] if arms[label][clock] else None
            for label in ("b", "c")}


def _window(calls, labels, window):
    arms = {label: _medians([call for call in calls if call["label"] == label]) for label in labels}
    wall, cpu = _ratios(arms, "wall_ns"), _ratios(arms, "cpu_ns")
    slowdowns = {label: ratio > 1.05 for label, ratio in wall.items()}
    values = [arm["wall_ns"] for arm in arms.values()]
    return {"window": window, "arms": arms, "a_over_control_wall_ratio": wall,
        "a_over_control_cpu_ratio": cpu, "apparent_slowdown_over_5pct": slowdowns,
        "any_apparent_slowdown_over_5pct": any(slowdowns.values()),
        "symmetric_wall_max_min_ratio": max(values) / min(values)}


def _phases(calls):
    result = _medians(calls)
    names = sorted({name for call in calls for name in call["phases"]})
    result["phases"] = {name: {
        **{clock: median(call["phases"].get(name, {}).get(clock, 0) for call in calls)
           for clock in ("wall_ns", "cpu_ns")},
        "total_calls": sum(call["phases"].get(name, {}).get("calls", 0) for call in calls)}
        for name in names}
    for clock in ("wall_ns", "cpu_ns"):
        for prefix in ("primary_phase_", "residual_"):
            result[prefix + clock] = median(call[prefix + clock] for call in calls)
        result["initial_lp_less_linprog_" + clock] = median(
            call["phases"].get("initial_lp", {}).get(clock, 0)
            - call["phases"].get("initial_lp_linprog", {}).get(clock, 0) for call in calls)
    result["gc_events"] = sum(len(call["gc_events"]) for call in calls)
    result["gc_pause_wall_ns"] = sum(event["wall_ns"] for call in calls for event in call["gc_events"])
    result["gc_pause_fraction_of_whole"] = result["gc_pause_wall_ns"] / sum(call["wall_ns"] for call in calls)
    return result


def summarize(protocol, record):
    errors = _validate(protocol, record)
    calls = [call for row in record["states"] for call in row["calls"]]
    counts = {stage: {kind: sum(call["stage"] == stage and call["kind"] == kind for call in calls)
                      for kind in ("warmup", "timed")} for stage in ("aa", "phase")}
    result = {"decision": "INVALID_EXECUTION" if errors else "AA_DIAGNOSIS_COMPLETE",
        "execution_checks_passed": not errors, "execution_failures": errors,
        "prior_decision": protocol["prior_decision"], "prior_hold_retained": True,
        "executed_selector_calls": len(calls), "call_counts": counts,
        "selector_calls_by_implementation": {
            label: sum(("candidate" if call["stage"] == "aa" else call["label"]) == label for call in calls)
            for label in protocol["phase_labels"]},
        "executed_reference_calls": 0, "executed_oracle_calls": 0,
        "borrowed_reference_count": len(protocol["states"]),
        "states": [], "work": record["work"], "elapsed_s": record["elapsed_s"],
        "limitations": protocol["limitations"]}
    if errors:
        return result
    for state, row in zip(protocol["states"], record["states"], strict=True):
        aa = [call for call in row["calls"] if call["stage"] == "aa" and call["kind"] == "timed"]
        windows = [_window([call for call in aa if call["window"] == window], protocol["labels"], window)
                   for window in range(protocol["windows"])]
        result["states"].append({key: state[key] for key in ("name", "node", "lanes", "queue")} | {
            "aa": {"aggregate_arms": {label: _medians([call for call in aa if call["label"] == label])
                                       for label in protocol["labels"]}, "windows": windows,
                "windows_with_apparent_slowdown": sum(window["any_apparent_slowdown_over_5pct"] for window in windows),
                "windows_with_symmetric_spread_over_5pct": sum(window["symmetric_wall_max_min_ratio"] > 1.05 for window in windows)},
            "phase": {label: _phases([call for call in row["calls"]
                if call["stage"] == "phase" and call["kind"] == "timed" and call["label"] == label])
                for label in protocol["phase_labels"]}})
    windows = [window for state in result["states"] for window in state["aa"]["windows"]]
    result["aa_readout"] = {
        "state_windows": len(windows),
        "windows_with_apparent_slowdown": sum(window["any_apparent_slowdown_over_5pct"] for window in windows),
        "apparent_slowdown_counts_by_control": {label: sum(window["apparent_slowdown_over_5pct"][label] for window in windows)
                                               for label in ("b", "c")},
        "windows_with_symmetric_spread_over_5pct": sum(window["symmetric_wall_max_min_ratio"] > 1.05 for window in windows),
        "max_a_over_control_wall_ratio": max(ratio for window in windows for ratio in window["a_over_control_wall_ratio"].values()),
        "max_symmetric_wall_max_min_ratio": max(window["symmetric_wall_max_min_ratio"] for window in windows),
        "interpretation": "Fixed label a versus b/c uses the prior >5% direction. Symmetric spread is a separate descriptive measure. State-windows share states and one process; counts are not independent error-rate estimates. Instrumented phases are not confirmation latency."}
    return result
