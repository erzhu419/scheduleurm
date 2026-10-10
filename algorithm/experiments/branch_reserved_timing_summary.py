"""Four-arm balanced timing with frozen old gates and a direct-predecessor gate."""
from __future__ import annotations

from collections import Counter, defaultdict
import math
from statistics import median

from algorithm.experiments.aligned_hardware_integer_capacity_development import COUNTERS, _combine_metrics
from algorithm.experiments.aligned_hardware_integer_capacity_timing_summary import _latency
from algorithm.experiments.branch_reserved_development_summary import audit_reserved_stage
from algorithm.experiments.certificate_routed_development_summary import audit_candidate, audit_result, _work as _old_work
from algorithm.experiments.integer_capacity_large_r_summary import _check_single_call
from algorithm.experiments.score_face_completion_confirmation_summary import ORACLE_COUNTERS, _pointer, _reference
from algorithm.experiments.strict_score_proof_timing_summary import _repeat_results, _same

LABELS = ("candidate", "routed", "probe", "long")
CONTROLS = LABELS[1:]
RUNTIME_KEYS = ("runtime_start", "runtime_before_warmup", "runtime_after_warmup", "runtime_end")


def _work(results, label):
    return _old_work(results, "candidate" if label == "routed" else label)


def _keys(items):
    return {(item["node"], item["lanes"], tuple(item["queue"])) for item in items}


def _identity(state):
    return tuple(state.get(key) for key in ("state_id", "panel", "cohort", "seed", "source_index", "source_state_id",
        "previous_integer_trigger", "score_unproved_regression", "previous_untimed_probe_trigger")), tuple(state["queue"])


def orders_valid(orders, count, repeats):
    if repeats != 8 or len(orders["state_orders"]) != repeats or len(orders["label_orders"]) != count:
        return False
    if any(sorted(order) != list(range(count)) for order in orders["state_orders"]):
        return False
    for blocks in orders["label_orders"]:
        if len(blocks) != repeats or any(sorted(order) != sorted(LABELS) for order in blocks):
            return False
        sequences = Counter(tuple(order) for order in blocks)
        positions = Counter((label, position) for order in blocks for position, label in enumerate(order))
        adjacency = Counter(pair for order in blocks for pair in zip(order, order[1:]))
        if len(sequences) != 4 or set(sequences.values()) != {2} or set(positions.values()) != {2} or len(adjacency) != 12 or set(adjacency.values()) != {2}:
            return False
    return True


def _durations_valid(state, repeats):
    return all(len(state[clock][label]) == repeats and all(type(value) is int and value >= (1 if clock == "wall_ns" else 0)
               for value in state[clock][label]) for label in LABELS for clock in ("wall_ns", "cpu_ns"))


def _audit(result, state, protocol, model, lanes, label, expected):
    if label in ("candidate", "routed"):
        initial = state["expected_reserved_initial_lp"] if label == "candidate" else state["expected_initial_lp"]
        metric, errors, quality = audit_candidate(result, {**state, "expected_initial_lp": initial}, protocol, model, lanes, expected)
        if not _same(result, state["expected_results"][label]):
            errors.append("actual selector differs from its independently executed frozen work-confirmation result")
        if label == "candidate":
            roots = len(result["initial_lp"]["thresholds"])
            for name, cap in (("initial_lp", protocol["initial_lp_budget"]), ("final_lp", protocol["lp_node_budget"] if result["resumed"] else protocol["initial_lp_budget"])):
                errors.extend(audit_reserved_stage(result[name], cap, roots))
        return metric, errors, quality
    return audit_result(result, {**state, "expected_probe_result": state["first_results"]["probe"]}, protocol, model, lanes, label, expected)


def _latencies(states, repeats):
    if not states or any(not _durations_valid(state, repeats) for state in states):
        return {"queue_count": len(states), "latency": None, "comparisons": None}
    latency = {label: _latency(states, label) for label in LABELS}
    for label in LABELS:
        ordered = sorted(median(state["wall_ns"][label]) for state in states)
        latency[label].update(p99_state_median_wall_ms=ordered[math.ceil(.99 * len(ordered)) - 1] / 1e6,
                              max_state_median_wall_ms=ordered[-1] / 1e6)
    comparisons = {label: {
        "wall_ratio": latency["candidate"]["sum_state_median_wall_ns"] / latency[label]["sum_state_median_wall_ns"],
        "cpu_ratio": latency["candidate"]["sum_state_median_cpu_ns"] / latency[label]["sum_state_median_cpu_ns"] if latency[label]["sum_state_median_cpu_ns"] else None,
        "p99_state_median_ratio": latency["candidate"]["p99_state_median_wall_ms"] / latency[label]["p99_state_median_wall_ms"],
        "max_paired_state_slowdown": max(median(state["wall_ns"]["candidate"]) / median(state["wall_ns"][label]) for state in states)} for label in CONTROLS}
    return {"queue_count": len(states), "latency": latency, "comparisons": comparisons}


def _group(states, repeats):
    return {**_latencies(states, repeats), "work": {label: _work([result for state in states for result in _repeat_results(state, label, repeats)], label)
            for label in LABELS}, "metrics": {label: _combine_metrics(state["metrics"][label] for state in states) for label in LABELS}}


def summarize(protocol, rows):
    repeats = protocol["repeats"]
    failures, quality_failures, entries, cells, repeat_failures = [], [], [], [], []
    warm_results, warm_metrics = ({label: [] for label in LABELS} for _ in range(2))
    models = {model["node"]: model for model in protocol["models"]}
    keys = {name: _keys(protocol[field]) for name, field in (("trigger", "historical_trigger_keys"), ("score", "score_unproved_keys"),
        ("untimed", "previous_untimed_probe_keys"), ("tail", "ordinary_tail_keys"))}
    if not keys["score"] <= keys["trigger"] or not keys["untimed"] <= keys["trigger"] or keys["score"] & keys["untimed"]:
        failures.append({"errors": ["fixed regression subgroups differ from disjoint trigger subsets"]})
    expected_cells = [(cell["node"], cell["lanes"]) for cell in protocol["cells"]]
    if [(row["node"], row["lanes"]) for row in rows] != expected_cells:
        failures.append({"errors": ["all registered timing cells must occur once in order"]})
    complete = pinned = stable = True
    for cell, row in zip(protocol["cells"], rows):
        model = models[cell["node"]]
        pointer = {"node": row["node"], "lanes": row["lanes"]}
        errors = []
        if row["registered_at_utc"] != protocol["registered_at_utc"] or row["physical_lanes"] != model["lanes"] or row["option_count"] != len(model["options"]):
            errors.append("cell registration or physical model differs from the frozen protocol")
        if list(map(_identity, row["states"])) != list(map(_identity, cell["states"])):
            errors.append("timing states differ from the complete frozen roster")
        if not orders_valid(cell["orders"], len(cell["states"]), repeats):
            errors.append("registered orders lack Williams position, adjacency or complete-state balance")
        if row["state_orders"] != cell["orders"]["state_orders"]:
            errors.append("actual per-sweep state order differs from registration")
        if (row["new_reference_calls"] or row["new_oracle_calls"] or row["screen_selector_calls"]
                or row["borrowed_reference_count"] != len(row["states"])):
            errors.append("timing must borrow references without new reference, oracle or screening calls")
        pinned &= all(row[key]["cpu_affinity"] == [protocol["cpu"]] for key in RUNTIME_KEYS)
        if any(not row[key]["gc_enabled"] for key in RUNTIME_KEYS):
            errors.append("normal automatic GC must remain enabled throughout the cell")
        if errors:
            failures.append({**pointer, "errors": errors})
        cell_entries = []
        references = []
        for index, state in enumerate(row["states"]):
            point, errors, expected = _pointer(row, state), [], None
            key = row["node"], row["lanes"], tuple(state["queue"])
            for flag, group in (("previous_integer_trigger", "trigger"), ("score_unproved_regression", "score"), ("previous_untimed_probe_trigger", "untimed")):
                if state[flag] != (key in keys[group]):
                    errors.append(flag + " differs from its frozen keys")
            if (state["cohort"] == "ordinary_tail_regression") != (key in keys["tail"]):
                errors.append("ordinary tail cohort differs from its frozen keys")
            try:
                _check_single_call(state["reference_metrics"], state["reference"], reference_complete=state["reference_complete"])
                expected, extra = _reference(state, protocol, len(model["options"]))
                errors.extend(extra)
            except ValueError as error:
                errors.append(str(error))
            if state["reference"]["lp_node_budget"] != protocol["reference_budget"] or state["reference_metrics"]["failures"] or state["reference_metrics"]["reference_comparable_count"] != int(state["reference_complete"]) or any(
                    state["reference_metrics"]["search_work"][counter] != state["reference"][counter] for counter in COUNTERS):
                errors.append("borrowed reference budget, feasibility or paid work is invalid")
            complete &= expected is not None
            references.append(expected)
            if not _durations_valid(state, repeats):
                errors.append("all eight registered wall and CPU observations are required")
            if state["blocks"] != [{"repeat": repeat, "order": list(cell["orders"]["label_orders"][index][repeat])} for repeat in range(repeats)]:
                errors.append("actual four-arm block orders differ from registration")
            for label in LABELS:
                actuals = _repeat_results(state, label, repeats)
                metrics = []
                stable &= not state["repeat_mismatches"][label]
                for item in state["repeat_mismatches"][label]:
                    repeat_failures.append({**point, "label": label, **item})
                for repeat, result in enumerate(actuals):
                    metric, extra, quality = _audit(result, state, protocol, model, row["lanes"], label, expected)
                    errors.extend(f"{label} repeat{repeat}: " + message for message in extra)
                    metrics.append(metric)
                    if not quality:
                        quality_failures.append({**point, "label": label, "repeat": repeat})
                if not _same(_combine_metrics(metrics), state["metrics"][label]):
                    errors.append(label + " timed metrics differ from actual repeat outputs")
            if errors:
                failures.append({**point, "errors": errors})
            entries.append((row, state))
            cell_entries.append(state)
        expected_warm = [(label, 0) for _ in range(protocol["warmups_per_label"]) for label in LABELS]
        if [(item["label"], item["state_index"]) for item in row["warmups"]] != expected_warm:
            failures.append({**pointer, "errors": ["all registered warmups must use the first timing state"]})
        for item in row["warmups"]:
            state = row["states"][item["state_index"]]
            label = item["label"]
            metric, errors, quality = _audit(item["result"], state, protocol, model, row["lanes"], label, references[item["state_index"]])
            if not _same(metric, item["metrics"]):
                errors.append("warmup metrics differ from actual output")
            if errors:
                failures.append({**_pointer(row, state), "phase": "warmup", "label": label, "errors": errors})
            if not quality:
                quality_failures.append({**_pointer(row, state), "phase": "warmup", "label": label})
            warm_results[label].append(item["result"])
            warm_metrics[label].append(metric)
        cells.append({**pointer, "timed": _group(cell_entries, repeats), "fresh_timed": _group([state for state in cell_entries if state["cohort"] == "fresh_confirmation"], repeats),
                      **{key: row[key] for key in RUNTIME_KEYS}})
    all_keys = {(row["node"], row["lanes"], tuple(state["queue"])) for row, state in entries}
    if not (keys["trigger"] | keys["tail"]) <= all_keys:
        failures.append({"errors": ["every fixed trigger and ordinary tail must occur in timing"]})
    subsets = {"fresh_timing": [state for _, state in entries if state["cohort"] == "fresh_confirmation"],
        "historical_trigger_timing": [state for _, state in entries if state["previous_integer_trigger"]],
        "score_unproved_timing": [state for _, state in entries if state["score_unproved_regression"]],
        "other_trigger_timing": [state for _, state in entries if state["previous_integer_trigger"] and not state["score_unproved_regression"]],
        "previous_untimed_probe_timing": [state for _, state in entries if state["previous_untimed_probe_trigger"]],
        "ordinary_tail_timing": [state for _, state in entries if state["cohort"] == "ordinary_tail_regression"],
        "historical_all_timing": [state for row, state in entries if (row["node"], row["lanes"], tuple(state["queue"])) in keys["trigger"] | keys["tail"]]}
    groups = {name: _group(states, repeats) for name, states in subsets.items()}
    def within(name, label, field, limit):
        group = groups[name]
        return bool(group["comparisons"] and group["comparisons"][label][field] <= limit)
    fresh_limits = {label: protocol["direct_predecessor_limit"] if label == "routed" else protocol["fresh_nonregression_limit"] for label in CONTROLS}
    fresh_gates = {label: {
        "pooled_no_material_regression": within("fresh_timing", label, "wall_ratio", fresh_limits[label]),
        "no_cell_material_regression": bool(subsets["fresh_timing"] and all(cell["fresh_timed"]["comparisons"] and cell["fresh_timed"]["comparisons"][label]["wall_ratio"] <= fresh_limits[label] for cell in cells if cell["fresh_timed"]["queue_count"])),
        "p99_no_material_regression": within("fresh_timing", label, "p99_state_median_ratio", fresh_limits[label])} for label in CONTROLS}
    regression_gates = {
        "score_unproved_probe_gain": within("score_unproved_timing", "probe", "wall_ratio", protocol["score_gain_limit"]),
        "score_unproved_probe_no_state_regression": within("score_unproved_timing", "probe", "max_paired_state_slowdown", protocol["state_ratio_limit"]),
        "score_unproved_long_no_regression": within("score_unproved_timing", "long", "wall_ratio", protocol["nonregression_limit"]),
        "other_trigger_probe_no_regression": within("other_trigger_timing", "probe", "wall_ratio", protocol["nonregression_limit"]),
        "other_trigger_probe_no_state_regression": within("other_trigger_timing", "probe", "max_paired_state_slowdown", protocol["state_ratio_limit"]),
        "trigger_long_no_regression": within("historical_trigger_timing", "long", "wall_ratio", protocol["trigger_long_ratio_limit"]),
        "trigger_long_p99_no_regression": within("historical_trigger_timing", "long", "p99_state_median_ratio", protocol["nonregression_limit"]),
        **{f"tail_no_state_regression_vs_{label}": within("ordinary_tail_timing", label, "max_paired_state_slowdown", protocol["tail_state_limit"]) for label in ("probe", "long")}}
    direct_gates = {field: within("historical_all_timing", "routed", field, protocol["direct_predecessor_limit"])
                    for field in ("wall_ratio", "p99_state_median_ratio", "max_paired_state_slowdown")}
    timed_calls, warm_calls = repeats * len(LABELS) * len(entries), sum(len(items) for items in warm_results.values())
    if (timed_calls != protocol["expected_timed_calls"] or warm_calls != protocol["expected_warmup_calls"]
            or timed_calls + warm_calls != protocol["expected_selector_calls"]):
        failures.append({"errors": ["executed timed/warmup method counts differ from registration"]})
    valid = not failures
    execution = valid and pinned and stable
    candidate_quality = valid and complete and not any(item["label"] == "candidate" for item in quality_failures)
    comparison_quality = valid and complete and not quality_failures
    timing_passed = all(all(gates.values()) for gates in fresh_gates.values()) and all(regression_gates.values()) and all(direct_gates.values())
    advance = execution and candidate_quality and comparison_quality and timing_passed
    decision = ("INVALID_EXECUTION" if not valid or not pinned else "INCOMPLETE_REFERENCE" if not complete else
        "BRANCH_RESERVED_TIMING_QUALITY_HOLD" if not comparison_quality else "INVALID_TIMING_EXECUTION" if not stable else
        "BRANCH_RESERVED_TIMING_ADVANCE" if advance else "BRANCH_RESERVED_TIMING_HOLD")
    warm_work = {label: _work(warm_results[label], label) for label in LABELS}
    measured = _group([state for _, state in entries], repeats)
    def views(states):
        wanted = {id(state) for state in states}
        return [{**_pointer(row, state), **_latencies([state], repeats), "wall_ns": state["wall_ns"], "cpu_ns": state["cpu_ns"],
                 "repeat_mismatches": state["repeat_mismatches"]} for row, state in entries if id(state) in wanted]
    oracles = [state["oracle"] for _, state in entries if state["oracle"] is not None]
    seed_groups = defaultdict(list)
    for state in subsets["fresh_timing"]:
        seed_groups[state["seed"]].append(state)
    return {"schema_version": 1, "experiment": protocol["experiment"], "registered_at_utc": protocol["registered_at_utc"],
        "decision": decision, "queue_count": len(entries), "timed_queue_count": len(entries),
        "timed_selector_calls": timed_calls, "warmup_calls": warm_calls, "executed_selector_calls": timed_calls + warm_calls,
        "executed_reference_calls": sum(row["new_reference_calls"] for row in rows),
        "executed_oracle_calls": sum(row["new_oracle_calls"] for row in rows),
        "screen_selector_calls": sum(row["screen_selector_calls"] for row in rows),
        "borrowed_reference_count": len(entries), "total_executed_lp_selector_calls": timed_calls + warm_calls,
        "certificate_checks_passed": valid, "effective_references_complete": complete, "repeat_paths_stable": stable,
        "cpu_affinity_preserved": pinned, "execution_checks_passed": execution, "candidate_quality_gate_passed": candidate_quality,
        "comparison_quality_gate_passed": comparison_quality, "confirmation_gate_passed": advance,
        "fresh_speedup_established": bool(execution and comparison_quality and all(within("fresh_timing", label, "wall_ratio", protocol["fresh_speedup_limit"]) for label in CONTROLS)),
        "timing_gates": {"fresh": fresh_gates, "regression": regression_gates, "direct_routed_historical": direct_gates},
        "accounting_failures": failures, "quality_failures": quality_failures, "repeat_failures": repeat_failures,
        "timed": measured, "warmup_work": warm_work, "warmup_metrics": {label: _combine_metrics(warm_metrics[label]) for label in LABELS},
        "executed_method_search_work": {counter: sum(measured["work"][label]["search_work"][counter] + warm_work[label]["search_work"][counter] for label in LABELS) for counter in COUNTERS},
        "executed_method_integer_states": sum(measured["work"][label]["shared_integer_states"] + warm_work[label]["shared_integer_states"] for label in LABELS),
        **groups, "fresh_by_seed": {str(seed): _group(states, repeats) for seed, states in seed_groups.items()},
        "historical_trigger_states": views(subsets["historical_trigger_timing"]), "ordinary_tail_states": views(subsets["ordinary_tail_timing"]),
        "previous_untimed_probe_states": views(subsets["previous_untimed_probe_timing"]),
        "fresh_worst_paired_states": {label: sorted(views(subsets["fresh_timing"]), key=lambda item: item["comparisons"][label]["wall_ratio"] if item["comparisons"] else float("inf"), reverse=True)[:10] for label in CONTROLS},
        "borrowed_reference_work": _combine_metrics(state["reference_metrics"] for _, state in entries),
        "borrowed_reference_oracle_work": {"calls": len(oracles), "work": {counter: sum(oracle["work"][counter] for oracle in oracles) for counter in ORACLE_COUNTERS}},
        "cells": cells, "prior_decisions": protocol["prior_decisions"], "prior_hold_retained": True, "limitations": protocol["limitations"]}
