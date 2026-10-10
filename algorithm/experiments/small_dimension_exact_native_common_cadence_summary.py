"""Whole-selector savings on a common clock; flow identity is not a flow gain."""
from __future__ import annotations

from collections import defaultdict
from math import isclose, isfinite
from statistics import mean

from algorithm.experiments.integer_capacity_decision_delay_summary import COUNTS, FLOW_METRICS
from algorithm.experiments.integer_capacity_trajectory_summary import CUTOFFS, common_cohort
from algorithm.experiments.small_dimension_exact_native_decision_delay_summary import (
    _add_costs, _add_work, _costs, _execution_valid, _runtime_valid,
)


LABELS = ("candidate", "direct")
TIMING_FIELDS = ("selector_wall_s", "selector_cpu_s", "window_padding_s", "window_overrun_s")
IDENTITY_FIELDS = ("start", "dispatch_time", "queue", "option_indices", "service", "duration",
                   "executed_service", "dispatched")


def _trajectory_identity(candidate, reference):
    """Compare actual execution, including final cancelled decisions and both cutoffs."""
    if len(candidate["frames"]) != len(reference["frames"]):
        return {"passed": False, "first_mismatch": {"field": "frame_count"}}
    for index, (first, second) in enumerate(zip(candidate["frames"], reference["frames"], strict=True)):
        for key in IDENTITY_FIELDS:
            a, b = first[key], second[key]
            if key in ("queue", "option_indices", "service", "executed_service"):
                a, b = tuple(a), tuple(b)
            if a != b:
                return {"passed": False, "first_mismatch": {"frame": index, "field": key,
                                                             "candidate": a, "reference": b}}
    for key in ("jobs", "snapshots"):
        if candidate[key] != reference[key]:
            return {"passed": False, "first_mismatch": {"field": key}}
    return {"passed": True, "first_mismatch": None}


def _causal_fifo_valid(result):
    """Reconstruct released queues and the served FIFO prefix at every solve start."""
    jobs = sorted(result["jobs"], key=lambda job: (job["arrival"], job["coordinate"], job["index"]))
    departed = set()
    for frame in result["frames"]:
        waiting = [[] for _ in frame["queue"]]
        for job in jobs:
            key = job["coordinate"], job["index"]
            if job["arrival"] <= frame["start"] and key not in departed:
                waiting[job["coordinate"]].append(job)
        if tuple(map(len, waiting)) != tuple(frame["queue"]):
            return False
        assigned = [job for job in jobs if job["start"] == frame["dispatch_time"]]
        expected = [job for coordinate, count in enumerate(frame["executed_service"])
                    for job in waiting[coordinate][:count]]
        actual_ids = {(job["coordinate"], job["index"]) for job in assigned}
        expected_ids = {(job["coordinate"], job["index"]) for job in expected}
        if (actual_ids != expected_ids
                or any(job["completion"] > frame["dispatch_time"] + frame["duration"] for job in assigned)):
            return False
        departed.update(actual_ids)
    return True


def _clock_valid(result, protocol, measured):
    window, tolerance = protocol["computation_window_s"], protocol["float_relative_tolerance"]
    close = lambda a, b: isclose(a, b, rel_tol=tolerance, abs_tol=0)
    valid = result["computation_window_s"] == window
    sums = dict.fromkeys(TIMING_FIELDS, 0.0)
    overruns = 0
    for frame in result["frames"]:
        wall, cpu, padding, overrun = (frame[name] for name in TIMING_FIELDS)
        valid &= (all(isfinite(value) and value >= 0 for value in (wall, cpu, padding, overrun))
                  and close(padding, max(window - wall, 0.0))
                  and close(overrun, max(wall - window, 0.0))
                  and close(frame["decision_delay_s"], max(window, wall) if measured else window))
        overruns += wall > window
        for key in sums:
            sums[key] += frame[key]
    valid &= (all(close(result["total_" + name], value) for name, value in sums.items())
              and result["window_overrun_count"] == overruns
              and close(result["selector_metrics"]["operational_elapsed_s"], sums["selector_wall_s"])
              and close(result["selector_metrics"]["operational_cpu_s"], sums["selector_cpu_s"]))
    return bool(valid)


def _measured_costs():
    return {**_costs(), **dict.fromkeys(TIMING_FIELDS, 0.0), "window_overrun_count": 0}


def _add_measured_costs(target, result):
    _add_costs(target, result)
    for key in TIMING_FIELDS:
        target[key] += result["total_" + key]
    target["window_overrun_count"] += result["window_overrun_count"]


def _ratio(candidate, direct):
    return candidate / direct if isfinite(candidate) and isfinite(direct) and direct > 0 else None


def _timing_comparison(costs):
    return {name + "_ratio": _ratio(costs["candidate"][name], costs["direct"][name])
            for name in ("wall_s", "cpu_s")}


def _p99(values):
    values = sorted(values)
    if not values:
        return None
    position = .99 * (len(values) - 1)
    left = int(position)
    right = min(left + 1, len(values) - 1)
    return values[left] + (position - left) * (values[right] - values[left])


def summarize(models, protocol, rows):
    """Average six real whole trajectories within each tape before pooling costs."""
    if tuple(protocol["labels"]) != LABELS or protocol["repeats"] != 6:
        raise ValueError("require candidate/direct and six paired timing repeats")
    expected_cells = {(model.node, lanes) for model in models for lanes in protocol["lanes"]}
    actual_cells = [(row["node"], row["lanes"]) for row in rows]
    if (len(actual_cells) != len(set(actual_cells)) or set(actual_cells) != expected_cells
            or actual_cells != [tuple(cell) for cell in protocol["execution_order"]]):
        raise ValueError("all registered node/R cells must appear exactly once in order")
    tapes = {tape["tape_id"]: tape for tape in protocol["tapes"]}
    if len(tapes) != len(protocol["tapes"]):
        raise ValueError("registered arrival tape IDs must be unique")
    expected_runs = {(load, seed) for load in protocol["loads"] for seed in protocol["arrival_seeds"]}
    execution = runtime = certificates = references = actions = clock = causal = no_overrun = identity = True
    paired, grouped, seen_tapes = [], defaultdict(list), set()
    totals = {"reference": _measured_costs(), "measured": {label: _measured_costs() for label in LABELS}}
    audit_totals = {"calls": 0, "oracle_calls": 0, "failures": 0, "incomplete_count": 0, "work": {}}
    actual_wall = {label: [] for label in LABELS}
    for row in rows:
        runtime &= _runtime_valid(row, protocol)
        actual_runs = [(run["load"], run["seed"]) for run in row["runs"]]
        if len(actual_runs) != len(set(actual_runs)) or set(actual_runs) != expected_runs:
            raise ValueError("all registered load/seed pairs must appear exactly once per cell")
        for run in row["runs"]:
            tape = tapes[run["tape_id"]]
            if run["tape_id"] in seen_tapes or any(tape[key] != value for key, value in (
                    ("node", row["node"]), ("lanes", row["lanes"]), ("load", run["load"]), ("seed", run["seed"]))):
                raise ValueError("trajectory pairing differs from its registered tape")
            seen_tapes.add(run["tape_id"])
            repeats = [repeat["repeat"] for repeat in run["measured"]]
            if len(repeats) != len(set(repeats)) or set(repeats) != set(range(protocol["repeats"])):
                raise ValueError("each registered timing repeat must appear exactly once")
            orders = [tuple(repeat["order"]) for repeat in run["measured"]]
            if orders.count(LABELS) != 3 or orders.count(tuple(reversed(LABELS))) != 3:
                raise ValueError("both complete whole-call orders must occur exactly three times")
            for repeat in run["measured"]:
                if (repeat["order"] != protocol["timing_orders"][run["tape_id"]][repeat["repeat"]]
                        or set(repeat["policies"]) != set(LABELS)):
                    raise ValueError("timed policies differ from their registered adjacent order")
            expected_jobs = sorted(tuple(event) for event in tape["arrivals"])
            audit = run["reference_audit"]
            references &= audit["incomplete_count"] == 0
            certificates &= audit["failures"] == 0
            for key in audit_totals:
                if key == "work":
                    _add_work(audit_totals[key], audit[key])
                else:
                    audit_totals[key] += audit[key]
            results = [("full_reference", run["reference"], False)]
            results.extend((label, repeat["policies"][label], True)
                           for repeat in run["measured"] for label in LABELS)
            for label, result, measured in results:
                metrics = result["selector_metrics"]
                execution &= _execution_valid(result, expected_jobs, protocol, label, True)
                clock &= _clock_valid(result, protocol, measured)
                causal &= _causal_fifo_valid(result)
                certificates &= metrics["failures"] == 0 and all(not entry["errors"] for entry in result["decision_diagnostics"])
                references &= metrics["reference_incomplete_count"] == 0 and result["reference_audit"]["incomplete_count"] == 0
                if measured:
                    actions &= (metrics["canonical_exact_count"] == metrics["calls"]
                                and metrics["score_mismatch_count"] == metrics["canonical_action_mismatch_count"] == 0
                                and all(frame["canonical_exact"] for frame in result["frames"]))
                    no_overrun &= all(frame["selector_wall_s"] <= protocol["computation_window_s"] for frame in result["frames"])
                    _add_measured_costs(totals["measured"][label], result)
                    actual_wall[label].extend(frame["selector_wall_s"] for frame in result["frames"])
                else:
                    _add_measured_costs(totals["reference"], result)
            matches = {f"{repeat['repeat']}:{label}": _trajectory_identity(repeat["policies"][label], run["reference"])
                       for repeat in run["measured"] for label in LABELS}
            identity &= all(match["passed"] for match in matches.values())
            costs = {label: {name + "_s": mean(repeat["policies"][label]["total_selector_" + name + "_s"]
                                              for repeat in run["measured"])
                             for name in ("wall", "cpu")} for label in LABELS}
            cohort_policies = {"reference": run["reference"],
                               **{f"{repeat['repeat']}:{label}": repeat["policies"][label]
                                  for repeat in run["measured"] for label in LABELS}}
            pair = {"node": row["node"], "lanes": row["lanes"], "load": run["load"], "seed": run["seed"],
                    "tape_id": run["tape_id"], "repeat_count": len(repeats), "trajectory_identity": matches,
                    "mean_actual_costs": costs, "timing_comparison": _timing_comparison(costs),
                    "common_completed_cohorts": {cutoff: common_cohort(cohort_policies, protocol[cutoff])
                                                 for cutoff in CUTOFFS},
                    "reference_snapshots": {cutoff: {key: run["reference"]["snapshots"][cutoff][key]
                                                     for key in (*COUNTS, *FLOW_METRICS)
                                                     if key in run["reference"]["snapshots"][cutoff]}
                                            for cutoff in CUTOFFS}}
            paired.append(pair)
            grouped[(row["node"], row["lanes"], run["load"])].append(pair)
    if seen_tapes != set(tapes):
        raise ValueError("all registered arrival tapes must appear exactly once")
    groups, gates = [], []
    for (node, lanes, load), pairs in sorted(grouped.items()):
        costs = {label: {name: sum(pair["mean_actual_costs"][label][name] for pair in pairs)
                         for name in ("wall_s", "cpu_s")} for label in LABELS}
        comparison = _timing_comparison(costs)
        passed = comparison["wall_s_ratio"] is not None and comparison["wall_s_ratio"] <= protocol["maximum_group_wall_ratio"]
        group = {"node": node, "lanes": lanes, "load": load, "seed_count": len(pairs),
                 "repeats_per_seed": protocol["repeats"], "mean_actual_costs_sum": costs,
                 "timing_comparison": comparison, "wall_gate_passed": passed}
        groups.append(group)
        gates.append({"name": f"group:{node}:R{lanes}:load{load}", "ratio": comparison["wall_s_ratio"],
                      "maximum": protocol["maximum_group_wall_ratio"], "passed": passed})
    pooled = {label: {name: sum(pair["mean_actual_costs"][label][name] for pair in paired)
                      for name in ("wall_s", "cpu_s")} for label in LABELS}
    comparison = _timing_comparison(pooled)
    gates.append({"name": "pooled", "ratio": comparison["wall_s_ratio"],
                  "maximum": protocol["maximum_pooled_wall_ratio"],
                  "passed": comparison["wall_s_ratio"] is not None
                            and comparison["wall_s_ratio"] <= protocol["maximum_pooled_wall_ratio"]})
    speed = all(gate["passed"] for gate in gates)
    execution &= runtime and clock and causal
    integration = execution and certificates and references and actions
    common_clock = integration and no_overrun and identity
    advance = common_clock and speed
    decision = ("INVALID_EXECUTION" if not execution or not certificates else
                "INCOMPLETE_REFERENCE" if not references else
                "SMALL_DIMENSION_EXACT_NATIVE_COMMON_CADENCE_INTEGRATION_HOLD" if not integration else
                "SMALL_DIMENSION_EXACT_NATIVE_COMMON_CADENCE_CLOCK_HOLD" if not common_clock else
                "SMALL_DIMENSION_EXACT_NATIVE_COMMON_CADENCE_TIMING_HOLD" if not speed else
                "SMALL_DIMENSION_EXACT_NATIVE_COMMON_CADENCE_TIMING_ADVANCE")
    p99 = {label: _p99(actual_wall[label]) for label in LABELS}
    return {
        "schema_version": protocol["schema_version"], "experiment": protocol["experiment"],
        "registered_at_utc": protocol["registered_at_utc"], "decision": decision,
        "execution_checks_passed": bool(execution), "runtime_checks_passed": bool(runtime),
        "clock_accounting_passed": bool(clock), "causal_fifo_checks_passed": bool(causal),
        "certificate_checks_passed": bool(certificates), "references_complete": bool(references),
        "canonical_action_gate_passed": bool(actions), "integration_gate_passed": bool(integration),
        "no_overrun_gate_passed": bool(no_overrun), "trajectory_identity_passed": bool(identity),
        "common_clock_gate_passed": bool(common_clock), "timing_gate_passed": bool(speed),
        "common_cadence_gate_passed": bool(advance), "flow_gain_established": False,
        "tape_count": len(paired), "trajectory_count": len(paired) * (1 + protocol["repeats"] * len(LABELS)),
        "computation_window_s": protocol["computation_window_s"], "totals": totals,
        "reference_audit_totals": audit_totals, "groups": groups, "paired_runs": paired,
        "pooled_mean_actual_costs_sum": pooled, "pooled_timing_comparison": comparison,
        "actual_call_wall_p99_s": p99, "actual_call_wall_p99_ratio": _ratio(p99["candidate"], p99["direct"])
            if all(value is not None for value in p99.values()) else None,
        "gates": gates, "gate_count": len(gates), "failed_gate_count": sum(not gate["passed"] for gate in gates),
        "prior_decisions": protocol["prior_decisions"], "prior_hold_retained": True,
        "limitations": protocol["limitations"],
    }
