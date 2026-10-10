"""Paired decision-delay readout, retaining censored jobs and timing repeats."""
from __future__ import annotations

from collections import defaultdict
from math import isclose, isfinite

from algorithm.experiments.integer_capacity_trajectory_summary import (
    CONTROLS, CUTOFFS, UTILITY_METRICS, _average, _identity, _not_worse, common_cohort,
)


COUNTS = ("arrivals", "completed_jobs", "queued_jobs", "inflight_jobs", "unfinished_jobs")
FLOW_METRICS = (*UTILITY_METRICS, "completed_mean_flow_s", "completed_p90_flow_s", "completion_rate")
SCALAR_COSTS = ("calls", "failures", "canonical_exact_count", "score_mismatch_count",
                "canonical_action_mismatch_count", "cap_reached_count",
                "uncertified_cap_reached_count", "operational_elapsed_s")
NESTED_COSTS = ("search_work", "preprocessing", "capacity_propagation")


def _costs():
    return {**dict.fromkeys(SCALAR_COSTS, 0), **{key: {} for key in NESTED_COSTS},
            "total_decision_delay_s": 0.0, "executed_frame_count": 0}


def _add_costs(target, result, measured):
    metrics = result["selector_metrics"]
    for key in SCALAR_COSTS:
        target[key] += metrics[key]
    for key in NESTED_COSTS:
        for name, value in metrics[key].items():
            target[key][name] = target[key].get(name, 0) + value
    target["total_decision_delay_s"] += result["total_decision_delay_s"] if measured else 0.0
    target["executed_frame_count"] += result["executed_frame_count"] if measured else result["frame_count"]


def _aggregate(snapshots, sum_counts=False):
    return {
        **{key: _average(snapshot[key] for snapshot in snapshots) for key in FLOW_METRICS},
        **{key: sum(snapshot[key] for snapshot in snapshots) if sum_counts
           else _average(snapshot[key] for snapshot in snapshots) for key in COUNTS},
    }


def _delta(first, second):
    return {key: first[key] - second[key]
            if first[key] is not None and second[key] is not None else None
            for key in (*UTILITY_METRICS, "unfinished_jobs")}


def _overheads(zero, measured):
    overhead = {label: _delta(snapshot, zero[label]) for label, snapshot in measured.items()}
    return overhead, {control: _delta(overhead["integer_capacity"], overhead[control])
                      for control in CONTROLS}


def _comparisons(policies, tolerance):
    comparisons = {}
    for control in CONTROLS:
        first, second = policies["integer_capacity"], policies[control]
        checks = {key: _not_worse(first[key], second[key], tolerance)
                  for key in (*UTILITY_METRICS, "unfinished_jobs")}
        comparisons[control] = {"passed": all(checks.values()), "checks": checks,
                                "deltas": _delta(first, second)}
    return comparisons


def _execution_valid(result, expected_jobs, protocol, lanes, label, measured):
    frames, metrics = result["frames"], result["selector_metrics"]
    valid = (all(result["invariants"].values())
             and result["frame_count"] == len(frames) == metrics["calls"]
             and sorted((job["arrival"], job["coordinate"], job["index"])
                        for job in result["jobs"]) == expected_jobs
             and result["budget"] == (protocol["reference_budget"] if label == "full_reference"
                                     else protocol["budgets"][str(lanes)])
             and all(result[key] == protocol[key] for key in CUTOFFS))
    if measured:
        valid &= (result["decision_count"] == len(frames)
                  and result["executed_frame_count"] == sum(frame["dispatched"] for frame in frames)
                  and isclose(result["total_decision_delay_s"],
                              sum(frame["decision_delay_s"] for frame in frames),
                              rel_tol=protocol["float_relative_tolerance"], abs_tol=0))
        for frame in frames:
            valid &= (isfinite(frame["decision_delay_s"]) and frame["decision_delay_s"] >= 0
                      and isclose(frame["dispatch_time"], frame["start"] + frame["decision_delay_s"],
                                  rel_tol=protocol["float_relative_tolerance"], abs_tol=0)
                      and frame["dispatched"] == (frame["dispatch_time"] < protocol["observation_end"])
                      and tuple(frame["executed_service"]) == (tuple(frame["service"]) if frame["dispatched"]
                                                               else (0,) * len(frame["service"])))
    return valid


def summarize(models, protocol, rows):
    """Average timing repeats within each tape before comparing arrival seeds."""
    labels, zero_labels = protocol["labels"], protocol["zero_labels"]
    expected_cells = {(model.node, lanes) for model in models for lanes in protocol["lanes"]}
    actual_cells = [(row["node"], row["lanes"]) for row in rows]
    if len(actual_cells) != len(set(actual_cells)) or set(actual_cells) != expected_cells:
        raise ValueError("all registered node/R cells must appear exactly once")
    tapes = {tape["tape_id"]: tape for tape in protocol["tapes"]}
    expected_runs = {(load, seed) for load in protocol["loads"] for seed in protocol["arrival_seeds"]}
    execution = certificates = references = actions = identity = True
    paired, grouped = [], defaultdict(list)
    totals = {"zero": {label: _costs() for label in zero_labels},
              "measured": {label: _costs() for label in labels}}
    audit_totals = {"calls": 0, "failures": 0, "incomplete_count": 0,
                    "search_work": {}, "operational_elapsed_s": 0.0}
    seen_tapes = set()
    for row in rows:
        actual_runs = [(run["load"], run["seed"]) for run in row["runs"]]
        if len(actual_runs) != len(set(actual_runs)) or set(actual_runs) != expected_runs:
            raise ValueError("all registered load/seed pairs must appear exactly once per cell")
        for run in row["runs"]:
            tape = tapes[run["tape_id"]]
            if (run["tape_id"] in seen_tapes or any(tape[key] != value for key, value in (
                    ("node", row["node"]), ("lanes", row["lanes"]),
                    ("load", run["load"]), ("seed", run["seed"])))
                    or set(run["zero"]) != set(zero_labels)):
                raise ValueError("trajectory pairing differs from registered tape or zero policies")
            seen_tapes.add(run["tape_id"])
            repeats = [repeat["repeat"] for repeat in run["measured"]]
            if len(repeats) != len(set(repeats)) or set(repeats) != set(range(protocol["repeats"])):
                raise ValueError("each registered timing repeat must appear exactly once")
            for repeat in run["measured"]:
                if (repeat["order"] != protocol["timing_orders"][run["tape_id"]][repeat["repeat"]]
                        or set(repeat["policies"]) != set(labels)):
                    raise ValueError("timed policy order differs from registered order")
            expected_jobs = sorted(tuple(event) for event in tape["arrivals"])
            execution &= run["arrival_count"] == len(expected_jobs)
            audit = run["reference_audit"]
            references &= audit["incomplete_count"] == 0
            certificates &= audit["failures"] == 0
            for key in audit_totals:
                if key == "search_work":
                    for name, value in audit[key].items():
                        audit_totals[key][name] = audit_totals[key].get(name, 0) + value
                else:
                    audit_totals[key] += audit[key]
            results = [(f"zero:{label}", label, result, False) for label, result in run["zero"].items()]
            results.extend((f"measured:{repeat['repeat']}:{label}", label, result, True)
                           for repeat in run["measured"] for label, result in repeat["policies"].items())
            for _, label, result, measured in results:
                metrics = result["selector_metrics"]
                execution &= _execution_valid(result, expected_jobs, protocol, row["lanes"], label, measured)
                certificates &= metrics["failures"] == 0
                references &= result["reference_audit"]["incomplete_frame_count"] == 0
                if label == "integer_capacity":
                    actions &= (metrics["canonical_exact_count"] == metrics["calls"]
                                and metrics["reference_comparable_count"] == metrics["calls"]
                                and metrics["score_mismatch_count"] == 0
                                and metrics["canonical_action_mismatch_count"] == 0)
                _add_costs(totals["measured" if measured else "zero"][label], result, measured)
            match = _identity(run["zero"]["integer_capacity"], run["zero"]["full_reference"])
            identity &= match["passed"]
            pair = {"node": row["node"], "lanes": row["lanes"], "load": run["load"],
                    "seed": run["seed"], "tape_id": run["tape_id"], "repeat_count": len(repeats),
                    "zero_trajectory_identity": match, "snapshots": {}}
            for cutoff in CUTOFFS:
                cohort = common_cohort({key: result for key, _, result, _ in results}, protocol[cutoff])
                snapshots = {}
                for key, _, result, _ in results:
                    snapshot = result["snapshots"][cutoff]
                    snapshots[key] = {**{name: snapshot[name] for name in (*COUNTS, *FLOW_METRICS)
                                        if name in snapshot},
                                      "common_mean_flow_s": cohort["policies"][key]["mean_flow_s"],
                                      "common_p90_flow_s": cohort["policies"][key]["p90_flow_s"]}
                    execution &= (snapshot["arrivals"] == len(expected_jobs)
                                  and snapshot["unfinished_jobs"] == snapshot["queued_jobs"] + snapshot["inflight_jobs"]
                                  and snapshot["arrivals"] == snapshot["completed_jobs"] + snapshot["unfinished_jobs"])
                zero = {label: snapshots[f"zero:{label}"] for label in zero_labels}
                measured = {label: _aggregate([snapshots[f"measured:{repeat}:{label}"] for repeat in repeats])
                            for label in labels}
                overhead, difference = _overheads(zero, measured)
                paired_snapshot = {"common_cohort_count": cohort["count"], "zero": zero,
                                   "measured": measured, "overhead_vs_zero": overhead,
                                   "difference_of_differences": difference}
                pair["snapshots"][cutoff] = paired_snapshot
                grouped[(row["node"], row["lanes"], run["load"], cutoff)].append(paired_snapshot)
            paired.append(pair)
    if seen_tapes != set(tapes):
        raise ValueError("all registered arrival tapes must appear exactly once")
    tolerance = protocol["float_relative_tolerance"]
    groups = []
    for (node, lanes, load, cutoff), snapshots in sorted(grouped.items()):
        zero = {label: _aggregate([snapshot["zero"][label] for snapshot in snapshots], sum_counts=True)
                for label in zero_labels}
        measured = {label: _aggregate([snapshot["measured"][label] for snapshot in snapshots], sum_counts=True)
                    for label in labels}
        overhead, difference = _overheads(zero, measured)
        groups.append({"node": node, "lanes": lanes, "load": load, "cutoff": cutoff,
                       "seed_count": len(snapshots), "repeats_per_seed": protocol["repeats"],
                       "zero": zero, "measured": measured, "comparisons": _comparisons(measured, tolerance),
                       "zero_comparisons": _comparisons(zero, tolerance),
                       "overhead_vs_zero": overhead, "difference_of_differences": difference})
    utility = all(comparison["passed"] for group in groups for comparison in group["comparisons"].values())
    pooled = {label: _average(pair["snapshots"]["observation_end"]["measured"][label]["common_mean_flow_s"]
                             for pair in paired) for label in labels}
    pooled_zero = {label: _average(pair["snapshots"]["observation_end"]["zero"][label]["common_mean_flow_s"]
                                  for pair in paired) for label in zero_labels}
    improvements = {control: 1 - pooled["integer_capacity"] / pooled[control]
                    if pooled["integer_capacity"] is not None and pooled[control] is not None and pooled[control] > 0
                    else None for control in CONTROLS}
    integration = execution and certificates and references and actions and identity
    gain = integration and utility and all(value is not None and value >= protocol["minimum_pooled_flow_improvement"]
                                          for value in improvements.values())
    decision = ("INVALID_EXECUTION" if not execution or not certificates else
                "INCOMPLETE_REFERENCE" if not references else
                "INTEGER_CAPACITY_DECISION_DELAY_INTEGRATION_HOLD" if not integration else
                "INTEGER_CAPACITY_DECISION_DELAY_UTILITY_HOLD" if not utility else
                "INTEGER_CAPACITY_DECISION_DELAY_FLOW_GAIN" if gain else
                "INTEGER_CAPACITY_DECISION_DELAY_PRESERVED")
    for arm in totals.values():
        for costs in arm.values():
            costs["mean_selector_wall_s"] = costs["operational_elapsed_s"] / costs["calls"] if costs["calls"] else None
    return {
        "decision": decision, "execution_checks_passed": execution,
        "certificate_checks_passed": certificates, "references_complete": references,
        "integer_action_gate_passed": actions, "zero_trajectory_identity_passed": identity,
        "integration_gate_passed": integration, "utility_gate_passed": utility,
        "flow_gain_established": gain, "tape_count": len(paired),
        "trajectory_count": len(paired) * (len(zero_labels) + protocol["repeats"] * len(labels)),
        "totals": totals, "reference_audit_totals": audit_totals, "groups": groups, "paired_runs": paired,
        "pooled_common_mean_flow_s_at_T": pooled, "pooled_zero_common_mean_flow_s_at_T": pooled_zero,
        "pooled_flow_improvement": improvements,
        "pooled_overhead_vs_zero_common_mean_flow_s_at_T": {
            label: pooled[label] - pooled_zero[label]
            if pooled[label] is not None and pooled_zero[label] is not None else None for label in labels},
        "limitations": protocol["limitations"],
    }
