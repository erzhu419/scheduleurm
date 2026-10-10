"""Paired native-selector replay: integration, utility and flow gain are separate."""
from __future__ import annotations

from collections import defaultdict
from math import isclose, isfinite

from algorithm.experiments.aligned_hardware_integer_capacity_timing import THREAD_ENV
from algorithm.experiments.integer_capacity_decision_delay_summary import (
    COUNTS, FLOW_METRICS, _aggregate, _delta,
)
from algorithm.experiments.integer_capacity_trajectory_replay import trajectory_snapshot
from algorithm.experiments.integer_capacity_trajectory_summary import (
    CUTOFFS, UTILITY_METRICS, _average, _identity, _not_worse, common_cohort,
)


LABELS = ("candidate", "direct")
ZERO_LABELS = ("full_reference", *LABELS)
SCALAR_COSTS = ("calls", "failures", "canonical_exact_count", "score_mismatch_count",
                "canonical_action_mismatch_count", "reference_incomplete_count")


def _add_work(target, work):
    """Sum the existing nested numeric work ledger without renaming its counters."""
    for key, value in work.items():
        if isinstance(value, dict):
            _add_work(target.setdefault(key, {}), value)
        else:
            target[key] = target.get(key, 0) + value


def _costs():
    return {**dict.fromkeys(SCALAR_COSTS, 0), "whole_work": {}, "delay_s": 0.0,
            "executed_frame_count": 0, "cancelled_decision_count": 0}


def _add_costs(target, result):
    metrics = result["selector_metrics"]
    for key in SCALAR_COSTS:
        target[key] += metrics[key]
    _add_work(target["whole_work"], metrics["whole_work"])
    target["delay_s"] += result["total_decision_delay_s"]
    target["executed_frame_count"] += result["executed_frame_count"]
    target["cancelled_decision_count"] += result["decision_count"] - result["executed_frame_count"]


def _runtime_valid(row, protocol):
    return all(
        runtime["cpu_affinity"] == [protocol["cpu"]] and runtime["gc_enabled"]
        and all(runtime["thread_environment"][name] == "1" for name in THREAD_ENV)
        and all(pool["num_threads"] == 1 for pool in runtime["threadpool_info"])
        for runtime in (row["runtime_start"], row["runtime_end"])
    )


def _execution_valid(result, expected_jobs, protocol, label, measured):
    """Each solve is charged once, including a final solve that cannot dispatch."""
    frames, metrics, tolerance = result["frames"], result["selector_metrics"], protocol["float_relative_tolerance"]
    close = lambda first, second: isclose(first, second, rel_tol=tolerance, abs_tol=0)
    valid = (
        result["policy"] == label and all(result["invariants"].values())
        and result["budget"] == protocol["reference_budget" if label == "full_reference" else "lp_node_budget"]
        and all(result[key] == protocol[key] for key in CUTOFFS)
        and result["frame_count"] == result["decision_count"] == len(frames) == metrics["calls"]
        and result["executed_frame_count"] == sum(frame["dispatched"] for frame in frames)
        and len(result["decision_diagnostics"]) == len(frames)
        and sorted((job["arrival"], job["coordinate"], job["index"])
                   for job in result["jobs"]) == expected_jobs
        and close(result["total_decision_delay_s"], sum(frame["decision_delay_s"] for frame in frames))
    )
    now = 0.0
    dispatched_times = set()
    for number, frame in enumerate(frames):
        delay, dispatch = frame["decision_delay_s"], frame["dispatch_time"]
        valid &= (
            isfinite(delay) and delay >= 0 and (measured or delay == 0)
            and isfinite(frame["duration"]) and frame["duration"] > 0
            and frame["start"] < protocol["observation_end"] and close(frame["start"], now)
            and close(dispatch, frame["start"] + delay)
            and frame["dispatched"] == (dispatch < protocol["observation_end"])
            and tuple(frame["executed_service"]) == (
                tuple(frame["service"]) if frame["dispatched"] else (0,) * len(frame["service"]))
        )
        if frame["dispatched"]:
            dispatched_times.add(dispatch)
            now = dispatch + frame["duration"]
        else:
            valid &= number == len(frames) - 1
            now = protocol["observation_end"]
    valid &= (close(result["last_frame_end"], now)
              and result["last_decision_end"] == (frames[-1]["dispatch_time"] if frames else None))
    for job in result["jobs"]:
        start, completion = job["start"], job["completion"]
        valid &= ((start is None and completion is None) or
                  (start in dispatched_times and completion is not None
                   and job["arrival"] <= start <= completion and isfinite(completion)))
    for cutoff in CUTOFFS:
        actual = trajectory_snapshot(result["jobs"], protocol[cutoff], len(frames[0]["queue"]) if frames else 3)
        snapshot = result["snapshots"][cutoff]
        valid &= all(snapshot[key] == actual[key] for key in (*COUNTS, "restricted_mean_flow_s",
                                                              "completed_mean_flow_s", "completed_p90_flow_s",
                                                              "completion_rate"))
    return bool(valid)


def _comparison(policies, tolerance):
    candidate, direct = policies["candidate"], policies["direct"]
    checks = {key: _not_worse(candidate[key], direct[key], tolerance)
              for key in (*UTILITY_METRICS, "unfinished_jobs")}
    return {"passed": all(checks.values()), "checks": checks, "deltas": _delta(candidate, direct)}


def _overheads(zero, measured):
    increments = {label: _delta(measured[label], zero[label]) for label in LABELS}
    return increments, _delta(increments["candidate"], increments["direct"])


def summarize(models, protocol, rows):
    """Use one 15-output cohort per tape; timing repeats are never arrival seeds."""
    if tuple(protocol["labels"]) != LABELS or tuple(protocol["zero_labels"]) != ZERO_LABELS or protocol["repeats"] != 6:
        raise ValueError("require candidate/direct, three zero policies and six timing repeats")
    expected_cells = {(model.node, lanes) for model in models for lanes in protocol["lanes"]}
    actual_cells = [(row["node"], row["lanes"]) for row in rows]
    if set(actual_cells) != expected_cells or actual_cells != [tuple(cell) for cell in protocol["execution_order"]]:
        raise ValueError("all registered node/R cells must appear exactly once in order")
    tapes = {tape["tape_id"]: tape for tape in protocol["tapes"]}
    if len(tapes) != len(protocol["tapes"]):
        raise ValueError("registered arrival tape IDs must be unique")
    expected_runs = {(load, seed) for load in protocol["loads"] for seed in protocol["arrival_seeds"]}
    execution = runtime = certificates = references = actions = identity = True
    paired, grouped, seen_tapes = [], defaultdict(list), set()
    totals = {"zero": {label: _costs() for label in ZERO_LABELS},
              "measured": {label: _costs() for label in LABELS}}
    audit_totals = {"calls": 0, "oracle_calls": 0, "failures": 0, "incomplete_count": 0, "work": {}}
    for row in rows:
        runtime &= _runtime_valid(row, protocol)
        actual_runs = [(run["load"], run["seed"]) for run in row["runs"]]
        if len(actual_runs) != len(set(actual_runs)) or set(actual_runs) != expected_runs:
            raise ValueError("all registered load/seed pairs must appear exactly once per cell")
        for run in row["runs"]:
            tape = tapes[run["tape_id"]]
            if (run["tape_id"] in seen_tapes or any(tape[key] != value for key, value in (
                    ("node", row["node"]), ("lanes", row["lanes"]),
                    ("load", run["load"]), ("seed", run["seed"])))
                    or set(run["zero"]) != set(ZERO_LABELS)):
                raise ValueError("trajectory pairing differs from its registered tape or zero policies")
            seen_tapes.add(run["tape_id"])
            repeats = [repeat["repeat"] for repeat in run["measured"]]
            if len(repeats) != len(set(repeats)) or set(repeats) != set(range(protocol["repeats"])):
                raise ValueError("each registered timing repeat must appear exactly once")
            for repeat in run["measured"]:
                if (repeat["order"] != protocol["timing_orders"][run["tape_id"]][repeat["repeat"]]
                        or set(repeat["policies"]) != set(LABELS)):
                    raise ValueError("timed policies differ from the registered adjacent order")
            expected_jobs = sorted(tuple(event) for event in tape["arrivals"])
            audit = run["reference_audit"]
            references &= audit["incomplete_count"] == 0
            certificates &= audit["failures"] == 0
            for key in audit_totals:
                if key == "work":
                    _add_work(audit_totals[key], audit[key])
                else:
                    audit_totals[key] += audit[key]
            results = [(f"zero:{label}", label, result, False) for label, result in run["zero"].items()]
            results.extend((f"measured:{repeat['repeat']}:{label}", label, result, True)
                           for repeat in run["measured"] for label, result in repeat["policies"].items())
            for _, label, result, measured in results:
                metrics = result["selector_metrics"]
                execution &= _execution_valid(result, expected_jobs, protocol, label, measured)
                certificates &= metrics["failures"] == 0 and all(not entry["errors"] for entry in result["decision_diagnostics"])
                references &= (metrics["reference_incomplete_count"] == 0
                               and result["reference_audit"]["incomplete_count"] == 0)
                if label in LABELS:
                    actions &= (metrics["canonical_exact_count"] == metrics["calls"]
                                and metrics["score_mismatch_count"] == 0
                                and metrics["canonical_action_mismatch_count"] == 0
                                and all(frame["canonical_exact"] for frame in result["frames"]))
                _add_costs(totals["measured" if measured else "zero"][label], result)
            matches = {label: _identity(run["zero"][label], run["zero"]["full_reference"]) for label in LABELS}
            identity &= all(match["passed"] for match in matches.values())
            pair = {"node": row["node"], "lanes": row["lanes"], "load": run["load"],
                    "seed": run["seed"], "tape_id": run["tape_id"], "repeat_count": len(repeats),
                    "zero_trajectory_identity": matches, "snapshots": {}}
            for cutoff in CUTOFFS:
                cohort = common_cohort({key: result for key, _, result, _ in results}, protocol[cutoff])
                snapshots = {key: {
                    **{name: result["snapshots"][cutoff][name] for name in (*COUNTS, *FLOW_METRICS)
                       if name in result["snapshots"][cutoff]},
                    "common_mean_flow_s": cohort["policies"][key]["mean_flow_s"],
                    "common_p90_flow_s": cohort["policies"][key]["p90_flow_s"],
                } for key, _, result, _ in results}
                zero = {label: snapshots[f"zero:{label}"] for label in ZERO_LABELS}
                measured = {label: _aggregate([snapshots[f"measured:{repeat}:{label}"] for repeat in repeats])
                            for label in LABELS}
                overhead, difference = _overheads(zero, measured)
                snapshot = {"common_cohort_count": cohort["count"], "common_job_ids": cohort["job_ids"],
                            "zero": zero, "measured": measured, "overhead_vs_zero": overhead,
                            "difference_of_differences": difference}
                pair["snapshots"][cutoff] = snapshot
                grouped[(row["node"], row["lanes"], run["load"], cutoff)].append(snapshot)
            paired.append(pair)
    if seen_tapes != set(tapes):
        raise ValueError("all registered arrival tapes must appear exactly once")
    groups = []
    for (node, lanes, load, cutoff), snapshots in sorted(grouped.items()):
        zero = {label: _aggregate([snapshot["zero"][label] for snapshot in snapshots], sum_counts=True)
                for label in ZERO_LABELS}
        measured = {label: _aggregate([snapshot["measured"][label] for snapshot in snapshots], sum_counts=True)
                    for label in LABELS}
        overhead, difference = _overheads(zero, measured)
        groups.append({"node": node, "lanes": lanes, "load": load, "cutoff": cutoff,
                       "seed_count": len(snapshots), "repeats_per_seed": protocol["repeats"],
                       "common_cohorts_nonempty": all(snapshot["common_cohort_count"] > 0 for snapshot in snapshots),
                       "zero": zero, "measured": measured,
                       "comparison": _comparison(measured, protocol["float_relative_tolerance"]),
                       "zero_comparison": _comparison(zero, protocol["float_relative_tolerance"]),
                       "overhead_vs_zero": overhead, "difference_of_differences": difference})
    utility = bool(groups) and all(group["comparison"]["passed"] and group["common_cohorts_nonempty"] for group in groups)
    pooled = {label: _average(pair["snapshots"]["observation_end"]["measured"][label]["common_mean_flow_s"]
                             for pair in paired) for label in LABELS}
    pooled_zero = {label: _average(pair["snapshots"]["observation_end"]["zero"][label]["common_mean_flow_s"]
                                  for pair in paired) for label in ZERO_LABELS}
    improvement = (1 - pooled["candidate"] / pooled["direct"]
                   if pooled["candidate"] is not None and pooled["direct"] is not None and pooled["direct"] > 0 else None)
    execution &= runtime
    integration = execution and certificates and references and actions and identity
    gain = (integration and utility and improvement is not None
            and improvement >= protocol["minimum_pooled_flow_improvement"])
    decision = ("INVALID_EXECUTION" if not execution or not certificates else
                "INCOMPLETE_REFERENCE" if not references else
                "SMALL_DIMENSION_EXACT_NATIVE_DECISION_DELAY_INTEGRATION_HOLD" if not integration else
                "SMALL_DIMENSION_EXACT_NATIVE_DECISION_DELAY_UTILITY_HOLD" if not utility else
                "SMALL_DIMENSION_EXACT_NATIVE_DECISION_DELAY_FLOW_GAIN" if gain else
                "SMALL_DIMENSION_EXACT_NATIVE_DECISION_DELAY_PRESERVED")
    return {
        "schema_version": protocol["schema_version"], "experiment": protocol["experiment"],
        "registered_at_utc": protocol["registered_at_utc"],
        "decision": decision, "execution_checks_passed": bool(execution), "runtime_checks_passed": bool(runtime),
        "certificate_checks_passed": bool(certificates), "references_complete": bool(references),
        "canonical_action_gate_passed": bool(actions), "zero_trajectory_identity_passed": bool(identity),
        "integration_gate_passed": bool(integration), "utility_gate_passed": bool(utility), "flow_gain_established": bool(gain),
        "tape_count": len(paired), "trajectory_count": len(paired) * (len(ZERO_LABELS) + protocol["repeats"] * len(LABELS)),
        "totals": totals, "reference_audit_totals": audit_totals, "groups": groups, "paired_runs": paired,
        "pooled_common_mean_flow_s_at_T": pooled, "pooled_zero_common_mean_flow_s_at_T": pooled_zero,
        "pooled_flow_improvement": improvement,
        "prior_decisions": protocol["prior_decisions"], "prior_hold_retained": True, "limitations": protocol["limitations"],
    }
