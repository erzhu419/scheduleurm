"""Service-clock performance of joint lane refill; runtime is auxiliary only."""
from __future__ import annotations

from collections import defaultdict
from math import isclose, isfinite

from algorithm.experiments.integer_capacity_trajectory_replay import trajectory_snapshot
from algorithm.experiments.integer_capacity_trajectory_summary import (
    CUTOFFS, UTILITY_METRICS, _average, _not_worse, common_cohort,
)
from algorithm.experiments.small_dimension_exact_native_decision_delay_summary import (
    SCALAR_COSTS, _add_work,
)


LABELS = ("sync_joint", "refill_joint", "refill_single")
RULES = dict(zip(LABELS, ("synchronous", "refill", "single"), strict=True))
COUNTS = ("arrivals", "completed_jobs", "queued_jobs", "inflight_jobs", "unfinished_jobs")
EXPOSURES = ("occupied_lane_time_s", "barrier_idle_lane_time_s", "scheduled_idle_lane_time_s",
             "waiting_area_s", "total_selector_wall_s")


def _add_whole_work(target, work):
    maximum = max(target.get("max_shared_integer_states", 0), work.get("max_shared_integer_states", 0))
    _add_work(target, work)
    target["max_shared_integer_states"] = maximum


def _trajectory_checks(model, result, tape, protocol, label, lanes):
    """Reconstruct each admission and lane reservation, including simultaneous calls."""
    frames, jobs, events = result["frames"], result["jobs"], result["lane_events"]
    metrics, diagnostics = result["selector_metrics"], result["decision_diagnostics"]
    tolerance, end = protocol["float_relative_tolerance"], protocol["observation_end"]
    close = lambda a, b: isclose(a, b, rel_tol=tolerance, abs_tol=0)
    expected_jobs = sorted(tuple(event) for event in tape["arrivals"])
    execution = (result["policy"] == label and result["dispatch_rule"] == RULES[label]
                 and result["budget"] == protocol["lp_node_budget"]
                 and all(result[key] == protocol[key] for key in CUTOFFS)
                 and result["frame_count"] == result["decision_count"]
                 == result["executed_frame_count"] == len(frames) == metrics["calls"]
                 == len(diagnostics)
                 and metrics["whole_work"]["calls"] == metrics["calls"]
                 and all(result["invariants"].values())
                 and sorted((job["arrival"], job["coordinate"], job["index"])
                            for job in jobs) == expected_jobs
                 and len(events) == sum(frame["lanes_called"] for frame in frames))
    clock = result["total_decision_delay_s"] == 0
    fifo = lane_accounting = True
    ordered = sorted(jobs, key=lambda job: (job["arrival"], job["coordinate"], job["index"]))
    admitted, available, event_cursor = set(), [0.0] * lanes, 0
    exposures = dict.fromkeys(EXPOSURES, 0.0)
    exposure = lambda start, finish: max(0.0, min(finish, end) - start)
    for number, frame in enumerate(frames):
        start, called = frame["start"], frame["lanes_called"]
        indices, lane_ids = tuple(frame["option_indices"]), tuple(frame["lane_ids"])
        diagnostic = diagnostics[number] if number < len(diagnostics) else {}
        clock &= (isfinite(start) and 0 <= start < end
                  and frame["decision_delay_s"] == 0 and frame["dispatch_time"] == start
                  and frame["dispatched"] is True and close(start, min(available)))
        free = tuple(lane for lane, finish in enumerate(available) if finish == start)
        expected_lanes = free[:1] if label == "refill_single" else free
        lane_accounting &= (type(called) is int and 1 <= called <= lanes
                            and len(lane_ids) == len(indices) == called
                            and lane_ids == expected_lanes
                            and (label != "sync_joint" or called == lanes))
        execution &= (diagnostic.get("frame") == number
                      and diagnostic.get("lanes_called") == called
                      and 0 <= diagnostic.get("actual_lp_budget", -1) <= protocol["lp_node_budget"]
                      and 0 <= diagnostic.get("lp_calls", -1) <= protocol["lp_node_budget"]
                      and 0 <= diagnostic.get("shared_integer_states", -1) <= protocol["integer_state_budget"])
        waiting = [[job for job in ordered if job["coordinate"] == coordinate
                    and job["arrival"] <= start
                    and (job["coordinate"], job["index"]) not in admitted]
                   for coordinate in range(len(model.workloads))]
        fifo &= tuple(map(len, waiting)) == tuple(frame["queue"])
        offsets = [[] for _ in model.workloads]
        local_durations = []
        for lane, option_index in zip(lane_ids, indices):
            valid_index = option_index is None or (type(option_index) is int
                                                  and 0 <= option_index < len(model.options))
            execution &= valid_index
            if not valid_index:
                continue
            option = None if option_index is None else model.options[option_index]
            local = 1.0 if option is None else max(1.0, float(option.duration))
            local_durations.append(local)
            local_end = start + local
            reserved_end = start + frame["duration"] if label == "sync_joint" else local_end
            if event_cursor < len(events):
                event = events[event_cursor]
                lane_accounting &= (event["lane"] == lane and event["start"] == start
                                    and event["option_index"] == option_index
                                    and close(event["local_end"], local_end)
                                    and close(event["reserved_end"], reserved_end))
            else:
                lane_accounting = False
            event_cursor += 1
            if type(lane) is not int or not 0 <= lane < lanes:
                lane_accounting = False
                continue
            lane_accounting &= available[lane] <= start
            available[lane] = reserved_end
            exposures["barrier_idle_lane_time_s"] += exposure(local_end, reserved_end)
            exposures["scheduled_idle_lane_time_s" if option is None
                      else "occupied_lane_time_s"] += exposure(start, local_end)
            if option is not None:
                execution &= (len(option.completion_offsets) == len(model.workloads)
                              and all(len(values) == option.requirements[coordinate]
                                      and all(isfinite(value) and 0 <= value <= option.duration
                                              for value in values)
                                      for coordinate, values in enumerate(option.completion_offsets)))
                for coordinate, values in enumerate(option.completion_offsets):
                    offsets[coordinate].extend(values)
        execution &= (isfinite(frame["duration"]) and frame["duration"] > 0
                      and close(frame["duration"], max((1.0, *local_durations)))
                      and tuple(map(len, offsets)) == tuple(frame["service"])
                      == tuple(frame["executed_service"]))
        for coordinate, values in enumerate(offsets):
            fifo &= len(values) <= len(waiting[coordinate])
            for job, offset in zip(waiting[coordinate], sorted(values)):
                fifo &= (job["start"] == start and job["completion"] is not None
                         and close(job["completion"], start + offset))
                admitted.add((job["coordinate"], job["index"]))
        wall = frame["selector_wall_s"]
        execution &= isfinite(wall) and wall >= 0
        exposures["total_selector_wall_s"] += wall
    lane_accounting &= event_cursor == len(events)
    clock &= (close(result["last_frame_end"], max(available))
              and result["last_decision_end"] == (frames[-1]["start"] if frames else None))
    for job in jobs:
        key, start, completion = (job["coordinate"], job["index"]), job["start"], job["completion"]
        fifo &= ((key in admitted and start is not None and completion is not None
                  and isfinite(start) and isfinite(completion)
                  and job["arrival"] <= start <= completion)
                 or (key not in admitted and start is None and completion is None))
        exposures["waiting_area_s"] += min(end, start if start is not None else end) - job["arrival"]
    for cutoff in CUTOFFS:
        execution &= result["snapshots"][cutoff] == trajectory_snapshot(
            jobs, protocol[cutoff], len(model.workloads))
    lane_accounting &= all(isfinite(result[key]) and close(result[key], value)
                           for key, value in exposures.items())
    return {"execution": bool(execution), "clock": bool(clock),
            "causal_fifo": bool(fifo), "lane_accounting": bool(lane_accounting)}


def _comparison(candidate, baseline, tolerance):
    checks = {key: _not_worse(candidate[key], baseline[key], tolerance)
              for key in UTILITY_METRICS}
    checks["unfinished_jobs"] = candidate["unfinished_jobs"] <= baseline["unfinished_jobs"]
    return {"passed": all(checks.values()), "checks": checks,
            "deltas": {key: candidate[key] - baseline[key]
                       if candidate[key] is not None and baseline[key] is not None else None
                       for key in (*UTILITY_METRICS, "unfinished_jobs")}}


def _improvement(candidate, baseline):
    return (1 - candidate / baseline
            if candidate is not None and baseline is not None and isfinite(candidate)
            and isfinite(baseline) and baseline > 0 else None)


def summarize(models, protocol, rows):
    """Gate performance against the synchronous control; single-lane is descriptive."""
    if tuple(protocol["labels"]) != LABELS:
        raise ValueError("require the three registered lane dispatch policies")
    expected_cells = {(model.node, lanes) for model in models for lanes in protocol["lanes"]}
    actual_cells = [(row["node"], row["lanes"]) for row in rows]
    if (len(actual_cells) != len(set(actual_cells)) or set(actual_cells) != expected_cells
            or actual_cells != [tuple(cell) for cell in protocol["execution_order"]]):
        raise ValueError("all registered node/R cells must appear exactly once in order")
    tapes = {tape["tape_id"]: tape for tape in protocol["tapes"]}
    if len(tapes) != len(protocol["tapes"]):
        raise ValueError("registered arrival tape IDs must be unique")
    expected_runs = {(load, seed) for load in protocol["loads"] for seed in protocol["arrival_seeds"]}
    flags = dict.fromkeys(("execution", "clock", "causal_fifo", "lane_accounting"), True)
    certificates = references = actions = True
    totals = {label: {**dict.fromkeys(SCALAR_COSTS, 0), "whole_work": {},
                      **dict.fromkeys(EXPOSURES, 0.0)} for label in LABELS}
    audit_totals = {"calls": 0, "oracle_calls": 0, "failures": 0, "incomplete_count": 0, "work": {}}
    paired, grouped, seen = [], defaultdict(list), set()
    for row in rows:
        actual_runs = [(run["load"], run["seed"]) for run in row["runs"]]
        if len(actual_runs) != len(set(actual_runs)) or set(actual_runs) != expected_runs:
            raise ValueError("all registered load/seed pairs must appear exactly once per cell")
        model = next(model for model in models if model.node == row["node"])
        for run in row["runs"]:
            tape = tapes[run["tape_id"]]
            if (run["tape_id"] in seen or set(run["policies"]) != set(LABELS)
                    or any(tape[key] != value for key, value in (
                        ("node", row["node"]), ("lanes", row["lanes"]),
                        ("load", run["load"]), ("seed", run["seed"])))):
                raise ValueError("trajectory pairing differs from its registered tape or policies")
            seen.add(run["tape_id"])
            flags["execution"] &= run["arrival_count"] == len(tape["arrivals"])
            audit = run["reference_audit"]
            certificates &= audit["failures"] == 0
            references &= audit["incomplete_count"] == 0
            for key in audit_totals:
                if key == "work":
                    _add_work(audit_totals[key], audit[key])
                else:
                    audit_totals[key] += audit[key]
            checks = {}
            for label in LABELS:
                result = run["policies"][label]
                checks[label] = _trajectory_checks(model, result, tape, protocol, label, row["lanes"])
                for key in flags:
                    flags[key] &= checks[label][key]
                metrics, diagnostics = result["selector_metrics"], result["decision_diagnostics"]
                certificates &= metrics["failures"] == 0 and all(not item["errors"] for item in diagnostics)
                references &= (metrics["reference_incomplete_count"] == 0
                               and result["reference_audit"]["incomplete_count"] == 0
                               and all(item["reference_complete"] for item in diagnostics))
                actions &= (metrics["canonical_exact_count"] == metrics["calls"]
                            and metrics["score_mismatch_count"] == metrics["canonical_action_mismatch_count"] == 0
                            and all(frame["canonical_exact"] for frame in result["frames"])
                            and all(item["canonical_exact"] for item in diagnostics))
                for key in SCALAR_COSTS:
                    totals[label][key] += metrics[key]
                _add_whole_work(totals[label]["whole_work"], metrics["whole_work"])
                for key in EXPOSURES:
                    totals[label][key] += result[key]
            pair = {"node": row["node"], "lanes": row["lanes"], "load": run["load"],
                    "seed": run["seed"], "tape_id": run["tape_id"],
                    "trajectory_checks": checks, "snapshots": {}}
            for cutoff in CUTOFFS:
                cohort = common_cohort(run["policies"], protocol[cutoff])
                snapshots = {label: {**result["snapshots"][cutoff],
                                    "common_mean_flow_s": cohort["policies"][label]["mean_flow_s"],
                                    "common_p90_flow_s": cohort["policies"][label]["p90_flow_s"]}
                             for label, result in run["policies"].items()}
                pair["snapshots"][cutoff] = {"common_cohort": cohort, "policies": snapshots}
                grouped[(row["node"], row["lanes"], run["load"], cutoff)].append(snapshots)
            paired.append(pair)
    if seen != set(tapes):
        raise ValueError("all registered arrival tapes must appear exactly once")
    if (len(paired) != protocol["expected_tape_count"]
            or len(paired) * len(LABELS) != protocol["expected_trajectory_count"]):
        raise ValueError("trajectory coverage differs from registered counts")
    groups = []
    tolerance = protocol["float_relative_tolerance"]
    for (node, lanes, load, cutoff), snapshots in sorted(grouped.items()):
        policies = {label: {
            **{key: _average(snapshot[label][key] for snapshot in snapshots) for key in UTILITY_METRICS},
            **{key: sum(snapshot[label][key] for snapshot in snapshots) for key in COUNTS},
        } for label in LABELS}
        comparisons = {control: _comparison(policies["refill_joint"], policies[control], tolerance)
                       for control in ("sync_joint", "refill_single")}
        groups.append({"node": node, "lanes": lanes, "load": load, "cutoff": cutoff,
                       "seed_count": len(snapshots), "policies": policies, "comparisons": comparisons})
    if len(groups) != protocol["expected_group_count"]:
        raise ValueError("utility group coverage differs from registration")
    utility = all(group["comparisons"]["sync_joint"]["passed"] for group in groups)
    single_utility = all(group["comparisons"]["refill_single"]["passed"] for group in groups)
    pooled = {label: {key: _average(pair["snapshots"]["observation_end"]["policies"][label][key]
                                 for pair in paired)
                      for key in ("restricted_mean_flow_s", "common_mean_flow_s")}
              for label in LABELS}
    improvements = {control: {key: _improvement(pooled["refill_joint"][key], pooled[control][key])
                             for key in ("restricted_mean_flow_s", "common_mean_flow_s")}
                    for control in ("sync_joint", "refill_single")}
    gain_value = improvements["sync_joint"]["restricted_mean_flow_s"]
    threshold = gain_value is not None and gain_value >= protocol["minimum_pooled_flow_improvement"]
    execution = all(flags.values())
    integration = execution and certificates and references and actions
    advance = integration and utility and threshold
    reason = ("EXECUTION_HOLD" if not execution or not certificates else
              "REFERENCE_HOLD" if not references else "INTEGRATION_HOLD" if not integration else
              "UTILITY_HOLD" if not utility else "GAIN_HOLD" if not threshold else "ADVANCE")
    return {
        "schema_version": protocol["schema_version"], "experiment": protocol["experiment"],
        "registered_at_utc": protocol["registered_at_utc"],
        "decision": "LANE_REFILL_PERFORMANCE_DEVELOPMENT_" + reason,
        "execution_checks_passed": bool(execution), "clock_accounting_passed": bool(flags["clock"]),
        "causal_fifo_checks_passed": bool(flags["causal_fifo"]),
        "lane_accounting_passed": bool(flags["lane_accounting"]),
        "certificate_checks_passed": bool(certificates), "references_complete": bool(references),
        "canonical_action_gate_passed": bool(actions), "integration_gate_passed": bool(integration),
        "utility_gate_passed": bool(utility), "flow_gain_threshold_passed": bool(threshold),
        "performance_gate_passed": bool(advance), "flow_gain_established": bool(advance),
        "refill_single_utility_descriptive_passed": bool(single_utility),
        "tape_count": len(paired), "trajectory_count": len(paired) * len(LABELS),
        "utility_group_count": len(groups), "failed_utility_group_count": sum(
            not group["comparisons"]["sync_joint"]["passed"] for group in groups),
        "totals": totals, "reference_audit_totals": audit_totals,
        "groups": groups, "paired_runs": paired,
        "pooled_flow_at_T": pooled, "pooled_flow_improvement": improvements,
        "primary_control": "sync_joint", "descriptive_control": "refill_single",
        "prior_decisions": protocol["prior_decisions"], "prior_hold_retained": True,
        "limitations": protocol["limitations"],
    }
