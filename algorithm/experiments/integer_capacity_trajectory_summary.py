"""Paired trajectory decisions with shared completed cohorts and censored tails."""
from __future__ import annotations

from collections import defaultdict
from math import floor, isfinite, isclose
from statistics import mean


CONTROLS = ("old_shared", "fractional_shared")
CUTOFFS = ("arrival_horizon", "observation_end")
UTILITY_METRICS = ("restricted_mean_flow_s", "common_mean_flow_s", "common_p90_flow_s")


def _flow_stats(values):
    values = sorted(values)
    if not values:
        return {"mean_flow_s": None, "p90_flow_s": None}
    position = .9 * (len(values) - 1)
    left = floor(position)
    right = min(left + 1, len(values) - 1)
    return {"mean_flow_s": mean(values),
            "p90_flow_s": values[left] + (position - left) * (values[right] - values[left])}


def common_cohort(policies, cutoff):
    """Use the intersection of actual completed IDs across every policy."""
    completed = {
        label: {(job["coordinate"], job["index"]): job["completion"] - job["arrival"]
                for job in result["jobs"] if job["arrival"] <= cutoff
                and job["completion"] is not None and job["completion"] <= cutoff}
        for label, result in policies.items()
    }
    ids = sorted(set.intersection(*(set(jobs) for jobs in completed.values())))
    return {"job_ids": [list(job_id) for job_id in ids], "count": len(ids),
            "policies": {label: _flow_stats(jobs[job_id] for job_id in ids)
                         for label, jobs in completed.items()}}


def _identity(candidate, reference):
    # Search effort and certificates may differ without changing execution.
    for index, (first, second) in enumerate(zip(candidate["frames"], reference["frames"])):
        for key in ("start", "duration", "queue", "option_indices", "service"):
            a, b = first[key], second[key]
            if key in ("queue", "option_indices", "service"):
                a, b = tuple(a), tuple(b)
            if a != b:
                return {"passed": False, "first_mismatch": {"frame": index, "field": key,
                                                            "candidate": a, "reference": b}}
    if len(candidate["frames"]) != len(reference["frames"]):
        return {"passed": False, "first_mismatch": {"field": "frame_count"}}
    if candidate["jobs"] != reference["jobs"]:
        return {"passed": False, "first_mismatch": {"field": "jobs"}}
    return {"passed": True, "first_mismatch": None}


def _not_worse(value, baseline, tolerance):
    return (value is not None and baseline is not None and isfinite(value) and isfinite(baseline)
            and (value <= baseline or isclose(value, baseline, rel_tol=tolerance, abs_tol=0)))


def _average(values):
    values = list(values)
    return mean(values) if values and all(value is not None for value in values) else None


def summarize(models, protocol, rows):
    labels = protocol["labels"]
    expected_cells = {(model.node, lanes) for model in models for lanes in protocol["lanes"]}
    actual_cells = [(row["node"], row["lanes"]) for row in rows]
    if len(actual_cells) != len(set(actual_cells)) or set(actual_cells) != expected_cells:
        raise ValueError("all registered node/R cells must appear exactly once")
    tapes = {tape["tape_id"]: tape for tape in protocol["tapes"]}
    expected_runs = {(load, seed) for load in protocol["loads"] for seed in protocol["arrival_seeds"]}
    execution = certificates = references = actions = identity = True
    paired, grouped = [], defaultdict(list)
    totals = {label: {"calls": 0, "failures": 0, "canonical_exact_count": 0,
                      "score_mismatch_count": 0, "canonical_action_mismatch_count": 0,
                      "cap_reached_count": 0, "uncertified_cap_reached_count": 0,
                      "operational_elapsed_s": 0.0, "search_work": {},
                      "preprocessing": {}, "capacity_propagation": {}} for label in labels}
    audit_totals = {"calls": 0, "failures": 0, "incomplete_count": 0,
                    "search_work": {}, "operational_elapsed_s": 0.0}
    for row in rows:
        actual_runs = [(run["load"], run["seed"]) for run in row["runs"]]
        if len(actual_runs) != len(set(actual_runs)) or set(actual_runs) != expected_runs:
            raise ValueError("all registered load/seed pairs must appear exactly once per cell")
        for run in row["runs"]:
            tape = tapes[run["tape_id"]]
            if any(tape[key] != value for key, value in (
                    ("node", row["node"]), ("lanes", row["lanes"]),
                    ("load", run["load"]), ("seed", run["seed"]))) or set(run["policies"]) != set(labels):
                raise ValueError("trajectory pairing differs from registered tape or policies")
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
            for label, result in run["policies"].items():
                metrics = result["selector_metrics"]
                execution &= (all(result["invariants"].values())
                              and result["frame_count"] == len(result["frames"]) == metrics["calls"]
                              and sorted((job["arrival"], job["coordinate"], job["index"])
                                         for job in result["jobs"]) == expected_jobs
                              and result["budget"] == (protocol["reference_budget"] if label == "full_reference"
                                                      else protocol["budgets"][str(row["lanes"])])
                              and all(result[key] == protocol[key] for key in CUTOFFS))
                certificates &= metrics["failures"] == 0
                references &= result["reference_audit"]["incomplete_frame_count"] == 0
                if label == "integer_capacity":
                    actions &= (metrics["canonical_exact_count"] == metrics["calls"]
                                and metrics["reference_comparable_count"] == metrics["calls"]
                                and metrics["score_mismatch_count"] == 0
                                and metrics["canonical_action_mismatch_count"] == 0)
                for key in totals[label]:
                    if isinstance(totals[label][key], dict):
                        for name, value in metrics[key].items():
                            totals[label][key][name] = totals[label][key].get(name, 0) + value
                    else:
                        totals[label][key] += metrics[key]
            match = _identity(run["policies"]["integer_capacity"], run["policies"]["full_reference"])
            identity &= match["passed"]
            pair = {"node": row["node"], "lanes": row["lanes"], "load": run["load"],
                    "seed": run["seed"], "tape_id": run["tape_id"],
                    "trajectory_identity": match, "snapshots": {}}
            for cutoff in CUTOFFS:
                cohort = common_cohort(run["policies"], protocol[cutoff])
                snapshots = {label: {**result["snapshots"][cutoff],
                                    "common_mean_flow_s": cohort["policies"][label]["mean_flow_s"],
                                    "common_p90_flow_s": cohort["policies"][label]["p90_flow_s"]}
                             for label, result in run["policies"].items()}
                for snapshot in snapshots.values():
                    execution &= (snapshot["arrivals"] == len(expected_jobs)
                                  and snapshot["unfinished_jobs"] == snapshot["queued_jobs"] + snapshot["inflight_jobs"]
                                  and snapshot["arrivals"] == snapshot["completed_jobs"] + snapshot["unfinished_jobs"])
                pair["snapshots"][cutoff] = {"common_cohort": cohort, "policies": snapshots}
                grouped[(row["node"], row["lanes"], run["load"], cutoff)].append(snapshots)
            paired.append(pair)
    tolerance = protocol["float_relative_tolerance"]
    groups = []
    for (node, lanes, load, cutoff), snapshots in sorted(grouped.items()):
        policies = {label: {
            **{key: _average(snapshot[label][key] for snapshot in snapshots) for key in UTILITY_METRICS},
            **{key: sum(snapshot[label][key] for snapshot in snapshots)
               for key in ("arrivals", "completed_jobs", "queued_jobs", "inflight_jobs", "unfinished_jobs")},
        } for label in labels}
        comparisons = {}
        for control in CONTROLS:
            candidate, baseline = policies["integer_capacity"], policies[control]
            checks = {key: _not_worse(candidate[key], baseline[key], tolerance) for key in UTILITY_METRICS}
            checks["unfinished_jobs"] = candidate["unfinished_jobs"] <= baseline["unfinished_jobs"]
            comparisons[control] = {"passed": all(checks.values()), "checks": checks,
                                    "deltas": {key: candidate[key] - baseline[key]
                                               if candidate[key] is not None and baseline[key] is not None else None
                                               for key in (*UTILITY_METRICS, "unfinished_jobs")}}
        groups.append({"node": node, "lanes": lanes, "load": load, "cutoff": cutoff,
                       "seed_count": len(snapshots), "policies": policies, "comparisons": comparisons})
    utility = all(comparison["passed"] for group in groups for comparison in group["comparisons"].values())
    pooled = {label: _average(pair["snapshots"]["observation_end"]["policies"][label]["common_mean_flow_s"]
                             for pair in paired) for label in labels}
    improvements = {control: 1 - pooled["integer_capacity"] / pooled[control]
                    if pooled["integer_capacity"] is not None and pooled[control] is not None and pooled[control] > 0
                    else None for control in CONTROLS}
    integration = execution and certificates and references and actions and identity
    gain = integration and utility and all(value is not None and value >= protocol["minimum_pooled_flow_improvement"]
                                          for value in improvements.values())
    decision = ("INVALID_EXECUTION" if not execution or not certificates else
                "INCOMPLETE_REFERENCE" if not references else
                "INTEGER_CAPACITY_TRAJECTORY_INTEGRATION_HOLD" if not integration else
                "INTEGER_CAPACITY_TRAJECTORY_UTILITY_HOLD" if not utility else
                "INTEGER_CAPACITY_TRAJECTORY_FLOW_GAIN" if gain else
                "INTEGER_CAPACITY_TRAJECTORY_PRESERVED")
    return {
        "decision": decision, "execution_checks_passed": execution,
        "certificate_checks_passed": certificates, "references_complete": references,
        "integer_action_gate_passed": actions, "trajectory_identity_passed": identity,
        "integration_gate_passed": integration, "utility_gate_passed": utility,
        "flow_gain_established": gain, "tape_count": len(paired), "trajectory_count": len(paired) * len(labels),
        "totals": totals, "reference_audit_totals": audit_totals, "groups": groups, "paired_runs": paired,
        "pooled_common_mean_flow_s_at_T": pooled, "pooled_flow_improvement": improvements,
        "limitations": protocol["limitations"],
    }
