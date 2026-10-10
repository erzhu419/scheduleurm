"""Frozen scheduling-performance screen for the known-rate lane rollout."""
from __future__ import annotations

from collections import defaultdict
from statistics import mean


POLICIES = ("maxweight", "arrival_rollout")
CUTOFFS = ("arrival_horizon", "observation_end")
FLOW_METRICS = ("restricted_mean_flow_s", "common_mean_flow_s", "common_p90_flow_s")
METRICS = (*FLOW_METRICS, "unfinished_jobs")


def _aggregate(samples):
    return {policy: {
        **{key: mean(sample[policy][key] for sample in samples) for key in FLOW_METRICS},
        "unfinished_jobs": sum(sample[policy]["unfinished_jobs"] for sample in samples),
        "common_jobs": sum(sample[policy]["common_jobs"] for sample in samples),
    } for policy in POLICIES}


def _ratio(candidate, baseline):
    return candidate / baseline if baseline else (1.0 if candidate == 0 else None)


def _contrast(policies):
    candidate, baseline = policies["arrival_rollout"], policies["maxweight"]
    return {
        "deltas": {key: candidate[key] - baseline[key] for key in METRICS},
        "ratios": {key: _ratio(candidate[key], baseline[key]) for key in METRICS},
        "regressed_metrics": [key for key in METRICS if candidate[key] > baseline[key]],
    }


def summarize(rows, protocol):
    """Keep the previous flow, load and censored-tail gates unchanged."""
    tapes = {tape["tape_id"]: tape for tape in protocol["tapes"]}
    expected = {(model["node"], model["lanes"], load, seed)
                for model in protocol["models"] for load in protocol["loads"]
                for seed in protocol["arrival_seeds"]}
    identities = [(row["node"], row["lanes"], row["load"], row["seed"]) for row in rows]
    if (len(tapes) != len(protocol["tapes"]) or len(rows) != len(tapes)
            or len(set(identities)) != len(rows) or set(identities) != expected
            or {row["tape_id"] for row in rows} != set(tapes)):
        raise ValueError("all registered node/lane/load/seed tapes must appear exactly once")
    if any(any(row[key] != tapes[row["tape_id"]][key]
               for key in ("node", "lanes", "load", "seed")) for row in rows):
        raise ValueError("paired trajectory identity differs from its registered tape")

    grouped, by_load, by_seed = defaultdict(list), defaultdict(list), defaultdict(list)
    totals = {policy: {} for policy in POLICIES}
    paired_runs = []
    for row in rows:
        for cutoff in CUTOFFS:
            grouped[row["node"], row["lanes"], row["load"], cutoff].append(row["snapshots"][cutoff])
        by_load[row["load"]].append(row)
        by_seed[row["seed"]].append(row)
        paired_runs.append({**row, "comparisons": {
            cutoff: _contrast(row["snapshots"][cutoff]) for cutoff in CUTOFFS}})
        for policy in POLICIES:
            for key, value in row["work"][policy].items():
                previous = totals[policy].get(key, 0)
                totals[policy][key] = (max(previous, value) if key == "max_rollout_calls"
                                       else previous + value)

    tolerance = protocol["performance"]["group_secondary_tolerance"]
    groups, failed_secondary = [], []
    for (node, lanes, load, cutoff), samples in sorted(grouped.items()):
        policies = _aggregate(samples)
        candidate, baseline = policies["arrival_rollout"], policies["maxweight"]
        checks = {key: candidate[key] <= (1 + tolerance) * baseline[key]
                  for key in ("common_p90_flow_s", "unfinished_jobs")}
        group = {"node": node, "lanes": lanes, "load": load, "cutoff": cutoff,
                 "paired_seeds": len(samples), "policies": policies,
                 "comparison": {**_contrast(policies), "checks": checks,
                                "secondary_passed": all(checks.values())}}
        groups.append(group)
        if not all(checks.values()):
            failed_secondary.append({key: group[key] for key in ("node", "lanes", "load", "cutoff")})
            failed_secondary[-1]["failed_metrics"] = [key for key, passed in checks.items() if not passed]

    pooled = {cutoff: _aggregate([row["snapshots"][cutoff] for row in rows]) for cutoff in CUTOFFS}
    loads = []
    for load, samples in sorted(by_load.items()):
        snapshots = {cutoff: _aggregate([row["snapshots"][cutoff] for row in samples]) for cutoff in CUTOFFS}
        primary = snapshots["observation_end"]
        loads.append({"load": load, "tapes": len(samples), "snapshots": snapshots,
                      "comparisons": {cutoff: _contrast(snapshot) for cutoff, snapshot in snapshots.items()},
                      "mean_nonregression_passed": primary["arrival_rollout"]["restricted_mean_flow_s"]
                                                   <= primary["maxweight"]["restricted_mean_flow_s"]})
    seed_aggregates = []
    for seed, samples in sorted(by_seed.items()):
        snapshots = {cutoff: _aggregate([row["snapshots"][cutoff] for row in samples]) for cutoff in CUTOFFS}
        seed_aggregates.append({"seed": seed, "paired_cells": len(samples), "snapshots": snapshots,
                                "comparisons": {cutoff: _contrast(snapshot) for cutoff, snapshot in snapshots.items()}})

    end = pooled["observation_end"]
    flow_ratio = _ratio(end["arrival_rollout"]["restricted_mean_flow_s"], end["maxweight"]["restricted_mean_flow_s"])
    flow_gain = flow_ratio is not None and flow_ratio <= 1 - protocol["performance"]["min_gain"]
    load_passed = all(load["mean_nonregression_passed"] for load in loads)
    pooled_tail_checks = {cutoff: policies["arrival_rollout"]["unfinished_jobs"]
                                 <= policies["maxweight"]["unfinished_jobs"]
                          for cutoff, policies in pooled.items()}
    reproduction = all(row["baseline_reproduction_passed"] for row in rows)
    rollout_checks = all(row["rollout_checks_passed"] for row in rows)
    execution = reproduction and rollout_checks
    performance = execution and flow_gain and load_passed and not failed_secondary and all(pooled_tail_checks.values())
    decision = ("ARRIVAL_AWARE_LANE_ROLLOUT_REPLAY_FAIL" if not execution else
                "ARRIVAL_AWARE_LANE_ROLLOUT_PERFORMANCE_PASS" if performance else
                "ARRIVAL_AWARE_LANE_ROLLOUT_PERFORMANCE_HOLD")
    return {
        "experiment": protocol["experiment"], "registered_at_utc": protocol["registered_at_utc"],
        "decision": decision, "execution_checks_passed": execution,
        "operational_integrity_passed": execution,
        "baseline_reproduction_passed": reproduction, "rollout_checks_passed": rollout_checks,
        "performance_gain_established": performance, "flow_gain_threshold_passed": flow_gain,
        "load_mean_nonregression_passed": load_passed,
        "group_secondary_passed": not failed_secondary,
        "pooled_unfinished_checks": pooled_tail_checks,
        "pooled_T_flow_ratio": flow_ratio,
        "pooled_T_flow_improvement": None if flow_ratio is None else 1 - flow_ratio,
        "tapes": len(rows), "trajectories": len(rows) * len(POLICIES),
        "independent_seed_replicates": len(seed_aggregates),
        "statistical_scope": "Development screen on three shared seeds; catalogue/load cells are paired within each seed, not independent replicates. Seed aggregates are descriptive, with no 24-pair significance test.",
        "pooled": pooled, "groups": groups, "loads": loads,
        "seed_aggregates": seed_aggregates, "failed_secondary_groups": failed_secondary,
        "totals_work": totals, "paired_runs": paired_runs, "limitations": protocol["limitations"],
    }
