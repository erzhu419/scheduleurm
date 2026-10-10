"""Paired seed-block uncertainty for the frozen BG confirmation population.

Resample the 12 actual-arrival seeds, keeping all eight catalogue/load cells and
both policies together. Flow statistics are equal-tape means; common P90 is the
mean of per-tape P90s on shared completed job IDs, not a pooled job percentile.
Callers must compute that shared-ID population before supplying the snapshots.
These pointwise percentile intervals are exploratory and conditional on the
fixed planning streams. They neither change the original gates nor constitute
a new independent confirmation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from math import fsum, isfinite
from pathlib import Path
from random import Random


ROOT = Path(__file__).resolve().parents[2]
ARRIVAL_SEEDS = tuple(range(100001, 100013))
CELLS = tuple((node, lanes, load)
              for node, lanes in (("cpu_a", 2), ("node007", 4),
                                  ("gpu_a", 2), ("gpu_b", 2))
              for load in (0.8, 0.95))
CUTOFF_TIMES = {"arrival_horizon": 900.0, "observation_end": 1200.0}
METRICS = ("restricted_mean_flow_s", "common_p90_flow_s", "unfinished_jobs")
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 20261003


def _identity(row):
    return row["node"], row["lanes"], row["load"], row["seed"]


def _validate_roster(rows):
    expected = {(*cell, seed) for cell in CELLS for seed in ARRIVAL_SEEDS}
    identities = [_identity(row) for row in rows]
    tape_ids = [row["tape_id"] for row in rows]
    if (len(rows) != 96 or len(set(identities)) != 96
            or set(identities) != expected or len(set(tape_ids)) != 96):
        raise ValueError("require all 12 registered seed blocks and their eight paired cells exactly once")


def _percentile_interval(values):
    ordered = sorted(values)

    def quantile(probability):
        position = (len(ordered) - 1) * probability
        index = int(position)
        fraction = position - index
        return ordered[index] + fraction * (ordered[min(index + 1, len(ordered) - 1)] - ordered[index])

    return [quantile(0.025), quantile(0.975)]


def block_intervals(rows, labels=("maxweight", "arrival_rollout"),
                    cutoff="observation_end", *, bootstrap_replicates=20_000,
                    seed=20261003):
    """Return candidate-minus-baseline pointwise 95% paired cluster intervals.

    Input rows contain tape_id/node/lanes/load/seed and snapshots[cutoff][label].
    The fixed roster is the existing 12-seed, four-catalogue, two-load population.
    A missing common P90 makes that whole metric not estimable: no cell or seed
    is dropped. Unfinished jobs use total-count differences only, without ratios.
    """
    rows = list(rows)
    labels = tuple(labels)
    if len(labels) != 2 or labels[0] == labels[1]:
        raise ValueError("supply distinct baseline and candidate labels")
    if cutoff not in CUTOFF_TIMES or bootstrap_replicates < 1:
        raise ValueError("require a registered cutoff and positive bootstrap replicate count")
    _validate_roster(rows)
    blocks = defaultdict(list)
    for row in rows:
        pair = row["snapshots"][cutoff]
        baseline, candidate = (pair[label] for label in labels)
        if (baseline["time"] != CUTOFF_TIMES[cutoff]
                or candidate["time"] != CUTOFF_TIMES[cutoff]
                or baseline["arrivals"] != candidate["arrivals"]
                or baseline["common_jobs"] != candidate["common_jobs"]):
            raise ValueError("paired snapshots must share cutoff, birth cohort and common-ID count")
        blocks[row["seed"]].append(row)
    for group in blocks.values():
        group.sort(key=lambda row: (row["node"], row["lanes"], row["load"]))

    rng = Random(seed)
    draws = [tuple(rng.randrange(12) for _ in ARRIVAL_SEEDS)
             for _ in range(bootstrap_replicates)]
    metrics = {}
    for metric in METRICS:
        per_block = []
        missing = 0
        for arrival_seed in ARRIVAL_SEEDS:
            policy_values = []
            for label in labels:
                values = [row["snapshots"][cutoff][label][metric]
                          for row in blocks[arrival_seed]]
                missing += sum(value is None for value in values)
                if any(value is not None and (not isfinite(value) or value < 0)
                       for value in values):
                    raise ValueError("flow and unfinished metrics must be finite and nonnegative")
                if any(value is None for value in values):
                    policy_values.append(None)
                else:
                    total = fsum(values)
                    policy_values.append(total if metric == "unfinished_jobs" else total / 8)
            per_block.append(policy_values)
        if missing:
            metrics[metric] = {"estimable": False,
                               "reason": "missing common-ID statistic; all 12 blocks retained",
                               "missing_policy_cell_values": missing}
            if metric != "common_p90_flow_s":
                raise ValueError("restricted mean and unfinished counts must be available for all cells")
            continue

        scale = 1.0 if metric == "unfinished_jobs" else 1 / 12
        point = [fsum(block[policy] for block in per_block) * scale for policy in (0, 1)]
        differences, relative_changes = [], []
        undefined_ratios = 0
        for indices in draws:
            baseline = fsum(per_block[index][0] for index in indices) * scale
            candidate = fsum(per_block[index][1] for index in indices) * scale
            differences.append(candidate - baseline)
            if metric != "unfinished_jobs":
                if baseline == 0:
                    undefined_ratios += 1
                else:
                    relative_changes.append(candidate / baseline - 1)
        entry = {
            "estimable": True, "baseline": point[0], "candidate": point[1],
            "aggregation": "total_jobs" if metric == "unfinished_jobs" else "equal_tape_mean",
            "paired_difference": {"estimate": point[1] - point[0],
                                  "ci95": _percentile_interval(differences)},
        }
        if metric != "unfinished_jobs":
            entry["relative_change"] = {
                "estimate": point[1] / point[0] - 1 if point[0] else None,
                "ci95": None if undefined_ratios else _percentile_interval(relative_changes),
                "undefined_bootstrap_replicates": undefined_ratios,
                "definition": "candidate_mean / baseline_mean - 1",
            }
        metrics[metric] = entry
    return {
        "labels": labels, "cutoff": cutoff, "time": CUTOFF_TIMES[cutoff],
        "paired_tapes": len(rows), "arrival_seed_blocks": len(ARRIVAL_SEEDS),
        "cells_per_block": len(CELLS), "arrival_seeds": ARRIVAL_SEEDS,
        "bootstrap_replicates": bootstrap_replicates, "bootstrap_seed": seed,
        "bootstrap_generator": "Python random.Random.randrange; 12 whole blocks with replacement",
        "interval": "pointwise percentile 95%; linear interpolation at 0.025/0.975",
        "scope": "Exploratory paired actual-arrival seed-block uncertainty, conditional on fixed planning streams; no multiplicity adjustment or new PASS gate.",
        "metrics": metrics,
    }


def analyse_confirmation(result, protocol, design):
    """Analyse the existing cached result only after matching its frozen epoch."""
    epoch = design["source_registered_at_utc"]
    if result["registered_at_utc"] != epoch or protocol["registered_at_utc"] != epoch:
        raise ValueError("cached result and protocol must match the design's source epoch")
    if not result["execution_checks_passed"]:
        raise ValueError("cached source execution checks did not pass")
    if (tuple(design["labels"]) != ("maxweight", "arrival_rollout")
            or tuple(design["cutoffs"]) != tuple(CUTOFF_TIMES)
            or design["bootstrap_replicates"] != BOOTSTRAP_REPLICATES
            or design["bootstrap_seed"] != BOOTSTRAP_SEED):
        raise ValueError("require the frozen BG labels, cutoffs and bootstrap parameters")
    rows, tapes = result["paired_runs"], protocol["tapes"]
    _validate_roster(rows)
    _validate_roster(tapes)
    registered = {tape["tape_id"]: _identity(tape) for tape in tapes}
    if any(registered.get(row["tape_id"]) != _identity(row) for row in rows):
        raise ValueError("cached paired tape identity differs from the registered source")
    return {
        "analysis": "cached_BG_confirmation_paired_seed_block_intervals",
        "source_registered_at_utc": epoch,
        "source_result": design["source_result"], "source_protocol": design["source_protocol"],
        "source_decision": result["decision"], "new_trajectories": 0,
        "birth_cohort": protocol["birth_cohort"],
        "snapshots": {cutoff: block_intervals(
            rows, labels=design["labels"], cutoff=cutoff,
            bootstrap_replicates=design["bootstrap_replicates"], seed=design["bootstrap_seed"])
            for cutoff in design["cutoffs"]},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    design = json.loads(args.design.read_text())
    result = json.loads((ROOT / design["source_result"]).read_text())
    protocol = json.loads((ROOT / design["source_protocol"]).read_text())
    summary = analyse_confirmation(result, protocol, design)
    summary["design_path"] = str(args.design)
    with args.output.open("x") as out:
        json.dump(summary, out, indent=2, allow_nan=False)
        out.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()
