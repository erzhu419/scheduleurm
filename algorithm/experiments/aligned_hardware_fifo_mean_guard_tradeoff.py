"""Describe completed mean-guard24 shard JSONs without loading trajectories."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean

STEM = "aligned_hardware_fifo_mean_guard_development_20261003"
ARTIFACTS = Path(__file__).resolve().parents[2] / "md/experiment_artifacts"
METRICS = ("restricted_mean_flow_s", "common_mean_flow_s", "common_p90_flow_s",
           "unfinished_jobs", "common_jobs")
COUNTS = ("completed_new_stage_frames", "chosen_true_S1_above_BG_frames",
          "chosen_true_S2_above_BG_frames")
WORK = ("mean_guard_comparisons", "mean_guard_allowed", "mean_guard_filtered")


def read(path):
    return json.loads(path.read_text())


def analyze(directory):
    summary = read(directory / (STEM + "_summary.json"))
    if (not summary["execution_checks_passed"] or summary["completed_shards"] != 24
            or summary["newly_executed_candidate_trajectories"] != 24):
        raise ValueError("completed execution-valid24 summary required")
    epoch = summary["registered_at_utc"]
    baseline = summary["comparisons"]["finite_current"]
    cells = {(g["node"], g["lanes"], g["load"]) for g in baseline["groups"]}
    seeds = {g["seed"] for g in baseline["seed_aggregates"]}
    expected = {(*cell, seed) for cell in cells for seed in seeds}
    if len(cells) != 8 or len(seeds) != 3 or len(expected) != 24:
        raise ValueError("eight catalogue/load cells and three shared seeds required")
    rows = []
    for index in range(24):
        doc = read(directory / (STEM + f"_shard_{index:02d}.json"))
        row = doc["row"]
        if (doc["index"] != index or doc["registered_at_utc"] != epoch
                or not row["rollout_checks_passed"] or not row["baseline_reproduction_passed"]):
            raise ValueError("shard identity, epoch or execution checks differ")
        rows.append(row)
    identities = [(r["node"], r["lanes"], r["load"], r["seed"]) for r in rows]
    if (len(set(identities)) != 24 or set(identities) != expected
            or len({r["tape_id"] for r in rows}) != 24):
        raise ValueError("summary and shard24 cell/seed identities differ")

    records, signs = [], {"equal": 0, "positive": 0, "negative": 0}
    for row in rows:
        comparison = row["action_comparison"]
        first = comparison["first_divergence"]
        gap = sign = None
        if first is not None:
            candidate, parent = first["candidate_true_S1"], first["parent_true_S1"]
            gap = candidate - parent
            sign = "equal" if candidate == parent else "positive" if candidate > parent else "negative"
            signs[sign] += 1
        records.append({**{k: row[k] for k in ("tape_id", "node", "lanes", "load", "seed")},
            "action_comparison": comparison, "reported_float_S1_gap_s": gap,
            "reported_float_S1_gap_sign": sign})
    divergent = sum(signs.values())
    if divergent != summary["divergent_tapes"]:
        raise ValueError("summary and shard first-divergence counts differ")
    work = {name: sum(r["work"]["arrival_rollout"][name] for r in rows) for name in WORK}
    if work != summary["mean_guard_work"]:
        raise ValueError("summary and shard mean-guard totals differ")

    classes = defaultdict(list)
    for row in rows:
        for item in row["class_snapshots"]:
            classes[row["load"], item["workload"]].append(item["comparisons"])
    class_means = []
    for (load, workload), samples in sorted(classes.items()):
        if len(samples) != 12:
            raise ValueError("each load/workload requires twelve paired cells")
        comparisons = {}
        for role in summary["comparisons"]:
            comparisons[role] = {}
            for cutoff in ("arrival_horizon", "observation_end"):
                scores = {policy: {metric: mean(s[role][cutoff][policy][metric] for s in samples)
                                   for metric in METRICS}
                          for policy in ("maxweight", "arrival_rollout")}
                comparisons[role][cutoff] = {"reference": scores["maxweight"],
                    "candidate": scores["arrival_rollout"],
                    "candidate_minus_reference": {m: scores["arrival_rollout"][m]
                        - scores["maxweight"][m] for m in METRICS}}
        class_means.append({"load": load, "workload": workload, "paired_cells": 12,
                            "comparisons": comparisons})

    def areas(selected):
        candidate = mean(r["candidate_middle_queue_area_s"] for r in selected)
        refs = {role: mean(r["reference_middle_queue_area_s"][role] for r in selected)
                for role in summary["comparisons"]}
        return {"paired_cells": len(selected), "candidate": candidate, "references": refs,
                "candidate_minus_reference": {role: candidate - value for role, value in refs.items()}}

    return {"registered_at_utc": epoch, "decision": summary["decision"],
        "performance_gain_established": summary["performance_gain_established"],
        "original_joint_gate_passed": summary["original_joint_gate_passed"],
        "parent_increment": summary["parent_increment"], "completed_shards": 24,
        "independent_seed_replicates": 3, "shared_seeds": sorted(seeds),
        "statistical_scope": "Same24 descriptive cells; eight paired cells per shared seed, not24 independent seeds. No new simulation or promotion gate.",
        "first_divergence": {"divergent_tapes": divergent, "equal_action_tapes": 24 - divergent,
            "reported_float_S1_gap_counts": signs,
            "precision": "Exact equality/order of saved JSON float reports; gap subtraction is floating point, not an exact Fraction comparison."},
        "action_counts": {name: sum(r["action_comparison"][name] for r in rows) for name in COUNTS},
        "action_scope": "First-divergence scores share a causal prefix; all-frame counts also include later divergent physical contexts.",
        "mean_guard_work": work,
        "class_scope": "Arithmetic means of twelve catalogue/seed cell reports per load/workload, not job-pooled means or P90. Unfinished/common-job counts are cell averages. Common-ID cohorts are paired separately for each reference.",
        "class_cell_means": class_means,
        "middle_queue_area_scope": "Arithmetic mean of existing queue-area[300,900] reports over the same24 cells; includes all jobs, not only the birth cohort.",
        "middle_queue_area_s_cell_mean": areas(rows),
        "middle_queue_area_s_by_load": [{"load": load, **areas([r for r in rows if r["load"] == load])}
                                         for load in sorted({r["load"] for r in rows})],
        "tapes": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACTS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.artifact_dir)
    output = args.output or args.artifact_dir / (STEM + "_tradeoff.json")
    with output.open("x") as target:
        json.dump(result, target, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        target.write("\n")
    print(json.dumps({"output": str(output), "decision": result["decision"],
                      "first_divergence": result["first_divergence"]}), flush=True)


if __name__ == "__main__":
    main()
