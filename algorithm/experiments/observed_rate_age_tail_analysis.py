"""Read-only descriptive explanation of a frozen observed-rate experiment."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
from statistics import mean


def analyze(prefix, known_rate_result=None):
    prefix = Path(prefix)
    protocol = json.loads(prefix.with_name(prefix.name + "_protocol.json").read_text())
    result = json.loads(prefix.with_suffix(".json").read_text())
    with gzip.open(prefix.with_name(prefix.name + "_trajectories.json.gz"), "rt") as source:
        trajectories = json.load(source)
    runs = {run["tape_id"]: run for run in trajectories["runs"]}
    if set(runs) != {row["tape_id"] for row in result["paired_runs"]}:
        raise ValueError("result and trajectories describe different tapes")
    models = {model["node"]: model for model in protocol["models"]}
    true_rates = {(row["node"], row["load"]): row["arrival_rates"] for row in protocol["capacities"]}
    known = json.loads(Path(known_rate_result).read_text()) if known_rate_result else None
    known_rows = {row["tape_id"]: row for row in known["paired_runs"]} if known else {}
    classes, rate_groups = defaultdict(list), defaultdict(list)
    changed_actions = Counter()
    horizon = protocol["arrival_horizon"]
    for row in result["paired_runs"]:
        node, load = row["node"], row["load"]
        reference = known_rows.get(row["tape_id"])
        for cutoff in ("arrival_horizon", "observation_end"):
            snapshots = row["snapshots"][cutoff]
            for coordinate, workload in enumerate(models[node]["workloads"]):
                candidate = snapshots["arrival_rollout"]["per_coordinate"][coordinate]
                controls = {"maxweight": snapshots["maxweight"]["per_coordinate"][coordinate]}
                if reference:
                    controls["known_rate"] = reference["snapshots"][cutoff]["arrival_rollout"]["per_coordinate"][coordinate]
                classes[node, load, workload, cutoff].append({label: {
                    key: candidate[key] - control[key] for key in
                    ("restricted_mean_flow_s", "completed_p90_flow_s", "unfinished_jobs")}
                    for label, control in controls.items()})
        periods = defaultdict(list)
        for frame in runs[row["tape_id"]]["policies"]["arrival_rollout"]["frames"]:
            period = "0_to_60" if frame["start"] < 60 else "60_to_H" if frame["start"] < horizon else "H_and_later"
            periods[period].append(frame["rollout"]["estimated_arrival_rates"])
            if frame["rollout"]["changed"]:
                changed_actions[models[node]["options"][frame["option_indices"][0]]["action_id"]] += 1
        for period, estimates in periods.items():
            nominal = true_rates[node, load]
            rate_groups[node, load, period].append({
                "decisions": len(estimates),
                "mean_estimate": [mean(values[c] for values in estimates) for c in range(len(nominal))],
                "mean_absolute_error": [mean(abs(values[c] - nominal[c]) for values in estimates) for c in range(len(nominal))],
            })
    per_class = []
    for (node, load, workload, cutoff), samples in sorted(classes.items()):
        contrasts = {}
        for label in samples[0]:
            matched = [sample[label] for sample in samples if label in sample]
            contrasts[label] = {"paired_tapes": len(matched),
                **{key: (sum(sample[key] for sample in matched) if key == "unfinished_jobs"
                         else mean(sample[key] for sample in matched)) for key in matched[0]}}
        per_class.append({"node": node, "load": load, "workload": workload, "cutoff": cutoff,
                          "candidate_minus_control": contrasts})
    rates = []
    for (node, load, period), samples in sorted(rate_groups.items()):
        nominal = true_rates[node, load]
        estimated = [mean(sample["mean_estimate"][c] for sample in samples) for c in range(len(nominal))]
        rates.append({"node": node, "load": load, "period": period,
            "paired_tapes": len(samples), "decisions": sum(sample["decisions"] for sample in samples),
            "pre_H_nominal_rates": nominal, "equal_tape_mean_estimates": estimated,
            "relative_bias": [(estimated[c] - nominal[c]) / nominal[c] for c in range(len(nominal))],
            "relative_mean_absolute_error": [mean(sample["mean_absolute_error"][c] for sample in samples) / nominal[c]
                                             for c in range(len(nominal))]})
    return {
        "decision": result["decision"], "registered_at_utc": result["registered_at_utc"],
        "performance_gain_established": result["performance_gain_established"],
        "pooled": result["pooled"], "pooled_T_flow_improvement": result["pooled_T_flow_improvement"],
        "previous_known_rate_comparison": result.get("previous_known_rate_comparison"),
        "failed_secondary_groups": result["failed_secondary_groups"],
        "local_class_regressions": [group for group in per_class if any(value > 0
            for contrast in group["candidate_minus_control"].values()
            for key, value in contrast.items() if key != "paired_tapes")],
        "per_class_paired_deltas": per_class,
        "estimated_rates_by_decision_period": rates,
        "changed_root_action_counts": dict(changed_actions),
        "actual_work": result["totals_work"],
        "scope": "Read-only description; no policy changes or new experiments. Primary common P90 is the equal-tape mean of each tape's common-cohort P90, not a pooled-job quantile. Per-class completed P90 uses each policy's completed class jobs and is descriptive. Rate means and absolute errors first average decisions within each tape, then weight tapes equally; decisions are not independent replicates. After H the estimate represents the historical pre-H rate, while the planning arrival window is zero. Known-rate per-class contrasts require identical tape IDs; no rematching of seeds.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", type=Path)
    parser.add_argument("--known-rate-result", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.prefix, args.known_rate_result)
    if args.output:
        with args.output.open("x") as out:
            json.dump(result, out, separators=(",", ":"), allow_nan=False)
        print(json.dumps({"output": str(args.output), "decision": result["decision"],
                          "pooled_T_flow_improvement": result["pooled_T_flow_improvement"],
                          "local_class_regression_groups": len(result["local_class_regressions"])}, indent=2))
    else:
        print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
