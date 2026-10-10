"""Single first-difference intervention after the frozen clearing rollout failed.

This diagnoses the first altered action with ordinary MaxWeight continuation;
it introduces neither a new scheduling candidate nor a performance advance gate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
from statistics import mean

from algorithm.experiments.aligned_hardware_performance_necessity import common_metrics, load_models
from algorithm.experiments.contextual_lane_refill_replay import simulate_contextual_trajectory
from algorithm.visible_queue_clearing_rollout import select_context_maxweight, select_visible_queue_clearing


ROOT = Path(__file__).resolve().parents[2]
PREVIOUS = ROOT / "md/experiment_artifacts/aligned_hardware_visible_queue_clearing_development_20260930"
DEFAULT = ROOT / "md/experiment_artifacts/visible_queue_first_change_diagnosis_20260930"
SOURCE = "algorithm/experiments/visible_queue_first_change_diagnosis.py"
ONE_SHOT = "first_change_mw_continuation"
LABELS = ("maxweight", "clearing_rollout", ONE_SHOT)
CUTOFFS = ("arrival_horizon", "observation_end")
PREFIX_FIELDS = ("start", "queue", "available_before", "lane_ids", "option_indices", "service", "duration")


class FirstChangeMaxWeight:
    """Use the unchanged rollout until one changed decision, then only MaxWeight."""
    def __init__(self):
        self.first_change_frame = None
        self.calls = 0

    def __call__(self, options, queue, available, now, lane):
        decision = (select_visible_queue_clearing(options, queue, available, now, lane)
                    if self.first_change_frame is None else
                    select_context_maxweight(options, queue, available, now, lane))
        if decision.changed:
            self.first_change_frame = self.calls
        self.calls += 1
        return decision


def prefix_matches(first, clearing, baseline, changed_frame):
    expected = next((i for i, frame in enumerate(clearing["frames"])
                     if frame["rollout"]["changed"]), None)
    actual = [i for i, frame in enumerate(first["frames"]) if frame["rollout"]["changed"]]
    if changed_frame != expected or actual != ([] if expected is None else [expected]):
        return False
    length = len(clearing["frames"]) if expected is None else expected + 1
    if len(first["frames"]) < length or len(baseline["frames"]) < length:
        return False
    normalized = lambda value: json.loads(json.dumps(value))
    for i in range(length):
        if any(normalized(first["frames"][i][key]) != clearing["frames"][i][key]
               for key in (*PREFIX_FIELDS, "rollout")):
            return False
        if i != expected and any(normalized(first["frames"][i][key]) != baseline["frames"][i][key]
                                 for key in PREFIX_FIELDS):
            return False
    return expected is not None or normalized(first["jobs"]) == baseline["jobs"]


def job_effects(intervention, baseline, cutoff, tolerance):
    """All released jobs count, including their unfinished restricted-flow tails."""
    base = {(job["coordinate"], job["index"]): job for job in baseline["jobs"]}
    deltas, starts, completions = [], [], []
    for job in intervention["jobs"]:
        old = base[job["coordinate"], job["index"]]
        if job["arrival"] > cutoff:
            continue
        restricted = lambda row: min(cutoff, row["completion"] if row["completion"] is not None else cutoff) - row["arrival"]
        deltas.append(restricted(job) - restricted(old))
        for field, values in (("start", starts), ("completion", completions)):
            if job[field] is not None and old[field] is not None:
                values.append(job[field] - old[field])

    def counts(values):
        return {"earlier": sum(value < -tolerance for value in values),
                "later": sum(value > tolerance for value in values),
                "equal": sum(abs(value) <= tolerance for value in values),
                "compared_jobs": len(values), "total_delta_s": sum(values)}

    return {"restricted_flow": counts(deltas), "start": counts(starts),
            "completion": counts(completions), "restricted_mean_delta_s": mean(deltas)}


def first_change_job_groups(intervention, baseline, changed, protocol):
    """Partition actual FIFO IDs before the call, including earlier same-time calls."""
    if changed is None:
        return None
    frame = intervention["frames"][changed]
    now, dimension = frame["start"], len(frame["queue"])
    admitted = [sum(prior["service"][coordinate] for prior in intervention["frames"][:changed])
                for coordinate in range(dimension)]
    waiting, prefix = [], []
    for coordinate in range(dimension):
        released = sorted((job for job in baseline["jobs"]
                           if job["coordinate"] == coordinate and job["arrival"] <= now),
                          key=lambda job: (job["arrival"], job["index"]))
        prefix.extend(released[:admitted[coordinate]])
        waiting.extend(released[admitted[coordinate]:])
    if tuple(sum(job["coordinate"] == coordinate for job in waiting)
             for coordinate in range(dimension)) != tuple(frame["queue"]):
        raise ValueError("first-change visible IDs differ from the causal FIFO queue")
    groups = {"visible_waiting": waiting, "previously_admitted": prefix,
              "future_arrivals": [job for job in baseline["jobs"] if job["arrival"] > now]}
    index = {(job["coordinate"], job["index"]): job for job in intervention["jobs"]}
    result = {"job_ids": {name: [(job["coordinate"], job["index"]) for job in jobs]
                          for name, jobs in groups.items()}, "snapshots": {}}
    for cutoff in CUTOFFS:
        stats = {}
        for name, jobs in groups.items():
            released = [job for job in jobs if job["arrival"] <= protocol[cutoff]]
            if released:
                stats[name] = job_effects({"jobs": [index[job["coordinate"], job["index"]] for job in released]},
                                          {"jobs": released}, protocol[cutoff], protocol["diagnostic_zero_tolerance_s"])
            else:
                stats[name] = {"restricted_flow": {"earlier": 0, "later": 0, "equal": 0,
                                                   "compared_jobs": 0, "total_delta_s": 0.},
                               "restricted_mean_delta_s": 0.}
        result["snapshots"][cutoff] = stats
    return result


def run_tape(model, tape, protocol, reused):
    selector = FirstChangeMaxWeight()
    result = simulate_contextual_trajectory(model, tape["arrivals"], protocol["arrival_horizon"],
        protocol["observation_end"], selector, policy=ONE_SHOT)
    baseline, clearing = reused["maxweight"], reused["clearing_rollout"]
    matching = prefix_matches(result, clearing, baseline, selector.first_change_frame)
    policies = {**reused, ONE_SHOT: result}
    changed = selector.first_change_frame
    evidence = None
    if changed is not None:
        frame = result["frames"][changed]
        old = select_context_maxweight(model.options, frame["queue"], frame["available_before"],
                                       frame["start"], frame["lane_ids"][0])
        old_index, = old.selection.option_indices
        new_index, = frame["option_indices"]
        decision = frame["rollout"]
        evidence = {"frame": changed, **{key: frame[key] for key in PREFIX_FIELDS},
                    "busy_remaining_s": [max(0., end - frame["start"]) for end in frame["available_before"]],
                    "old_root": model.options[old_index].snapshot(),
                    "new_root": model.options[new_index].snapshot(),
                    "baseline_predicted_completion_s": decision["baseline_predicted_completion_s"],
                    "chosen_predicted_completion_s": decision["chosen_predicted_completion_s"],
                    "predicted_gain_s": decision["baseline_predicted_completion_s"] - decision["chosen_predicted_completion_s"]}
    tolerance = protocol["diagnostic_zero_tolerance_s"]
    return {**{key: tape[key] for key in ("tape_id", "node", "lanes", "load", "seed")},
            "prefix_and_first_action_match": matching, "first_change": evidence,
            "snapshots": {cutoff: common_metrics(policies, protocol[cutoff]) for cutoff in CUTOFFS},
            "job_time_effects_vs_maxweight": {
                cutoff: {label: job_effects(policies[label], baseline, protocol[cutoff], tolerance)
                         for label in ("clearing_rollout", ONE_SHOT)} for cutoff in CUTOFFS},
            "first_change_job_groups": first_change_job_groups(result, baseline, changed, protocol),
            "new_work": result["rollout_work"]}


def build_protocol(previous):
    for path, source in previous["execution_sources"].items():
        if (ROOT / path).read_text() != source:
            raise ValueError(f"original frozen development source changed: {path}")
    return {"experiment": "visible_queue_first_change_diagnosis_v1",
            "previous_protocol": str(PREVIOUS.with_name(PREVIOUS.name + "_protocol.json").relative_to(ROOT)),
            "previous_trajectories": str(PREVIOUS.with_name(PREVIOUS.name + "_trajectories.json.gz").relative_to(ROOT)),
            "previous_registered_at_utc": previous["registered_at_utc"],
            **{key: previous[key] for key in ("models", "tapes", "loads", "arrival_seeds", "arrival_horizon", "observation_end", "method", "budget", "clock", "performance")},
            "diagnostic_zero_tolerance_s": 1e-9,
            "diagnostic_rule": "Every original development tape is replayed. Use the unchanged clearing selector until its first changed decision, adopt exactly that root, then ordinary contextual MaxWeight forever. No parameter or seed selection. Validate action/time/Q/available prefix and first changed action against the full frozen clearing trajectory. Reuse original MaxWeight and full clearing outcomes; recompute three-policy common cohorts at H/T. Count all restricted-flow jobs and preserve per-class outcomes. Identify visible FIFO waiting IDs by per-class service admitted before the first changed call, including earlier same-time calls; separately retain those previously admitted and future arrival IDs. Sum H/T realized restricted-flow differences in these fixed groups. No performance promotion gate.",
            "limitations": "Retrospective deterministic-tape single-intervention mechanism diagnosis on three shared development seeds, not a new candidate, independent confirmation, physical performance or asynchronous stability claim. The realized first-action effect includes later ordinary MaxWeight reactions and the fixed real arrivals; predicted zero-arrival clearing gain does not establish realized flow gain.",
            "execution_sources": {**previous["execution_sources"], SOURCE: (ROOT / SOURCE).read_text()}}


def summarize(rows, protocol):
    tolerance = protocol["diagnostic_zero_tolerance_s"]
    pooled = {cutoff: {label: {
        "restricted_mean_flow_s": mean(row["snapshots"][cutoff][label]["restricted_mean_flow_s"] for row in rows),
        "common_p90_flow_s": mean(row["snapshots"][cutoff][label]["common_p90_flow_s"] for row in rows),
        "unfinished_jobs": sum(row["snapshots"][cutoff][label]["unfinished_jobs"] for row in rows)}
        for label in LABELS} for cutoff in CUTOFFS}
    def effects(samples):
        values = [row["job_time_effects_vs_maxweight"]["observation_end"][ONE_SHOT]["restricted_mean_delta_s"] for row in samples]
        return {"tapes": len(samples), "improved_tapes": sum(value < -tolerance for value in values),
                "worse_tapes": sum(value > tolerance for value in values),
                "unchanged_tapes": sum(abs(value) <= tolerance for value in values),
                "mean_T_restricted_flow_delta_s": mean(values),
                "total_T_restricted_flow_delta_s": sum(row["job_time_effects_vs_maxweight"]["observation_end"][ONE_SHOT]["restricted_flow"]["total_delta_s"] for row in samples)}
    seeds = [{"seed": seed, **effects([row for row in rows if row["seed"] == seed])} for seed in protocol["arrival_seeds"]]
    group_deltas = {cutoff: {name: sum(row["first_change_job_groups"]["snapshots"][cutoff][name]["restricted_flow"]["total_delta_s"]
                                    for row in rows if row["first_change_job_groups"] is not None)
                            for name in ("visible_waiting", "previously_admitted", "future_arrivals")}
                    for cutoff in CUTOFFS}
    valid = all(row["prefix_and_first_action_match"] for row in rows)
    return {"experiment": protocol["experiment"], "registered_at_utc": protocol["registered_at_utc"],
            "decision": "FIRST_CHANGE_DIAGNOSIS_COMPLETE" if valid else "FIRST_CHANGE_DIAGNOSIS_REPLAY_HOLD",
            "execution_checks_passed": valid, "new_trajectories": len(rows), "reused_trajectories": 2 * len(rows),
            "first_change_tapes": sum(row["first_change"] is not None for row in rows),
            **effects(rows), "improved_seed_aggregates": sum(seed["mean_T_restricted_flow_delta_s"] < -tolerance for seed in seeds),
            "worse_seed_aggregates": sum(seed["mean_T_restricted_flow_delta_s"] > tolerance for seed in seeds),
            "pooled": pooled, "first_change_group_total_flow_deltas_s": group_deltas,
            "seed_aggregates": seeds, "paired_runs": rows, "limitations": protocol["limitations"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--register", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    previous = json.loads(PREVIOUS.with_name(PREVIOUS.name + "_protocol.json").read_text())
    path = args.output.with_name(args.output.name + "_protocol.json")
    if args.register:
        with path.open("x") as out:
            json.dump({**build_protocol(previous), "registered_at_utc": datetime.now(timezone.utc).isoformat()}, out, separators=(",", ":"), allow_nan=False)
        print(path, flush=True)
        return
    protocol = json.loads(path.read_text())
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != build_protocol(previous):
        raise ValueError("registered first-change diagnostic or source development changed")
    output = args.output.with_suffix(".json")
    if output.exists():
        raise FileExistsError("preserve existing diagnostic results")
    with gzip.open(ROOT / protocol["previous_trajectories"], "rt") as source:
        original = json.load(source)
    reused = {run["tape_id"]: run["policies"] for run in original["runs"]}
    if (original["registered_at_utc"] != protocol["previous_registered_at_utc"]
            or len(reused) != len(original["runs"])
            or set(reused) != {tape["tape_id"] for tape in protocol["tapes"]}):
        raise ValueError("reused development trajectories differ from registered tapes")
    models = {model.node: model for model in load_models(protocol["models"])}
    rows = []
    for tape in protocol["tapes"]:
        row = run_tape(models[tape["node"]], tape, protocol, reused[tape["tape_id"]])
        rows.append(row)
        print(f"{tape['tape_id']}: first-change={'none' if row['first_change'] is None else row['first_change']['frame']} prefix={row['prefix_and_first_action_match']}", flush=True)
    result = summarize(rows, protocol)
    with output.open("x") as out:
        json.dump(result, out, separators=(",", ":"), allow_nan=False)
    print(json.dumps({key: result[key] for key in ("decision", "first_change_tapes", "improved_tapes", "worse_tapes", "unchanged_tapes", "mean_T_restricted_flow_delta_s", "pooled")}), flush=True)


if __name__ == "__main__":
    main()
