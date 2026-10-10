"""Reassess joint scheduling against plain controls on frozen arrival tapes.

This is a development diagnosis, with known joint outcomes reused explicitly.
No parameters are selected from the new controls' outcomes. The simple exact
MaxWeight replay must reproduce every old serial action and job timestamp.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from functools import partial
import gzip
import json
from pathlib import Path
from statistics import mean

from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption
from algorithm.experiments.integer_capacity_trajectory_replay import trajectory_snapshot
from algorithm.experiments.lane_refill_performance_replay import simulate_lane_refill_trajectory
from algorithm.lane_refill_simple_controls import POLICIES, select_single_lane_control


ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "md/experiment_artifacts/aligned_hardware_lane_refill_performance_development_20260929"
DEFAULT = ROOT / "md/experiment_artifacts/aligned_hardware_performance_necessity_development_20260930"
SOURCES = ("algorithm/lane_refill_simple_controls.py", "algorithm/experiments/aligned_hardware_performance_necessity.py",
           "algorithm/experiments/lane_refill_performance_replay.py", "algorithm/experiments/integer_capacity_trajectory_replay.py")
FRAME_FIELDS = ("start", "queue", "lane_ids", "option_indices", "service", "duration")


def load_models(snapshots):
    return tuple(FrameModel(**{**row, "workloads": tuple(row["workloads"]), "scales": tuple(row["scales"]),
        "excluded_actions": tuple(row["excluded_actions"]),
        "options": tuple(FrameOption(**{**option, "requirements": tuple(option["requirements"]),
            "completion_offsets": tuple(tuple(offsets) for offsets in option["completion_offsets"]),
            "source_action_ids": tuple(option["source_action_ids"]),
            "phases": tuple(tuple(phase) for phase in option["phases"])}) for option in row["options"])}) for row in snapshots)


def protocol_body():
    previous = json.loads(OLD.with_name(OLD.name + "_protocol.json").read_text())
    result = json.loads(OLD.with_suffix(".json").read_text())
    if not result["integration_gate_passed"]:
        raise ValueError("preceding same-clock replay must be valid")
    for path in SOURCES[2:]:
        if (ROOT / path).read_text() != previous["execution_sources"][path]:
            raise ValueError(f"retain the preceding replay clock and job accounting: {path}")
    return {"experiment": "performance_necessity_development_v1",
        **{key: previous[key] for key in ("models", "tapes", "capacities", "arrival_horizon", "observation_end", "execution_order")},
        "policies": list(POLICIES), "reused_policies": ["refill_joint", "refill_single"],
        "clock": previous["clock"], "dispatch_rule": "single", "idle_duration": 1.0,
        "primary_control": "enum_maxweight", "additional_controls": ["delay", "fixed_p1"],
        "rules": {"enum_maxweight": "Full feasible single-lane catalogue; exact rational Q*d/max(1,duration), ties duration then original index; idle participates.",
                  "delay": "Full feasible catalogue; smallest arithmetic mean of all admitted job completion offsets including loaded residents; ties larger exact support, duration, index. No intentional idle when a job can run.",
                  "fixed_p1": "Dedicated nonloaded width1 options only; choose largest exact MaxWeight support, ties duration/index. Restricted configuration may have a smaller capacity region; its failure is not evidence for joint optimization."},
        "performance_rule": "Reused joint must lower pooled per-tape T all-job restricted mean by at least1% versus EACH control and must not increase mean restricted flow, common-cohort mean/P90 or unfinished count in any of16 node/load/H,T groups. Relative tolerance1e-9; no wall-time gate. All five policies use one completed-job ID intersection at each cutoff. Per-class restricted outcomes retained.",
        "reproduction_rule": "enum_maxweight must match every old refill_single job and recorded action/start/Q/lane/service/duration exactly; otherwise report REPLAY_HOLD and interpret no performance contrast.",
        "accounting": "72 new plain-control trajectories;48 explicitly reused joint/serial trajectories; all24 original development tapes, with no search or parameter sweep. Count actual option probes/candidates; no LP or integer search. Physical service clock excludes computation.",
        "limitations": "Retrospective direction diagnosis on reused development seeds and virtualR8 independent cached lanes. Shortest-mean-completion is a heuristic, not a flow oracle. Fixed p1 has restricted actions/capacity. No new policy, independent confirmation, asynchronous stability or production deployment claim.",
        "sources": {path: (ROOT / path).read_text() for path in SOURCES}}


def replay_matches(new, old):
    return (new["jobs"] == old["jobs"] and len(new["frames"]) == len(old["frames"])
            and all(all(json.loads(json.dumps(a[key])) == b[key] for key in FRAME_FIELDS)
                    for a, b in zip(new["frames"], old["frames"], strict=True)))


def common_metrics(policies, cutoff):
    done = [{(job["coordinate"], job["index"]): job for job in result["jobs"]
             if job["completion"] is not None and job["completion"] <= cutoff}
            for result in policies.values()]
    cohort = set.intersection(*(set(jobs) for jobs in done))
    if not cohort:
        raise ValueError("registered performance contrast needs a nonempty common completed cohort")
    return {label: {**trajectory_snapshot(result["jobs"], cutoff,
                                          len(result["snapshots"]["observation_end"]["per_coordinate"])),
                   "common_mean_flow_s": mean(jobs[job]["completion"] - jobs[job]["arrival"] for job in cohort),
                   "common_p90_flow_s": trajectory_snapshot([jobs[job] for job in cohort], cutoff, 3)["completed_p90_flow_s"],
                   "common_jobs": len(cohort)}
            for (label, result), jobs in zip(policies.items(), done, strict=True)}


def summarize(rows, protocol, reproduction, structure):
    if {row["tape_id"] for row in rows} != {tape["tape_id"] for tape in protocol["tapes"]} or len(rows) != len(protocol["tapes"]):
        raise ValueError("all registered tapes must appear exactly once")
    groups = defaultdict(list)
    for row in rows:
        for cutoff in ("arrival_horizon", "observation_end"):
            groups[row["node"], row["load"], cutoff].append(row["snapshots"][cutoff])
    comparisons = []
    metrics = ("restricted_mean_flow_s", "common_mean_flow_s", "common_p90_flow_s")
    for (node, load, cutoff), samples in sorted(groups.items()):
        stats = {label: {**{key: mean(sample[label][key] for sample in samples) for key in metrics},
                         "unfinished_jobs": sum(sample[label]["unfinished_jobs"] for sample in samples)}
                 for label in (*protocol["reused_policies"], *POLICIES)}
        contrasts = {label: {"passed": all(stats["refill_joint"][key] <= stats[label][key] * (1 + 1e-9) for key in metrics)
                                       and stats["refill_joint"]["unfinished_jobs"] <= stats[label]["unfinished_jobs"],
                            "flow_ratio": stats["refill_joint"]["restricted_mean_flow_s"] / stats[label]["restricted_mean_flow_s"]}
                     for label in POLICIES}
        comparisons.append({"node": node, "load": load, "cutoff": cutoff, "policies": stats, "joint_vs_control": contrasts})
    if len(comparisons) != 16 or any(len(samples) != 3 for samples in groups.values()):
        raise ValueError("retain all16 node/load/H,T groups with three paired seeds")
    pooled = {label: {"restricted_mean_flow_s": mean(row["snapshots"]["observation_end"][label]["restricted_mean_flow_s"] for row in rows),
                     "common_p90_flow_s": mean(row["snapshots"]["observation_end"][label]["common_p90_flow_s"] for row in rows),
                     "unfinished_H": sum(row["snapshots"]["arrival_horizon"][label]["unfinished_jobs"] for row in rows),
                     "unfinished_T": sum(row["snapshots"]["observation_end"][label]["unfinished_jobs"] for row in rows)}
              for label in (*protocol["reused_policies"], *POLICIES)}
    advantage = reproduction and all(group["joint_vs_control"][label]["passed"] for group in comparisons for label in POLICIES)
    advantage &= all(pooled["refill_joint"]["restricted_mean_flow_s"] <= .99 * pooled[label]["restricted_mean_flow_s"] for label in POLICIES)
    fixed_loads = []
    for cell in protocol["capacities"]:
        model = next(model for model in protocol["models"] if model["node"] == cell["node"])
        durations = [min(max(1.0, option["duration"]) for option in model["options"]
                         if not option["loaded"] and sum(option["requirements"]) == 1 and option["requirements"][i] == 1)
                     for i in range(len(model["workloads"]))]
        offered = sum(rate * duration for rate, duration in zip(cell["certificate"]["arrival_rates"], durations, strict=True)) / cell["lanes"]
        fixed_loads.append({"node": cell["node"], "load_on_full_catalogue": cell["load"], "fixed_p1_offered_lane_load": offered})
    return {"experiment": protocol["experiment"], "registered_at_utc": protocol["registered_at_utc"],
            "decision": "REPLAY_HOLD" if not reproduction else "JOINT_PERFORMANCE_DEVELOPMENT_ADVANCE" if advantage else "JOINT_PERFORMANCE_NECESSITY_HOLD",
            "serial_reproduction_passed": reproduction, "joint_advantage_established": advantage,
            "tapes": len(rows), "new_trajectories": len(rows) * len(POLICIES), "reused_trajectories": len(rows) * 2,
            "structure": structure, "pooled_T": pooled, "groups": comparisons, "paired_runs": rows,
            "fixed_p1_capacity_diagnostic": fixed_loads,
            "limitations": protocol["limitations"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--register", action="store_true")
    args = parser.parse_args()
    base = args.output
    protocol_path = base.with_name(base.name + "_protocol.json")
    if args.register:
        payload = {**protocol_body(), "registered_at_utc": datetime.now(timezone.utc).isoformat()}
        with protocol_path.open("x") as out:
            json.dump(payload, out, separators=(",", ":"), allow_nan=False)
        print(protocol_path, flush=True)
        return
    protocol = json.loads(protocol_path.read_text())
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != protocol_body():
        raise ValueError("registered plain controls, sources or original tapes changed")
    output, trajectories = base.with_suffix(".json"), base.with_name(base.name + "_trajectories.json.gz")
    if output.exists() or trajectories.exists():
        raise FileExistsError("preserve existing development results")
    models = {model.node: model for model in load_models(protocol["models"])}
    all_runs, rows, reproduction, structure = [], [], True, Counter()
    for node, lanes in protocol["execution_order"]:
        path = OLD.with_name(f"{OLD.name}_cell_{node}_r{lanes}.json.gz")
        with gzip.open(path, "rt") as source:
            old = {run["tape_id"]: run for run in json.load(source)["cell"]["runs"]}
        for tape in (tape for tape in protocol["tapes"] if tape["node"] == node):
            prior = old[tape["tape_id"]]
            policies = {label: prior["policies"][label] for label in protocol["reused_policies"]}
            work = {}
            for policy in POLICIES:
                result, selections = simulate_lane_refill_trajectory(replace(models[node], lanes=lanes), tape["arrivals"],
                    protocol["arrival_horizon"], protocol["observation_end"], 0,
                    partial(select_single_lane_control, policy=policy), policy=policy, dispatch_rule="single")
                policies[policy] = result
                work[policy] = {"cold_calls": len(selections), "option_probes": sum(s.option_probes for s in selections),
                                "candidate_evaluations": sum(s.candidate_evaluations for s in selections), "lp_calls": 0, "integer_states": 0}
            matched = replay_matches(policies["enum_maxweight"], policies["refill_single"])
            reproduction &= matched
            frames = policies["refill_joint"]["frames"]
            structure.update(calls=len(frames), single_lane_calls=sum(f["lanes_called"] == 1 for f in frames),
                             service_calls=sum(any(f["service"]) for f in frames),
                             multi_lane_service_calls=sum(f["lanes_called"] > 1 and any(f["service"]) for f in frames))
            snapshots = {label: common_metrics(policies, protocol[label]) for label in ("arrival_horizon", "observation_end")}
            rows.append({**{key: tape[key] for key in ("tape_id", "node", "load", "seed")},
                         "serial_reproduction_passed": matched, "snapshots": snapshots, "work": work})
            all_runs.append({"tape_id": tape["tape_id"], "policies": {label: policies[label] for label in POLICIES}})
            print(f"{tape['tape_id']}: three plain controls complete; serial identity={matched}", flush=True)
    with gzip.open(trajectories, "xt") as out:
        json.dump({"registered_at_utc": protocol["registered_at_utc"], "runs": all_runs}, out, separators=(",", ":"), allow_nan=False)
    result = summarize(rows, protocol, reproduction, dict(structure))
    with output.open("x") as out:
        json.dump(result, out, separators=(",", ":"), allow_nan=False)
    print(json.dumps({key: result[key] for key in ("decision", "serial_reproduction_passed", "pooled_T")}), flush=True)


if __name__ == "__main__":
    main()
