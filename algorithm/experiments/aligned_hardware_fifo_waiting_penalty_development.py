"""Frozen full24 development of a finite FIFO waiting-penalty chooser."""
import argparse
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import socket
from time import perf_counter, process_time

from algorithm.parallel_fifo_waiting_penalty_lane_rollout import ParallelFIFOWaitingPenaltyCurrentPolicyLaneRollout
from algorithm.experiments import aligned_hardware_two_step_current_policy_bg_incumbent_development as finite
from algorithm.experiments.aligned_hardware_continuous_arrival_age_tail import cohort_metrics, queue_area
from algorithm.experiments.aligned_hardware_exact_reuse_sharded import accumulate_reuse
from algorithm.experiments.aligned_hardware_performance_necessity import load_models
from algorithm.experiments.aligned_hardware_persistent_first_divergence import CONTEXT, PREFIX, ORIGINAL
from algorithm.experiments.arrival_aware_lane_rollout_summary import summarize
from algorithm.experiments.fifo_waiting_penalty_checks import FIFO_WORK, check_rollout
from algorithm.experiments.observed_rate_contextual_lane_refill_replay import simulate_observed_rate_contextual_trajectory

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
STEM = "aligned_hardware_fifo_waiting_penalty_development_20261003"
REFERENCE = "aligned_hardware_parallel_root_sharded_20261001"
SOURCE = str(Path(__file__).resolve().relative_to(ROOT))
SOURCES = (SOURCE, "algorithm/fifo_waiting_penalty_lane_rollout.py",
    "algorithm/parallel_fifo_waiting_penalty_lane_rollout.py",
    "algorithm/exact_reuse_lane_rollout.py", "algorithm/exact_reuse_inner_decision.py",
    "algorithm/dyadic_background_rollout.py",
    "algorithm/experiments/fifo_waiting_penalty_checks.py",
    "algorithm/experiments/aligned_hardware_persistent_first_divergence.py",
    "algorithm/experiments/aligned_hardware_exact_reuse_sharded.py",
    "md/experiment_artifacts/" + STEM + "_design.json")
COMPARISONS = ("MW", "BG", "finite_current")


def path(directory, suffix):
    return Path(directory) / (STEM + suffix)


def read(file):
    return json.loads(Path(file).read_text())


def dump(file, data):
    with Path(file).open("x") as out:
        json.dump(data, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write("\n")


def register(directory, workers=13):
    cached_protocol = read(Path(directory) / (REFERENCE + "_protocol.json"))
    cached_result = read(Path(directory) / (REFERENCE + ".json"))
    science = cached_protocol["scientific_protocol"]
    if (not cached_result["execution_checks_passed"]
            or cached_result["registered_at_utc"] != science["registered_at_utc"]
            or science["planning_samples"] != 8 or science["planning_event_cap"] != 128
            or len(science["tapes"]) != 24 or workers != 13
            or [(r["shard_index"], r["tape_id"]) for r in cached_protocol["assignments"]]
                != [(i, t["tape_id"]) for i, t in enumerate(science["tapes"])]):
        raise ValueError("execution-valid frozen24 finite reference and 13 workers required")
    sources = {**science["execution_sources"], **cached_protocol["execution_sources"],
        **{name: (ROOT / name).read_text() for name in SOURCES}}
    for name, literal in sources.items():
        if (ROOT / name).read_text() != literal:
            raise ValueError("frozen source changed: " + name)
    by_id = {r["tape_id"]: r for r in cached_result["paired_runs"]}
    references = []
    for assignment in cached_protocol["assignments"]:
        doc, row = read(assignment["row_path"]), by_id[assignment["tape_id"]]
        if (doc["tape_id"] != assignment["tape_id"]
                or doc["scientific_registered_at_utc"] != science["registered_at_utc"]
                or not all(row[k] for k in ("baseline_reproduction_passed", "rollout_checks_passed",
                                            "cached_reference_checks_passed"))
                or not Path(doc["trajectory_file"]).is_file()):
            raise ValueError("certified complete finite/MW/BG reference required")
        references.append({"tape_id": assignment["tape_id"], "row_path": assignment["row_path"],
            "trajectory_file": doc["trajectory_file"],
            "distribution_registered_at_utc": doc["distribution_registered_at_utc"],
            "cached_checks_passed": True})
    protocol = {k: science[k] for k in ("models", "tapes", "loads", "arrival_seeds", "planning_samples",
        "planning_event_cap", "birth_cohort", "environment_arrival_end", "arrival_horizon", "observation_end",
        "performance", "clock", "cutoff_semantics", "limitations")}
    protocol.update(experiment="fifo_waiting_penalty_finite_full24_development_v1",
        registered_at_utc=datetime.now(timezone.utc).isoformat(), workers=workers,
        finite_scientific_registered_at_utc=science["registered_at_utc"], references=references,
        BG_trajectories=science["previous_trajectories"],
        BG_registered_at_utc=cached_protocol["reference_registrations"]["ordinary_BG"],
        prior_full24_decision=cached_result["decision"], execution_sources=sources,
        method="Change only finite parent's outer root ordering to J=S1_tag+sum(W_tag^2)/tau_max with fixed coefficient1. True S1 and S2 remain reported. Keep ordinary BG S2<=MW roots, BG-first augmented-score ties, native two OLD-current calls then MW, all original eligibility and overflow fallbacks. Eight independent paths and cap128 at every planner, no mirrored paths, class/state gate, arrival oracle or parameter sweep.",
        execution="Execute 24 new candidate trajectories from time0 to1200 on the literal development tapes. Reuse certified complete MW/BG/finite-parent references on server. Each causal trajectory has one persistent process pool with13 workers, two-sample root chunks, ascending service-lane dispatch and no computing delay. New gzip trajectories stay on server; only summaries are pulled.",
        budget="Native finite logical ledgers and partial overflow fees stay unchanged. Each consumed tagged admission adds exactly one waiting subtraction, square, tau division and penalty addition; each returned new-stage root adds one augmented-score addition. Record all physically executed worker chunks including discarded chunks, constructors and CPU separately; constructor CPU is already included in worker CPU.",
        gate="Original MW/BG joint gate stays fixed: >=1% pooled restricted mean improvement, both loads nonregress, each node/load/cutoff common P90 and unfinished <=1.05 times baseline, pooled unfinished nonincrease and zero tail stays zero. Additionally require direct finite-parent observation_end pooled restricted mean strictly improves, each load mean nonregresses, and pooled common P90 and unfinished nonregress at both900/1200. Three shared development seeds; confirmation remains sealed.",
        parent_increment_rules={"observation_end_pooled_restricted_mean": "strict_improvement",
            "each_load_observation_end_restricted_mean": "nonregression",
            "each_cutoff_pooled_common_p90": "nonregression",
            "each_cutoff_pooled_unfinished": "nonregression"})
    dump(path(directory, "_protocol.json"), protocol)
    print(json.dumps({"registered_at_utc": protocol["registered_at_utc"], "tapes": 24,
        "candidate_trajectories": 24, "workers_per_tape": workers}), flush=True)


def load_protocol(directory):
    protocol = read(path(directory, "_protocol.json"))
    for name, literal in protocol["execution_sources"].items():
        if (ROOT / name).read_text() != literal:
            raise ValueError("registered source changed: " + name)
    return protocol


def references(protocol, index):
    ref = protocol["references"][index]
    doc = read(ref["row_path"])
    if (doc["tape_id"] != ref["tape_id"] or doc["trajectory_file"] != ref["trajectory_file"]
            or doc["scientific_registered_at_utc"] != protocol["finite_scientific_registered_at_utc"]
            or doc["distribution_registered_at_utc"] != ref["distribution_registered_at_utc"]):
        raise ValueError("registered cached reference differs")
    with gzip.open(ref["trajectory_file"], "rt") as source:
        saved = json.load(source)
    if saved["registered_at_utc"] != protocol["finite_scientific_registered_at_utc"]:
        raise ValueError("finite reference trajectory registration differs")
    policies = next(r["policies"] for r in saved["runs"] if r["tape_id"] == ref["tape_id"])
    with gzip.open(ROOT / protocol["BG_trajectories"], "rt") as source:
        bg = json.load(source)
    if bg["registered_at_utc"] != protocol["BG_registered_at_utc"]:
        raise ValueError("BG reference trajectory registration differs")
    bg_policy = next(r["policies"]["arrival_rollout"] for r in bg["runs"] if r["tape_id"] == ref["tape_id"])
    return {"MW": policies["maxweight"], "BG": bg_policy, "finite_current": policies["arrival_rollout"]}


def snapshots(candidate, controls, protocol):
    return {role: {cutoff: cohort_metrics({"maxweight": control, "arrival_rollout": candidate},
        protocol[cutoff], protocol["birth_cohort"]) for cutoff in ("arrival_horizon", "observation_end")}
        for role, control in controls.items()}


def action_comparison(candidate, parent):
    """Compare equal physical contexts through the first action divergence."""
    old, new = parent["frames"], candidate["frames"]
    first = None
    for index, (a, b) in enumerate(zip(old, new)):
        ra, rb = a["rollout"], json.loads(json.dumps(b["rollout"]))
        # Cached JSON stores lists; the fresh physical simulator returns tuples.
        if (any(a[k] != json.loads(json.dumps(b[k])) for k in CONTEXT)
                or any(ra[k] != rb[k] for k in PREFIX + ORIGINAL)
                or any(ra[k] - ra["background_" + k] != rb[k] - rb["background_" + k]
                       for k in ("planning_random_draws", "planning_rate_reads"))):
            raise ValueError("causal context/prefix/original BG differs before action divergence")
        if tuple(a["option_indices"]) != tuple(b["option_indices"]):
            first = {"frame_index": index, "start": b["start"], "queue": b["queue"],
                "parent_root": a["option_indices"][0], "candidate_root": b["option_indices"][0],
                "parent_true_S1": ra["chosen_predicted_completion_s"],
                "candidate_true_S1": rb["chosen_predicted_completion_s"],
                **{k: rb[k] for k in ("baseline_predicted_augmented_score_s",
                    "chosen_predicted_augmented_score_s", "baseline_predicted_waiting_penalty_s",
                    "chosen_predicted_waiting_penalty_s")}}
            break
    if first is None and len(old) != len(new):
        raise ValueError("equal-action physical trajectories have different lengths")
    scored = [f["rollout"] for f in new if f["rollout"]["candidate_count"] and not f["rollout"]["planning_overflow"]]
    return {"first_divergence": first, "candidate_frames": len(new), "parent_frames": len(old),
        "eligible_frames": sum(f["rollout"]["eligible"] for f in new),
        "completed_new_stage_frames": len(scored),
        "chosen_true_S1_above_BG_frames": sum(r["chosen_predicted_completion_s"] > r["baseline_predicted_completion_s"] for r in scored),
        "chosen_true_S2_above_BG_frames": sum(r["chosen_predicted_squared_flow_s2"] > r["baseline_predicted_squared_flow_s2"] for r in scored),
        "scope": "Paired causal prefix through first changed action; all-frame tradeoff counts include later divergent contexts."}


def run_shard(index, directory):
    protocol = load_protocol(directory)
    tape, controls = protocol["tapes"][index], references(protocol, index)
    print(tape["tape_id"], "candidate trajectory starts at time0; observation_end", protocol["observation_end"], flush=True)
    model = next(m for m in load_models(protocol["models"]) if m.node == tape["node"])
    wall, cpu = perf_counter(), process_time()
    with ParallelFIFOWaitingPenaltyCurrentPolicyLaneRollout(model.options,
            f"arrival-plan-v1:{model.node}:{tape['load']}:online", protocol["planning_samples"],
            protocol["planning_event_cap"], workers=protocol["workers"]) as planner:
        def select(*context):
            decision = planner(*context)
            if planner.decision_counter % 20 == 0:
                print(tape["tape_id"], "candidate dispatch", planner.decision_counter,
                    "service time", context[3], flush=True)
            return decision

        candidate = simulate_observed_rate_contextual_trajectory(model, tape["arrivals"],
            protocol["environment_arrival_end"], protocol["observation_end"], select, policy="arrival_rollout")
        valid = check_rollout(candidate, model, protocol, planner)
        comparison = action_comparison(candidate, controls["finite_current"])
        work = finite.two_step_work(candidate, planner, protocol["arrival_horizon"])
        work.update({"fifo_" + name: sum(f["rollout"]["fifo_waiting_work"][name] for f in candidate["frames"])
                     for name in FIFO_WORK})
        executed = planner.reuse_work
    if not valid:
        raise ValueError("FIFO candidate native integrity failed: " + tape["tape_id"])
    classes = []
    for c, name in enumerate(model.workloads):
        filtered = {**candidate, "jobs": [j for j in candidate["jobs"] if j["coordinate"] == c]}
        filtered_controls = {role: {**control, "jobs": [j for j in control["jobs"] if j["coordinate"] == c]}
                             for role, control in controls.items()}
        classes.append({"coordinate": c, "workload": name, "comparisons": snapshots(filtered, filtered_controls, protocol)})
    row = {k: tape[k] for k in ("tape_id", "node", "lanes", "load", "seed")}
    worker_cpu = executed["parallel_root_work"].get("worker_cpu_s", 0.)
    row.update(baseline_reproduction_passed=protocol["references"][index]["cached_checks_passed"],
        baseline_check_scope="Reused certified MW/BG/finite-current checks; no baseline trajectory rerun.",
        rollout_checks_passed=True, comparison_snapshots=snapshots(candidate, controls, protocol),
        class_snapshots=classes, action_comparison=comparison,
        work={"maxweight": {}, "arrival_rollout": work}, executed_reuse_work=executed,
        controller_cpu_s=process_time() - cpu, worker_cpu_s=worker_cpu,
        candidate_middle_queue_area_s=queue_area(candidate),
        reference_middle_queue_area_s={role: queue_area(control) for role, control in controls.items()})
    row["charged_cpu_s"] = row["controller_cpu_s"] + worker_cpu
    trajectory_file = path(directory, f"_shard_{index:02d}_trajectories.json.gz")
    with gzip.open(trajectory_file, "wt") as output:
        json.dump({"registered_at_utc": protocol["registered_at_utc"],
            "runs": [{"tape_id": tape["tape_id"], "policies": {"arrival_rollout": candidate}}]},
            output, separators=(",", ":"), allow_nan=False)
    dump(path(directory, f"_shard_{index:02d}.json"), dict(index=index,
        registered_at_utc=protocol["registered_at_utc"], row=row, trajectory_file=str(trajectory_file),
        execution_host=socket.gethostname(), wall_s=perf_counter() - wall))
    print(json.dumps({"index": index, "checks_passed": True, "tape_id": tape["tape_id"]}), flush=True)


def parent_increment(result):
    pooled = result["pooled"]
    end = pooled["observation_end"]
    checks = {"pooled_mean_strictly_improved": end["arrival_rollout"]["restricted_mean_flow_s"] < end["maxweight"]["restricted_mean_flow_s"],
        "each_load_mean_nonregression": all(r["mean_nonregression_passed"] for r in result["loads"])}
    checks.update({cutoff + "_" + metric + "_nonregression": scores["arrival_rollout"][metric] <= scores["maxweight"][metric]
                  for cutoff, scores in pooled.items() for metric in ("common_p90_flow_s", "unfinished_jobs")})
    return {"passed": all(checks.values()), "checks": checks}


def compact(result):
    return {k: v for k, v in result.items() if k not in ("paired_runs", "totals_work")}


def merge(directory):
    protocol = load_protocol(directory)
    rows = []
    for i, tape in enumerate(protocol["tapes"]):
        doc = read(path(directory, f"_shard_{i:02d}.json"))
        if (doc["index"] != i or doc["registered_at_utc"] != protocol["registered_at_utc"]
                or doc["row"]["tape_id"] != tape["tape_id"]):
            raise ValueError("all24 registered candidate shards required")
        rows.append(doc["row"])
    comparisons = {}
    for role in COMPARISONS:
        selected = [{**r, "snapshots": r["comparison_snapshots"][role],
            "middle_queue_area_s": {"maxweight": r["reference_middle_queue_area_s"][role],
                "arrival_rollout": r["candidate_middle_queue_area_s"]}} for r in rows]
        comparisons[role] = summarize(selected, protocol)
        comparisons[role].update(policy_roles={"maxweight": "cached_" + role, "arrival_rollout": "fifo_waiting_penalty_candidate"},
            baseline_rerun=False, performance_gate_applied=role != "finite_current",
            parent_increment_gate_applied=role == "finite_current")
    executed = all(r["execution_checks_passed"] for r in comparisons.values())
    original_joint = executed and all(comparisons[r]["performance_gain_established"] for r in ("MW", "BG"))
    increment = parent_increment(comparisons["finite_current"])
    passed = original_joint and increment["passed"]
    reuse = {}
    for row in rows:
        accumulate_reuse(reuse, row["executed_reuse_work"])
    summary = dict(decision="FIFO_WAITING_PENALTY_DEVELOPMENT_" + ("REPLAY_FAIL" if not executed else "PASS" if passed else "HOLD"),
        execution_checks_passed=executed, performance_gain_established=passed,
        original_joint_gate_passed=original_joint, parent_increment=increment,
        registered_at_utc=protocol["registered_at_utc"], completed_shards=24,
        newly_executed_candidate_trajectories=24, cached_reference_trajectories=72,
        prior_full24_decision=protocol["prior_full24_decision"],
        comparisons={role: compact(result) for role, result in comparisons.items()},
        executed_reuse_work=reuse, charged_cpu_s=sum(r["charged_cpu_s"] for r in rows),
        fifo_logical_work={name: sum(r["work"]["arrival_rollout"]["fifo_" + name] for r in rows) for name in FIFO_WORK},
        divergent_tapes=sum(r["action_comparison"]["first_divergence"] is not None for r in rows),
        interpretation="Full closed loops on the same24 development tapes with three shared seeds. Original MW/BG joint gate and preregistered finite-parent increment both decide promotion; true S1/S2 tradeoffs and failures are retained. No fresh confirmation or independent24-replicate claim.")
    dump(path(directory, ".json"), {**summary, "comparisons": comparisons, "rows": rows})
    dump(path(directory, "_summary.json"), summary)
    dump(path(directory, "_records_summary.json"), {"registered_at_utc": protocol["registered_at_utc"],
        "records": [{**{k: r[k] for k in ("tape_id", "node", "lanes", "load", "seed")},
            "comparisons": {role: result["paired_runs"][i]["comparisons"] for role, result in comparisons.items()},
            "action_comparison": r["action_comparison"]} for i, r in enumerate(rows)]})
    print(json.dumps({"decision": summary["decision"], "original_joint_gate_passed": original_joint,
        "parent_increment": increment}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--register", action="store_true")
    mode.add_argument("--run-shard", type=int, choices=range(24))
    mode.add_argument("--merge", action="store_true")
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACTS)
    parser.add_argument("--workers", type=int, default=13)
    args = parser.parse_args()
    if args.register:
        register(args.artifact_dir, args.workers)
    elif args.run_shard is not None:
        run_shard(args.run_shard, args.artifact_dir)
    else:
        merge(args.artifact_dir)


if __name__ == "__main__":
    main()
