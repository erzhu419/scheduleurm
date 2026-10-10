"""Complete frozen24 closed loops with persistent OLD-current scoring."""
import argparse
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import socket
from time import perf_counter

from algorithm.parallel_persistent_current_policy_lane_rollout import ParallelPersistentCurrentPolicyBGIncumbentLaneRollout
from algorithm.persistent_current_policy_bg_incumbent_lane_rollout import LATE_CURRENT_WORK
from algorithm.experiments import aligned_hardware_current_policy_third_root_labels as labels
from algorithm.experiments import aligned_hardware_two_step_current_policy_bg_incumbent_development as finite
from algorithm.experiments.aligned_hardware_continuous_arrival_age_tail import cohort_metrics, queue_area
from algorithm.experiments.aligned_hardware_performance_necessity import load_models
from algorithm.experiments.arrival_aware_lane_rollout_summary import summarize
from algorithm.experiments.observed_rate_contextual_lane_refill_replay import simulate_observed_rate_contextual_trajectory
from algorithm.experiments.persistent_current_policy_checks import check_rollout

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
STEM = "aligned_hardware_persistent_current_policy_development_20261002"
REFERENCE = "aligned_hardware_parallel_root_sharded_20261001"
SOURCE = str(Path(__file__).resolve().relative_to(ROOT))
SOURCES = (SOURCE, "algorithm/persistent_current_policy_bg_incumbent_lane_rollout.py",
    "algorithm/parallel_persistent_current_policy_lane_rollout.py", "algorithm/parallel_root_lane_rollout.py",
    "algorithm/experiments/persistent_current_policy_checks.py")
COMPARISONS = ("MW", "BG", "finite_current")


def path(directory, suffix):
    return Path(directory) / (STEM + suffix)


def read(file):
    return json.loads(Path(file).read_text())


def dump(file, data):
    Path(file).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def register(directory, workers):
    prior = labels.load_protocol(directory)
    completed = read(labels.path(directory, "_summary.json"))
    cached_protocol = read(Path(directory) / (REFERENCE + "_protocol.json"))
    cached_result = read(Path(directory) / (REFERENCE + ".json"))
    science = cached_protocol["scientific_protocol"]
    if (not completed["execution_checks_passed"] or completed["native_third_roots_without_labels"]
            or completed["registered_at_utc"] != prior["registered_at_utc"]
            or not cached_result["execution_checks_passed"]
            or cached_result["registered_at_utc"] != science["registered_at_utc"]
            or science["models"] != prior["models"] or science["tapes"] != prior["tapes"]
            or science["planning_samples"] != 8 or science["planning_event_cap"] != 128):
        raise ValueError("completed labels and the same execution-valid full24 finite parent required")
    if len(science["tapes"]) != 24 or [(r["shard_index"], r["tape_id"]) for r in cached_protocol["assignments"]] != [
            (i, t["tape_id"]) for i, t in enumerate(science["tapes"])]:
        raise ValueError("all24 frozen tapes and references required once")
    sources = {**prior["execution_sources"], **{name: (ROOT / name).read_text() for name in SOURCES}}
    for name, literal in sources.items():
        if (ROOT / name).read_text() != literal:
            raise ValueError("frozen source changed: " + name)
    by_id = {r["tape_id"]: r for r in cached_result["paired_runs"]}
    references = []
    for assignment in cached_protocol["assignments"]:
        doc = read(assignment["row_path"])
        row = by_id[assignment["tape_id"]]
        if (doc["tape_id"] != assignment["tape_id"]
                or doc["scientific_registered_at_utc"] != science["registered_at_utc"]
                or not all(row[k] for k in ("baseline_reproduction_passed", "rollout_checks_passed", "cached_reference_checks_passed"))
                or not Path(doc["trajectory_file"]).is_file()):
            raise ValueError("execution-valid persisted finite/MW/BG reference required")
        references.append({"tape_id": assignment["tape_id"], "row_path": assignment["row_path"],
            "trajectory_file": doc["trajectory_file"], "distribution_registered_at_utc": doc["distribution_registered_at_utc"],
            "cached_checks_passed": True})
    protocol = {k: science[k] for k in ("models", "tapes", "loads", "arrival_seeds", "planning_samples", "planning_event_cap",
        "birth_cohort", "environment_arrival_end", "arrival_horizon", "observation_end", "performance", "clock", "cutoff_semantics", "limitations")}
    protocol.update(experiment="persistent_old_current_full24_development_v1",
        registered_at_utc=datetime.now(timezone.utc).isoformat(), workers=workers,
        finite_scientific_registered_at_utc=science["registered_at_utc"],
        labels_registered_at_utc=prior["registered_at_utc"],
        prior_full24_decision=cached_result["decision"], references=references,
        BG_trajectories=science["previous_trajectories"], BG_registered_at_utc=cached_protocol["reference_registrations"]["ordinary_BG"],
        method="From time0 continuously apply persistent OLD-current native chooser on all24 literal tapes. Retain finite visible+tau_max tagged population,8 samples/cap128, original ordinary-BG S2<=MW risk set, BG-first S1 ties and every original fallback. Every nonempty virtual call uses persistent OLD-current, whose own suffix remains twoBG thenMW; no recursive new candidate. No class/root5/seed gate or parameter scan. Parallel root/two-sample chunks preserve root-major first-overflow logical accounting and final child states.",
        execution="24 newly executed candidate trajectories only. Reuse certified complete native MW, BG and finite-current references on server; no new baseline reproduction claim. One persistent process pool per trajectory; real service frames stay causal and serial. Retain new gzip trajectories on server, pull only summaries.",
        budget="All real and nested sample budgets8/cap128. Charge every first/second/late current call and its native nested work. Additional late fields are subsets, never added twice to total probes. Logical outer continuation calls are bounded by the same finite tagged-population dispatch bound, not two calls per root/sample. Separately record all physically executed parallel chunks, including work discarded after the first native overflow.",
        gate="Original MW/BG joint gate unchanged: >=1% pooled restricted mean improvement, both loads nonregress, each node/load/cutoff common P90 and unfinished <=1.05 times baseline, pooled unfinished nonincrease, zero baseline tail stays zero. Finite-current increments are reported separately with no added or retuned gate. Three shared development seeds, not24 independent replications; confirmation remains sealed.",
        execution_sources=sources)
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


def run_shard(index, directory):
    protocol = load_protocol(directory)
    tape = protocol["tapes"][index]
    controls = references(protocol, index)
    model = next(m for m in load_models(protocol["models"]) if m.node == tape["node"])
    started = perf_counter()
    with ParallelPersistentCurrentPolicyBGIncumbentLaneRollout(model.options,
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
        work = finite.two_step_work(candidate, planner, protocol["arrival_horizon"])
        work.update({name: sum(f["rollout"][name] for f in candidate["frames"]) for name in LATE_CURRENT_WORK})
        executed = planner.reuse_work
    if not valid:
        raise ValueError("persistent candidate native integrity failed: " + tape["tape_id"])
    classes = []
    for c, name in enumerate(model.workloads):
        filtered_candidate = {**candidate, "jobs": [j for j in candidate["jobs"] if j["coordinate"] == c]}
        filtered_controls = {role: {**control, "jobs": [j for j in control["jobs"] if j["coordinate"] == c]}
            for role, control in controls.items()}
        classes.append({"coordinate": c, "workload": name,
            "comparisons": snapshots(filtered_candidate, filtered_controls, protocol)})
    row = {k: tape[k] for k in ("tape_id", "node", "lanes", "load", "seed")}
    row.update(baseline_reproduction_passed=protocol["references"][index]["cached_checks_passed"],
        baseline_check_scope="Reused certified MW/BG/finite-current checks; no baseline trajectory rerun.",
        rollout_checks_passed=True, comparison_snapshots=snapshots(candidate, controls, protocol),
        class_snapshots=classes, work={"maxweight": {}, "arrival_rollout": work},
        executed_reuse_work=executed, candidate_middle_queue_area_s=queue_area(candidate),
        reference_middle_queue_area_s={role: queue_area(control) for role, control in controls.items()})
    trajectory_file = path(directory, f"_shard_{index:02d}_trajectories.json.gz")
    with gzip.open(trajectory_file, "wt") as output:
        json.dump({"registered_at_utc": protocol["registered_at_utc"],
            "runs": [{"tape_id": tape["tape_id"], "policies": {"arrival_rollout": candidate}}]},
            output, separators=(",", ":"), allow_nan=False)
    dump(path(directory, f"_shard_{index:02d}.json"), dict(index=index,
        registered_at_utc=protocol["registered_at_utc"], row=row, trajectory_file=str(trajectory_file),
        execution_host=socket.gethostname(), wall_s=perf_counter() - started))
    print(json.dumps({"index": index, "checks_passed": True, "tape_id": tape["tape_id"]}), flush=True)


def compact(result):
    return {k: v for k, v in result.items() if k not in ("paired_runs", "totals_work")}


def merge(directory):
    protocol = load_protocol(directory)
    documents = [read(path(directory, f"_shard_{i:02d}.json")) for i in range(24)]
    rows = []
    for i, doc in enumerate(documents):
        if (doc["index"] != i or doc["registered_at_utc"] != protocol["registered_at_utc"]
                or doc["row"]["tape_id"] != protocol["tapes"][i]["tape_id"]):
            raise ValueError("all24 registered new candidate shards required")
        rows.append(doc["row"])
    comparisons = {}
    for role in COMPARISONS:
        selected = [{**r, "snapshots": r["comparison_snapshots"][role],
            "middle_queue_area_s": {"maxweight": r["reference_middle_queue_area_s"][role],
                "arrival_rollout": r["candidate_middle_queue_area_s"]}} for r in rows]
        comparison = summarize(selected, protocol)
        comparison.update(policy_roles={"maxweight": "cached_" + role, "arrival_rollout": "persistent_current_candidate"},
            baseline_rerun=False, baseline_check_scope="Reuse registered execution-valid complete reference.",
            performance_gate_applied=role != "finite_current")
        comparisons[role] = comparison
    executed = all(r["execution_checks_passed"] for r in comparisons.values())
    passed = executed and all(comparisons[r]["performance_gain_established"] for r in ("MW", "BG"))
    summary = dict(decision="PERSISTENT_CURRENT_POLICY_DEVELOPMENT_" + ("REPLAY_FAIL" if not executed else "PASS" if passed else "HOLD"),
        execution_checks_passed=executed, performance_gain_established=passed,
        registered_at_utc=protocol["registered_at_utc"], completed_shards=24, newly_executed_candidate_trajectories=24,
        cached_reference_trajectories=72, prior_full24_decision=protocol["prior_full24_decision"],
        comparisons={role: compact(result) for role, result in comparisons.items()},
        finite_current_pooled_mean_improved=comparisons["finite_current"]["pooled_T_flow_improvement"] > 0,
        interpretation="Complete closed loops on the same24 development tapes, with three shared seeds. Original MW/BG joint gate decides performance; finite-current increment is reported separately. No fresh confirmation or independent24-replicate claim.")
    dump(path(directory, ".json"), {**summary, "comparisons": comparisons, "rows": rows})
    dump(path(directory, "_summary.json"), summary)
    dump(path(directory, "_records_summary.json"), {"registered_at_utc": protocol["registered_at_utc"],
        "records": [{**{k: r[k] for k in ("tape_id", "node", "lanes", "load", "seed")},
            "comparisons": {role: result["paired_runs"][i]["comparisons"] for role, result in comparisons.items()},
            "class_snapshots": r["class_snapshots"]} for i, r in enumerate(rows)]})
    print(json.dumps({"decision": summary["decision"], "finite_current_increment": comparisons["finite_current"]["pooled_T_flow_improvement"]}), flush=True)


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
