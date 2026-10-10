"""Evaluate the frozen24 tapes with persistent root/sample process pools."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import socket
from time import perf_counter, process_time

from algorithm.continuous_arrival_age_tail_lane_rollout import ContinuousArrivalAgeTailLaneRollout
from algorithm.parallel_root_lane_rollout import ParallelRootTwoStepCurrentPolicyBGIncumbentLaneRollout
from algorithm.experiments import aligned_hardware_exact_reuse_benchmark as benchmark
from algorithm.experiments import aligned_hardware_exact_reuse_handoff as handoff
from algorithm.experiments import aligned_hardware_exact_reuse_sharded as fast
from algorithm.experiments import aligned_hardware_two_step_current_policy_sharded as original
from algorithm.experiments.aligned_hardware_continuous_arrival_age_tail import cohort_metrics, queue_area
from algorithm.experiments.aligned_hardware_performance_necessity import replay_matches
from algorithm.experiments.observed_rate_contextual_lane_refill_replay import simulate_observed_rate_contextual_trajectory


ROOT = Path(__file__).resolve().parents[2]
STEM = "aligned_hardware_parallel_root_sharded_20261001"
DEFAULT_DIR = ROOT / "md/experiment_artifacts"
SOURCES = ("algorithm/parallel_root_lane_rollout.py",
           "algorithm/experiments/aligned_hardware_parallel_root_sharded.py")
WORKERS = 22
LABELS = fast.LABELS
check_rollout, two_step_work = fast.check_rollout, fast.two_step_work


def register_benchmark(artifact_dir=DEFAULT_DIR):
    directory = Path(artifact_dir)
    reference = benchmark.load_protocol(directory)
    registration = {"registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_registered_at_utc": reference["registered_at_utc"],
        "workers": WORKERS, "execution_sources": {name: (ROOT / name).read_text() for name in SOURCES}}
    destination = directory / (STEM + "_benchmark_protocol.json")
    with destination.open("x") as out:
        json.dump(registration, out, separators=(",", ":"), allow_nan=False)
    print(json.dumps({"registration": str(destination), "workers": WORKERS}), flush=True)
    return destination


def load_benchmark(artifact_dir):
    directory = Path(artifact_dir)
    registration = json.loads((directory / (STEM + "_benchmark_protocol.json")).read_text())
    for name, source in registration["execution_sources"].items():
        if (ROOT / name).read_text() != source:
            raise ValueError("registered parallel benchmark source changed: " + name)
    reference = benchmark.load_protocol(directory)
    if (reference["registered_at_utc"] != registration["benchmark_registered_at_utc"]
            or registration["workers"] != WORKERS):
        raise ValueError("the fixed parallel benchmark inputs changed")
    return registration, reference


def register(artifact_dir=DEFAULT_DIR):
    directory = Path(artifact_dir)
    previous = json.loads((directory / (handoff.STEM + "_protocol.json")).read_text())
    if previous["execution_source"] != (ROOT / handoff.SOURCE).read_text():
        raise ValueError("registered handoff source changed")
    origins = handoff.registrations(previous)
    registration = origins["optimized"]
    science = registration["scientific_protocol"]
    anchors, _ = load_benchmark(directory)
    anchor_results = [json.loads((directory / f"{STEM}_anchor_{index:02d}.json").read_text())
                      for index in range(4)]
    if any(result["parallel_benchmark_registered_at_utc"] != anchors["registered_at_utc"]
           or result["anchor_index"] != index or not all(result["checks"].values())
           for index, result in enumerate(anchor_results)):
        raise ValueError("four matching exact benchmark proposals required before tape execution")
    assignments = []
    for assignment, tape in zip(previous["assignments"], science["tapes"], strict=True):
        selected = dict(assignment)
        if Path(selected["row_path"]).is_file():
            handoff.read_document(selected, tape, origins[selected["origin"]])
            selected["completed_at_handoff"] = True
        else:
            selected.update(origin="parallel", completed_at_handoff=False,
                row_path=str((directory / (STEM + "_shards") / f"{selected['shard_index']:02d}.json").absolute()))
        assignments.append(selected)
    protocol = {
        "distribution_registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_protocol": science,
        "reference_registrations": registration["reference_registrations"],
        "previous_handoff_registered_at_utc": previous["registered_at_utc"],
        "legacy_origins": previous["origins"],
        "parallel_benchmark_registered_at_utc": anchors["registered_at_utc"],
        "workers": WORKERS,
        "assignments": assignments,
        "execution": "Retain completed handoff rows without reading service outcomes. Remaining tapes use persistent local process pools; each real decision submits feasible root by consecutive two-prefix-sample chunks. Keep all8 samples/cap128 and original ordered overflow accounting, physical simulator, integrity checks and performance gates. Real trajectory frames remain sequential. Completed trajectories stay on the server.",
        "accounting": "Frozen logical policy fees count only the root-major sample prefix through the first overflow. Actual worker ledgers include all submitted chunks, including speculative chunks after overflow, and constructor work. Reused original native work and completed serial optimized work retain their respective origins. Cancelled unpersisted work is unavailable.",
        "execution_sources": {name: (ROOT / name).read_text() for name in SOURCES},
    }
    destination = directory / (STEM + "_protocol.json")
    with destination.open("x") as out:
        json.dump(protocol, out, separators=(",", ":"), allow_nan=False)
    print(json.dumps({"registration": str(destination), "workers": WORKERS,
        "completed_reused": sum(row["completed_at_handoff"] for row in assignments),
        "pending_parallel": [row["shard_index"] for row in assignments if not row["completed_at_handoff"]]}), flush=True)
    return destination


def load_registration(artifact_dir=DEFAULT_DIR):
    directory = Path(artifact_dir)
    registration = json.loads((directory / (STEM + "_protocol.json")).read_text())
    for name, source in registration["execution_sources"].items():
        if (ROOT / name).read_text() != source:
            raise ValueError("registered parallel source changed: " + name)
    previous = json.loads((directory / (handoff.STEM + "_protocol.json")).read_text())
    origins = handoff.registrations(previous)
    if (registration["previous_handoff_registered_at_utc"] != previous["registered_at_utc"]
            or registration["legacy_origins"] != previous["origins"]
            or registration["scientific_protocol"] != origins["optimized"]["scientific_protocol"]
            or registration["reference_registrations"] != origins["optimized"]["reference_registrations"]
            or registration["workers"] != WORKERS):
        raise ValueError("registered parallel execution inputs changed")
    tapes = registration["scientific_protocol"]["tapes"]
    if [(row["shard_index"], row["tape_id"]) for row in registration["assignments"]] != [
            (index, tape["tape_id"]) for index, tape in enumerate(tapes)]:
        raise ValueError("the literal24 scientific roster is required once")
    return registration, origins


def run_tape(model, tape, protocol, frozen_policies, frozen_checks, parent_policies, parent_checks,
             workers=WORKERS):
    seed = f"arrival-plan-v1:{model.node}:{tape['load']}:online"
    with ParallelRootTwoStepCurrentPolicyBGIncumbentLaneRollout(
            model.options, seed, protocol["planning_samples"], protocol["planning_event_cap"], workers) as candidate:
        planners = {"maxweight": ContinuousArrivalAgeTailLaneRollout(model.options, seed,
            protocol["planning_samples"], protocol["planning_event_cap"]), "arrival_rollout": candidate}
        return summarize_tape(model, tape, protocol, frozen_policies, frozen_checks,
                              parent_policies, parent_checks, planners)


def summarize_tape(model, tape, protocol, frozen_policies, frozen_checks,
                   parent_policies, parent_checks, planners):
    def select_candidate(*context):
        decision = planners["arrival_rollout"](*context)
        counter = planners["arrival_rollout"].decision_counter
        if counter % 20 == 0:
            print(tape["tape_id"], "candidate dispatch", counter,
                  "service time", context[3], flush=True)
        return decision

    policies = {label: simulate_observed_rate_contextual_trajectory(model, tape["arrivals"],
        protocol["environment_arrival_end"], protocol["observation_end"],
        planners[label].maxweight if label == "maxweight" else select_candidate, policy=label)
        for label in LABELS}
    executed_reuse_work = planners["arrival_rollout"].reuse_work
    reference = {"maxweight": frozen_policies["arrival_rollout"], "arrival_rollout": policies["arrival_rollout"]}
    parent_reference = {"maxweight": parent_policies["arrival_rollout"],
                        "arrival_rollout": policies["arrival_rollout"]}
    class_snapshots = []
    for coordinate, name in enumerate(model.workloads):
        filtered = {label: {**result, "jobs": [job for job in result["jobs"]
                    if job["coordinate"] == coordinate]} for label, result in policies.items()}
        class_reference = {"maxweight": {**frozen_policies["arrival_rollout"], "jobs": [
            job for job in frozen_policies["arrival_rollout"]["jobs"] if job["coordinate"] == coordinate]},
            "arrival_rollout": filtered["arrival_rollout"]}
        class_parent = {"maxweight": {**parent_policies["arrival_rollout"], "jobs": [
            job for job in parent_policies["arrival_rollout"]["jobs"] if job["coordinate"] == coordinate]},
            "arrival_rollout": filtered["arrival_rollout"]}
        class_snapshots.append({"class": name, "coordinate": coordinate,
            "snapshots": {key: cohort_metrics(filtered, protocol[key], protocol["birth_cohort"])
                          for key in ("arrival_horizon", "observation_end")},
            "reference_snapshots": {key: cohort_metrics(class_reference, protocol[key], protocol["birth_cohort"])
                                    for key in ("arrival_horizon", "observation_end")},
            "parent_snapshots": {key: cohort_metrics(class_parent, protocol[key], protocol["birth_cohort"])
                                 for key in ("arrival_horizon", "observation_end")}})
    row = {
        **{key: tape[key] for key in ("tape_id", "node", "lanes", "load", "seed")},
        "baseline_reproduction_passed": replay_matches(policies["maxweight"], frozen_policies["maxweight"]),
        "rollout_checks_passed": check_rollout(policies["arrival_rollout"], model, protocol, planners["arrival_rollout"]),
        "cached_reference_checks_passed": frozen_checks["baseline_reproduction_passed"] and frozen_checks["rollout_checks_passed"],
        "cached_parent_checks_passed": parent_checks["baseline_reproduction_passed"]
            and parent_checks["rollout_checks_passed"],
        "snapshots": {key: cohort_metrics(policies, protocol[key], protocol["birth_cohort"])
                      for key in ("arrival_horizon", "observation_end")},
        "reference_snapshots": {key: cohort_metrics(reference, protocol[key], protocol["birth_cohort"])
                                for key in ("arrival_horizon", "observation_end")},
        "parent_snapshots": {key: cohort_metrics(parent_reference, protocol[key], protocol["birth_cohort"])
                             for key in ("arrival_horizon", "observation_end")},
        "class_snapshots": class_snapshots,
        "current_continuation_diagnostics": {
            "completed_second_stages": sum(frame["rollout"]["candidate_count"] > 0
                and not frame["rollout"]["planning_overflow"]
                for frame in policies["arrival_rollout"]["frames"]),
            "applied_two_model_S2_above_BG": sum(
                frame["rollout"]["chosen_predicted_squared_flow_s2"]
                > frame["rollout"]["baseline_predicted_squared_flow_s2"]
                for frame in policies["arrival_rollout"]["frames"]
                if frame["rollout"]["candidate_count"] > 0
                and not frame["rollout"]["planning_overflow"]),
        },
        "work": {label: two_step_work(result, planners[label], protocol["arrival_horizon"])
                 for label, result in policies.items()},
        "middle_queue_area_s": {label: queue_area(result) for label, result in policies.items()},
        "reference_middle_queue_area_s": queue_area(frozen_policies["arrival_rollout"]),
        "parent_middle_queue_area_s": queue_area(parent_policies["arrival_rollout"])}
    row["executed_reuse_work"] = executed_reuse_work
    return row, policies



def run_shard(shard_index, artifact_dir=DEFAULT_DIR):
    directory = Path(artifact_dir)
    registration, _ = load_registration(directory)
    science = registration["scientific_protocol"]
    tape = science["tapes"][shard_index]
    assignment = registration["assignments"][shard_index]
    if assignment["origin"] != "parallel":
        raise ValueError("completed legacy tapes must be reused, not rerun")
    row_path = Path(assignment["row_path"])
    row_path.parent.mkdir(exist_ok=True)
    if row_path.exists():
        handoff.read_document(assignment, tape, registration)
        print(tape["tape_id"], "already persisted", flush=True)
        return row_path
    references = original.load_references(science, tape["tape_id"], registration["reference_registrations"])
    model = next(model for model in fast.frozen.load_models(science["models"]) if model.node == tape["node"])
    row, policies = run_tape(model, tape, science, *references, workers=registration["workers"])
    metadata = {"shard_index": shard_index, "tape_id": tape["tape_id"],
        "scientific_registered_at_utc": science["registered_at_utc"],
        "distribution_registered_at_utc": registration["distribution_registered_at_utc"],
        "execution_host": socket.gethostname()}
    trajectory = row_path.with_name(f"{shard_index:02d}_trajectories.json.gz")
    trajectory_tmp = trajectory.with_name(trajectory.name + ".tmp")
    with gzip.open(trajectory_tmp, "wt") as out:
        json.dump({**metadata, "registered_at_utc": science["registered_at_utc"],
            "runs": [{"tape_id": tape["tape_id"], "policies": policies}]}, out,
            separators=(",", ":"), allow_nan=False)
    trajectory_tmp.replace(trajectory)
    temporary = row_path.with_name(row_path.name + ".tmp")
    with temporary.open("w") as out:
        json.dump({**metadata, "trajectory_file": str(trajectory.absolute()), "row": row},
                  out, separators=(",", ":"), allow_nan=False)
    temporary.replace(row_path)
    print(tape["tape_id"], "persisted", row["baseline_reproduction_passed"], row["rollout_checks_passed"], flush=True)
    return row_path


def merge(artifact_dir=DEFAULT_DIR):
    directory = Path(artifact_dir)
    registration, origins = load_registration(directory)
    science = registration["scientific_protocol"]
    rows, source_counts, worker_work, serial_work, original_work = [], {}, {}, {}, []
    for assignment, tape in zip(registration["assignments"], science["tapes"], strict=True):
        role = assignment["origin"]
        document = handoff.read_document(assignment, tape,
            registration if role == "parallel" else origins[role])
        row = document["row"]
        rows.append({key: value for key, value in row.items() if key != "executed_reuse_work"})
        source_counts[role] = source_counts.get(role, 0) + 1
        if role == "parallel":
            fast.accumulate_reuse(worker_work, row["executed_reuse_work"])
        elif role == "optimized":
            fast.accumulate_reuse(serial_work, row["executed_reuse_work"])
        else:
            original_work.append({"tape_id": tape["tape_id"], "native_work": row["work"]})
    result = original.summarize_rows(rows, science)
    result.update(distribution_registered_at_utc=registration["distribution_registered_at_utc"],
        distributed_execution_protocol=str(directory / (STEM + "_protocol.json")),
        execution_origins=registration["assignments"], completed_origin_counts=source_counts,
        parallel_executed_reuse_work=worker_work, serial_executed_reuse_work=serial_work,
        original_completed_native_work=original_work, execution_accounting=registration["accounting"])
    destination = directory / (STEM + ".json")
    with destination.open("x") as out:
        json.dump(result, out, separators=(",", ":"), allow_nan=False)
    print(json.dumps({"decision": result["decision"], "execution_checks_passed": result["execution_checks_passed"],
        "performance_gain_established": result["performance_gain_established"],
        "completed_origin_counts": source_counts}), flush=True)
    return destination


def run_anchor(index, artifact_dir=DEFAULT_DIR):
    directory = Path(artifact_dir)
    registration, reference = load_benchmark(directory)
    row = reference["contexts"][index]
    tape, context = row["tape"], row["context"]
    model = next(model for model in benchmark.load_models(reference["models"]) if model.node == tape["node"])

    class RecordingParallel(ParallelRootTwoStepCurrentPolicyBGIncumbentLaneRollout):
        def score_background_roots(self, *args):
            self.last_scores = super().score_background_roots(*args)
            return self.last_scores

    outputs, timing = {}, {}
    for label in ("serial_optimized", "parallel"):
        planner = (benchmark.recording_policy(fast.ExactReuseTwoStepCurrentPolicyBGIncumbentLaneRollout,
            model, tape, reference["planning_samples"], reference["planning_event_cap"])
            if label == "serial_optimized" else RecordingParallel(model.options,
            f"arrival-plan-v1:{model.node}:{tape['load']}:online", reference["planning_samples"],
            reference["planning_event_cap"], registration["workers"]))
        planner.last_scores = None
        planner.rng.setstate(benchmark.rng_tuple(row["state"]["rng_state"]))
        planner.decision_counter = row["state"]["decision_counter"]
        wall, cpu = perf_counter(), process_time()
        try:
            decision = planner(model.options, tuple(context["queue"]), tuple(context["available_before"]),
                context["start"], context["lane_ids"][0], tuple(tuple(group) for group in context["waiting_ages"]),
                tuple(context["observed_arrival_counts"]))
            timing[label] = {"wall_s": perf_counter() - wall, "parent_cpu_s": process_time() - cpu,
                "worker_cpu_s": planner.parallel_work["worker_cpu_s"] if label == "parallel" else 0}
            timing[label]["total_cpu_s"] = timing[label]["parent_cpu_s"] + timing[label]["worker_cpu_s"]
            outputs[label] = asdict(decision), planner.last_scores, benchmark.planner_states(planner)
            if label == "parallel":
                timing[label]["actual_work"] = planner.reuse_work
        finally:
            if label == "parallel":
                planner.close()
        print(tape["tape_id"], label, timing[label]["wall_s"], flush=True)
    checks = {"decision_and_logical_work_exact": outputs["serial_optimized"][0] == outputs["parallel"][0],
        "fraction_scores_exact": outputs["serial_optimized"][1] == outputs["parallel"][1],
        "three_controller_states_exact": outputs["serial_optimized"][2] == outputs["parallel"][2]}
    result = {"parallel_benchmark_registered_at_utc": registration["registered_at_utc"],
        "anchor_index": index, "tape_id": tape["tape_id"], "checks": checks, "timing": timing,
        "wall_speedup": timing["serial_optimized"]["wall_s"] / timing["parallel"]["wall_s"],
        "cpu_multiplier": timing["parallel"]["total_cpu_s"] / timing["serial_optimized"]["total_cpu_s"],
        "scope": "Four previously fixed real contexts; complete8/cap128 proposals. No new native reference rerun. Parallel wall includes pool startup and transfer; measured CPU is parent plus executed chunk bodies including worker planner construction, excluding child process startup/import. No full-tape speedup inferred."}
    with (directory / f"{STEM}_anchor_{index:02d}.json").open("x") as out:
        json.dump(result, out, separators=(",", ":"), allow_nan=False)
    if not all(checks.values()):
        raise ValueError("parallel proposal changed a frozen result")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--register-benchmark", action="store_true")
    mode.add_argument("--register", action="store_true")
    mode.add_argument("--run-shard", type=int, choices=range(24))
    mode.add_argument("--run-anchor", type=int, choices=range(4))
    mode.add_argument("--merge", action="store_true")
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()
    if args.register_benchmark:
        register_benchmark(args.artifact_dir)
    elif args.register:
        register(args.artifact_dir)
    elif args.run_shard is not None:
        run_shard(args.run_shard, args.artifact_dir)
    elif args.run_anchor is not None:
        run_anchor(args.run_anchor, args.artifact_dir)
    else:
        merge(args.artifact_dir)


if __name__ == "__main__":
    main()
