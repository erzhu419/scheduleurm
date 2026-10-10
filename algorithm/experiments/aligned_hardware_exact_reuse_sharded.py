"""Evaluate exact execution reuse on the unchanged frozen24 development tapes."""
import argparse
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import socket

from algorithm.continuous_arrival_age_tail_lane_rollout import ContinuousArrivalAgeTailLaneRollout
from algorithm.exact_reuse_lane_rollout import ExactReuseTwoStepCurrentPolicyBGIncumbentLaneRollout
from algorithm.experiments import aligned_hardware_two_step_current_policy_bg_incumbent_development as frozen
from algorithm.experiments import aligned_hardware_two_step_current_policy_sharded as previous
from algorithm.experiments.aligned_hardware_continuous_arrival_age_tail import cohort_metrics, queue_area
from algorithm.experiments.aligned_hardware_performance_necessity import replay_matches
from algorithm.experiments.observed_rate_contextual_lane_refill_replay import simulate_observed_rate_contextual_trajectory


ROOT = Path(__file__).resolve().parents[2]
STEM = "aligned_hardware_exact_reuse_sharded_20261001"
DEFAULT_DIR = ROOT / "md/experiment_artifacts"
SOURCE = "algorithm/experiments/aligned_hardware_exact_reuse_sharded.py"
SOURCES = (SOURCE, previous.SOURCE, "algorithm/exact_reuse_lane_rollout.py",
           "algorithm/exact_reuse_inner_decision.py", "algorithm/dyadic_background_rollout.py")
LABELS = frozen.LABELS
check_rollout = frozen.check_rollout
two_step_work = frozen.two_step_work
ACCOUNTING = (
    "work retains frozen logical policy fees for unchanged integrity checks and comparisons. "
    "executed_reuse_work is the planner's cumulative actual reuse ledger; saved fields "
    "describe avoided logical work. Merged cache peaks use max, other counters use sum."
)


def accumulate_reuse(total, addition):
    for group, fields in addition.items():
        target = total.setdefault(group, {})
        for name, value in fields.items():
            target[name] = (max(target.get(name, 0), value) if "_peak_" in name
                            else target.get(name, 0) + value)
    return total


def build_protocol():
    parent = previous.build_protocol()
    return {
        "experiment": "exact_reuse_frozen_current_policy_distributed_execution_v1",
        **{key: parent[key] for key in ("scientific_protocol_path", "scientific_protocol",
                                      "reference_registrations", "shard_count")},
        "execution": "One native MW/exact-reuse-candidate pair for each unchanged literal tape. Reuse the original physical simulator, integrity checks, cohort metrics and MW/BG performance gates. Retain trajectories in new per-shard gzip files and merge only new completed rows. No original shard outcome or trajectory is borrowed as the new candidate execution.",
        "accounting": ACCOUNTING,
        "execution_sources": {
            **parent["scientific_protocol"]["execution_sources"],
            parent["scientific_protocol_path"]: (ROOT / parent["scientific_protocol_path"]).read_text(),
            **{name: (ROOT / name).read_text() for name in SOURCES}},
    }


def register(artifact_dir=DEFAULT_DIR):
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    protocol = {**build_protocol(), "distribution_registered_at_utc": datetime.now(timezone.utc).isoformat()}
    destination = artifact_dir / (STEM + "_protocol.json")
    with destination.open("x") as out:
        json.dump(protocol, out, separators=(",", ":"), allow_nan=False)
    return destination


def load_registration(artifact_dir):
    protocol = json.loads((Path(artifact_dir) / (STEM + "_protocol.json")).read_text())
    expected = build_protocol()
    if {key: value for key, value in protocol.items() if key != "distribution_registered_at_utc"} != expected:
        raise ValueError("registered exact-reuse sources or frozen scientific inputs changed")
    return protocol


def run_tape(model, tape, protocol, frozen_policies, frozen_checks,
             parent_policies, parent_checks):
    seed = f"arrival-plan-v1:{model.node}:{tape['load']}:online"
    planners = {
        "maxweight": ContinuousArrivalAgeTailLaneRollout(model.options, seed,
            protocol["planning_samples"], protocol["planning_event_cap"]),
        "arrival_rollout": ExactReuseTwoStepCurrentPolicyBGIncumbentLaneRollout(model.options, seed,
            protocol["planning_samples"], protocol["planning_event_cap"])}

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


def shard_identity_matches(document, index, tape, registration):
    return (document["shard_index"] == index and document["tape_id"] == tape["tape_id"]
        and document["row"]["tape_id"] == tape["tape_id"]
        and document["scientific_registered_at_utc"] == registration["scientific_protocol"]["registered_at_utc"]
        and document["distribution_registered_at_utc"] == registration["distribution_registered_at_utc"])


def run_shard(shard_index, artifact_dir=DEFAULT_DIR):
    artifact_dir = Path(artifact_dir)
    registration = load_registration(artifact_dir)
    scientific = registration["scientific_protocol"]
    if not 0 <= shard_index < len(scientific["tapes"]):
        raise ValueError("shard index must address a literal registered tape")
    directory = artifact_dir / (STEM + "_shards")
    directory.mkdir(exist_ok=True)
    row_path = directory / f"{shard_index:02d}.json"
    trajectory_path = directory / f"{shard_index:02d}_trajectories.json.gz"
    tape = scientific["tapes"][shard_index]
    if row_path.exists():
        if not shard_identity_matches(json.loads(row_path.read_text()), shard_index, tape, registration):
            raise ValueError("completed exact-reuse shard belongs to another registered tape")
        print(tape["tape_id"], "already persisted", flush=True)
        return row_path
    references = previous.load_references(scientific, tape["tape_id"], registration["reference_registrations"])
    model = next(model for model in frozen.load_models(scientific["models"]) if model.node == tape["node"])
    row, policies = run_tape(model, tape, scientific, *references)
    metadata = {"shard_index": shard_index, "tape_id": tape["tape_id"],
        "scientific_registered_at_utc": scientific["registered_at_utc"],
        "distribution_registered_at_utc": registration["distribution_registered_at_utc"],
        "execution_host": socket.gethostname()}
    trajectory_tmp = trajectory_path.with_name(trajectory_path.name + ".tmp")
    row_tmp = row_path.with_name(row_path.name + ".tmp")
    with gzip.open(trajectory_tmp, "wt") as out:
        json.dump({**metadata, "registered_at_utc": scientific["registered_at_utc"],
            "runs": [{"tape_id": tape["tape_id"], "policies": policies}]},
            out, separators=(",", ":"), allow_nan=False)
    trajectory_tmp.replace(trajectory_path)
    with row_tmp.open("w") as out:
        json.dump({**metadata, "trajectory_file": str(trajectory_path.absolute()), "row": row},
                  out, separators=(",", ":"), allow_nan=False)
    row_tmp.replace(row_path)
    print(tape["tape_id"], "persisted", row["baseline_reproduction_passed"], row["rollout_checks_passed"], flush=True)
    return row_path


def merge(artifact_dir=DEFAULT_DIR):
    artifact_dir = Path(artifact_dir)
    registration = load_registration(artifact_dir)
    scientific = registration["scientific_protocol"]
    paths = sorted((artifact_dir / (STEM + "_shards")).glob("*.json"))
    documents = [json.loads(path.read_text()) for path in paths]
    if (len(documents) != len(scientific["tapes"])
            or {doc["shard_index"] for doc in documents} != set(range(len(scientific["tapes"])))):
        raise ValueError("all24 new exact-reuse completed shards exactly once required")
    by_index = {doc["shard_index"]: doc for doc in documents}
    rows, trajectories, executed = [], [], {}
    for index, tape in enumerate(scientific["tapes"]):
        document = by_index[index]
        if not shard_identity_matches(document, index, tape, registration):
            raise ValueError("exact-reuse shard identity or registered execution differs")
        rows.append(document["row"])
        accumulate_reuse(executed, document["row"]["executed_reuse_work"])
        trajectories.append({"shard_index": index, "tape_id": tape["tape_id"],
            "execution_host": document["execution_host"], "trajectory_file": document["trajectory_file"]})
    result = previous.summarize_rows(rows, scientific)
    original_decision = result["decision"]
    result["decision"] = original_decision.replace("TWO_STEP_CURRENT_POLICY_BG_INCUMBENT", "EXACT_REUSE_TWO_STEP_CURRENT_POLICY_BG_INCUMBENT", 1)
    result.update(scientific_gate_decision=original_decision,
        distribution_registered_at_utc=registration["distribution_registered_at_utc"],
        distributed_execution_protocol=str(artifact_dir / (STEM + "_protocol.json")),
        completed_shards=len(rows), trajectory_shards=trajectories,
        trajectory_scope="New native MW/candidate trajectories are retained in this runner's shard gzip files. Merge reads rows only.",
        work_scope=ACCOUNTING, executed_reuse_work=executed,
        per_tape_executed_reuse_work=[{"tape_id": row["tape_id"], "executed_reuse_work": row["executed_reuse_work"]} for row in rows])
    for role in ("reference_comparison", "parent_comparison"):
        result[role]["work_scope"] = "Same frozen logical candidate work for comparison; cached baseline has no new execution work. Actual execution reuse is recorded once in the top-level executed_reuse_work."
    destination = artifact_dir / (STEM + ".json")
    with destination.open("x") as out:
        json.dump(result, out, separators=(",", ":"), allow_nan=False)
    print(json.dumps({"decision": result["decision"], "completed_shards": len(rows),
        "reference_flow_improvement": result["reference_comparison"]["pooled_T_flow_improvement"],
        "reference_failed_secondary_groups": result["reference_comparison"]["failed_secondary_groups"]}), flush=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--register", action="store_true")
    mode.add_argument("--run-shard", action="store_true")
    mode.add_argument("--merge", action="store_true")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()
    if args.run_shard:
        if args.shard_index is None:
            parser.error("--run-shard requires --shard-index")
        run_shard(args.shard_index, args.artifact_dir)
    elif args.shard_index is not None:
        parser.error("--shard-index is used only with --run-shard")
    elif args.register:
        print(register(args.artifact_dir), flush=True)
    else:
        merge(args.artifact_dir)


if __name__ == "__main__":
    main()
