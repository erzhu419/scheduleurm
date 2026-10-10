"""Run the frozen current-policy development experiment as 24 paired tape shards."""
import argparse
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import socket

from algorithm.experiments import aligned_hardware_two_step_current_policy_bg_incumbent_development as frozen
from algorithm.experiments.arrival_aware_lane_rollout_summary import summarize


ROOT = Path(__file__).resolve().parents[2]
STEM = "aligned_hardware_two_step_current_policy_sharded_20261001"
DEFAULT_DIR = ROOT / "md/experiment_artifacts"
SOURCE = "algorithm/experiments/aligned_hardware_two_step_current_policy_sharded.py"
LABELS = frozen.LABELS


def build_protocol():
    original_path = frozen.path(frozen.BASE, "_protocol.json")
    scientific = json.loads(original_path.read_text())
    for name, source in scientific["execution_sources"].items():
        if (ROOT / name).read_text() != source:
            raise ValueError("frozen scientific source changed: " + name)
    if len(scientific["tapes"]) != 24 or len({tape["tape_id"] for tape in scientific["tapes"]}) != 24:
        raise ValueError("all24 literal frozen tapes required")
    references = {}
    for role, key in (("ordinary_BG", "previous_result"),
                      ("frozen_current_parent", "continuation_parent_result")):
        result = json.loads((ROOT / scientific[key]).read_text())
        if not result["execution_checks_passed"]:
            raise ValueError("execution-valid cached reference required: " + role)
        references[role] = result["registered_at_utc"]
    return {"experiment": "two_step_current_policy_distributed_execution_v1",
        "scientific_protocol_path": str(original_path.relative_to(ROOT)),
        "scientific_protocol": scientific, "reference_registrations": references,
        "shard_count": 24,
        "execution": "One complete native MW/candidate pair per literal tape, unchanged frozen run_tape. Each shard writes its two native trajectories first and its completed row last. Execute the24 independent shards across node001-node006. Merge all24 completed rows in literal roster order, retaining the same service metrics, native work and MW/BG gates. Full trajectories remain in per-shard server gzip files; merge reads only rows.",
        "aborted_local_run": {"scheduler_task_id": "t116636", "scheduler_status": "cancelled",
            "completed_tapes_in_log": 3, "persisted_rows": 0, "reused_trajectories": 0,
            "partial_work_available": False,
            "accounting": "Discarded local execution is separate from the48 newly completed distributed trajectories. Its partial native work was not persisted and is not reconstructed or added to final completed-run totals."},
        "execution_sources": {SOURCE: (ROOT / SOURCE).read_text()}}


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
    if {key: value for key, value in protocol.items() if key != "distribution_registered_at_utc"} != build_protocol():
        raise ValueError("registered distribution or frozen scientific inputs changed")
    return protocol


def load_references(scientific, tape_id, reference_registrations):
    pairs = []
    for role, result_key, trajectory_key in (
            ("ordinary_BG", "previous_result", "previous_trajectories"),
            ("frozen_current_parent", "continuation_parent_result", "continuation_parent_trajectories")):
        result = json.loads((ROOT / scientific[result_key]).read_text())
        if result["registered_at_utc"] != reference_registrations[role] or not result["execution_checks_passed"]:
            raise ValueError("cached reference registration differs: " + role)
        checks = next(row for row in result["paired_runs"] if row["tape_id"] == tape_id)
        with gzip.open(ROOT / scientific[trajectory_key], "rt") as source:
            trajectories = json.load(source)
        if trajectories["registered_at_utc"] != result["registered_at_utc"]:
            raise ValueError("cached trajectory registration differs: " + role)
        policies = next(run["policies"] for run in trajectories["runs"] if run["tape_id"] == tape_id)
        pairs.extend((policies, checks))
    return tuple(pairs)


def run_shard(shard_index, artifact_dir=DEFAULT_DIR):
    artifact_dir = Path(artifact_dir)
    distribution = load_registration(artifact_dir)
    scientific = distribution["scientific_protocol"]
    if not 0 <= shard_index < len(scientific["tapes"]):
        raise ValueError("shard index must address a literal registered tape")
    shard_dir = artifact_dir / (STEM + "_shards")
    shard_dir.mkdir(exist_ok=True)
    row_path = shard_dir / f"{shard_index:02d}.json"
    trajectory_path = shard_dir / f"{shard_index:02d}_trajectories.json.gz"
    tape = scientific["tapes"][shard_index]
    if row_path.exists():
        completed = json.loads(row_path.read_text())
        if (completed["shard_index"] != shard_index or completed["tape_id"] != tape["tape_id"]
                or completed["row"]["tape_id"] != tape["tape_id"]
                or completed["scientific_registered_at_utc"] != scientific["registered_at_utc"]
                or completed["distribution_registered_at_utc"] != distribution["distribution_registered_at_utc"]):
            raise ValueError("completed shard belongs to a different registered tape")
        print(tape["tape_id"], "already persisted", flush=True)
        return row_path
    references = load_references(scientific, tape["tape_id"], distribution["reference_registrations"])
    model = next(model for model in frozen.load_models(scientific["models"]) if model.node == tape["node"])
    row, policies = frozen.run_tape(model, tape, scientific, *references)
    metadata = {"shard_index": shard_index, "tape_id": tape["tape_id"],
        "scientific_registered_at_utc": scientific["registered_at_utc"],
        "distribution_registered_at_utc": distribution["distribution_registered_at_utc"],
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


def summarize_rows(rows, protocol):
    result = summarize(rows, protocol)
    reference_rows = [{**row, "snapshots": row["reference_snapshots"],
        "baseline_reproduction_passed": row["baseline_reproduction_passed"] and row["cached_reference_checks_passed"],
        "work": {"maxweight": {}, "arrival_rollout": row["work"]["arrival_rollout"]},
        "middle_queue_area_s": {"maxweight": row["reference_middle_queue_area_s"],
                                "arrival_rollout": row["middle_queue_area_s"]["arrival_rollout"]}}
        for row in rows]
    reference = summarize(reference_rows, protocol)
    reference.update(policy_roles={"maxweight": "cached_confirmed_background_method_on_development_tapes",
                                   "arrival_rollout": "two_current_policy_BG_incumbent_candidate"},
        cached_reference_reused=True, reference_rerun=False,
        baseline_check_scope="Frozen BG passing checks plus newly executed native MW identity; no new BG reproduction.",
        work_scope="Candidate work repeats the same executed candidate accounting; cached baseline has no new execution work.")
    parent_rows = [{**row, "snapshots": row["parent_snapshots"],
        "baseline_reproduction_passed": row["baseline_reproduction_passed"] and row["cached_parent_checks_passed"],
        "work": {"maxweight": {}, "arrival_rollout": row["work"]["arrival_rollout"]},
        "middle_queue_area_s": {"maxweight": row["parent_middle_queue_area_s"],
                                "arrival_rollout": row["middle_queue_area_s"]["arrival_rollout"]}}
        for row in rows]
    parent_comparison = summarize(parent_rows, protocol)
    parent_comparison.update(
        policy_roles={"maxweight": "cached_frozen_original_risk_current_parent",
                      "arrival_rollout": "two_current_policy_BG_incumbent_candidate"},
        cached_reference_reused=True, reference_rerun=False, performance_gate_applied=False,
        baseline_check_scope="Frozen parent execution checks plus newly executed native MW identity; no parent rerun.",
        work_scope="Same executed candidate work repeated for comparison; cached parent has no new work.")
    executed = (result["execution_checks_passed"] and reference["execution_checks_passed"]
                and parent_comparison["execution_checks_passed"])
    passed = result["performance_gain_established"] and reference["performance_gain_established"]
    result["performance_vs_maxweight_established"] = result["performance_gain_established"]
    result["performance_gain_established"], result["execution_checks_passed"] = passed, executed
    result["decision"] = "TWO_STEP_CURRENT_POLICY_BG_INCUMBENT_DEVELOPMENT_" + ("REPLAY_FAIL" if not executed else "PASS" if passed else "HOLD")
    result["reference_comparison"] = reference
    result["parent_comparison"] = parent_comparison
    result["current_continuation_diagnostics"] = {
        key: sum(row["current_continuation_diagnostics"][key] for row in rows)
        for key in rows[0]["current_continuation_diagnostics"]}
    result["continuation_parent_result"] = protocol["continuation_parent_result"]
    result["birth_cohort"], result["cutoff_semantics"] = protocol["birth_cohort"], protocol["cutoff_semantics"]
    result["total_executed_trajectories"], result["reused_reference_trajectories"] = 2 * len(rows), len(rows)
    result["historical_reference_work_location"] = protocol["previous_result"]
    result["middle_queue_area_s"] = {label: sum(row["middle_queue_area_s"][label] for row in rows) for label in LABELS}
    result["reference_middle_queue_area_s"] = sum(row["reference_middle_queue_area_s"] for row in rows)
    result["parent_middle_queue_area_s"] = sum(row["parent_middle_queue_area_s"] for row in rows)
    result["reused_parent_trajectories"] = len(rows)
    return result


def merge(artifact_dir=DEFAULT_DIR):
    artifact_dir = Path(artifact_dir)
    distribution = load_registration(artifact_dir)
    scientific = distribution["scientific_protocol"]
    paths = sorted((artifact_dir / (STEM + "_shards")).glob("*.json"))
    documents = [json.loads(path.read_text()) for path in paths]
    if (len(documents) != len(scientific["tapes"])
            or {doc["shard_index"] for doc in documents} != set(range(len(scientific["tapes"])) )):
        raise ValueError("all24 completed shards exactly once required")
    by_index = {doc["shard_index"]: doc for doc in documents}
    rows, trajectory_shards = [], []
    for index, tape in enumerate(scientific["tapes"]):
        doc = by_index[index]
        if (doc["tape_id"] != tape["tape_id"] or doc["row"]["tape_id"] != tape["tape_id"]
                or doc["scientific_registered_at_utc"] != scientific["registered_at_utc"]
                or doc["distribution_registered_at_utc"] != distribution["distribution_registered_at_utc"]):
            raise ValueError("shard identity or registered execution differs")
        rows.append(doc["row"])
        trajectory_shards.append({"shard_index": index, "tape_id": tape["tape_id"],
            "execution_host": doc["execution_host"], "trajectory_file": doc["trajectory_file"]})
    result = summarize_rows(rows, scientific)
    result.update(distribution_registered_at_utc=distribution["distribution_registered_at_utc"],
        distributed_execution_protocol=str(artifact_dir / (STEM + "_protocol.json")),
        completed_shards=len(rows), trajectory_shards=trajectory_shards,
        trajectory_scope="Each complete native MW/candidate pair is retained in its server-side shard gzip. No full trajectory collection or transfer is needed for this merge.",
        aborted_local_run=distribution["aborted_local_run"])
    destination = artifact_dir / (STEM + ".json")
    with destination.open("x") as out:
        json.dump(result, out, separators=(",", ":"), allow_nan=False)
    print(json.dumps({"decision": result["decision"], "completed_shards": len(rows),
        "pooled": result["pooled"], "reference_flow_improvement": result["reference_comparison"]["pooled_T_flow_improvement"],
        "reference_failed_secondary_groups": result["reference_comparison"]["failed_secondary_groups"],
        "parent_flow_improvement": result["parent_comparison"]["pooled_T_flow_improvement"]}), flush=True)
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
