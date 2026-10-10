"""Continue the fixed24 evaluation using completed native and optimized shards."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from algorithm.experiments import aligned_hardware_exact_reuse_sharded as fast
from algorithm.experiments import aligned_hardware_two_step_current_policy_sharded as original


ROOT = Path(__file__).resolve().parents[2]
STEM = "aligned_hardware_exact_reuse_handoff_20261001"
SOURCE = "algorithm/experiments/aligned_hardware_exact_reuse_handoff.py"
DEFAULT_DIR = ROOT / "md/experiment_artifacts"


def registrations(manifest):
    sources = {
        "original": original.load_registration(manifest["origins"]["original"]["artifact_dir"]),
        "optimized": fast.load_registration(manifest["origins"]["optimized"]["artifact_dir"]),
    }
    if sources["original"]["scientific_protocol"] != sources["optimized"]["scientific_protocol"]:
        raise ValueError("handoff requires the unchanged scientific protocol")
    for role, registration in sources.items():
        if registration["distribution_registered_at_utc"] != manifest["origins"][role]["distribution_registered_at_utc"]:
            raise ValueError("handoff origin registration changed: " + role)
    if sources["original"]["scientific_protocol"]["registered_at_utc"] != manifest["scientific_registered_at_utc"]:
        raise ValueError("handoff scientific registration changed")
    return sources


def read_document(assignment, tape, registration):
    document = json.loads(Path(assignment["row_path"]).read_text())
    if not fast.shard_identity_matches(document, assignment["shard_index"], tape, registration):
        raise ValueError("handoff shard or origin identity differs: " + assignment["row_path"])
    if not Path(document["trajectory_file"]).is_file():
        raise ValueError("completed row needs its persisted trajectory")
    return document


def register(original_dir, optimized_dir, artifact_dir=DEFAULT_DIR):
    directories = {"original": Path(original_dir), "optimized": Path(optimized_dir)}
    distributions = {"original": original.load_registration(original_dir),
                     "optimized": fast.load_registration(optimized_dir)}
    scientific = distributions["original"]["scientific_protocol"]
    if scientific != distributions["optimized"]["scientific_protocol"]:
        raise ValueError("the optimized execution must retain the original science")
    stems = {"original": original.STEM, "optimized": fast.STEM}
    assignments = []
    for index, tape in enumerate(scientific["tapes"]):
        paths = {role: directory / (stems[role] + "_shards") / f"{index:02d}.json"
                 for role, directory in directories.items()}
        role = "optimized" if paths["optimized"].exists() else (
            "original" if paths["original"].exists() else "optimized")
        assignment = {"shard_index": index, "tape_id": tape["tape_id"], "origin": role,
                      "row_path": str(paths[role].absolute()),
                      "completed_at_handoff": paths[role].exists()}
        if assignment["completed_at_handoff"]:
            read_document(assignment, tape, distributions[role])
        assignments.append(assignment)
    manifest = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_registered_at_utc": scientific["registered_at_utc"],
        "selection": "At handoff prefer an already completed optimized shard, otherwise a completed original shard, otherwise execute optimized. Selection uses completed rows only, without consulting service outcomes. Keep one result per literal tape.",
        "origins": {role: {"artifact_dir": str(directories[role].absolute()),
            "distribution_registered_at_utc": distribution["distribution_registered_at_utc"]}
            for role, distribution in distributions.items()},
        "assignments": assignments, "execution_source": (ROOT / SOURCE).read_text(),
        "aborted_work": "Unfinished original workers are cancelled. Their unpersisted partial work is unavailable and excluded from completed scientific and execution ledgers. Existing completed original outputs remain retained.",
    }
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    destination = artifact_dir / (STEM + "_protocol.json")
    with destination.open("x") as out:
        json.dump(manifest, out, separators=(",", ":"), allow_nan=False)
    print(json.dumps({"registration": str(destination), "pending_optimized": [
        row["shard_index"] for row in assignments if not row["completed_at_handoff"]]}), flush=True)
    return destination


def merge(artifact_dir=DEFAULT_DIR):
    artifact_dir = Path(artifact_dir)
    manifest = json.loads((artifact_dir / (STEM + "_protocol.json")).read_text())
    if (ROOT / SOURCE).read_text() != manifest["execution_source"]:
        raise ValueError("registered handoff source changed")
    distributions = registrations(manifest)
    scientific = distributions["original"]["scientific_protocol"]
    assignments = manifest["assignments"]
    if [(row["shard_index"], row["tape_id"]) for row in assignments] != [
            (index, tape["tape_id"]) for index, tape in enumerate(scientific["tapes"])]:
        raise ValueError("handoff must retain the literal scientific roster once")
    rows, origins, optimized_work, original_work = [], [], {}, []
    for assignment, tape in zip(assignments, scientific["tapes"], strict=True):
        role = assignment["origin"]
        document = read_document(assignment, tape, distributions[role])
        row = document["row"]
        rows.append({key: value for key, value in row.items() if key != "executed_reuse_work"})
        origins.append({**assignment, "execution_host": document["execution_host"],
                        "trajectory_file": document["trajectory_file"]})
        if role == "optimized":
            fast.accumulate_reuse(optimized_work, row["executed_reuse_work"])
        else:
            original_work.append({"tape_id": tape["tape_id"], "native_work": row["work"]})
    result = original.summarize_rows(rows, scientific)
    result.update(handoff_registered_at_utc=manifest["registered_at_utc"],
        handoff_protocol=str(artifact_dir / (STEM + "_protocol.json")),
        execution_origins=origins,
        original_completed_tapes=sum(row["origin"] == "original" for row in assignments),
        optimized_completed_tapes=sum(row["origin"] == "optimized" for row in assignments),
        optimized_executed_reuse_work=optimized_work, original_completed_native_work=original_work,
        execution_accounting="Scientific work counts one selected completed pair per literal tape. Original native work is retained separately; optimized executed reuse ledgers cover optimized tapes only. Other retained completed parity outputs and cancelled unpersisted executions are not part of these selected totals. No whole24 speedup is inferred from partial optimized ledgers.",
        aborted_work=manifest["aborted_work"])
    with (artifact_dir / (STEM + ".json")).open("x") as out:
        json.dump(result, out, separators=(",", ":"), allow_nan=False)
    print(json.dumps({key: result[key] for key in (
        "decision", "performance_gain_established", "execution_checks_passed",
        "original_completed_tapes", "optimized_completed_tapes")}), flush=True)
    return artifact_dir / (STEM + ".json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--register", action="store_true")
    mode.add_argument("--merge", action="store_true")
    parser.add_argument("--original-artifact-dir", type=Path)
    parser.add_argument("--optimized-artifact-dir", type=Path)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()
    if args.register:
        if args.original_artifact_dir is None or args.optimized_artifact_dir is None:
            parser.error("--register requires both original and optimized artifact directories")
        register(args.original_artifact_dir, args.optimized_artifact_dir, args.artifact_dir)
    else:
        merge(args.artifact_dir)


if __name__ == "__main__":
    main()
