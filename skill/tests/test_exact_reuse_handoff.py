"""Mixed completed execution origins retain one unchanged scientific row per tape."""
from copy import deepcopy
import gzip
import json
from pathlib import Path

import pytest

from algorithm.experiments import aligned_hardware_exact_reuse_handoff as handoff


ARTIFACTS = Path(__file__).resolve().parents[2] / "md/experiment_artifacts"
OLD_INDICES = (0, 1, 2, 5, 10, 16, 22)


@pytest.fixture(scope="module")
def scientific_rows():
    protocol = json.loads((ARTIFACTS / "aligned_hardware_two_step_current_policy_bg_incumbent_development_20261001_protocol.json").read_text())
    cached = json.loads((ARTIFACTS / "aligned_hardware_two_step_bg_incumbent_original_risk_development_20261001.json").read_text())
    by_id = {row["tape_id"]: row for row in cached["paired_runs"]}
    rows = [deepcopy(by_id[tape["tape_id"]]) for tape in protocol["tapes"]]
    # Retain real small service summaries; these added fields are merge-only inputs.
    for index, row in enumerate(rows):
        row["parent_snapshots"] = deepcopy(row["snapshots"])
        row["cached_parent_checks_passed"] = True
        row["parent_middle_queue_area_s"] = row["middle_queue_area_s"]["arrival_rollout"]
        row["current_continuation_diagnostics"] = {
            "completed_second_stages": index + 1, "applied_two_model_S2_above_BG": index % 3}
    return protocol, rows


@pytest.fixture
def completed_handoff(tmp_path, monkeypatch, scientific_rows):
    scientific, rows = scientific_rows
    directories = {role: tmp_path / role for role in ("original", "optimized")}
    modules = {"original": handoff.original, "optimized": handoff.fast}
    distributions = {role: {"scientific_protocol": scientific,
        "distribution_registered_at_utc": role + "-registered"} for role in modules}
    for role, module in modules.items():
        (directories[role] / (module.STEM + "_shards")).mkdir(parents=True)
        monkeypatch.setattr(module, "load_registration",
                            lambda directory, role=role: distributions[role])
    documents = {}

    def write_row(role, index):
        folder = directories[role] / (modules[role].STEM + "_shards")
        trajectory = folder / f"{index:02d}_trajectories.json.gz"
        with gzip.open(trajectory, "wt") as out:
            json.dump({"fixture": "metadata-only persisted trajectory"}, out)
        row = deepcopy(rows[index])
        if role == "optimized":
            row["executed_reuse_work"] = {
                "mw_reuse_work": {"queries": 100 + index, "cache_hits": index + 1},
                "inner_decision_reuse_work": {"current_choose_calls": 20 + index,
                                              "current_cache_peak_entries": index % 5 + 1,
                                              "BG_cache_peak_entries": index % 3 + 1},
                "kernel_work": {"scorer_calls": 2 * index + 1},
            }
        document = {
            "shard_index": index, "tape_id": row["tape_id"],
            "scientific_registered_at_utc": scientific["registered_at_utc"],
            "distribution_registered_at_utc": distributions[role]["distribution_registered_at_utc"],
            "execution_host": "node001", "trajectory_file": str(trajectory), "row": row,
        }
        row_path = folder / f"{index:02d}.json"
        row_path.write_text(json.dumps(document))
        documents[role, index] = document
        return row_path

    for index in OLD_INDICES:
        write_row("original", index)
    write_row("original", 4)
    write_row("optimized", 4)
    destination = tmp_path / "handoff"
    manifest_path = handoff.register(directories["original"], directories["optimized"], destination)
    manifest = json.loads(manifest_path.read_text())
    for assignment in manifest["assignments"]:
        if not assignment["completed_at_handoff"]:
            write_row("optimized", assignment["shard_index"])
    # A late original completion must not change the registered optimized assignment.
    write_row("original", 3)
    return destination, manifest, documents, distributions


def test_mixed_handoff_preserves_exact_scientific_gates_and_separates_optimized_actual_work(
        completed_handoff, scientific_rows):
    directory, manifest, documents, _ = completed_handoff
    scientific, rows = scientific_rows
    expected = handoff.original.summarize_rows(rows, scientific)
    actual = json.loads(handoff.merge(directory).read_text())
    assert {key: actual[key] for key in expected} == expected
    assert actual["original_completed_tapes"] == 7
    assert actual["optimized_completed_tapes"] == 17
    assert actual["total_executed_trajectories"] == 48
    assert [item["tape_id"] for item in actual["execution_origins"]] == [tape["tape_id"] for tape in scientific["tapes"]]
    optimized_indices = [index for index in range(24) if index not in OLD_INDICES]
    assert [item["shard_index"] for item in actual["execution_origins"] if item["origin"] == "optimized"] == optimized_indices
    assert manifest["assignments"][4]["origin"] == "optimized"
    assert manifest["assignments"][4]["completed_at_handoff"]
    assert not manifest["assignments"][3]["completed_at_handoff"]
    ledger = actual["optimized_executed_reuse_work"]
    assert ledger["mw_reuse_work"]["queries"] == sum(100 + index for index in optimized_indices)
    assert ledger["mw_reuse_work"]["cache_hits"] == sum(index + 1 for index in optimized_indices)
    assert ledger["inner_decision_reuse_work"]["current_choose_calls"] == sum(20 + index for index in optimized_indices)
    assert ledger["inner_decision_reuse_work"]["current_cache_peak_entries"] == max(index % 5 + 1 for index in optimized_indices)
    assert ledger["inner_decision_reuse_work"]["BG_cache_peak_entries"] == max(index % 3 + 1 for index in optimized_indices)
    assert ledger["kernel_work"]["scorer_calls"] == sum(2 * index + 1 for index in optimized_indices)
    assert actual["original_completed_native_work"] == [
        {"tape_id": rows[index]["tape_id"], "native_work": rows[index]["work"]}
        for index in OLD_INDICES]
    assert all("executed_reuse_work" not in documents["original", index]["row"] for index in OLD_INDICES)
    assert all("executed_reuse_work" not in row for row in actual["paired_runs"])


def test_handoff_rejects_missing_completion_row_and_wrong_document_or_origin_registration(completed_handoff):
    directory, manifest, documents, distributions = completed_handoff
    assignment = manifest["assignments"][3]
    row_path = Path(assignment["row_path"])
    original = documents["optimized", 3]
    row_path.unlink()
    with pytest.raises(FileNotFoundError):
        handoff.merge(directory)
    assert not (directory / (handoff.STEM + ".json")).exists()
    row_path.write_text(json.dumps({**original,
        "distribution_registered_at_utc": distributions["original"]["distribution_registered_at_utc"]}))
    with pytest.raises(ValueError, match="origin identity differs"):
        handoff.merge(directory)
    assert not (directory / (handoff.STEM + ".json")).exists()
    row_path.write_text(json.dumps(original))
    distributions["optimized"]["distribution_registered_at_utc"] = "changed-optimized-registration"
    with pytest.raises(ValueError, match="origin registration changed: optimized"):
        handoff.merge(directory)
    assert not (directory / (handoff.STEM + ".json")).exists()
