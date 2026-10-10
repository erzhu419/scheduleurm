"""Retained serial rows and new parallel rows keep the original scientific gate."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from algorithm.experiments import aligned_hardware_parallel_root_sharded as parallel
from skill.tests.test_exact_reuse_handoff import scientific_rows, completed_handoff


def test_parallel_distribution_freezes_completed_origins_and_keeps_science_once(
        completed_handoff, scientific_rows, monkeypatch):
    directory, old_manifest, documents, distributions = completed_handoff
    scientific, rows = scientific_rows
    for distribution in distributions.values():
        distribution["reference_registrations"] = {"ordinary_BG": "bg", "frozen_current_parent": "parent"}
    for index in (3, 6):
        Path(old_manifest["assignments"][index]["row_path"]).unlink()
    benchmark_registration = {"registered_at_utc": "parallel-benchmark"}
    monkeypatch.setattr(parallel, "load_benchmark", lambda path: (benchmark_registration, {}))
    for index in range(4):
        (directory / f"{parallel.STEM}_anchor_{index:02d}.json").write_text(json.dumps({
            "parallel_benchmark_registered_at_utc": "parallel-benchmark", "anchor_index": index,
            "checks": {"exact": True}}))
    registration = json.loads(parallel.register(directory).read_text())
    assert [row["shard_index"] for row in registration["assignments"] if row["origin"] == "parallel"] == [3, 6]
    assert registration["workers"] == 22
    assert registration["assignments"][4]["origin"] == "optimized"
    # A serial result arriving after registration cannot replace its selected parallel execution.
    late = Path(old_manifest["assignments"][3]["row_path"])
    late.write_text(json.dumps(documents["optimized", 3]))
    for index in (3, 6):
        assignment = registration["assignments"][index]
        row_path = Path(assignment["row_path"])
        row_path.parent.mkdir(exist_ok=True)
        document = deepcopy(documents["optimized", index])
        document["distribution_registered_at_utc"] = registration["distribution_registered_at_utc"]
        document["row"]["executed_reuse_work"]["parallel_root_work"] = {
            "submitted_chunks": 20, "included_chunks": 18, "discarded_chunks": 2, "worker_cpu_s": 1.5}
        row_path.write_text(json.dumps(document))
    final_row = Path(registration["assignments"][6]["row_path"])
    saved = final_row.read_text()
    final_row.unlink()
    with pytest.raises(FileNotFoundError):
        parallel.merge(directory)
    assert not (directory / (parallel.STEM + ".json")).exists()
    final_row.write_text(saved)
    actual = json.loads(parallel.merge(directory).read_text())
    expected = parallel.original.summarize_rows(rows, scientific)
    assert {key: actual[key] for key in expected} == expected
    assert actual["completed_origin_counts"] == {"original": 7, "optimized": 15, "parallel": 2}
    assert actual["total_executed_trajectories"] == 48
    assert actual["parallel_executed_reuse_work"]["parallel_root_work"] == {
        "submitted_chunks": 40, "included_chunks": 36, "discarded_chunks": 4, "worker_cpu_s": 3.0}
    assert "parallel_root_work" not in actual["serial_executed_reuse_work"]
    assert all("executed_reuse_work" not in row for row in actual["paired_runs"])
