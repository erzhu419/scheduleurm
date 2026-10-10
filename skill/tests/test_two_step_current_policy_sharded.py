"""Tape sharding preserves the frozen roster, scientific summaries, and native fees."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import algorithm.experiments.aligned_hardware_two_step_current_policy_sharded as module
from algorithm.experiments.arrival_aware_lane_rollout_summary import summarize


ARTIFACTS = Path(__file__).resolve().parents[2] / "md/experiment_artifacts"


@pytest.fixture(scope="module")
def registered_rows():
    protocol = json.loads((ARTIFACTS / "aligned_hardware_two_step_current_policy_bg_incumbent_development_20261001_protocol.json").read_text())
    previous = json.loads((ARTIFACTS / "aligned_hardware_two_step_bg_incumbent_original_risk_development_20261001.json").read_text())
    by_id = {row["tape_id"]: row for row in previous["paired_runs"]}
    rows = [deepcopy(by_id[tape["tape_id"]]) for tape in protocol["tapes"]]
    assert len(rows) == 24
    # Reuse small real service summaries; added work is an aggregation fixture.
    for index, row in enumerate(rows):
        row["parent_snapshots"] = deepcopy(row["snapshots"])
        row["cached_parent_checks_passed"] = True
        row["parent_middle_queue_area_s"] = row["middle_queue_area_s"]["arrival_rollout"]
        row["current_continuation_diagnostics"] = {
            "completed_second_stages": index + 1, "applied_two_model_S2_above_BG": index % 3}
        work = row["work"]["arrival_rollout"]
        work.update(inner_current_incumbent_root_option_probes=index + 1,
                    inner_current_recourse_root_option_probes=2 * (index + 1),
                    inner_current_recourse_calls=index + 1,
                    inner_current_recourse_second_calls=index,
                    controller_constructors=1, recourse_controller_constructors=2,
                    total_controller_constructors=3)
        work["total_option_probes"] += 3 * (index + 1)
    return protocol, rows


def test_sharded_summary_matches_unsharded_scientific_and_reference_aggregation(registered_rows):
    protocol, rows = registered_rows
    actual = module.summarize_rows(rows, protocol)
    direct = summarize(rows, protocol)
    references = {}
    for name, checks, area in (("reference", "cached_reference_checks_passed", "reference_middle_queue_area_s"),
                               ("parent", "cached_parent_checks_passed", "parent_middle_queue_area_s")):
        references[name] = summarize([{**row, "snapshots": row[name + "_snapshots"],
            "baseline_reproduction_passed": row["baseline_reproduction_passed"] and row[checks],
            "work": {"maxweight": {}, "arrival_rollout": row["work"]["arrival_rollout"]},
            "middle_queue_area_s": {"maxweight": row[area],
                                    "arrival_rollout": row["middle_queue_area_s"]["arrival_rollout"]}}
            for row in rows], protocol)
    for key in ("pooled", "groups", "loads", "seed_aggregates", "pooled_unfinished_checks",
                "failed_secondary_groups", "totals_work", "pooled_T_flow_improvement"):
        assert actual[key] == direct[key]
        assert actual["reference_comparison"][key] == references["reference"][key]
        assert actual["parent_comparison"][key] == references["parent"][key]
    work = actual["totals_work"]["arrival_rollout"]
    assert work["inner_current_recourse_calls"] == sum(range(1, 25))
    assert work["inner_current_incumbent_root_option_probes"] == sum(range(1, 25))
    assert work["controller_constructors"] == 24 and work["recourse_controller_constructors"] == 48
    assert work["total_controller_constructors"] == 72
    assert work["max_rollout_calls"] == max(row["work"]["arrival_rollout"]["max_rollout_calls"] for row in rows)
    assert actual["total_executed_trajectories"] == 48
    assert actual["reused_reference_trajectories"] == actual["reused_parent_trajectories"] == 24


@pytest.fixture
def completed_shards(tmp_path, monkeypatch, registered_rows):
    scientific, rows = registered_rows
    expected = module.build_protocol()
    assert expected["scientific_protocol"] == scientific
    monkeypatch.setattr(module, "build_protocol", lambda: expected)
    registration_path = module.register(tmp_path)
    registration = json.loads(registration_path.read_text())
    by_id = {row["tape_id"]: row for row in rows}
    calls = []
    unchanged_run_tape = module.frozen.run_tape

    def references(protocol, tape_id, epochs):
        assert protocol == scientific and epochs == expected["reference_registrations"]
        return ({"BG": tape_id}, {"BG_checks": tape_id},
                {"parent": tape_id}, {"parent_checks": tape_id})

    def native_pair(model, tape, protocol, bg, bg_checks, parent, parent_checks):
        index = next(index for index, item in enumerate(scientific["tapes"])
                     if item["tape_id"] == tape["tape_id"])
        assert tape == scientific["tapes"][index] and protocol == scientific
        assert model.node == tape["node"]
        assert (bg, bg_checks, parent, parent_checks) == references(protocol, tape["tape_id"], expected["reference_registrations"])
        calls.append(tape["tape_id"])
        return deepcopy(by_id[tape["tape_id"]]), {
            "maxweight": {"identity": tape["tape_id"] + ":MW"},
            "arrival_rollout": {"identity": tape["tape_id"] + ":candidate"}}

    monkeypatch.setattr(module, "load_references", references)
    monkeypatch.setattr(module.frozen, "load_models", lambda models: [SimpleNamespace(**model) for model in models])
    monkeypatch.setattr(module.frozen, "run_tape", native_pair)
    for index in reversed(range(24)):
        module.run_shard(index, tmp_path)
    return tmp_path, registration, calls, unchanged_run_tape


def test_shard_pair_roundtrip_and_merge_keep_literal_order_and_native_work_once(completed_shards, registered_rows):
    directory, registration, calls, unchanged_run_tape = completed_shards
    scientific, rows = registered_rows
    assert unchanged_run_tape.__module__ == module.frozen.__name__
    assert calls == [tape["tape_id"] for tape in reversed(scientific["tapes"])]
    shard_dir = directory / (module.STEM + "_shards")
    previous_calls = list(calls)
    assert module.run_shard(0, directory) == shard_dir / "00.json"
    assert calls == previous_calls
    for index, tape in enumerate(scientific["tapes"]):
        row_file = shard_dir / f"{index:02d}.json"
        doc = json.loads(row_file.read_text())
        assert doc["row"] == rows[index]
        assert doc["scientific_registered_at_utc"] == scientific["registered_at_utc"]
        assert doc["distribution_registered_at_utc"] == registration["distribution_registered_at_utc"]
        with gzip.open(doc["trajectory_file"], "rt") as source:
            trajectories = json.load(source)
        assert trajectories["registered_at_utc"] == scientific["registered_at_utc"]
        assert trajectories["runs"] == [{"tape_id": tape["tape_id"], "policies": {
            "maxweight": {"identity": tape["tape_id"] + ":MW"},
            "arrival_rollout": {"identity": tape["tape_id"] + ":candidate"}}}]
        # Collection filenames need not sort in scientific roster order.
        row_file.rename(shard_dir / f"part{23 - index:02d}.json")
    merged = json.loads(module.merge(directory).read_text())
    direct = module.summarize_rows(rows, scientific)
    assert {key: merged[key] for key in direct} == direct
    assert [row["tape_id"] for row in merged["paired_runs"]] == [tape["tape_id"] for tape in scientific["tapes"]]
    assert [row["tape_id"] for row in merged["trajectory_shards"]] == [tape["tape_id"] for tape in scientific["tapes"]]
    assert merged["completed_shards"] == 24 and merged["total_executed_trajectories"] == 48
    assert merged["totals_work"]["arrival_rollout"]["inner_current_recourse_calls"] == sum(range(1, 25))
    assert merged["aborted_local_run"]["reused_trajectories"] == 0


def test_unfinished_shard_with_trajectory_but_no_completion_row_cannot_merge(completed_shards):
    directory, _, _, _ = completed_shards
    shard_dir = directory / (module.STEM + "_shards")
    (shard_dir / "23.json").unlink()
    (shard_dir / "23.json.tmp").write_text('{"shard_index":23,')
    assert (shard_dir / "23_trajectories.json.gz").exists()
    with pytest.raises(ValueError, match="all24 completed shards"):
        module.merge(directory)
    assert not (directory / (module.STEM + ".json")).exists()


def test_duplicate_index_wrong_tape_or_wrong_registration_cannot_enter_aggregate(completed_shards):
    directory, registration, _, _ = completed_shards
    row_file = directory / (module.STEM + "_shards") / "01.json"
    original = json.loads(row_file.read_text())
    for key, value in (("shard_index", 0),
                       ("tape_id", registration["scientific_protocol"]["tapes"][0]["tape_id"]),
                       ("scientific_registered_at_utc", "a different scientific registration")):
        changed = {**original, key: value}
        row_file.write_text(json.dumps(changed))
        with pytest.raises(ValueError):
            module.merge(directory)
        assert not (directory / (module.STEM + ".json")).exists()
    row_file.write_text(json.dumps(original))


def test_retry_replaces_orphaned_trajectory_and_stale_temps_then_reuses_completed_work(completed_shards, registered_rows):
    directory, _, calls, _ = completed_shards
    scientific, rows = registered_rows
    shard_dir = directory / (module.STEM + "_shards")
    (shard_dir / "00.json").unlink()
    row_tmp = shard_dir / "00.json.tmp"
    trajectory_tmp = shard_dir / "00_trajectories.json.gz.tmp"
    row_tmp.write_text('{"shard_index":0,')
    trajectory_tmp.write_bytes(b"interrupted gzip write")
    before = len(calls)
    completed = module.run_shard(0, directory)
    assert len(calls) == before + 1 and calls[-1] == scientific["tapes"][0]["tape_id"]
    assert not row_tmp.exists() and not trajectory_tmp.exists()
    doc = json.loads(completed.read_text())
    assert doc["row"] == rows[0]
    with gzip.open(doc["trajectory_file"], "rt") as source:
        assert json.load(source)["runs"][0]["tape_id"] == scientific["tapes"][0]["tape_id"]
    assert module.run_shard(0, directory) == completed and len(calls) == before + 1
    merged = json.loads(module.merge(directory).read_text())
    assert merged["total_executed_trajectories"] == 48 and merged["completed_shards"] == 24
    assert merged["totals_work"]["arrival_rollout"]["inner_current_recourse_calls"] == sum(range(1, 25))
