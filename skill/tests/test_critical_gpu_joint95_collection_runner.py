from __future__ import annotations

import copy
import json

import pytest

from algorithm.experiments.critical_gpu_joint95_collection_runner import (
    CALIBRATION_WAVES,
    HOLDOUT_WAVE,
    build_collection_manifest,
    local_readiness_snapshot,
    validate_manifest,
    write_registration,
)


def test_manifest_registers_exact_twenty_joint_units_and_180_observations():
    report = build_collection_manifest()

    assert report["validation"]["pass"] is True
    assert report["measurements_executed"] is False
    assert report["network_access_performed"] is False
    assert report["launch_capability"] is False
    assert len(report["empty_contract"]) == 7
    assert len(report["loaded_contract"]) == 2
    assert len(report["scored_functionals"]) == 9
    assert [row["functional"] for row in report["scored_functionals"]].count(
        "drain_completion"
    ) == 7
    assert [row["functional"] for row in report["scored_functionals"]].count(
        "target_completion_time"
    ) == 2
    assert report["score_contract"]["empty_mean_task_jct_role"].endswith(
        "not_scored"
    )
    assert report["score_contract"]["loaded_resident_overlap_service_role"].endswith(
        "not_scored"
    )
    assert [row["joint_wave"] for row in report["joint_work_items"]] == [
        *CALIBRATION_WAVES,
        HOLDOUT_WAVE,
    ]
    assert [row["split_role"] for row in report["joint_work_items"]] == [
        *(["calibration"] * 19),
        "holdout",
    ]
    assert sum(row["observation_count"] for row in report["joint_work_items"]) == 180
    assert report["coverage_contract"]["finite_sample_rank"] == 19
    assert report["historical_training_inputs"]["eligible_joint_calibration_units"] == 0
    assert report["observations"] == []


def test_semantic_audit_keeps_old_waves_predictor_only():
    audit = build_collection_manifest()["compatibility_audit"]

    assert audit["status"] == "COMPATIBLE_FOR_FROZEN_PREDICTOR_ONLY"
    assert audit["training_predictor_reuse_ready"] is True
    assert audit["old_calibration_or_holdout_units_eligible"] == 0
    assert audit["empty_contract"]["match"] is True
    assert audit["loaded_contract"]["match"] is True
    assert audit["empty_code"]["exact_code_match"] is False
    assert audit["empty_code"]["unknown_changed_paths"] == []
    assert {
        row["classification"] for row in audit["empty_code"]["findings"]
    } >= {
        "readiness_parser_changed_estimand_preserved",
        "runtime_environment_changed_estimand_preserved",
    }


def test_validation_rejects_a_partial_joint_visit():
    report = build_collection_manifest()
    broken = copy.deepcopy(report)
    broken["joint_work_items"][0]["empty_suite"]["service_cell_ids"].pop()

    validation = validate_manifest(broken)

    assert validation["pass"] is False
    assert any("empty cells differ" in error for error in validation["errors"])
    assert any("joint work items differ" in error for error in validation["errors"])

    wrong_seed = copy.deepcopy(report)
    wrong_seed["joint_work_items"][1]["loaded_suite"]["seed_base"] += 1
    assert any(
        "joint work items differ" in error
        for error in validate_manifest(wrong_seed)["errors"]
    )


def test_local_readiness_reports_four_running_bapr_tasks_and_busy_gpus(tmp_path):
    node_status = tmp_path / "nodes.json"
    node_status.write_text(
        json.dumps(
            {
                "ts": 123.0,
                "states": {
                    "gpu_b": {
                        "alive": True,
                        "gpus": [
                            {"idx": 0, "used_mb": 6000, "util_pct": 99},
                            {"idx": 1, "used_mb": 6000, "util_pct": 98},
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    queue_status = tmp_path / "queue.json"
    queue_status.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "id": f"t{index}",
                        "status": "running",
                        "project": "BAPR",
                        "node": "gpu_b",
                        "gpu_idx": index % 2,
                    }
                    for index in range(4)
                ]
            }
        ),
        encoding="utf-8",
    )

    readiness = local_readiness_snapshot(
        node_status_path=node_status,
        queue_status_path=queue_status,
    )

    assert readiness["ready"] is False
    assert readiness["active_task_count"] == 4
    assert {row["code"] for row in readiness["blockers"]} == {
        "TARGET_GPUS_NOT_EMPTY",
        "ACTIVE_SCHEDULER_TASKS_ON_TARGET_NODE",
    }


def test_registration_writes_only_the_manifest(tmp_path):
    output = tmp_path / "registered.json"
    report = build_collection_manifest()

    written = write_registration(report, output)

    assert written["registered"] is True
    assert written["measurements_executed"] is False
    assert written["observations"] == []
    assert json.loads(output.read_text(encoding="utf-8"))["joint_work_items"] == written[
        "joint_work_items"
    ]
    with pytest.raises(FileExistsError):
        write_registration(report, output)
