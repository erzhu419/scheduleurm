from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from algorithm.experiments import critical_gpu_joint95_v3_registration as v3


def test_v3_registers_fresh_19_plus_1_without_prior_data():
    report = v3.build_registration()
    assert report["validation"]["pass"] is True
    assert report["coverage_contract"]["calibration_joint_waves"] == list(range(54, 73))
    assert report["coverage_contract"]["holdout_joint_wave"] == 73
    assert report["coverage_contract"]["finite_sample_rank"] == 19
    assert [row["split_role"] for row in report["joint_work_items"]] == ["calibration"] * 19 + ["holdout"]
    assert sum(row["observation_count"] for row in report["joint_work_items"]) == 180
    assert report["observations"] == []
    assert report["completed_joint_units"] == 0
    assert report["launch_authorized"] is False
    assert report["evidence_status"] == "NO_V3_EMPIRICAL_EVIDENCE"


def test_v3_preserves_measurement_contract_and_excludes_v1_v2():
    report = v3.build_registration()
    source = json.loads(v3.DEFAULT_V1_MANIFEST.read_text(encoding="utf-8"))
    for key in ("measurement_bundles", "empty_contract", "loaded_contract", "scored_functionals", "score_contract", "historical_training_inputs"):
        assert report[key] == source[key]
    replacement = report["replacement_basis"]
    for version in ("v1", "v2"):
        assert replacement[f"{version}_is_retry_parent"] is False
        assert replacement[f"{version}_joint_units_eligible_for_v3_calibration"] == 0
        assert replacement[f"{version}_joint_scores_eligible_for_v3_margin"] == 0
    seeds = report["seed_independence"]
    for version in ("v1", "v2"):
        for suite in ("empty", "loaded"):
            assert set(seeds[f"{version}_{suite}_seed_bases"]).isdisjoint(seeds[f"v3_{suite}_seed_bases"])


def test_v3_guard_retains_unknown_and_live_prior_child_as_overlap():
    report = v3.build_registration()
    rule = report["scheduler_interval_guard"]["auto_adopted_child_rule"]
    assert rule["unknown_or_live_prior_child"] == "count as overlap and fail-stop"
    assert rule["unrelated_target_node_task"] == "count as overlap and fail-stop"
    assert "prove process exit" in rule["prior_call"]
    assert "own-child provenance and process-exit boundary audit" in v3.V3_EXECUTOR_TEST_SCOPE


def test_v3_validation_rejects_prior_data_and_guard_changes():
    source = json.loads(v3.DEFAULT_V1_MANIFEST.read_text(encoding="utf-8"))
    report = v3.build_registration()
    reused = copy.deepcopy(report)
    reused["replacement_basis"]["v2_joint_units_eligible_for_v3_calibration"] = 1
    assert v3.validate_registration(reused, v1_manifest=source)["pass"] is False
    unguarded = copy.deepcopy(report)
    unguarded["scheduler_interval_guard"]["auto_adopted_child_rule"]["prior_call"] = "always exclude"
    assert v3.validate_registration(unguarded, v1_manifest=source)["pass"] is False


def test_write_v3_registration_stays_unlaunched(tmp_path: Path):
    output = tmp_path / "manifest.json"
    results = tmp_path / "results"
    report = v3.build_registration(results_dir=results)
    registered = v3.write_registration(report, output=output, results_dir=results)
    assert registered["status"] == "REGISTERED_AWAITING_V3_EXECUTOR_VALIDATION"
    assert registered["launch_authorized"] is False
    assert list(results.iterdir()) == []
    with pytest.raises(FileExistsError):
        v3.write_registration(report, output=output, results_dir=results)
