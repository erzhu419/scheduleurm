from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from algorithm.experiments import critical_gpu_joint95_v2_registration as v2


def test_v2_registers_disjoint_19_plus_1_without_evidence():
    report = v2.build_registration()

    assert report["validation"]["pass"] is True
    assert report["coverage_contract"]["calibration_joint_waves"] == list(
        range(34, 53)
    )
    assert report["coverage_contract"]["holdout_joint_wave"] == 53
    assert report["coverage_contract"]["finite_sample_rank"] == 19
    assert [row["split_role"] for row in report["joint_work_items"]] == [
        *("calibration" for _ in range(19)),
        "holdout",
    ]
    assert sum(row["observation_count"] for row in report["joint_work_items"]) == 180
    assert report["observations"] == []
    assert report["completed_joint_units"] == 0
    assert report["measurements_executed"] is False
    assert report["launch_authorized"] is False
    assert report["evidence_status"] == "NO_V2_EMPIRICAL_EVIDENCE"


def test_v2_preserves_contract_but_excludes_all_v1_joint_data():
    report = v2.build_registration()
    source = json.loads(v2.DEFAULT_V1_MANIFEST.read_text(encoding="utf-8"))

    assert report["measurement_bundles"] == source["measurement_bundles"]
    assert report["empty_contract"] == source["empty_contract"]
    assert report["loaded_contract"] == source["loaded_contract"]
    assert report["scored_functionals"] == source["scored_functionals"]
    assert report["score_contract"] == source["score_contract"]
    assert report["historical_training_inputs"] == source[
        "historical_training_inputs"
    ]
    assert report["predictor_compatibility_audit"] == source[
        "compatibility_audit"
    ]
    replacement = report["replacement_basis"]
    assert replacement["classification"] == v2.REPLACEMENT_CLASSIFICATION
    assert replacement["v1_data_interpretation"] == (
        "mechanically_terminated_not_a_stochastic_outcome"
    )
    assert replacement["v1_is_retry_parent"] is False
    assert replacement["v1_joint_units_eligible_for_v2_calibration"] == 0
    assert replacement["v1_joint_scores_eligible_for_v2_margin"] == 0
    assert report["predictor_provenance"][
        "v1_measurements_used_to_fit_or_select_predictors"
    ] is False


def test_v2_waves_and_seed_bases_are_disjoint_from_v1():
    report = v2.build_registration()
    audit = report["seed_independence"]

    assert audit["wave_identities_disjoint"] is True
    assert audit["empty_seed_bases_disjoint"] is True
    assert audit["loaded_seed_bases_disjoint"] is True
    assert set(audit["v1_joint_waves"]).isdisjoint(audit["v2_joint_waves"])
    assert set(audit["v1_empty_seed_bases"]).isdisjoint(
        audit["v2_empty_seed_bases"]
    )
    assert set(audit["v1_loaded_seed_bases"]).isdisjoint(
        audit["v2_loaded_seed_bases"]
    )
    assert [row["empty_suite"]["seed_base"] for row in report["joint_work_items"]] == audit[
        "v2_empty_seed_bases"
    ]
    assert [row["loaded_suite"]["seed_base"] for row in report["joint_work_items"]] == audit[
        "v2_loaded_seed_bases"
    ]


def test_interval_overlap_audit_is_a_required_per_measurement_launch_gate():
    report = v2.build_registration()
    guard = report["scheduler_interval_guard"]

    assert guard["required"] is True
    assert guard["audit_function"] == "_scheduler_history_interval_audit"
    assert guard["measurement_boundaries"]["empty"]["call"] == "_run_cell"
    assert guard["measurement_boundaries"]["loaded"]["call"] == (
        "_run_loaded_trajectory"
    )
    assert report["launch_preconditions"][
        "scheduler_history_interval_audit_after_each_measurement_required"
    ] is True
    assert guard["terminal_history_refresh"] == {
        "required_before_pass": True,
        "intervals": "the original seven empty and two loaded measurement intervals",
        "row_and_top_level_audits_replaced_together": True,
        "purpose": "catch scheduler tasks recorded after an immediate per-call audit",
    }
    assert "strict attempt01 result identity" in v2.V2_EXECUTOR_TEST_SCOPE
    assert "unknown JSON result rejection" in v2.V2_EXECUTOR_TEST_SCOPE
    assert (
        "terminal refresh of the original nine scheduler intervals"
        in v2.V2_EXECUTOR_TEST_SCOPE
    )
    assert all(
        row["empty_suite"][
            "scheduler_interval_audit_required_after_each_cell"
        ]
        for row in report["joint_work_items"]
    )
    assert all(
        row["loaded_suite"][
            "scheduler_interval_audit_required_after_each_scenario"
        ]
        for row in report["joint_work_items"]
    )


def test_validation_rejects_v1_data_reuse_seed_reuse_and_optional_guard():
    report = v2.build_registration()
    source = json.loads(v2.DEFAULT_V1_MANIFEST.read_text(encoding="utf-8"))

    reused = copy.deepcopy(report)
    reused["replacement_basis"][
        "v1_joint_units_eligible_for_v2_calibration"
    ] = 9
    assert v2.validate_registration(reused, v1_manifest=source)["pass"] is False

    collided = copy.deepcopy(report)
    collided["joint_work_items"][0]["empty_suite"]["seed_base"] = collided[
        "seed_independence"
    ]["v1_empty_seed_bases"][0]
    assert v2.validate_registration(collided, v1_manifest=source)["pass"] is False

    unguarded = copy.deepcopy(report)
    unguarded["scheduler_interval_guard"]["required"] = False
    assert v2.validate_registration(unguarded, v1_manifest=source)["pass"] is False


def test_write_registration_creates_only_fresh_empty_v2_state(tmp_path: Path):
    output = tmp_path / "v2_manifest.json"
    results = tmp_path / "v2_results"
    report = v2.build_registration(results_dir=results)

    registered = v2.write_registration(
        report,
        output=output,
        results_dir=results,
    )

    assert output.is_file()
    assert results.is_dir()
    assert list(results.iterdir()) == []
    assert registered["status"] == "REGISTERED_AWAITING_V2_EXECUTOR_VALIDATION"
    assert registered["registration_snapshot"]["results_file_count"] == 0
    assert registered["launch_authorized"] is False
    assert registered["measurements_executed"] is False
    with pytest.raises(FileExistsError):
        v2.write_registration(report, output=output, results_dir=results)


def test_registration_refuses_a_nonempty_results_directory(tmp_path: Path):
    output = tmp_path / "v2_manifest.json"
    results = tmp_path / "v2_results"
    results.mkdir()
    (results / "unexpected.json").write_text("{}\n", encoding="utf-8")
    report = v2.build_registration(results_dir=results)

    with pytest.raises(ValueError, match="not empty"):
        v2.write_registration(report, output=output, results_dir=results)


def test_registration_refuses_a_different_results_directory(tmp_path: Path):
    output = tmp_path / "v2_manifest.json"
    registered_results = tmp_path / "registered_results"
    different_results = tmp_path / "different_results"
    report = v2.build_registration(results_dir=registered_results)

    with pytest.raises(ValueError, match="differs"):
        v2.write_registration(
            report,
            output=output,
            results_dir=different_results,
        )
