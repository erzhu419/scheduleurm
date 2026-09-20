import json
import math
from pathlib import Path

import pytest

from algorithm.experiments import aligned_hardware_joint95_capacity_closure as closure


def _registration(results_dir: Path):
    return closure.build_registration(results_dir=results_dir)


def _write(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _visit_payload(item, predictor_anchor, score, *, started_at):
    observations = []
    for row in predictor_anchor:
        predicted = float(row["predicted_time_s"])
        actual = predicted * (1.0 + score)
        actual_score = actual / predicted - 1.0
        observations.append(
            {
                "functional_id": row["functional_id"],
                "actual_time_s": actual,
                "frozen_predicted_time_s": predicted,
                "ratio_score": actual_score,
            }
        )
    return {
        "schema_version": 1,
        "gate": "critical_gpu_joint95_registered_visit",
        "protocol": "critical_gpu_joint95_natural_completion_v1",
        "manifest_registered": True,
        "joint_unit_id": item["joint_unit_id"],
        "joint_wave": item["joint_wave"],
        "split_role": item["split_role"],
        "status": "PASS",
        "pass": True,
        "terminal": True,
        "started_at": started_at,
        "suite_order": ["empty", "loaded"],
        "measurement_bundles_unchanged": True,
        "empty_rows": [
            {
                "registered_service_cell_id": identity,
                "ready": True,
                "status": "READY",
                "capacity_boundary": False,
                "wave": item["joint_wave"],
                "split_role": item["split_role"],
            }
            for identity in item["empty_suite"]["service_cell_ids"]
        ],
        "loaded_campaign": {
            "rows": [
                {
                    "scenario_id": identity,
                    "ready": True,
                    "capacity_boundary": False,
                    "wave": item["joint_wave"],
                    "split_role": item["split_role"],
                }
                for identity in item["loaded_suite"]["scenario_ids"]
            ]
        },
        "observations": observations,
        "joint_wave_score": max(row["ratio_score"] for row in observations),
    }


def _complete_synthetic_results(protocol, *, calibration_score=-0.02, holdout_score=-0.03):
    results_dir = Path(protocol["source_paths"]["joint_results_directory"])
    manifest = json.loads(
        Path(protocol["source_paths"]["collection_manifest"]).read_text(
            encoding="utf-8"
        )
    )
    by_wave = {row["joint_wave"]: row for row in manifest["joint_work_items"]}
    calibration_paths = []
    calibration_scores = []
    for wave in closure.CALIBRATION_WAVES:
        item = by_wave[wave]
        payload = _visit_payload(
            item,
            protocol["predictor_anchor"],
            calibration_score,
            started_at=float(wave),
        )
        path = results_dir / f"{item['joint_unit_id']}_attempt01.json"
        _write(path, payload)
        calibration_paths.append(str(path))
        calibration_scores.append(payload["joint_wave_score"])
    raw_rank, margin = closure._rank19_margin(calibration_scores)
    assert raw_rank < 0.0
    assert margin == 0.0
    release = {
        "schema_version": 1,
        "gate": "critical_gpu_joint95_holdout_release",
        "protocol": "critical_gpu_joint95_natural_completion_v1",
        "status": "HOLDOUT_RELEASED",
        "released_at": 100.0,
        "holdout_wave": closure.HOLDOUT_WAVE,
        "calibration_waves": list(closure.CALIBRATION_WAVES),
        "calibration_joint_unit_ids": [
            by_wave[wave]["joint_unit_id"] for wave in closure.CALIBRATION_WAVES
        ],
        "calibration_result_paths": calibration_paths,
        "calibration_joint_wave_scores": calibration_scores,
        "finite_sample_rank": 19,
        "calibration_margin_frozen": True,
        "frozen_margin": margin,
    }
    _write(results_dir / "holdout_release.json", release)
    item = by_wave[closure.HOLDOUT_WAVE]
    holdout = _visit_payload(
        item,
        protocol["predictor_anchor"],
        holdout_score,
        started_at=101.0,
    )
    _write(
        results_dir / f"{item['joint_unit_id']}_attempt01.json",
        holdout,
    )


def test_registration_does_not_read_joint_results(monkeypatch, tmp_path):
    forbidden = (tmp_path / "joint-results").resolve()
    forbidden.mkdir()
    (forbidden / "critical_gpu_joint95_v1_gpu_b_r14_attempt01.json").write_text(
        "not-json", encoding="utf-8"
    )
    original = closure._read_object
    reads = []

    def guarded(path):
        resolved = Path(path).resolve()
        reads.append(resolved)
        if resolved == forbidden or forbidden in resolved.parents:
            raise AssertionError("registration read the joint results directory")
        return original(path)

    monkeypatch.setattr(closure, "_read_object", guarded)
    protocol = closure.build_registration(results_dir=forbidden)
    assert protocol["registration_scope"] == {
        "joint_results_directory_read": False,
        "joint_visit_values_read": False,
        "margin_computed": False,
        "trajectory_built": False,
        "selector_invoked": False,
        "server_pull_performed": False,
        "rng_draws": 0,
    }
    assert all(path != forbidden and forbidden not in path.parents for path in reads)
    assert protocol["status"] == "REGISTERED_AWAITING_JOINT95_DATA"


def test_upper_time_score_uses_reciprocal_service_transform(tmp_path):
    protocol = _registration(tmp_path / "unused-results")
    point = closure._model_from_snapshot(protocol["point_reference_model"])
    model, envelopes = closure._planned_baseline_model(
        point,
        protocol["predictor_anchor"],
        protocol["action_mapping"],
        margin=0.25,
    )
    anchor = {
        row["functional_id"]: row["predicted_time_s"]
        for row in protocol["predictor_anchor"]
    }
    by_id = {row["action_id"]: row for row in protocol["action_mapping"]}
    for option, envelope in zip(model.options, envelopes, strict=True):
        expected_duration = sum(
            anchor[functional_id] * 1.25
            for functional_id in by_id[option.action_id]["functional_ids"]
        )
        assert option.duration == pytest.approx(expected_duration)
        assert all(
            component["upper_time_s"]
            == pytest.approx(component["predicted_time_s"] * 1.25)
            for component in envelope["components"]
        )
        assert envelope["complete_job_lower_service_rates_per_s"] == pytest.approx(
            [value / max(1.0, expected_duration) for value in option.requirements]
        )
        # The registered conversion is 1/(1+m), not the invalid 1-m shortcut.
        if expected_duration > 1.0 and any(option.requirements):
            baseline_duration = expected_duration / 1.25
            baseline_rate = sum(option.requirements) / baseline_duration
            upper_rate = sum(envelope["complete_job_lower_service_rates_per_s"])
            assert upper_rate == pytest.approx(baseline_rate / 1.25)
            assert not math.isclose(upper_rate, baseline_rate * 0.75)


def test_negative_calibration_scores_do_not_enlarge_margin():
    raw, margin = closure._rank19_margin([-0.01 - index / 1000 for index in range(19)])
    assert raw < 0.0
    assert margin == 0.0
    raw, margin = closure._rank19_margin([index / 1000 for index in range(19)])
    assert raw == pytest.approx(0.018)
    assert margin == pytest.approx(0.018)


def test_old_trajectory_epsilon_is_not_compared_across_contracts(tmp_path):
    protocol = _registration(tmp_path / "unused-results")
    point = closure._model_from_snapshot(protocol["point_reference_model"])
    model, _ = closure._planned_baseline_model(
        point,
        protocol["predictor_anchor"],
        protocol["action_mapping"],
        margin=0.0,
    )
    capacity = closure._capacity_analysis(protocol, model)
    audit = closure._candidate_support_audit(protocol, model, capacity)
    assert audit["status"] == "CANDIDATE_SUPPORT_EVIDENCE_UNAVAILABLE"
    assert audit["numeric_comparison_count"] == 0
    assert set(audit["unavailable_reasons"]) == {
        "ACTION_DURATION_CATALOGUE_MISMATCH",
        "VISITED_STATE_POPULATION_MISMATCH",
    }
    assert all(not row["numeric_comparison_performed"] for row in audit["cases"])
    assert all("eta_minus_epsilon" not in row for row in audit["cases"])


def test_run_refuses_incomplete_collection_without_writing_result(tmp_path):
    protocol_path = tmp_path / "protocol.json"
    result_path = tmp_path / "result.json"
    protocol = _registration(tmp_path / "empty-results")
    _write(protocol_path, protocol)
    with pytest.raises(closure.ClosureRefusal, match="wave 14 must have exactly one"):
        closure.run_registered(protocol_path, result_path)
    assert not result_path.exists()


def test_complete_synthetic_split_reports_capacity_then_candidate_pending(tmp_path):
    protocol_path = tmp_path / "protocol.json"
    result_path = tmp_path / "result.json"
    protocol = _registration(tmp_path / "complete-results")
    _write(protocol_path, protocol)
    _complete_synthetic_results(protocol)
    report = closure.run_registered(protocol_path, result_path)
    assert report["joint95_audit"]["raw_rank19_score"] < 0.0
    assert report["joint95_audit"]["one_sided_margin"] == 0.0
    assert report["joint95_audit"]["holdout_covered"] is True
    assert report["capacity"]["all_registered_capacity_margins_positive"] is True
    assert report["candidate_support"]["status"] == (
        "CANDIDATE_SUPPORT_EVIDENCE_UNAVAILABLE"
    )
    assert report["scientific_decision"] == (
        "CAPACITY_ESTABLISHED_CANDIDATE_PENDING"
    )
    assert report["capacity_established"] is True
    assert report["closure_passed"] is False
    assert report["trajectory_runs"] == report["selector_calls"] == 0
    assert result_path.is_file()


@pytest.mark.parametrize(
    ("coverage", "capacity", "candidate", "expected"),
    [
        (False, True, "CANDIDATE_SUPPORT_PASS", "JOINT95_HOLDOUT_COVERAGE_FAIL_STOP"),
        (True, False, "CANDIDATE_SUPPORT_PASS", "JOINT95_NO_POSITIVE_CAPACITY_MARGIN_STOP"),
        (True, True, "CANDIDATE_SUPPORT_EVIDENCE_UNAVAILABLE", "CAPACITY_ESTABLISHED_CANDIDATE_PENDING"),
        (True, True, "CANDIDATE_SUPPORT_FAIL", "JOINT95_CANDIDATE_SUPPORT_FAIL_STOP"),
        (True, True, "CANDIDATE_SUPPORT_PASS", "JOINT95_ALIGNED_CAPACITY_CLOSURE_PASS"),
    ],
)
def test_decision_branches(coverage, capacity, candidate, expected):
    assert closure._decision(coverage, capacity, candidate) == expected
