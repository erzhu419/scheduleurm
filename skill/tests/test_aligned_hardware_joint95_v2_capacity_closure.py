import json
from pathlib import Path

import pytest

from algorithm.experiments import aligned_hardware_joint95_v2_capacity_closure as closure


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _bound_manifest(tmp_path: Path) -> tuple[Path, Path, dict]:
    manifest = json.loads(closure.DEFAULT_COLLECTION_MANIFEST.read_text())
    results_dir = (tmp_path / "v2-results").resolve()
    manifest["results_contract"]["results_directory"] = str(results_dir)
    manifest["authorization_snapshot"]["results_directory"] = str(results_dir)
    path = tmp_path / "v2-manifest.json"
    _write(path, manifest)
    return path, results_dir, manifest


def _protocol(tmp_path: Path) -> tuple[dict, Path, dict]:
    manifest_path, results_dir, manifest = _bound_manifest(tmp_path)
    protocol = closure.build_registration(
        collection_manifest_path=manifest_path,
        results_dir=results_dir,
    )
    return protocol, results_dir, manifest


def _audit(kind: str) -> dict:
    return {
        "status": "PASS",
        "overlap_count": 0,
        "measurement_kind": kind,
        "target_node": closure.campaign.NODE,
    }


def _visit_payload(item: dict, predictor_anchor: list[dict], score: float) -> dict:
    audits = [_audit("empty_cell") for _ in range(7)] + [
        _audit("loaded_scenario") for _ in range(2)
    ]
    observations = []
    for row in predictor_anchor:
        predicted = float(row["predicted_time_s"])
        actual = predicted * (1.0 + score)
        observations.append(
            {
                "functional_id": row["functional_id"],
                "actual_time_s": actual,
                "frozen_predicted_time_s": predicted,
                "ratio_score": actual / predicted - 1.0,
            }
        )
    return {
        "schema_version": 1,
        "gate": closure.dispatcher.RESULT_GATE,
        "protocol": closure.campaign.PROTOCOL,
        "manifest_registered": True,
        "joint_unit_id": item["joint_unit_id"],
        "joint_wave": item["joint_wave"],
        "split_role": item["split_role"],
        "status": "PASS",
        "pass": True,
        "terminal": True,
        "started_at": float(item["joint_wave"]),
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
                "joint95_scheduler_interval_audit": audits[index],
            }
            for index, identity in enumerate(
                item["empty_suite"]["service_cell_ids"]
            )
        ],
        "loaded_campaign": {
            "rows": [
                {
                    "scenario_id": identity,
                    "ready": True,
                    "capacity_boundary": False,
                    "wave": item["joint_wave"],
                    "split_role": item["split_role"],
                    "joint95_scheduler_interval_audit": audits[7 + index],
                }
                for index, identity in enumerate(
                    item["loaded_suite"]["scenario_ids"]
                )
            ]
        },
        "scheduler_interval_audits": audits,
        "observations": observations,
        "joint_wave_score": max(row["ratio_score"] for row in observations),
    }


def _write_all_visits(
    protocol: dict,
    results_dir: Path,
    manifest: dict,
    *,
    release: bool,
) -> None:
    by_wave = {row["joint_wave"]: row for row in manifest["joint_work_items"]}
    calibration_paths = []
    calibration_scores = []
    for wave in closure.campaign.CALIBRATION_WAVES:
        item = by_wave[wave]
        payload = _visit_payload(item, protocol["predictor_anchor"], -0.02)
        path = results_dir / f"{item['joint_unit_id']}_attempt01.json"
        _write(path, payload)
        calibration_paths.append(str(path))
        calibration_scores.append(payload["joint_wave_score"])
    if release:
        _write(
            results_dir / "holdout_release.json",
            {
                "schema_version": 1,
                "gate": closure.dispatcher.RELEASE_GATE,
                "protocol": closure.campaign.PROTOCOL,
                "status": "HOLDOUT_RELEASED",
                "released_at": 52.5,
                "holdout_wave": closure.campaign.HOLDOUT_WAVE,
                "calibration_waves": list(closure.campaign.CALIBRATION_WAVES),
                "calibration_joint_unit_ids": [
                    by_wave[wave]["joint_unit_id"]
                    for wave in closure.campaign.CALIBRATION_WAVES
                ],
                "calibration_result_paths": calibration_paths,
                "calibration_joint_wave_scores": calibration_scores,
                "finite_sample_rank": 19,
                "calibration_margin_frozen": True,
                "frozen_margin": 0.0,
                "v1_joint_units_used": 0,
            },
        )
    item = by_wave[closure.campaign.HOLDOUT_WAVE]
    payload = _visit_payload(item, protocol["predictor_anchor"], -0.03)
    payload["started_at"] = 53.0
    _write(results_dir / f"{item['joint_unit_id']}_attempt01.json", payload)


def test_registration_freezes_v2_split_and_reads_no_results(monkeypatch, tmp_path):
    manifest_path, results_dir, _ = _bound_manifest(tmp_path)
    original = closure.base._read_object
    reads = []

    def guarded(path):
        resolved = Path(path).resolve()
        reads.append(resolved)
        if resolved == results_dir or results_dir in resolved.parents:
            raise AssertionError("registration read the v2 results directory")
        return original(path)

    monkeypatch.setattr(closure.base, "_read_object", guarded)
    protocol = closure.build_registration(
        collection_manifest_path=manifest_path,
        results_dir=results_dir,
    )
    assert protocol["status"] == closure.STATUS
    assert protocol["source_paths"]["joint_results_directory"] == str(results_dir)
    assert protocol["joint_data_eligibility"] == {
        "eligible_campaign": "critical_gpu_joint95_v2",
        "eligible_calibration_waves": list(range(34, 53)),
        "eligible_holdout_wave": 53,
        "v1_joint_units_used": 0,
        "v1_joint_scores_used": 0,
        "v1_joint_results_read": 0,
    }
    items = protocol["collection_contract"]["joint_work_items"]
    assert [row["joint_wave"] for row in items] == list(range(34, 54))
    assert [row["split_role"] for row in items] == ["calibration"] * 19 + [
        "holdout"
    ]
    assert all(path != results_dir and results_dir not in path.parents for path in reads)
    assert not results_dir.exists()


def test_run_refuses_missing_v2_results_without_writing_output(tmp_path):
    protocol, _, _ = _protocol(tmp_path)
    registration = tmp_path / "protocol.json"
    output = tmp_path / "result.json"
    _write(registration, protocol)
    with pytest.raises(closure.ClosureRefusal, match="v2 wave 34 requires exactly one"):
        closure.run_registered(registration, output)
    assert not output.exists()


def test_run_rejects_v1_artifact_in_bound_v2_directory(tmp_path):
    protocol, results_dir, _ = _protocol(tmp_path)
    registration = tmp_path / "protocol.json"
    output = tmp_path / "result.json"
    _write(registration, protocol)
    _write(results_dir / "critical_gpu_joint95_v1_jtl110gpu2_r14_attempt01.json", {})
    with pytest.raises(closure.ClosureRefusal, match="v1 or unregistered"):
        closure.run_registered(registration, output)
    assert not output.exists()


def test_holdout_terminal_without_release_remains_locked(tmp_path):
    protocol, results_dir, manifest = _protocol(tmp_path)
    registration = tmp_path / "protocol.json"
    output = tmp_path / "result.json"
    _write(registration, protocol)
    _write_all_visits(protocol, results_dir, manifest, release=False)
    with pytest.raises(closure.ClosureRefusal, match="holdout release is unavailable"):
        closure.run_registered(registration, output)
    assert not output.exists()


def test_complete_synthetic_v2_cohort_runs_capacity_with_zero_v1_units(tmp_path):
    protocol, results_dir, manifest = _protocol(tmp_path)
    registration = tmp_path / "protocol.json"
    output = tmp_path / "result.json"
    _write(registration, protocol)
    _write_all_visits(protocol, results_dir, manifest, release=True)
    report = closure.run_registered(registration, output)
    assert report["joint95_v2_audit"]["one_sided_margin"] == 0.0
    assert report["joint95_v2_audit"]["holdout_covered"] is True
    assert report["joint_data_accounting"] == {
        "v2_calibration_units": 19,
        "v2_holdout_units": 1,
        "v1_joint_units": 0,
        "v1_joint_scores": 0,
    }
    assert report["capacity"]["all_registered_capacity_margins_positive"] is True
    assert report["scientific_decision"] == "CAPACITY_ESTABLISHED_CANDIDATE_PENDING"
    assert output.is_file()
