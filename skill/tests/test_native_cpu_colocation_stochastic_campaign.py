from __future__ import annotations

import json

import pytest

from algorithm.experiments import (
    native_cpu_colocation_stochastic_calibration_campaign as campaign,
)


def _write_json(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_allow_launch_requires_passed_p1_certificate(monkeypatch):
    def forbidden_run(**kwargs):
        raise AssertionError("p2 launched without a p1 certificate")

    monkeypatch.setattr(
        campaign,
        "_run_native_cpu_colocation_stochastic_campaign_unlocked",
        forbidden_run,
    )

    report = campaign.build_native_cpu_colocation_stochastic_campaign(
        allow_launch=True,
        tag="missing-p1",
        profiles=(2,),
        nodes=("node001",),
        kinds=("freqduet",),
    )

    assert report["pass"] is False
    assert report["status"] == "P1_PREREQUISITE_GAP"
    assert report["completed_wave_count"] == 0
    assert "p1_campaign_path_missing" in report["p1_prerequisite"]["errors"]
    assert "p1_gate_path_missing" in report["p1_prerequisite"]["errors"]


def test_p1_prerequisite_binds_gate_and_all_evidence_hashes(tmp_path):
    training = tmp_path / "training.json"
    calibration = tmp_path / "calibration.json"
    holdout = tmp_path / "holdout.json"
    for path in (training, calibration, holdout):
        _write_json(path, {"rows": []})
    gate_path = tmp_path / "gate.json"
    _write_json(
        gate_path,
        {
            "gate": campaign.P1_GATE_NAME,
            "status": "PASS",
            "pass": True,
            "all_rows_ready": True,
            "all_lower_service_valid_on_holdout": True,
            "training_paths": [str(training)],
            "calibration_paths": [str(calibration)],
            "holdout_path": str(holdout),
            "rows": [
                {
                    "node": "node001",
                    "workload_key": "freqduet_cpu_native",
                    "colocation_count": 1,
                    "ready": True,
                    "dispatch_state_ready": True,
                    "measurement_protocol": campaign.MEASUREMENT_PROTOCOL,
                    "dispatch_sample_count": (
                        campaign.DISPATCH_CPU_SAMPLE_COUNT
                    ),
                    "lower_service_valid_on_holdout": True,
                    "lower_service_units_per_s": 0.25,
                }
            ],
        },
    )
    campaign_path = tmp_path / "campaign.json"
    _write_json(
        campaign_path,
        {
            "status": "PASS",
            "gate_pass": True,
            "measurement_protocol": campaign.P1_MEASUREMENT_PROTOCOL,
            "gate_path": str(gate_path),
        },
    )

    certificate = campaign._validate_p1_prerequisite(
        campaign_path=campaign_path,
        gate_path=gate_path,
        required_nodes=("node001",),
        required=True,
        required_kinds=("freqduet",),
    )

    assert certificate["ready"] is True
    assert certificate["status"] == "PASS"
    assert len(certificate["campaign_sha256"]) == 64
    assert len(certificate["gate_sha256"]) == 64
    assert len(certificate["evidence_sha256"]) == 3


def test_profile_one_campaign_bootstraps_without_p1(
    monkeypatch,
):
    launched = []

    def fake_run(**kwargs):
        launched.append(kwargs)
        return {"pass": True, "status": "PASS"}

    monkeypatch.setattr(
        campaign,
        "_run_native_cpu_colocation_stochastic_campaign_unlocked",
        fake_run,
    )
    monkeypatch.setattr(
        campaign,
        "_acquire_node_locks",
        lambda nodes, tag: ([], [{"node": "node001", "tag": tag}]),
    )

    report = campaign.build_native_cpu_colocation_stochastic_campaign(
        allow_launch=True,
        tag="p1-bootstrap",
        profiles=(1,),
        nodes=("node001",),
        kinds=("freqduet",),
    )

    assert launched
    assert report["status"] == "PASS"
    assert report["p1_prerequisite"]["required"] is False
    assert report["p1_prerequisite"]["ready"] is True


def test_p1_prerequisite_requires_exact_workload_node_cell(tmp_path):
    training = tmp_path / "training.json"
    calibration = tmp_path / "calibration.json"
    holdout = tmp_path / "holdout.json"
    for path in (training, calibration, holdout):
        _write_json(path, {"rows": []})
    gate_path = tmp_path / "gate.json"
    _write_json(
        gate_path,
        {
            "gate": campaign.P1_GATE_NAME,
            "status": "PASS",
            "pass": True,
            "all_rows_ready": True,
            "all_lower_service_valid_on_holdout": True,
            "training_paths": [str(training)],
            "calibration_paths": [str(calibration)],
            "holdout_path": str(holdout),
            "rows": [
                {
                    "node": "node001",
                    "workload_key": "freqduet_cpu_native",
                    "colocation_count": 1,
                    "ready": True,
                    "dispatch_state_ready": True,
                    "measurement_protocol": campaign.MEASUREMENT_PROTOCOL,
                    "dispatch_sample_count": (
                        campaign.DISPATCH_CPU_SAMPLE_COUNT
                    ),
                    "lower_service_valid_on_holdout": True,
                    "lower_service_units_per_s": 0.25,
                }
            ],
        },
    )
    campaign_path = tmp_path / "campaign.json"
    _write_json(
        campaign_path,
        {
            "status": "PASS",
            "gate_pass": True,
            "measurement_protocol": campaign.P1_MEASUREMENT_PROTOCOL,
            "gate_path": str(gate_path),
        },
    )

    certificate = campaign._validate_p1_prerequisite(
        campaign_path=campaign_path,
        gate_path=gate_path,
        required_nodes=("node001",),
        required=True,
        required_kinds=("freqduet", "sumo"),
    )

    assert certificate["ready"] is False
    assert any(
        error.startswith("p1_gate_missing_exact_cells:")
        and "sumo_eval_cpu_native" in error
        for error in certificate["errors"]
    )


def test_existing_artifact_with_wrong_scope_is_rejected_not_reused(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(campaign, "ARTIFACT_ROOT", tmp_path)
    stale = (
        tmp_path
        / "native_cpu_colocation_stochastic_p2_training_r01_scope-test.json"
    )
    _write_json(
        stale,
        {
            "gate": "native_cpu_colocation_completion_matrix",
            "all_completion_models_ready": True,
            "selected_count": 1,
            "rows": [],
        },
    )

    def forbidden_launch(**kwargs):
        raise AssertionError("stale artifact must fail closed, not relaunch")

    monkeypatch.setattr(
        campaign,
        "build_native_cpu_colocation_completion_matrix",
        forbidden_launch,
    )
    report = (
        campaign._run_native_cpu_colocation_stochastic_campaign_unlocked(
            allow_launch=True,
            tag="scope-test",
            profiles=(2,),
            nodes=("node001",),
            kinds=("freqduet",),
        )
    )

    assert report["pass"] is False
    assert report["status"] == "STOPPED_ON_GAP"
    assert report["completed_wave_count"] == 1
    row = report["wave_rows"][0]
    assert row["action"] == "REJECTED_STALE_OR_INCOMPLETE_ARTIFACT"
    assert "measurement_scope_sha256_mismatch" in row[
        "artifact_scope_errors"
    ]
    assert "calibration_cell_set_mismatch" in row["artifact_scope_errors"]


def test_node_lock_rejects_overlapping_campaign(monkeypatch, tmp_path):
    monkeypatch.setattr(campaign, "LOCK_ROOT", tmp_path / "locks")
    handles, rows = campaign._acquire_node_locks(
        ("node001", "node002"),
        tag="first",
    )
    try:
        assert {row["node"] for row in rows} == {"node001", "node002"}
        with pytest.raises(BlockingIOError, match="node001"):
            campaign._acquire_node_locks(("node001",), tag="second")
    finally:
        campaign._release_node_locks(handles)
