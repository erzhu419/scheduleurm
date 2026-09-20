import copy
import json

import pytest

from algorithm.experiments import (
    aligned_hardware_joint95_v5_envelope_normalized_static_diagnostic as diagnostic,
)


def test_registration_retains_negative_v5_result_and_frozen_design():
    protocol = diagnostic.build_registration()

    assert protocol["post_result_development"] is True
    assert protocol["common_capacity_scale"] == pytest.approx(0.507946412715907)
    assert protocol["capacity_result_snapshot"]["scientific_decision"] == (
        diagnostic.NEGATIVE_CAPACITY_DECISION
    )
    assert protocol["capacity_result_snapshot"]["capacity_established"] is False
    assert protocol["capacity_result_snapshot"]["closure_passed"] is False
    design = protocol["parent_design_contract"]
    assert design["selector_contract"]["fixed_budget_by_lane"] == {
        "8": 36,
        "16": 72,
    }
    assert design["selector_contract"]["cold_canonical_reference_budget"] == 4096
    assert design["queue_roster_contract"]["seed"] == 13041
    assert len(design["queue_roster_contract"]["queues"]) == 512
    assert protocol["registration_scope"]["selector_invoked"] is False
    assert protocol["registration_scope"]["trajectory_built"] is False


def test_normalized_loads_are_fractions_of_final_envelope_boundary():
    protocol = diagnostic.build_registration()
    scale = protocol["common_capacity_scale"]
    rows = protocol["normalized_load_contract"]

    assert [row["rho"] for row in rows] == [0.5, 0.8, 0.95]
    assert [row["point_boundary_equivalent_load"] for row in rows] == pytest.approx(
        [0.5 * scale, 0.8 * scale, 0.95 * scale]
    )
    assert rows[-1]["point_boundary_equivalent_load"] < scale


def test_protocol_refuses_normalized_load_drift():
    protocol = diagnostic.build_registration()
    protocol["normalized_load_contract"][1]["rho"] = 0.81

    with pytest.raises(
        diagnostic.EnvelopeDiagnosticRefusal,
        match="normalized load contract changed",
    ):
        diagnostic._validate_protocol(protocol)


def test_fixed_sources_cover_capacity_and_model_dependencies():
    paths = set(diagnostic.FIXED_SOURCE_PATHS)
    assert "algorithm/experiments/aligned_hardware_joint95_capacity_closure.py" in paths
    assert "algorithm/experiments/aligned_hardware_frame_model.py" in paths
    assert "algorithm/experiments/aligned_hardware_lcb_frame_envelope_shadow.py" in paths


def test_capacity_snapshot_refuses_reclassification():
    protocol = diagnostic._read_object(diagnostic.DEFAULT_CAPACITY_REGISTRATION)
    result = diagnostic._read_object(diagnostic.DEFAULT_CAPACITY_RESULT)
    changed = copy.deepcopy(result)
    changed["capacity_established"] = True

    with pytest.raises(
        diagnostic.EnvelopeDiagnosticRefusal,
        match="negative capacity artifact differs",
    ):
        diagnostic._validated_capacity_snapshot(changed, protocol)


def test_run_reports_static_prerequisite_without_reclassifying_capacity(
    tmp_path, monkeypatch
):
    protocol = {
        "registered_at_utc": "test",
        **diagnostic.build_registration(),
    }
    registration = tmp_path / "protocol.json"
    fixed = tmp_path / "fixed.json"
    output = tmp_path / "result.json"
    registration.write_text(json.dumps(protocol), encoding="utf-8")
    fixed.write_text(json.dumps(diagnostic._current_fixed_sources()), encoding="utf-8")

    def fake_screen(_model, queues, lanes, capacity_rows):
        assert len(queues) == 512
        assert len(capacity_rows) == 3
        candidate = {field: 1 for field in diagnostic.canonical.WORK_FIELDS}
        reference = {field: 2 for field in diagnostic.canonical.WORK_FIELDS}
        return (
            {
                "lanes": lanes,
                "state_count": 512,
                "candidate_actual_work": candidate,
                "cold_canonical_reference_actual_work": reference,
                "support_gate_passed": True,
                "actual_compute_gate_passed": True,
            },
            {"synthetic_execution_check": True},
        )

    monkeypatch.setattr(diagnostic.canonical, "_screen_lane", fake_screen)
    report = diagnostic.run_registered(registration, fixed, output)

    assert report["status"] == "PASS"
    assert report["scientific_decision"] == diagnostic.PASS_DECISION
    assert report["trajectory_registration_prerequisite_passed"] is True
    assert report["retained_v5_capacity_decision"] == (
        diagnostic.NEGATIVE_CAPACITY_DECISION
    )
    assert report["retained_v5_capacity_established"] is False
    assert report["retained_v5_closure_passed"] is False
    assert report["capacity_closure_reclassified"] is False
    assert report["trajectory_runs"] == 0
    assert report["arrival_generations"] == 0
    assert report["rng_draws"] == 0
