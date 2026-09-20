from __future__ import annotations

import json
from pathlib import Path

import pytest

from algorithm.experiments import aligned_hardware_joint95_capacity_closure as closure
from algorithm.experiments import aligned_hardware_joint95_static_selector_screen as screen


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _source_copy(tmp_path: Path, *, empty_results: bool = True) -> Path:
    source = json.loads(screen.DEFAULT_CAPACITY_REGISTRATION.read_text(encoding="utf-8"))
    if empty_results:
        results = (tmp_path / "joint-results").resolve()
        results.mkdir(parents=True)
        source["source_paths"]["joint_results_directory"] = str(results)
        source["source_paths"]["holdout_release"] = str(results / "holdout_release.json")
    path = tmp_path / "capacity-protocol.json"
    _write(path, source)
    return path


def _register_tmp(tmp_path: Path):
    capacity_protocol = _source_copy(tmp_path)
    registration = tmp_path / "static-protocol.json"
    roster = tmp_path / "static-roster.json"
    fixed = tmp_path / "static-fixed.json"
    capacity_result = tmp_path / "future-capacity-result.json"
    protocol = screen.register(
        registration,
        roster,
        fixed,
        capacity_registration_path=capacity_protocol,
        capacity_result_path=capacity_result,
        previous_roster_path=screen.DEFAULT_PREVIOUS_ROSTER,
    )
    return protocol, registration, roster, fixed, capacity_result


def test_common_margin_uniformly_scales_dedicated_and_loaded_sums():
    source = json.loads(screen.DEFAULT_CAPACITY_REGISTRATION.read_text(encoding="utf-8"))
    audit = screen._uniform_margin_scaling_audit(source)
    assert audit["all_nine_action_durations_uniformly_scaled"] is True
    assert audit["issue_count"] == 0
    assert audit["action_count"] == 9
    assert audit["dedicated_action_count"] == 7
    assert audit["loaded_action_count"] == 2
    assert audit["minimum_zero_margin_duration_s"] > 1.0
    assert audit["one_second_denominator_clipping_issue"] is False
    assert all(
        row["component_count"] == (2 if row["action_kind"] == "loaded" else 1)
        for row in audit["actions"]
    )
    assert all(
        row["loaded_sum_included"]
        for row in audit["actions"]
        if row["action_kind"] == "loaded"
    )

    point = closure._model_from_snapshot(source["point_reference_model"])
    baseline = {
        row["action_id"]: float(row["upper_complete_frame_duration_s"])
        for row in source["zero_margin_action_envelopes"]
    }
    margin = 0.375
    model, envelopes = closure._planned_baseline_model(
        point,
        source["predictor_anchor"],
        source["action_mapping"],
        margin=margin,
    )
    assert len(model.options) == len(envelopes) == 9
    assert all(
        row["upper_complete_frame_duration_s"]
        == pytest.approx(baseline[row["action_id"]] * (1.0 + margin))
        for row in envelopes
    )


def test_roster_is_deterministic_unique_and_disjoint_from_both_old_rosters():
    previous_payload = json.loads(screen.DEFAULT_PREVIOUS_ROSTER.read_text(encoding="utf-8"))
    previous = screen._validated_previous_roster(previous_payload)
    first = screen.generate_roster(previous)
    second = screen.generate_roster(previous)
    assert first == second
    queues = tuple(tuple(row) for row in first["queues"])
    assert first["seed"] == 10041
    assert first["candidate_draws"] == 525
    assert first["rng_integer_draws"] == 1575
    assert queues[:3] == ((25, 10, 18), (47, 15, 15), (15, 27, 6))
    assert len(queues) == len(set(queues)) == 512
    assert not set(queues) & screen._old_base_grid()
    assert not set(queues) & set(previous)
    assert (0, 0, 0) not in queues


def test_registration_does_not_read_joint_or_capacity_results_or_run_science(
    tmp_path, monkeypatch
):
    capacity_protocol = _source_copy(tmp_path)
    source = json.loads(capacity_protocol.read_text(encoding="utf-8"))
    joint_results = Path(source["source_paths"]["joint_results_directory"])
    (joint_results / "would-be-result.json").write_text("not-json", encoding="utf-8")
    capacity_result = tmp_path / "future-capacity-result.json"
    capacity_result.write_text("not-json", encoding="utf-8")

    original_read = screen._read_object
    reads = []

    def guarded_read(path):
        resolved = Path(path).resolve()
        reads.append(resolved)
        if resolved == capacity_result.resolve() or joint_results in resolved.parents:
            raise AssertionError("registration read future result evidence")
        return original_read(path)

    def forbidden(*args, **kwargs):
        raise AssertionError("registration performed capacity or selector work")

    monkeypatch.setattr(screen, "_read_object", guarded_read)
    monkeypatch.setattr(screen, "select_count_space_bnb_frame", forbidden)
    monkeypatch.setattr(screen, "capacity_certificate", forbidden)
    monkeypatch.setattr(screen, "_uniform_capacity_slack", forbidden)
    monkeypatch.setattr(closure, "_planned_baseline_model", forbidden)

    registration = tmp_path / "static-protocol.json"
    roster = tmp_path / "static-roster.json"
    fixed = tmp_path / "static-fixed.json"
    protocol = screen.register(
        registration,
        roster,
        fixed,
        capacity_registration_path=capacity_protocol,
        capacity_result_path=capacity_result,
        previous_roster_path=screen.DEFAULT_PREVIOUS_ROSTER,
    )
    scope = protocol["registration_scope"]
    assert scope["joint_results_directory_read"] is False
    assert scope["joint_visit_values_read"] is False
    assert scope["capacity_closure_result_read"] is False
    assert scope["joint_margin_computed"] is False
    assert scope["capacity_computed"] is False
    assert scope["selector_invoked"] is False
    assert scope["exact_reference_invoked"] is False
    assert scope["roster_rng_initializations"] == 1
    assert scope["other_rng_draws"] == 0
    assert capacity_result.resolve() not in reads
    assert all(joint_results not in path.parents for path in reads)
    assert registration.is_file() and roster.is_file() and fixed.is_file()


def test_run_refuses_incomplete_joint_collection_before_capacity_or_selector(
    tmp_path, monkeypatch
):
    _, registration, roster, fixed, _ = _register_tmp(tmp_path)
    output = tmp_path / "static-result.json"

    def forbidden(*args, **kwargs):
        raise AssertionError("incomplete joint evidence reached scientific work")

    monkeypatch.setattr(screen, "select_count_space_bnb_frame", forbidden)
    monkeypatch.setattr(screen, "capacity_certificate", forbidden)
    monkeypatch.setattr(screen, "_uniform_capacity_slack", forbidden)
    with pytest.raises(screen.StaticScreenRefusal, match="incomplete or invalid"):
        screen.run_registered(registration, roster, fixed, output)
    assert not output.exists()


def test_runtime_accepts_coverage_even_when_old_point_load_gate_is_false(
    tmp_path, monkeypatch
):
    source_path = _source_copy(tmp_path, empty_results=False)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source_contract = screen._capacity_registration_contract(source)
    previous = json.loads(screen.DEFAULT_PREVIOUS_ROSTER.read_text(encoding="utf-8"))
    roster = screen.generate_roster(screen._validated_previous_roster(previous))
    capacity_result_path = tmp_path / "capacity-result.json"
    protocol = screen._protocol_constants(
        capacity_registration_path=source_path,
        capacity_result_path=capacity_result_path,
        previous_roster_path=screen.DEFAULT_PREVIOUS_ROSTER,
        source_contract=source_contract,
        previous_roster=previous,
        roster=roster,
    )

    point = closure._model_from_snapshot(source["point_reference_model"])
    joint = {
        "calibration": [{"joint_wave": wave} for wave in closure.CALIBRATION_WAVES],
        "release": {"audit_passed": True},
        "holdout": {"joint_wave": closure.HOLDOUT_WAVE},
        "holdout_covered": True,
        "one_sided_margin": 0.25,
    }
    model, envelopes = closure._planned_baseline_model(
        point,
        source["predictor_anchor"],
        source["action_mapping"],
        margin=joint["one_sided_margin"],
    )
    capacity = {
        "common_capacity_scale": 0.7,
        "all_registered_capacity_margins_positive": False,
        "case_count": 6,
    }
    capacity_result = {
        "schema_version": 1,
        "gate": screen.CAPACITY_GATE,
        "experiment": screen.CAPACITY_EXPERIMENT,
        "status": "PASS",
        "protocol": source,
        "joint95_audit": joint,
        "physical_complete_frame_envelope": {
            "model": screen._json(model.snapshot()),
            "actions": screen._json(envelopes),
        },
        "capacity": capacity,
        "capacity_established": False,
        "scientific_decision": "JOINT95_NO_POSITIVE_CAPACITY_MARGIN_STOP",
    }
    _write(capacity_result_path, capacity_result)
    monkeypatch.setattr(
        closure,
        "_current_source_snapshot",
        lambda registered: {"point_model": point, "manifest": {}},
    )
    monkeypatch.setattr(closure, "_validate_registered_sources", lambda *args: None)
    monkeypatch.setattr(closure, "_complete_joint_evidence", lambda *args: joint)
    monkeypatch.setattr(closure, "_capacity_analysis", lambda *args: capacity)

    context = screen._runtime_context(protocol)
    assert context["common_capacity_scale"] == pytest.approx(0.7)
    assert context["joint"]["holdout_covered"] is True
    assert context["capacity_closure"]["capacity_established"] is False
    assert context["capacity_closure"]["capacity"][
        "all_registered_capacity_margins_positive"
    ] is False


@pytest.mark.parametrize(
    ("execution", "support", "compute", "expected"),
    [
        (False, True, True, "INVALID_EXECUTION"),
        (True, False, True, "STATIC_SUPPORT_FAIL_STOP"),
        (True, True, False, "STATIC_COMPUTE_FAIL_STOP"),
        (
            True,
            True,
            True,
            "STATIC_SUPPORT_AND_COMPUTE_PASS_ADVANCE_TO_SEPARATE_PHYSICAL_TRAJECTORY",
        ),
    ],
)
def test_decision_branches(execution, support, compute, expected):
    lanes = [
        {"support_gate_passed": support, "actual_compute_gate_passed": compute},
        {"support_gate_passed": support, "actual_compute_gate_passed": compute},
    ]
    assert screen._decision(execution, lanes, compute) == expected
