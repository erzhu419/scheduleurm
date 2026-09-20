from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path
import random

import pytest

from algorithm.experiments import (
    aligned_hardware_joint95_v5_envelope_normalized_trajectory_transition as trajectory,
)
from algorithm.experiments.aligned_frame_flow_decomposition import decompose_flow
from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption


def _write_json(path: Path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def frozen_protocol():
    return trajectory.build_registration()


@pytest.fixture(scope="module")
def v5_model(frozen_protocol):
    return trajectory.baseline._model_from_snapshot(
        frozen_protocol["physical_complete_frame_model"]
    )


def test_registration_binds_static_confirmation_four_piece_and_seed_roster():
    protocol = trajectory.build_registration()
    paths = protocol["source_paths"]

    assert paths == {
        "confirmation_registration": str(
            trajectory.DEFAULT_CONFIRMATION_REGISTRATION.resolve()
        ),
        "confirmation_roster": str(trajectory.DEFAULT_CONFIRMATION_ROSTER.resolve()),
        "confirmation_fixed_sources": str(
            trajectory.DEFAULT_CONFIRMATION_FIXED_SOURCES.resolve()
        ),
        "confirmation_result": str(trajectory.DEFAULT_CONFIRMATION_RESULT.resolve()),
        "baseline_registration": str(
            trajectory.DEFAULT_BASELINE_REGISTRATION.resolve()
        ),
        "baseline_fixed_sources": str(
            trajectory.DEFAULT_BASELINE_FIXED_SOURCES.resolve()
        ),
    }
    source_protocol = trajectory._read_object(
        trajectory.DEFAULT_CONFIRMATION_REGISTRATION
    )
    source_roster = trajectory._read_object(trajectory.DEFAULT_CONFIRMATION_ROSTER)
    source_result = trajectory._read_object(trajectory.DEFAULT_CONFIRMATION_RESULT)
    baseline_protocol = trajectory._read_object(
        trajectory.DEFAULT_BASELINE_REGISTRATION
    )

    assert source_result["protocol"] == source_protocol
    assert source_result["scientific_decision"] == trajectory.confirmation.PASS_DECISION
    assert source_result["status"] == "PASS"
    assert source_result["trajectory_registration_prerequisite_passed"] is True
    assert source_result["envelope_normalized_trajectory_registration_authorized"] is True
    assert source_roster == source_protocol["queue_roster_contract"]
    source_contract = protocol["source_confirmation_contract"]
    assert {
        key: value
        for key, value in source_contract.items()
        if key
        not in ("confirmation_roster_payload", "confirmation_fixed_source_keys")
    } == source_result
    assert source_contract["confirmation_roster_payload"] == source_roster
    assert source_contract["confirmation_fixed_source_keys"] == sorted(
        trajectory._read_object(trajectory.DEFAULT_CONFIRMATION_FIXED_SOURCES)
    )

    frozen = protocol["trajectory_contract"]
    baseline_frozen = baseline_protocol["trajectory_contract"]
    assert frozen["arrival_seeds"] == baseline_frozen["arrival_seeds"]
    assert frozen["arrival_seeds"] == list(range(12041, 12061))
    assert frozen["lanes"] == [8, 16]
    assert frozen["loads"] == [0.5, 0.8, 0.95]
    assert frozen["arrival_horizon_s"] == 600.0
    assert frozen["case_seed_tapes"] == 120
    assert frozen["trajectory_runs"] == 240
    assert frozen["executed_policies"] == ["candidate", "cold_k4096_reference"]
    assert protocol["baseline_seed_contract"]["arrival_seeds"] == frozen[
        "arrival_seeds"
    ]
    assert protocol["baseline_seed_contract"][
        "corresponding_result_required_absent"
    ] is True
    assert protocol["baseline_seed_contract"]["corresponding_result_path"] == str(
        trajectory.DEFAULT_BASELINE_RESULT.resolve()
    )


def test_registration_is_zero_work_and_does_not_recompute_old_capacity(
    tmp_path, monkeypatch
):
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("registration performed trajectory or model work")

    monkeypatch.setattr(random, "Random", forbidden)
    monkeypatch.setattr(trajectory, "poisson_arrivals", forbidden)
    monkeypatch.setattr(
        trajectory, "select_canonical_count_space_bnb_frame", forbidden
    )
    monkeypatch.setattr(
        trajectory,
        "select_canonical_count_space_bnb_single_exchange_frame",
        forbidden,
    )
    monkeypatch.setattr(
        trajectory,
        "select_canonical_support_rescue_bulk_tie_frame",
        forbidden,
    )
    monkeypatch.setattr(trajectory.baseline, "select_exact_renewal_mw", forbidden)
    monkeypatch.setattr(trajectory.baseline, "_model_from_snapshot", forbidden)

    registration = tmp_path / "protocol.json"
    fixed = tmp_path / "fixed_sources.json"
    result = tmp_path / "result.json"
    protocol = trajectory.register(registration, fixed, result)

    assert calls == []
    assert registration.is_file() and fixed.is_file()
    assert not result.exists()
    scope = protocol["registration_scope"]
    assert scope["rng_initializations"] == scope["rng_draws"] == 0
    assert scope["arrival_generations"] == 0
    assert scope["selector_calls"] == 0
    assert scope["trajectory_runs"] == 0
    assert scope["capacity_recomputations"] == 0

    source_text = Path(trajectory.__file__).read_text(encoding="utf-8")
    assert "capacity_certificate(" not in source_text
    assert not hasattr(trajectory, "capacity_certificate")
    normalized = protocol["physical_envelope_capacity_normalized_loads"]
    assert set(normalized) == {"8", "16"}
    assert all(len(normalized[str(lanes)]) == 3 for lanes in trajectory.LANES)


def test_work_helpers_require_and_preserve_every_registered_work_field():
    fields = tuple(trajectory.WORK_FIELDS)
    assert fields
    assert trajectory._new_work() == {field: 0 for field in fields}

    target = trajectory._new_work()
    trajectory._add_work(target, {field: index + 1 for index, field in enumerate(fields)})
    assert target == {field: index + 1 for index, field in enumerate(fields)}

    missing = {field: 1 for field in fields[:-1]}
    with pytest.raises(KeyError):
        trajectory._add_work(trajectory._new_work(), missing)


def test_fixed_sources_cover_the_seed_registration_validator():
    frozen = trajectory._current_fixed_sources()
    assert trajectory.baseline_v5.THIS_SOURCE in frozen
    assert trajectory.BASELINE_SOURCE in frozen
    assert trajectory.THIS_SOURCE in frozen


@pytest.mark.parametrize(
    "piece",
    ("confirmation_registration", "confirmation_roster", "confirmation_fixed", "confirmation_result"),
)
def test_registration_rejects_each_changed_confirmation_piece(tmp_path, piece):
    kwargs = {}
    if piece == "confirmation_registration":
        payload = trajectory._read_object(trajectory.DEFAULT_CONFIRMATION_REGISTRATION)
        payload["selector_contract"]["fixed_budget_by_lane"]["8"] = 47
        kwargs["confirmation_registration"] = _write_json(
            tmp_path / "changed_confirmation_protocol.json", payload
        )
    elif piece == "confirmation_roster":
        payload = trajectory._read_object(trajectory.DEFAULT_CONFIRMATION_ROSTER)
        payload["queues"][0] = list(payload["queues"][1])
        kwargs["confirmation_roster"] = _write_json(
            tmp_path / "changed_confirmation_roster.json", payload
        )
    elif piece == "confirmation_fixed":
        payload = trajectory._read_object(
            trajectory.DEFAULT_CONFIRMATION_FIXED_SOURCES
        )
        payload[next(iter(payload))] += "\n# changed"
        kwargs["confirmation_fixed_sources"] = _write_json(
            tmp_path / "changed_confirmation_fixed.json", payload
        )
    else:
        payload = trajectory._read_object(trajectory.DEFAULT_CONFIRMATION_RESULT)
        payload["scientific_decision"] = "CHANGED"
        kwargs["confirmation_result"] = _write_json(
            tmp_path / "changed_confirmation_result.json", payload
        )

    with pytest.raises(trajectory.EnvelopeTrajectoryRefusal):
        trajectory.build_registration(**kwargs)


def test_registration_rejects_changed_unused_arrival_seed_roster(tmp_path):
    payload = trajectory._read_object(trajectory.DEFAULT_BASELINE_REGISTRATION)
    payload["trajectory_contract"]["arrival_seeds"][0] = 12040
    changed = _write_json(tmp_path / "changed_baseline_protocol.json", payload)

    with pytest.raises(trajectory.EnvelopeTrajectoryRefusal):
        trajectory.build_registration(baseline_registration=changed)


def test_registration_rejects_seed_roster_marked_as_observed(tmp_path):
    payload = trajectory._read_object(trajectory.DEFAULT_BASELINE_REGISTRATION)
    payload["registration_scope"]["arrival_tape_constructed"] = True
    payload["registration_scope"]["rng_initializations"] = 1
    changed = _write_json(tmp_path / "observed_baseline_protocol.json", payload)

    with pytest.raises(trajectory.EnvelopeTrajectoryRefusal):
        trajectory.build_registration(baseline_registration=changed)


def test_r8_r16_dispatch_rescue_trigger_and_complete_charging(v5_model):
    r8 = trajectory._candidate_call(replace(v5_model, lanes=8), (12, 4, 44))
    assert r8["valid"] and r8["method_relation"]
    assert r8["exchange_triggered"] is True
    assert r8["exchange_attempts"] > 0
    assert r8["rescue_triggered"] is False
    assert r8["rescue_work"] == trajectory._new_work()

    safe_r16 = trajectory._candidate_call(
        replace(v5_model, lanes=16), (0, 0, 10)
    )
    assert safe_r16["valid"] and safe_r16["method_relation"]
    assert safe_r16["rescue_triggered"] is False
    assert safe_r16["work"] == safe_r16["primary_work"]
    assert safe_r16["rescue_work"] == trajectory._new_work()

    rescued_r16 = trajectory._candidate_call(
        replace(v5_model, lanes=16), (0, 0, 80)
    )
    assert rescued_r16["valid"] and rescued_r16["method_relation"]
    assert rescued_r16["rescue_triggered"] is True
    assert rescued_r16["rescue_exact"] is True
    assert any(rescued_r16["rescue_work"].values())
    assert rescued_r16["work"] == {
        field: rescued_r16["primary_work"][field]
        + rescued_r16["rescue_work"][field]
        for field in trajectory.WORK_FIELDS
    }
    for row in (r8, safe_r16, rescued_r16):
        assert set(row["work"]) == set(trajectory.WORK_FIELDS)
        assert set(row["primary_work"]) == set(trajectory.WORK_FIELDS)
        assert set(row["rescue_work"]) == set(trajectory.WORK_FIELDS)


def _one_class_r8_model():
    return FrameModel(
        node="fixture",
        scenario_id="fixture",
        lanes=8,
        workloads=("a",),
        scales=(1.0,),
        options=(
            FrameOption(
                action_id="a1",
                requirements=(1,),
                duration=2.0,
                completion_offsets=((2.0,),),
            ),
        ),
    )


def test_simulation_is_online_fifo_and_does_not_apply_static_queue_scaling(
    monkeypatch,
):
    observed_queues = []
    real_selector = trajectory.select_canonical_count_space_bnb_single_exchange_frame

    def recorded_selector(options, queues, lanes, budget):
        observed_queues.append(tuple(queues))
        return real_selector(options, queues, lanes, budget)

    monkeypatch.setattr(
        trajectory,
        "select_canonical_count_space_bnb_single_exchange_frame",
        recorded_selector,
    )
    model = _one_class_r8_model()
    tape = ((0.0, 0, 0), (0.5, 0, 1))
    metrics, frames = trajectory.simulate_trajectory(
        model, tape, 0.5, "candidate", audit_limit=8
    )
    flow = decompose_flow(model, tape, frames, 0.5)

    assert observed_queues == [(1,), (1,)]
    assert [frame["queue_before"] for frame in frames] == [(1,), (1,)]
    assert [frame["departures"] for frame in frames] == [(1,), (1,)]
    assert frames[0]["start_s"] == 0.0
    assert frames[1]["start_s"] == 2.0
    assert metrics["job_count"] == metrics["completed_jobs"] == 2
    assert metrics["maximum_queue_recursion_violation"] == 0
    assert metrics["mean_flow_s"] == pytest.approx(2.75)
    assert flow["status"] == "PASS"
    assert flow["mean_flow_s"] == pytest.approx(metrics["mean_flow_s"])


def test_every_candidate_frame_has_exact_same_state_shadow_action_and_regret():
    model = _one_class_r8_model()
    tape = ((0.0, 0, 0), (0.5, 0, 1))
    metrics, _ = trajectory.simulate_trajectory(
        model, tape, 0.5, "candidate", audit_limit=8
    )

    assert metrics["same_state_shadow_calls"] == metrics["frame_count"]
    assert metrics["same_state_shadow_exact_count"] == metrics["frame_count"]
    assert metrics["same_state_action_match_count"] == metrics["frame_count"]
    assert metrics["regret_bounded_by_epsilon_count"] == metrics["frame_count"]
    assert metrics["maximum_actual_normalized_regret"] == pytest.approx(0.0)
    assert set(metrics["work"]) == set(trajectory.WORK_FIELDS)
    assert set(metrics["primary_work"]) == set(trajectory.WORK_FIELDS)
    assert set(metrics["rescue_work"]) == set(trajectory.WORK_FIELDS)
    assert set(metrics["same_state_shadow_work"]) == set(trajectory.WORK_FIELDS)
    assert all(
        row["same_state_shadow_score_exact"] is True
        and row["same_state_shadow_canonical_exact"] is True
        and row["same_state_shadow_action_match"] is True
        and row["actual_normalized_regret"] == pytest.approx(0.0)
        for row in metrics["frame_audit_prefix"]
    )


def test_case_seed_generates_one_tape_and_replays_it_under_both_arms(monkeypatch):
    tape = ((0.0, 0, 0),)
    observed = []
    generations = []

    def fake_arrivals(rates, horizon, seed):
        generations.append((tuple(rates), horizon, seed))
        return tape

    def fake_simulate(_model, arrivals, horizon, policy):
        observed.append((arrivals, horizon, policy))
        return ({"policy": policy, "mean_flow_s": 1.0}, [])

    def fake_decompose(_model, arrivals, frames, horizon):
        assert arrivals == tape and frames == [] and horizon == 600.0
        return {
            "status": "PASS",
            "job_count": 1,
            "completed_jobs": 1,
            "mean_flow_s": 1.0,
            "flow_area_job_seconds": 1.0,
            "flow_identity_error_job_seconds": 0.0,
            "mean_deferred_visible_wait_s": 0.0,
            "mean_arrival_barrier_wait_s": 0.0,
            "mean_dispatch_to_completion_s": 1.0,
            "time_regions": {
                "arrival_window": {"mean_flow_contribution_s": 1.0},
                "drain": {"mean_flow_contribution_s": 0.0},
            },
        }

    monkeypatch.setattr(trajectory, "poisson_arrivals", fake_arrivals)
    monkeypatch.setattr(trajectory, "simulate_trajectory", fake_simulate)
    monkeypatch.setattr(trajectory, "decompose_flow", fake_decompose)
    row = trajectory._run_tape(
        (_one_class_r8_model(), {"arrival_rates": [0.1]}, 8, 0.5, 12041, 600.0)
    )

    assert len(generations) == 1
    assert observed == [
        (tape, 600.0, "candidate"),
        (tape, 600.0, "cold_k4096_reference"),
    ]
    assert row["arrival_tape"]["arrival_generation_count"] == 1
    assert row["arrival_tape"]["executed_policy_count"] == 2
    assert row["arrival_tape"]["shadow_trajectory_runs"] == 0


def _synthetic_case_rows(ratio: float):
    work = {field: 1 for field in trajectory.WORK_FIELDS}
    shadow = {field: 2 for field in trajectory.WORK_FIELDS}
    rows = []
    for seed in trajectory.ARRIVAL_SEEDS:
        candidate_metrics = {
            "job_count": 1,
            "completed_jobs": 1,
            "mean_flow_s": ratio,
            "frame_count": 1,
            "valid_selection_frame_count": 1,
            "candidate_valid_selection_frame_count": 1,
            "exact_selection_frame_count": 0,
            "method_relation_frame_count": 1,
            "same_state_shadow_valid_count": 1,
            "same_state_shadow_exact_count": 1,
            "same_state_action_match_count": 1,
            "regret_bounded_by_epsilon_count": 1,
            "rescue_trigger_frame_count": 0,
            "exact_rescue_frame_count": 0,
            "maximum_certified_gap_per_job": 0.0,
            "maximum_actual_normalized_regret": 0.0,
            "work": dict(work),
            "same_state_shadow_work": dict(shadow),
        }
        reference_metrics = {
            "job_count": 1,
            "completed_jobs": 1,
            "mean_flow_s": 1.0,
            "frame_count": 1,
            "valid_selection_frame_count": 1,
            "exact_selection_frame_count": 1,
        }
        rows.append(
            {
                "lanes": 8,
                "load": 0.5,
                "seed": seed,
                "arrival_tape": {"job_count": 1},
                "policies": [
                    {
                        "policy": "candidate",
                        "metrics": candidate_metrics,
                        "flow_decomposition": {
                            "status": "PASS",
                            "mean_flow_s": ratio,
                        },
                    },
                    {
                        "policy": "cold_k4096_reference",
                        "metrics": reference_metrics,
                        "flow_decomposition": {
                            "status": "PASS",
                            "mean_flow_s": 1.0,
                        },
                    },
                ],
            }
        )
    return rows


def test_paired_bonferroni_boundary_and_decision_direction_are_frozen():
    at_limit = trajectory._paired_case_summary(
        _synthetic_case_rows(1.01), 8, 0.5, eta=0.1
    )
    above_limit = trajectory._paired_case_summary(
        _synthetic_case_rows(1.0101), 8, 0.5, eta=0.1
    )

    assert at_limit["paired_mean_flow_ratios"] == pytest.approx([1.01] * 20)
    assert at_limit["paired_ratio_upper_confidence_bound"] == pytest.approx(1.01)
    assert at_limit["noninferiority_acceptance_upper"] == pytest.approx(1.01)
    assert at_limit["one_sided_confidence_level"] == pytest.approx(
        1.0 - 0.05 / 6
    )
    assert at_limit["paired_mean_flow_noninferiority_passed"] is True
    assert at_limit["case_gate_passed"] is True
    assert above_limit["paired_mean_flow_noninferiority_passed"] is False
    assert above_limit["case_gate_passed"] is False

    passing = {
        "trajectory_support_gate_passed": True,
        "same_state_action_and_regret_gate_passed": True,
        "paired_mean_flow_noninferiority_passed": True,
    }
    assert trajectory._decision(False, [passing], True) == trajectory.INVALID_DECISION
    assert trajectory._decision(
        True, [{**passing, "trajectory_support_gate_passed": False}], True
    ) == trajectory.SUPPORT_FAIL_DECISION
    assert trajectory._decision(
        True,
        [{**passing, "same_state_action_and_regret_gate_passed": False}],
        True,
    ) == trajectory.ACTION_FAIL_DECISION
    assert trajectory._decision(
        True,
        [{**passing, "paired_mean_flow_noninferiority_passed": False}],
        True,
    ) == trajectory.FLOW_FAIL_DECISION
    assert trajectory._decision(True, [passing], False) == trajectory.COMPUTE_FAIL_DECISION
    assert trajectory._decision(
        True,
        [{**passing, "paired_mean_flow_noninferiority_passed": False}],
        False,
    ) == trajectory.COMPUTE_FAIL_DECISION
    assert trajectory._decision(True, [passing], True) == trajectory.PASS_DECISION


def test_protocol_tamper_is_rejected_before_rng_model_or_selector_work(
    tmp_path, monkeypatch
):
    protocol = {
        "registered_at_utc": "test",
        **copy.deepcopy(trajectory.build_registration()),
    }
    protocol["trajectory_contract"]["arrival_horizon_s"] = 601.0
    registration = _write_json(tmp_path / "protocol.json", protocol)
    fixed = _write_json(
        tmp_path / "fixed_sources.json", trajectory._current_fixed_sources()
    )
    output = tmp_path / "result.json"

    def forbidden(*_args, **_kwargs):
        raise AssertionError("tampered protocol reached RNG, model, or selector work")

    monkeypatch.setattr(random, "Random", forbidden)
    monkeypatch.setattr(trajectory, "poisson_arrivals", forbidden)
    monkeypatch.setattr(trajectory.baseline, "_model_from_snapshot", forbidden)
    monkeypatch.setattr(trajectory, "_candidate_call", forbidden)
    monkeypatch.setattr(trajectory.baseline, "select_exact_renewal_mw", forbidden)

    with pytest.raises(trajectory.EnvelopeTrajectoryRefusal, match="protocol changed"):
        trajectory.run_registered(registration, fixed, output)

    assert not output.exists()
