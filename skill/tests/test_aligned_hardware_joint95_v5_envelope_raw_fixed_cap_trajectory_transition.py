from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from algorithm.experiments import (
    aligned_hardware_joint95_v5_envelope_raw_fixed_cap_trajectory_transition
    as trajectory,
)
from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption


def _work(value: int) -> dict[str, int]:
    return {field: value for field in trajectory.WORK_FIELDS}


def _selection(
    *,
    indices=(0,),
    service=(1,),
    score=1.0,
    upper=1.0,
    epsilon=0.0,
    exact=True,
    duration=1.0,
    work=1,
):
    kwargs = {
        field: work for field in trajectory.WORK_FIELDS
    }
    return SimpleNamespace(
        option_indices=tuple(indices),
        service=tuple(service),
        score=float(score),
        upper_bound=float(upper),
        gap=max(0.0, float(upper) - float(score)),
        epsilon=float(epsilon),
        score_exact=bool(exact),
        canonical_exact=bool(exact),
        duration=float(duration),
        **kwargs,
    )


def _model(lanes=8) -> FrameModel:
    return FrameModel(
        node="fixture",
        scenario_id="fixture",
        lanes=lanes,
        workloads=("a",),
        scales=(1.0,),
        options=(
            FrameOption(
                action_id="a1",
                requirements=(1,),
                duration=1.0,
                completion_offsets=((1.0,),),
            ),
        ),
    )


def _capacity():
    return {
        str(lanes): [
            {
                "lanes": lanes,
                "load": load,
                "arrival_rates": [0.1],
                "capacity_projection_passed": True,
                "conditional_negative_drift_certified": True,
                "positive_effective_slack": True,
                "effective_slack_eta": 0.1,
            }
            for load in trajectory.LOADS
        ]
        for lanes in trajectory.LANES
    }


def _source_fixture():
    protocol = {
        "physical_complete_frame_model": {
            "node": "fixture",
            "scenario_id": "fixture",
            "lanes": 8,
            "workloads": ["a"],
            "scales": [1.0],
            "options": [{}] * 9,
        },
        "lane_minimum_effective_slack_eta": {"8": 0.1, "16": 0.1},
    }
    contract = {
        "scientific_decision": trajectory.development.PASS_DECISION,
        "selected_lane_caps": {"8": 288, "16": 576},
    }
    return contract, protocol, _capacity()


def _policy(policy, *, mean_flow=1.0, work=1, exact=True):
    frame_count = 2
    metrics = {
        "policy": policy,
        "job_count": 2,
        "completed_jobs": 2,
        "mean_flow_s": mean_flow,
        "frame_count": frame_count,
        "valid_selection_frame_count": frame_count,
        "exact_selection_frame_count": frame_count if exact else 0,
        "inexact_selection_frame_count": 0 if exact else frame_count,
        "same_state_shadow_calls": frame_count if policy == "candidate" else 0,
        "same_state_shadow_valid_count": frame_count if policy == "candidate" else None,
        "same_state_shadow_exact_count": frame_count if policy == "candidate" else None,
        "same_state_action_match_count": frame_count if policy == "candidate" else None,
        "regret_bounded_by_epsilon_count": frame_count if policy == "candidate" else None,
        "maximum_certified_gap_per_job": 0.0,
        "maximum_actual_normalized_regret": 0.0 if policy == "candidate" else None,
        "maximum_queue_recursion_violation": 0,
        "work": _work(work),
        "same_state_shadow_work": _work(work + 1) if policy == "candidate" else None,
    }
    return {
        "policy": policy,
        "metrics": metrics,
        "flow_decomposition": {"status": "PASS", "mean_flow_s": mean_flow},
    }


def _run(lanes, load, seed, *, candidate_work=1, shadow_work_offset=1):
    candidate = _policy("candidate", work=candidate_work, exact=False)
    candidate["metrics"]["same_state_shadow_work"] = _work(
        candidate_work + shadow_work_offset
    )
    return {
        "lanes": lanes,
        "load": load,
        "seed": seed,
        "arrival_tape": {
            "job_count": 2,
            "arrival_generation_count": 1,
            "shadow_trajectory_runs": 0,
        },
        "policies": [candidate, _policy("cold_k4096_reference", work=2)],
    }


def test_fresh_seed_roster_is_disjoint_and_fixed():
    assert trajectory.ARRIVAL_SEEDS == tuple(range(12061, 12081))
    assert len(trajectory.PREVIOUSLY_USED_ARRIVAL_SEEDS) == 133
    assert not (
        set(trajectory.ARRIVAL_SEEDS)
        & set(trajectory.PREVIOUSLY_USED_ARRIVAL_SEEDS)
    )
    assert trajectory.STATIC_ROSTER_RNG_SEED not in trajectory.ARRIVAL_SEEDS


def test_registration_is_zero_work_and_freezes_raw_caps(monkeypatch):
    monkeypatch.setattr(
        trajectory, "_validated_development_bundle", lambda paths: _source_fixture()
    )
    forbidden_calls = []

    def forbidden(*args, **kwargs):
        forbidden_calls.append((args, kwargs))
        raise AssertionError("registration executed experiment work")

    monkeypatch.setattr(trajectory, "poisson_arrivals", forbidden)
    monkeypatch.setattr(
        trajectory, "select_canonical_count_space_bnb_frame", forbidden
    )
    payload = trajectory.build_registration()

    assert forbidden_calls == []
    assert payload["trajectory_contract"]["candidate_fixed_budget_by_lane"] == {
        "8": 288,
        "16": 576,
    }
    assert payload["trajectory_contract"]["arrival_seeds"] == list(
        range(12061, 12081)
    )
    assert payload["registration_scope"]["rng_initializations"] == 0
    assert payload["registration_scope"]["selector_calls"] == 0
    assert payload["paper_claim_authorized"] is False


@pytest.mark.parametrize("lanes,budget", ((8, 288), (16, 576)))
def test_candidate_call_uses_one_native_fixed_budget(monkeypatch, lanes, budget):
    model = _model(lanes)
    selected = _selection(indices=(0,) * lanes, service=(lanes,))
    calls = []

    def select(options, queue, selected_lanes, selected_budget):
        calls.append((tuple(queue), selected_lanes, selected_budget))
        return selected

    monkeypatch.setattr(trajectory, "select_canonical_count_space_bnb_frame", select)
    monkeypatch.setattr(trajectory, "_certificate_valid", lambda *args, **kwargs: True)
    row = trajectory._candidate_call(model, (lanes,))

    assert calls == [((lanes,), lanes, budget)]
    assert row["selection"] is selected
    assert row["work"] == _work(1)
    assert row["valid"] is True


def test_reference_requires_native_exact_certificate(monkeypatch):
    model = _model(8)
    selected = _selection(indices=(0,) * 8, service=(8,))
    observed = []
    monkeypatch.setattr(
        trajectory,
        "select_canonical_count_space_bnb_frame",
        lambda options, queue, lanes, budget: selected,
    )

    def certificate(model, queue, budget, selection, *, exact=False):
        observed.append((budget, exact))
        return True

    monkeypatch.setattr(trajectory, "_certificate_valid", certificate)
    row = trajectory._reference_call(model, (8,))

    assert observed == [(4096, True)]
    assert row["valid"] is True


def test_candidate_trajectory_reads_only_released_queue_and_is_fifo(monkeypatch):
    model = _model(8)
    selected = _selection(
        indices=(0, None, None, None, None, None, None, None), service=(1,)
    )
    seen = []

    def candidate(_model, queue):
        seen.append(tuple(queue))
        return {"selection": selected, "work": _work(1), "valid": True}

    def shadow(_model, queue):
        return {"selection": selected, "work": _work(2), "valid": True}

    monkeypatch.setattr(trajectory, "_candidate_call", candidate)
    monkeypatch.setattr(trajectory, "_reference_call", shadow)
    metrics, frames = trajectory.simulate_trajectory(
        model, ((0.0, 0, 0), (0.5, 0, 1)), 0.5, "candidate"
    )

    assert seen == [(1,), (1,)]
    assert [row["queue_before"] for row in frames] == [(1,), (1,)]
    assert metrics["completed_jobs"] == 2
    assert metrics["same_state_shadow_calls"] == 2
    assert metrics["work"] == _work(2)
    assert metrics["same_state_shadow_work"] == _work(4)


def test_case_compute_allows_uniform_parity_and_exactness_is_diagnostic():
    runs = [
        _run(16, 0.5, seed, candidate_work=2, shadow_work_offset=0)
        for seed in trajectory.ARRIVAL_SEEDS
    ]
    summary = trajectory._paired_case_summary(runs, 16, 0.5, eta=0.1)

    assert summary["case_compute_gate_passed"] is True
    assert summary["charged_work_comparison"]["gate_fields_no_greater"] is True
    assert summary["charged_work_comparison"]["gate_fields_strictly_lower"] is False
    assert summary["candidate_inexact_frame_count"] == 40
    assert summary["candidate_exactness_is_diagnostic_not_a_gate"] is True
    assert summary["case_gate_passed"] is True


def test_lane_and_aggregate_compute_require_strict_saving():
    runs = []
    cases = []
    for lanes in trajectory.LANES:
        for load in trajectory.LOADS:
            offset = 0 if (lanes, load) == (16, 0.5) else 1
            members = [
                _run(lanes, load, seed, shadow_work_offset=offset)
                for seed in trajectory.ARRIVAL_SEEDS
            ]
            runs.extend(members)
            cases.append(
                trajectory._paired_case_summary(members, lanes, load, eta=0.1)
            )
    gates = trajectory._aggregate_work_gates(runs, cases)

    assert gates["uniform_case_no_regression_gate_passed"] is True
    assert all(row["gate_passed"] for row in gates["by_lane"])
    assert gates["aggregate"]["gate_passed"] is True
    assert gates["gate_passed"] is True


def test_any_case_work_regression_fails_uniform_compute_gate():
    members = [
        _run(8, 0.8, seed, candidate_work=2, shadow_work_offset=-1)
        for seed in trajectory.ARRIVAL_SEEDS
    ]
    summary = trajectory._paired_case_summary(members, 8, 0.8, eta=0.1)
    assert summary["case_compute_gate_passed"] is False
    assert summary["case_gate_passed"] is False


def test_decision_preserves_support_action_compute_and_flow_failures():
    base = {
        "trajectory_support_gate_passed": True,
        "same_state_action_and_regret_gate_passed": True,
        "paired_mean_flow_noninferiority_passed": True,
    }
    assert trajectory._decision(False, [base], True) == trajectory.INVALID_DECISION
    support = {**base, "trajectory_support_gate_passed": False}
    assert trajectory._decision(True, [support], True) == trajectory.SUPPORT_FAIL_DECISION
    action = {**base, "same_state_action_and_regret_gate_passed": False}
    assert trajectory._decision(True, [action], True) == trajectory.ACTION_FAIL_DECISION
    assert trajectory._decision(True, [base], False) == trajectory.COMPUTE_FAIL_DECISION
    flow = {**base, "paired_mean_flow_noninferiority_passed": False}
    assert trajectory._decision(True, [flow], True) == trajectory.FLOW_FAIL_DECISION
    assert trajectory._decision(True, [base], True) == trajectory.PASS_DECISION


def test_register_refuses_overwrite(tmp_path):
    registration = tmp_path / "protocol.json"
    registration.write_text("{}", encoding="utf-8")
    with pytest.raises(
        trajectory.RawFixedCapTrajectoryRefusal, match="already exists"
    ):
        trajectory.register(
            registration,
            tmp_path / "fixed.json",
            tmp_path / "result.json",
        )
