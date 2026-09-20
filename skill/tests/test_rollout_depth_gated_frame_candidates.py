"""The rollout-depth gate routes once and preserves both frozen branches."""
from __future__ import annotations

import random

import pytest

import algorithm.rollout_depth_gated_frame_candidates as gated
from algorithm.budgeted_frame_candidates import select_budgeted_frame
from algorithm.experiments.aligned_hardware_frame_model import capacity_certificate, load_model
from algorithm.experiments.aligned_hardware_lookahead import terminal_service_rates
from algorithm.mixed_successor_rollout_frame_candidates import (
    MixedSuccessorRolloutFrameSelection,
    select_mixed_successor_rollout_frame,
)


class NoDraws:
    def expovariate(self, rate):
        pytest.fail("the inactive v4 branch must not advance planning RNGs")


def model_inputs(load=.8):
    model = load_model()
    certificate = capacity_certificate(model, load)
    return model, certificate["arrival_rates"], terminal_service_rates(model, certificate)


def test_strict_boundary_routes_to_exactly_one_selector(monkeypatch):
    model, rates, service_rates = model_inputs()
    gate = gated.build_backlog_gate(model.options, model.lanes)
    assert gate.weights == (8, 8, 3) and gate.capacity == 48
    calls = {"v4": 0, "mixed": 0}
    sentinels = {"v4": object(), "mixed": object()}

    def v4(*args, **kwargs):
        calls["v4"] += 1
        return sentinels["v4"]

    def mixed(*args, **kwargs):
        calls["mixed"] += 1
        return sentinels["mixed"]

    monkeypatch.setattr(gated, "select_budgeted_frame", v4)
    monkeypatch.setattr(gated, "select_mixed_successor_rollout_frame", mixed)
    common = dict(gate=gate, arrival_rates=rates, terminal_service_rates=service_rates,
                  arrival_rngs=(NoDraws(),) * 3)

    # 8*12 = 2*C remains inactive; 8*11 + 3*3 = 2*C+1 is active.
    assert gated.select_rollout_depth_gated_frame(
        model.options, (12, 0, 0), model.lanes, 26, **common) is sentinels["v4"]
    assert calls == {"v4": 1, "mixed": 0}
    assert gated.select_rollout_depth_gated_frame(
        model.options, (11, 0, 3), model.lanes, 26, **common) is sentinels["mixed"]
    assert calls == {"v4": 1, "mixed": 1}


def test_inactive_branch_equals_frozen_v4_and_never_draws():
    model, rates, service_rates = model_inputs()
    gate = gated.build_backlog_gate(model.options, model.lanes)
    result = gated.select_rollout_depth_gated_frame(
        model.options, (12, 0, 0), model.lanes, 26,
        gate=gate, arrival_rates=rates, terminal_service_rates=service_rates,
        arrival_rngs=(NoDraws(),) * 3,
    )
    assert result == select_budgeted_frame(model.options, (12, 0, 0), model.lanes, 26)
    assert not isinstance(result, MixedSuccessorRolloutFrameSelection)


def test_active_branch_equals_frozen_mixed_with_the_same_rng_endpoint():
    model, rates, service_rates = model_inputs()
    gate = gated.build_backlog_gate(model.options, model.lanes)
    left = tuple(random.Random(f"rollout-depth-gate:{coordinate}") for coordinate in range(3))
    right = tuple(random.Random(f"rollout-depth-gate:{coordinate}") for coordinate in range(3))
    queues = (4, 5, 10)  # weighted backlog 102 > 96
    result = gated.select_rollout_depth_gated_frame(
        model.options, queues, model.lanes, 26,
        gate=gate, arrival_rates=rates, terminal_service_rates=service_rates,
        arrival_rngs=left,
    )
    expected = select_mixed_successor_rollout_frame(
        model.options, queues, model.lanes, 26,
        arrival_rates=rates, terminal_service_rates=service_rates,
        arrival_rngs=right,
    )
    assert result == expected
    assert tuple(rng.getstate() for rng in left) == tuple(rng.getstate() for rng in right)


def test_known_active_queue_preserves_budget_support_draw_and_lane_accounting():
    model, rates, service_rates = model_inputs()
    gate = gated.build_backlog_gate(model.options, model.lanes)
    selected = gated.select_rollout_depth_gated_frame(
        model.options, (4, 5, 10), model.lanes, 26,
        gate=gate, arrival_rates=rates, terminal_service_rates=service_rates,
        arrival_rngs=tuple(random.Random(f"known-active:{coordinate}") for coordinate in range(3)),
    )
    assert isinstance(selected, MixedSuccessorRolloutFrameSelection)
    assert selected.rollout_used
    assert selected.attempts == selected.root_attempts + selected.rollout_attempts <= 26
    assert selected.rollout_attempts == 4
    assert selected.rollout_cost_candidate_evaluations == 2
    assert selected.score >= selected.seed_support
    assert selected.score >= selected.support_floor
    assert selected.relative_support_loss <= .01 + 1e-12
    assert selected.rollout_random_draws == selected.rollout_arrival_event_generations + 3
    assert selected.rollout_mixed_priority_coordinate_evaluations == selected.rollout_class_evaluations
    assert selected.rollout_mixed_profile_probes == selected.rollout_profile_probes
    assert (0 < selected.rollout_temporary_queue_updates
            <= selected.rollout_lane_fill_evaluations <= 8)
    assert selected.rollout_temporary_queue_updates <= selected.rollout_queue_coordinate_evaluations
