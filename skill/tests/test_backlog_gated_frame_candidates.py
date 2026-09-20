"""Verify the fixed backlog gate and dispatch to exactly one frozen selector."""
from dataclasses import asdict
import itertools

import pytest

from algorithm import backlog_gated_frame_candidates as gated
from algorithm.budgeted_frame_candidates import select_budgeted_frame
from algorithm.experiments.aligned_hardware_frame_model import load_model
from algorithm.lookahead_frame_candidates import select_lookahead_frame


def test_registered_weights_and_capacity_cover_every_two_lane_assignment():
    model = load_model()
    gate = gated.build_backlog_gate(model.options, model.lanes)
    assert gate.weights == (8, 8, 3)
    assert gate.capacity == 48
    assert gate.setup_requirement_reads == 2 * len(model.options) * len(model.workloads)
    maximum = 0
    for assignment in itertools.product((None, *range(len(model.options))), repeat=model.lanes):
        departures = tuple(
            sum(model.options[index].requirements[i] for index in assignment if index is not None)
            for i in range(len(model.workloads))
        )
        weighted_departures = sum(weight * count for weight, count in zip(gate.weights, departures))
        assert weighted_departures <= gate.capacity
        maximum = max(maximum, weighted_departures)
    assert maximum == gate.capacity
    assert gate.load((6, 0, 0)) == gate.load((3, 3, 0)) == gate.load((0, 0, 16)) == 48
    assert gate.load((7, 0, 0)) == 56
    assert gate.load((3, 2, 2)) == 46
    assert gate.load((3, 2, 3)) == 49


def test_all_225_small_queues_return_the_exact_frozen_branch_result():
    model = load_model()
    gate = gated.build_backlog_gate(model.options, model.lanes)
    arrival_rates = (0.05, 0.15, 0.2)
    terminal_rates = (0.4, 0.9, 1.2)
    for queues in itertools.product(range(5), range(5), range(9)):
        active = sum(weight * queue for weight, queue in zip((8, 8, 3), queues)) > 48
        for budget in (10, 14, 26, 120):
            selected = gated.select_backlog_gated_frame(
                model.options, queues, model.lanes, budget, gate=gate,
                arrival_rates=arrival_rates, terminal_service_rates=terminal_rates,
            )
            if active:
                expected = select_lookahead_frame(
                    model.options, queues, model.lanes, budget,
                    arrival_rates=arrival_rates, terminal_service_rates=terminal_rates,
                    terminal_weight=1.0, max_relative_support_loss=0.01,
                )
                assert selected.score >= selected.seed_support
                assert selected.score >= 0.99 * selected.support_incumbent_score
            else:
                expected = select_budgeted_frame(model.options, queues, model.lanes, budget)
            context = (queues, budget, active)
            assert type(selected) is type(expected), context
            assert asdict(selected) == asdict(expected), context
            assert selected.attempts <= budget, context
            assert selected.gap == max(0.0, selected.upper_bound - selected.score), context
            assert selected.exact == (selected.gap == 0.0), context


@pytest.mark.parametrize("queues,expected_branch", (
    ((6, 0, 0), "v4"),
    ((7, 0, 0), "lookahead"),
    ((3, 2, 2), "v4"),
    ((3, 2, 3), "lookahead"),
))
def test_strict_boundary_dispatches_one_search_only(monkeypatch, queues, expected_branch):
    model = load_model()
    gate = gated.build_backlog_gate(model.options, model.lanes)
    calls = []

    def v4_spy(*args, **kwargs):
        calls.append("v4")
        return select_budgeted_frame(*args, **kwargs)

    def lookahead_spy(*args, **kwargs):
        calls.append("lookahead")
        assert kwargs["terminal_weight"] == 1.0
        assert kwargs["max_relative_support_loss"] == 0.01
        return select_lookahead_frame(*args, **kwargs)

    monkeypatch.setattr(gated, "select_budgeted_frame", v4_spy)
    monkeypatch.setattr(gated, "select_lookahead_frame", lookahead_spy)
    gated.select_backlog_gated_frame(
        model.options, queues, model.lanes, 26, gate=gate,
        arrival_rates=(0.05, 0.15, 0.2), terminal_service_rates=(0.4, 0.9, 1.2),
    )
    assert calls == [expected_branch]
