"""Independent accounting and Poisson expectations for one-frame lookahead."""
from dataclasses import dataclass
import itertools
import math

import pytest

from algorithm.budgeted_frame_candidates import select_budgeted_frame
from algorithm.experiments.aligned_hardware_frame_model import load_model
from algorithm.lookahead_frame_candidates import select_lookahead_frame
from algorithm.waiting_cost_frame_candidates import select_waiting_cost_frame


@dataclass(frozen=True)
class Option:
    requirements: tuple[int, ...]
    duration: float
    completion_offsets: tuple[tuple[float, ...], ...]


def _objective(options, queues, selected, arrival_rates, terminal_rates, weight):
    departures = tuple(
        sum(options[index].requirements[i] for index in selected.option_indices if index is not None)
        for i in range(len(queues))
    )
    duration = max(1.0, *(options[index].duration if index is not None else 1.0
                          for index in selected.option_indices))
    offsets = sum(
        offset for index in selected.option_indices if index is not None
        for group in options[index].completion_offsets for offset in group
    )
    area = ((sum(queues) - sum(departures)) * duration + offsets
            + sum(arrival_rates) * duration * duration / 2.0)
    next_queue = tuple(q - d + rate * duration for q, d, rate in zip(queues, departures, arrival_rates))
    terminal_area = sum(
        (future * future + rate * duration) / (2.0 * service)
        for future, rate, service in zip(next_queue, arrival_rates, terminal_rates)
    )
    return departures, duration, area, next_queue, terminal_area, area + weight * terminal_area


def _poisson_terminal_expectation(remaining_queue, duration, arrival_rates, terminal_rates):
    # Independent marginal summation, without using the closed-form variance.
    expected = 0.0
    for remaining, rate, service in zip(remaining_queue, arrival_rates, terminal_rates):
        arrival_mean = rate * duration
        probability = math.exp(-arrival_mean)
        for arrivals in range(40):
            expected += probability * (remaining + arrivals) ** 2 / (2.0 * service)
            probability *= arrival_mean / (arrivals + 1)
    return expected


def test_real_small_queues_preserve_generation_pool_caps_and_evaluation_accounting():
    model = load_model()
    arrival_rates = (0.05, 0.15, 0.2)
    terminal_rates = (0.4, 0.9, 1.2)
    generator_fields = (
        "attempts", "seed_attempts", "feasible_candidate_count",
        "local_score_evaluations", "bound_option_probes",
        "bound_threshold_evaluations", "frontier_peak",
    )
    for queues in itertools.product(range(5), range(5), range(9)):
        for budget in (10, 14, 26, 120):
            reference = select_budgeted_frame(model.options, queues, model.lanes, budget)
            old_cost = select_waiting_cost_frame(
                model.options, queues, model.lanes, budget,
                arrival_rate=sum(arrival_rates), max_relative_support_loss=0.01,
            )
            for weight in (0.0, 1.0):
                selected = select_lookahead_frame(
                    model.options, queues, model.lanes, budget,
                    arrival_rates=arrival_rates, terminal_service_rates=terminal_rates,
                    terminal_weight=weight, max_relative_support_loss=0.01,
                )
                context = (queues, budget, weight, selected)
                for name in generator_fields:
                    assert getattr(selected, name) == getattr(reference, name), (name, context)
                assert selected.seed_support == old_cost.seed_support, context
                assert selected.support_floor == old_cost.support_floor, context
                assert selected.score >= selected.seed_support, context
                assert selected.score >= 0.99 * reference.score, context
                assert selected.support_incumbent_score == reference.score, context
                assert selected.upper_bound == reference.upper_bound, context
                assert selected.eligible_candidate_count == old_cost.eligible_candidate_count, context
                assert selected.support_cap_excluded_candidate_count == old_cost.support_cap_excluded_candidate_count, context
                assert selected.cost_candidate_evaluations == selected.eligible_candidate_count, context
                assert selected.completion_offset_reads == old_cost.completion_offset_reads, context
                assert selected.next_queue_coordinate_evaluations == selected.eligible_candidate_count * len(queues), context
                assert selected.terminal_value_coordinate_evaluations == selected.eligible_candidate_count * len(queues), context
                departures, duration, area, future, terminal_area, cost = _objective(
                    model.options, queues, selected, arrival_rates, terminal_rates, weight,
                )
                assert selected.service == departures, context
                assert all(d <= q for d, q in zip(departures, queues)), context
                assert selected.duration == duration, context
                assert selected.score == sum(q * d for q, d in zip(queues, departures)) / duration, context
                assert selected.current_holding_area == pytest.approx(area), context
                assert selected.expected_next_queue == pytest.approx(future), context
                assert selected.expected_terminal_area == pytest.approx(terminal_area), context
                assert selected.lookahead_cost == pytest.approx(cost), context
                incumbent_cost = _objective(
                    model.options, queues, reference, arrival_rates, terminal_rates, weight,
                )[-1]
                assert selected.support_incumbent_lookahead_cost == pytest.approx(incumbent_cost), context
                assert selected.lookahead_cost <= incumbent_cost + 1e-10, context
                assert selected.gap == max(0.0, selected.upper_bound - selected.score), context
                assert selected.exact == (selected.gap == 0.0), context
                assert selected.epsilon == (selected.gap / sum(queues) if sum(queues) else 0.0), context
                assert selected.support_search_exact == reference.exact, context
                assert selected.relative_support_loss == (
                    (reference.score - selected.score) / reference.score if reference.score else 0.0
                ), context


def test_terminal_value_changes_the_decision_and_includes_poisson_variance():
    options = (
        Option((2, 0), 2.0, ((2.0, 2.0), ())),
        Option((0, 1), 1.0, ((), (1.0,))),
    )
    queues = (2, 2)
    arrival_rates = (0.1, 0.1)
    terminal_rates = (0.1, 10.0)
    results = []
    for weight in (0.0, 1.0):
        selected = select_lookahead_frame(
            options, queues, 1, 10, arrival_rates=arrival_rates,
            terminal_service_rates=terminal_rates, terminal_weight=weight,
        )
        remaining = tuple(q - d for q, d in zip(queues, selected.service))
        independently_summed = _poisson_terminal_expectation(
            remaining, selected.duration, arrival_rates, terminal_rates,
        )
        assert selected.expected_terminal_area == pytest.approx(independently_summed, abs=1e-12)
        mean_only = sum(future * future / (2 * service)
                        for future, service in zip(selected.expected_next_queue, terminal_rates))
        variance_contribution = sum(rate * selected.duration / (2 * service)
                                    for rate, service in zip(arrival_rates, terminal_rates))
        assert variance_contribution > 0
        assert selected.expected_terminal_area == pytest.approx(mean_only + variance_contribution)
        assert selected.lookahead_cost == pytest.approx(
            selected.current_holding_area + weight * selected.expected_terminal_area
        )
        assert selected.next_queue_coordinate_evaluations == selected.eligible_candidate_count * 2
        assert selected.terminal_value_coordinate_evaluations == selected.eligible_candidate_count * 2
        results.append(selected)
    area_only, lookahead = results
    assert area_only.option_indices == (1,)
    assert lookahead.option_indices == (0,)
    assert area_only.score == lookahead.score == area_only.seed_support == 2.0
    assert area_only.current_holding_area == pytest.approx(4.1)
    assert lookahead.current_holding_area == pytest.approx(8.4)
    assert area_only.expected_next_queue == pytest.approx((2.1, 1.1))
    assert lookahead.expected_next_queue == pytest.approx((0.2, 2.2))
    assert area_only.expected_terminal_area == pytest.approx(22.6155)
    assert lookahead.expected_terminal_area == pytest.approx(1.452)
    assert area_only.lookahead_cost == pytest.approx(4.1)
    assert lookahead.lookahead_cost == pytest.approx(9.852)


def test_terminal_value_cannot_bypass_the_service_support_floor():
    options = (
        Option((5, 0), 5.0, ((5.0,) * 5, ())),
        Option((0, 1), 1.0, ((), (1.0,))),
    )
    selected = select_lookahead_frame(
        options, (5, 1), 1, 10,
        arrival_rates=(0.0, 0.0), terminal_service_rates=(0.1, 0.0001),
    )
    assert selected.seed_support == selected.score == 5.0
    assert selected.option_indices == (0,)
    assert selected.lookahead_cost == pytest.approx(30.0 + 5000.0)
    # Serving only b would give J=6+125, but has support 1 and is ineligible.
    assert 6.0 + 125.0 < selected.lookahead_cost
