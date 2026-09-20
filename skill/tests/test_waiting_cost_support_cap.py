"""Bound the holding-cost choice's support loss within the unchanged paid pool."""
from dataclasses import dataclass
import itertools

import pytest

from algorithm.budgeted_frame_candidates import select_budgeted_frame
from algorithm.experiments.aligned_hardware_frame_model import load_model
from algorithm.waiting_cost_frame_candidates import select_waiting_cost_frame


@dataclass(frozen=True)
class Option:
    requirements: tuple[int, ...]
    duration: float
    completion_offsets: tuple[tuple[float, ...], ...]


def _seed_support(options, queues, lanes):
    return max((
        min(lanes, *(q // n for q, n in zip(queues, option.requirements) if n))
        * sum(q * n for q, n in zip(queues, option.requirements))
        / max(1.0, option.duration)
        for option in options
    ), default=0.0)


def _holding(options, queues, selected, arrival_rate):
    offsets = sum(
        offset for index in selected.option_indices if index is not None
        for group in options[index].completion_offsets for offset in group
    )
    return (sum(queues) - sum(selected.service)
            + offsets / selected.duration + arrival_rate * selected.duration / 2.0)


def test_real_225_queues_keep_generation_counts_seed_and_relative_caps():
    model = load_model()
    counters = (
        "attempts", "seed_attempts", "feasible_candidate_count",
        "local_score_evaluations", "bound_option_probes",
        "bound_threshold_evaluations", "frontier_peak",
    )
    for queues in itertools.product(range(5), range(5), range(9)):
        seed_support = _seed_support(model.options, queues, model.lanes)
        for budget in (10, 14, 26, 120):
            reference = select_budgeted_frame(model.options, queues, model.lanes, budget)
            seed_eligible_count = None
            for cap in (0.0, 0.01):
                selected = select_waiting_cost_frame(
                    model.options, queues, model.lanes, budget,
                    arrival_rate=0.6, max_relative_support_loss=cap,
                )
                context = (queues, budget, cap, selected)
                assert selected.seed_support == pytest.approx(seed_support), context
                assert selected.support_incumbent_score == reference.score, context
                assert selected.upper_bound == reference.upper_bound, context
                assert selected.support_floor == max(seed_support, (1.0 - cap) * reference.score), context
                assert selected.score >= selected.seed_support, context
                assert selected.score >= (1.0 - cap) * reference.score, context
                actual_relative_loss = (
                    (reference.score - selected.score) / reference.score if reference.score else 0.0
                )
                assert selected.relative_support_loss == actual_relative_loss, context
                assert 0.0 <= selected.relative_support_loss <= cap + 1e-12, context
                for name in counters:
                    assert getattr(selected, name) == getattr(reference, name), (name, context)
                assert selected.holding_rate_cost == pytest.approx(
                    _holding(model.options, queues, selected, 0.6)
                ), context
                assert selected.holding_rate_cost <= selected.support_incumbent_holding_rate_cost + 1e-12, context
                assert selected.cost_candidate_evaluations == selected.eligible_candidate_count, context
                assert selected.support_cap_excluded_candidate_count >= 0, context
                original_seed_eligible = (selected.eligible_candidate_count
                                          + selected.support_cap_excluded_candidate_count)
                assert original_seed_eligible <= reference.feasible_candidate_count, context
                if seed_eligible_count is not None:
                    assert original_seed_eligible == seed_eligible_count, context
                seed_eligible_count = original_seed_eligible
                assert selected.gap == max(0.0, selected.upper_bound - selected.score), context
                assert selected.exact == (selected.gap == 0.0), context
                assert selected.support_search_exact == reference.exact, context
                if cap == 0.0:
                    assert selected.score == reference.score, context
                    assert selected.gap == reference.gap, context
                    assert selected.exact == reference.exact, context


def _two_speed_options(second_duration):
    return (
        Option((1, 0), 1.0, ((1.0,), ())),
        Option((0, 1), second_duration, ((), (second_duration,))),
    )


def test_one_percent_cap_allows_a_half_percent_loss_with_lower_holding_cost():
    options = _two_speed_options(1.99)
    selected = select_waiting_cost_frame(
        options, (1, 1), 2, 30, arrival_rate=2.0, max_relative_support_loss=0.01,
    )
    strict = select_waiting_cost_frame(
        options, (1, 1), 2, 30, arrival_rate=2.0, max_relative_support_loss=0.0,
    )
    assert selected.score == selected.seed_support == 1.0
    assert selected.relative_support_loss == pytest.approx(0.005)
    assert selected.score < strict.score
    assert selected.holding_rate_cost < strict.holding_rate_cost
    assert not selected.exact
    assert strict.exact


def test_one_percent_cap_rejects_the_previous_quarter_support_loss():
    options = _two_speed_options(1.5)
    uncapped = select_waiting_cost_frame(
        options, (1, 1), 2, 30, arrival_rate=2.0, max_relative_support_loss=1.0,
    )
    capped = select_waiting_cost_frame(
        options, (1, 1), 2, 30, arrival_rate=2.0, max_relative_support_loss=0.01,
    )
    assert uncapped.relative_support_loss == pytest.approx(0.25)
    assert capped.score == capped.support_incumbent_score
    assert capped.relative_support_loss == 0.0
    assert capped.support_cap_excluded_candidate_count > 0
    assert uncapped.support_cap_excluded_candidate_count == 0
    assert capped.holding_rate_cost > uncapped.holding_rate_cost


def test_zero_cap_can_improve_holding_cost_among_equal_support_actions():
    options = (
        Option((1, 0), 1.0, ((1.0,), ())),
        Option((1, 1), 2.0, ((0.1,), (2.0,))),
    )
    reference = select_budgeted_frame(options, (1, 1), 1, 10)
    selected = select_waiting_cost_frame(
        options, (1, 1), 1, 10, arrival_rate=0.1, max_relative_support_loss=0.0,
    )
    assert reference.option_indices == (0,)
    assert selected.option_indices == (1,)
    assert selected.score == reference.score == 1.0
    assert selected.relative_support_loss == 0.0
    assert selected.holding_rate_cost < selected.support_incumbent_holding_rate_cost
    assert selected.exact


def test_fixed_queue_and_budget_have_nested_eligibility_as_cap_relaxes():
    model = load_model()
    for queues in ((0, 0, 0), (1, 0, 1), (0, 1, 3), (4, 1, 5), (3, 2, 8)):
        for budget in (10, 14, 26, 120):
            previous_cost = float("inf")
            previous_eligible = 0
            seed_eligible_count = None
            for cap in (0.0, 0.01, 1.0):
                selected = select_waiting_cost_frame(
                    model.options, queues, model.lanes, budget,
                    arrival_rate=0.6, max_relative_support_loss=cap,
                )
                context = (queues, budget, cap)
                assert selected.holding_rate_cost <= previous_cost + 1e-12, context
                assert selected.eligible_candidate_count >= previous_eligible, context
                original_count = selected.eligible_candidate_count + selected.support_cap_excluded_candidate_count
                if seed_eligible_count is not None:
                    assert original_count == seed_eligible_count, context
                seed_eligible_count = original_count
                if cap == 1.0:
                    assert selected.support_cap_excluded_candidate_count == 0, context
                    assert selected.support_floor == selected.seed_support, context
                previous_cost = selected.holding_rate_cost
                previous_eligible = selected.eligible_candidate_count
    # No cross-budget holding monotonicity is asserted: a larger search can
    # increase the support incumbent, raising the relative eligibility floor.
