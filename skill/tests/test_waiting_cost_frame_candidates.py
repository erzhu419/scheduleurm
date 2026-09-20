"""Check holding-cost selection without changing the paid support-search pool."""
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


def _homogeneous_support(options, queues, lanes, idle_duration=1.0):
    maximum = 0.0
    for option in options:
        copies = lanes
        for available, required in zip(queues, option.requirements):
            if required:
                copies = min(copies, available // required)
        numerator = copies * sum(q * n for q, n in zip(queues, option.requirements))
        maximum = max(maximum, numerator / max(idle_duration, option.duration))
    return maximum


def _holding_rate(options, queues, selection, arrival_rate):
    offset_sum = sum(
        offset
        for index in selection.option_indices if index is not None
        for offsets in options[index].completion_offsets
        for offset in offsets
    )
    return (sum(queues) - sum(selection.service)
            + offset_sum / selection.duration
            + arrival_rate * selection.duration / 2.0)


def test_real_small_queues_preserve_paid_search_and_homogeneous_support():
    model = load_model()
    generation_counters = (
        "attempts", "seed_attempts", "feasible_candidate_count",
        "local_score_evaluations", "bound_option_probes",
        "bound_threshold_evaluations", "frontier_peak",
    )
    arrival_rate = 0.6
    maximum_frame_departures = model.lanes * max(sum(option.requirements) for option in model.options)
    for queues in itertools.product(range(5), range(5), range(9)):
        previous_cost = float("inf")
        previous_support_incumbent = -1.0
        previous_upper = float("inf")
        seed = _homogeneous_support(model.options, queues, model.lanes)
        for budget in (10, 14, 26, 120):
            reference = select_budgeted_frame(model.options, queues, model.lanes, budget)
            selected = select_waiting_cost_frame(
                model.options, queues, model.lanes, budget, arrival_rate=arrival_rate,
            )
            context = (queues, budget, selected)
            actual_departures = tuple(
                sum(model.options[index].requirements[i]
                    for index in selected.option_indices if index is not None)
                for i in range(len(queues))
            )
            actual_duration = max(1.0, *(
                model.options[index].duration if index is not None else 1.0
                for index in selected.option_indices
            ))
            assert selected.service == actual_departures, context
            assert all(d <= q for d, q in zip(actual_departures, queues)), context
            assert selected.duration == actual_duration, context
            assert selected.score == sum(q * d for q, d in zip(queues, actual_departures)) / actual_duration, context
            assert selected.seed_support == pytest.approx(seed), context
            assert selected.score >= seed - 1e-12, context
            assert selected.support_incumbent_score == reference.score, context
            assert selected.upper_bound == reference.upper_bound, context
            for name in generation_counters:
                assert getattr(selected, name) == getattr(reference, name), (name, context)
            assert selected.holding_rate_cost == pytest.approx(
                _holding_rate(model.options, queues, selected, arrival_rate)
            ), context
            assert selected.support_incumbent_holding_rate_cost == pytest.approx(
                _holding_rate(model.options, queues, reference, arrival_rate)
            ), context
            assert (selected.holding_rate_cost
                    <= selected.support_incumbent_holding_rate_cost + 1e-12), context
            assert selected.holding_rate_cost <= previous_cost + 1e-12, context
            assert selected.support_incumbent_score >= previous_support_incumbent, context
            assert selected.upper_bound <= previous_upper + 1e-12, context
            assert 1 <= selected.eligible_candidate_count <= reference.feasible_candidate_count, context
            assert selected.cost_candidate_evaluations == selected.eligible_candidate_count, context
            assert sum(selected.service) <= selected.completion_offset_reads, context
            assert selected.completion_offset_reads <= selected.cost_candidate_evaluations * min(
                sum(queues), maximum_frame_departures
            ), context
            assert selected.gap == max(0.0, selected.upper_bound - selected.score), context
            assert selected.epsilon == (selected.gap / sum(queues) if sum(queues) else 0.0), context
            assert selected.exact == (selected.gap == 0.0), context
            assert selected.support_search_exact == reference.exact, context
            previous_cost = selected.holding_rate_cost
            previous_support_incumbent = selected.support_incumbent_score
            previous_upper = selected.upper_bound


def test_end_of_frame_completions_cancel_departure_count_in_holding_ratio():
    options = (
        Option((1,), 4.0, ((4.0,),)),
        Option((3,), 4.0, ((4.0, 4.0, 4.0),)),
    )
    selected = select_waiting_cost_frame(options, (5,), 1, 10, arrival_rate=0.8)
    assert selected.service == (3,)
    # All admitted jobs remain in the system throughout this frame, regardless
    # of batch width: H = sum(Q) + Lambda*tau/2.
    assert selected.holding_rate_cost == pytest.approx(5.0 + 0.8 * 4.0 / 2.0)


def test_low_holding_cost_cannot_bypass_the_homogeneous_support_floor():
    options = (
        Option((5, 0), 5.0, ((5.0,) * 5, ())),
        Option((0, 1), 1.0, ((), (1.0,))),
    )
    selected = select_waiting_cost_frame(options, (5, 1), 1, 10, arrival_rate=2.0)
    assert selected.seed_support == 5.0
    assert selected.score == 5.0
    assert selected.option_indices == (0,)
    assert selected.holding_rate_cost == 11.0
    # The one-second b action would cost 7 but has support 1, so is ineligible.
    assert 6.0 + 2.0 / 2.0 < selected.holding_rate_cost


def test_exact_support_search_does_not_mislabel_a_lower_support_cost_choice():
    options = (
        Option((1, 0), 1.0, ((1.0,), ())),
        Option((0, 1), 1.5, ((), (1.5,))),
    )
    selected = select_waiting_cost_frame(options, (1, 1), 2, 30, arrival_rate=2.0)
    assert selected.support_search_exact
    assert selected.support_incumbent_score == pytest.approx(4.0 / 3.0)
    assert selected.seed_support == selected.score == 1.0
    assert selected.holding_rate_cost == 3.0
    assert selected.support_incumbent_holding_rate_cost == pytest.approx(19.0 / 6.0)
    assert selected.gap == pytest.approx(1.0 / 3.0)
    assert selected.epsilon == pytest.approx(1.0 / 6.0)
    assert not selected.exact
