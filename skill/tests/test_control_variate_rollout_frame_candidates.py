"""A fixed unit control-variate correction without another sample or search."""
from dataclasses import dataclass, fields, replace
import random

import pytest

import algorithm.control_variate_rollout_frame_candidates as cv
from algorithm.experiments.aligned_hardware_frame_model import load_model
from algorithm.poisson_rollout_frame_candidates import PoissonRolloutFrameSelection, select_poisson_rollout_frame


@dataclass(frozen=True)
class Option:
    requirements: tuple[int, ...]
    duration: float
    completion_offsets: tuple[tuple[float, ...], ...]


class GapStream:
    def __init__(self, gaps):
        self.gaps, self.draws = iter(gaps), 0

    def expovariate(self, rate):
        self.draws += 1
        return next(self.gaps)


class NoDraws:
    def expovariate(self, rate):
        pytest.fail("this stream must not be advanced")


OPTIONS = (Option((1,), 1.0, ((1.0,),)), Option((2,), 2.0, ((2.0, 2.0),)))
RATES = {"arrival_rates": (0.1,), "terminal_service_rates": (0.1,)}


def test_unit_correction_uses_sampled_next_queue_and_keeps_negative_values():
    rng = GapStream((0.5, 10.0))
    result = cv.select_control_variate_rollout_frame(OPTIONS, (2,), 1, 10,
                                                    arrival_rngs=(rng,), **RATES)
    # Both sampled paths have X=6.5. The single has current area 2.5 and
    # Qnext=2; the batch has current area 5.5 and Qnext=1. Thus Y is 22.5
    # and 10.5 respectively, whereas E[Y] is 8.6 and 5.4.
    assert result.incumbent_raw_rollout_cost == pytest.approx(6.5)
    assert result.analytic_challenger_raw_rollout_cost == pytest.approx(6.5)
    assert result.incumbent_sampled_control_value == pytest.approx(22.5)
    assert result.analytic_challenger_sampled_control_value == pytest.approx(10.5)
    assert result.support_incumbent_lookahead_cost == pytest.approx(8.6)
    assert result.root_lookahead_cost == pytest.approx(5.4)
    assert result.incumbent_rollout_cost == pytest.approx(6.5 - (22.5 - 8.6))
    assert result.analytic_challenger_rollout_cost == pytest.approx(6.5 - (10.5 - 5.4))
    assert result.selected_rollout_cost == pytest.approx(-7.4)
    assert result.selected_rollout_cost < 0.0 and result.option_indices == (0,)
    assert result.raw_rollout_margin == pytest.approx(0.0)
    assert result.corrected_rollout_margin == pytest.approx(8.8)
    assert rng.draws == result.rollout_random_draws == 2
    assert result.rollout_attempts == 4


def test_empty_sample_retains_the_analytic_arrival_correction_at_unit_weight():
    rng = GapStream((10.0,))
    result = cv.select_control_variate_rollout_frame(OPTIONS, (2,), 1, 10,
                                                    arrival_rngs=(rng,), **RATES)
    # No sampled arrivals: X=(3,4), Y=(2+5,4+0). Adding E[Y]=(8.6,5.4)
    # yields (4.6,5.4), rather than treating this path as a zero-rate model.
    assert result.incumbent_raw_rollout_cost == pytest.approx(3.0)
    assert result.analytic_challenger_raw_rollout_cost == pytest.approx(4.0)
    assert result.incumbent_sampled_control_value == pytest.approx(7.0)
    assert result.analytic_challenger_sampled_control_value == pytest.approx(4.0)
    assert result.incumbent_rollout_cost == pytest.approx(4.6)
    assert result.analytic_challenger_rollout_cost == pytest.approx(5.4)
    assert result.raw_rollout_margin == pytest.approx(1.0)
    assert result.corrected_rollout_margin == pytest.approx(0.8)
    assert result.rollout_arrival_event_generations == 0
    assert rng.draws == result.rollout_random_draws == 1


def test_zero_rates_preserve_old_path_cost_and_add_only_two_value_coordinates():
    rates = {"arrival_rates": (0.0,), "terminal_service_rates": (0.1,)}
    old = select_poisson_rollout_frame(OPTIONS, (2,), 1, 10, arrival_rngs=(NoDraws(),), **rates)
    result = cv.select_control_variate_rollout_frame(OPTIONS, (2,), 1, 10,
                                                    arrival_rngs=(NoDraws(),), **rates)
    assert result.rollout_used and result.rollout_random_draws == 0
    for field in fields(PoissonRolloutFrameSelection):
        if field.name == "terminal_value_coordinate_evaluations":
            assert getattr(result, field.name) == getattr(old, field.name) + 2
        else:
            assert getattr(result, field.name) == getattr(old, field.name), field.name
    assert result.cv_terminal_value_coordinate_evaluations == 2
    assert result.raw_rollout_margin == result.corrected_rollout_margin


def test_identical_rng_states_keep_sampling_search_pool_and_budget_identical():
    model = load_model()
    rates = {"arrival_rates": (0.05, 0.15, 0.2), "terminal_service_rates": (0.4, 0.9, 1.2)}
    changed_fields = {
        "option_indices", "service", "duration", "score", "gap", "epsilon", "exact",
        "relative_support_loss", "current_holding_area", "expected_next_queue", "expected_terminal_area",
        "lookahead_cost", "selected_rollout_cost", "incumbent_rollout_cost",
        "analytic_challenger_rollout_cost", "rollout_vetoed", "terminal_value_coordinate_evaluations",
    }
    active = 0
    for queues in ((0, 0, 0), (1, 0, 0), (0, 4, 8), (2, 4, 8), (2, 5, 8), (10, 12, 16)):
        for budget in (10, 14, 26):
            left = tuple(random.Random(seed) for seed in (17, 18, 19))
            right = tuple(random.Random(seed) for seed in (17, 18, 19))
            old = select_poisson_rollout_frame(model.options, queues, model.lanes, budget,
                                               arrival_rngs=left, **rates)
            result = cv.select_control_variate_rollout_frame(model.options, queues, model.lanes, budget,
                                                              arrival_rngs=right, **rates)
            assert tuple(rng.getstate() for rng in left) == tuple(rng.getstate() for rng in right)
            for field in fields(PoissonRolloutFrameSelection):
                if field.name not in changed_fields:
                    assert getattr(result, field.name) == getattr(old, field.name), (queues, budget, field.name)
            extra = 2 * len(queues) if old.rollout_used else 0
            assert result.cv_terminal_value_coordinate_evaluations == extra
            assert result.terminal_value_coordinate_evaluations == old.terminal_value_coordinate_evaluations + extra
            assert result.attempts == result.root_attempts + result.rollout_attempts <= budget
            assert result.score >= result.seed_support
            assert result.score >= 0.99 * result.support_incumbent_score
            assert result.gap == max(0.0, result.upper_bound - result.score)
            active += old.rollout_used
    assert active > 0


def test_extra_value_accounting_matches_actual_reads_including_zero_coordinates(monkeypatch):
    reads = []
    shared_paths = []
    original = cv.evaluate_arrival_rollout

    class QueueCoordinates(tuple):
        def __iter__(self):
            for value in super().__iter__():
                reads.append(value)
                yield value

    def evaluate(*args):
        result = original(*args)
        shared_paths.append(args[-1])
        first = {**result.steps[0], "queue_after": QueueCoordinates(result.steps[0]["queue_after"])}
        return replace(result, steps=(first, *result.steps[1:]))

    monkeypatch.setattr(cv, "evaluate_arrival_rollout", evaluate)
    options = (Option((1, 0), 1.0, ((1.0,), ())), Option((2, 0), 2.0, ((2.0, 2.0), ())),
               Option((0, 1), 1.0, ((), (1.0,))))
    rates = {"arrival_rates": (0.1, 0.0), "terminal_service_rates": (0.1, 1.0)}
    result = cv.select_control_variate_rollout_frame(options, (2, 0), 1, 12,
                                                    arrival_rngs=(GapStream((0.5, 10.0)), NoDraws()), **rates)
    assert len(shared_paths) == 2 and shared_paths[0] is shared_paths[1]
    assert reads == [1, 0, 2, 0]  # challenger then incumbent; each d=2 exactly once.
    assert result.cv_terminal_value_coordinate_evaluations == len(reads) == 4
    assert result.rollout_attempts == 4 and result.rollout_random_draws == 2
    old = select_poisson_rollout_frame(options, (2, 0), 1, 12,
                                       arrival_rngs=(GapStream((0.5, 10.0)), NoDraws()), **rates)
    assert result.terminal_value_coordinate_evaluations == old.terminal_value_coordinate_evaluations + 4
    assert result.rollout_terminal_value_coordinate_evaluations == old.rollout_terminal_value_coordinate_evaluations


def test_inactive_search_keeps_all_old_fields_and_performs_no_correction(monkeypatch):
    def forbidden(*args):
        pytest.fail("inactive rollout must not evaluate a control variate")

    monkeypatch.setattr(cv, "build_poisson_arrival_path", forbidden)
    monkeypatch.setattr(cv, "evaluate_arrival_rollout", forbidden)
    for queues, budget in (((2,), 6), ((1,), 10)):
        old = select_poisson_rollout_frame(OPTIONS, queues, 1, budget, arrival_rngs=(NoDraws(),), **RATES)
        result = cv.select_control_variate_rollout_frame(OPTIONS, queues, 1, budget,
                                                        arrival_rngs=(NoDraws(),), **RATES)
        for field in fields(PoissonRolloutFrameSelection):
            assert getattr(result, field.name) == getattr(old, field.name), field.name
        assert result.cv_terminal_value_coordinate_evaluations == 0
        assert result.raw_rollout_margin is None and result.corrected_rollout_margin is None
