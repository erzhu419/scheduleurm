"""Planning-only Poisson streams and shared, fully charged path sampling."""
from dataclasses import dataclass, fields
import math
import random

import pytest

from algorithm.arrival_rollout_frame_candidates import ArrivalRolloutFrameSelection, select_arrival_rollout_frame
from algorithm.arrival_rollout_terminal import evaluate_arrival_rollout
import algorithm.poisson_rollout_frame_candidates as poisson
from algorithm.poisson_rollout_terminal import build_poisson_arrival_path


class GapStream:
    def __init__(self, gaps):
        self.gaps = iter(gaps)
        self.rates = []

    def expovariate(self, rate):
        self.rates.append(rate)
        return next(self.gaps)


class NoDraws:
    def expovariate(self, rate):
        pytest.fail("this stream must not be advanced")


@dataclass(frozen=True)
class Option:
    requirements: tuple[int, ...]
    duration: float
    completion_offsets: tuple[tuple[float, ...], ...]


OPTIONS = (Option((1,), 1.0, ((1.0,),)), Option((2,), 2.0, ((2.0, 2.0),)))
RATES = {"arrival_rates": (0.1,), "terminal_service_rates": (0.1,)}


def _assert_old_fields_equal(new, old):
    for field in fields(ArrivalRolloutFrameSelection):
        assert getattr(new, field.name) == getattr(old, field.name), field.name


def test_gaps_accumulate_and_stopping_draws_count_even_without_an_event():
    first = GapStream((0.25, 0.75, 0.5, 0.25))
    second = GapStream((2.0,))
    path = build_poisson_arrival_path((2.0, 0.5, 0.0), 1.5, (first, second, NoDraws()))
    assert path.times == ((0.25, 1.0, 1.5), (), ())
    assert path.horizon == 1.5
    assert first.rates == [2.0] * 4 and second.rates == [0.5]
    assert path.rate_reads == 3
    assert path.event_generations == 3
    assert path.random_draws == 5  # three events plus two H-exceeding draws.


def test_real_random_streams_match_inverse_exponential_sums_and_resume_their_state():
    rates, horizon = (0.5, 0.8, 0.25), 3.0
    rngs = tuple(random.Random(seed) for seed in (0, 1, 2))
    references = tuple(random.Random(seed) for seed in (0, 1, 2))
    original_states = tuple(rng.getstate() for rng in rngs)
    for _ in range(2):
        expected_times, expected_draws = [], 0
        for rate, reference in zip(rates, references):
            events, elapsed = [], 0.0
            while True:
                # Independent inverse-CDF calculation, without expovariate().
                elapsed += -math.log1p(-reference.random()) / rate
                expected_draws += 1
                if elapsed > horizon:
                    break
                events.append(elapsed)
            expected_times.append(tuple(events))
        path = build_poisson_arrival_path(rates, horizon, rngs)
        for actual, expected in zip(path.times, expected_times):
            assert actual == pytest.approx(expected)
        assert path.random_draws == expected_draws
        assert tuple(rng.getstate() for rng in rngs) == tuple(rng.getstate() for rng in references)
    assert tuple(rng.getstate() for rng in rngs) != original_states


def test_one_class_cannot_consume_another_class_random_stream():
    left = tuple(random.Random(seed) for seed in (11, 12, 13))
    right = tuple(random.Random(seed) for seed in (11, 12, 13))
    zero_class_state = left[2].getstate()
    baseline = build_poisson_arrival_path((0.5, 0.8, 0.0), 4.0, left)
    changed = build_poisson_arrival_path((5.0, 0.8, 0.25), 4.0, right)
    assert baseline.times[1] == changed.times[1]
    assert left[1].getstate() == right[1].getstate()
    assert left[0].getstate() != right[0].getstate()
    assert left[2].getstate() == zero_class_state
    assert right[2].getstate() != zero_class_state


def test_inactive_and_recovery_branches_do_not_sample_or_change_old_fields(monkeypatch):
    def forbidden(*args):
        pytest.fail("inactive current search must not construct a planning path")

    monkeypatch.setattr(poisson, "build_poisson_arrival_path", forbidden)
    monkeypatch.setattr(poisson, "evaluate_arrival_rollout", forbidden)
    for queues, budget in (((2,), 6), ((1,), 10)):
        old = select_arrival_rollout_frame(OPTIONS, queues, 1, budget, **RATES)
        new = poisson.select_poisson_rollout_frame(OPTIONS, queues, 1, budget,
                                                  arrival_rngs=(NoDraws(),), **RATES)
        assert not new.rollout_used
        assert new.rollout_random_draws == 0
        _assert_old_fields_equal(new, old)


def test_two_contenders_share_one_sampled_path_and_the_frozen_area_integrator(monkeypatch):
    assert poisson.evaluate_arrival_rollout is evaluate_arrival_rollout
    built, evaluations = [], []
    original_build, original_evaluate = poisson.build_poisson_arrival_path, poisson.evaluate_arrival_rollout

    def build(*args):
        path = original_build(*args)
        built.append(path)
        return path

    def evaluate(*args):
        result = original_evaluate(*args)
        evaluations.append((args[-1], result))
        return result

    monkeypatch.setattr(poisson, "build_poisson_arrival_path", build)
    monkeypatch.setattr(poisson, "evaluate_arrival_rollout", evaluate)
    stream = GapStream((0.5, 10.0))
    selected = poisson.select_poisson_rollout_frame(OPTIONS, (2,), 1, 10,
                                                    arrival_rngs=(stream,), **RATES)
    assert len(built) == 1 and len(evaluations) == 2
    assert all(path is built[0] for path, _ in evaluations)
    assert built[0].times == ((0.5,),) and built[0].horizon == 6.0
    assert stream.rates == [0.1, 0.1]
    assert selected.rollout_random_draws == built[0].random_draws == 2
    assert selected.rollout_arrival_event_generations == 1
    assert selected.rollout_population_jobs == 3
    assert selected.rollout_attempts == sum(result.attempts for _, result in evaluations) == 4
    assert selected.attempts == selected.root_attempts + 4 <= 10
    assert selected.score >= selected.seed_support
    assert selected.score >= 0.99 * selected.support_incumbent_score
    # The one shared arrival waits .5 in the single frame or 1.5 in the batch.
    # Subsequent integer dispatch yields total area 6.5 in either branch.
    assert selected.incumbent_rollout_cost == pytest.approx(6.5)
    assert selected.analytic_challenger_rollout_cost == pytest.approx(6.5)
    for _, result in evaluations:
        assert result.cost == pytest.approx(6.5)
        assert result.population_jobs == 3
        assert result.population_error == pytest.approx(0.0, abs=1e-12)


def test_zero_arrival_active_comparison_reproduces_midpoint_selector_without_draws():
    rates = {"arrival_rates": (0.0,), "terminal_service_rates": (0.1,)}
    old = select_arrival_rollout_frame(OPTIONS, (2,), 1, 10, **rates)
    selected = poisson.select_poisson_rollout_frame(OPTIONS, (2,), 1, 10,
                                                    arrival_rngs=(NoDraws(),), **rates)
    assert selected.rollout_used and selected.rollout_attempts == 4
    assert selected.rollout_random_draws == selected.rollout_arrival_event_generations == 0
    _assert_old_fields_equal(selected, old)
