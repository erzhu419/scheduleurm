"""Changing only successor priority preserves the paid search and sampling."""
from dataclasses import dataclass, fields
import random

import pytest

import algorithm.poisson_rollout_frame_candidates as poisson
import algorithm.pressure_rollout_frame_candidates as pressure
from algorithm.experiments.aligned_hardware_frame_model import load_model


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
        pytest.fail("inactive planning must not advance the stream")


OPTIONS = (Option((1,), 1.0, ((1.0,),)), Option((2,), 2.0, ((2.0, 2.0),)))
RATES = {"arrival_rates": (0.1,), "terminal_service_rates": (0.1,)}


def test_same_successor_order_preserves_every_old_field_and_charges_static_setup(monkeypatch):
    evaluated = []
    original = pressure.evaluate_pressure_rollout

    def evaluate(*args):
        result = original(*args)
        evaluated.append((args, result))
        return result

    monkeypatch.setattr(pressure, "evaluate_pressure_rollout", evaluate)
    left, right = GapStream((0.5, 10.0)), GapStream((0.5, 10.0))
    old = poisson.select_poisson_rollout_frame(OPTIONS, (2,), 1, 10,
                                               arrival_rngs=(left,), **RATES)
    result = pressure.select_pressure_rollout_frame(OPTIONS, (2,), 1, 10,
                                                     arrival_rngs=(right,), **RATES)
    assert old.rollout_used and result.rollout_used
    for field in fields(poisson.PoissonRolloutFrameSelection):
        assert getattr(result, field.name) == getattr(old, field.name), field.name
    assert len(evaluated) == 2
    assert evaluated[0][0][-2] is evaluated[1][0][-2]  # one shared sampled path
    assert evaluated[0][0][-1] is evaluated[1][0][-1]  # one shared static rate setup
    assert result.incumbent_rollout_cost == result.analytic_challenger_rollout_cost == 6.5
    assert result.rollout_pressure_rates == (1.0,)
    assert result.rollout_pressure_profile_reads == result.rollout_pressure_rate_evaluations == 2
    assert result.rollout_priority_coordinate_evaluations == sum(row.class_evaluations for _, row in evaluated)
    assert result.local_score_evaluations == len(OPTIONS) + result.rollout_priority_coordinate_evaluations
    assert left.draws == right.draws == result.rollout_random_draws == 2


def test_real_catalogue_keeps_rng_stream_current_contenders_and_paid_budget(monkeypatch):
    model = load_model()
    rates = {"arrival_rates": (0.05, 0.15, 0.2), "terminal_service_rates": (0.4, 0.9, 1.2)}
    old_rows, new_rows = [], []
    old_evaluate, new_evaluate = poisson.evaluate_arrival_rollout, pressure.evaluate_pressure_rollout

    def observe_old(*args):
        result = old_evaluate(*args)
        old_rows.append((args, result))
        return result

    def observe_new(*args):
        result = new_evaluate(*args)
        new_rows.append((args, result))
        return result

    monkeypatch.setattr(poisson, "evaluate_arrival_rollout", observe_old)
    monkeypatch.setattr(pressure, "evaluate_pressure_rollout", observe_new)
    left = tuple(random.Random(seed) for seed in (17, 18, 19))
    right = tuple(random.Random(seed) for seed in (17, 18, 19))
    # Streams persist across decisions, including inactive ones. Changes to
    # the successor order cannot alter planning draws or the pre-draw search.
    unchanged = (
        "upper_bound", "attempts", "seed_attempts", "feasible_candidate_count",
        "bound_option_probes", "bound_threshold_evaluations", "frontier_peak",
        "seed_support", "support_floor", "support_cap_excluded_candidate_count",
        "eligible_candidate_count", "support_incumbent_score", "support_search_exact",
        "support_incumbent_lookahead_cost", "cost_candidate_evaluations",
        "next_queue_coordinate_evaluations", "root_budget", "root_attempts",
        "rollout_attempts", "rollout_used", "analytic_challenger_score",
        "root_lookahead_cost", "rollout_setup_requirement_reads",
        "rollout_cost_candidate_evaluations", "physical_action_completion_offset_reads",
        "initial_root_budget", "initial_root_attempts", "reclaimed_reserve",
        "recycled_candidate_attempts", "current_search_resumed",
        "initial_cost_candidate_evaluations", "recycled_cost_candidate_evaluations",
        "rollout_horizon", "rollout_population_jobs", "rollout_arrival_event_generations",
        "rollout_arrival_rate_reads", "rollout_random_draws",
    )
    active = inactive = 0
    for queues in ((0, 0, 0), (1, 0, 0), (1, 4, 8), (2, 4, 8), (2, 5, 8), (10, 12, 16)):
        for budget in (10, 14, 26, 120):
            old_rows.clear()
            new_rows.clear()
            before = tuple(rng.getstate() for rng in left)
            old = poisson.select_poisson_rollout_frame(model.options, queues, model.lanes, budget,
                                                        arrival_rngs=left, **rates)
            result = pressure.select_pressure_rollout_frame(model.options, queues, model.lanes, budget,
                                                             arrival_rngs=right, **rates)
            assert tuple(rng.getstate() for rng in left) == tuple(rng.getstate() for rng in right)
            for name in unchanged:
                assert getattr(result, name) == getattr(old, name), (queues, budget, name)
            assert result.attempts == result.root_attempts + result.rollout_attempts <= budget
            assert result.score >= result.seed_support
            assert result.score >= 0.99 * result.support_incumbent_score
            assert result.gap == max(0.0, result.upper_bound - result.score)
            if result.rollout_used:
                active += 1
                assert len(old_rows) == len(new_rows) == 2
                for (old_args, old_cost), (new_args, new_cost) in zip(old_rows, new_rows, strict=True):
                    assert old_args == new_args[:-1]
                    assert old_cost.steps[0] == new_cost.steps[0]
                assert new_rows[0][0][-2] is new_rows[1][0][-2]
                assert result.rollout_attempts == 4
                assert result.rollout_pressure_profile_reads == result.rollout_pressure_rate_evaluations == 7
                assert result.rollout_pressure_rates == pytest.approx((
                    6.0 / model.options[3].duration,
                    6.0 / model.options[5].duration,
                    16.0 / model.options[8].duration,
                ))
                assert result.rollout_priority_coordinate_evaluations == result.rollout_class_evaluations
                assert result.local_score_evaluations == len(model.options) + result.rollout_priority_coordinate_evaluations
            else:
                inactive += 1
                assert not old_rows and not new_rows
                assert tuple(rng.getstate() for rng in left) == before
                assert result.rollout_pressure_rates == ()
                assert result.rollout_pressure_profile_reads == result.rollout_pressure_rate_evaluations == 0
                assert result.rollout_priority_coordinate_evaluations == 0
    assert active > 0 and inactive > 0


def test_inactive_decisions_never_build_rates_paths_or_successor_frames(monkeypatch):
    def forbidden(*args):
        pytest.fail("inactive decisions must skip all rollout setup and evaluation")

    monkeypatch.setattr(pressure, "static_pressure_rates", forbidden)
    monkeypatch.setattr(pressure, "build_poisson_arrival_path", forbidden)
    monkeypatch.setattr(pressure, "evaluate_pressure_rollout", forbidden)
    for queues, budget in (((2,), 6), ((1,), 10)):
        old = poisson.select_poisson_rollout_frame(OPTIONS, queues, 1, budget,
                                                    arrival_rngs=(NoDraws(),), **RATES)
        result = pressure.select_pressure_rollout_frame(OPTIONS, queues, 1, budget,
                                                         arrival_rngs=(NoDraws(),), **RATES)
        assert not result.rollout_used
        for field in fields(poisson.PoissonRolloutFrameSelection):
            assert getattr(result, field.name) == getattr(old, field.name), field.name
        assert result.rollout_pressure_rates == ()
        assert result.rollout_pressure_profile_reads == result.rollout_pressure_rate_evaluations == 0
        assert result.rollout_priority_coordinate_evaluations == 0
