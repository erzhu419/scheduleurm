"""Mixed successors preserve current search, shared samples, and paid attempts."""
from dataclasses import dataclass, fields
import random

import pytest

import algorithm.mixed_successor_rollout_frame_candidates as mixed
import algorithm.poisson_rollout_frame_candidates as poisson
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


def test_homogeneous_two_lane_actions_match_raw_costs_and_charge_lane_work():
    old_rng, new_rng = GapStream((10.0,)), GapStream((10.0,))
    old = poisson.select_poisson_rollout_frame(OPTIONS, (6,), 2, 14,
                                               arrival_rngs=(old_rng,), **RATES)
    result = mixed.select_mixed_successor_rollout_frame(OPTIONS, (6,), 2, 14,
                                                        arrival_rngs=(new_rng,), **RATES)
    # Challenger leaves2 jobs and fills one batch lane; incumbent leaves4
    # and fills both. Each then examines one empty lane in successor2.
    different_counts = {
        "local_score_evaluations", "rollout_class_evaluations", "rollout_profile_probes",
        "rollout_queue_coordinate_evaluations",
    }
    assert result.rollout_used
    for field in fields(poisson.PoissonRolloutFrameSelection):
        if field.name not in different_counts:
            assert getattr(result, field.name) == getattr(old, field.name), field.name
    assert result.incumbent_rollout_cost == 14.0 and result.analytic_challenger_rollout_cost == 16.0
    assert result.option_indices == (0, 0)
    assert result.rollout_lane_fill_evaluations == 6
    assert result.rollout_temporary_queue_updates == 3
    assert result.rollout_mixed_priority_coordinate_evaluations == result.rollout_class_evaluations == 3
    assert result.rollout_mixed_profile_probes == result.rollout_profile_probes == 3
    assert result.local_score_evaluations == len(OPTIONS) + 3
    assert result.rollout_queue_coordinate_evaluations == old.rollout_queue_coordinate_evaluations + 3
    assert result.rollout_attempts == 4
    assert old_rng.draws == new_rng.draws == result.rollout_random_draws == 1


def test_catalogue_preserves_pre_draw_candidates_path_rng_and_budget(monkeypatch):
    model = load_model()
    rates = {"arrival_rates": (0.05, 0.15, 0.2), "terminal_service_rates": (0.4, 0.9, 1.2)}
    old_rows, new_rows = [], []
    old_evaluate, new_evaluate = poisson.evaluate_arrival_rollout, mixed.evaluate_mixed_successor_rollout

    def observe_old(*args):
        result = old_evaluate(*args)
        old_rows.append((args, result))
        return result

    def observe_new(*args):
        result = new_evaluate(*args)
        new_rows.append((args, result))
        return result

    monkeypatch.setattr(poisson, "evaluate_arrival_rollout", observe_old)
    monkeypatch.setattr(mixed, "evaluate_mixed_successor_rollout", observe_new)
    left = tuple(random.Random(seed) for seed in (17, 18, 19))
    right = tuple(random.Random(seed) for seed in (17, 18, 19))
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
            result = mixed.select_mixed_successor_rollout_frame(model.options, queues, model.lanes, budget,
                                                                 arrival_rngs=right, **rates)
            assert tuple(rng.getstate() for rng in left) == tuple(rng.getstate() for rng in right)
            for name in unchanged:
                assert getattr(result, name) == getattr(old, name), (queues, budget, name)
            assert result.attempts == result.root_attempts + result.rollout_attempts <= budget
            assert result.score >= result.seed_support
            assert result.score >= 0.99 * result.support_incumbent_score
            if result.rollout_used:
                active += 1
                assert len(old_rows) == len(new_rows) == 2
                for (old_args, old_cost), (new_args, new_cost) in zip(old_rows, new_rows, strict=True):
                    assert old_args == new_args
                    assert old_cost.steps[0] == new_cost.steps[0]
                assert new_rows[0][0][-1] is new_rows[1][0][-1]
                assert result.rollout_attempts == 4
                for selection_name, terminal_name in (
                    ("rollout_lane_fill_evaluations", "lane_fill_evaluations"),
                    ("rollout_temporary_queue_updates", "temporary_queue_updates"),
                    ("rollout_mixed_priority_coordinate_evaluations", "class_evaluations"),
                    ("rollout_mixed_profile_probes", "profile_probes"),
                    ("rollout_queue_coordinate_evaluations", "queue_coordinate_evaluations"),
                ):
                    assert getattr(result, selection_name) == sum(getattr(cost, terminal_name) for _, cost in new_rows)
                assert result.rollout_mixed_priority_coordinate_evaluations == result.rollout_class_evaluations
                assert result.rollout_mixed_profile_probes == result.rollout_profile_probes
                assert result.local_score_evaluations == len(model.options) + result.rollout_class_evaluations
            else:
                inactive += 1
                assert not old_rows and not new_rows
                assert tuple(rng.getstate() for rng in left) == before
                assert result.rollout_lane_fill_evaluations == result.rollout_temporary_queue_updates == 0
                assert result.rollout_mixed_priority_coordinate_evaluations == result.rollout_mixed_profile_probes == 0
    assert active > 0 and inactive > 0


def test_inactive_decisions_skip_new_work_and_preserve_all_raw_fields(monkeypatch):
    def forbidden(*args):
        pytest.fail("inactive decisions must skip path building and mixed successors")

    monkeypatch.setattr(mixed, "build_poisson_arrival_path", forbidden)
    monkeypatch.setattr(mixed, "evaluate_mixed_successor_rollout", forbidden)
    for queues, budget in (((2,), 6), ((1,), 10)):
        old = poisson.select_poisson_rollout_frame(OPTIONS, queues, 1, budget,
                                                    arrival_rngs=(NoDraws(),), **RATES)
        result = mixed.select_mixed_successor_rollout_frame(OPTIONS, queues, 1, budget,
                                                             arrival_rngs=(NoDraws(),), **RATES)
        assert not result.rollout_used
        for field in fields(poisson.PoissonRolloutFrameSelection):
            assert getattr(result, field.name) == getattr(old, field.name), field.name
        assert result.rollout_lane_fill_evaluations == result.rollout_temporary_queue_updates == 0
        assert result.rollout_mixed_priority_coordinate_evaluations == result.rollout_mixed_profile_probes == 0
