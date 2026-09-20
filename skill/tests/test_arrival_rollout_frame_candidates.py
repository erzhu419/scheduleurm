"""Arrival-aware terminal costs retain the frozen current-search budget."""
from dataclasses import dataclass, fields

import pytest

import algorithm.arrival_rollout_frame_candidates as arrival
from algorithm.batch_rollout_frame_candidates import select_batch_rollout_frame
from algorithm.budgeted_frame_candidates import select_budgeted_frame
from algorithm.experiments.aligned_hardware_frame_model import load_model
from algorithm.lookahead_frame_candidates import select_lookahead_frame
from algorithm.recycled_rollout_frame_candidates import RecycledRolloutFrameSelection, select_recycled_rollout_frame


@dataclass(frozen=True)
class Option:
    requirements: tuple[int, ...]
    duration: float
    completion_offsets: tuple[tuple[float, ...], ...]


def test_representative_catalogue_states_preserve_search_caps_and_total_budget():
    model = load_model()
    rates = {"arrival_rates": (0.05, 0.15, 0.2), "terminal_service_rates": (0.4, 0.9, 1.2)}
    states = ((0, 0, 0), (1, 0, 0), (1, 1, 1), (3, 0, 3), (1, 4, 8),
              (4, 4, 8), (4, 1, 5), (10, 12, 16), (2, 5, 8), (1, 6, 4))
    active = recovered = 0
    for queues in states:
        for budget in (10, 14, 26, 120):
            old = select_recycled_rollout_frame(model.options, queues, model.lanes, budget, **rates)
            new = arrival.select_arrival_rollout_frame(model.options, queues, model.lanes, budget, **rates)
            context = (queues, budget)
            for name in ("initial_root_budget", "initial_root_attempts", "root_budget", "root_attempts",
                         "attempts", "seed_attempts", "feasible_candidate_count", "bound_option_probes",
                         "bound_threshold_evaluations", "frontier_peak", "seed_support", "support_floor",
                         "support_incumbent_score", "support_search_exact", "upper_bound", "reclaimed_reserve",
                         "recycled_candidate_attempts", "current_search_resumed", "rollout_used",
                         "rollout_attempts", "eligible_candidate_count", "cost_candidate_evaluations"):
                assert getattr(new, name) == getattr(old, name), (name, context)
            assert (new.local_score_evaluations - new.rollout_class_evaluations
                    == old.local_score_evaluations - old.rollout_class_evaluations), context
            assert new.rollout_attempts in (0, 4), context
            assert new.attempts == new.root_attempts + new.rollout_attempts <= budget, context
            if new.rollout_used:
                active += 1
                support = select_budgeted_frame(model.options, queues, model.lanes, new.root_budget)
                analytic = select_lookahead_frame(model.options, queues, model.lanes, new.root_budget, **rates)
                assert new.option_indices in (support.option_indices, analytic.option_indices), context
                assert new.selected_rollout_cost == min(new.incumbent_rollout_cost,
                                                       new.analytic_challenger_rollout_cost), context
            else:
                recovered += new.current_search_resumed
                assert new.option_indices == old.option_indices, context
            assert new.score >= new.seed_support, context
            assert new.score >= 0.99 * new.support_incumbent_score, context
            departures = tuple(sum(model.options[j].requirements[i] for j in new.option_indices if j is not None)
                               for i in range(len(queues)))
            duration = max((1.0, *(model.options[j].duration for j in new.option_indices if j is not None)))
            assert departures == new.service and all(d <= q for d, q in zip(departures, queues)), context
            assert duration == new.duration, context
            assert new.score == sum(q * d for q, d in zip(queues, departures)) / duration, context
            assert new.gap == max(0.0, new.upper_bound - new.score), context
            assert new.exact == (new.gap == 0.0), context
            assert new.epsilon == (new.gap / sum(queues) if sum(queues) else 0.0), context
    assert active > 0 and recovered > 0


def test_recovery_and_no_reserve_keep_all_old_fields_without_calling_arrival_prediction(monkeypatch):
    def forbidden(*args):
        pytest.fail("inactive rollout must not build or evaluate a prediction path")

    monkeypatch.setattr(arrival, "build_arrival_path", forbidden)
    monkeypatch.setattr(arrival, "evaluate_arrival_rollout", forbidden)
    model = load_model()
    rates = {"arrival_rates": (0.05, 0.15, 0.2), "terminal_service_rates": (0.4, 0.9, 1.2)}
    for queues, budget in (((1, 4, 8), 10), ((0, 0, 0), 14)):
        old = select_recycled_rollout_frame(model.options, queues, model.lanes, budget, **rates)
        new = arrival.select_arrival_rollout_frame(model.options, queues, model.lanes, budget, **rates)
        assert not new.rollout_used
        for field in fields(RecycledRolloutFrameSelection):
            assert getattr(new, field.name) == getattr(old, field.name), field.name
        assert new.rollout_horizon is None
        for name in ("rollout_population_jobs", "rollout_arrival_event_generations", "rollout_arrival_rate_reads",
                     "rollout_arrival_event_reads", "rollout_tail_segments", "rollout_queue_coordinate_evaluations",
                     "rollout_current_completion_offset_reads"):
            assert getattr(new, name) == 0, name


def test_active_comparison_builds_one_shared_path_and_pays_two_steps_each(monkeypatch):
    options = (Option((1,), 1.0, ((1.0,),)), Option((2,), 2.0, ((2.0, 2.0),)))
    built, evaluated = [], []
    original_build, original_evaluate = arrival.build_arrival_path, arrival.evaluate_arrival_rollout

    def build(*args):
        path = original_build(*args)
        built.append(path)
        return path

    def evaluate(*args):
        result = original_evaluate(*args)
        evaluated.append((args[-1], result))
        return result

    monkeypatch.setattr(arrival, "build_arrival_path", build)
    monkeypatch.setattr(arrival, "evaluate_arrival_rollout", evaluate)
    for rate in (0.0, 0.1):
        built.clear()
        evaluated.clear()
        rates = {"arrival_rates": (rate,), "terminal_service_rates": (0.1,)}
        new = arrival.select_arrival_rollout_frame(options, (2,), 1, 10, **rates)
        assert new.rollout_used and new.rollout_attempts == 4
        assert len(built) == 1 and len(evaluated) == 2
        assert all(path is built[0] for path, _ in evaluated)
        assert built[0].horizon == 6.0
        assert sum(result.attempts for _, result in evaluated) == 4
        assert new.attempts == new.root_attempts + 4 <= 10
        assert new.rollout_horizon == 6.0
        assert new.rollout_population_jobs == 2 + built[0].event_generations
        assert new.rollout_arrival_event_generations == built[0].event_generations
        assert new.rollout_arrival_rate_reads == built[0].rate_reads == 1
        for reported, measured in (("rollout_arrival_event_reads", "arrival_event_reads"),
                                   ("rollout_tail_segments", "tail_segments"),
                                   ("rollout_queue_coordinate_evaluations", "queue_coordinate_evaluations"),
                                   ("rollout_current_completion_offset_reads", "current_completion_offset_reads"),
                                   ("rollout_completion_offset_reads", "completion_offset_reads"),
                                   ("rollout_terminal_value_coordinate_evaluations", "terminal_value_coordinate_evaluations")):
            assert getattr(new, reported) == sum(getattr(result, measured) for _, result in evaluated), reported
        root = select_lookahead_frame(options, (2,), 1, new.root_budget, **rates)
        assert new.completion_offset_reads == (root.completion_offset_reads + new.physical_action_completion_offset_reads
            + new.rollout_completion_offset_reads + new.rollout_current_completion_offset_reads)
        assert new.terminal_value_coordinate_evaluations == (root.terminal_value_coordinate_evaluations
            + new.rollout_terminal_value_coordinate_evaluations)
        for _, result in evaluated:
            assert result.population_error == pytest.approx(0.0, abs=1e-12)
            assert result.horizon_excess == 0.0
            for step in result.steps:
                assert all(isinstance(q, int) and q >= 0 for q in step["queue_before"])
                assert all(d <= q for d, q in zip(step["departures"], step["queue_before"]))
        if rate == 0.0:
            old = select_batch_rollout_frame(options, (2,), 1, 10, **rates)
            assert new.option_indices == old.option_indices
            assert new.selected_rollout_cost == pytest.approx(old.selected_rollout_cost)
            assert new.incumbent_rollout_cost == pytest.approx(old.incumbent_rollout_cost)
            assert new.analytic_challenger_rollout_cost == pytest.approx(old.analytic_challenger_rollout_cost)
        else:
            # Shared t=5 arrival costs 1/(2*.1)=5 in either fluid tail.
            # Known-job costs are 3 for the single incumbent, 4 for the batch.
            assert built[0].times == ((5.0,),)
            assert new.incumbent_rollout_cost == pytest.approx(8.0)
            assert new.analytic_challenger_rollout_cost == pytest.approx(9.0)
            assert new.selected_rollout_cost == pytest.approx(8.0)
            assert all(result.population_jobs == 3 for _, result in evaluated)
