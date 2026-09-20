"""Paid two-step batch continuation, checked without an arrival tape."""
from dataclasses import dataclass
import itertools
from types import SimpleNamespace

import pytest

import algorithm.batch_rollout_frame_candidates as rollout
from algorithm.budgeted_frame_candidates import select_budgeted_frame
from algorithm.experiments.aligned_hardware_frame_model import load_model
from algorithm.lookahead_frame_candidates import select_lookahead_frame


@dataclass(frozen=True)
class Option:
    requirements: tuple[int, ...]
    duration: float
    completion_offsets: tuple[tuple[float, ...], ...]


ONE_CLASS = (Option((1,), 1.0, ((1.0,),)), Option((2,), 2.0, ((2.0, 2.0),)))
RATES = {"arrival_rates": (0.1,), "terminal_service_rates": (0.1,)}


def _support_oracle(options, queues, lanes):
    optimum = 0.0
    for indices in itertools.product((None, *range(len(options))), repeat=lanes):
        service = tuple(sum(options[j].requirements[i] for j in indices if j is not None)
                        for i in range(len(queues)))
        if all(d <= q for d, q in zip(service, queues)):
            duration = max((1.0, *(options[j].duration for j in indices if j is not None)))
            optimum = max(optimum, sum(q * d for q, d in zip(queues, service)) / duration)
    return optimum


def test_real_225_queues_charge_both_branches_and_preserve_current_search(monkeypatch):
    model = load_model()
    rates = {"arrival_rates": (0.05, 0.15, 0.2), "terminal_service_rates": (0.4, 0.9, 1.2)}
    original = rollout._rollout_old_jobs
    residuals = []

    def record_residual(options, residual, *args):
        assert all(isinstance(q, int) and q >= 0 for q in residual)
        residuals.append(tuple(residual))
        return original(options, residual, *args)

    monkeypatch.setattr(rollout, "_rollout_old_jobs", record_residual)
    used = 0
    for queues in itertools.product(range(5), range(5), range(9)):
        optimum = _support_oracle(model.options, queues, model.lanes)
        for budget in (10, 14, 26, 120):
            current_budget = budget - 4 if budget >= 14 else budget
            assert rollout.current_frame_budget(len(model.options), budget) == current_budget
            support = select_budgeted_frame(model.options, queues, model.lanes, current_budget)
            analytic = select_lookahead_frame(model.options, queues, model.lanes, current_budget, **rates)
            residuals.clear()
            chosen = rollout.select_batch_rollout_frame(model.options, queues, model.lanes, budget, **rates)
            context = (queues, budget)
            assert chosen.root_budget == current_budget, context
            assert chosen.root_attempts == support.attempts, context
            assert chosen.rollout_attempts in (0, 4), context
            assert chosen.attempts == support.attempts + chosen.rollout_attempts <= budget, context
            for name in ("seed_attempts", "bound_option_probes", "bound_threshold_evaluations", "frontier_peak"):
                assert getattr(chosen, name) == getattr(support, name), (name, context)
            assert chosen.feasible_candidate_count == support.feasible_candidate_count + chosen.rollout_attempts, context
            assert chosen.local_score_evaluations == support.local_score_evaluations + chosen.rollout_class_evaluations, context
            for name in ("seed_support", "support_floor", "support_cap_excluded_candidate_count",
                         "eligible_candidate_count", "cost_candidate_evaluations", "next_queue_coordinate_evaluations"):
                assert getattr(chosen, name) == getattr(analytic, name), (name, context)
            assert chosen.completion_offset_reads == (analytic.completion_offset_reads
                + chosen.physical_action_completion_offset_reads + chosen.rollout_completion_offset_reads), context
            assert chosen.terminal_value_coordinate_evaluations == (analytic.terminal_value_coordinate_evaluations
                + chosen.rollout_terminal_value_coordinate_evaluations), context
            assert chosen.rollout_cost_candidate_evaluations == 2 * chosen.rollout_used, context
            assert chosen.rollout_terminal_value_coordinate_evaluations == 4 * len(queues) * chosen.rollout_used, context
            assert chosen.rollout_setup_requirement_reads == len(model.options) * len(queues) * chosen.rollout_used, context
            assert len(residuals) == 2 * chosen.rollout_used, context
            if chosen.rollout_used:
                used += 1
                assert residuals == [tuple(q - d for q, d in zip(queues, candidate.service))
                                     for candidate in (analytic, support)], context
                assert chosen.selected_rollout_cost == min(chosen.incumbent_rollout_cost,
                                                           chosen.analytic_challenger_rollout_cost), context
                expected = support if chosen.rollout_vetoed else analytic
                assert chosen.option_indices == expected.option_indices, context
            else:
                assert chosen.option_indices == support.option_indices, context
                assert chosen.selected_rollout_cost is None, context
            assert chosen.root_lookahead_cost == analytic.lookahead_cost, context
            assert chosen.analytic_challenger_score == analytic.score, context
            assert chosen.support_incumbent_score == support.score, context
            assert chosen.score >= chosen.seed_support, context
            assert chosen.score >= 0.99 * support.score, context
            assert chosen.upper_bound == support.upper_bound, context
            assert chosen.support_search_exact == support.exact, context
            service = tuple(sum(model.options[j].requirements[i] for j in chosen.option_indices if j is not None)
                            for i in range(len(queues)))
            duration = max((1.0, *(model.options[j].duration for j in chosen.option_indices if j is not None)))
            assert chosen.service == service and all(d <= q for d, q in zip(service, queues)), context
            assert chosen.duration == duration, context
            assert chosen.score == sum(q * d for q, d in zip(queues, service)) / duration, context
            assert chosen.score <= optimum + 1e-12 <= chosen.upper_bound + 2e-12, context
            assert chosen.gap == max(0.0, chosen.upper_bound - chosen.score), context
            assert chosen.exact == (chosen.gap == 0.0), context
            assert chosen.epsilon == (chosen.gap / sum(queues) if sum(queues) else 0.0), context
    assert used > 0  # The catalogue test must exercise actual paid continuations.


def test_batch_correction_vetoes_the_fluid_challenger_with_four_paid_steps():
    analytic = select_lookahead_frame(ONE_CLASS, (2,), 1, 6, **RATES)
    chosen = rollout.select_batch_rollout_frame(ONE_CLASS, (2,), 1, 10, **RATES)
    assert analytic.option_indices == (1,)
    # Single: A=2.05, EV=((1.1)^2+.1)/.2=6.55, J=8.6.
    # Batch: A=4.2, EV=((.2)^2+.2)/.2=1.2, J=5.4.
    # Single's residual one job clears in 1 second then paid idle: V2=1,
    # replacing Vfluid=5. Batch's empty residual still pays two idle steps.
    assert chosen.support_incumbent_lookahead_cost == pytest.approx(8.6)
    assert chosen.root_lookahead_cost == pytest.approx(5.4)
    assert chosen.incumbent_rollout_cost == pytest.approx(4.6)
    assert chosen.analytic_challenger_rollout_cost == pytest.approx(5.4)
    assert chosen.selected_rollout_cost == pytest.approx(4.6)
    assert chosen.option_indices == (0,)
    assert chosen.lookahead_cost == pytest.approx(8.6)
    assert chosen.rollout_used and chosen.rollout_vetoed
    assert chosen.rollout_attempts == 4
    assert chosen.rollout_cost_candidate_evaluations == 2
    assert chosen.rollout_class_evaluations == 1
    assert chosen.rollout_profile_probes == 2
    assert chosen.rollout_completion_offset_reads == 1
    assert chosen.rollout_terminal_value_coordinate_evaluations == 4
    assert chosen.attempts == chosen.root_attempts + 4 <= 10


def test_insufficient_reserve_returns_support_without_any_future_construction(monkeypatch):
    def forbidden(*args):
        pytest.fail("a continuation was inspected without four reserved attempts")

    monkeypatch.setattr(rollout, "_build_rollout_profiles", forbidden)
    monkeypatch.setattr(rollout, "_rollout_old_jobs", forbidden)
    support = select_budgeted_frame(ONE_CLASS, (2,), 1, 6)
    analytic = select_lookahead_frame(ONE_CLASS, (2,), 1, 6, **RATES)
    assert support.option_indices != analytic.option_indices
    chosen = rollout.select_batch_rollout_frame(ONE_CLASS, (2,), 1, 6, **RATES)
    assert chosen.root_budget == 6
    assert chosen.option_indices == support.option_indices
    assert chosen.attempts == support.attempts
    assert chosen.rollout_attempts == chosen.rollout_setup_requirement_reads == 0
    assert chosen.selected_rollout_cost is None


def test_identical_current_actions_skip_reserved_rollouts(monkeypatch):
    def forbidden(*args):
        pytest.fail("identical current actions need no continuation comparison")

    monkeypatch.setattr(rollout, "_build_rollout_profiles", forbidden)
    monkeypatch.setattr(rollout, "_rollout_old_jobs", forbidden)
    chosen = rollout.select_batch_rollout_frame(ONE_CLASS, (1,), 1, 10, **RATES)
    assert chosen.root_budget == 6
    assert chosen.option_indices == (0,)
    assert not chosen.rollout_used
    assert chosen.rollout_attempts == chosen.rollout_cost_candidate_evaluations == 0
    assert chosen.attempts == chosen.root_attempts


class CountedTuple(tuple):
    def __new__(cls, values, reads, key):
        instance = super().__new__(cls, values)
        instance.reads, instance.key = reads, key
        return instance

    def __iter__(self):
        for value in super().__iter__():
            self.reads[self.key] = self.reads.get(self.key, 0) + 1
            yield value


def test_two_frames_use_integer_batches_and_only_probe_the_selected_profiles():
    reads = {}
    counts = ((1, 0), (3, 0), (8, 0), (0, 1), (0, 2), (1, 1))
    durations = (1.0, 2.0, 5.0, 1.0, 1.5, 1.0)
    requirements = tuple(CountedTuple(row, reads, "requirements") for row in counts)
    profiles, setup_reads = rollout._build_rollout_profiles(requirements, durations)
    assert setup_reads == reads["requirements"] == 12
    assert profiles == (((8, 2), (3, 1), (1, 0)), ((2, 4), (1, 3)))
    profiles = tuple(CountedTuple(row, reads, "profiles") for row in profiles)
    offsets = (((1.0,), ()), ((1.0, 1.5, 2.0), ()), ((5.0,) * 8, ()),
               ((), (1.0,)), ((), (1.5, 1.5)), ((1.0,), (1.0,)))
    # No requirement/duration attributes are provided: continuation must reuse
    # its paid index, and offset reads must touch only the constructed frames.
    options = tuple(SimpleNamespace(completion_offsets=tuple(
        CountedTuple(group, reads, f"offsets_{index}") for group in groups))
        for index, groups in enumerate(offsets))
    result = rollout._rollout_old_jobs(options, (5, 4), 2, 1.0, (1.0, 2.0), profiles, durations)
    # Frame 1: priorities 5 and 2, batch 3 -> residual (2,4), area 6*2+4.5.
    # Frame 2: tied priorities 2 and 2, two singles -> (0,4), area 4*1+2.
    # Initial fluid=25/2+16/4=16.5; tail=16/4=4.
    assert result.holding_area == pytest.approx(22.5)
    assert result.initial_fluid_value == pytest.approx(16.5)
    assert result.tail_fluid_value == pytest.approx(4.0)
    assert result.correction == pytest.approx(10.0)
    assert result.attempts == 2
    assert result.class_evaluations == 4
    assert result.profile_probes == reads["profiles"] == 5
    assert result.completion_offset_reads == reads["offsets_0"] + reads["offsets_1"] == 5
    assert not any(reads.get(f"offsets_{index}", 0) for index in (2, 3, 4, 5))
    assert reads["requirements"] == 12
    assert result.terminal_value_coordinate_evaluations == 4


def test_empty_old_backlog_still_pays_two_idle_frames():
    result = rollout._rollout_old_jobs((), (0, 0), 2, 1.0, (1.0, 2.0), ((), ()), ())
    assert result.attempts == 2
    assert result.holding_area == result.initial_fluid_value == result.tail_fluid_value == result.correction == 0
    assert result.class_evaluations == result.profile_probes == result.completion_offset_reads == 0
    assert result.terminal_value_coordinate_evaluations == 4


def test_physical_equivalence_uses_completion_offsets_not_only_total_service():
    options = (Option((1,), 2.0, ((1.0,),)), Option((1,), 2.0, ((2.0,),)),
               Option((2,), 2.0, ((1.5, 1.5),)))
    left = SimpleNamespace(indices=(0, 1), service=(2,), duration=2.0)
    permuted = SimpleNamespace(indices=(1, 0), service=(2,), duration=2.0)
    different = SimpleNamespace(indices=(2, None), service=(2,), duration=2.0)
    assert rollout._same_physical_action(left, permuted, options) == (True, 4)
    # Equal area and equal D/tau do not imply the same individual completions.
    assert rollout._same_physical_action(left, different, options) == (False, 4)
