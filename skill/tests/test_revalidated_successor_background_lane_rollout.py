"""A deferred successor must earn acceptance again from its current causal state."""
from dataclasses import asdict, replace
from fractions import Fraction

import pytest

import algorithm.background_competition_continuous_lane_rollout as background_module
import algorithm.deferred_incumbent_background_lane_rollout as deferred_module
import algorithm.revalidated_successor_background_lane_rollout as module
from algorithm.arrival_aware_lane_rollout import PlanningArrivals
from algorithm.deferred_incumbent_background_lane_rollout import DeferredIncumbentBackgroundLaneRollout
from algorithm.experiments.aligned_hardware_frame_model import FrameOption
from algorithm.revalidated_successor_background_lane_rollout import RevalidatedSuccessorBackgroundLaneRollout


OPTIONS = (FrameOption("a", (1, 0), 2., ((2.,), ())),
           FrameOption("b", (0, 1), 1., ((), (1.,))),
           FrameOption("a2", (2, 0), 3., ((3., 3.), ())))


def scripted_background(monkeypatch, gaps):
    instances = []

    class ScriptedRandom:
        def __init__(self, seed):
            self.gaps = iter(gaps)
            instances.append(self)

        def expovariate(self, rate):
            return next(self.gaps)

    monkeypatch.setattr(background_module, "Random", ScriptedRandom)
    return instances


def test_two_root_scores_preserve_loaded_reservations_fifo_and_tagged_only_background_cost(monkeypatch):
    options = (FrameOption("loaded", (1, 1), 4., ((1.,), (3.,)), loaded=True),
               *OPTIONS[:2])
    candidate = RevalidatedSuccessorBackgroundLaneRollout(options, "two", sample_count=1)
    candidate.arrival_rates = (0., 1.)
    instances = scripted_background(monkeypatch, (.5, 100.))
    rng_before = candidate.rng.getstate()
    scored = candidate.score_successor_roots(
        (2, 1), (1000., 1006.), 1000., 0, ((5., 1.), (2.,)),
        (PlanningArrivals(((4., 0),), False, 0, 0),), 0, 2)
    # Only loaded0 and MW B2 are scored; the feasible singleton A1 is excluded.
    # Loaded A/B complete at1/3 but its lane stays reserved until4. At6 the
    # other committed lane admits the remaining tagged A despite BG B service.
    assert scored.costs == {0: Fraction(22), 2: Fraction(21)}
    assert scored.squared_costs == {0: Fraction(126), 2: Fraction(125)}
    assert scored.candidate_count == scored.root_option_probes == 2
    assert scored.rollout_calls == 7 and scored.rollout_option_probes == 7 * len(options)
    assert scored.flow_evaluations == 8 and scored.background_admissions == 2
    assert scored.admitted_job_reads == scored.tag_flag_reads == 10
    assert scored.background_planning_events == 1 and scored.background_planning_random_draws == 2
    assert scored.background_seed_initializations == len(instances) == 1
    assert candidate.rng.getstate() == rng_before


@pytest.mark.parametrize("pending_cost,pending_square,accepted", [
    (19, 100, True),
    (20, 90, False),   # Lower S2 cannot turn an S1 tie into strict gain.
    (21, 90, False),
    (19, 101, False),  # S1 gain cannot bypass the current MW-root S2 gate.
])
def test_revalidation_uses_current_queue_ages_commitments_and_causal_rates(monkeypatch,
                                                                        pending_cost,
                                                                        pending_square,
                                                                        accepted):
    candidate = RevalidatedSuccessorBackgroundLaneRollout(OPTIONS, "causal")
    candidate.pending_successor_index = 2
    prefix = PlanningArrivals(((.5, 1),), False, 2, 2)
    sampled, scored_contexts = [], []

    def sampled_path(rates, window, rng, cap):
        sampled.append((tuple(rates), window, cap))
        return prefix

    actual_score = candidate.score_successor_roots

    def scored(*args):
        scored_contexts.append(args)
        actual = actual_score(*args)
        # Supply an exact objective contrast while retaining the real two-root
        # feasible set and all sampling, continuation and per-job accounting.
        return replace(actual, costs={1: Fraction(20), 2: Fraction(pending_cost)},
                       squared_costs={1: Fraction(100), 2: Fraction(pending_square)})

    monkeypatch.setattr(module, "sample_planning_arrivals", sampled_path)
    monkeypatch.setattr(candidate, "score_successor_roots", scored)
    current = (OPTIONS, (2, 3), (1000., 1009., 1011.), 1000., 0,
               ((11., 10.), (.5, .25, .1)), (200, 100))
    assert candidate.maxweight(*current).selection.option_indices == (1,)
    result = candidate(*current)
    assert sampled == [((.2, .1), 3., 128)] * 8
    assert len(scored_contexts) == 1
    seen = scored_contexts[0]
    assert seen[:5] == current[1:6]
    assert tuple(seen[5]) == (prefix,) * 8 and seen[-2:] == (2, 1)
    assert candidate.arrival_rates == (.2, .1)
    assert candidate.decision_counter == 1 and candidate.pending_successor_index is None
    assert result.successor_revalidated and result.successor_rejected == (not accepted)
    assert result.successor_executed == result.successor_changed == accepted
    assert result.selection.option_indices == ((2,) if accepted else (1,))
    assert result.candidate_count == 2 and result.sample_count == 8
    assert result.root_option_probes == len(OPTIONS) + 2
    assert result.tagged_planning_events == 8 and result.planning_random_draws >= 16
    assert result.chosen_predicted_completion_s <= result.baseline_predicted_completion_s
    assert result.chosen_predicted_squared_flow_s2 <= result.baseline_predicted_squared_flow_s2


@pytest.mark.parametrize("queue,ages,same_mw", [
    ((1, 2), ((.5,), (.5, .25)), False),
    ((2, 5), ((.5, .25), (.5, .4, .3, .2, .1)), False),
    ((2, 1), ((.5, .25), (.5,)), True),
])
def test_empty_hold_and_guard_failure_or_same_mw_skip_all_sampling(monkeypatch, queue, ages, same_mw):
    candidate = RevalidatedSuccessorBackgroundLaneRollout(OPTIONS, "guards")
    candidate.pending_successor_index = 2

    def forbidden(*args):
        raise AssertionError("empty/failed guards/same-MW calls must not sample or score")

    monkeypatch.setattr(module, "sample_planning_arrivals", forbidden)
    monkeypatch.setattr(deferred_module, "sample_planning_arrivals", forbidden)
    monkeypatch.setattr(candidate, "score_successor_roots", forbidden)
    rng_before = candidate.rng.getstate()
    empty = candidate(OPTIONS, (0, 0), (1000., 1009., 1011.), 1000., 0,
                      ((), ()), (0, 0))
    assert empty.selection.option_indices == (None,) and candidate.pending_successor_index == 2
    assert candidate.decision_counter == 1
    assert empty.successor_envelope_checks == empty.successor_feasibility_checks == 0
    current = (OPTIONS, queue, (1001., 1009., 1011.), 1001., 0, ages, queue)
    control = candidate.maxweight(*current)
    result = candidate(*current)
    assert result.selection == control.selection
    assert result.successor_executed == same_mw and not result.successor_changed
    assert not result.successor_revalidated
    assert result.sample_count == result.planning_events == result.rollout_calls == 0
    assert result.root_option_probes == len(OPTIONS)
    assert candidate.pending_successor_index is None and candidate.decision_counter == 2
    assert candidate.rng.getstate() == rng_before


def test_nonpending_joint_planning_remains_the_frozen_deferred_policy():
    candidate = RevalidatedSuccessorBackgroundLaneRollout(OPTIONS, "initial", sample_count=1)
    original = DeferredIncumbentBackgroundLaneRollout(OPTIONS, "initial", sample_count=1)
    current = (OPTIONS, (2, 1), (1000., 1009.), 1000., 0, ((10., 9.), (0.,)), (0, 0))
    result, reference = candidate(*current), original(*current)
    original_fields = asdict(result)
    assert not original_fields.pop("successor_revalidated")
    assert not original_fields.pop("successor_rejected")
    assert original_fields == asdict(reference)
    assert candidate.pending_successor_index == original.pending_successor_index
    assert candidate.rng.getstate() == original.rng.getstate()
    assert candidate.decision_counter == original.decision_counter


def test_background_overflow_rejects_pending_and_keeps_partial_actual_fees(monkeypatch):
    candidate = RevalidatedSuccessorBackgroundLaneRollout(
        OPTIONS, "overflow", sample_count=1, event_cap=2)
    candidate.pending_successor_index = 0
    scripted_background(monkeypatch, (100., .1, .1))
    monkeypatch.setattr(module, "sample_planning_arrivals",
        lambda *args: PlanningArrivals(((1., 0),), False, 2, 2))
    current = (OPTIONS, (2, 1), (1000., 1005.), 1000., 0,
               ((5., 1.), (2.,)), (200, 100))
    result = candidate(*current)
    assert result.selection == candidate.maxweight(*current).selection
    assert result.successor_revalidated and result.successor_rejected
    assert result.planning_overflow and result.background_rollout_overflow
    assert not result.successor_executed and not result.successor_changed
    assert candidate.pending_successor_index is None and candidate.decision_counter == 1
    assert result.planning_events == 3 and result.tagged_planning_events == 1
    assert result.background_planning_events == 2
    assert result.planning_random_draws == result.planning_rate_reads == 5
    assert result.rollout_calls == 1 and result.rollout_option_probes == len(OPTIONS)
    assert result.root_option_probes == len(OPTIONS) + 2
    assert result.flow_evaluations == result.admitted_job_reads == result.tag_flag_reads == 3
    assert result.baseline_predicted_completion_s is result.chosen_predicted_completion_s is None
