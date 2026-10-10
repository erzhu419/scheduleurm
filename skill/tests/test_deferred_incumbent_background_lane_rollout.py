"""A scored next-call commitment must execute using only its visible state."""
from dataclasses import replace
from fractions import Fraction

import pytest

import algorithm.background_competition_continuous_lane_rollout as background_module
import algorithm.deferred_incumbent_background_lane_rollout as module
from algorithm.arrival_aware_lane_rollout import PlanningArrivals
from algorithm.background_competition_continuous_lane_rollout import BackgroundCompetitionContinuousLaneRollout
from algorithm.deferred_incumbent_background_lane_rollout import DeferredIncumbentBackgroundLaneRollout
from algorithm.experiments.aligned_hardware_frame_model import FrameOption


SINGLES = (FrameOption("a", (1, 0), 2., ((2.,), ())),
           FrameOption("b", (0, 1), 1., ((), (1.,))))
OPTIONS = (*SINGLES, FrameOption("a2", (2, 0), 3., ((3., 3.), ())))


def population(*events):
    return PlanningArrivals(tuple(events), False, 0, 0)


def scripted_background(monkeypatch, gaps):
    instances = []

    class ScriptedRandom:
        def __init__(self, seed):
            self.gaps = iter(gaps)
            self.rates = []
            instances.append(self)

        def expovariate(self, rate):
            self.rates.append(rate)
            return next(self.gaps)

    monkeypatch.setattr(background_module, "Random", ScriptedRandom)
    return instances


def plan_fixture(monkeypatch, *, root=1, fixed_cost=9, fixed_square=90,
                 ordinary_cost=20):
    """Exercise decision/execution rules with specified exact plan contrasts.

    Keep the real scorer's feasible plan set and work counters. Only the exact
    cost contrast is supplied, so tests can isolate carry/execution behavior.
    """
    candidate = DeferredIncumbentBackgroundLaneRollout(OPTIONS, "pending", sample_count=1)
    actual_score = candidate.score_deferred_roots

    def scored(*args):
        actual = actual_score(*args)
        incumbent = args[-1]
        costs = dict.fromkeys(actual.costs, Fraction(20))
        squared = dict.fromkeys(actual.squared_costs, Fraction(100))
        costs[incumbent, None] = Fraction(10)
        costs[root, None] = Fraction(ordinary_cost)
        # When root is the current incumbent, its ordinary cost is the control.
        costs[incumbent, None] = Fraction(10)
        costs[root, incumbent] = Fraction(fixed_cost)
        squared[root, incumbent] = Fraction(fixed_square)
        return replace(actual, costs=costs, squared_costs=squared)

    monkeypatch.setattr(candidate, "score_deferred_roots", scored)
    args = (OPTIONS, (2, 1), (1000., 1009., 1009.), 1000., 0,
            ((10., 9.), (0.,)), (0, 0))
    return candidate, args


def forbid_resampling(monkeypatch, candidate):
    def forbidden(*args):
        raise AssertionError("a pending successor must not replan or sample future arrivals")

    monkeypatch.setattr(module, "sample_planning_arrivals", forbidden)
    monkeypatch.setattr(candidate, "score_deferred_roots", forbidden)


def test_original_background_plan_is_preserved_with_shared_suffix_and_tagged_only_costs(monkeypatch):
    instances = scripted_background(monkeypatch, (.5, 100.))
    candidate = DeferredIncumbentBackgroundLaneRollout(SINGLES, "plans", sample_count=1)
    candidate.arrival_rates = (0., 1.)
    state = ((2, 1), (1000., 1004.), 1000., 0, ((5., 1.), (2.,)),
             (population((2., 0)),))
    rng_before = candidate.rng.getstate()
    scored = candidate.score_deferred_roots(*state, 1)
    assert {root: scored.costs[root, None] for root in (0, 1)} == {
        0: Fraction(23), 1: Fraction(22)}
    assert {root: scored.squared_costs[root, None] for root in (0, 1)} == {
        0: Fraction(139), 1: Fraction(138)}
    assert scored.costs[0, 1] == scored.costs[0, None]
    assert scored.costs[1, 1] == scored.costs[1, None]
    assert scored.candidate_count == 2 and scored.plan_candidate_count == 4
    assert scored.rollout_calls == 16 and scored.rollout_option_probes == 32
    assert scored.flow_evaluations == 16 and scored.background_admissions == 4
    assert scored.admitted_job_reads == scored.tag_flag_reads == 20
    assert scored.background_planning_events == 1
    assert scored.background_planning_random_draws == 2
    assert scored.background_seed_initializations == len(instances) == 1
    assert candidate.rng.getstate() == rng_before
    # Independent old scorer confirms all original MW continuation branches.
    old = BackgroundCompetitionContinuousLaneRollout(SINGLES, "plans", sample_count=1)
    old.arrival_rates = (0., 1.)
    original = old.score_background_roots(*state)
    assert all(scored.costs[root, None] == original.costs[root] and
               scored.squared_costs[root, None] == original.squared_costs[root]
               for root in original.costs)


def test_fixed_incumbent_changes_only_first_nonempty_successor_and_keeps_complete_offsets():
    candidate = DeferredIncumbentBackgroundLaneRollout(SINGLES, "fixed", sample_count=1)
    scored = candidate.score_deferred_roots(
        (2, 0), (1000., 1004.), 1000., 0, ((5., 1.), ()),
        (population((1., 1)),), 0)
    # Ordinary MW serves B at t=2 then A at t=3. Fixed A consumes the one
    # commitment at t=2, completing A at t=4; ordinary MW then serves B at t=4.
    assert scored.costs == {(0, None): Fraction(15), (0, 0): Fraction(16)}
    assert scored.squared_costs == {(0, None): Fraction(89), (0, 0): Fraction(90)}
    assert scored.rollout_calls == 4 and scored.flow_evaluations == 6
    assert scored.background_admissions == 0


def test_loaded_root_uses_fifo_completion_offsets_without_releasing_lane_early():
    options = (FrameOption("loaded", (1, 1), 4., ((1.,), (3.,)), loaded=True),
               *SINGLES)
    candidate = DeferredIncumbentBackgroundLaneRollout(options, "loaded", sample_count=1)
    scored = candidate.score_deferred_roots(
        (2, 1), (1000., 1010.), 1000., 0, ((5., 1.), (2.,)),
        (population((3., 1)),), 2)
    # Loaded A finishes at t=1 and B at t=3, but the lane stays reserved until
    # t=4. Then the newly born B completes at t=5 and remaining A at t=7.
    # Oldest A binds to the loaded offset: tagged flows are 6, 5, 2, 8.
    assert scored.costs[0, None] == scored.costs[0, 2] == Fraction(21)
    assert scored.squared_costs[0, None] == scored.squared_costs[0, 2] == Fraction(129)


def test_chosen_successor_executes_against_new_visible_mw_without_replanning(monkeypatch):
    candidate, initial = plan_fixture(monkeypatch)
    first = candidate(*initial)
    assert first.selection.option_indices == (1,) and first.changed
    assert first.plan_changed and first.planned_successor_index == 2
    assert candidate.pending_successor_index == 2
    forbid_resampling(monkeypatch, candidate)
    rng_before = candidate.rng.getstate()
    current = (OPTIONS, (2, 3), (1001., 1009., 1009.), 1001., 0,
               ((11., 10.), (.5, .25, .1)), (2, 4))
    assert candidate.maxweight(*current).selection.option_indices == (1,)
    result = candidate(*current)
    assert result.selection.option_indices == (2,)
    assert result.successor_call and result.successor_executed and result.successor_changed
    assert not result.changed and not result.eligible
    assert result.planned_successor_index == 2 and candidate.pending_successor_index is None
    assert result.root_option_probes == len(OPTIONS)
    assert result.rollout_calls == result.sample_count == result.planning_random_draws == 0
    assert candidate.rng.getstate() == rng_before


@pytest.mark.parametrize("queues,ages", [
    ((1, 2), ((11.,), (.5, .25))),
    ((2, 5), ((11., 10.), (.5, .4, .3, .2, .1))),
])
def test_empty_call_holds_but_infeasible_or_over_envelope_nonempty_call_consumes(monkeypatch, queues, ages):
    candidate, initial = plan_fixture(monkeypatch)
    candidate(*initial)
    forbid_resampling(monkeypatch, candidate)
    rng_before = candidate.rng.getstate()
    empty = candidate(OPTIONS, (0, 0), (1001., 1009., 1009.), 1001., 0,
                      ((), ()), (2, 1))
    assert empty.selection.option_indices == (None,)
    assert candidate.pending_successor_index == 2
    assert empty.successor_envelope_checks == empty.successor_feasibility_checks == 0
    assert empty.planning_random_draws == empty.sample_count == 0
    current = (OPTIONS, queues, (1002., 1009., 1009.), 1002., 0, ages, (3, 6))
    expected = candidate.maxweight(*current).selection
    result = candidate(*current)
    assert result.selection == expected
    assert result.successor_call and not result.successor_executed
    assert not result.successor_changed and not result.changed
    assert result.planned_successor_index == 2 and candidate.pending_successor_index is None
    assert result.root_option_probes == len(OPTIONS) and result.rollout_calls == 0
    assert candidate.rng.getstate() == rng_before


def test_same_current_root_can_carry_a_strictly_better_successor(monkeypatch):
    candidate, args = plan_fixture(monkeypatch, root=2)
    result = candidate(*args)
    assert result.selection.option_indices == candidate.maxweight(*args).selection.option_indices == (2,)
    assert not result.changed and result.plan_changed
    assert result.planned_successor_index == candidate.pending_successor_index == 2
    assert result.chosen_predicted_completion_s < result.baseline_predicted_completion_s
    assert result.chosen_predicted_squared_flow_s2 <= result.baseline_predicted_squared_flow_s2


@pytest.mark.parametrize("fixed_cost,fixed_square,ordinary_cost,expected_root", [
    (9, 101, 20, 2),  # S1 improvement cannot bypass the MW/MW S2 control.
    (10, 90, 20, 2),  # Equal S1 preserves the complete baseline, with no carry.
    (9, 90, 9, 1),    # Equal nonbaseline scores prefer an ordinary MW suffix.
])
def test_risk_and_strict_improvement_rules_do_not_leave_a_commit(monkeypatch, fixed_cost,
                                                               fixed_square, ordinary_cost,
                                                               expected_root):
    candidate, args = plan_fixture(monkeypatch, fixed_cost=fixed_cost,
        fixed_square=fixed_square, ordinary_cost=ordinary_cost)
    result = candidate(*args)
    assert result.selection.option_indices == (expected_root,)
    assert result.planned_successor_index is None and candidate.pending_successor_index is None
    assert result.risk_feasible_candidates + result.risk_filtered_candidates == result.plan_candidate_count
    assert result.chosen_predicted_squared_flow_s2 <= result.baseline_predicted_squared_flow_s2


def test_combined_overflow_discards_partial_plans_and_preserves_actual_work(monkeypatch):
    candidate = DeferredIncumbentBackgroundLaneRollout(
        OPTIONS, "overflow", sample_count=1, event_cap=2)
    scripted_background(monkeypatch, (100., .1, .1))
    monkeypatch.setattr(module, "sample_planning_arrivals",
        lambda *args: PlanningArrivals(((1., 0),), False, 2, 2))
    args = (OPTIONS, (2, 1), (1000., 1005.), 1000., 0,
            ((5., 1.), (2.,)), (200, 100))
    result = candidate(*args)
    assert result.selection == candidate.maxweight(*args).selection
    assert result.planning_overflow and result.background_rollout_overflow
    assert not result.changed and not result.plan_changed
    assert result.planned_successor_index is None and candidate.pending_successor_index is None
    assert result.planning_events == 3 and result.tagged_planning_events == 1
    assert result.background_planning_events == 2
    assert result.planning_random_draws == result.planning_rate_reads == 5
    assert result.rollout_calls == 1 and result.rollout_option_probes == len(OPTIONS)
    assert result.flow_evaluations == result.admitted_job_reads == result.tag_flag_reads == 3
    assert result.baseline_predicted_completion_s is result.chosen_predicted_completion_s is None
    assert result.risk_feasible_candidates == result.risk_filtered_candidates == 0
