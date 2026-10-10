"""One later candidate decision must use the causal FIFO continuation state."""
from fractions import Fraction

from algorithm.age_tail_constrained_lane_rollout import AgeRootScores
from algorithm.arrival_aware_lane_rollout import PlanningArrivals
from algorithm.continuous_arrival_age_tail_lane_rollout import ContinuousArrivalAgeTailLaneRollout
from algorithm.experiments.aligned_hardware_frame_model import FrameOption
from algorithm.redecision_continuous_lane_rollout import RedecisionContinuousLaneRollout


OPTIONS = (FrameOption("a1", (1, 0), 2., ((2.,), ())),
           FrameOption("a2", (2, 0), 3., ((3., 3.), ())),
           FrameOption("b1", (0, 1), 1., ((), (1.,))))


def population(*events):
    return PlanningArrivals(tuple(events), False, 0, 0)


class ContinuationSpy:
    def __init__(self, planner, *, use_candidate=False):
        self.planner = planner
        self.use_candidate = use_candidate
        self.calls = []
        self.decisions = []

    def __getattr__(self, name):
        return getattr(self.planner, name)

    @property
    def rng(self):
        return self.planner.rng

    @rng.setter
    def rng(self, value):
        self.planner.rng = value

    def __call__(self, options, queue, available, now, lane, ages, counts):
        self.calls.append((tuple(queue), tuple(available), now, lane,
                           tuple(tuple(group) for group in ages), tuple(counts),
                           self.rng.getstate()))
        method = self.planner if self.use_candidate else self.planner.maxweight
        decision = method(options, queue, available, now, lane, ages, counts)
        self.decisions.append(decision)
        return decision


def with_spy(options=OPTIONS, *, use_candidate=False):
    candidate = RedecisionContinuousLaneRollout(options, "redecision-test")
    spy = ContinuationSpy(candidate.continuation, use_candidate=use_candidate)
    candidate.continuation = spy
    return candidate, spy


def test_inner_context_keeps_fifo_busy_lane_and_same_time_releases():
    candidate, spy = with_spy()
    scored = candidate.score_redecision_roots(
        (2, 1), (1000., 1003.), 1000., 0, ((5., 1.), (2.,)), (10, 20),
        (population((2., 1), (2.5, 0), (3., 1), (4., 0)),), (0, 1))
    assert [call[:6] for call in spy.calls] == [
        ((1, 2), (1002., 1003.), 1002., 0, ((3.,), (4., 0.)), (10, 21)),
        ((1, 3), (1003., 1003.), 1003., 0, ((.5,), (5., 1., 0.)), (11, 22)),
    ]
    # Identical hypothetical random streams couple the two roots in sample0.
    assert spy.calls[0][6] == spy.calls[1][6]
    assert scored.costs == {0: Fraction(59, 2), 1: Fraction(59, 2)}
    assert scored.squared_costs == {0: Fraction(609, 4), 1: Fraction(613, 4)}


def test_outer_unreleased_future_cannot_change_inner_context_or_rng():
    state = ((2, 1), (1000., 1003.), 1000., 0, ((5., 1.), (2.,)), (10, 20))
    calls = []
    for future in (((10., 0),), ((11., 1), (12., 1))):
        candidate, spy = with_spy()
        outer_rng = candidate.rng.getstate()
        candidate.score_redecision_roots(*state, (population((2., 1), *future),), (0,))
        assert candidate.rng.getstate() == outer_rng
        assert len(spy.calls) == 1
        calls.append(spy.calls[0])
    assert calls[0] == calls[1]


def test_over_envelope_continuation_is_called_once_then_plain_mw_drains():
    options = (FrameOption("single", (1,), 2., ((2.,),)),)
    candidate, spy = with_spy(options, use_candidate=True)
    scored = candidate.score_redecision_roots(
        (1,), (1000.,), 1000., 0, ((5.,),), (20,),
        (population((.5, 0), (1.5, 0), (3.5, 0)),), (0,))
    assert len(spy.calls) == 1
    assert spy.calls[0][:6] == ((2,), (1002.,), 1002., 0, ((1.5, .5),), (22,))
    assert not spy.decisions[0].eligible and not spy.decisions[0].changed
    assert scored.rollout_calls == 3
    assert scored.costs[0] == Fraction(39, 2)
    assert scored.squared_costs[0] == Fraction(407, 4)


def test_idle_polling_does_not_consume_the_single_inner_decision():
    options = (FrameOption("single", (1,), 2., ((2.,),)),)
    candidate, spy = with_spy(options)
    scored = candidate.score_redecision_roots(
        (1,), (1000.,), 1000., 0, ((5.,),), (20,),
        (population((10.25, 0)),), (0,))
    assert len(spy.calls) == 1
    assert spy.calls[0][:6] == ((1,), (1011.,), 1011., 0, ((.75,),), (21,))
    assert scored.rollout_idle_calls == 9
    assert scored.costs[0] == Fraction(39, 4)
    assert scored.squared_costs[0] == Fraction(905, 16)


def fixed_scores(costs, squared):
    return AgeRootScores(costs, squared, len(costs), 0, 0, 0, 0, 8, 3, 0)


def test_pair_squared_flow_rejection_restores_exact_mw(monkeypatch):
    candidate = RedecisionContinuousLaneRollout(OPTIONS, "rejection-test")
    state = (OPTIONS, (2, 1), (1000., 1003.), 1000., 0,
             ((0., 0.), (20.,)), (0, 0))
    baseline = candidate.maxweight(*state)
    monkeypatch.setattr(ContinuousArrivalAgeTailLaneRollout, "score_age_roots",
        lambda *args: fixed_scores({0: Fraction(12), 1: Fraction(11), 2: Fraction(10)},
                                  {0: Fraction(110), 1: Fraction(100), 2: Fraction(99)}))
    calls = []

    def pair(*args):
        calls.append(args[-1])
        return fixed_scores({1: Fraction(9), 2: Fraction(8)},
                            {1: Fraction(100), 2: Fraction(101)})

    monkeypatch.setattr(candidate, "score_redecision_roots", pair)
    result = candidate(*state)
    assert len(calls) == 1 and set(calls[0]) == {1, 2}
    assert not result.changed and result.selection == baseline.selection
    assert result.selection.canonical_exact and result.selection.score_exact
    assert result.chosen_predicted_squared_flow_s2 <= result.baseline_predicted_squared_flow_s2
    assert result.redecision_attempted and result.redecision_rejected
    assert result.chosen_redecision_flow_s == result.baseline_redecision_flow_s
    assert result.chosen_redecision_squared_flow_s2 == result.baseline_redecision_squared_flow_s2
    assert result.redecision_work["restore_control_option_probes"] == len(OPTIONS)


def test_unchanged_outer_proposal_does_not_start_second_search(monkeypatch):
    candidate = RedecisionContinuousLaneRollout(OPTIONS, "unchanged-test")

    def unexpected(*args):
        raise AssertionError("unchanged MW proposal must not rerank")

    monkeypatch.setattr(candidate, "score_redecision_roots", unexpected)
    result = candidate(OPTIONS, (2, 1), (1000., 1003.), 1000., 0,
                       ((10., 10.), (0.,)), (0, 0))
    assert not result.changed and result.selection.option_indices == (1,)
