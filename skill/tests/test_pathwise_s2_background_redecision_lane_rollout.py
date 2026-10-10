"""Samplewise S2 gates cannot hide a losing path in an aggregate gain."""
from dataclasses import replace
from fractions import Fraction
from itertools import chain, repeat

import pytest

import algorithm.background_competition_continuous_lane_rollout as background_module
import algorithm.pathwise_s2_background_redecision_lane_rollout as module
from algorithm.arrival_aware_lane_rollout import PlanningArrivals
from algorithm.background_competition_continuous_lane_rollout import _LazyBackgroundArrivals
from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption
from algorithm.experiments.aligned_hardware_pathwise_s2_background_redecision_development import (
    check_rollout, pathwise_work,
)
from algorithm.experiments.observed_rate_contextual_lane_refill_replay import simulate_observed_rate_contextual_trajectory
from algorithm.pathwise_s2_background_redecision_lane_rollout import PathwiseS2BackgroundRedecisionLaneRollout


OPTIONS = (FrameOption("single", (1,), 1., ((1.,),)),
           FrameOption("batch", (2,), 1.5, ((1.5, 1.5),)))


def population(*events):
    return PlanningArrivals(tuple(events), False, 0, 0)


def script_background(monkeypatch, gaps=(1000.,)):
    class ScriptedRandom:
        def __init__(self, seed):
            self.gaps = chain(gaps, repeat(1000.))

        def expovariate(self, rate):
            return next(self.gaps)

    class ScriptedSuffix(_LazyBackgroundArrivals):
        def release_through(self, start):
            original = background_module.Random
            background_module.Random = ScriptedRandom
            try:
                return super().release_through(start)
            finally:
                background_module.Random = original

    # Outer scoring is still the exact one-step physical simulator, while
    # ordinary inner BG samples retain their independent tagged population.
    monkeypatch.setattr(module, "_LazyBackgroundArrivals", ScriptedSuffix)
    monkeypatch.setattr(background_module, "_LazyBackgroundArrivals", ScriptedSuffix)


def empty_prefixes(monkeypatch):
    monkeypatch.setattr(background_module, "sample_planning_arrivals", lambda *args: population())
    monkeypatch.setattr(module, "sample_planning_arrivals", lambda *args: population())


def decision_args(options=OPTIONS):
    return (options, (2,), (1000., 1005.), 1000., 0, ((0., 0.),), (20,))


def test_native_distinct_prefixes_record_exact_sample_s2_and_aggregate(monkeypatch):
    options = OPTIONS[:1]
    candidate = PathwiseS2BackgroundRedecisionLaneRollout(options, "samples", sample_count=2)
    candidate.arrival_rates = (.02,)
    candidate._outer_observed_counts = (20,)
    script_background(monkeypatch)
    empty_prefixes(monkeypatch)
    before = candidate.rng.getstate()
    scored = candidate.score_background_roots(
        (2,), (1000., 1005.), 1000., 0, ((5., 1.),),
        (population(), population((1., 0))))
    # Same visible FIFO: flows6,3 in both paths; the second additionally
    # completes its A@1 at3, giving flow2 and exactly four extra S2 units.
    assert scored.costs == {0: Fraction(20)}
    assert scored.per_sample_squared_flow_costs == {0: (Fraction(45), Fraction(49))}
    assert scored.squared_costs == {0: Fraction(94)}
    assert scored.flow_evaluations == 5
    assert candidate.rng.getstate() == before and candidate.decision_counter == 0


@pytest.mark.parametrize("sample_squares,total_cost,changed,comparisons", [
    ((7, 1), 18, False, 3),   # Aggregate8 beats12, but sample0 loses: stop early.
    ((1, 11), 18, False, 4),  # Aggregate tie12 also hides a loss in sample1.
    ((5, 6), 18, True, 4),    # Both paths pass, including equality in sample1.
    ((5, 6), 20, False, 4),   # Every path passes, but S1 ties keep MW.
])
def test_pathwise_gate_rejects_hidden_losses_and_retains_mw_strict_ties(
        monkeypatch, sample_squares, total_cost, changed, comparisons):
    candidate = PathwiseS2BackgroundRedecisionLaneRollout(OPTIONS, "contrast", sample_count=2)
    script_background(monkeypatch)
    empty_prefixes(monkeypatch)
    real_score = candidate.score_background_roots
    observed = []

    def scored(*args):
        actual = real_score(*args)
        observed.append(actual)
        # Retain actual feasible roots, recourse simulation and store fees;
        # change only the exact two-path contrast used by the selection rule.
        return replace(actual, costs={0: Fraction(total_cost), 1: Fraction(20)},
                       squared_costs={0: Fraction(sum(sample_squares)), 1: Fraction(12)},
                       per_sample_squared_flow_costs={
                           0: tuple(map(Fraction, sample_squares)),
                           1: (Fraction(6), Fraction(6))})

    monkeypatch.setattr(candidate, "score_background_roots", scored)
    result = candidate(*decision_args())
    assert result.changed == changed
    assert result.selection.option_indices == ((0,) if changed else (1,))
    assert result.candidate_count == 2
    assert result.risk_feasible_candidates == (1 if max(sample_squares) > 6 else 2)
    assert result.risk_feasible_candidates + result.risk_filtered_candidates == 2
    assert observed[0].per_sample_squared_flow_costs.keys() == {0, 1}
    assert result.pathwise_sample_squared_flow_writes == 4
    assert result.pathwise_sample_squared_flow_reads == 4 + 2 * comparisons
    assert result.pathwise_sample_risk_comparisons == comparisons
    rejected = int(max(sample_squares) > 6)
    assert result.pathwise_aggregate_feasible_candidates == 2
    assert result.pathwise_rejected_aggregate_feasible_candidates == rejected
    assert result.pathwise_changed_aggregate_root == rejected
    assert result.pathwise_aggregate_risk_comparisons == 2
    assert result.pathwise_aggregate_squared_flow_reads == 3
    assert result.pathwise_aggregate_ordering_comparisons == 1
    assert result.pathwise_aggregate_improvement_comparisons == result.pathwise_root_change_comparisons == 1


def test_outer_overflow_keeps_completed_and_partial_sample_storage_without_gate(monkeypatch):
    options = OPTIONS[:1]
    candidate = PathwiseS2BackgroundRedecisionLaneRollout(options, "partial", sample_count=2, event_cap=2)
    script_background(monkeypatch, (.1, .1, 1000.))
    outer_prefixes = iter((population(), population((1., 0))))

    def sample(rates, window, rng, cap):
        return next(outer_prefixes) if rng is candidate.rng else population()

    monkeypatch.setattr(background_module, "sample_planning_arrivals", sample)
    monkeypatch.setattr(module, "sample_planning_arrivals", sample)
    scored = []
    real_score = candidate.score_background_roots

    def capture(*args):
        actual = real_score(*args)
        scored.append(actual)
        return actual

    monkeypatch.setattr(candidate, "score_background_roots", capture)
    args = (options, (3,), (1000., 1005., 1005.), 1000., 0, ((0., 0., 0.),), (20,))
    result = candidate(*args)
    assert result.selection == candidate.maxweight(*args).selection
    assert result.planning_overflow and result.background_rollout_overflow
    assert not result.changed and result.baseline_predicted_completion_s is None
    assert scored[0].per_sample_squared_flow_costs == {0: (Fraction(14), Fraction(5))}
    assert scored[0].squared_costs == {0: Fraction(19)}
    assert result.pathwise_sample_squared_flow_writes == 2
    assert result.pathwise_sample_squared_flow_reads == 2
    assert result.pathwise_sample_risk_comparisons == 0
    assert result.pathwise_aggregate_risk_comparisons == 0
    assert result.pathwise_aggregate_squared_flow_reads == 0
    assert result.risk_feasible_candidates == result.risk_filtered_candidates == 0
    assert result.flow_evaluations == 5 and result.recourse_calls == 2
    assert result.recourse_rollout_calls > 0


def test_native_toy_driver_aggregates_sample_gate_once_and_detects_missing_reads():
    model = FrameModel("toy", "sample-gate", 1, ("a",), (1.,), OPTIONS)
    candidate = PathwiseS2BackgroundRedecisionLaneRollout(OPTIONS, "native")
    result = simulate_observed_rate_contextual_trajectory(
        model, [(0., 0, 0), (0., 0, 1), (3., 0, 2)], 6., 8., candidate, policy="pathwise")
    protocol = {"planning_samples": 8, "planning_event_cap": 128}
    assert check_rollout(result, model, protocol, candidate)
    first = result["frames"][0]["rollout"]
    assert first["candidate_count"] == 2 and first["risk_feasible_candidates"] == 1
    # The singleton loses on the first path; MW's eight identical paths all
    # pass. Fees therefore include9 actual comparisons, rather than16 or2.
    assert first["pathwise_sample_risk_comparisons"] == 9
    assert first["pathwise_sample_squared_flow_writes"] == 16
    assert first["pathwise_sample_squared_flow_reads"] == 16 + 2 * 9
    work = pathwise_work(result, candidate, 6.)
    assert work["risk_gate_comparisons"] == work["pathwise_sample_risk_comparisons"]
    assert work["pathwise_sample_risk_comparisons"] == sum(
        frame["rollout"]["pathwise_sample_risk_comparisons"] for frame in result["frames"])
    assert work["pathwise_sample_squared_flow_reads"] == (
        work["pathwise_sample_squared_flow_writes"] + 2 * work["risk_gate_comparisons"])
    altered = {**result, "frames": [{**frame, "rollout": dict(frame["rollout"])}
                                     for frame in result["frames"]]}
    altered["frames"][0]["rollout"]["pathwise_sample_squared_flow_reads"] -= 1
    assert not check_rollout(altered, model, protocol, candidate)
