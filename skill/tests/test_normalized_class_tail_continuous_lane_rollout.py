"""The normalized class constraint replaces both older S2 gates."""
from fractions import Fraction

import pytest

import algorithm.normalized_class_tail_continuous_lane_rollout as module
from algorithm.arrival_aware_lane_rollout import PlanningArrivals
from algorithm.class_tail_continuous_lane_rollout import (
    ClassRootScores, ClassTailContinuousLaneRollout,
)
from algorithm.continuous_arrival_age_tail_lane_rollout import ContinuousArrivalAgeTailLaneRollout
from algorithm.experiments.aligned_hardware_frame_model import FrameOption
from algorithm.normalized_class_tail_continuous_lane_rollout import NormalizedClassTailContinuousLaneRollout


OPTIONS = (FrameOption("a1", (1, 0), 2., ((2.,), ())),
           FrameOption("a2", (2, 0), 3., ((3., 3.), ())),
           FrameOption("b1", (0, 1), 1., ((), (1.,))))
STATE = (OPTIONS, (2, 1), (1000., 1003.), 1000., 0,
         ((3., 1.), (10.,)), (0, 0))


def set_root_scores(monkeypatch, candidate, class_cost, candidate_flow=9):
    """Keep the physical MW root (a2), varying only its compared scores."""
    vectors = {0: tuple(map(Fraction, class_cost)),
               1: (Fraction(100), Fraction(1)),
               2: (Fraction(100), Fraction(1))}
    scored = ClassRootScores(
        costs={0: Fraction(candidate_flow), 1: Fraction(10), 2: Fraction(100)},
        squared_costs={root: sum(values, Fraction(0)) for root, values in vectors.items()},
        candidate_count=3, rollout_calls=0, rollout_option_probes=0,
        root_option_probes=3, rollout_idle_calls=0, sample_count=8,
        waiting_age_reads=3, flow_evaluations=0,
        class_squared_costs=vectors, class_squared_accumulations=0)
    monkeypatch.setattr(candidate, "score_age_roots", lambda *args: scored)
    return scored


@pytest.mark.parametrize("class_cost, accepted", [
    ((50, Fraction(3, 2)), True),
    ((50, 2), False),
    ((150, Fraction(1, 4)), True),
])
def test_relative_constraint_replaces_component_and_aggregate_gates(monkeypatch, class_cost, accepted):
    candidate = NormalizedClassTailContinuousLaneRollout(OPTIONS, "gates")
    scored = set_root_scores(monkeypatch, candidate, class_cost)
    actual = candidate(*STATE)
    relative = sum(value / baseline for value, baseline in
                   zip(scored.class_squared_costs[0], scored.class_squared_costs[1], strict=True))
    assert actual.changed is accepted
    assert actual.selection.option_indices == ((0,) if accepted else (1,))
    assert actual.risk_changed_unconstrained_root is not accepted
    assert (relative <= 2) is accepted
    assert actual.baseline_normalized_risk == 2.
    assert actual.chosen_normalized_risk == (float(relative) if accepted else 2.)
    assert actual.active_risk_classes == 2
    assert actual.normalization_divisions == actual.normalization_additions == 3 * 2
    assert actual.normalization_comparisons == 3
    assert actual.normalization_coordinate_reads == 2 + 3 * 2
    assert actual.class_risk_comparisons == 0
    if class_cost[0] == 50:
        assert scored.squared_costs[0] < scored.squared_costs[1]
        assert scored.class_squared_costs[0][1] > scored.class_squared_costs[1][1]
    else:
        assert scored.squared_costs[0] > scored.squared_costs[1]


@pytest.mark.parametrize("candidate_flow, selected", [(9, (0,)), (10, (1,))])
def test_exact_fraction_boundary_permits_root_and_preserves_mw_flow_tie(monkeypatch, candidate_flow, selected):
    candidate = NormalizedClassTailContinuousLaneRollout(OPTIONS, "fraction-equality")
    set_root_scores(monkeypatch, candidate, (Fraction(200, 3), Fraction(4, 3)), candidate_flow)
    actual = candidate(*STATE)
    assert actual.risk_feasible_candidates == 3
    assert actual.risk_filtered_candidates == 0
    assert actual.selection.option_indices == selected
    assert actual.changed is (candidate_flow == 9)
    if candidate_flow == 10:
        assert actual.selection == candidate.maxweight(*STATE).selection


def test_real_fifo_class_scores_match_existing_exact_flow_engine():
    current = NormalizedClassTailContinuousLaneRollout(OPTIONS, "FIFO")
    aggregate = ContinuousArrivalAgeTailLaneRollout(OPTIONS, "FIFO")
    classes = ClassTailContinuousLaneRollout(OPTIONS, "FIFO")
    state = STATE[1:6]
    paths = ((), ((.5, 0), (1., 1)))
    actual = current.score_age_roots(*state, paths)
    old = aggregate.score_age_roots(*state, paths)
    assert actual == classes.score_age_roots(*state, paths)
    assert actual.costs == old.costs and actual.squared_costs == old.squared_costs
    assert actual.rollout_calls == old.rollout_calls
    assert actual.flow_evaluations == old.flow_evaluations
    assert all(sum(values, Fraction(0)) == old.squared_costs[root]
               for root, values in actual.class_squared_costs.items())
    visible = current.score_age_roots(*state, ((),))
    # a2 binds older a-jobs first (flows 6 and 4), then b1 on the lane
    # released at t+3 (flow 14). The busy lane cannot start it earlier.
    assert visible.costs[1] == 24
    assert visible.class_squared_costs[1] == (52, 196)


def test_absent_class_zero_baseline_has_no_division_and_keeps_real_scoring():
    candidate = NormalizedClassTailContinuousLaneRollout(OPTIONS, "absent-class")
    args = (OPTIONS, (2, 0), (1000., 1003.), 1000., 0, ((3., 1.), ()), (0, 0))
    actual = candidate(*args)
    assert actual.baseline_class_squared_flow_s2 == (52., 0.)
    assert actual.chosen_class_squared_flow_s2[1] == 0.
    assert actual.risk_feasible_candidates >= 1
    assert not actual.planning_overflow
    assert actual.active_risk_classes == 1
    assert actual.normalization_divisions == actual.normalization_additions == 2
    assert actual.normalization_comparisons == 2
    assert actual.normalization_coordinate_reads == 2 + 2


def test_overflow_and_queue_envelope_skip_all_normalized_scoring(monkeypatch):
    candidate = NormalizedClassTailContinuousLaneRollout(OPTIONS, "skip-scoring")

    def forbidden(*args):
        raise AssertionError("MW fallback must not score or normalize roots")

    monkeypatch.setattr(candidate, "score_age_roots", forbidden)
    initial_rng = candidate.rng.getstate()
    for queue, ages in (((0, 0), ((), ())), ((4, 1), ((9., 8., 7., 6.), (2.,)))):
        args = (OPTIONS, queue, (1000., 1003.), 1000., 0, ages, (0, 0))
        actual = candidate(*args)
        assert actual.selection == candidate.maxweight(*args).selection
        assert not actual.eligible and not actual.changed
        assert actual.candidate_count == actual.flow_evaluations == 0
        assert not actual.baseline_class_squared_flow_s2
        assert actual.active_risk_classes == actual.normalization_divisions == 0
        assert actual.normalization_additions == actual.normalization_comparisons == 0
        assert actual.normalization_coordinate_reads == 0
        assert actual.baseline_normalized_risk is actual.chosen_normalized_risk is None
    assert candidate.rng.getstate() == initial_rng
    monkeypatch.setattr(module, "sample_planning_arrivals",
                        lambda *args: PlanningArrivals((), True, 1, 1))
    actual = candidate(*STATE)
    assert actual.planning_overflow and actual.sample_count == 1
    assert actual.selection == candidate.maxweight(*STATE).selection
    assert actual.candidate_count == actual.class_squared_accumulations == 0
    assert not actual.baseline_class_squared_flow_s2
    assert actual.active_risk_classes == actual.normalization_divisions == 0
    assert actual.normalization_additions == actual.normalization_comparisons == 0
    assert actual.normalization_coordinate_reads == 0
    assert actual.baseline_normalized_risk is actual.chosen_normalized_risk is None
