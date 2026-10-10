"""A carried class support adapts batch size in one causal catalogue scan."""
from fractions import Fraction

import pytest

import algorithm.background_competition_continuous_lane_rollout as background_module
import algorithm.support_priority_background_lane_rollout as module
from algorithm.arrival_aware_lane_rollout import PlanningArrivals
from algorithm.background_competition_continuous_lane_rollout import BackgroundCompetitionContinuousLaneRollout
from algorithm.experiments.aligned_hardware_frame_model import FrameOption
from algorithm.support_priority_background_lane_rollout import SupportPriorityBackgroundLaneRollout


OPTIONS = (FrameOption("a1", (1, 0), 2., ((2.,), ())),
           FrameOption("a2", (2, 0), 3., ((3., 3.), ())),
           FrameOption("a3", (3, 0), 4., ((4., 4., 4.), ())),
           FrameOption("b1", (0, 1), 1., ((), (1.,))))


class CountedRequirements(tuple):
    """Count actual catalogue entries visited, independently of reported fees."""
    visits = 0

    def __iter__(self):
        for counts in super().__iter__():
            self.visits += 1
            yield counts


def forbid_sampling(monkeypatch):
    def forbidden(*args):
        raise AssertionError("a carried support is consumed without a new arrival sample")

    monkeypatch.setattr(module, "sample_planning_arrivals", forbidden)


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


@pytest.mark.parametrize("saved,queue,chosen,global_index,reads,comparisons", [
    (2, (1, 1), 0, 3, 6, 1),   # Old A3 is infeasible: shrink to A1.
    (0, (3, 3), 2, 3, 10, 3),  # An old A1 priority grows to efficient A3.
    (0, (3, 0), 2, 2, 8, 3),   # Batch changes even though dynamic action equals MW.
])
def test_dynamic_batch_uses_one_catalogue_scan_and_no_new_draws(monkeypatch, saved,
                                                              queue, chosen,
                                                              global_index, reads,
                                                              comparisons):
    candidate = SupportPriorityBackgroundLaneRollout(OPTIONS, "dynamic")
    candidate.pending_successor_index = saved
    candidate.requirements = CountedRequirements(candidate.requirements)
    forbid_sampling(monkeypatch)
    rng_before = candidate.rng.getstate()
    ages = tuple(tuple(.5 - .1 * i for i in range(count)) for count in queue)
    result = candidate(OPTIONS, queue, (1000., 1009.), 1000., 0, ages, queue)
    assert result.selection.option_indices == (chosen,)
    assert result.planned_successor_index == saved and result.successor_option_index == chosen
    assert result.successor_executed and result.successor_batch_changed
    assert result.successor_changed == (chosen != global_index)
    assert candidate.pending_successor_index is None and candidate.decision_counter == 1
    assert candidate.requirements.visits == result.root_option_probes == len(OPTIONS)
    assert result.successor_support_coordinate_reads == reads
    assert result.successor_support_comparisons == comparisons
    assert result.successor_feasibility_coordinate_reads == 0
    assert result.planning_random_draws == result.rollout_calls == result.sample_count == 0
    assert candidate.rng.getstate() == rng_before


def test_multiclass_support_allows_a_subset_but_excludes_unpromised_classes(monkeypatch):
    options = (FrameOption("a", (1, 0, 0), 2., ((2.,), (), ())),
               FrameOption("b", (0, 1, 0), 1., ((), (1.,), ())),
               FrameOption("c", (0, 0, 1), 1., ((), (), (1.,))),
               FrameOption("ab", (1, 1, 0), 2., ((1.,), (2.,), ()), loaded=True))
    candidate = SupportPriorityBackgroundLaneRollout(options, "subset")
    candidate.pending_successor_index = 3
    forbid_sampling(monkeypatch)
    result = candidate(options, (1, 2, 3), (1000., 1009., 1010., 1011.), 1000., 0,
                       ((.5,), (.5, .4), (.5, .4, .3)), (1, 2, 3))
    # Global MW would serve C; support {A,B} instead selects B alone.
    assert result.selection.option_indices == (1,) and result.selection.service == (0, 1, 0)
    assert result.successor_executed and result.successor_changed and result.successor_batch_changed
    assert result.successor_support_coordinate_reads == 15
    assert result.successor_support_comparisons == 3


def test_restricted_mw_preserves_exact_duration_then_index_ties():
    options = (FrameOption("a1", (1, 0), 1., ((1.,), ())),
               FrameOption("a2", (2, 0), 2., ((2., 2.), ())),
               FrameOption("a1-copy", (1, 0), 1., ((1.,), ())),
               FrameOption("b", (0, 1), 1., ((), (1.,))))
    candidate = SupportPriorityBackgroundLaneRollout(options, "ties")
    scanned = candidate.scan_support_options((2, 3), 0)
    # All A options tie on exact support; A1 has shortest duration and lower
    # index than its duplicate. B remains the independent global MW incumbent.
    assert scanned.global_index == 3 and scanned.support_index == 0
    assert scanned.support_coordinate_reads == 10 and scanned.support_comparisons == 3


@pytest.mark.parametrize("queue,inside", [((0, 3), True), ((3, 4), False)])
def test_empty_holds_then_missing_family_or_envelope_failure_consumes_to_global_mw(monkeypatch, queue, inside):
    candidate = SupportPriorityBackgroundLaneRollout(OPTIONS, "guards")
    candidate.pending_successor_index = 2
    candidate.requirements = CountedRequirements(candidate.requirements)
    forbid_sampling(monkeypatch)
    rng_before = candidate.rng.getstate()
    empty = candidate(OPTIONS, (0, 0), (1000., 1009.), 1000., 0, ((), ()), (0, 0))
    assert empty.selection.option_indices == (None,) and candidate.pending_successor_index == 2
    assert candidate.requirements.visits == len(OPTIONS)
    assert empty.successor_envelope_checks == empty.successor_support_coordinate_reads == 0
    candidate.requirements.visits = 0
    ages = tuple(tuple(.5 - .1 * i for i in range(count)) for count in queue)
    result = candidate(OPTIONS, queue, (1001., 1009.), 1001., 0, ages, queue)
    assert result.selection.option_indices == (3,) and not result.successor_executed
    assert result.successor_option_index is None and not result.successor_batch_changed
    assert result.successor_envelope_passed == inside
    assert result.successor_feasibility_checks == int(inside)
    assert result.successor_support_coordinate_reads == (4 if inside else 0)
    assert result.successor_support_comparisons == 0
    assert candidate.requirements.visits == result.root_option_probes == len(OPTIONS)
    assert candidate.pending_successor_index is None and candidate.decision_counter == 2
    assert result.planning_random_draws == result.sample_count == 0
    assert candidate.rng.getstate() == rng_before


def test_virtual_first_nonempty_batch_matches_real_handoff(monkeypatch):
    candidate = SupportPriorityBackgroundLaneRollout(OPTIONS, "paired", sample_count=1)
    scans = []
    scan = candidate.scan_support_options

    def observed_scan(queue, support_option_index=None):
        selected = scan(queue, support_option_index)
        scans.append((tuple(queue), support_option_index, selected.support_index))
        return selected

    monkeypatch.setattr(candidate, "scan_support_options", observed_scan)
    scored = candidate.score_deferred_roots(
        (3, 1), (1000., 1010.), 1000., 0, ((5., 3., 1.), (2.,)),
        (PlanningArrivals((), False, 0, 0),), 2)
    assert [entry for entry in scans if entry[:2] == ((1, 1), 2)] == [((1, 1), 2, 0)]
    # Root A2 leaves A1+B1 at t=3. The priority serves A1 first; ordinary MW
    # serves B1 first. Both branches use complete offsets and identical FIFO.
    assert scored.costs[1, None] == Fraction(27) and scored.squared_costs[1, None] == Fraction(185)
    assert scored.costs[1, 2] == Fraction(28) and scored.squared_costs[1, 2] == Fraction(200)
    candidate.pending_successor_index = 2
    forbid_sampling(monkeypatch)
    actual = candidate(OPTIONS, (1, 1), (1003., 1010.), 1003., 0, ((4.,), (5.,)), (3, 1))
    assert actual.successor_option_index == 0 and scans[-1] == ((1, 1), 2, 0)
    assert actual.selection.option_indices == (0,) and candidate.pending_successor_index is None


def test_ordinary_plans_match_original_bg_with_loaded_busy_lanes_and_shared_tagged_objective(monkeypatch):
    options = (FrameOption("loaded", (1, 1), 4., ((1.,), (3.,)), loaded=True),
               OPTIONS[0], OPTIONS[3])
    candidate = SupportPriorityBackgroundLaneRollout(options, "bg", sample_count=1)
    candidate.arrival_rates = (0., 1.)
    instances = scripted_background(monkeypatch, (.5, 100.))
    state = ((2, 1), (1000., 1006.), 1000., 0, ((5., 1.), (2.,)),
             (PlanningArrivals(((4., 0),), False, 0, 0),))
    rng_before = candidate.rng.getstate()
    scored = candidate.score_deferred_roots(*state, 2)
    # Loaded completions1/3 never release its lane before4. The busy other
    # lane opens at6; BG B4.5 competes but only four tagged jobs enter each plan.
    assert {root: scored.costs[root, None] for root in range(3)} == {
        0: Fraction(22), 1: Fraction(22), 2: Fraction(21)}
    assert {root: scored.squared_costs[root, None] for root in range(3)} == {
        0: Fraction(126), 1: Fraction(126), 2: Fraction(125)}
    assert scored.rollout_calls == 22 and scored.rollout_option_probes == 66
    assert scored.flow_evaluations == 24 and scored.background_admissions == 6
    assert scored.admitted_job_reads == scored.tag_flag_reads == 30
    assert scored.background_planning_events == 1 and scored.background_planning_random_draws == 2
    assert scored.background_seed_initializations == len(instances) == 1
    assert candidate.rng.getstate() == rng_before
    original = BackgroundCompetitionContinuousLaneRollout(options, "bg", sample_count=1)
    original.arrival_rates = (0., 1.)
    baseline = original.score_background_roots(*state)
    assert all(scored.costs[root, None] == baseline.costs[root] and
               scored.squared_costs[root, None] == baseline.squared_costs[root]
               for root in baseline.costs)


def test_partial_background_overflow_discards_all_support_plans_but_keeps_actual_work(monkeypatch):
    candidate = SupportPriorityBackgroundLaneRollout(OPTIONS, "overflow", sample_count=1, event_cap=2)
    scripted_background(monkeypatch, (100., .1, .1))
    monkeypatch.setattr(module, "sample_planning_arrivals",
        lambda *args: PlanningArrivals(((1., 0),), False, 2, 2))
    current = (OPTIONS, (3, 1), (1000., 1009.), 1000., 0,
               ((5., 3., 1.), (2.,)), (200, 100))
    result = candidate(*current)
    assert result.selection.option_indices == (2,)
    assert result.planning_overflow and result.background_rollout_overflow
    assert not result.plan_changed and candidate.pending_successor_index is None
    assert result.planning_events == 3 and result.background_planning_events == 2
    assert result.planning_random_draws == result.planning_rate_reads == 5
    assert result.rollout_calls == 1 and result.rollout_option_probes == len(OPTIONS)
    assert result.root_option_probes == 2 * len(OPTIONS)
    assert result.flow_evaluations == result.admitted_job_reads == result.tag_flag_reads == 4
    assert result.baseline_predicted_completion_s is result.chosen_predicted_completion_s is None
