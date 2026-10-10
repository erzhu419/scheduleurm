"""Hand-computed common-horizon costs and native arrival/call boundaries."""
from collections import Counter
from fractions import Fraction
from random import Random
from types import SimpleNamespace
from unittest.mock import patch

from algorithm.arrival_aware_lane_rollout import ArrivalAwareLaneRollout, sample_planning_arrivals
from algorithm.background_competition_continuous_lane_rollout import _LazyBackgroundArrivals
from algorithm.exact_reuse_lane_rollout import ExactReuseTwoStepCurrentPolicyBGIncumbentLaneRollout
from algorithm import parallel_root_lane_rollout as native
from algorithm.experiments.aligned_hardware_frame_model import FrameOption
from algorithm.experiments.bg_incumbent_counterfactual_replay import FEES, STATUS
from algorithm.experiments.common_horizon_flow_rollout import _score_flow, _simulate_flow, evaluate_root_sample
from algorithm.experiments.persistent_continuation_cross_selector import (
    capture_planner_tree, restore_planner_tree,
)
from algorithm.two_step_bg_incumbent_background_redecision_lane_rollout import TWO_STEP_WORK


F = Fraction


class DispatchFixture:
    """Known finite proposals; terminal selection uses the real native MW scan."""
    _best_index = ArrivalAwareLaneRollout._best_index

    def __init__(self, options):
        self.options = tuple(options)
        self.requirements = tuple(option.requirements for option in options)
        self.durations = tuple(F(option.duration) for option in options)
        self.duration_ratios = tuple((value.numerator, value.denominator) for value in self.durations)
        self.sorted_completion_offsets = tuple(tuple(tuple(sorted(map(F, group)))
            for group in option.completion_offsets) for option in options)
        self.maximum_duration = float(max(self.durations))
        self.decision_counter = 7
        self.arrival_rates = (0.,)
        self._outer_observed_counts = (0,)
        self.rng = Random(31)
        self.reuse_work = {}
        self.calls = []

    def __call__(self, options, queue, available, now, lane, ages, counts):
        self.calls.append(dict(queue=queue, available=available, now=now,
            lane=lane, ages=ages, counts=counts, counter=self.decision_counter))
        self.decision_counter += 1
        self.rng.random()
        self.arrival_rates = tuple(count / now for count in counts)
        self._outer_observed_counts = tuple(counts)
        index, _ = self._best_index(queue)
        fields = dict.fromkeys(FEES + STATUS + TWO_STEP_WORK, 0)
        return SimpleNamespace(**fields, selection=SimpleNamespace(option_indices=(index,)),
            inner_current_work={}, inner_current_fallbacks={})


class ScriptedArrivals:
    """Reveal a fixed chronological suffix only through the requested boundary."""
    def __init__(self, events=()):
        self.pending = tuple((F(birth), coordinate) for birth, coordinate in events)
        self.events = []
        self.requested = []
        self.random_draws = self.rate_reads = self.seed_initializations = 0
        self.overflow = False

    def release_through(self, time):
        self.requested.append(time)
        self.events = [event for event in self.pending if event[0] <= time]
        return True


def context(ages):
    return dict(queue=(len(ages),), available=(100., 110.), now=100., lane=0,
        ages=(tuple(ages),), counts=(len(ages),))


def test_integral_counts_jobs_and_committed_completions_not_lane_reservations():
    # A batch is two unfinished jobs, and its early completion releases no lane.
    # The other lane remains reserved to 10 although its committed jobs end at 2/9.
    planner = DispatchFixture((FrameOption("pair", (2,), 4., ((.5, 3.5),)),
        FrameOption("single", (1,), 4., ((4.,),))))
    result = _simulate_flow(planner, context((7., 5., 3., 1.)), 0, (), ScriptedArrivals(),
        initial_inflight=(dict(id=("busy", 0), coordinate=0, birth=96., completion=102.),
                          dict(id=("busy", 1), coordinate=0, birth=94., completion=109.)))
    # Future completion costs: .5 + 3.5 + 4.5 + 7.5 + 2 + 9 = 27.
    assert result["complete"] and not result["overflow"]
    assert result["waiting_area"] == 8
    assert result["inflight_area"] == 18
    assert result["common_area"] == 26
    assert result["terminal_cost"] == 1
    assert result["cost"] == 27
    assert result["past_flow_constant"] == 26
    assert [call["now"] for call in planner.calls] == [104.]
    assert result["terminal_state"]["queue"] == (0,)
    assert result["terminal_state"]["lane_ends"] == (F(8), F(10))


def test_horizon_admits_boundary_births_then_clears_fifo_without_later_arrivals():
    planner = DispatchFixture((FrameOption("single", (1,), 3., ((1.,),)),))
    suffix = ScriptedArrivals(((4, 0), (6, 0), (F(13, 2), 0)))
    result = _simulate_flow(planner, context((5.,)), 0, ((F(9, 4), 0),), suffix)
    # Completions 1,4,7,10; births 0,2.25,4,6. Pre-anchor age 5 is common.
    # At H=6 two jobs wait; MW starts them at 6 and 9, not together at H.
    assert result["complete"] and not result["overflow"]
    assert result["common_area"] == F(19, 4)
    assert result["waiting_area"] == F(11, 4)
    assert result["inflight_area"] == 2
    assert result["terminal_cost"] == 5
    assert result["cost"] == F(39, 4)
    assert result["past_flow_constant"] == 5
    assert result["terminal_state"]["queue"] == (2,)
    assert result["terminal_state"]["lane_ends"] == (F(6), F(10))
    assert suffix.events == [(F(4), 0), (F(6), 0)]
    assert max(suffix.requested) == 6
    assert [call["now"] for call in planner.calls] == [103.]


def test_idle_native_call_is_consumed_and_horizon_between_dispatches_keeps_early_completions():
    options = (FrameOption("single", (1,), 2., ((.5,),)),)
    seed, samples, cap = "common-horizon-idle", 1, 16
    current = context((3.,))
    reference = ExactReuseTwoStepCurrentPolicyBGIncumbentLaneRollout(options, seed, samples, cap)
    reference(options, current["queue"], current["available"], current["now"],
        current["lane"], current["ages"], current["counts"])
    after = capture_planner_tree(reference, Counter())
    traced = []
    cls = native.ParallelRootTwoStepCurrentPolicyBGIncumbentLaneRollout
    native_call = cls.__call__

    def observe(planner, *arguments):
        entry = capture_planner_tree(planner, Counter())
        decision = native_call(planner, *arguments)
        traced.append((arguments, entry, decision, capture_planner_tree(planner, Counter())))
        return decision

    suffix = ScriptedArrivals(((F(5, 2), 0), (F(9, 2), 0)))
    with cls(options, seed, samples, cap, workers=2) as planner:
        with patch.object(cls, "__call__", observe):
            result = _score_flow(planner, current, 0, (), suffix, after)
    # The t=2 empty proposal consumes state; its one-second idle poll sees t=2.5 birth.
    # Completion at 3.5 is finished at H=4 even though the lane releases at 5.
    assert result["complete"] and not result["overflow"]
    assert [arguments[3] for arguments, *_ in traced] == [102., 103.]
    assert [arguments[1] for arguments, *_ in traced] == [(0,), (1,)]
    assert [arguments[6] for arguments, *_ in traced] == [(1,), (2,)]
    assert result["calls"] == 2 and result["idle_calls"] == 1
    assert result["initial_tree"] == after == traced[0][1]
    restore_planner_tree(reference, after)
    logical = Counter()
    for arguments, entry, proposed, exit_state in traced:
        assert capture_planner_tree(reference, Counter()) == entry
        expected = reference(*arguments)
        assert expected == proposed
        assert capture_planner_tree(reference, Counter()) == exit_state
        logical.update({name: getattr(expected, name) for name in FEES + TWO_STEP_WORK})
        for values in (expected.inner_current_work, expected.inner_current_fallbacks):
            logical.update({"inner_current_" + name: value for name, value in values.items()})
    assert dict(logical) == result["logical_work"]
    assert result["final_tree"] == capture_planner_tree(reference, Counter())
    assert result["final_tree"]["planners"][0]["decision_counter"] == (
        after["planners"][0]["decision_counter"] + 2)
    assert result["common_area"] == F(3, 2)
    assert result["terminal_cost"] == 0
    assert result["cost"] == F(3, 2)
    assert result["terminal_state"]["lane_ends"] == (F(5), F(10))
    assert not result["terminal_state"]["unfinished_ids"]
    assert suffix.events == [(F(5, 2), 0)]
    assert max(suffix.requested) == 4


def test_native_arrival_cap_keeps_each_sample_and_partial_cost_instead_of_clearing():
    # Both fixed paths overflow before H. Reaching the cap must not switch to a
    # no-arrival drain and publish that cheaper incomplete score as available.
    results = []
    for sample in range(2):
        planner = DispatchFixture((FrameOption("single", (1,), 2., ((.5,),)),))
        suffix = _LazyBackgroundArrivals((100.,), 2.,
            f"common-horizon-cap:background:7:{sample}", 1, 0)
        result = _simulate_flow(planner, context((3., 2., 1.)), 0, (), suffix)
        results.append(result)
        assert result["overflow"] and not result["complete"]
        assert suffix.overflow and len(suffix.events) == 2
        assert result["cost"] > 0 and result["common_area"] > 0
        assert result["terminal_cost"] == 0
        assert result["population"]["remaining"] > 0
        assert result["area_end"] < result["horizon"]
    assert len(results) == 2


def test_public_entry_preserves_real_sampler_triples_and_global_sample_identity():
    # Production PlanningArrivals carry (time, coordinate, class-local index).
    # Test the public conversion with a real nonempty sampler path, rather than
    # hand-authored pairs that bypass the format used by saved preparations.
    options = (FrameOption("single", (1,), 2., ((.5,),)),)
    seed, samples, cap = "common-horizon-native-prefix", 1, 16
    current = dict(context((3.,)), counts=(100,))
    path = sample_planning_arrivals((1.,), 2., Random(0), cap)
    original_events = path.events
    assert not path.overflow and len(original_events) == 1
    assert len(original_events[0]) == 3 and original_events[0][1:] == (0, 0)
    planner = ExactReuseTwoStepCurrentPolicyBGIncumbentLaneRollout(options, seed, samples, cap)
    before = capture_planner_tree(planner, Counter())
    planner(options, current["queue"], current["available"], current["now"],
        current["lane"], current["ages"], current["counts"])
    after = capture_planner_tree(planner, Counter())
    result = evaluate_root_sample(options, seed, (path,), before, after, current,
        0, 0, sample_count=samples, event_cap=cap, workers=2)
    assert path.events == original_events
    assert result["root"] == result["sample_index"] == 0
    environment, flow = result["environment"], result["flow"]
    assert environment["sample_index"] == environment["anchor_counter"] == 0
    assert environment["seed"] == f"{seed}:background:0:0"
    assert environment["prefix_events"] == ((F(original_events[0][0]), 0),)
    suffix = _LazyBackgroundArrivals((1.,), 2., environment["seed"], cap, 1)
    assert suffix.release_through(F(4))
    assert flow["background_events"] == tuple(suffix.events)
    assert flow["initial_tree"] == after
    assert flow["calls"] == 1 and flow["idle_calls"] == 0
    assert result["complete"] and flow["complete"] and not flow["overflow"]
    # Visible completes .5, the prefix arrival is served at 2 and completes 2.5.
    # All background births are in (2,4]; terminal MW fills the two retained
    # lane release clocks (4,10) one singleton at a time, without new arrivals.
    prefix_birth = F(original_events[0][0])
    expected = F(1, 2) + F(5, 2) - prefix_birth
    common = expected + sum((4 - birth for birth, _ in suffix.events), F(0))
    lane_ends = [F(4), F(10)]
    for birth, coordinate in suffix.events:
        assert coordinate == 0 and 2 < birth <= 4
        lane = min(range(2), key=lambda index: (lane_ends[index], index))
        expected += lane_ends[lane] + F(1, 2) - birth
        lane_ends[lane] += 2
    assert flow["common_area"] == common
    assert flow["cost"] == expected
    assert flow["terminal_cost"] == expected - common
    assert flow["past_flow_constant"] == 3
    assert flow["population"]["total"] == 2 + len(suffix.events)
    assert flow["population"]["sources"]["prefix"]["total"] == 1
    assert flow["population"]["sources"]["background"]["total"] == len(suffix.events)


if __name__ == "__main__":
    test_integral_counts_jobs_and_committed_completions_not_lane_reservations()
    test_horizon_admits_boundary_births_then_clears_fifo_without_later_arrivals()
    test_idle_native_call_is_consumed_and_horizon_between_dispatches_keeps_early_completions()
    test_native_arrival_cap_keeps_each_sample_and_partial_cost_instead_of_clearing()
    test_public_entry_preserves_real_sampler_triples_and_global_sample_identity()
    print("5 passed", flush=True)
