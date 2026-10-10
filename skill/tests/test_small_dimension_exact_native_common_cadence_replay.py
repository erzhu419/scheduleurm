"""Common windows preserve FIFO semantics and retain all paid overruns."""
from copy import deepcopy

import pytest

from algorithm.canonical_count_space_bnb_frame_candidates import (
    select_canonical_count_space_bnb_frame,
)
from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption
from algorithm.experiments.integer_capacity_decision_delay_replay import (
    simulate_delayed_trajectory,
)
from algorithm.experiments import small_dimension_exact_native_common_cadence_replay as replay
from algorithm.experiments.small_dimension_exact_native_decision_delay_audit import (
    ReplayReferenceAudit, selectors,
)


PROTOCOL = dict(initial_lp_budget=144, lp_node_budget=256, integer_state_budget=8192,
                reference_budget=4096, oracle_state_budget=5_000_000)


class SolveClock:
    def __init__(self, *delays):
        self.delays = iter(delays)
        self.calls = 0
        self.now = 0

    def __call__(self):
        if self.calls % 2:
            self.now += round(next(self.delays) * 1_000_000_000)
        self.calls += 1
        return self.now


def model(*, duration=2., offsets=(2.,), lanes=1):
    return FrameModel("small", "common_cadence", lanes, ("a",), (1.,), (
        FrameOption("a", (len(offsets),), duration, (offsets,)),))


def run(current, tape, *, horizon=1., cutoff=10., wall=(.2,) * 20,
        cpu=(.1,) * 20, selector=select_canonical_count_space_bnb_frame, policy="candidate",
        budget=4096, window=1.):
    wall_clock, cpu_clock = SolveClock(*wall), SolveClock(*cpu)
    result, original = replay.simulate_common_cadence_trajectory(
        current, tape, horizon, cutoff, budget, selector, window, policy=policy,
        wall_clock=wall_clock, cpu_clock=cpu_clock)
    return result, original, wall_clock, cpu_clock


def timeline(result):
    keys = ("start", "dispatch_time", "decision_delay_s", "duration", "queue",
            "option_indices", "service", "executed_service", "dispatched")
    return ([[frame[key] for key in keys] for frame in result["frames"]],
            result["jobs"], result["snapshots"])


def test_different_actual_runtimes_preserve_the_same_common_decision_trajectory():
    current = model(duration=1., offsets=(1.,), lanes=2)
    tape = ((0., 0, 0), (.5, 0, 1), (.75, 0, 2))
    fast, _, _, _ = run(current, tape, wall=(.1, .2), cpu=(.05, .1))
    slow, _, _, _ = run(current, tape, wall=(.8, .9), cpu=(.6, .7))
    assert timeline(fast) == timeline(slow)
    assert [frame["queue"] for frame in fast["frames"]] == [(1,), (2,)]
    assert [job["start"] for job in fast["jobs"]] == [1., 3., 3.]
    assert fast["total_selector_wall_s"] == pytest.approx(.3)
    assert slow["total_selector_wall_s"] == pytest.approx(1.7)
    assert fast["total_window_padding_s"] == pytest.approx(1.7)
    assert slow["total_window_padding_s"] == pytest.approx(.3)
    assert fast["total_decision_delay_s"] == slow["total_decision_delay_s"] == 2.


def test_common_window_reuses_the_original_fixed_delay_fifo_engine():
    current = model(duration=3., offsets=(1., 3.), lanes=2)
    tape = ((.25, 0, 0), (.25, 0, 1), (.75, 0, 2), (1., 0, 3))
    expected, _ = simulate_delayed_trajectory(
        current, tape, 1., 12., 4096, select_canonical_count_space_bnb_frame,
        clock=SolveClock(*(1.,) * 20))
    actual, _, _, _ = run(current, tape, cutoff=12.)
    assert timeline(actual) == timeline(expected)
    assert all(actual["invariants"].values())


def test_window_overrun_uses_full_actual_wall_and_retains_the_result():
    current = model(duration=1., offsets=(1.,))
    actual, original, _, _ = run(current, ((0., 0, 0), (.5, 0, 1)),
                                   wall=(1.5, .2), cpu=(1.1, .1))
    assert [frame["decision_delay_s"] for frame in actual["frames"]] == [1.5, 1.]
    assert [frame["start"] for frame in actual["frames"]] == [0., 2.5]
    assert [job["start"] for job in actual["jobs"]] == [1.5, 3.5]
    assert actual["window_overrun_count"] == 1
    assert actual["total_window_overrun_s"] == .5
    assert actual["total_selector_wall_s"] == pytest.approx(1.7)
    assert actual["total_window_padding_s"] == .8
    assert len(original) == actual["decision_count"] == 2


@pytest.mark.parametrize("wall", (.1, 1., 1.5))
def test_decision_at_or_past_cutoff_is_paid_with_no_dispatch(wall):
    actual, original, wall_clock, cpu_clock = run(
        model(), ((0., 0, 0),), cutoff=1., wall=(wall,), cpu=(.05,))
    frame = actual["frames"][0]
    assert actual["decision_count"] == len(original) == 1
    assert actual["executed_frame_count"] == 0
    assert frame["dispatch_time"] == max(1., wall)
    assert frame["selector_wall_s"] == wall
    assert frame["decision_delay_s"] == max(1., wall)
    assert frame["executed_service"] == (0,)
    assert actual["jobs"][0]["start"] is None
    assert actual["snapshots"]["observation_end"]["unfinished_jobs"] == 1
    assert wall_clock.calls == cpu_clock.calls == 2


def test_completion_offsets_crossing_cutoff_are_censored_after_common_window():
    actual, _, _, _ = run(model(duration=10., offsets=(8., 2.)),
                          ((0., 0, 0), (0., 0, 1), (0., 0, 2), (1., 0, 3)),
                          cutoff=5., wall=(.25,), cpu=(.1,))
    assert [job["completion"] for job in actual["jobs"]] == [3., 9., None, None]
    snapshot = actual["snapshots"]["observation_end"]
    assert (snapshot["completed_jobs"], snapshot["inflight_jobs"], snapshot["queued_jobs"]) == (1, 1, 2)
    assert snapshot["restricted_mean_flow_s"] == pytest.approx(17 / 4)
    assert actual["total_decision_delay_s"] == 1.
    assert actual["total_selector_wall_s"] == .25
    assert actual["last_frame_end"] == 11.


def test_actual_cold_whole_calls_are_not_shared_for_repeated_queue_states():
    calls, returned = [], []

    def selector(options, queue, lanes, cap):
        calls.append(queue)
        result = select_canonical_count_space_bnb_frame(options, queue, lanes, cap)
        returned.append(result)
        return result

    actual, original, wall_clock, cpu_clock = run(
        model(), ((0., 0, 0), (.5, 0, 1)), selector=selector,
        wall=(.2, .3), cpu=(.1, .15))
    assert calls == [(1,), (1,)]
    assert len(original) == len(returned) == 2
    assert original[0] is returned[0] and original[1] is returned[1]
    assert original[0] is not original[1]
    assert wall_clock.calls == cpu_clock.calls == 4
    assert actual["total_selector_cpu_s"] == .25


@pytest.mark.parametrize("policy", ("candidate", "direct"))
def test_native_audit_retains_whole_work_but_uses_actual_wall_and_cpu(policy):
    current = FrameModel("small", "routed_common", 4, ("a",), (1.,), (
        FrameOption("a1", (1,), 1., ((1.,),)),
        FrameOption("a3", (3,), 2., ((2., 2., 2.),))))
    trajectory, original, _, _ = run(
        current, ((0., 0, 0),), horizon=.5, cutoff=.5, selector=selectors(PROTOCOL)[policy],
        policy=policy, budget=PROTOCOL["lp_node_budget"], wall=(.2,), cpu=(.1,))
    frames_before, jobs_before = deepcopy(trajectory["frames"]), deepcopy(trajectory["jobs"])
    replay.audit_common_cadence_trajectory(
        current, trajectory, original, ReplayReferenceAudit(current, PROTOCOL), PROTOCOL)
    metrics = trajectory["selector_metrics"]
    assert metrics["operational_elapsed_s"] == .2
    assert metrics["operational_cpu_s"] == .1
    assert trajectory["total_decision_delay_s"] == 1.
    assert trajectory["frames"] == frames_before and trajectory["jobs"] == jobs_before
    assert metrics["whole_work"]["calls"] == metrics["calls"] == 1
    assert metrics["canonical_exact_count"] == 1
    assert not metrics["failures"] and not trajectory["reference_audit"]["incomplete_count"]


def test_idle_decision_keeps_the_existing_one_second_frame_after_computation_window():
    actual, _, _, _ = run(model(), ((.25, 0, 0),), cutoff=5.)
    assert actual["frames"][0]["queue"] == (0,)
    assert actual["frames"][0]["duration"] == 1.
    assert actual["frames"][0]["dispatch_time"] == 1.
    assert actual["frames"][1]["start"] == 2.
    assert actual["jobs"][0]["start"] == 3.


def test_actual_wall_equal_to_window_has_no_padding_or_overrun():
    actual, _, _, _ = run(model(), ((0., 0, 0),), wall=(1.,), cpu=(.75,))
    assert actual["total_window_padding_s"] == 0
    assert actual["total_window_overrun_s"] == actual["window_overrun_count"] == 0
    assert actual["total_decision_delay_s"] == actual["total_selector_wall_s"] == 1.


def test_offline_reference_overrun_is_recorded_without_changing_normative_clock():
    current, tape = model(), ((0., 0, 0),)
    reference, original = replay.simulate_fixed_reference_trajectory(
        current, tape, 1., 10., 4096, select_canonical_count_space_bnb_frame, 1.,
        wall_clock=SolveClock(2.5), cpu_clock=SolveClock(2.))
    normative, _ = simulate_delayed_trajectory(
        current, tape, 1., 10., 4096, select_canonical_count_space_bnb_frame,
        policy="full_reference", clock=SolveClock(1.))
    assert reference["policy"] == "full_reference"
    assert timeline(reference) == timeline(normative)
    assert reference["total_selector_wall_s"] == 2.5
    assert reference["total_selector_cpu_s"] == 2.
    assert reference["window_overrun_count"] == 1
    assert reference["total_window_overrun_s"] == 1.5
    assert reference["total_decision_delay_s"] == 1.
    replay.audit_common_cadence_trajectory(
        current, reference, original, ReplayReferenceAudit(current, PROTOCOL), PROTOCOL)
    assert reference["selector_metrics"]["operational_elapsed_s"] == 2.5
    assert reference["selector_metrics"]["operational_cpu_s"] == 2.
    assert not reference["selector_metrics"]["failures"]
