"""The service policy must match the frozen diagnostic without observers."""
from dataclasses import asdict, fields

from algorithm.arrival_aware_lane_rollout import PlanningArrivals
from algorithm.experiments import aligned_hardware_current_policy_continuation_alignment as alignment
from algorithm.experiments import aligned_hardware_two_step_current_policy_population_accounting as population
from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption
from algorithm.experiments.persistent_current_policy_checks import check_rollout
from algorithm.persistent_current_policy_bg_incumbent_lane_rollout import (
    LATE_CURRENT_WORK, PersistentCurrentPolicyBGIncumbentLaneRollout,
)
from skill.tests.test_current_policy_continuation_alignment import (
    OPTIONS, RecordingDeterministicCurrent,
)
from skill.tests.test_parallel_root_lane_rollout import semantic


SAMPLES, CAP = 8, 32


class ScoreCapture(PersistentCurrentPolicyBGIncumbentLaneRollout):
    def score_background_roots(self, *context):
        self.last_scores = super().score_background_roots(*context)
        return self.last_scores


def assert_native_match(native, observed):
    assert native.rng.getstate() == observed.rng.getstate()
    assert native.decision_counter == observed.decision_counter
    assert native._recourse_work == observed._recourse_work
    assert native._two_step_work == observed._two_step_work
    assert native._inner_current_work == observed._inner_current_work
    assert native._inner_current_fallbacks == observed._inner_current_fallbacks
    assert native.continuation_work == observed.continuation_work
    assert native.last_scores == observed.recorded_scores
    assert population.exact_reconstruction(observed)["all_roots_exact_Fraction_reconstruction_passed"]


def test_two_real_decisions_match_original_fields_scores_rng_and_all_native_fees():
    options = (
        FrameOption("a1", (1,), 1., ((1.,),)),
        FrameOption("a2", (2,), 1.5, ((1.5, 1.5),)),
        FrameOption("a4", (4,), 4., ((4., 4., 4., 4.),)),
    )
    native = ScoreCapture(options, "persistent-real-decisions", SAMPLES, CAP)
    observed = alignment.MatchedContinuationPolicy(options, "persistent-real-decisions", SAMPLES, CAP)
    model = FrameModel("toy", "persistent-real", 2, ("a",), (1.,), options)
    contexts = (
        (options, (8,), (1000., 1003.), 1000., 0, ((8., 7., 6., 5., 4., 3., 2., 1.),), (1,)),
        (options, (7,), (1001.5, 1003.), 1001.5, 0, ((8.5, 7.5, 6.5, 5.5, 4.5, 3.5, 2.5),), (2,)),
    )
    frames = []
    total_late = 0
    initial_rng = native.rng.getstate()
    for index, context in enumerate(contexts):
        decision, reference = native(*context), observed(*context)
        for attribute in fields(reference):
            assert getattr(decision, attribute.name) == getattr(reference, attribute.name), attribute.name
        for name in LATE_CURRENT_WORK:
            assert getattr(decision, name) == observed.continuation_work[name]
        assert_native_match(native, observed)
        assert native.decision_counter == index + 1
        total_late += decision.late_current_calls
        frames.append({"start": context[3], "queue": context[1], "waiting_ages": context[5],
            "observed_arrival_counts": context[6], "option_indices": decision.selection.option_indices,
            "rollout": asdict(decision)})
    assert native.rng.getstate() != initial_rng and total_late > 0
    protocol = {"planning_samples": SAMPLES, "planning_event_cap": CAP}
    assert check_rollout({"frames": frames}, model, protocol, native)
    # A lost later-call fee must fail rather than pass under the old two-call bound.
    damaged = [dict(frame, rollout=dict(frame["rollout"])) for frame in frames]
    selected = next(frame for frame in damaged if frame["rollout"]["late_current_calls"])
    selected["rollout"]["late_current_calls"] -= 1
    assert not check_rollout({"frames": damaged}, model, protocol, native)


def deterministic_pair(options, seed, cap, counts, now):
    native = ScoreCapture(options, seed, SAMPLES, cap)
    observed = alignment.MatchedContinuationPolicy(options, seed, SAMPLES, cap)
    for planner in (native, observed):
        planner.arrival_rates = tuple(count / now for count in counts)
        planner._outer_observed_counts = counts
        planner.decision_counter = 7
        planner.inner_planner = RecordingDeterministicCurrent(options, seed, SAMPLES, cap)
    return native, observed


def test_late_idle_counters_released_arrivals_and_rng_persist_without_observers():
    options = (
        FrameOption("a1", (1, 0), 1., ((1.,), ())),
        FrameOption("b1", (0, 1), 1., ((), (1.,))),
        FrameOption("long-loaded-ab", (1, 1), 10., ((.5,), (10.,)), loaded=True),
    )
    native, observed = deterministic_pair(options, "continuation-clock", CAP, (1, 0), 1000.)
    paths = (PlanningArrivals(((0., 0), (.2, 0), (3.2, 0), (6.2, 0), (9.2, 0)), False, 0, 0),) * SAMPLES
    context = ((1, 0), (1000., 1015.), 1000., 0, ((0.,), ()), paths)
    before = native.rng.getstate()
    native.score_background_roots(*context)
    observed.score_background_roots(*context)
    assert_native_match(native, observed)
    assert native.inner_planner.records == observed.inner_planner.records
    assert native.rng.getstate() == before and native.decision_counter == 7
    assert native.continuation_work["late_current_calls"] == 3 * SAMPLES
    assert native.continuation_work["late_virtual_idle_counter_updates"] == 5 * SAMPLES
    for sample in range(SAMPLES):
        calls = native.inner_planner.records[5 * sample:5 * sample + 5]
        assert [call["now"] for call in calls] == [1001., 1002., 1004., 1007., 1010.]
        assert [call["counts"] for call in calls] == [(3, 0), (3, 0), (4, 0), (5, 0), (6, 0)]
        assert [call["counter_before"] for call in calls] == [8, 9, 11, 14, 17]
        assert calls[0]["rng_before"] == before
        assert all(current["rng_before"] == prior["rng_after"]
                   for prior, current in zip(calls, calls[1:]))


def test_overflow_keeps_identical_partial_exact_scores_and_native_fallback_work():
    native, observed = deterministic_pair(OPTIONS, "parallel-overflow", 2, (500, 400, 300), 1000.1)
    paths = (PlanningArrivals(((.2, 2), (.9, 1)), False, 0, 0),) * SAMPLES
    context = ((2, 1, 1), (1000.1, 1005.3), 1000.1, 0, ((2.1, 0.), (1.2,), (0.,)), paths)
    native.score_background_roots(*context)
    observed.score_background_roots(*context)
    assert_native_match(native, observed)
    assert native.last_scores.background_rollout_overflow
    assert population.exact_reconstruction(observed)["complete_population"] is False
    assert native.last_scores.flow_evaluations > 0


def test_parallel_chunks_keep_every_decision_cost_semantic_state_and_discarded_fee():
    from algorithm.parallel_persistent_current_policy_lane_rollout import (
        ParallelPersistentCurrentPolicyBGIncumbentLaneRollout,
    )

    class ParallelScoreCapture(ParallelPersistentCurrentPolicyBGIncumbentLaneRollout):
        def score_background_roots(self, *context):
            self.last_scores = super().score_background_roots(*context)
            return self.last_scores

    options = (
        FrameOption("a1", (1,), 1., ((1.,),)),
        FrameOption("a2", (2,), 1.5, ((1.5, 1.5),)),
        FrameOption("a4", (4,), 4., ((4., 4., 4., 4.),)),
    )
    contexts = (
        (options, (8,), (1000., 1003.), 1000., 0, ((8., 7., 6., 5., 4., 3., 2., 1.),), (1,)),
        (options, (7,), (1001.5, 1003.), 1001.5, 0, ((8.5, 7.5, 6.5, 5.5, 4.5, 3.5, 2.5),), (2,)),
    )
    serial = ScoreCapture(options, "persistent-parallel-decisions", SAMPLES, CAP)
    with ParallelScoreCapture(options, "persistent-parallel-decisions", SAMPLES, CAP, workers=2) as parallel:
        total_late = 0
        for context in contexts:
            expected, actual = serial(*context), parallel(*context)
            assert actual == expected
            assert parallel.last_scores == serial.last_scores
            assert semantic(parallel) == semantic(serial)
            total_late += actual.late_current_calls
        assert total_late > 0
        work = parallel.reuse_work["parallel_root_work"]
        assert work["submitted_chunks"] == work["completed_chunks"] == work["included_chunks"]
        assert work["discarded_chunks"] == 0

    serial = ScoreCapture(OPTIONS, "parallel-overflow", SAMPLES, 2)
    with ParallelScoreCapture(OPTIONS, "parallel-overflow", SAMPLES, 2, workers=2) as parallel:
        for planner in (serial, parallel):
            planner.arrival_rates = (.5, .4, .3)
            planner._outer_observed_counts = (500, 400, 300)
            planner.decision_counter = 7
        context = ((2, 1, 1), (1000.1, 1005.3), 1000.1, 0, ((2.1, .3), (1.2,), (.1,)),
            (PlanningArrivals(((.2, 2), (.9, 1)), False, 0, 0),) * SAMPLES)
        expected = serial.score_background_roots(*context)
        actual = parallel.score_background_roots(*context)
        assert actual == expected and actual.background_rollout_overflow
        assert semantic(parallel) == semantic(serial)
        for name in ("_recourse_work", "_two_step_work", "_inner_current_work", "_inner_current_fallbacks",
                     "continuation_work"):
            assert getattr(parallel, name) == getattr(serial, name), name
        work = parallel.reuse_work["parallel_root_work"]
        assert work["submitted_chunks"] == work["completed_chunks"] == 20
        assert 0 < work["included_chunks"] < work["completed_chunks"]
        assert work["included_chunks"] + work["discarded_chunks"] == work["completed_chunks"]
        assert work["worker_cpu_s"] > 0
        assert parallel.reuse_work["mw_reuse_work"]["actual_option_probes"] > 0
        assert (parallel.reuse_work["inner_decision_reuse_work"]["current_choose_calls"]
                > serial._recourse_work["recourse_calls"])


if __name__ == "__main__":
    test_two_real_decisions_match_original_fields_scores_rng_and_all_native_fees()
    test_late_idle_counters_released_arrivals_and_rng_persist_without_observers()
    test_overflow_keeps_identical_partial_exact_scores_and_native_fallback_work()
    test_parallel_chunks_keep_every_decision_cost_semantic_state_and_discarded_fee()
    print("4 passed", flush=True)
