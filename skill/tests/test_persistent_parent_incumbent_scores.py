"""Parent guarding uses exact native scores without changing their execution."""
from collections import Counter
from dataclasses import asdict, replace
from fractions import Fraction
from unittest.mock import patch

from algorithm.arrival_aware_lane_rollout import PlanningArrivals
from algorithm.background_competition_continuous_lane_rollout import (
    BackgroundCompetitionContinuousLaneRollout, BackgroundRootScores,
)
from algorithm.experiments.aligned_hardware_frame_model import FrameOption
from algorithm.experiments.persistent_continuation_cross_selector import capture_planner_tree
from algorithm.experiments.persistent_parent_incumbent_scores import (
    RecordingPersistentCurrentPolicyScores, RecordingParallelPersistentCurrentPolicyScores,
    choose_parent_incumbent, last_touched_frames, recover_frame_child_states,
)
from algorithm.parallel_root_lane_rollout import _semantic_state
from algorithm.tick_persistent_current_policy_lane_rollout import (
    ParallelTickPersistentCurrentPolicyBGIncumbentLaneRollout,
    TickPersistentCurrentPolicyBGIncumbentLaneRollout, _initialize_worker, _run_root,
)


OPTIONS = (
    FrameOption("single", (1,), 1., ((1.,),)),
    FrameOption("pair", (2,), 1.5, ((1.5, 1.5),)),
    FrameOption("four", (4,), 4., ((4., 4., 4., 4.),)),
)
SEED, SAMPLES, CAP = "parent-score-native-prefix", 2, 16
CONTEXTS = (
    (OPTIONS, (8,), (1000., 1003.), 1000., 0, ((8., 7., 6., 5., 4., 3., 2., 1.),), (1,)),
    (OPTIONS, (7,), (1001.5, 1003.), 1001.5, 0, ((8.5, 7.5, 6.5, 5.5, 4.5, 3.5, 2.5),), (2,)),
)


def table(costs, squared):
    return BackgroundRootScores(
        {i: Fraction(v) for i, v in enumerate(costs)},
        {i: Fraction(v) for i, v in enumerate(squared)},
        len(costs), 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, False)


def tree(planner):
    return capture_planner_tree(planner, Counter())


def test_parent_rule_keeps_original_risk_set_and_strict_s1_with_parent_s2():
    original = table((20,) * 6, (100, 100, 100, 101, 100, 99))
    scored = table((10, 9, 1, 0, 10, 9), (100, 100, 101, 50, 90, 95))
    durations = tuple(map(Fraction, (7, 4, 1, 1, 1, 2)))
    assert choose_parent_incumbent(scored, original, 0, 4, durations) == 5
    # Among strict improvements, duration and then catalogue index break ties.
    assert choose_parent_incumbent(scored, original, 0, 4,
        (7, 2, 1, 1, 1, 2)) == 1
    tied = replace(scored, costs={**scored.costs, 1: Fraction(10), 5: Fraction(10)})
    assert choose_parent_incumbent(tied, original, 0, 4, durations) == 0
    # A native overflow contains partial costs and cannot establish a guard.
    for left, right in ((replace(scored, background_rollout_overflow=True), original),
            (scored, replace(original, background_rollout_overflow=True))):
        try:
            choose_parent_incumbent(left, right, 0, 4, durations)
        except ValueError:
            pass
        else:
            raise AssertionError("partial root costs were used as complete native scores")


def test_fraction_strict_gain_and_risk_boundary_survive_equal_float_predictions():
    epsilon = Fraction(1, 2 ** 100)
    original = table((2, 2, 2), (3, 3, 3))
    scored = table((1, 1 - epsilon, 0), (2, 2, 2 + epsilon))
    assert float(scored.costs[1]) == float(scored.costs[0])
    assert float(scored.squared_costs[2]) == float(scored.squared_costs[0])
    assert choose_parent_incumbent(scored, original, 0, 0, (3, 1, 1)) == 1
    original = replace(original, squared_costs={0: Fraction(3), 1: 3 + epsilon, 2: Fraction(3)})
    assert float(original.squared_costs[1]) == float(original.squared_costs[0])
    assert choose_parent_incumbent(scored, original, 0, 0, (3, 1, 1)) == 0


class NativeScoreCapture(TickPersistentCurrentPolicyBGIncumbentLaneRollout):
    def score_background_roots(self, *context):
        self.last_scores = super().score_background_roots(*context)
        return self.last_scores


def test_recording_preserves_every_native_field_fee_rng_state_and_single_bg_score():
    native = NativeScoreCapture(OPTIONS, SEED, SAMPLES, CAP)
    recorded = RecordingPersistentCurrentPolicyScores(OPTIONS, SEED, SAMPLES, CAP)
    calls = {id(native): [], id(recorded): []}
    bg_score = BackgroundCompetitionContinuousLaneRollout.score_background_roots

    def trace_original(planner, *context):
        value = bg_score(planner, *context)
        if id(planner) in calls:
            calls[id(planner)].append((value, planner.rng.getstate(), planner.decision_counter))
        return value

    with patch.object(BackgroundCompetitionContinuousLaneRollout, "score_background_roots", trace_original):
        for context in CONTEXTS:
            expected, actual = native(*context), recorded(*context)
            assert actual == expected
            assert tree(recorded) == tree(native)
            assert recorded.reuse_work == native.reuse_work
            assert recorded.last_reuse_work == native.last_reuse_work
            original, rng, counter = calls[id(native)][-1]
            assert recorded.recorded_original_scores == original
            assert recorded.recorded_original_scores is calls[id(recorded)][-1][0]
            assert recorded.recorded_scores == native.last_scores
            assert recorded.recorded_post_prefix_state == dict(rng_state=rng, decision_counter=counter)
            assert not actual.planning_overflow
            assert all(isinstance(v, Fraction) for scored in
                (recorded.recorded_original_scores, recorded.recorded_scores)
                for values in (scored.costs, scored.squared_costs) for v in values.values())
        assert all(len(values) == len(CONTEXTS) for values in calls.values())
        extra = dict(recorded.recording_work)
        assert extra == {"recorded_post_prefix_rng_words_copied": len(CONTEXTS) * 625,
            "recorded_post_prefix_counter_reads": len(CONTEXTS)}
        idle = (OPTIONS, (0,), (1004., 1005.), 1004., 0, ((),), (2,))
        assert recorded(*idle) == native(*idle)
        assert tree(recorded) == tree(native)
        assert recorded.recorded_original_scores is recorded.recorded_scores is None
        assert recorded.recorded_post_prefix_state is None
        assert dict(recorded.recording_work) == extra
        assert all(len(values) == len(CONTEXTS) for values in calls.values())


class PayloadCapture(RecordingParallelPersistentCurrentPolicyScores):
    def score_background_roots(self, queues, available, now, lane, ages, paths):
        self.payload = dict(queue=tuple(queues), available=tuple(available), now=now,
            lane=lane, ages=tuple(map(tuple, ages)), paths=tuple(paths), outer_state=_semantic_state(self),
            input_states=(_semantic_state(self.inner_planner),
                _semantic_state(self.inner_planner.inner_planner)))
        return super().score_background_roots(queues, available, now, lane, ages, paths)


def frame_for(context, decision):
    return dict(queue=context[1], available_before=context[2], start=context[3],
        lane_ids=(context[4],), waiting_ages=context[5], observed_arrival_counts=context[6],
        option_indices=decision.selection.option_indices, rollout=asdict(decision))


def test_parallel_actual_prefix_and_last_touched_recovery_are_exact():
    dummy = TickPersistentCurrentPolicyBGIncumbentLaneRollout(OPTIONS, SEED, SAMPLES, CAP)
    dummy_states = (_semantic_state(dummy.inner_planner), _semantic_state(dummy.inner_planner.inner_planner))
    frames, touched = [], set()
    native_cls = ParallelTickPersistentCurrentPolicyBGIncumbentLaneRollout
    with native_cls(OPTIONS, SEED, SAMPLES, CAP, workers=2) as native, \
            PayloadCapture(OPTIONS, SEED, SAMPLES, CAP, workers=2) as recorded:
        for context in CONTEXTS:
            before = (_semantic_state(recorded.inner_planner), _semantic_state(recorded.inner_planner.inner_planner))
            assert before == (_semantic_state(native.inner_planner), _semantic_state(native.inner_planner.inner_planner))
            expected, actual = native(*context), recorded(*context)
            assert actual == expected and tree(recorded) == tree(native)
            assert not actual.planning_overflow
            frame = frame_for(context, actual)
            frames.append(frame)
            assert recorded.payload["input_states"] == before
            _initialize_worker(OPTIONS, SEED, SAMPLES, CAP)
            # Compare a real chunk with genuine incoming prefix state against
            # constructor input. Only touched child states may be recovered.
            payload = {**recorded.payload, "root": next(reversed(recorded.recorded_scores.costs)),
                "sample_start": 0}
            real = _run_root(payload)
            fresh = _run_root({**payload, "input_states": dummy_states})
            excluded = {"states", "actual_reuse_work", "worker_cpu_s", "worker_wall_s",
                "constructor_cpu_s", "constructor_wall_s", "preprocessing"}
            assert {k: v for k, v in real.items() if k not in excluded} == {
                k: v for k, v in fresh.items() if k not in excluded}
            for level, flag in enumerate(("current_touched", "BG_touched")):
                if real[flag]:
                    touched.add(level)
                    assert real["states"][level] == fresh["states"][level]
                else:
                    assert real["states"][level] == before[level]
                    assert fresh["states"][level] == dummy_states[level]
            recovered, work = recover_frame_child_states(recorded, frame,
                recorded.payload["paths"], recorded.payload["outer_state"],
                needed=tuple(level for level, count in enumerate((actual.recourse_calls,
                    actual.inner_current_work["recourse_calls"])) if count))
            expected_states = (_semantic_state(native.inner_planner),
                _semantic_state(native.inner_planner.inner_planner))
            for level, state in recovered.items():
                assert state == expected_states[level]
            last = last_touched_frames(frames, len(frames))
            for level, index in enumerate(last):
                if index == len(frames) - 1:
                    assert level in recovered
                elif index is None:
                    assert expected_states[level] == dummy_states[level]
            assert work
        assert touched == {0, 1}
        assert last_touched_frames(frames, 0) == (None, None)
        first_touch = last_touched_frames(frames, 1)
        assert all(index in (None, 0) for index in first_touch)
        assert recorded.payload["input_states"] != dummy_states
    # A native partial overflow commits only the included prefix, even though
    # other chunks have already run. Recover that prefix in forward order.
    options = (
        FrameOption("a1", (1, 0, 0), 1., ((1.,), (), ())),
        FrameOption("b1", (0, 1, 0), 1.25, ((), (1.25,), ())),
        FrameOption("c1", (0, 0, 1), 1.5, ((), (), (1.5,))),
        FrameOption("loaded-ab", (1, 1, 0), 1.75, ((.5,), (1.75,), ()), loaded=True),
        FrameOption("a2", (2, 0, 0), 1.5, ((1.5, 1.5), (), ())),
    )
    with PayloadCapture(options, "parallel-overflow", 8, 2, workers=2) as overflow:
        overflow.arrival_rates = (.5, .4, .3)
        overflow._outer_observed_counts = (500, 400, 300)
        overflow.decision_counter = 7
        paths = (PlanningArrivals(((.2, 2), (.9, 1)), False, 0, 0),) * 8
        context = ((2, 1, 1), (1000.1, 1005.3), 1000.1, 0,
            ((2.1, .3), (1.2,), (.1,)), paths)
        scored = overflow.score_background_roots(*context)
        assert scored.background_rollout_overflow
        frame = dict(queue=context[0], available_before=context[1], start=context[2],
            lane_ids=(context[3],), waiting_ages=context[4],
            rollout=dict(background_rollout_overflow=True))
        needed = tuple(level for level, count in enumerate((overflow._recourse_work["recourse_calls"],
            overflow._inner_current_work["recourse_calls"])) if count)
        state_before = tree(overflow)
        recovered, work = recover_frame_child_states(overflow, frame, paths,
            overflow.payload["outer_state"], needed=needed)
        assert tree(overflow) == state_before
        assert set(recovered) == set(needed)
        children = (overflow.inner_planner, overflow.inner_planner.inner_planner)
        assert all(state == _semantic_state(children[level]) for level, state in recovered.items())
        assert work["prefix_recovery_chunks"] == overflow.parallel_work["included_chunks"]
        assert overflow.parallel_work["discarded_chunks"] > 0


if __name__ == "__main__":
    test_parent_rule_keeps_original_risk_set_and_strict_s1_with_parent_s2()
    test_fraction_strict_gain_and_risk_boundary_survive_equal_float_predictions()
    test_recording_preserves_every_native_field_fee_rng_state_and_single_bg_score()
    test_parallel_actual_prefix_and_last_touched_recovery_are_exact()
    print("4 passed", flush=True)
