"""Explicit small tapes for shared-prefix, single-action continuation checks."""
import itertools

import pytest

from algorithm.experiments import aligned_lookahead_continuation as continuation
from algorithm.budgeted_frame_candidates import select_budgeted_frame
from algorithm.experiments.aligned_hardware_frame_experiment import simulate_frames
from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption
from algorithm.lookahead_frame_candidates import select_lookahead_frame


def _single_class_model():
    return FrameModel(
        node="toy", scenario_id="toy", lanes=1, workloads=("jobs",), scales=(1.0,),
        options=(
            FrameOption("single", (1,), 1.0, ((1.0,),)),
            FrameOption("batch", (2,), 2.0, ((2.0, 2.0),)),
        ),
    )


def _baseline():
    model = _single_class_model()
    tape = ((0.2, 0, 0), (0.3, 0, 1), (2.5, 0, 2))
    certificate = {"arrival_rates": (0.1,), "availability_loss_C": 1.0,
                   "second_moment_B": 2.12, "effective_slack_eta": 0.9}
    actions = []

    def record(options, queues, lanes, budget):
        selected = select_budgeted_frame(options, queues, lanes, budget)
        actions.append((tuple(queues), selected))
        return selected

    metrics = simulate_frames(model, tape, 3.5, certificate, 10, selector=record)
    return model, tape, certificate, actions, metrics


def _paired_completion_model():
    return FrameModel(
        node="toy", scenario_id="toy", lanes=1, workloads=("a", "b"), scales=(1.0, 1.0),
        options=(FrameOption("pair", (1, 1), 4.0, ((1.0,), (4.0,)), loaded=True),),
    )


def test_expected_parts_match_current_area_and_poisson_terminal_hand_calculation():
    model = _paired_completion_model()
    selected = select_budgeted_frame(model.options, (2, 1), 1, 10)
    result = continuation.expected_parts(model, (2, 1), selected, (0.25, 0.5), (1.0, 2.0))
    # Current: 1 deferred job*4s + completions at 1s and 4s + .75*4²/2 = 15.
    # Next means=(2,2), variances=(1,2): (4+1)/2 + (4+2)/4 = 4.
    assert result == {"current_area": 15.0, "terminal_area": 4.0, "total": 19.0}


def test_realized_area_counts_inside_arrivals_once_and_none_after_cutoff():
    model = _paired_completion_model()
    selected = select_budgeted_frame(model.options, (2, 1), 1, 10)
    times = (9.0, 10.0, 10.0, 11.0, 13.0, 14.0)
    prefix_sums = (0.0, *itertools.accumulate(times))
    # At start 10, the first three jobs are already represented in Q=(2,1).
    # Visible area=4+1+4=9; arrivals at 11/13/14 add 3+1+0, hence 13.
    assert continuation.realized_current_area(model, (2, 1), selected, 10.0, times, prefix_sums) == 13.0
    # This explicit tape ends at 14. A later frame has only its visible area.
    assert continuation.realized_current_area(model, (2, 1), selected, 20.0, times, prefix_sums) == 9.0


def test_replacing_an_action_with_itself_reproduces_every_frozen_metric():
    model, tape, certificate, actions, original = _baseline()
    called = []

    def future(queues):
        called.append(queues)
        return select_budgeted_frame(model.options, queues, model.lanes, 10)

    replay = continuation.PrefixReplay(actions, 1, actions[1][1], future)
    repeated = simulate_frames(model, tape, 3.5, certificate, 10, selector=replay)
    assert repeated == original
    assert original["mean_flow_s"] == pytest.approx(2.0)
    assert called == [(1,), (1,)]
    assert [frame["start_s"] for frame in replay.frames] == [0.0, 1.0, 2.0, 3.0]
    assert replay.now == 4.0
    assert replay.frame_index == 4


def test_one_legal_batch_replacement_then_v4_matches_independent_total_flow():
    model, tape, certificate, actions, original = _baseline()
    replacement = select_lookahead_frame(
        model.options, (2,), model.lanes, 10,
        arrival_rates=(0.1,), terminal_service_rates=(0.1,),
    )
    assert replacement.option_indices == (1,)
    assert replacement.score == replacement.seed_support == 2.0
    called = []

    def future(queues):
        called.append(queues)
        return select_budgeted_frame(model.options, queues, model.lanes, 10)

    replay = continuation.PrefixReplay(actions, 1, replacement, future)
    changed = simulate_frames(model, tape, 3.5, certificate, 10, selector=replay)
    assert replay.frames[0]["queue_before"] == (0,)
    assert replay.frames[0]["option_indices"] == actions[0][1].option_indices
    assert replay.frames[1]["start_s"] == 1.0
    assert replay.frames[1]["queue_before"] == (2,)
    assert replay.frames[1]["option_indices"] == (1,)
    assert replay.frames[2]["start_s"] == 3.0
    assert replay.frames[2]["option_indices"] == (0,)
    assert called == [(1,)]
    # Original completions at 2,3,4 give 1.8+2.7+1.5=6 job-seconds.
    # Replaced completions at 3,3,4 give 2.8+2.7+1.5=7 job-seconds.
    assert original["mean_flow_s"] * 3 == pytest.approx(6.0)
    assert changed["mean_flow_s"] * 3 == pytest.approx(7.0)
    assert changed["completed_jobs"] == 3
    assert changed["makespan_s"] == replay.now == 4.0
    assert replay.frame_index == 3


def test_decision_cache_charges_only_actual_separate_searches(monkeypatch):
    model = _single_class_model()
    invoked = []

    def support_spy(*args, **kwargs):
        result = select_budgeted_frame(*args, **kwargs)
        invoked.append(("support", result))
        return result

    def lookahead_spy(*args, **kwargs):
        result = select_lookahead_frame(*args, **kwargs)
        invoked.append(("lookahead", result))
        return result

    monkeypatch.setattr(continuation, "select_budgeted_frame", support_spy)
    monkeypatch.setattr(continuation, "select_lookahead_frame", lookahead_spy)
    pairs = continuation.DecisionPairs(model, (0.1,), (0.1,), budget=10)
    first_support = pairs.support((2,))
    first_pair = pairs.get((2,))
    charged = pairs.accounting()
    assert first_pair[0] is first_support
    assert pairs.get((2,)) == first_pair
    assert pairs.support((2,)) is first_support
    assert pairs.accounting() == charged
    assert [branch for branch, _ in invoked] == ["support", "lookahead"]
    assert charged["calls"] == 2
    assert charged["support_calls"] == charged["lookahead_calls"] == 1
    pairs.support((1,))
    counts = pairs.accounting()
    assert counts["calls"] == 3
    assert counts["support_calls"] == 2
    assert counts["lookahead_calls"] == 1
    assert counts["candidate_attempts"] == sum(result.attempts for _, result in invoked)
    assert counts["option_probes"] == sum(result.bound_option_probes for _, result in invoked)
    assert counts["threshold_evaluations"] == sum(result.bound_threshold_evaluations for _, result in invoked)
    for name in ("local_score_evaluations", "cost_candidate_evaluations", "completion_offset_reads",
                 "next_queue_coordinate_evaluations", "terminal_value_coordinate_evaluations"):
        assert counts[name] == sum(getattr(result, name, 0) for _, result in invoked)
    # These are actual offline diagnostic searches. Cached native .attempts
    # values are not added again or presented as new online work.
