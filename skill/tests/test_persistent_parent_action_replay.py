"""Replay a true persistent checkpoint and change only its first action."""
from collections import Counter
from dataclasses import replace
from fractions import Fraction
from functools import lru_cache
import json
from math import inf, nextafter

from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption
from algorithm.experiments.aligned_hardware_background_root_counterfactual import paired_outcomes
from algorithm.experiments.observed_rate_contextual_lane_refill_replay import simulate_observed_rate_contextual_trajectory
from algorithm.experiments.persistent_continuation_cross_selector import (
    PersistentContinuationCrossSelector, capture_planner_tree, restore_planner_tree,
)
from algorithm.experiments.single_action_counterfactual_replay import prepare_checkpoint, resume_single_action_trajectory
from algorithm.tick_persistent_current_policy_lane_rollout import TickPersistentCurrentPolicyBGIncumbentLaneRollout


OPTIONS = (
    FrameOption("single", (1,), 1.3, ((1.3,),)),
    FrameOption("loaded-pair", (2,), 1.7, ((.3, .9),), loaded=True),
    FrameOption("triple", (3,), 2.8, ((2.8, 2.8, 2.8),)),
    FrameOption("parent-loaded-triple", (3,), 2.9, ((.3, 1.8, 2.9),), loaded=True),
)
MODEL = FrameModel("toy", "parent-action", 2, ("a",), (1.,), OPTIONS)
TAPE = [(850.2, 0, 0), (850.2, 0, 1), (852.1, 0, 2), (852.2, 0, 3),
    (852.3, 0, 4), (899.25, 0, 5), (900., 0, 6), (1199.25, 0, 7)]
PROTOCOL = {"birth_cohort": [300., 900.], "arrival_horizon": 900., "observation_end": 1200.}


def planner():
    return TickPersistentCurrentPolicyBGIncumbentLaneRollout(OPTIONS, "parent-action-replay", 2, 16)


def tree(candidate):
    return capture_planner_tree(candidate, Counter())


def normalized(value):
    return json.loads(json.dumps(value))


@lru_cache(maxsize=1)
def native_fixture():
    candidate, states = planner(), {}
    capture_next = False

    def select(*context):
        nonlocal capture_next
        if any(context[1]) or capture_next:
            states[candidate.decision_counter] = tree(candidate)
        proposed = candidate(*context)
        capture_next = any(context[1])
        return proposed

    factual = simulate_observed_rate_contextual_trajectory(MODEL, TAPE, 1200., 1200., select, policy="native")
    return factual, states, tree(candidate)


def test_native_checkpoint_reproduces_float_clock_fifo_all_fees_and_cutoff_snapshots():
    factual, states, final_state = native_fixture()
    cached = normalized(factual)
    anchor = next(i for i, frame in enumerate(factual["frames"]) if frame["queue"] == (3,))
    frame = factual["frames"][anchor]
    assert Fraction.from_float(frame["start"]) != Fraction(str(frame["start"]))
    assert min(frame["available_before"]) == frame["start"]
    candidate = planner()
    restore_planner_tree(candidate, states[anchor])
    assert tree(candidate) == states[anchor]
    assert states[anchor]["planners"][1:] != tree(planner())["planners"][1:]
    checkpoint = prepare_checkpoint(cached, anchor)
    selector = PersistentContinuationCrossSelector(candidate)
    resumed = resume_single_action_trajectory(MODEL, 1200., 1200., selector, checkpoint, policy="native")
    assert normalized(resumed["frames"]) == cached["frames"][anchor:]
    assert resumed["jobs"] == cached["jobs"]
    assert normalized(resumed["lane_events"]) == cached["lane_events"][anchor:]
    for name in ("snapshots", "waiting_area_s", "last_frame_end", "clock"):
        assert normalized(resumed[name]) == cached[name]
    assert resumed["snapshots"]["observation_end"]["arrivals"] == 8
    assert resumed["snapshots"]["observation_end"]["unfinished_jobs"] == 1
    # Production simulates births through1200; the registered [300,900]
    # cohort and its two observation cutoffs are recomputed from whole jobs.
    outcomes = paired_outcomes(cached, resumed, PROTOCOL, MODEL.workloads)
    for cutoff, common, unfinished in (("arrival_horizon", 5, 2), ("observation_end", 7, 0)):
        for group in outcomes["snapshots"][cutoff].values():
            assert group["arrivals"] == 7 and group["common_jobs"] == common
            assert group["unfinished_jobs"] == unfinished
    assert all(outcomes[key] == 0 for key in ("delta_T_mean", "delta_T_common_p90",
        "delta_900_unfinished", "delta_1200_unfinished", "delta_middle_queue_area_s"))
    assert tree(candidate) == final_state
    assert selector.calls == len(factual["frames"]) - anchor and selector.work["override_calls"] == 0


def test_busy_prefix_admissions_and_committed_completion_times_survive_resume():
    factual, states, final_state = native_fixture()
    index = next(i for i, frame in enumerate(factual["frames"])
        if frame["option_indices"] == (1,)) + 1
    frame = factual["frames"][index]
    checkpoint = prepare_checkpoint(normalized(factual), index)
    assert checkpoint["available"] == list(frame["available_before"])
    # The earlier lane was admitted at exactly this timestamp, before its tie.
    committed = factual["jobs"][:2]
    assert all(job["start"] == frame["start"] < job["completion"] for job in committed)
    assert max(checkpoint["available"]) > frame["start"]
    assert checkpoint["jobs"][:2] == committed
    assert all(job["start"] is None and job["completion"] is None for job in checkpoint["jobs"][2:])
    candidate = planner()
    restore_planner_tree(candidate, states[index])
    resumed = resume_single_action_trajectory(MODEL, 1200., 1200., candidate, checkpoint, policy="native")
    assert resumed["jobs"] == factual["jobs"]
    assert normalized(resumed["frames"]) == normalized(factual["frames"][index:])
    assert tree(candidate) == final_state
    assert resumed["jobs"][:2] == committed


def test_forced_parent_action_matches_manual_native_call_then_free_persistent_suffix():
    factual, states, _ = native_fixture()
    anchor = next(i for i, frame in enumerate(factual["frames"]) if frame["queue"] == (3,))
    checkpoint = prepare_checkpoint(normalized(factual), anchor)
    candidate, reference = planner(), planner()
    restore_planner_tree(candidate, states[anchor])
    restore_planner_tree(reference, states[anchor])
    root = 3
    selector = PersistentContinuationCrossSelector(candidate, root=root)
    manual_calls, first_native = 0, None

    def manual(*context):
        nonlocal manual_calls, first_native
        proposed = reference(*context)
        manual_calls += 1
        if manual_calls != 1:
            return proposed
        first_native = proposed
        control = reference.maxweight(*context)
        mw, = control.selection.option_indices
        support = Fraction(sum(q * d for q, d in zip(context[1], OPTIONS[root].requirements))) / reference.durations[root]
        best = Fraction(sum(q * d for q, d in zip(context[1], control.selection.service))) / reference.durations[mw]
        exact = support == best
        upper = float(support) if exact else control.selection.upper_bound
        if not exact and Fraction.from_float(upper) < best:
            upper = nextafter(upper, inf)
        selected = replace(control.selection, option_counts=(0, 0, 0, 1), option_indices=(root,),
            service=(3,), duration=2.9, score=float(support), upper_bound=upper,
            gap=max(0., upper - float(support)), exact=False, score_exact=exact, canonical_exact=False)
        return replace(proposed, selection=selected, changed=root != mw,
            improved_incumbent=root != proposed.incumbent_root_index,
            risk_changed_unconstrained_root=root != proposed.unconstrained_root_index,
            chosen_predicted_completion_s=None, chosen_predicted_squared_flow_s2=None,
            incumbent_actual_predicted_squared_flow_s2=None)

    actual = resume_single_action_trajectory(MODEL, 1200., 1200., selector, checkpoint, policy="forced")
    expected = resume_single_action_trajectory(MODEL, 1200., 1200., manual, checkpoint, policy="forced")
    assert actual == expected and tree(candidate) == tree(reference)
    assert selector.first_native_decision == first_native
    assert first_native.selection.option_indices != (root,)
    assert selector.calls == manual_calls > 1 and selector.work["override_calls"] == 1
    assert selector.status["persistent_proposal_calls"] == len(actual["frames"])
    assert selector.active_planner is candidate and selector.BG_planner is None
    assert actual["frames"][0]["option_indices"] == (root,)
    assert actual["lane_events"][0]["reserved_end"] == actual["frames"][0]["start"] + 2.9
    assert actual["jobs"][:2] == factual["jobs"][:2]
    assert [(j["coordinate"], j["index"], j["arrival"]) for j in actual["jobs"]] == [
        (j["coordinate"], j["index"], j["arrival"]) for j in factual["jobs"]]


if __name__ == "__main__":
    test_native_checkpoint_reproduces_float_clock_fifo_all_fees_and_cutoff_snapshots()
    test_busy_prefix_admissions_and_committed_completion_times_survive_resume()
    test_forced_parent_action_matches_manual_native_call_then_free_persistent_suffix()
    print("3 passed", flush=True)
