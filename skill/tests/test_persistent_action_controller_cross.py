"""Finite continuation restores its own prefix and overrides only one action."""
from collections import Counter
from dataclasses import replace
from fractions import Fraction
from functools import lru_cache
from math import inf, nextafter

from algorithm.exact_reuse_lane_rollout import ExactReuseTwoStepCurrentPolicyBGIncumbentLaneRollout
from algorithm.experiments.aligned_hardware_background_root_counterfactual import paired_outcomes
from algorithm.experiments.aligned_hardware_persistent_action_controller_cross import prepare_finite_prefix
from algorithm.experiments.bg_incumbent_counterfactual_replay import FEES
from algorithm.experiments.observed_rate_contextual_lane_refill_replay import simulate_observed_rate_contextual_trajectory
from algorithm.experiments.persistent_action_controller_cross_replay import (
    FiniteActionCrossSelector, capture_planner_tree, restore_planner_tree,
)
from algorithm.experiments.single_action_counterfactual_replay import prepare_checkpoint, resume_single_action_trajectory
from algorithm.two_step_bg_incumbent_background_redecision_lane_rollout import TWO_STEP_WORK
from skill.tests.test_persistent_parent_action_replay import MODEL, OPTIONS, PROTOCOL, TAPE, normalized


def planner():
    return ExactReuseTwoStepCurrentPolicyBGIncumbentLaneRollout(OPTIONS, "finite-action-cross", 2, 16)


def tree(candidate):
    return capture_planner_tree(candidate, Counter())


@lru_cache(maxsize=1)
def finite_fixture():
    candidate, states = planner(), {}
    capture_next = False

    def select(*context):
        nonlocal capture_next
        if any(context[1]) or capture_next:
            states[candidate.decision_counter] = tree(candidate)
        proposed = candidate(*context)
        capture_next = any(context[1])
        return proposed

    factual = simulate_observed_rate_contextual_trajectory(MODEL, TAPE, 1200., 1200., select, policy="finite")
    return normalized(factual), states, tree(candidate)


def test_true_finite_prefix_and_null_suffix_reproduce_complete_cached_trajectory():
    factual, states, final_state = finite_fixture()
    anchor = next(i for i, frame in enumerate(factual["frames"]) if frame["queue"] == [3])
    busy = next(i for i, frame in enumerate(factual["frames"]) if frame["option_indices"] == [1]) + 1
    later = next(i for i, frame in enumerate(factual["frames"]) if frame["start"] >= 899 and any(frame["queue"]))
    assert Fraction.from_float(factual["frames"][anchor]["start"]) != Fraction(str(factual["frames"][anchor]["start"]))
    assert states[anchor]["planners"][1:] != tree(planner())["planners"][1:]
    assert all(actual != initial for actual, initial in zip(
        states[later]["planners"][1:], tree(planner())["planners"][1:]))
    for index in (busy, anchor, later):
        candidate, work = planner(), Counter()
        last, before = prepare_finite_prefix(candidate, factual, index, work)
        assert before == states[index] and tree(candidate) == states[index]
        assert all(i is None or i < index for i in last)
        checkpoint = prepare_checkpoint(factual, index)
        committed = [dict(job) for job in checkpoint["jobs"] if job["start"] is not None]
        if index == busy:
            assert max(checkpoint["available"]) > factual["frames"][index]["start"]
            assert any(job["completion"] > factual["frames"][index]["start"] for job in committed)
        selector = FiniteActionCrossSelector(candidate)
        resumed = resume_single_action_trajectory(MODEL, 1200., 1200., selector, checkpoint, policy="finite")
        assert normalized(resumed["frames"]) == factual["frames"][index:]
        assert normalized(resumed["lane_events"]) == factual["lane_events"][index:]
        assert resumed["jobs"] == factual["jobs"] and normalized(resumed["snapshots"]) == factual["snapshots"]
        assert resumed["last_frame_end"] == factual["last_frame_end"]
        assert tree(candidate) == final_state and selector.work["override_calls"] == 0
        assert selector.status["finite_proposal_calls"] == len(resumed["frames"])
        assert selector.first_proposal["metadata_scope"] == "native_finite_current_proposal"
        assert not hasattr(selector.first_native_decision, "late_current_calls")
        assert all(any(job == original for job in resumed["jobs"]) for original in committed)
        paired = paired_outcomes(factual, resumed, PROTOCOL, MODEL.workloads)
        assert all(paired[key] == 0 for key in ("delta_T_mean", "delta_T_common_p90",
            "delta_900_unfinished", "delta_1200_unfinished", "delta_middle_queue_area_s"))
        for cutoff, common in (("arrival_horizon", 5), ("observation_end", 7)):
            assert all(group["arrivals"] == 7 and group["common_jobs"] == common
                for group in paired["snapshots"][cutoff].values())


def test_one_forced_action_consumes_native_finite_proposal_then_freely_replans():
    factual, states, _ = finite_fixture()
    anchor = next(i for i, frame in enumerate(factual["frames"]) if frame["queue"] == [3])
    checkpoint = prepare_checkpoint(factual, anchor)
    candidate, reference = planner(), planner()
    for current in (candidate, reference):
        restore_planner_tree(current, states[anchor])
    selector = FiniteActionCrossSelector(candidate, root=3)
    native_work, calls, first_native = Counter(), 0, None

    def manual(*context):
        nonlocal calls, first_native
        proposed = reference(*context)
        calls += 1
        native_work.update({key: getattr(proposed, key) for key in FEES + TWO_STEP_WORK})
        native_work.update({"inner_current_" + key: value
            for values in (proposed.inner_current_work, proposed.inner_current_fallbacks)
            for key, value in values.items()})
        if calls != 1:
            return proposed
        first_native = proposed
        control = reference.maxweight(*context)
        mw, = control.selection.option_indices
        support = Fraction(3 * context[1][0]) / reference.durations[3]
        best = Fraction(sum(q * d for q, d in zip(context[1], control.selection.service))) / reference.durations[mw]
        exact = support == best
        upper = float(support) if exact else control.selection.upper_bound
        if not exact and Fraction.from_float(upper) < best:
            upper = nextafter(upper, inf)
        selected = replace(control.selection, option_counts=(0, 0, 0, 1), option_indices=(3,),
            service=(3,), duration=2.9, score=float(support), upper_bound=upper,
            gap=max(0., upper - float(support)), exact=False, score_exact=exact, canonical_exact=False)
        return replace(proposed, selection=selected, changed=3 != mw,
            improved_incumbent=3 != proposed.incumbent_root_index,
            risk_changed_unconstrained_root=3 != proposed.unconstrained_root_index,
            chosen_predicted_completion_s=None, chosen_predicted_squared_flow_s2=None,
            incumbent_actual_predicted_squared_flow_s2=None)

    actual = resume_single_action_trajectory(MODEL, 1200., 1200., selector, checkpoint, policy="forced")
    expected = resume_single_action_trajectory(MODEL, 1200., 1200., manual, checkpoint, policy="forced")
    assert actual == expected and tree(candidate) == tree(reference)
    assert selector.first_native_decision == first_native
    assert first_native.selection.option_indices != (3,) and not first_native.improved_incumbent
    assert selector.calls == calls > 1 and selector.work["override_calls"] == 1
    assert selector.work["override_root_option_probes"] == len(OPTIONS)
    assert selector.active_planner is candidate and selector.BG_planner is None
    assert selector.status["finite_proposal_calls"] == len(actual["frames"])
    assert all(selector.work[key] == count for key, count in native_work.items())
    assert not any(key.startswith("late_") for key in selector.work)
    assert actual["frames"][0]["option_indices"] == (3,)
    assert actual["lane_events"][0]["reserved_end"] == actual["frames"][0]["start"] + 2.9
    assert actual["jobs"][:2] == factual["jobs"][:2]
    assert [(job["coordinate"], job["index"], job["arrival"]) for job in actual["jobs"]] == [
        (job["coordinate"], job["index"], job["arrival"]) for job in factual["jobs"]]


if __name__ == "__main__":
    test_true_finite_prefix_and_null_suffix_reproduce_complete_cached_trajectory()
    test_one_forced_action_consumes_native_finite_proposal_then_freely_replans()
    print("2 passed", flush=True)
