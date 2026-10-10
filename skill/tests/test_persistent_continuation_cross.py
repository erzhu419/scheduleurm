"""A fixed first root consumes a full proposal before free persistent replay."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, fields, replace
from fractions import Fraction
from functools import lru_cache
import json

from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption
from algorithm.experiments.aligned_hardware_background_root_counterfactual import paired_outcomes
from algorithm.experiments.aligned_hardware_persistent_continuation_cross import (
    METRICS, compare_effects, native_first_frame,
)
from algorithm.experiments.aligned_hardware_shared_threshold_scaling_screen import check_selection
from algorithm.experiments.bg_incumbent_counterfactual_replay import FEES
from algorithm.experiments.observed_rate_contextual_lane_refill_replay import simulate_observed_rate_contextual_trajectory
from algorithm.experiments.persistent_continuation_cross_selector import (
    PersistentContinuationCrossSelector, capture_planner_tree, restore_planner_tree,
)
from algorithm.experiments.persistent_current_policy_checks import check_rollout
from algorithm.experiments.single_action_counterfactual_replay import prepare_checkpoint, resume_single_action_trajectory
from algorithm.persistent_current_policy_bg_incumbent_lane_rollout import LATE_CURRENT_WORK
from algorithm.tick_persistent_current_policy_lane_rollout import TickPersistentCurrentPolicyBGIncumbentLaneRollout
from algorithm.two_step_bg_incumbent_background_redecision_lane_rollout import TWO_STEP_WORK


OPTIONS = (
    FrameOption("single", (1,), 1., ((1.,),)),
    FrameOption("loaded-pair", (2,), 1.5, ((.5, 1.),), loaded=True),
    FrameOption("triple", (3,), 2., ((2., 2., 2.),)),
    FrameOption("third-loaded-triple", (3,), 2.7, ((.3, 1.8, 2.7),), loaded=True),
)
MODEL = FrameModel("toy", "persistent-cross", 2, ("a",), (1.,), OPTIONS)
TAPE = [(0., 0, 0), (0., 0, 1), (3., 0, 2), (3., 0, 3), (3., 0, 4), (6., 0, 5), (8., 0, 6)]


def planner():
    return TickPersistentCurrentPolicyBGIncumbentLaneRollout(OPTIONS, "persistent-cross-toy", 2, 16)


def tree(candidate):
    return capture_planner_tree(candidate, Counter())


def normalized(value):
    return json.loads(json.dumps(value))


def test_forced_first_preserves_complete_proposal_rng_fees_and_freely_replans_afterwards():
    reference, candidate = planner(), planner()
    for current in (reference, candidate):
        current.decision_counter = 7
    selector = PersistentContinuationCrossSelector(candidate, root=3, capture=(1,))
    context = (OPTIONS, (3,), (1000., 1009.), 1000., 0, ((3., 2., 1.),), (3,))
    before = reference.rng.getstate()
    proposed = reference(*context)
    selected = selector(*context)
    assert reference.rng.getstate() != before
    assert tree(candidate) == tree(reference)
    assert candidate.decision_counter == 8
    assert selector.first_native_decision == proposed
    assert selector.first_proposal["rollout"] == {k: v for k, v in asdict(proposed).items() if k != "selection"}
    assert selected.selection.option_indices == (3,)
    assert not any(check_selection(selected.selection, OPTIONS, context[1], 1, 0).values())
    permitted_changes = {"selection", "changed", "improved_incumbent", "risk_changed_unconstrained_root",
        "chosen_predicted_completion_s", "chosen_predicted_squared_flow_s2", "incumbent_actual_predicted_squared_flow_s2"}
    for attribute in fields(proposed):
        if attribute.name not in permitted_changes:
            assert getattr(selected, attribute.name) == getattr(proposed, attribute.name), attribute.name
    for name in ("chosen_predicted_completion_s", "chosen_predicted_squared_flow_s2", "incumbent_actual_predicted_squared_flow_s2"):
        assert getattr(selected, name) is None
    for name in FEES + TWO_STEP_WORK + LATE_CURRENT_WORK:
        assert selector.work[name] == getattr(proposed, name), name
    for name, value in {**proposed.inner_current_work, **proposed.inner_current_fallbacks}.items():
        assert selector.work["inner_current_" + name] == value, name
    assert selector.work["override_calls"] == 1
    assert selector.work["override_root_option_probes"] == len(OPTIONS)
    following = (OPTIONS, (2,), (1002.7, 1009.), 1002.7, 0, ((.7, 0.),), (5,))
    expected, actual = reference(*following), selector(*following)
    assert actual == expected and tree(candidate) == tree(reference)
    assert selector.states[1]["planners"][0]["decision_counter"] == 8
    assert selector.active_planner is candidate and selector.BG_planner is None
    assert selector.status["persistent_proposal_calls"] == selector.calls == 2
    assert selector.work["override_calls"] == 1
    for name in FEES + TWO_STEP_WORK + LATE_CURRENT_WORK:
        assert selector.work[name] == getattr(proposed, name) + getattr(expected, name), name


@lru_cache(maxsize=1)
def factual_fixture():
    candidate, states = planner(), {}

    def select(*context):
        states[candidate.decision_counter] = tree(candidate)
        return candidate(*context)

    result = simulate_observed_rate_contextual_trajectory(MODEL, TAPE, 8., 12., select, policy="factual")
    return result, states, tree(candidate)


def test_null_checkpoint_reproduces_births_fifo_busy_lanes_clock_and_complete_tree():
    factual, states, final_state = factual_fixture()
    target = next(i for i, frame in enumerate(factual["frames"]) if frame["queue"] == (3,))
    assert factual["frames"][0]["start"] == factual["frames"][1]["start"] == 0.
    checkpoint = prepare_checkpoint(factual, 1)
    assert checkpoint["available"] == (1.5, 0.)
    assert [job["completion"] for job in checkpoint["jobs"][:2]] == [.5, 1.]
    assert all(job["start"] is None and job["completion"] is None for job in checkpoint["jobs"][2:])
    for index in (1, target):
        checkpoint = prepare_checkpoint(factual, index)
        assert [(j["coordinate"], j["index"], j["arrival"]) for j in checkpoint["jobs"]] == [
            (j["coordinate"], j["index"], j["arrival"]) for j in factual["jobs"]]
        ages = tuple(tuple(factual["frames"][index]["start"] - checkpoint["jobs"][j]["arrival"]
            for j in queue) for queue in checkpoint["queues"])
        assert ages == factual["frames"][index]["waiting_ages"]
        candidate = planner()
        work = restore_planner_tree(candidate, states[index])
        assert work["checkpoint_planner_state_writes"] == 3
        assert work["checkpoint_rng_words_restored"] == 3 * 625
        assert tree(candidate) == states[index]
        selector = PersistentContinuationCrossSelector(candidate)
        resumed = resume_single_action_trajectory(MODEL, 8., 12., selector, checkpoint, policy="null")
        assert resumed["jobs"] == factual["jobs"]
        assert normalized(factual["frames"][:index] + resumed["frames"]) == normalized(factual["frames"])
        assert resumed["clock"] == factual["clock"] and resumed["last_frame_end"] == factual["last_frame_end"]
        assert tree(candidate) == final_state
        assert selector.work["override_calls"] == 0
    # Polls and equal-time calls do not release a birth twice or skip a counter.
    assert any(frame["option_indices"] == (None,) for frame in factual["frames"])
    assert final_state["planners"][0]["decision_counter"] == len(factual["frames"])
    for frame in factual["frames"]:
        assert frame["start"] == min(frame["available_before"])
        assert frame["lane_ids"] == (frame["available_before"].index(frame["start"]),)
        assert frame["observed_arrival_counts"] == (sum(birth <= frame["start"] for birth, _, _ in TAPE),)


def test_forced_checkpoint_matches_manual_first_action_and_only_new_native_suffix():
    factual, states, _ = factual_fixture()
    index = next(i for i, frame in enumerate(factual["frames"]) if frame["queue"] == (3,))
    checkpoint = prepare_checkpoint(factual, index)
    candidate, reference = planner(), planner()
    restore_planner_tree(candidate, states[index])
    restore_planner_tree(reference, states[index])
    selector = PersistentContinuationCrossSelector(candidate, root=3)
    manual_calls = 0

    def manual(*context):
        nonlocal manual_calls
        proposed = reference(*context)
        manual_calls += 1
        if manual_calls != 1:
            return proposed
        control = reference.maxweight(*context)
        q = context[1][0]
        support = float(Fraction(3 * q) / reference.durations[3])
        selected = replace(control.selection, option_counts=(0, 0, 0, 1), option_indices=(3,),
            service=(3,), duration=2.7, score=support, upper_bound=control.selection.upper_bound,
            gap=control.selection.upper_bound - support, exact=False, score_exact=False, canonical_exact=False)
        return replace(proposed, selection=selected, changed=True,
            improved_incumbent=proposed.incumbent_root_index != 3,
            risk_changed_unconstrained_root=proposed.unconstrained_root_index != 3,
            chosen_predicted_completion_s=None, chosen_predicted_squared_flow_s2=None,
            incumbent_actual_predicted_squared_flow_s2=None)

    actual = resume_single_action_trajectory(MODEL, 8., 12., selector, checkpoint, policy="forced")
    expected = resume_single_action_trajectory(MODEL, 8., 12., manual, checkpoint, policy="forced")
    assert actual == expected and tree(candidate) == tree(reference)
    assert selector.calls == manual_calls and selector.work["override_calls"] == 1
    assert actual["frames"][0]["option_indices"] == (3,)
    assert actual["frames"][0]["duration"] == 2.7
    assert actual["lane_events"][0]["reserved_end"] == actual["frames"][0]["start"] + 2.7
    assert actual["jobs"][:2] == factual["jobs"][:2]
    assert [(j["coordinate"], j["index"], j["arrival"]) for j in actual["jobs"]] == [
        (j["coordinate"], j["index"], j["arrival"]) for j in factual["jobs"]]
    assert len(actual["frames"]) > 1
    protocol = {"planning_samples": 2, "planning_event_cap": 16,
        "birth_cohort": [0., 8.], "arrival_horizon": 8., "observation_end": 12.}
    checked = {**actual, "frames": [native_first_frame(actual["frames"][0], selector.first_native_decision)]
        + actual["frames"][1:]}
    assert check_rollout(checked, MODEL, protocol, candidate)
    # The persisted arms omit frames but retain the snapshot coordinate count.
    compact = [{k: result[k] for k in ("jobs", "snapshots")} for result in (factual, actual)]
    assert paired_outcomes(*compact, protocol, MODEL.workloads) == paired_outcomes(
        factual, actual, protocol, MODEL.workloads)


def test_conditional_effect_signs_and_p90_require_the_full_same_birth_population():
    population = 7
    support = {"snapshots": {"observation_end": {
        label: {"arrivals": population, "common_jobs": population}
        for label in ("factual", "replacement")}}}
    old, new = (deepcopy(support) for _ in range(2))
    old.update(zip(METRICS, (-.5, -2., 1, 0, None), strict=True))
    new.update(zip(METRICS, (.25, 1., -1, 1, .75), strict=True))
    result = compare_effects(old, new, population)
    assert result["delta_T_mean"]["crossing"] == "old_help_new_loss"
    assert result["delta_T_mean"]["difference_of_conditional_effects"] == .75
    assert result["delta_T_common_p90"]["crossing"] == "old_help_new_loss"
    assert result["delta_T_common_p90"]["difference_of_conditional_effects"] == 3.
    assert result["delta_900_unfinished"]["crossing"] == "old_loss_new_help"
    assert result["delta_1200_unfinished"]["crossing"] == "contains_zero"
    assert result["delta_middle_queue_area_s"]["crossing"] == "unavailable"
    for incomplete in (old, new):
        incomplete["snapshots"]["observation_end"]["replacement"]["common_jobs"] -= 1
        result = compare_effects(old, new, population)
        assert result["delta_T_common_p90"]["crossing"] == "unavailable"
        assert not result["delta_T_common_p90"]["comparable_population"]
        assert result["delta_T_common_p90"]["difference_of_conditional_effects"] is None
        assert result["delta_T_mean"]["crossing"] == "old_help_new_loss"
        incomplete["snapshots"]["observation_end"]["replacement"]["common_jobs"] += 1
    new["snapshots"]["observation_end"]["factual"]["arrivals"] += 1
    assert compare_effects(old, new, population)["delta_T_common_p90"]["crossing"] == "unavailable"


if __name__ == "__main__":
    test_forced_first_preserves_complete_proposal_rng_fees_and_freely_replans_afterwards()
    test_null_checkpoint_reproduces_births_fifo_busy_lanes_clock_and_complete_tree()
    test_forced_checkpoint_matches_manual_first_action_and_only_new_native_suffix()
    test_conditional_effect_signs_and_p90_require_the_full_same_birth_population()
    print("4 passed", flush=True)
