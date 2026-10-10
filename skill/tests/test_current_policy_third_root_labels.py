"""A third-root label changes one action without borrowing a native prediction."""
from dataclasses import replace
from fractions import Fraction

from algorithm.experiments import aligned_hardware_current_policy_third_root_labels as labels
from algorithm.experiments.aligned_hardware_frame_model import FrameOption
from algorithm.experiments.aligned_hardware_shared_threshold_scaling_screen import check_selection
from algorithm.experiments.two_step_bg_incumbent_continuation_cross_replay import restore_planner_state
from algorithm.two_step_bg_incumbent_original_risk_lane_rollout import TwoStepBGIncumbentOriginalRiskLaneRollout


OPTIONS = (
    FrameOption("single", (1,), 1., ((1.,),)),
    FrameOption("loaded-pair", (2,), 1.5, ((.5, 1.),), loaded=True),
    FrameOption("triple", (3,), 2., ((2., 2., 2.),)),
    FrameOption("third-loaded-triple", (3,), 2.7, ((.3, 1.8, 2.7),), loaded=True),
)


def planner():
    candidate = TwoStepBGIncumbentOriginalRiskLaneRollout(
        OPTIONS, "third-root-toy", sample_count=1, event_cap=16)
    native_score = candidate.score_background_roots

    def strict_first_improvement(*context):
        scored = native_score(*context)
        if context[0] == (3,) and context[2] == 0.:
            # A harness fixture, not a scientific label: preserve native work
            # while constructing the strict A/B anchor required by the selector.
            scored = replace(scored, costs={**scored.costs,
                0: scored.costs[1] - Fraction(1, 2)})
        return scored

    candidate.score_background_roots = strict_first_improvement
    return candidate


def test_third_root_is_one_swap_preserving_native_proposal_rng_counter_and_valid_certificate():
    null = labels.FirstRootSelector(planner())
    forced = labels.FirstRootSelector(planner(), root=3)
    state = {"rng_state": null.planner.rng.getstate(), "decision_counter": 7}
    restore_planner_state(null.planner, state)
    restore_planner_state(forced.planner, state)
    context = (OPTIONS, (3,), (0., 9.), 0., 0, ((0., 0., 0.),), (0,))
    native, swapped = null(*context), forced(*context)
    assert native.improved_incumbent
    assert native.selection.option_indices == (0,) and native.incumbent_root_index == 1
    assert forced.first_proposal == null.first_proposal
    assert forced.first_proposal["original"] == 0
    assert forced.first_proposal["replacement"] == 1
    assert forced.first_proposal["rollout"]["chosen_predicted_completion_s"] is not None
    assert swapped.selection.option_indices == (3,)
    assert swapped.selection.service == (3,) and swapped.selection.duration == 2.7
    assert not any(check_selection(swapped.selection, OPTIONS, context[1], 1, 0).values())
    assert not swapped.selection.score_exact and swapped.selection.upper_bound > swapped.selection.score
    assert swapped.chosen_predicted_completion_s is None
    assert swapped.chosen_predicted_squared_flow_s2 is None
    assert swapped.incumbent_actual_predicted_squared_flow_s2 is None
    assert forced.intervention["actual_applied_option_index"] == 3
    assert forced.work["override_calls"] == 1
    assert forced.work["override_root_option_probes"] == len(OPTIONS)
    assert forced.planner.decision_counter == null.planner.decision_counter == 8
    assert forced.planner.rng.getstate() == null.planner.rng.getstate() == state["rng_state"]

    reference = planner()
    restore_planner_state(reference, {"rng_state": null.planner.rng.getstate(),
                                     "decision_counter": null.planner.decision_counter})
    following = (OPTIONS, (2,), (2.7, 9.), 2.7, 0, ((.7, 0.),), (0,))
    expected = reference(*following)
    assert forced(*following) == null(*following) == expected
    assert forced.calls == 2 and forced.work["override_calls"] == 1
    assert forced.work["override_root_option_probes"] == len(OPTIONS)
    assert forced.planner.decision_counter == reference.decision_counter == 9
    assert forced.planner.rng.getstate() == reference.rng.getstate()


def test_conditional_metric_missingness_seed_support_and_two_label_scope():
    records = [
        {"seed": 41, "conditional_metrics": {"delta_T_mean": -.5,
            "delta_T_common_p90": None, "delta_900_unfinished": 0,
            "delta_1200_unfinished": None}},
        {"seed": 42, "conditional_metrics": {"delta_T_mean": 1.5,
            "delta_T_common_p90": -2., "delta_900_unfinished": 1,
            "delta_1200_unfinished": None}},
        {"seed": 41, "conditional_metrics": {"delta_T_mean": None,
            "delta_T_common_p90": 0., "delta_900_unfinished": -1,
            "delta_1200_unfinished": None}},
    ]
    result = labels.summarize(records)
    assert result["states"] == 3
    assert result["metrics"]["delta_T_mean"] == {
        "available": 2, "mean": .5, "helped": 1, "harmed": 1, "unchanged": 0}
    assert result["metrics"]["delta_T_common_p90"] == {
        "available": 2, "mean": -1., "helped": 1, "harmed": 0, "unchanged": 1}
    assert result["metrics"]["delta_900_unfinished"] == {
        "available": 3, "mean": 0., "helped": 1, "harmed": 1, "unchanged": 1}
    missing = {"available": 0, "mean": None, "helped": 0, "harmed": 0, "unchanged": 0}
    assert result["metrics"]["delta_1200_unfinished"] == missing

    seeds = {seed: labels.summarize([r for r in records if r["seed"] == seed])
             for seed in (41, 42, 43)}
    assert [seeds[seed]["states"] for seed in (41, 42, 43)] == [2, 1, 0]
    assert seeds[41]["metrics"]["delta_T_mean"] == {
        "available": 1, "mean": -.5, "helped": 1, "harmed": 0, "unchanged": 0}
    assert seeds[42]["metrics"]["delta_T_common_p90"] == {
        "available": 1, "mean": -2., "helped": 1, "harmed": 0, "unchanged": 0}
    assert all(metric == missing for metric in seeds[43]["metrics"].values())
    # Two new labels retain two-state support; they cannot represent 22 states
    # or establish the repeated-action service performance of a full policy.
    two_labels = labels.summarize(records[:2])
    assert two_labels["states"] == 2
    assert two_labels["metrics"]["delta_T_mean"]["available"] == 2
    assert "performance_gain_established" not in two_labels


if __name__ == "__main__":
    test_third_root_is_one_swap_preserving_native_proposal_rng_counter_and_valid_certificate()
    test_conditional_metric_missingness_seed_support_and_two_label_scope()
    print("2 passed", flush=True)
