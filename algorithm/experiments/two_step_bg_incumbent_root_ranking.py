"""Compare once- and twice-BG root rankings at a cached post-prefix state."""
from collections import Counter

from algorithm.background_competition_continuous_lane_rollout import (
    BackgroundCompetitionContinuousLaneRollout,
)
from algorithm.bg_incumbent_background_redecision_lane_rollout import INCUMBENT_SCORE_WORK
from algorithm.one_step_background_redecision_lane_rollout import RECOURSE_WORK
from algorithm.two_step_bg_incumbent_background_redecision_lane_rollout import TWO_STEP_WORK


def compare_root_scores(once, twice, frame, paths, prior_record, work):
    """Rank native scores without sampling another prefix or applying an action.

    The caller has reconstructed the once planner immediately after its outer
    prefix draws and before advancing its decision counter. The same causal
    state is copied into the twice planner. Saved BG-restoration outcomes stay
    untouched; they describe the original intervention, not a third root.
    """
    info = frame["rollout"]
    original = prior_record["factual_option_index"]
    bg = prior_record["replacement_option_index"]
    if (not info["improved_incumbent"] or info["planning_overflow"]
            or once.decision_counter != prior_record["decision_counter"]
            or original != frame["option_indices"][0] or bg != info["incumbent_root_index"]):
        raise ValueError("original target must be a complete same-state incumbent improvement")

    state = once.rng.getstate()
    twice.rng.setstate(state)
    twice.decision_counter = once.decision_counter
    twice._outer_observed_counts = tuple(count for count in once._outer_observed_counts)
    twice.arrival_rates = tuple(rate for rate in once.arrival_rates)
    work["comparison_rng_words_copied"] += len(state[1])
    work["comparison_rng_words_restored"] += len(state[1])
    work["comparison_counter_reads"] += 1
    work["comparison_counter_writes"] += 1
    work["comparison_count_coordinate_reads"] += len(once._outer_observed_counts)
    work["comparison_count_coordinate_writes"] += len(once._outer_observed_counts)
    work["comparison_rate_coordinate_reads"] += len(once.arrival_rates)
    work["comparison_rate_coordinate_writes"] += len(once.arrival_rates)

    context = (frame["queue"], frame["available_before"], frame["start"],
               frame["lane_ids"][0], frame["waiting_ages"])
    control = once.maxweight(once.options, *context)
    work["comparison_maxweight_root_option_probes"] += control.root_option_probes
    mw, = control.selection.option_indices
    if mw != info["incumbent_maxweight_root_index"]:
        raise ValueError("native MW root differs from the cached root")

    old = BackgroundCompetitionContinuousLaneRollout.score_background_roots(once, *context, paths)
    work.update({"old_stage_" + key: getattr(old, key) for key in INCUMBENT_SCORE_WORK})
    if old.background_rollout_overflow:
        raise ValueError("cached complete ordinary-BG stage now overflows")
    old_permitted = frozenset(index for index in old.costs
        if old.squared_costs[index] <= old.squared_costs[mw])
    old_ordering = lambda index: (old.costs[index], index != mw, once.durations[index], index)
    min(old.costs, key=old_ordering)
    old_bg = min(old_permitted, key=old_ordering)
    old_improved = old.costs[old_bg] < old.costs[mw]
    if not old_improved:
        old_bg = mw
    work["old_stage_risk_comparisons"] += old.candidate_count
    work["old_stage_ordering_comparisons"] += old.candidate_count - 1 + len(old_permitted) - 1
    work["old_stage_strict_improvement_comparisons"] += 1
    if old_bg != bg:
        raise ValueError("native ordinary-BG incumbent differs from the cache")

    once._recourse_work = Counter(recourse_root_count_reads=len(once._outer_observed_counts))
    one = once.score_background_roots(*context, paths)
    work.update({"once_" + key: getattr(one, key) for key in INCUMBENT_SCORE_WORK})
    work.update({"once_" + key: once._recourse_work[key] for key in RECOURSE_WORK})
    if one.background_rollout_overflow:
        raise ValueError("cached complete once-BG stage now overflows")
    predictions = {
        "incumbent_mw_predicted_completion_s": float(old.costs[mw] / len(paths)),
        "incumbent_predicted_completion_s": float(old.costs[bg] / len(paths)),
        "incumbent_mw_predicted_squared_flow_s2": float(old.squared_costs[mw] / len(paths)),
        "incumbent_predicted_squared_flow_s2": float(old.squared_costs[bg] / len(paths)),
        "incumbent_actual_predicted_squared_flow_s2": float(old.squared_costs[original] / len(paths)),
        "baseline_predicted_completion_s": float(one.costs[bg] / len(paths)),
        "chosen_predicted_completion_s": float(one.costs[original] / len(paths)),
        "baseline_predicted_squared_flow_s2": float(one.squared_costs[bg] / len(paths)),
        "chosen_predicted_squared_flow_s2": float(one.squared_costs[original] / len(paths)),
    }
    if any(value != info[key] for key, value in predictions.items()):
        raise ValueError("native original-stage/once-BG predictions differ from the cache")

    def choose(scored, prefix):
        permitted = []
        for index in scored.costs:
            work[prefix + "membership_checks"] += 1
            if index not in old_permitted:
                continue
            work[prefix + "risk_comparisons"] += 1
            if scored.squared_costs[index] <= scored.squared_costs[bg]:
                permitted.append(index)
        ordering = lambda index: (scored.costs[index], index != bg, once.durations[index], index)
        unconstrained = min(scored.costs, key=ordering)
        chosen = min(permitted, key=ordering)
        improved = scored.costs[chosen] < scored.costs[bg]
        work[prefix + "ordering_comparisons"] += scored.candidate_count - 1 + len(permitted) - 1
        work[prefix + "strict_improvement_comparisons"] += 1
        return chosen if improved else bg, unconstrained, permitted

    one_chosen, one_unconstrained, one_permitted = choose(one, "once_")
    if one_chosen != original:
        raise ValueError("native once-BG chooser differs from the factual root")

    def rows(scored, permitted, model):
        return [{"option_index": index,
            "S1": float(scored.costs[index] / len(paths)),
            "S2": float(scored.squared_costs[index] / len(paths)),
            "original_stage_S1": float(old.costs[index] / len(paths)),
            "original_stage_S2": float(old.squared_costs[index] / len(paths)),
            "old_stage_permitted": index in old_permitted,
            model + "_S2_permitted": scored.squared_costs[index] <= scored.squared_costs[bg],
            "permitted": index in permitted} for index in scored.costs]

    twice._recourse_work = Counter(recourse_root_count_reads=len(twice._outer_observed_counts))
    twice._two_step_work = Counter()
    two = twice.score_background_roots(*context, paths)
    work.update({"twice_" + key: getattr(two, key) for key in INCUMBENT_SCORE_WORK})
    work.update({"twice_" + key: twice._recourse_work[key] for key in RECOURSE_WORK})
    work.update({"twice_" + key: twice._two_step_work[key] for key in TWO_STEP_WORK})
    two_chosen, two_unconstrained, two_permitted = bg, None, []
    two_predictions, two_rows = None, []
    if not two.background_rollout_overflow:
        two_chosen, two_unconstrained, two_permitted = choose(two, "twice_")
        two_rows = rows(two, two_permitted, "two_step")
        two_predictions = {
            "baseline_predicted_completion_s": float(two.costs[bg] / len(paths)),
            "chosen_predicted_completion_s": float(two.costs[two_chosen] / len(paths)),
            "baseline_predicted_squared_flow_s2": float(two.squared_costs[bg] / len(paths)),
            "chosen_predicted_squared_flow_s2": float(two.squared_costs[two_chosen] / len(paths)),
        }
    return {**prior_record, "predictions_match_passed": True,
        "choice_kind": "original" if two_chosen == original else "BG" if two_chosen == bg else "third",
        "two_step_option_index": two_chosen,
        "two_step_background_overflow": two.background_rollout_overflow,
        "BG_root_index": bg, "MW_root_index": mw, "once_option_index": one_chosen,
        "once_unconstrained_root_index": one_unconstrained,
        "two_step_unconstrained_root_index": two_unconstrained,
        "once_risk_feasible_candidates": len(one_permitted),
        "two_step_risk_feasible_candidates": len(two_permitted) if two_rows else None,
        "once_scores": rows(one, one_permitted, "once"), "two_step_scores": two_rows,
        "same_model_predictions": predictions, "two_step_predictions": two_predictions}
