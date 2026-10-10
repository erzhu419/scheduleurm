"""Restore one class-protected root to its BG incumbent, then freely replan."""
from dataclasses import replace

from algorithm.class_protected_bg_incumbent_background_redecision_lane_rollout import (
    ClassProtectedBGIncumbentBackgroundRedecisionLaneRollout, CLASS_PROTECTION_WORK,
)
from algorithm.experiments.bg_incumbent_counterfactual_replay import (
    BGIncumbentCounterfactualSelector, FEES as BG_FEES, STATUS as BG_STATUS,
    collect_work, registered_targets, restore_planner_state,
)


FEES = BG_FEES + CLASS_PROTECTION_WORK
CLASS_STATUS = ("aggregate_risk_feasible_candidates", "class_risk_filtered_candidates")
STATUS = BG_STATUS + CLASS_STATUS


def make_planner(model, tape, protocol):
    return ClassProtectedBGIncumbentBackgroundRedecisionLaneRollout(model.options,
        f"arrival-plan-v1:{model.node}:{tape['load']}:online",
        protocol["planning_samples"], protocol["planning_event_cap"])


class ClassProtectedBGIncumbentCounterfactualSelector(BGIncumbentCounterfactualSelector):
    """Charge the full proposal while changing only the first applied root.

    The inherited selector preserves the original proposal and charges its
    ordinary and recourse work, including one extra MW scan for an override.
    Class costs and feasible counts are computed by that same proposal; no
    second scoring call is needed to account for them or restore the BG vector.
    """
    def __call__(self, *context):
        selected = super().__call__(*context)
        self.work.update({key: getattr(selected, key) for key in CLASS_PROTECTION_WORK})
        for key in CLASS_STATUS:
            self.status["proposal_" + key] += getattr(selected, key)
            self.status["applied_" + key] += getattr(selected, key)
        if self.force_first and self.calls == 1:
            selected = replace(selected,
                chosen_predicted_class_squared_flow_s2=(
                    selected.baseline_predicted_class_squared_flow_s2))
        return selected
