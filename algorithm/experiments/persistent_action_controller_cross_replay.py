"""One fixed physical root followed by the same finite native controller."""
from dataclasses import replace
from fractions import Fraction
from math import inf, nextafter

from algorithm.experiments.bg_incumbent_counterfactual_replay import BGIncumbentCounterfactualSelector
from algorithm.experiments.persistent_continuation_cross_selector import (
    capture_planner_tree, restore_planner_tree, collect_work,
)
from algorithm.two_step_bg_incumbent_background_redecision_lane_rollout import TWO_STEP_WORK


class FiniteActionCrossSelector(BGIncumbentCounterfactualSelector):
    """Consume a complete finite proposal, forcing only its first action.

    The first original decision and its complete logical work remain available
    for cached-proposal validation. A forced root carries a feasible support
    certificate but no forecast claim; every later call freely uses the same
    supplied finite controller, including its original two-current cutoff.
    """
    def __init__(self, planner, root=None):
        super().__init__(planner, force_first=False)
        self.root = root
        self.active_planner, self.BG_planner = planner, None
        self.first_native_decision = None

    def __call__(self, *context):
        proposed = super().__call__(*context)
        self.work.update({name: getattr(proposed, name) for name in TWO_STEP_WORK})
        self.work.update({"inner_current_" + name: value
            for name, value in proposed.inner_current_work.items()})
        self.work.update({"inner_current_" + name: value
            for name, value in proposed.inner_current_fallbacks.items()})
        self.status["finite_proposal_calls"] += 1
        if self.calls != 1:
            return proposed
        self.first_native_decision = proposed
        self.first_proposal["metadata_scope"] = "native_finite_current_proposal"
        if self.root is None:
            return proposed
        root, queue = self.root, context[1]
        if any(need > count for need, count in zip(self.planner.requirements[root], queue, strict=True)):
            raise ValueError("registered first root is infeasible")
        control = self.planner.maxweight(*context)
        self.work["override_root_option_probes"] += control.root_option_probes
        self.work["override_calls"] += 1
        mw, = control.selection.option_indices
        support = Fraction(sum(q * d for q, d in zip(
            queue, self.planner.requirements[root], strict=True))) / self.planner.durations[root]
        best = Fraction(sum(q * d for q, d in zip(
            queue, control.selection.service, strict=True))) / self.planner.durations[mw]
        exact = support == best
        upper = float(support) if exact else control.selection.upper_bound
        if not exact and Fraction.from_float(upper) < best:
            upper = nextafter(upper, inf)
        selected = replace(control.selection,
            option_counts=tuple(int(i == root) for i in range(len(self.planner.options))),
            option_indices=(root,), service=self.planner.requirements[root],
            duration=float(self.planner.durations[root]), score=float(support), upper_bound=upper,
            gap=max(0., upper - float(support)), exact=False, score_exact=exact, canonical_exact=False)
        applied = dict(changed=root != mw, improved_incumbent=root != proposed.incumbent_root_index,
            risk_changed_unconstrained_root=root != proposed.unconstrained_root_index)
        for name, value in applied.items():
            self.status["applied_" + name] += int(value) - int(getattr(proposed, name))
        self.intervention = dict(original=self.first_proposal["original"], replacement=root,
            actual_applied_option_index=root, registered_first_root=root, maxweight_option_index=mw,
            suffix="finite_current", forced_single_action=True,
            retained_feasibility_metadata_scope="native finite proposal; forced action has no risk/forecast claim")
        return replace(proposed, selection=selected, **applied,
            chosen_predicted_completion_s=None, chosen_predicted_squared_flow_s2=None,
            incumbent_actual_predicted_squared_flow_s2=None)
