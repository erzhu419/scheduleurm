"""Force one newly selected root, then continue the frozen once-BG policy."""
from dataclasses import replace
from fractions import Fraction
from math import inf, nextafter

from algorithm.experiments.bg_incumbent_counterfactual_replay import (
    BGIncumbentCounterfactualSelector, STATUS,
)


ONCE_PREDICTION_FIELDS = (
    "chosen_predicted_completion_s", "chosen_predicted_squared_flow_s2",
    "incumbent_actual_predicted_squared_flow_s2",
)


class BGIncumbentSelectedRootSelector(BGIncumbentCounterfactualSelector):
    """Override only the first native proposal with a feasible third root.

    The full once-BG proposal consumes its normal randomness and work before
    the intervention. Its candidate and feasibility metadata remain attributes
    of that proposal; forcing a root does not establish native once-BG gate
    acceptance. Predicted values for the forced root come from the caller's
    same-state once-BG score reconstruction. Later calls freely use once-BG.
    """
    def __init__(self, planner, selected_root, selected_once_predictions, capture=()):
        super().__init__(planner, capture=capture)
        self.selected_root = selected_root
        self.selected_once_predictions = {
            key: selected_once_predictions[key] for key in ONCE_PREDICTION_FIELDS}

    def __call__(self, *context):
        first = self.calls == 0
        proposed = super().__call__(*context)
        if not first:
            return proposed
        original, = proposed.selection.option_indices
        bg, replacement = proposed.incumbent_root_index, self.selected_root
        queues = context[1]
        if (not proposed.improved_incumbent or bg is None
                or replacement in (original, bg)
                or not 0 <= replacement < len(self.planner.options)
                or not all(need <= count for need, count in zip(
                    self.planner.requirements[replacement], queues, strict=True))):
            raise ValueError("forced target must be a feasible third root at a native BG-incumbent improvement")

        control = self.planner.maxweight(*context)
        self.work["override_root_option_probes"] += control.root_option_probes
        self.work["override_calls"] += 1
        maxweight_root, = control.selection.option_indices
        selection = control.selection
        if replacement != maxweight_root:
            support = Fraction(sum(q * d for q, d in zip(
                queues, self.planner.requirements[replacement], strict=True))) / self.planner.durations[replacement]
            best_support = Fraction(sum(q * d for q, d in zip(
                queues, selection.service, strict=True))) / self.planner.durations[maxweight_root]
            score, score_exact = float(support), support == best_support
            upper = score if score_exact else selection.upper_bound
            if not score_exact and Fraction.from_float(upper) < best_support:
                upper = nextafter(upper, inf)
            selection = replace(selection,
                option_counts=tuple(int(i == replacement) for i in range(len(self.planner.options))),
                option_indices=(replacement,), service=self.planner.requirements[replacement],
                duration=float(self.planner.durations[replacement]), score=score, upper_bound=upper,
                gap=max(0., upper - score), exact=False, score_exact=score_exact, canonical_exact=False)

        selected = replace(proposed, selection=selection,
            changed=replacement != maxweight_root, improved_incumbent=replacement != bg,
            risk_changed_unconstrained_root=replacement != proposed.unconstrained_root_index,
            **self.selected_once_predictions)
        self.status.update({"applied_" + key: getattr(selected, key) - getattr(proposed, key)
                            for key in STATUS})
        self.intervention = {
            "original": original, "replacement": replacement,
            "BG_incumbent_option_index": bg, "maxweight_option_index": maxweight_root,
            "actual_applied_option_index": replacement,
            "override_root_option_probes": control.root_option_probes,
            "selected_once_predictions": dict(self.selected_once_predictions),
            "forced_single_action": True,
            "retained_feasibility_metadata_scope": "native_once_BG_proposal",
        }
        return selected
