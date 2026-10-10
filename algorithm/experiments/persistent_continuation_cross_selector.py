"""One forced physical root followed by free persistent-current replanning."""
from collections import Counter
from dataclasses import replace
from fractions import Fraction
from math import inf, nextafter

from algorithm.experiments.bg_incumbent_counterfactual_replay import (
    BGIncumbentCounterfactualSelector, collect_work as incumbent_collect_work,
)
from algorithm.persistent_current_policy_bg_incumbent_lane_rollout import LATE_CURRENT_WORK
from algorithm.two_step_bg_incumbent_background_redecision_lane_rollout import TWO_STEP_WORK


def _planner_nodes(planner):
    nodes = []
    while planner is not None:
        nodes.append(planner)
        planner = getattr(planner, "inner_planner", None)
    return nodes


def capture_planner_tree(planner, work):
    """Capture every controller's semantic state; replay engines own queues."""
    states = []
    for node in _planner_nodes(planner):
        rng = node.rng.getstate()
        rates = tuple(node.arrival_rates)
        counts = (tuple(node._outer_observed_counts)
            if hasattr(node, "_outer_observed_counts") else None)
        states.append(dict(rng_state=rng, decision_counter=node.decision_counter,
            arrival_rates=rates, observed_counts=counts))
        work["checkpoint_planner_state_reads"] += 1
        work["checkpoint_rng_words_copied"] += len(rng[1])
        work["checkpoint_decision_counter_reads"] += 1
        work["checkpoint_arrival_rate_reads"] += len(rates)
        work["checkpoint_observed_count_reads"] += len(counts) if counts is not None else 0
    return {"planners": states}


def restore_planner_tree(planner, state):
    """Restore the common outer/current/BG state before either forced arm."""
    work = Counter()
    for node, saved in zip(_planner_nodes(planner), state["planners"], strict=True):
        rng = saved["rng_state"]
        node.rng.setstate((rng[0], tuple(rng[1]), rng[2]))
        node.decision_counter = saved["decision_counter"]
        node.arrival_rates = tuple(saved["arrival_rates"])
        counts = saved["observed_counts"]
        if counts is not None:
            node._outer_observed_counts = tuple(counts)
        work["checkpoint_planner_state_writes"] += 1
        work["checkpoint_rng_words_restored"] += len(rng[1])
        work["checkpoint_decision_counter_writes"] += 1
        work["checkpoint_arrival_rate_writes"] += len(node.arrival_rates)
        work["checkpoint_observed_count_writes"] += len(counts) if counts is not None else 0
    return work


class PersistentContinuationCrossSelector(BGIncumbentCounterfactualSelector):
    """Consume each complete native proposal, forcing only the first action.

    The supplied planner can be serial persistent or its Tick/parallel exact
    implementation. It remains active for the entire suffix. First-action
    native metadata is retained in first_proposal, while a forced application
    clears forecasts for that applied root. Capture requests retain the full
    three-controller state instead of the earlier outer-only checkpoints.
    """
    def __init__(self, planner, root=None, capture=()):
        super().__init__(planner, capture=(), force_first=False)
        self.root, self.tree_capture = root, set(capture)
        self.active_planner, self.BG_planner = planner, None
        self.first_native_decision = None

    def __call__(self, *context):
        if self.calls in self.tree_capture:
            self.states[self.calls] = capture_planner_tree(self.planner, self.work)
        proposed = super().__call__(*context)
        self.work.update({name: getattr(proposed, name) for name in TWO_STEP_WORK + LATE_CURRENT_WORK})
        self.work.update({"inner_current_" + name: value
            for name, value in proposed.inner_current_work.items()})
        self.work.update({"inner_current_" + name: value
            for name, value in proposed.inner_current_fallbacks.items()})
        self.status["persistent_proposal_calls"] += 1
        if self.calls != 1:
            return proposed
        self.first_native_decision = proposed
        self.first_proposal["metadata_scope"] = "native_persistent_current_proposal"
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
            suffix="persistent_current", forced_single_action=True,
            retained_feasibility_metadata_scope="native persistent proposal; forced action has no risk/forecast claim")
        return replace(proposed, selection=selected, **applied,
            chosen_predicted_completion_s=None, chosen_predicted_squared_flow_s2=None,
            incumbent_actual_predicted_squared_flow_s2=None)


def collect_work(selector, result):
    """Charge outer/current/BG construction and native proposal work once."""
    work = incumbent_collect_work(selector, result)
    work["recourse_controller_constructors"] += 1
    work["total_controller_constructors"] = (
        work["controller_constructors"] + work["recourse_controller_constructors"])
    return work
