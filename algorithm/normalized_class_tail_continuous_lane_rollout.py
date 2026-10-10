"""Class-normalized second-moment protection with causal continuous arrivals."""
from dataclasses import dataclass, fields, replace
from fractions import Fraction
from math import inf, nextafter

from algorithm.arrival_aware_lane_rollout import sample_planning_arrivals
from algorithm.class_tail_continuous_lane_rollout import (
    ClassTailContinuousLaneRollout, ClassTailDecision,
)


@dataclass(frozen=True)
class NormalizedClassTailDecision(ClassTailDecision):
    baseline_normalized_risk: float | None = None
    chosen_normalized_risk: float | None = None
    active_risk_classes: int = 0
    normalization_divisions: int = 0
    normalization_additions: int = 0
    normalization_comparisons: int = 0
    normalization_coordinate_reads: int = 0


class NormalizedClassTailContinuousLaneRollout(ClassTailContinuousLaneRollout):
    """Minimize S1 under the sum of same-state MW-normalized class S2 ratios.

    The common sampled FIFO population gives every root the same class jobs.
    Only classes with positive MW S2 enter the ratio; the MW root's sum equals
    their count. This replaces the aggregate S2 gate and the per-class gate.
    """

    def maxweight(self, options, queues, available, now, lane, waiting_ages,
                  observed_arrival_counts=None):
        decision = super().maxweight(options, queues, available, now, lane,
                                     waiting_ages)
        return NormalizedClassTailDecision(
            **{field.name: getattr(decision, field.name)
               for field in fields(ClassTailDecision)})

    def _choose(self, options, queues, available, now, lane, waiting_ages):
        baseline = self.maxweight(options, queues, available, now, lane, waiting_ages)
        if not 0 < sum(queues) <= baseline.queue_limit:
            return baseline
        window = min(self.maximum_duration, max(0., self.arrival_horizon - now))
        paths = []
        for _ in range(self.sample_count):
            paths.append(sample_planning_arrivals(self.arrival_rates, window,
                                                 self.rng, self.event_cap))
            if paths[-1].overflow:
                break
        metadata = dict(
            eligible=True,
            planning_events=sum(len(path.events) for path in paths),
            planning_random_draws=sum(path.random_draws for path in paths),
            planning_rate_reads=sum(path.rate_reads for path in paths),
            planning_overflow=any(path.overflow for path in paths),
            window_s=window, sample_count=len(paths))
        if metadata["planning_overflow"]:
            return replace(baseline, **metadata)

        scored = self.score_age_roots(queues, available, now, lane, waiting_ages, paths)
        incumbent, = baseline.selection.option_indices
        control = scored.class_squared_costs[incumbent]
        active = tuple((coordinate, value) for coordinate, value in enumerate(control)
                       if value > 0)
        normalized = {}
        for index in scored.costs:
            risk = Fraction(0)
            for coordinate, denominator in active:
                risk += scored.class_squared_costs[index][coordinate] / denominator
            normalized[index] = risk
        threshold = Fraction(len(active))
        permitted = tuple(index for index, risk in normalized.items()
                          if risk <= threshold)

        def ordering(index):
            return scored.costs[index], index != incumbent, self.durations[index], index

        unconstrained = min(scored.costs, key=ordering)
        chosen = min(permitted, key=ordering)
        changed = scored.costs[chosen] < scored.costs[incumbent]
        selected = baseline.selection
        if changed:
            support = Fraction(sum(q * d for q, d in zip(
                queues, self.requirements[chosen], strict=True))) / self.durations[chosen]
            best_support = Fraction(sum(q * d for q, d in zip(
                queues, selected.service, strict=True))) / self.durations[incumbent]
            score, score_exact = float(support), support == best_support
            upper = score if score_exact else selected.upper_bound
            if not score_exact and Fraction.from_float(upper) < best_support:
                upper = nextafter(upper, inf)
            selected = replace(
                selected,
                option_counts=tuple(int(i == chosen) for i in range(len(self.options))),
                option_indices=(chosen,), service=self.requirements[chosen],
                duration=float(self.durations[chosen]), score=score, upper_bound=upper,
                gap=max(0., upper - score), exact=False, score_exact=score_exact,
                canonical_exact=False)
        actual = chosen if changed else incumbent
        return replace(
            baseline, selection=selected, candidate_count=scored.candidate_count,
            rollout_calls=scored.rollout_calls,
            rollout_option_probes=scored.rollout_option_probes,
            root_option_probes=baseline.root_option_probes + scored.root_option_probes,
            baseline_predicted_completion_s=float(scored.costs[incumbent] / len(paths)),
            chosen_predicted_completion_s=float(scored.costs[actual] / len(paths)),
            baseline_predicted_squared_flow_s2=float(
                scored.squared_costs[incumbent] / len(paths)),
            chosen_predicted_squared_flow_s2=float(
                scored.squared_costs[actual] / len(paths)),
            baseline_class_squared_flow_s2=tuple(
                float(value / len(paths)) for value in control),
            chosen_class_squared_flow_s2=tuple(
                float(value / len(paths)) for value in scored.class_squared_costs[actual]),
            class_squared_accumulations=scored.class_squared_accumulations,
            baseline_normalized_risk=float(normalized[incumbent]),
            chosen_normalized_risk=float(normalized[actual]),
            active_risk_classes=len(active),
            normalization_divisions=len(normalized) * len(active),
            normalization_additions=len(normalized) * len(active),
            normalization_comparisons=len(normalized),
            normalization_coordinate_reads=len(control) + len(normalized) * len(active),
            changed=changed, rollout_idle_calls=scored.rollout_idle_calls,
            risk_filtered_candidates=scored.candidate_count - len(permitted),
            risk_feasible_candidates=len(permitted),
            waiting_age_reads=scored.waiting_age_reads,
            flow_evaluations=scored.flow_evaluations,
            unconstrained_root_index=unconstrained,
            risk_changed_unconstrained_root=actual != unconstrained, **metadata)
