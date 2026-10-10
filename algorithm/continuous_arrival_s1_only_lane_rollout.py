"""Budget-matched continuous-arrival S1 root selection without an S2 gate.

The original age-aware scorer still computes S1 and S2 on the same eight sampled
prefixes with MW continuation. S2 is recorded only as a diagnostic; its work is
retained. The policy otherwise keeps the frozen queue envelope, prefix overflow
fallback, FIFO offsets, strict S1 improvement, MW ties and support certificate.
"""
from dataclasses import dataclass, fields, replace
from fractions import Fraction
from math import inf, nextafter

from algorithm.arrival_aware_lane_rollout import sample_planning_arrivals
from algorithm.continuous_arrival_age_tail_lane_rollout import ContinuousArrivalAgeTailLaneRollout
from algorithm.observed_rate_age_tail_lane_rollout import ObservedRateDecision


@dataclass(frozen=True)
class ContinuousArrivalS1OnlyDecision(ObservedRateDecision):
    chosen_exceeds_MW_S2: bool = False
    diagnostic_S2_comparisons: int = 0
    root_ordering_comparisons: int = 0
    strict_improvement_comparisons: int = 0


class ContinuousArrivalS1OnlyLaneRollout(ContinuousArrivalAgeTailLaneRollout):
    """Choose the exact sampled S1 minimum over all feasible roots."""

    def __call__(self, options, queues, available, now, lane, waiting_ages,
                 observed_arrival_counts):
        exposure = float(now)
        counts = tuple(observed_arrival_counts)
        self.arrival_rates = tuple(count / exposure if exposure > 0 else 0.0 for count in counts)
        control = self.maxweight(options, queues, available, now, lane, waiting_ages)
        baseline = ContinuousArrivalS1OnlyDecision(
            **{field.name: getattr(control, field.name) for field in fields(ObservedRateDecision)})
        baseline = replace(baseline, estimated_arrival_rates=self.arrival_rates,
            observed_arrival_counts=counts, rate_exposure_s=exposure,
            rate_estimation_coordinate_reads=len(counts),
            rate_estimation_divisions=len(counts) if exposure > 0 else 0)
        if not 0 < sum(queues) <= baseline.queue_limit:
            return baseline

        paths = []
        for _ in range(self.sample_count):
            paths.append(sample_planning_arrivals(
                self.arrival_rates, self.maximum_duration, self.rng, self.event_cap))
            if paths[-1].overflow:
                break
        metadata = dict(
            eligible=True, planning_events=sum(len(path.events) for path in paths),
            planning_random_draws=sum(path.random_draws for path in paths),
            planning_rate_reads=sum(path.rate_reads for path in paths),
            planning_overflow=any(path.overflow for path in paths),
            window_s=self.maximum_duration, sample_count=len(paths))
        if metadata["planning_overflow"]:
            return replace(baseline, **metadata)

        scored = self.score_age_roots(queues, available, now, lane, waiting_ages, paths)
        incumbent, = baseline.selection.option_indices

        def ordering(index):
            return scored.costs[index], index != incumbent, self.durations[index], index

        chosen = min(scored.costs, key=ordering)
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
            selected = replace(selected,
                option_counts=tuple(int(i == chosen) for i in range(len(self.options))),
                option_indices=(chosen,), service=self.requirements[chosen],
                duration=float(self.durations[chosen]), score=score, upper_bound=upper,
                gap=max(0., upper - score), exact=False, score_exact=score_exact,
                canonical_exact=False)
        actual = chosen if changed else incumbent
        exceeds_s2 = scored.squared_costs[actual] > scored.squared_costs[incumbent]
        return replace(baseline, selection=selected, candidate_count=scored.candidate_count,
            rollout_calls=scored.rollout_calls, rollout_option_probes=scored.rollout_option_probes,
            root_option_probes=baseline.root_option_probes + scored.root_option_probes,
            baseline_predicted_completion_s=float(scored.costs[incumbent] / len(paths)),
            chosen_predicted_completion_s=float(scored.costs[actual] / len(paths)),
            baseline_predicted_squared_flow_s2=float(scored.squared_costs[incumbent] / len(paths)),
            chosen_predicted_squared_flow_s2=float(scored.squared_costs[actual] / len(paths)),
            changed=changed, rollout_idle_calls=scored.rollout_idle_calls,
            risk_filtered_candidates=0, risk_feasible_candidates=scored.candidate_count,
            waiting_age_reads=scored.waiting_age_reads, flow_evaluations=scored.flow_evaluations,
            unconstrained_root_index=chosen, risk_changed_unconstrained_root=False,
            chosen_exceeds_MW_S2=exceeds_s2, diagnostic_S2_comparisons=1,
            root_ordering_comparisons=scored.candidate_count - 1,
            strict_improvement_comparisons=1, **metadata)
