"""Causal age-tail planning over two maximum catalogue frame periods."""
from dataclasses import replace
from fractions import Fraction
from math import inf, nextafter

from algorithm.arrival_aware_lane_rollout import sample_planning_arrivals
from algorithm.continuous_arrival_age_tail_lane_rollout import ContinuousArrivalAgeTailLaneRollout


class TwoFrameWindowContinuousLaneRollout(ContinuousArrivalAgeTailLaneRollout):
    """Keep the original root rule and MW continuation; sample arrivals to 2*tau_max."""
    def _choose(self, options, queues, available, now, lane, waiting_ages):
        baseline = self.maxweight(options, queues, available, now, lane, waiting_ages)
        if not 0 < sum(queues) <= baseline.queue_limit:
            return baseline
        window = 2 * self.maximum_duration
        paths = []
        for _ in range(self.sample_count):
            paths.append(sample_planning_arrivals(self.arrival_rates, window, self.rng,
                                                 self.event_cap))
            if paths[-1].overflow:
                break
        metadata = dict(eligible=True, planning_events=sum(len(path.events) for path in paths),
            planning_random_draws=sum(path.random_draws for path in paths),
            planning_rate_reads=sum(path.rate_reads for path in paths),
            planning_overflow=any(path.overflow for path in paths),
            window_s=window, sample_count=len(paths))
        if metadata["planning_overflow"]:
            return replace(baseline, **metadata)
        scored = self.score_age_roots(queues, available, now, lane, waiting_ages, paths)
        incumbent, = baseline.selection.option_indices
        permitted = tuple(index for index in scored.costs
                          if scored.squared_costs[index] <= scored.squared_costs[incumbent])
        def ordering(index):
            return scored.costs[index], index != incumbent, self.durations[index], index
        unconstrained = min(scored.costs, key=ordering)
        chosen = min(permitted, key=ordering)
        changed = scored.costs[chosen] < scored.costs[incumbent]
        selected = baseline.selection
        if changed:
            support = Fraction(sum(q * d for q, d in zip(queues, self.requirements[chosen], strict=True))) / self.durations[chosen]
            best_support = Fraction(sum(q * d for q, d in zip(queues, selected.service, strict=True))) / self.durations[incumbent]
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
        return replace(baseline, selection=selected, candidate_count=scored.candidate_count,
            rollout_calls=scored.rollout_calls, rollout_option_probes=scored.rollout_option_probes,
            root_option_probes=baseline.root_option_probes + scored.root_option_probes,
            baseline_predicted_completion_s=float(scored.costs[incumbent] / len(paths)),
            chosen_predicted_completion_s=float(scored.costs[actual] / len(paths)),
            baseline_predicted_squared_flow_s2=float(scored.squared_costs[incumbent] / len(paths)),
            chosen_predicted_squared_flow_s2=float(scored.squared_costs[actual] / len(paths)),
            changed=changed, rollout_idle_calls=scored.rollout_idle_calls,
            risk_filtered_candidates=scored.candidate_count - len(permitted),
            risk_feasible_candidates=len(permitted), waiting_age_reads=scored.waiting_age_reads,
            flow_evaluations=scored.flow_evaluations, unconstrained_root_index=unconstrained,
            risk_changed_unconstrained_root=actual != unconstrained, **metadata)

    def __call__(self, options, queues, available, now, lane, waiting_ages,
                 observed_arrival_counts):
        rates = tuple(count / now if now > 0 else 0.0 for count in observed_arrival_counts)
        self.arrival_rates = rates
        selected = self._choose(options, queues, available, now, lane, waiting_ages)
        return replace(selected, estimated_arrival_rates=rates,
            observed_arrival_counts=tuple(observed_arrival_counts), rate_exposure_s=float(now),
            rate_estimation_coordinate_reads=len(rates),
            rate_estimation_divisions=len(rates) if now > 0 else 0)
