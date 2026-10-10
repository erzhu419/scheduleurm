"""Per-class unfinished-population protection with causal rates and no stopping time."""
from collections import deque
from dataclasses import dataclass, fields, replace
from fractions import Fraction
from math import inf, nextafter

from algorithm.arrival_aware_lane_rollout import (
    PlanningArrivals,
    sample_planning_arrivals,
)


from algorithm.age_tail_constrained_lane_rollout import AgeRootScores
from algorithm.continuous_arrival_age_tail_lane_rollout import ContinuousArrivalAgeTailLaneRollout
from algorithm.observed_rate_age_tail_lane_rollout import ObservedRateDecision


@dataclass(frozen=True)
class WindowRootScores(AgeRootScores):
    class_unfinished_counts: dict
    unfinished_accumulations: int


@dataclass(frozen=True)
class WindowTailDecision(ObservedRateDecision):
    baseline_class_unfinished_counts: tuple = ()
    chosen_class_unfinished_counts: tuple = ()
    unfinished_accumulations: int = 0
    unfinished_completion_checks: int = 0
    aggregate_risk_comparisons: int = 0
    class_unfinished_comparisons: int = 0


class WindowTailContinuousLaneRollout(ContinuousArrivalAgeTailLaneRollout):
    """Keep aggregate S2; require each class unfinished count at tau_max <= MW."""
    def maxweight(self, options, queues, available, now, lane, waiting_ages,
                  observed_arrival_counts=None):
        """The control ignores ages and reproduces ordinary exact MW."""
        decision = super().maxweight(options, queues, available, now, lane, waiting_ages)
        return WindowTailDecision(**{field.name: getattr(decision, field.name)
                                    for field in fields(ObservedRateDecision)})

    def score_age_roots(self, queues, available, now, lane, waiting_ages, paths):
        """Return exact summed S1/S2 for every feasible root and common sample.

        Each class's visible ages arrive oldest first. They become negative
        relative arrival times; sampled future arrivals append to the same
        FIFO. Sorted completion offsets bind the oldest jobs first in every
        real option. The complete option period reserves the lane even when
        one admitted target finishes earlier. Already admitted costs are
        common to all roots and omitted, while their reservations remain.
        """
        queue = tuple(queues)
        if len(waiting_ages) != len(queue) or any(len(group) != count
                                                 for group, count in zip(waiting_ages, queue, strict=True)):
            raise ValueError("visible FIFO age counts differ from the waiting queue")
        arrivals = tuple(tuple(-Fraction.from_float(float(age)) for age in group)
                         for group in waiting_ages)
        roots = tuple(index for index, counts in enumerate(self.requirements)
                      if all(need <= count for need, count in zip(counts, queue, strict=True)))
        relative = tuple(max(Fraction(0), Fraction.from_float(float(end))
                             - Fraction.from_float(float(now))) for end in available)
        prepared_paths = []
        for path in paths:
            if isinstance(path, PlanningArrivals):
                if path.overflow:
                    raise ValueError("an overflowing planning population cannot be scored")
                path = path.events
            prepared_paths.append(tuple((Fraction.from_float(float(event[0])), event[1])
                                        for event in path))
        costs = dict.fromkeys(roots, Fraction(0))
        squared = dict.fromkeys(roots, Fraction(0))
        class_counts = {root: [0] * len(queue) for root in roots}
        cutoff = Fraction.from_float(self.maximum_duration)
        calls = idle_calls = evaluations = 0

        def admit(index, start, waiting):
            nonlocal evaluations
            first = second = Fraction(0)
            for coordinate, offsets in enumerate(self.sorted_completion_offsets[index]):
                for offset in offsets:
                    completion = start + offset
                    flow = completion - waiting[coordinate].popleft()
                    first += flow
                    square = flow * flow
                    second += square
                    class_unfinished[coordinate] += int(completion > cutoff)
                    evaluations += 1
            return first, second

        for root in roots:
            for events in prepared_paths:
                class_unfinished = [0] * len(queue)
                waiting = [deque(group) for group in arrivals]
                lane_ends = list(relative)
                lane_ends[lane] = self.durations[root]
                first, second = admit(root, Fraction(0), waiting)
                cursor = 0
                while cursor < len(events) or any(waiting):
                    next_lane = min(range(len(lane_ends)), key=lambda i: (lane_ends[i], i))
                    start = lane_ends[next_lane]
                    while cursor < len(events) and events[cursor][0] <= start:
                        waiting[events[cursor][1]].append(events[cursor][0])
                        cursor += 1
                    chosen, _ = self._best_index(tuple(len(group) for group in waiting))
                    calls += 1
                    if chosen is None:
                        idle_calls += 1
                        lane_ends[next_lane] = start + 1
                        continue
                    additional_first, additional_second = admit(chosen, start, waiting)
                    first += additional_first
                    second += additional_second
                    lane_ends[next_lane] = start + self.durations[chosen]
                costs[root] += first
                squared[root] += second
                for coordinate, value in enumerate(class_unfinished):
                    class_counts[root][coordinate] += value
        return WindowRootScores(costs, squared, len(roots), calls,
            calls * len(self.options), len(self.options), idle_calls,
            len(prepared_paths), sum(queue), evaluations,
            {root: tuple(values) for root, values in class_counts.items()},
            evaluations + len(roots) * len(prepared_paths) * len(queue))

    def _choose(self, options, queues, available, now, lane, waiting_ages):
        baseline = self.maxweight(options, queues, available, now, lane, waiting_ages)
        if not 0 < sum(queues) <= baseline.queue_limit:
            return baseline
        window = min(self.maximum_duration, max(0., self.arrival_horizon - now))
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
        comparisons = aggregate_comparisons = 0
        def permitted_class_risk(index):
            nonlocal comparisons, aggregate_comparisons
            aggregate_comparisons += 1
            if scored.squared_costs[index] > scored.squared_costs[incumbent]:
                return False
            for candidate, control in zip(scored.class_unfinished_counts[index],
                                          scored.class_unfinished_counts[incumbent], strict=True):
                comparisons += 1
                if candidate > control:
                    return False
            return True
        permitted = tuple(index for index in scored.costs if permitted_class_risk(index))
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
            baseline_class_unfinished_counts=scored.class_unfinished_counts[incumbent],
            chosen_class_unfinished_counts=scored.class_unfinished_counts[actual],
            unfinished_accumulations=scored.unfinished_accumulations,
            class_unfinished_comparisons=comparisons,
            unfinished_completion_checks=scored.flow_evaluations,
            aggregate_risk_comparisons=aggregate_comparisons,
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
