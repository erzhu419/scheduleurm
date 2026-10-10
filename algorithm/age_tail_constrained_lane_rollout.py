"""Arrival-aware root improvement constrained by the same-population flow square.

Visible FIFO waiting ages provide the past part of each job's flow time. Every
feasible root is evaluated on the same independently sampled future jobs and
ordinary MaxWeight continuation. A root may improve predicted total flow only
when its predicted sum of squared flows is no larger than the MW root's.
"""
from collections import deque
from dataclasses import dataclass, fields, replace
from fractions import Fraction
from math import inf, nextafter

from algorithm.arrival_aware_lane_rollout import (
    ArrivalAwareDecision, ArrivalAwareLaneRollout, PlanningArrivals,
    sample_planning_arrivals,
)


@dataclass(frozen=True)
class AgeRootScores:
    costs: dict
    squared_costs: dict
    candidate_count: int
    rollout_calls: int
    rollout_option_probes: int
    root_option_probes: int
    rollout_idle_calls: int
    sample_count: int
    waiting_age_reads: int
    flow_evaluations: int


@dataclass(frozen=True)
class AgeTailDecision(ArrivalAwareDecision):
    baseline_predicted_squared_flow_s2: float | None = None
    chosen_predicted_squared_flow_s2: float | None = None
    risk_filtered_candidates: int = 0
    risk_feasible_candidates: int = 0
    waiting_age_reads: int = 0
    flow_evaluations: int = 0
    unconstrained_root_index: int | None = None
    risk_changed_unconstrained_root: bool = False


class AgeTailConstrainedLaneRollout(ArrivalAwareLaneRollout):
    """Keep the frozen arrival planner, adding observable ages and an S2 gate."""
    def __init__(self, options, arrival_rates, arrival_horizon, planning_seed,
                 sample_count=8, event_cap=128):
        super().__init__(options, arrival_rates, arrival_horizon, planning_seed,
                         sample_count, event_cap)
        profiles, offset_reads = [], 0
        for option in self.options:
            profile = tuple(tuple(sorted(Fraction.from_float(float(value)) for value in group))
                            for group in option.completion_offsets)
            profiles.append(profile)
            offset_reads += sum(len(group) for group in profile)
        self.sorted_completion_offsets = tuple(profiles)
        self.preprocessing_work.update(
            age_profile_option_probes=len(self.options),
            age_profile_completion_offset_reads=offset_reads)

    def maxweight(self, options, queues, available, now, lane, waiting_ages):
        """The control ignores ages and reproduces ordinary exact MW."""
        decision = super().maxweight(options, queues, available, now, lane)
        return AgeTailDecision(**{field.name: getattr(decision, field.name)
                                  for field in fields(ArrivalAwareDecision)})

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
        calls = idle_calls = evaluations = 0

        def admit(index, start, waiting):
            nonlocal evaluations
            first = second = Fraction(0)
            for coordinate, offsets in enumerate(self.sorted_completion_offsets[index]):
                for offset in offsets:
                    flow = start + offset - waiting[coordinate].popleft()
                    first += flow
                    second += flow * flow
                    evaluations += 1
            return first, second

        for root in roots:
            for events in prepared_paths:
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
        return AgeRootScores(costs, squared, len(roots), calls,
            calls * len(self.options), len(self.options), idle_calls,
            len(prepared_paths), sum(queue), evaluations)

    def __call__(self, options, queues, available, now, lane, waiting_ages):
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
