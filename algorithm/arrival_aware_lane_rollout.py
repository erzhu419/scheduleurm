"""Independent-arrival, complete-drain lookahead on asynchronous lane renewals.

The registered arrival law and its public stopping time are model information.
The planner owns separate randomness and never receives the experiment tape.
All feasible roots share one population per planning sample; busy commitments
remain fixed and future jobs become dispatchable only at lane polling events.
"""
from dataclasses import dataclass, replace
from fractions import Fraction
from math import inf, nextafter
from random import Random

from algorithm.lane_refill_simple_controls import SingleLaneSelection


@dataclass(frozen=True)
class PlanningArrivals:
    events: tuple
    overflow: bool
    random_draws: int
    rate_reads: int


def sample_planning_arrivals(rates, window, rng, event_cap=128):
    """Sample independent class Poisson processes, stopping at cap + 1 events.

    Overflow paths are never scored or truncated into a smaller population.
    The first exponential gap beyond the common window is charged. A zero
    window consumes no draws, matching the known post-H stopping contract.
    """
    if window <= 0:
        return PlanningArrivals((), False, 0, 0)
    events, draws, reads = [], 0, 0
    for coordinate, rate in enumerate(rates):
        reads += 1
        if rate <= 0:
            continue
        time, index = 0.0, 0
        while True:
            time += rng.expovariate(rate)
            draws += 1
            if time > window:
                break
            events.append((time, coordinate, index))
            index += 1
            if len(events) > event_cap:
                return PlanningArrivals(tuple(sorted(events)), True, draws, reads)
    return PlanningArrivals(tuple(sorted(events)), False, draws, reads)


@dataclass(frozen=True)
class RootScores:
    costs: dict
    candidate_count: int
    rollout_calls: int
    rollout_option_probes: int
    root_option_probes: int
    rollout_idle_calls: int
    sample_count: int


@dataclass(frozen=True)
class ArrivalAwareDecision:
    selection: SingleLaneSelection
    eligible: bool
    candidate_count: int
    rollout_calls: int
    rollout_option_probes: int
    root_option_probes: int
    baseline_predicted_completion_s: float | None
    chosen_predicted_completion_s: float | None
    changed: bool
    queue_limit: int
    planning_events: int = 0
    planning_random_draws: int = 0
    planning_rate_reads: int = 0
    planning_overflow: bool = False
    window_s: float = 0.0
    sample_count: int = 0
    configured_sample_count: int = 8
    rollout_idle_calls: int = 0
    preprocessing_option_probes: int = 0


class ArrivalAwareLaneRollout:
    """Eight-sample bounded-queue root improvement with exact MW continuation.

    ``options`` are bound at construction; the context call's catalogue is the
    same registered catalogue. Preprocessing is charged once through
    ``preprocessing_work``; every actual current/continuation scan is charged
    in the returned decision. Costs are exact sums of the supplied float data.
    """
    def __init__(self, options, arrival_rates, arrival_horizon, planning_seed,
                 sample_count=8, event_cap=128):
        self.options = tuple(options)
        self.arrival_rates = tuple(arrival_rates)
        self.arrival_horizon = float(arrival_horizon)
        self.sample_count, self.event_cap = sample_count, event_cap
        self.rng = Random(planning_seed)
        requirements, durations, offsets, sizes = [], [], [], []
        requirement_reads = offset_reads = singleton_checks = 0
        singletons = [False] * len(self.arrival_rates)
        for option in self.options:
            counts = tuple(option.requirements)
            requirements.append(counts)
            requirement_reads += len(counts)
            size = sum(counts)
            sizes.append(size)
            durations.append(Fraction.from_float(max(1., float(option.duration))))
            values = tuple(value for group in option.completion_offsets for value in group)
            offsets.append(sum((Fraction.from_float(float(value)) for value in values), Fraction(0)))
            offset_reads += len(values)
            singleton_checks += 1
            if not option.loaded and size == 1:
                singletons[counts.index(1)] = True
        if not all(singletons):
            raise ValueError("complete-drain models need a dedicated singleton per arrival class")
        self.requirements, self.durations = tuple(requirements), tuple(durations)
        self.offset_sums, self.sizes = tuple(offsets), tuple(sizes)
        self.duration_ratios = tuple((value.numerator, value.denominator) for value in durations)
        self.batch_width, self.maximum_duration = max(sizes), float(max(durations))
        self.preprocessing_work = {
            "option_probes": len(self.options), "requirement_reads": requirement_reads,
            "completion_offset_reads": offset_reads, "singleton_checks": singleton_checks,
        }

    def _best_index(self, queue):
        """One full scan, comparing exact support ratios by integer products."""
        best, best_numerator, best_denominator, feasible = None, 0, 1, 0
        for index, counts in enumerate(self.requirements):
            if any(need > count for need, count in zip(counts, queue, strict=True)):
                continue
            feasible += 1
            weight = sum(q * d for q, d in zip(queue, counts, strict=True))
            numerator, denominator = self.duration_ratios[index]
            candidate_numerator, candidate_denominator = weight * denominator, numerator
            cross = candidate_numerator * best_denominator - best_numerator * candidate_denominator
            if cross > 0 or (cross == 0 and (self.durations[index], index)
                            < (Fraction(1) if best is None else self.durations[best],
                               len(self.options) if best is None else best)):
                best = index
                best_numerator, best_denominator = candidate_numerator, candidate_denominator
        return best, feasible

    def _selection(self, queue):
        index, feasible = self._best_index(queue)
        service = (0,) * len(queue) if index is None else self.requirements[index]
        duration = Fraction(1) if index is None else self.durations[index]
        support = Fraction(sum(q * d for q, d in zip(queue, service, strict=True))) / duration
        score = float(support)
        return SingleLaneSelection(tuple(int(i == index) for i in range(len(self.options))),
            (index,), service, float(duration), score, score, 0., True, True, True,
            0, feasible + 1, len(self.options))

    def maxweight(self, options, queues, available, now, lane):
        """Plain exact MW, sharing only constructor preprocessing with planning."""
        if not 0 <= lane < len(available) or available[lane] > now:
            raise ValueError("the dispatched lane must already be available")
        return ArrivalAwareDecision(self._selection(tuple(queues)), False, 0, 0, 0,
            len(self.options), None, None, False, len(available) * self.batch_width,
            configured_sample_count=self.sample_count)

    def score_roots(self, queues, available, now, lane, paths):
        """Score every feasible root on all supplied common populations.

        ``paths`` accepts PlanningArrivals objects or their event tuples. Costs
        are sums, not sample averages. Each continuation call, including idle
        polling, scans the whole catalogue and is charged. Previously admitted
        jobs have root-independent completion costs and are omitted, while
        their complete lane reservation times remain in the simulation.
        """
        queue = tuple(queues)
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
            events = tuple((Fraction.from_float(float(event[0])), event[1]) for event in path)
            prepared_paths.append((events, sum((event[0] for event in events), Fraction(0))))
        calls = idle_calls = 0
        costs = dict.fromkeys(roots, Fraction(0))
        for index in roots:
            for events, arrival_sum in prepared_paths:
                remaining = [q - need for q, need in zip(queue, self.requirements[index], strict=True)]
                lane_ends = list(relative)
                lane_ends[lane] = self.durations[index]
                cost, cursor = self.offset_sums[index], 0
                while cursor < len(events) or any(remaining):
                    next_lane = min(range(len(lane_ends)), key=lambda i: (lane_ends[i], i))
                    start = lane_ends[next_lane]
                    while cursor < len(events) and events[cursor][0] <= start:
                        remaining[events[cursor][1]] += 1
                        cursor += 1
                    chosen, _ = self._best_index(remaining)
                    calls += 1
                    if chosen is None:
                        idle_calls += 1
                        lane_ends[next_lane] = start + 1
                        continue
                    cost += self.sizes[chosen] * start + self.offset_sums[chosen]
                    remaining = [q - need for q, need in zip(remaining, self.requirements[chosen], strict=True)]
                    lane_ends[next_lane] = start + self.durations[chosen]
                costs[index] += cost - arrival_sum
        return RootScores(costs, len(roots), calls, calls * len(self.options),
                          len(self.options), idle_calls, len(prepared_paths))

    def __call__(self, options, queues, available, now, lane):
        baseline = self.maxweight(options, queues, available, now, lane)
        population = sum(queues)
        if not 0 < population <= baseline.queue_limit:
            return baseline
        window = min(self.maximum_duration, max(0., self.arrival_horizon - now))
        paths = []
        for _ in range(self.sample_count):
            paths.append(sample_planning_arrivals(self.arrival_rates, window, self.rng, self.event_cap))
            if paths[-1].overflow:
                break
        fields = dict(eligible=True, planning_events=sum(len(path.events) for path in paths),
            planning_random_draws=sum(path.random_draws for path in paths),
            planning_rate_reads=sum(path.rate_reads for path in paths),
            planning_overflow=any(path.overflow for path in paths), window_s=window,
            sample_count=len(paths))
        if fields["planning_overflow"]:
            return replace(baseline, **fields)
        scored = self.score_roots(queues, available, now, lane, paths)
        incumbent, = baseline.selection.option_indices
        chosen = min(scored.costs, key=lambda index: (scored.costs[index], index != incumbent,
                     self.durations[index], index))
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
        return replace(baseline, selection=selected, candidate_count=scored.candidate_count,
            rollout_calls=scored.rollout_calls, rollout_option_probes=scored.rollout_option_probes,
            root_option_probes=baseline.root_option_probes + scored.root_option_probes,
            baseline_predicted_completion_s=float(scored.costs[incumbent] / len(paths)),
            chosen_predicted_completion_s=float(scored.costs[chosen if changed else incumbent] / len(paths)),
            changed=changed, rollout_idle_calls=scored.rollout_idle_calls, **fields)
