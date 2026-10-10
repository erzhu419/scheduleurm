"""Exact ordinary-BG scoring with cached integer background clocks."""
from collections import deque
from fractions import Fraction
from math import inf
from random import Random

from algorithm.arrival_aware_lane_rollout import PlanningArrivals
from algorithm.background_competition_continuous_lane_rollout import BackgroundRootScores


class _TickBackgroundArrivals:
    """Keep native Poisson draws and ties; convert each next float only once."""
    def __init__(self, rates, window, window_ticks, seed, event_cap, prefix_events,
                 scale, work):
        self.rates = tuple(rates)
        self.window, self.window_ticks = float(window), window_ticks
        self.seed, self.event_cap, self.prefix_events = seed, event_cap, prefix_events
        self.scale, self.work = scale, work
        self.events = []
        self.random_draws = self.rate_reads = self.seed_initializations = 0
        self.overflow = False
        self.rng = None
        self.next_times, self.next_ticks = [], []

    def _ticks(self, time):
        numerator, denominator = time.as_integer_ratio()
        self.work["background_tick_conversions"] += 1
        return numerator * (self.scale // denominator)

    def release_through(self, start):
        if start <= self.window_ticks:
            return True
        if self.rng is None:
            self.rng = Random(self.seed)
            self.seed_initializations += 1
            for rate in self.rates:
                self.rate_reads += 1
                if rate > 0:
                    time = self.window + self.rng.expovariate(rate)
                    self.next_times.append(time)
                    self.next_ticks.append(self._ticks(time))
                    self.random_draws += 1
                else:
                    self.next_times.append(inf)
                    self.next_ticks.append(None)
        while True:
            coordinate = min(range(len(self.next_times)),
                key=lambda item: (self.next_times[item], item))
            time = self.next_times[coordinate]
            if time == inf or self.next_ticks[coordinate] > start:
                return True
            self.events.append((self.next_ticks[coordinate], coordinate))
            if self.prefix_events + len(self.events) > self.event_cap:
                self.overflow = True
                return False
            self.rate_reads += 1
            time = time + self.rng.expovariate(self.rates[coordinate])
            self.next_times[coordinate] = time
            self.next_ticks[coordinate] = self._ticks(time)
            self.random_draws += 1


def score_background_roots(planner, queues, available, now, lane, waiting_ages, paths):
    """Return native Fraction scores and fees while simulating on integer ticks.

    Native durations are at least one second, so background float births have
    denominators at most2**52. The same dyadic scale includes supplied clocks,
    ages, completion offsets and sampled prefix births. Each shared suffix
    keeps its original float recurrence/RNG; only its comparison/admission
    clocks are cached integers. Conversion work counts every generated finite
    next time, including values not yet released and the overflow event.
    """
    queue = tuple(queues)
    if len(waiting_ages) != len(queue) or any(
            len(group) != count for group, count in zip(waiting_ages, queue, strict=True)):
        raise ValueError("visible FIFO age counts differ from the waiting queue")
    roots = tuple(index for index, counts in enumerate(planner.requirements)
        if all(need <= count for need, count in zip(counts, queue, strict=True)))
    origin = Fraction.from_float(float(now))
    relative = tuple(max(Fraction(0), Fraction.from_float(float(end)) - origin)
        for end in available)
    visible = tuple(tuple(-Fraction.from_float(float(age)) for age in group)
        for group in waiting_ages)
    window = Fraction.from_float(planner.maximum_duration)
    prepared = []
    values = [*relative, *planner.durations, window]
    values.extend(birth for group in visible for birth in group)
    values.extend(offset for profile in planner.sorted_completion_offsets
        for group in profile for offset in group)
    work = planner.kernel_work
    work["scorer_calls"] += 1
    work["root_evaluations"] += len(roots)
    for path in paths:
        if isinstance(path, PlanningArrivals):
            if path.overflow:
                raise ValueError("an overflowing planning population cannot be scored")
            path = path.events
        events = tuple((Fraction.from_float(float(event[0])), event[1]) for event in path)
        prepared.append(events)
        values.extend(time for time, _ in events)
    scale = max(2**52, max(value.denominator for value in values))

    def ticks(value):
        return value.numerator * (scale // value.denominator)

    window_ticks = ticks(window)
    work["window_tick_conversions"] += 1
    suffixes = [_TickBackgroundArrivals(
        planner.arrival_rates, planner.maximum_duration, window_ticks,
        f"{planner.planning_seed}:background:{planner.decision_counter}:{sample_index}",
        planner.event_cap, len(events), scale, work)
        for sample_index, events in enumerate(prepared)]
    visible = tuple(tuple((ticks(birth), True) for birth in group) for group in visible)
    relative = tuple(map(ticks, relative))
    durations = tuple(map(ticks, planner.durations))
    offsets = tuple(tuple(tuple(map(ticks, group)) for group in profile)
        for profile in planner.sorted_completion_offsets)
    prepared = tuple(tuple((ticks(time), coordinate) for time, coordinate in events)
        for events in prepared)
    costs, squared = dict.fromkeys(roots, 0), dict.fromkeys(roots, 0)
    calls = idle_calls = evaluations = admitted_reads = background_admissions = 0

    def scores(overflow=False):
        work["rollout_calls"] += calls
        work["flow_evaluations"] += evaluations
        work["admitted_job_reads"] += admitted_reads
        actual_background_events = sum(len(suffix.events) for suffix in suffixes)
        actual_draws = sum(suffix.random_draws for suffix in suffixes)
        actual_rate_reads = sum(suffix.rate_reads for suffix in suffixes)
        actual_seeds = sum(suffix.seed_initializations for suffix in suffixes)
        work["background_planning_events"] += actual_background_events
        work["background_planning_random_draws"] += actual_draws
        work["background_planning_rate_reads"] += actual_rate_reads
        work["background_seed_initializations"] += actual_seeds
        return BackgroundRootScores(
            {root: Fraction(value, scale) for root, value in costs.items()},
            {root: Fraction(value, scale * scale) for root, value in squared.items()},
            len(roots), calls, calls * len(planner.options), len(planner.options),
            idle_calls, len(prepared), sum(queue), evaluations, actual_background_events,
            actual_draws, actual_rate_reads, actual_seeds, admitted_reads,
            admitted_reads, background_admissions, overflow)

    def admit(index, start, waiting):
        nonlocal evaluations, admitted_reads, background_admissions, tagged_remaining
        first = second = 0
        for coordinate, group in enumerate(offsets[index]):
            for offset in group:
                birth, tagged = waiting[coordinate].popleft()
                admitted_reads += 1
                if tagged:
                    flow = start + offset - birth
                    first += flow
                    second += flow * flow
                    evaluations += 1
                    tagged_remaining -= 1
                else:
                    background_admissions += 1
        return first, second

    for root in roots:
        for sample_index, events in enumerate(prepared):
            waiting = [deque(group) for group in visible]
            lane_ends = list(relative)
            lane_ends[lane] = durations[root]
            tagged_remaining = sum(queue) + len(events)
            first, second = admit(root, 0, waiting)
            cursor = background_cursor = 0
            while tagged_remaining:
                next_lane = min(range(len(lane_ends)), key=lambda i: (lane_ends[i], i))
                start = lane_ends[next_lane]
                while cursor < len(events) and events[cursor][0] <= start:
                    birth, coordinate = events[cursor]
                    waiting[coordinate].append((birth, True))
                    cursor += 1
                work["suffix_release_calls"] += 1
                suffix = suffixes[sample_index]
                if not suffix.release_through(start):
                    costs[root] += first
                    squared[root] += second
                    return scores(True)
                while background_cursor < len(suffix.events):
                    birth, coordinate = suffix.events[background_cursor]
                    if birth > start:
                        break
                    waiting[coordinate].append((birth, False))
                    background_cursor += 1
                chosen, _ = planner._best_index(tuple(map(len, waiting)))
                calls += 1
                if chosen is None:
                    idle_calls += 1
                    lane_ends[next_lane] = start + scale
                    continue
                additional_first, additional_second = admit(chosen, start, waiting)
                first += additional_first
                second += additional_second
                lane_ends[next_lane] = start + durations[chosen]
            costs[root] += first
            squared[root] += second
    return scores()
