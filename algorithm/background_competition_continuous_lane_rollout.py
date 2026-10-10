"""Score a finite tagged FIFO population under continuing sampled competition."""
from collections import deque
from dataclasses import dataclass, fields, replace
from fractions import Fraction
from math import inf, nextafter
from random import Random

from algorithm.age_tail_constrained_lane_rollout import AgeRootScores
from algorithm.arrival_aware_lane_rollout import PlanningArrivals, sample_planning_arrivals
from algorithm.continuous_arrival_age_tail_lane_rollout import ContinuousArrivalAgeTailLaneRollout
from algorithm.observed_rate_age_tail_lane_rollout import ObservedRateDecision


@dataclass(frozen=True)
class BackgroundRootScores(AgeRootScores):
    background_planning_events: int
    background_planning_random_draws: int
    background_planning_rate_reads: int
    background_seed_initializations: int
    admitted_job_reads: int
    tag_flag_reads: int
    background_admissions: int
    background_rollout_overflow: bool


@dataclass(frozen=True)
class BackgroundCompetitionDecision(ObservedRateDecision):
    tagged_planning_events: int = 0
    background_planning_events: int = 0
    background_planning_random_draws: int = 0
    background_planning_rate_reads: int = 0
    background_seed_initializations: int = 0
    admitted_job_reads: int = 0
    tag_flag_reads: int = 0
    background_admissions: int = 0
    background_rollout_overflow: bool = False


class _LazyBackgroundArrivals:
    """A shared chronological Poisson suffix with branch-independent cursors.

    Prefix sampling already consumed its terminal gaps. Memorylessness permits
    fresh independent clocks at the cutoff. The suffix owns separate randomness
    so longer counterfactual branches cannot change later online prefix samples.
    """
    def __init__(self, rates, window, seed, event_cap, prefix_events):
        self.rates = tuple(rates)
        self.window = float(window)
        self.seed = seed
        self.event_cap = event_cap
        self.prefix_events = prefix_events
        self.events = []
        self.random_draws = self.rate_reads = self.seed_initializations = 0
        self.overflow = False
        self.rng = None
        self.next_times = []

    def release_through(self, start):
        if start <= Fraction.from_float(self.window):
            return True
        if self.rng is None:
            self.rng = Random(self.seed)
            self.seed_initializations += 1
            for rate in self.rates:
                self.rate_reads += 1
                if rate > 0:
                    self.next_times.append(self.window + self.rng.expovariate(rate))
                    self.random_draws += 1
                else:
                    self.next_times.append(inf)
        while True:
            coordinate = min(range(len(self.next_times)),
                             key=lambda item: (self.next_times[item], item))
            time = self.next_times[coordinate]
            if time == inf or Fraction.from_float(time) > start:
                return True
            self.events.append((Fraction.from_float(time), coordinate))
            if self.prefix_events + len(self.events) > self.event_cap:
                self.overflow = True
                return False
            self.rate_reads += 1
            self.next_times[coordinate] = time + self.rng.expovariate(self.rates[coordinate])
            self.random_draws += 1


class BackgroundCompetitionContinuousLaneRollout(ContinuousArrivalAgeTailLaneRollout):
    """Keep the original S2-constrained root rule; continue MW background arrivals.

    Visible jobs and arrivals within tau_max form the scored population. Later
    sampled births join the same FIFO queues and affect MW but contribute no
    flow cost. A branch ends once every scored job has been admitted, since its
    full completion offset is then known; later background queues need no drain.
    """
    def __init__(self, options, planning_seed, sample_count=8, event_cap=128):
        super().__init__(options, planning_seed, sample_count, event_cap)
        self.planning_seed = planning_seed
        self.decision_counter = 0

    def maxweight(self, options, queues, available, now, lane, waiting_ages,
                  observed_arrival_counts=None):
        decision = super().maxweight(options, queues, available, now, lane, waiting_ages)
        return BackgroundCompetitionDecision(
            **{field.name: getattr(decision, field.name)
               for field in fields(ObservedRateDecision)})

    def score_background_roots(self, queues, available, now, lane, waiting_ages, paths):
        queue = tuple(queues)
        if len(waiting_ages) != len(queue) or any(
                len(group) != count for group, count in zip(waiting_ages, queue, strict=True)):
            raise ValueError("visible FIFO age counts differ from the waiting queue")
        visible = tuple(tuple((-Fraction.from_float(float(age)), True) for age in group)
                        for group in waiting_ages)
        roots = tuple(index for index, counts in enumerate(self.requirements)
                      if all(need <= count for need, count in zip(counts, queue, strict=True)))
        relative = tuple(max(Fraction(0), Fraction.from_float(float(end))
                             - Fraction.from_float(float(now))) for end in available)
        prepared_paths, suffixes = [], []
        for sample_index, path in enumerate(paths):
            if isinstance(path, PlanningArrivals):
                if path.overflow:
                    raise ValueError("an overflowing planning population cannot be scored")
                path = path.events
            events = tuple((Fraction.from_float(float(event[0])), event[1]) for event in path)
            prepared_paths.append(events)
            suffixes.append(_LazyBackgroundArrivals(
                self.arrival_rates, self.maximum_duration,
                f"{self.planning_seed}:background:{self.decision_counter}:{sample_index}",
                self.event_cap, len(events)))
        costs = dict.fromkeys(roots, Fraction(0))
        squared = dict.fromkeys(roots, Fraction(0))
        calls = idle_calls = evaluations = admitted_reads = tag_reads = background_admissions = 0

        def scores(overflow=False):
            return BackgroundRootScores(
                costs, squared, len(roots), calls, calls * len(self.options),
                len(self.options), idle_calls, len(prepared_paths), sum(queue), evaluations,
                sum(len(suffix.events) for suffix in suffixes),
                sum(suffix.random_draws for suffix in suffixes),
                sum(suffix.rate_reads for suffix in suffixes),
                sum(suffix.seed_initializations for suffix in suffixes),
                admitted_reads, tag_reads, background_admissions, overflow)

        def admit(index, start, waiting):
            nonlocal evaluations, admitted_reads, tag_reads, background_admissions, tagged_remaining
            first = second = Fraction(0)
            for coordinate, offsets in enumerate(self.sorted_completion_offsets[index]):
                for offset in offsets:
                    birth, tagged = waiting[coordinate].popleft()
                    admitted_reads += 1
                    tag_reads += 1
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
            for events, suffix in zip(prepared_paths, suffixes, strict=True):
                waiting = [deque(group) for group in visible]
                lane_ends = list(relative)
                lane_ends[lane] = self.durations[root]
                tagged_remaining = sum(queue) + len(events)
                first, second = admit(root, Fraction(0), waiting)
                cursor = background_cursor = 0
                while tagged_remaining:
                    next_lane = min(range(len(lane_ends)), key=lambda i: (lane_ends[i], i))
                    start = lane_ends[next_lane]
                    while cursor < len(events) and events[cursor][0] <= start:
                        birth, coordinate = events[cursor]
                        waiting[coordinate].append((birth, True))
                        cursor += 1
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
        return scores()

    def _choose(self, options, queues, available, now, lane, waiting_ages):
        baseline = self.maxweight(options, queues, available, now, lane, waiting_ages)
        if not 0 < sum(queues) <= baseline.queue_limit:
            return baseline
        paths = []
        for _ in range(self.sample_count):
            paths.append(sample_planning_arrivals(
                self.arrival_rates, self.maximum_duration, self.rng, self.event_cap))
            if paths[-1].overflow:
                break
        metadata = dict(
            eligible=True, tagged_planning_events=sum(len(path.events) for path in paths),
            planning_events=sum(len(path.events) for path in paths),
            planning_random_draws=sum(path.random_draws for path in paths),
            planning_rate_reads=sum(path.rate_reads for path in paths),
            planning_overflow=any(path.overflow for path in paths),
            window_s=self.maximum_duration, sample_count=len(paths))
        if metadata["planning_overflow"]:
            return replace(baseline, **metadata)
        scored = self.score_background_roots(queues, available, now, lane, waiting_ages, paths)
        metadata["planning_events"] += scored.background_planning_events
        metadata["planning_random_draws"] += scored.background_planning_random_draws
        metadata["planning_rate_reads"] += scored.background_planning_rate_reads
        work = dict(
            candidate_count=scored.candidate_count,
            rollout_calls=scored.rollout_calls,
            rollout_option_probes=scored.rollout_option_probes,
            root_option_probes=baseline.root_option_probes + scored.root_option_probes,
            rollout_idle_calls=scored.rollout_idle_calls,
            waiting_age_reads=scored.waiting_age_reads, flow_evaluations=scored.flow_evaluations,
            **{field.name: getattr(scored, field.name)
               for field in fields(BackgroundRootScores)
               if field.name not in {base.name for base in fields(AgeRootScores)}})
        if scored.background_rollout_overflow:
            metadata["planning_overflow"] = True
            return replace(baseline, **work, **metadata)
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
            support = Fraction(sum(q * d for q, d in zip(
                queues, self.requirements[chosen], strict=True))) / self.durations[chosen]
            best_support = Fraction(sum(q * d for q, d in zip(
                queues, selected.service, strict=True))) / self.durations[incumbent]
            score, score_exact = float(support), support == best_support
            upper = score if score_exact else selected.upper_bound
            if not score_exact and Fraction.from_float(upper) < best_support:
                upper = nextafter(upper, inf)
            selected = replace(
                selected, option_counts=tuple(int(i == chosen) for i in range(len(self.options))),
                option_indices=(chosen,), service=self.requirements[chosen],
                duration=float(self.durations[chosen]), score=score, upper_bound=upper,
                gap=max(0., upper - score), exact=False, score_exact=score_exact,
                canonical_exact=False)
        actual = chosen if changed else incumbent
        return replace(
            baseline, selection=selected,
            baseline_predicted_completion_s=float(scored.costs[incumbent] / len(paths)),
            chosen_predicted_completion_s=float(scored.costs[actual] / len(paths)),
            baseline_predicted_squared_flow_s2=float(scored.squared_costs[incumbent] / len(paths)),
            chosen_predicted_squared_flow_s2=float(scored.squared_costs[actual] / len(paths)),
            changed=changed, risk_filtered_candidates=scored.candidate_count - len(permitted),
            risk_feasible_candidates=len(permitted), unconstrained_root_index=unconstrained,
            risk_changed_unconstrained_root=actual != unconstrained, **work, **metadata)

    def __call__(self, options, queues, available, now, lane, waiting_ages,
                 observed_arrival_counts):
        rates = tuple(count / now if now > 0 else 0.0 for count in observed_arrival_counts)
        self.arrival_rates = rates
        selected = self._choose(options, queues, available, now, lane, waiting_ages)
        self.decision_counter += 1
        return replace(selected, estimated_arrival_rates=rates,
            observed_arrival_counts=tuple(observed_arrival_counts), rate_exposure_s=float(now),
            rate_estimation_coordinate_reads=len(rates),
            rate_estimation_divisions=len(rates) if now > 0 else 0)
