"""Value BG-incumbent improvements with two causal BG continuation calls."""
from collections import Counter, deque
from dataclasses import dataclass, fields, replace
from fractions import Fraction

from algorithm.arrival_aware_lane_rollout import PlanningArrivals
from algorithm.background_competition_continuous_lane_rollout import (
    BackgroundRootScores, _LazyBackgroundArrivals,
)
from algorithm.bg_incumbent_background_redecision_lane_rollout import (
    BGIncumbentBackgroundRedecisionDecision, BGIncumbentBackgroundRedecisionLaneRollout,
)
from algorithm.one_step_background_redecision_lane_rollout import INNER_WORK


TWO_STEP_WORK = (
    "recourse_first_calls", "recourse_second_calls", "recourse_virtual_idle_counter_updates",
)


@dataclass(frozen=True)
class TwoStepBGIncumbentBackgroundRedecisionDecision(BGIncumbentBackgroundRedecisionDecision):
    recourse_first_calls: int = 0
    recourse_second_calls: int = 0
    recourse_virtual_idle_counter_updates: int = 0


class TwoStepBGIncumbentBackgroundRedecisionLaneRollout(BGIncumbentBackgroundRedecisionLaneRollout):
    """Use at most two nonempty ordinary-BG calls, then MW, per root/sample.

    The ordinary-BG incumbent and both aggregate S2 gates are inherited. The
    first virtual BG restores the post-prefix RNG and causal call counter;
    the second continues that inner state. Released births update cumulative
    counts until the second call. Intervening idle calls advance only the inner
    counter, without spending either nonempty-call budget or drawing samples.
    The actual controller freely replans, and the outer scorer state is retained.
    """
    def __init__(self, options, planning_seed, sample_count=8, event_cap=128):
        super().__init__(options, planning_seed, sample_count, event_cap)
        self._two_step_work = Counter()

    def maxweight(self, options, queues, available, now, lane, waiting_ages,
                  observed_arrival_counts=None):
        decision = super().maxweight(options, queues, available, now, lane, waiting_ages)
        return TwoStepBGIncumbentBackgroundRedecisionDecision(
            **{field.name: getattr(decision, field.name)
               for field in fields(BGIncumbentBackgroundRedecisionDecision)})

    def _choose(self, options, queues, available, now, lane, waiting_ages):
        self._two_step_work = Counter()
        decision = super()._choose(options, queues, available, now, lane, waiting_ages)
        return replace(decision, **{name: self._two_step_work[name] for name in TWO_STEP_WORK})

    def score_background_roots(self, queues, available, now, lane, waiting_ages, paths):
        queue = tuple(queues)
        if len(waiting_ages) != len(queue) or any(
                len(group) != count for group, count in zip(waiting_ages, queue, strict=True)):
            raise ValueError("visible FIFO age counts differ from the waiting queue")
        visible = tuple(tuple((-Fraction.from_float(float(age)), True) for age in group)
                        for group in waiting_ages)
        roots = tuple(index for index, counts in enumerate(self.requirements)
                      if all(need <= count for need, count in zip(counts, queue, strict=True)))
        absolute_origin = Fraction.from_float(float(now))
        relative = tuple(max(Fraction(0), Fraction.from_float(float(end)) - absolute_origin)
                         for end in available)
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
        work, two_step_work = self._recourse_work, self._two_step_work
        post_prefix_state = self.rng.getstate()
        outer_counter = self.decision_counter
        work["recourse_rng_words_copied"] += len(post_prefix_state[1])
        work["recourse_counter_reads"] += 1
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
                counts = list(self._outer_observed_counts)
                work["recourse_count_state_reads"] += len(counts)
                tagged_remaining = sum(queue) + len(events)
                first, second = admit(root, Fraction(0), waiting)
                cursor = background_cursor = prior_idle_calls = recourse_steps = 0
                while tagged_remaining:
                    next_lane = min(range(len(lane_ends)), key=lambda i: (lane_ends[i], i))
                    start = lane_ends[next_lane]
                    while cursor < len(events) and events[cursor][0] <= start:
                        birth, coordinate = events[cursor]
                        waiting[coordinate].append((birth, True))
                        if recourse_steps < 2:
                            counts[coordinate] += 1
                            work["recourse_arrival_counter_updates"] += 1
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
                        if recourse_steps < 2:
                            counts[coordinate] += 1
                            work["recourse_arrival_counter_updates"] += 1
                        background_cursor += 1
                    current_queue = tuple(len(group) for group in waiting)
                    chosen, _ = self._best_index(current_queue)
                    calls += 1
                    if chosen is None:
                        idle_calls += 1
                        if recourse_steps == 0:
                            prior_idle_calls += 1
                        elif recourse_steps == 1:
                            self.inner_planner.decision_counter += 1
                            work["recourse_counter_reads"] += 1
                            work["recourse_counter_writes"] += 1
                            two_step_work["recourse_virtual_idle_counter_updates"] += 1
                        lane_ends[next_lane] = start + 1
                        continue
                    if recourse_steps < 2:
                        if recourse_steps == 0:
                            self.inner_planner.rng.setstate(post_prefix_state)
                            self.inner_planner.decision_counter = outer_counter + 1 + prior_idle_calls
                            work["recourse_rng_words_restored"] += len(post_prefix_state[1])
                            work["recourse_counter_writes"] += 1
                            two_step_work["recourse_first_calls"] += 1
                        else:
                            two_step_work["recourse_second_calls"] += 1
                        absolute_now = float(absolute_origin + start)
                        absolute_ends = tuple(float(absolute_origin + end) for end in lane_ends)
                        ages = tuple(tuple(float(start - birth) for birth, _ in group) for group in waiting)
                        work["recourse_context_age_reads"] += sum(current_queue)
                        work["recourse_context_coordinate_reads"] += len(counts)
                        work["recourse_context_lane_reads"] += len(lane_ends)
                        inner = self.inner_planner(self.options, current_queue, absolute_ends,
                                                   absolute_now, next_lane, ages, tuple(counts))
                        work["recourse_calls"] += 1
                        work["recourse_eligible_calls"] += inner.eligible
                        work["recourse_changed_calls"] += inner.changed
                        work["recourse_planning_overflows"] += inner.planning_overflow
                        work["recourse_background_overflows"] += inner.background_rollout_overflow
                        work["recourse_risk_changed_unconstrained_calls"] += inner.risk_changed_unconstrained_root
                        for name in INNER_WORK:
                            work["recourse_" + name] += getattr(inner, name)
                        if inner.eligible and not inner.planning_overflow:
                            work["recourse_risk_comparisons"] += inner.candidate_count
                            work["recourse_ordering_comparisons"] += (
                                inner.candidate_count - 1 + inner.risk_feasible_candidates - 1)
                            work["recourse_strict_improvement_comparisons"] += 1
                        chosen, = inner.selection.option_indices
                        recourse_steps += 1
                    additional_first, additional_second = admit(chosen, start, waiting)
                    first += additional_first
                    second += additional_second
                    lane_ends[next_lane] = start + self.durations[chosen]
                costs[root] += first
                squared[root] += second
        return scores()
