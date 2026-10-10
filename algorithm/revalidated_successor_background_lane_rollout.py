"""Revalidate a carried successor against current-state MW before execution."""
from collections import deque
from dataclasses import dataclass, fields, replace
from fractions import Fraction

from algorithm.age_tail_constrained_lane_rollout import AgeRootScores
from algorithm.arrival_aware_lane_rollout import PlanningArrivals, sample_planning_arrivals
from algorithm.background_competition_continuous_lane_rollout import (
    BackgroundRootScores, _LazyBackgroundArrivals,
)
from algorithm.deferred_incumbent_background_lane_rollout import (
    DeferredIncumbentBackgroundLaneRollout, DeferredIncumbentDecision,
)


@dataclass(frozen=True)
class RevalidatedSuccessorDecision(DeferredIncumbentDecision):
    successor_revalidated: bool = False
    successor_rejected: bool = False


class RevalidatedSuccessorBackgroundLaneRollout(DeferredIncumbentBackgroundLaneRollout):
    """Keep the original root planner, then score a feasible changed successor.

    At execution, current FIFO ages, queues, lane commitments and causal rates
    replace the original prediction context. Exactly two roots share eight
    fresh prefixes and continuing background paths, with MW continuation. The
    carried root executes only for strict S1 improvement and nonworse S2. This
    later revalidation is not nested into the original two-plan root scoring;
    that initial continuation still predicts its single fixed successor.
    """
    def maxweight(self, options, queues, available, now, lane, waiting_ages,
                  observed_arrival_counts=None):
        decision = super().maxweight(options, queues, available, now, lane, waiting_ages)
        return RevalidatedSuccessorDecision(
            **{field.name: getattr(decision, field.name)
               for field in fields(DeferredIncumbentDecision)})

    def _choose(self, options, queues, available, now, lane, waiting_ages):
        if self.pending_successor_index is None:
            return super()._choose(options, queues, available, now, lane, waiting_ages)
        baseline = self.maxweight(options, queues, available, now, lane, waiting_ages)
        return self._consume_successor(baseline, queues, available, now, lane, waiting_ages)

    def _consume_successor(self, baseline, queues, available, now, lane, waiting_ages):
        successor = self.pending_successor_index
        metadata = dict(successor_call=True, planned_successor_index=successor)
        population = sum(queues)
        if not population:
            return replace(baseline, **metadata)
        self.pending_successor_index = None
        envelope = population <= baseline.queue_limit
        metadata.update(successor_envelope_checks=1, successor_envelope_passed=envelope)
        if not envelope:
            return replace(baseline, **metadata)
        coordinates = tuple(need <= count for need, count in zip(
            self.requirements[successor], queues, strict=True))
        feasible = all(coordinates)
        metadata.update(successor_feasibility_coordinate_reads=len(coordinates),
                        successor_feasibility_checks=1, successor_feasibility_passed=feasible)
        if not feasible:
            return replace(baseline, **metadata)
        incumbent, = baseline.selection.option_indices
        if successor == incumbent:
            return replace(baseline, successor_executed=True, successor_override_calls=1, **metadata)

        paths = []
        for _ in range(self.sample_count):
            paths.append(sample_planning_arrivals(
                self.arrival_rates, self.maximum_duration, self.rng, self.event_cap))
            if paths[-1].overflow:
                break
        metadata.update(
            eligible=True, successor_revalidated=True,
            tagged_planning_events=sum(len(path.events) for path in paths),
            planning_events=sum(len(path.events) for path in paths),
            planning_random_draws=sum(path.random_draws for path in paths),
            planning_rate_reads=sum(path.rate_reads for path in paths),
            planning_overflow=any(path.overflow for path in paths),
            window_s=self.maximum_duration, sample_count=len(paths))
        if metadata["planning_overflow"]:
            return replace(baseline, successor_rejected=True, **metadata)
        scored = self.score_successor_roots(
            queues, available, now, lane, waiting_ages, paths, successor, incumbent)
        metadata["planning_events"] += scored.background_planning_events
        metadata["planning_random_draws"] += scored.background_planning_random_draws
        metadata["planning_rate_reads"] += scored.background_planning_rate_reads
        work = dict(
            candidate_count=scored.candidate_count, plan_candidate_count=scored.candidate_count,
            rollout_calls=scored.rollout_calls, rollout_option_probes=scored.rollout_option_probes,
            root_option_probes=baseline.root_option_probes + scored.root_option_probes,
            rollout_idle_calls=scored.rollout_idle_calls, waiting_age_reads=scored.waiting_age_reads,
            flow_evaluations=scored.flow_evaluations,
            **{field.name: getattr(scored, field.name)
               for field in fields(BackgroundRootScores)
               if field.name not in {base.name for base in fields(AgeRootScores)}})
        if scored.background_rollout_overflow:
            metadata["planning_overflow"] = True
            return replace(baseline, successor_rejected=True, **work, **metadata)
        permitted = tuple(index for index in scored.costs
                          if scored.squared_costs[index] <= scored.squared_costs[incumbent])
        accepted = (scored.costs[successor] < scored.costs[incumbent]
                    and successor in permitted)
        actual = successor if accepted else incumbent
        unconstrained = min(scored.costs, key=lambda index: (
            scored.costs[index], index != incumbent, self.durations[index], index))
        return replace(
            baseline, selection=self._root_selection(actual, baseline.selection, queues),
            successor_executed=accepted, successor_changed=accepted, successor_rejected=not accepted,
            successor_override_calls=int(accepted),
            baseline_predicted_completion_s=float(scored.costs[incumbent] / len(paths)),
            chosen_predicted_completion_s=float(scored.costs[actual] / len(paths)),
            baseline_predicted_squared_flow_s2=float(scored.squared_costs[incumbent] / len(paths)),
            chosen_predicted_squared_flow_s2=float(scored.squared_costs[actual] / len(paths)),
            risk_filtered_candidates=scored.candidate_count - len(permitted),
            risk_feasible_candidates=len(permitted), unconstrained_root_index=unconstrained,
            risk_changed_unconstrained_root=actual != unconstrained, **work, **metadata)

    def score_successor_roots(self, queues, available, now, lane, waiting_ages,
                              paths, successor, incumbent):
        """Score the two already feasible roots without scanning other roots."""
        queue = tuple(queues)
        if len(waiting_ages) != len(queue) or any(
                len(group) != count for group, count in zip(waiting_ages, queue, strict=True)):
            raise ValueError("visible FIFO age counts differ from the waiting queue")
        visible = tuple(tuple((-Fraction.from_float(float(age)), True) for age in group)
                        for group in waiting_ages)
        roots = (incumbent, successor)
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
                costs, squared, 2, calls, calls * len(self.options), 2,
                idle_calls, len(prepared_paths), sum(queue), evaluations,
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
