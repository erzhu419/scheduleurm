"""Joint root and deferred-current-MW planning under background competition."""
from collections import deque
from dataclasses import dataclass, fields, replace
from fractions import Fraction
from math import inf, nextafter

from algorithm.age_tail_constrained_lane_rollout import AgeRootScores
from algorithm.arrival_aware_lane_rollout import PlanningArrivals, sample_planning_arrivals
from algorithm.background_competition_continuous_lane_rollout import (
    BackgroundCompetitionContinuousLaneRollout, BackgroundCompetitionDecision,
    BackgroundRootScores, _LazyBackgroundArrivals,
)


@dataclass(frozen=True)
class DeferredPlanScores(BackgroundRootScores):
    plan_candidate_count: int
    successor_feasibility_coordinate_reads: int
    successor_envelope_checks: int
    successor_feasibility_checks: int
    successor_override_calls: int


@dataclass(frozen=True)
class DeferredIncumbentDecision(BackgroundCompetitionDecision):
    plan_candidate_count: int = 0
    plan_changed: bool = False
    planned_successor_index: int | None = None
    successor_call: bool = False
    successor_executed: bool = False
    successor_changed: bool = False
    successor_feasibility_coordinate_reads: int = 0
    successor_envelope_checks: int = 0
    successor_feasibility_checks: int = 0
    successor_override_calls: int = 0
    successor_envelope_passed: bool | None = None
    successor_feasibility_passed: bool | None = None


class DeferredIncumbentBackgroundLaneRollout(BackgroundCompetitionContinuousLaneRollout):
    """Score two plans per root and honor a selected fixed successor once.

    The fixed successor is the current same-state MW option. Empty dispatches
    retain it; the first nonempty dispatch consumes it, using MW instead if the
    original queue envelope or option feasibility fails. Ordinary-MW successor
    plans retain normal online replanning. Both modes share all planning births.
    """
    def __init__(self, options, planning_seed, sample_count=8, event_cap=128):
        super().__init__(options, planning_seed, sample_count, event_cap)
        self.pending_successor_index = None

    def maxweight(self, options, queues, available, now, lane, waiting_ages,
                  observed_arrival_counts=None):
        decision = super().maxweight(options, queues, available, now, lane, waiting_ages)
        return DeferredIncumbentDecision(
            **{field.name: getattr(decision, field.name)
               for field in fields(BackgroundCompetitionDecision)})

    def _root_selection(self, index, baseline, queues):
        incumbent, = baseline.option_indices
        if index == incumbent:
            return baseline
        support = Fraction(sum(q * d for q, d in zip(
            queues, self.requirements[index], strict=True))) / self.durations[index]
        best_support = Fraction(sum(q * d for q, d in zip(
            queues, baseline.service, strict=True))) / self.durations[incumbent]
        score, score_exact = float(support), support == best_support
        upper = score if score_exact else baseline.upper_bound
        if not score_exact and Fraction.from_float(upper) < best_support:
            upper = nextafter(upper, inf)
        return replace(
            baseline, option_counts=tuple(int(i == index) for i in range(len(self.options))),
            option_indices=(index,), service=self.requirements[index],
            duration=float(self.durations[index]), score=score, upper_bound=upper,
            gap=max(0., upper - score), exact=False, score_exact=score_exact,
            canonical_exact=False)

    def _consume_successor(self, baseline, queues):
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
        return replace(baseline, selection=self._root_selection(successor, baseline.selection, queues),
                       successor_executed=True, successor_changed=successor != incumbent,
                       successor_override_calls=1, **metadata)

    def score_deferred_roots(self, queues, available, now, lane, waiting_ages, paths, incumbent):
        queue = tuple(queues)
        if len(waiting_ages) != len(queue) or any(
                len(group) != count for group, count in zip(waiting_ages, queue, strict=True)):
            raise ValueError("visible FIFO age counts differ from the waiting queue")
        visible = tuple(tuple((-Fraction.from_float(float(age)), True) for age in group)
                        for group in waiting_ages)
        roots = tuple(index for index, counts in enumerate(self.requirements)
                      if all(need <= count for need, count in zip(counts, queue, strict=True)))
        plans = tuple((root, successor) for root in roots for successor in (None, incumbent))
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
        costs = dict.fromkeys(plans, Fraction(0))
        squared = dict.fromkeys(plans, Fraction(0))
        calls = idle_calls = evaluations = admitted_reads = tag_reads = background_admissions = 0
        successor_reads = envelope_checks = feasibility_checks = override_calls = 0
        queue_limit = len(available) * self.batch_width

        def scores(overflow=False):
            return DeferredPlanScores(
                costs, squared, len(roots), calls, calls * len(self.options),
                len(self.options), idle_calls, len(prepared_paths), sum(queue), evaluations,
                sum(len(suffix.events) for suffix in suffixes),
                sum(suffix.random_draws for suffix in suffixes),
                sum(suffix.rate_reads for suffix in suffixes),
                sum(suffix.seed_initializations for suffix in suffixes),
                admitted_reads, tag_reads, background_admissions, overflow, len(plans),
                successor_reads, envelope_checks, feasibility_checks, override_calls)

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

        for plan in plans:
            root, successor = plan
            for events, suffix in zip(prepared_paths, suffixes, strict=True):
                waiting = [deque(group) for group in visible]
                lane_ends = list(relative)
                lane_ends[lane] = self.durations[root]
                tagged_remaining = sum(queue) + len(events)
                first, second = admit(root, Fraction(0), waiting)
                cursor = background_cursor = 0
                pending = successor is not None
                while tagged_remaining:
                    next_lane = min(range(len(lane_ends)), key=lambda i: (lane_ends[i], i))
                    start = lane_ends[next_lane]
                    while cursor < len(events) and events[cursor][0] <= start:
                        birth, coordinate = events[cursor]
                        waiting[coordinate].append((birth, True))
                        cursor += 1
                    if not suffix.release_through(start):
                        costs[plan] += first
                        squared[plan] += second
                        return scores(True)
                    while background_cursor < len(suffix.events):
                        birth, coordinate = suffix.events[background_cursor]
                        if birth > start:
                            break
                        waiting[coordinate].append((birth, False))
                        background_cursor += 1
                    current_queue = tuple(len(group) for group in waiting)
                    chosen, _ = self._best_index(current_queue)
                    calls += 1
                    if chosen is None:
                        idle_calls += 1
                        lane_ends[next_lane] = start + 1
                        continue
                    if pending:
                        pending = False
                        envelope_checks += 1
                        if sum(current_queue) <= queue_limit:
                            coordinates = tuple(need <= count for need, count in zip(
                                self.requirements[successor], current_queue, strict=True))
                            successor_reads += len(coordinates)
                            feasibility_checks += 1
                            if all(coordinates):
                                chosen = successor
                                override_calls += 1
                    additional_first, additional_second = admit(chosen, start, waiting)
                    first += additional_first
                    second += additional_second
                    lane_ends[next_lane] = start + self.durations[chosen]
                costs[plan] += first
                squared[plan] += second
        return scores()

    def _choose(self, options, queues, available, now, lane, waiting_ages):
        baseline = self.maxweight(options, queues, available, now, lane, waiting_ages)
        if self.pending_successor_index is not None:
            return self._consume_successor(baseline, queues)
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
        incumbent, = baseline.selection.option_indices
        scored = self.score_deferred_roots(
            queues, available, now, lane, waiting_ages, paths, incumbent)
        metadata["planning_events"] += scored.background_planning_events
        metadata["planning_random_draws"] += scored.background_planning_random_draws
        metadata["planning_rate_reads"] += scored.background_planning_rate_reads
        work = dict(
            candidate_count=scored.candidate_count, rollout_calls=scored.rollout_calls,
            rollout_option_probes=scored.rollout_option_probes,
            root_option_probes=baseline.root_option_probes + scored.root_option_probes,
            rollout_idle_calls=scored.rollout_idle_calls, waiting_age_reads=scored.waiting_age_reads,
            flow_evaluations=scored.flow_evaluations,
            **{field.name: getattr(scored, field.name)
               for field in fields(DeferredPlanScores)
               if field.name not in {base.name for base in fields(AgeRootScores)}})
        if scored.background_rollout_overflow:
            metadata["planning_overflow"] = True
            return replace(baseline, **work, **metadata)
        control_plan = (incumbent, None)
        permitted = tuple(plan for plan in scored.costs
                          if scored.squared_costs[plan] <= scored.squared_costs[control_plan])

        def ordering(plan):
            root, successor = plan
            return (scored.costs[plan], plan != control_plan, successor is not None,
                    self.durations[root], root)

        unconstrained = min(scored.costs, key=ordering)
        chosen = min(permitted, key=ordering)
        plan_changed = scored.costs[chosen] < scored.costs[control_plan]
        actual = chosen if plan_changed else control_plan
        root, successor = actual
        self.pending_successor_index = successor
        return replace(
            baseline, selection=self._root_selection(root, baseline.selection, queues),
            baseline_predicted_completion_s=float(scored.costs[control_plan] / len(paths)),
            chosen_predicted_completion_s=float(scored.costs[actual] / len(paths)),
            baseline_predicted_squared_flow_s2=float(scored.squared_costs[control_plan] / len(paths)),
            chosen_predicted_squared_flow_s2=float(scored.squared_costs[actual] / len(paths)),
            changed=plan_changed and root != incumbent, plan_changed=plan_changed,
            planned_successor_index=successor,
            risk_filtered_candidates=scored.plan_candidate_count - len(permitted),
            risk_feasible_candidates=len(permitted), unconstrained_root_index=unconstrained[0],
            risk_changed_unconstrained_root=root != unconstrained[0], **work, **metadata)
