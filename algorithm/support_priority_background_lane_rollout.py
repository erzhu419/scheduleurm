"""Carry a service-class priority while adapting its batch to the next queue."""
from collections import deque
from dataclasses import dataclass, fields, replace
from fractions import Fraction

from algorithm.age_tail_constrained_lane_rollout import AgeRootScores
from algorithm.arrival_aware_lane_rollout import PlanningArrivals, sample_planning_arrivals
from algorithm.background_competition_continuous_lane_rollout import _LazyBackgroundArrivals
from algorithm.deferred_incumbent_background_lane_rollout import (
    DeferredIncumbentBackgroundLaneRollout, DeferredIncumbentDecision, DeferredPlanScores,
)
from algorithm.lane_refill_simple_controls import SingleLaneSelection


@dataclass(frozen=True)
class SupportScan:
    global_index: int | None
    support_index: int | None
    feasible_options: int
    support_coordinate_reads: int
    support_comparisons: int


@dataclass(frozen=True)
class SupportPriorityPlanScores(DeferredPlanScores):
    successor_support_coordinate_reads: int
    successor_support_comparisons: int


@dataclass(frozen=True)
class SupportPriorityDecision(DeferredIncumbentDecision):
    successor_option_index: int | None = None
    successor_batch_changed: bool = False
    successor_support_coordinate_reads: int = 0
    successor_support_comparisons: int = 0


class SupportPriorityBackgroundLaneRollout(DeferredIncumbentBackgroundLaneRollout):
    """Jointly score MW continuation and one current-MW class-support handoff.

    A saved option identifies its nonempty served-class support, not its batch.
    The first nonempty virtual or real handoff uses the same single-scan rule:
    maximize exact MW over feasible options whose served support is a nonempty
    subset of the saved support, otherwise use global MW. The original queue
    envelope still applies, empty calls retain the priority, and a nonempty
    call consumes it. No later veto or arrival resampling occurs at handoff.
    Only this first handoff rule is aligned: subsequent scoring still uses MW,
    while later online calls normally replan.
    """
    def maxweight(self, options, queues, available, now, lane, waiting_ages,
                  observed_arrival_counts=None):
        decision = super().maxweight(options, queues, available, now, lane, waiting_ages)
        return SupportPriorityDecision(
            **{field.name: getattr(decision, field.name)
               for field in fields(DeferredIncumbentDecision)})

    def scan_support_options(self, queues, support_option_index=None):
        """Find global and support-restricted MW in one complete option scan.

        Support reads include constructing the saved C-coordinate mask and
        reading every coordinate for each globally feasible option. Restricted
        comparisons count every supported feasible option entering its maximum.
        """
        support = None if support_option_index is None else tuple(
            need > 0 for need in self.requirements[support_option_index])
        reads = 0 if support is None else len(support)
        best = restricted = None
        best_numerator = restricted_numerator = 0
        best_denominator = restricted_denominator = 1
        feasible = comparisons = 0
        for index, counts in enumerate(self.requirements):
            if any(need > count for need, count in zip(counts, queues, strict=True)):
                continue
            feasible += 1
            weight = sum(q * d for q, d in zip(queues, counts, strict=True))
            numerator, denominator = self.duration_ratios[index]
            candidate_numerator, candidate_denominator = weight * denominator, numerator
            cross = candidate_numerator * best_denominator - best_numerator * candidate_denominator
            if cross > 0 or (cross == 0 and (self.durations[index], index)
                            < (Fraction(1) if best is None else self.durations[best],
                               len(self.options) if best is None else best)):
                best = index
                best_numerator, best_denominator = candidate_numerator, candidate_denominator
            if support is None:
                continue
            allowed, nonempty = True, False
            for need, included in zip(counts, support, strict=True):
                reads += 1
                if need:
                    nonempty = True
                    if not included:
                        allowed = False
            if not allowed or not nonempty:
                continue
            comparisons += 1
            cross = (candidate_numerator * restricted_denominator
                     - restricted_numerator * candidate_denominator)
            if cross > 0 or (cross == 0 and (self.durations[index], index)
                            < (Fraction(1) if restricted is None else self.durations[restricted],
                               len(self.options) if restricted is None else restricted)):
                restricted = index
                restricted_numerator, restricted_denominator = candidate_numerator, candidate_denominator
        return SupportScan(best, restricted, feasible, reads, comparisons)

    def _baseline_from_scan(self, scanned, queues, available, now, lane):
        if not 0 <= lane < len(available) or available[lane] > now:
            raise ValueError("the dispatched lane must already be available")
        index = scanned.global_index
        service = (0,) * len(queues) if index is None else self.requirements[index]
        duration = Fraction(1) if index is None else self.durations[index]
        support = Fraction(sum(q * d for q, d in zip(queues, service, strict=True))) / duration
        score = float(support)
        selection = SingleLaneSelection(
            tuple(int(i == index) for i in range(len(self.options))), (index,), service,
            float(duration), score, score, 0., True, True, True,
            0, scanned.feasible_options + 1, len(self.options))
        return SupportPriorityDecision(
            selection=selection, eligible=False, candidate_count=0, rollout_calls=0,
            rollout_option_probes=0, root_option_probes=len(self.options),
            baseline_predicted_completion_s=None, chosen_predicted_completion_s=None,
            changed=False, queue_limit=len(available) * self.batch_width,
            configured_sample_count=self.sample_count)

    def _consume_support(self, queues, available, now, lane):
        saved = self.pending_successor_index
        population = sum(queues)
        envelope = population <= len(available) * self.batch_width if population else None
        scanned = self.scan_support_options(queues, saved if envelope else None)
        baseline = self._baseline_from_scan(scanned, queues, available, now, lane)
        metadata = dict(successor_call=True, planned_successor_index=saved)
        if not population:
            return replace(baseline, **metadata)
        self.pending_successor_index = None
        metadata.update(successor_envelope_checks=1, successor_envelope_passed=envelope)
        if not envelope:
            return replace(baseline, **metadata)
        dynamic = scanned.support_index
        executed = dynamic is not None
        metadata.update(
            successor_feasibility_checks=1, successor_feasibility_passed=executed,
            successor_support_coordinate_reads=scanned.support_coordinate_reads,
            successor_support_comparisons=scanned.support_comparisons)
        if not executed:
            return replace(baseline, **metadata)
        return replace(
            baseline, selection=self._root_selection(dynamic, baseline.selection, queues),
            successor_executed=True, successor_changed=dynamic != scanned.global_index,
            successor_option_index=dynamic, successor_batch_changed=dynamic != saved,
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
        envelope_checks = feasibility_checks = override_calls = support_reads = support_comparisons = 0
        queue_limit = len(available) * self.batch_width

        def scores(overflow=False):
            return SupportPriorityPlanScores(
                costs, squared, len(roots), calls, calls * len(self.options),
                len(self.options), idle_calls, len(prepared_paths), sum(queue), evaluations,
                sum(len(suffix.events) for suffix in suffixes),
                sum(suffix.random_draws for suffix in suffixes),
                sum(suffix.rate_reads for suffix in suffixes),
                sum(suffix.seed_initializations for suffix in suffixes),
                admitted_reads, tag_reads, background_admissions, overflow, len(plans),
                0, envelope_checks, feasibility_checks, override_calls,
                support_reads, support_comparisons)

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
            root, saved = plan
            for events, suffix in zip(prepared_paths, suffixes, strict=True):
                waiting = [deque(group) for group in visible]
                lane_ends = list(relative)
                lane_ends[lane] = self.durations[root]
                tagged_remaining = sum(queue) + len(events)
                first, second = admit(root, Fraction(0), waiting)
                cursor = background_cursor = 0
                pending = saved is not None
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
                    population = sum(current_queue)
                    envelope = pending and population > 0 and population <= queue_limit
                    scanned = self.scan_support_options(current_queue, saved if envelope else None)
                    chosen = scanned.global_index
                    calls += 1
                    if chosen is None:
                        idle_calls += 1
                        lane_ends[next_lane] = start + 1
                        continue
                    if pending:
                        pending = False
                        envelope_checks += 1
                        if envelope:
                            feasibility_checks += 1
                            support_reads += scanned.support_coordinate_reads
                            support_comparisons += scanned.support_comparisons
                            if scanned.support_index is not None:
                                chosen = scanned.support_index
                                override_calls += 1
                    additional_first, additional_second = admit(chosen, start, waiting)
                    first += additional_first
                    second += additional_second
                    lane_ends[next_lane] = start + self.durations[chosen]
                costs[plan] += first
                squared[plan] += second
        return scores()

    def _choose(self, options, queues, available, now, lane, waiting_ages):
        if self.pending_successor_index is not None:
            return self._consume_support(queues, available, now, lane)
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
        incumbent, = baseline.selection.option_indices
        scored = self.score_deferred_roots(queues, available, now, lane, waiting_ages, paths, incumbent)
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
               for field in fields(SupportPriorityPlanScores)
               if field.name not in {base.name for base in fields(AgeRootScores)}})
        if scored.background_rollout_overflow:
            metadata["planning_overflow"] = True
            return replace(baseline, **work, **metadata)
        control = (incumbent, None)
        permitted = tuple(plan for plan in scored.costs
                          if scored.squared_costs[plan] <= scored.squared_costs[control])

        def ordering(plan):
            root, successor = plan
            return (scored.costs[plan], plan != control, successor is not None,
                    self.durations[root], root)

        unconstrained = min(scored.costs, key=ordering)
        chosen = min(permitted, key=ordering)
        plan_changed = scored.costs[chosen] < scored.costs[control]
        actual = chosen if plan_changed else control
        root, saved = actual
        self.pending_successor_index = saved
        return replace(
            baseline, selection=self._root_selection(root, baseline.selection, queues),
            baseline_predicted_completion_s=float(scored.costs[control] / len(paths)),
            chosen_predicted_completion_s=float(scored.costs[actual] / len(paths)),
            baseline_predicted_squared_flow_s2=float(scored.squared_costs[control] / len(paths)),
            chosen_predicted_squared_flow_s2=float(scored.squared_costs[actual] / len(paths)),
            changed=plan_changed and root != incumbent, plan_changed=plan_changed,
            planned_successor_index=saved,
            risk_filtered_candidates=scored.plan_candidate_count - len(permitted),
            risk_feasible_candidates=len(permitted), unconstrained_root_index=unconstrained[0],
            risk_changed_unconstrained_root=root != unconstrained[0], **work, **metadata)
