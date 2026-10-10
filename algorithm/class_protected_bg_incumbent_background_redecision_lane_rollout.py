"""Protect each class relative to the ordinary-BG incumbent in one-step scoring."""
from collections import Counter, deque
from dataclasses import dataclass, fields, replace
from fractions import Fraction
from math import inf, nextafter

from algorithm.age_tail_constrained_lane_rollout import AgeRootScores
from algorithm.arrival_aware_lane_rollout import PlanningArrivals, sample_planning_arrivals
from algorithm.background_competition_continuous_lane_rollout import (
    BackgroundCompetitionContinuousLaneRollout, BackgroundRootScores, _LazyBackgroundArrivals,
)
from algorithm.bg_incumbent_background_redecision_lane_rollout import (
    BGIncumbentBackgroundRedecisionDecision, BGIncumbentBackgroundRedecisionLaneRollout,
    INCUMBENT_SCORE_WORK, INCUMBENT_WORK,
)
from algorithm.one_step_background_redecision_lane_rollout import INNER_WORK, RECOURSE_WORK


CLASS_PROTECTION_WORK = (
    "class_squared_initializations", "class_squared_accumulations",
    "class_squared_score_reads", "class_risk_comparisons",
)


@dataclass(frozen=True)
class ClassProtectedBackgroundRootScores(BackgroundRootScores):
    class_squared_costs: dict
    class_squared_initializations: int
    class_squared_accumulations: int


@dataclass(frozen=True)
class ClassProtectedBGIncumbentBackgroundRedecisionDecision(BGIncumbentBackgroundRedecisionDecision):
    aggregate_risk_feasible_candidates: int = 0
    class_risk_filtered_candidates: int = 0
    baseline_predicted_class_squared_flow_s2: tuple = ()
    chosen_predicted_class_squared_flow_s2: tuple = ()
    class_squared_initializations: int = 0
    class_squared_accumulations: int = 0
    class_squared_score_reads: int = 0
    class_risk_comparisons: int = 0


class ClassProtectedBGIncumbentBackgroundRedecisionLaneRollout(
        BGIncumbentBackgroundRedecisionLaneRollout):
    """Improve aggregate S1 without transferring squared-flow cost between classes.

    Ordinary BG supplies the incumbent and its original S2-versus-MW feasible
    set. The one-step scorer uses the same tagged visible and prefix jobs, then
    admits only roots whose aggregate and each class's S2 do not exceed that
    incumbent under the same one-step model. Later background jobs compete in
    FIFO without contributing cost. No added samples or successor commitments
    are needed. Ordinary-BG overflow returns MW; later overflow retains BG.
    """
    def maxweight(self, options, queues, available, now, lane, waiting_ages,
                  observed_arrival_counts=None):
        decision = super().maxweight(options, queues, available, now, lane, waiting_ages)
        return ClassProtectedBGIncumbentBackgroundRedecisionDecision(
            **{field.name: getattr(decision, field.name)
               for field in fields(BGIncumbentBackgroundRedecisionDecision)})

    def _choose(self, options, queues, available, now, lane, waiting_ages):
        self._recourse_work = Counter(recourse_root_count_reads=len(self._outer_observed_counts))
        incumbent_work = Counter()
        class_work = Counter()

        def finish(decision):
            return replace(decision,
                **{name: self._recourse_work[name] for name in RECOURSE_WORK},
                **{name: incumbent_work[name] for name in INCUMBENT_WORK},
                **{name: class_work[name] for name in CLASS_PROTECTION_WORK})

        baseline = self.maxweight(options, queues, available, now, lane, waiting_ages)
        if not 0 < sum(queues) <= baseline.queue_limit:
            return finish(baseline)
        maxweight_root, = baseline.selection.option_indices
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
            return finish(replace(baseline, **metadata))
        original = BackgroundCompetitionContinuousLaneRollout.score_background_roots(
            self, queues, available, now, lane, waiting_ages, paths)
        for name in INCUMBENT_SCORE_WORK:
            incumbent_work["incumbent_" + name] = getattr(original, name)
        baseline = replace(baseline,
            incumbent_candidate_count=original.candidate_count,
            incumbent_background_rollout_overflow=original.background_rollout_overflow)
        if original.background_rollout_overflow:
            metadata["planning_overflow"] = True
            return finish(replace(baseline, **metadata))

        original_permitted = frozenset(index for index in original.costs
            if original.squared_costs[index] <= original.squared_costs[maxweight_root])

        def original_ordering(index):
            return original.costs[index], index != maxweight_root, self.durations[index], index

        min(original.costs, key=original_ordering)
        incumbent = min(original_permitted, key=original_ordering)
        original_changed = original.costs[incumbent] < original.costs[maxweight_root]
        if not original_changed:
            incumbent = maxweight_root
        incumbent_work["incumbent_risk_comparisons"] = original.candidate_count
        incumbent_work["incumbent_ordering_comparisons"] = (
            original.candidate_count - 1 + len(original_permitted) - 1)
        incumbent_work["incumbent_strict_improvement_comparisons"] = 1
        baseline = replace(baseline,
            incumbent_root_index=incumbent, incumbent_changed=incumbent != maxweight_root,
            incumbent_risk_feasible_candidates=len(original_permitted),
            incumbent_risk_filtered_candidates=original.candidate_count - len(original_permitted),
            incumbent_mw_predicted_completion_s=float(original.costs[maxweight_root] / len(paths)),
            incumbent_predicted_completion_s=float(original.costs[incumbent] / len(paths)),
            incumbent_mw_predicted_squared_flow_s2=float(
                original.squared_costs[maxweight_root] / len(paths)),
            incumbent_predicted_squared_flow_s2=float(original.squared_costs[incumbent] / len(paths)),
            incumbent_actual_predicted_squared_flow_s2=float(
                original.squared_costs[incumbent] / len(paths)))

        def selection_for_root(root):
            selected = baseline.selection
            if root == maxweight_root:
                return selected
            support = Fraction(sum(q * d for q, d in zip(
                queues, self.requirements[root], strict=True))) / self.durations[root]
            best_support = Fraction(sum(q * d for q, d in zip(
                queues, selected.service, strict=True))) / self.durations[maxweight_root]
            score, score_exact = float(support), support == best_support
            upper = score if score_exact else selected.upper_bound
            if not score_exact and Fraction.from_float(upper) < best_support:
                upper = nextafter(upper, inf)
            return replace(selected,
                option_counts=tuple(int(i == root) for i in range(len(self.options))),
                option_indices=(root,), service=self.requirements[root],
                duration=float(self.durations[root]), score=score, upper_bound=upper,
                gap=max(0., upper - score), exact=False, score_exact=score_exact,
                canonical_exact=False)

        scored = self.score_background_roots(queues, available, now, lane, waiting_ages, paths)
        class_work["class_squared_initializations"] = scored.class_squared_initializations
        class_work["class_squared_accumulations"] = scored.class_squared_accumulations
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
            return finish(replace(baseline,
                selection=selection_for_root(incumbent), changed=incumbent != maxweight_root,
                **work, **metadata))

        permitted = []
        aggregate_permitted = 0
        for index in scored.costs:
            incumbent_work["incumbent_membership_checks"] += 1
            if index not in original_permitted:
                continue
            incumbent_work["incumbent_lookahead_risk_comparisons"] += 1
            if scored.squared_costs[index] > scored.squared_costs[incumbent]:
                continue
            aggregate_permitted += 1
            for candidate, control in zip(scored.class_squared_costs[index],
                                          scored.class_squared_costs[incumbent], strict=True):
                class_work["class_squared_score_reads"] += 2
                class_work["class_risk_comparisons"] += 1
                if candidate > control:
                    break
            else:
                permitted.append(index)

        def ordering(index):
            return scored.costs[index], index != incumbent, self.durations[index], index

        unconstrained = min(scored.costs, key=ordering)
        chosen = min(permitted, key=ordering)
        improved = scored.costs[chosen] < scored.costs[incumbent]
        incumbent_work["incumbent_lookahead_ordering_comparisons"] = (
            scored.candidate_count - 1 + len(permitted) - 1)
        incumbent_work["incumbent_lookahead_strict_improvement_comparisons"] = 1
        actual = chosen if improved else incumbent
        class_work["class_squared_score_reads"] += 2 * len(queues)
        return finish(replace(baseline,
            selection=selection_for_root(actual), changed=actual != maxweight_root,
            improved_incumbent=improved,
            aggregate_risk_feasible_candidates=aggregate_permitted,
            class_risk_filtered_candidates=aggregate_permitted - len(permitted),
            baseline_predicted_class_squared_flow_s2=tuple(float(value / len(paths))
                for value in scored.class_squared_costs[incumbent]),
            chosen_predicted_class_squared_flow_s2=tuple(float(value / len(paths))
                for value in scored.class_squared_costs[actual]),
            incumbent_actual_predicted_squared_flow_s2=float(original.squared_costs[actual] / len(paths)),
            baseline_predicted_completion_s=float(scored.costs[incumbent] / len(paths)),
            chosen_predicted_completion_s=float(scored.costs[actual] / len(paths)),
            baseline_predicted_squared_flow_s2=float(scored.squared_costs[incumbent] / len(paths)),
            chosen_predicted_squared_flow_s2=float(scored.squared_costs[actual] / len(paths)),
            risk_filtered_candidates=scored.candidate_count - len(permitted),
            risk_feasible_candidates=len(permitted), unconstrained_root_index=unconstrained,
            risk_changed_unconstrained_root=actual != unconstrained, **work, **metadata))

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
        work = self._recourse_work
        post_prefix_state = self.rng.getstate()
        outer_counter = self.decision_counter
        work["recourse_rng_words_copied"] += len(post_prefix_state[1])
        work["recourse_counter_reads"] += 1
        costs = dict.fromkeys(roots, Fraction(0))
        squared = dict.fromkeys(roots, Fraction(0))
        class_squared = {root: [Fraction(0)] * len(queue) for root in roots}
        calls = idle_calls = evaluations = admitted_reads = tag_reads = background_admissions = 0

        def scores(overflow=False):
            return ClassProtectedBackgroundRootScores(
                costs, squared, len(roots), calls, calls * len(self.options),
                len(self.options), idle_calls, len(prepared_paths), sum(queue), evaluations,
                sum(len(suffix.events) for suffix in suffixes),
                sum(suffix.random_draws for suffix in suffixes),
                sum(suffix.rate_reads for suffix in suffixes),
                sum(suffix.seed_initializations for suffix in suffixes),
                admitted_reads, tag_reads, background_admissions, overflow,
                {root: tuple(values) for root, values in class_squared.items()},
                len(roots) * len(queue), evaluations)

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
                        squared_flow = flow * flow
                        second += squared_flow
                        class_squared[root][coordinate] += squared_flow
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
                cursor = background_cursor = prior_idle_calls = 0
                recursed = False
                while tagged_remaining:
                    next_lane = min(range(len(lane_ends)), key=lambda i: (lane_ends[i], i))
                    start = lane_ends[next_lane]
                    while cursor < len(events) and events[cursor][0] <= start:
                        birth, coordinate = events[cursor]
                        waiting[coordinate].append((birth, True))
                        if not recursed:
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
                        if not recursed:
                            counts[coordinate] += 1
                            work["recourse_arrival_counter_updates"] += 1
                        background_cursor += 1
                    current_queue = tuple(len(group) for group in waiting)
                    chosen, _ = self._best_index(current_queue)
                    calls += 1
                    if chosen is None:
                        idle_calls += 1
                        if not recursed:
                            prior_idle_calls += 1
                        lane_ends[next_lane] = start + 1
                        continue
                    if not recursed:
                        self.inner_planner.rng.setstate(post_prefix_state)
                        self.inner_planner.decision_counter = outer_counter + 1 + prior_idle_calls
                        work["recourse_rng_words_restored"] += len(post_prefix_state[1])
                        work["recourse_counter_writes"] += 1
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
                        recursed = True
                    additional_first, additional_second = admit(chosen, start, waiting)
                    first += additional_first
                    second += additional_second
                    lane_ends[next_lane] = start + self.durations[chosen]
                costs[root] += first
                squared[root] += second
        return scores()
