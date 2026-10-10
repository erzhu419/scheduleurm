"""Pair an existing improved root with MW using one causal candidate redecision."""
from collections import Counter, deque
from dataclasses import dataclass, field, fields, replace
from fractions import Fraction
from math import ceil
from random import Random

from algorithm.age_tail_constrained_lane_rollout import AgeRootScores
from algorithm.continuous_arrival_age_tail_lane_rollout import ContinuousArrivalAgeTailLaneRollout
from algorithm.observed_rate_age_tail_lane_rollout import ObservedRateDecision


@dataclass(frozen=True)
class RedecisionDecision(ObservedRateDecision):
    redecision_attempted: bool = False
    redecision_rejected: bool = False
    proposal_option_index: int | None = None
    baseline_redecision_flow_s: float | None = None
    chosen_redecision_flow_s: float | None = None
    baseline_redecision_squared_flow_s2: float | None = None
    chosen_redecision_squared_flow_s2: float | None = None
    redecision_work: dict = field(default_factory=dict)
    search_option_probe_limit: int = 0
    search_flow_evaluation_limit: int = 0


class RedecisionContinuousLaneRollout(ContinuousArrivalAgeTailLaneRollout):
    """Filter original strict improvements by a bounded two-root comparison.

    A branch predicts exactly one subsequent nonidle decision of the frozen
    original policy, using fresh independent samples and causally updated
    counts. Remaining outer jobs then drain with MW. This is a finite-depth
    continuation approximation, not an exact value of the new closed loop.
    """
    def __init__(self, options, planning_seed, sample_count=8, event_cap=128):
        super().__init__(options, planning_seed, sample_count, event_cap)
        self.continuation = ContinuousArrivalAgeTailLaneRollout(
            self.options, planning_seed, sample_count, event_cap)
        self.preprocessing_work.update({"continuation_" + key: value
                                        for key, value in self.continuation.preprocessing_work.items()})
        self.planning_seed = planning_seed
        self.decision_counter = 0
        self._redecision_work = Counter()

    def score_age_roots(self, *args):
        self._saved_paths = tuple(args[-1])
        return super().score_age_roots(*args)

    def budget_limits(self, queues, lanes):
        m, s = len(self.options), self.sample_count
        population = lanes * self.batch_width + self.event_cap
        calls = population + lanes * (ceil(self.maximum_duration) + 1)
        # Initial all-root score, two paired outer roots, <=2*s inner calls.
        option_limit = (4 * m + 2 + s * m * m * calls
                        + 2 * s * (m * calls + 2 * m + s * m * m * calls))
        flow_limit = s * m * population + 2 * s * (population + s * m * population)
        return option_limit, flow_limit

    def score_redecision_roots(self, queues, available, now, lane, waiting_ages,
                              observed_counts, paths, roots):
        work = self._redecision_work
        work["pair_waiting_age_reads"] += sum(queues)
        work["pair_root_option_probes"] += len(roots)
        arrivals = tuple(tuple(-Fraction.from_float(float(age)) for age in group)
                         for group in waiting_ages)
        relative = tuple(max(Fraction(0), Fraction.from_float(float(end))
                             - Fraction.from_float(float(now))) for end in available)
        prepared = tuple(tuple((Fraction.from_float(float(event[0])), event[1])
                               for event in path.events) for path in paths)
        costs, squared = dict.fromkeys(roots, Fraction(0)), dict.fromkeys(roots, Fraction(0))

        def admit(index, start, waiting):
            first = second = Fraction(0)
            for coordinate, offsets in enumerate(self.sorted_completion_offsets[index]):
                for offset in offsets:
                    flow = start + offset - waiting[coordinate].popleft()
                    first += flow
                    second += flow * flow
                    work["pair_flow_evaluations"] += 1
            return first, second

        for root in roots:
            for sample_index, events in enumerate(prepared):
                waiting = [deque(group) for group in arrivals]
                ends, counts = list(relative), list(observed_counts)
                ends[lane] = self.durations[root]
                first, second = admit(root, Fraction(0), waiting)
                cursor, redecided = 0, False
                while cursor < len(events) or any(waiting):
                    next_lane = min(range(len(ends)), key=lambda index: (ends[index], index))
                    start = ends[next_lane]
                    while cursor < len(events) and events[cursor][0] <= start:
                        birth, coordinate = events[cursor]
                        waiting[coordinate].append(birth)
                        counts[coordinate] += 1
                        work["pair_event_releases"] += 1
                        cursor += 1
                    queue = tuple(map(len, waiting))
                    chosen, _ = self._best_index(queue)
                    work["pair_rollout_calls"] += 1
                    work["pair_rollout_option_probes"] += len(self.options)
                    if chosen is None:
                        work["pair_idle_calls"] += 1
                        ends[next_lane] = start + 1
                        continue
                    if not redecided:
                        # Common independent stream across the two outer roots.
                        # The online planner's RNG never advances here.
                        self.continuation.rng = Random(
                            f"{self.planning_seed}:next:{self.decision_counter}:{sample_index}")
                        work["continuation_seed_initializations"] += 1
                        absolute_now = float(Fraction.from_float(float(now)) + start)
                        absolute_ends = tuple(float(Fraction.from_float(float(now)) + end) for end in ends)
                        ages = tuple(tuple(float(start - birth) for birth in group) for group in waiting)
                        inner = self.continuation(self.options, queue, absolute_ends, absolute_now,
                                                  next_lane, ages, tuple(counts))
                        work["continuation_context_age_reads"] += sum(queue)
                        work["continuation_context_coordinate_reads"] += len(counts)
                        work["continuation_calls"] += 1
                        work["continuation_eligible_decisions"] += inner.eligible
                        work["continuation_changed_decisions"] += inner.changed
                        work["continuation_planning_overflows"] += inner.planning_overflow
                        for key in ("root_option_probes", "rollout_calls", "rollout_option_probes",
                                    "planning_events", "planning_random_draws", "planning_rate_reads",
                                    "rollout_idle_calls", "waiting_age_reads", "flow_evaluations",
                                    "rate_estimation_coordinate_reads", "rate_estimation_divisions",
                                    "risk_filtered_candidates", "risk_feasible_candidates"):
                            work["continuation_" + key] += getattr(inner, key)
                        chosen, = inner.selection.option_indices
                        redecided = True
                    additional_first, additional_second = admit(chosen, start, waiting)
                    first += additional_first
                    second += additional_second
                    ends[next_lane] = start + self.durations[chosen]
                costs[root] += first
                squared[root] += second
        return AgeRootScores(costs, squared, len(roots), work["pair_rollout_calls"],
            work["pair_rollout_option_probes"], len(roots), work["pair_idle_calls"],
            len(paths), sum(queues), work["pair_flow_evaluations"])

    def __call__(self, options, queues, available, now, lane, waiting_ages, observed_arrival_counts):
        self.decision_counter += 1
        self._redecision_work = Counter()
        proposal = super().__call__(options, queues, available, now, lane, waiting_ages,
                                    observed_arrival_counts)
        decision = RedecisionDecision(**{item.name: getattr(proposal, item.name)
                                         for item in fields(ObservedRateDecision)})
        option_limit, flow_limit = self.budget_limits(queues, len(available))
        decision = replace(decision, search_option_probe_limit=option_limit,
                           search_flow_evaluation_limit=flow_limit)
        if not proposal.changed:
            return decision
        candidate, = proposal.selection.option_indices
        incumbent, _ = self._best_index(tuple(queues))
        self._redecision_work["pair_control_option_probes"] += len(self.options)
        scored = self.score_redecision_roots(queues, available, now, lane, waiting_ages,
                                             observed_arrival_counts, self._saved_paths,
                                             (incumbent, candidate))
        self._redecision_work["pair_risk_comparisons"] += 2
        accepted = (scored.costs[candidate] < scored.costs[incumbent]
                    and scored.squared_costs[candidate] <= scored.squared_costs[incumbent])
        actual = candidate if accepted else incumbent
        if not accepted:
            selected = self._selection(tuple(queues))
            self._redecision_work["restore_control_option_probes"] += len(self.options)
            decision = replace(decision, selection=selected, changed=False,
                chosen_predicted_completion_s=decision.baseline_predicted_completion_s,
                chosen_predicted_squared_flow_s2=decision.baseline_predicted_squared_flow_s2,
                risk_changed_unconstrained_root=incumbent != decision.unconstrained_root_index)
        return replace(decision, redecision_attempted=True, redecision_rejected=not accepted,
            proposal_option_index=candidate,
            baseline_redecision_flow_s=float(scored.costs[incumbent] / len(self._saved_paths)),
            chosen_redecision_flow_s=float(scored.costs[actual] / len(self._saved_paths)),
            baseline_redecision_squared_flow_s2=float(scored.squared_costs[incumbent] / len(self._saved_paths)),
            chosen_redecision_squared_flow_s2=float(scored.squared_costs[actual] / len(self._saved_paths)),
            redecision_work=dict(self._redecision_work))
