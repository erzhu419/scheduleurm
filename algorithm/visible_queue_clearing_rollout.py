"""Bounded visible-queue rollout on complete, asynchronous lane renewals.

Only the currently waiting jobs are cleared; no future arrivals are read or
predicted. Busy jobs have fixed completion costs and are excluded. The queue
size envelope comes from the catalogue's maximal batch width and lane count.
"""
from dataclasses import dataclass, replace
from fractions import Fraction
from math import inf, nextafter

from algorithm.lane_refill_simple_controls import (
    SingleLaneSelection, select_single_lane_control,
)


@dataclass(frozen=True)
class VisibleQueueDecision:
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


def _baseline(options, queues, available, now, lane):
    if not 0 <= lane < len(available) or available[lane] > now:
        raise ValueError("the dispatched lane must already be available")
    selected = select_single_lane_control(options, queues, 1, 0)
    limit = len(available) * max(sum(option.requirements) for option in options)
    return VisibleQueueDecision(selected, False, 0, 0, 0, len(options),
                                None, None, False, limit)


def select_context_maxweight(options, queues, available, now, lane):
    """Plain exact single-lane MaxWeight with the same context interface."""
    return _baseline(tuple(options), tuple(queues), available, now, lane)


def select_visible_queue_clearing(options, queues, available, now, lane):
    """Choose a root only when its exact clearing cost improves on MaxWeight.

    For each feasible nonidle root, the continuation uses ordinary MaxWeight
    at the earliest lane availability, breaking lane ties by original ID.
    Completion offsets count each admitted job; complete option durations,
    with the selector's one-second floor, reserve the lane. One continuation
    call serves at least one job on the registered singleton-complete models.
    """
    options, queue = tuple(options), tuple(queues)
    baseline = _baseline(options, queue, available, now, lane)
    population = sum(queue)
    if not 0 < population <= baseline.queue_limit:
        return baseline

    # One shared, paid full-catalogue preprocessing pass; the subsequent root
    # feasibility enumeration is also a full pass. Continuations retain the
    # original indices so their ordinary MaxWeight tie rule stays unchanged.
    prepared = tuple((Fraction.from_float(max(1., float(option.duration))),
                      sum((Fraction.from_float(float(offset))
                           for offsets in option.completion_offsets
                           for offset in offsets), Fraction(0))) for option in options)
    durations, offset_sums = tuple(zip(*prepared, strict=True))
    root_indices = tuple(index for index, option in enumerate(options)
                         if all(need <= count for need, count in
                                zip(option.requirements, queue, strict=True)))
    relative = tuple(max(Fraction(0), Fraction.from_float(float(end))
                         - Fraction.from_float(float(now))) for end in available)
    rollout_calls = rollout_probes = 0

    def clearing_cost(root):
        nonlocal rollout_calls, rollout_probes
        remaining = tuple(count - need for count, need in
                          zip(queue, options[root].requirements, strict=True))
        lane_ends = list(relative)
        lane_ends[lane] = durations[root]
        cost = offset_sums[root]
        while any(remaining):
            next_lane = min(range(len(lane_ends)), key=lambda i: (lane_ends[i], i))
            start = lane_ends[next_lane]
            chosen = select_single_lane_control(options, remaining, 1, 0)
            rollout_calls += 1
            rollout_probes += chosen.option_probes
            index, = chosen.option_indices
            if index is None:
                raise ValueError("visible waiting class has no feasible singleton service")
            cost += sum(chosen.service) * start + offset_sums[index]
            remaining = tuple(count - need for count, need in
                              zip(remaining, chosen.service, strict=True))
            lane_ends[next_lane] = start + durations[index]
        return cost

    costs = {index: clearing_cost(index) for index in root_indices}
    incumbent, = baseline.selection.option_indices
    if incumbent is None:
        raise ValueError("nonempty registered queue needs feasible singleton service")
    chosen_index = min(root_indices, key=lambda index: (
        costs[index], index != incumbent, durations[index], index))
    changed = costs[chosen_index] < costs[incumbent]
    selected = baseline.selection
    if changed:
        option = options[chosen_index]
        score_exact_value = Fraction(sum(q * d for q, d in
                                    zip(queue, option.requirements, strict=True))) / durations[chosen_index]
        incumbent_value = Fraction(sum(q * d for q, d in
                                  zip(queue, selected.service, strict=True))) / durations[incumbent]
        score = float(score_exact_value)
        score_exact = score_exact_value == incumbent_value
        upper = score if score_exact else selected.upper_bound
        if not score_exact and Fraction.from_float(upper) < incumbent_value:
            upper = nextafter(upper, inf)
        selected = replace(selected,
            option_counts=tuple(int(i == chosen_index) for i in range(len(options))),
            option_indices=(chosen_index,), service=tuple(option.requirements),
            duration=float(durations[chosen_index]), score=score, upper_bound=upper,
            gap=max(0., upper - score), exact=False, score_exact=score_exact,
            canonical_exact=False)
    return VisibleQueueDecision(selected, True, len(root_indices), rollout_calls,
        rollout_probes, 3 * len(options), float(costs[incumbent]),
        float(costs[chosen_index] if changed else costs[incumbent]), changed,
        baseline.queue_limit)
