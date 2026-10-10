"""Independent original-count canonical oracle conditional on a proven score.

This audit uses exact arithmetic and bounded depth-first enumeration, with no LP
or selector pruning code.  Duration thresholds are examined in increasing order;
counts of each original option are examined in decreasing order.  Thus the first
solution has the minimum duration and sorted option-index tuple (idle last).
"""
from __future__ import annotations

from fractions import Fraction
import math


def _inputs(options, queues, lanes, idle_duration):
    queue = tuple(queues)
    if lanes < 1 or int(lanes) != lanes or any(q < 0 or int(q) != q for q in queue):
        raise ValueError("positive integer lanes and nonnegative integer queues required")
    if not math.isfinite(idle_duration) or idle_duration <= 0:
        raise ValueError("positive finite idle duration required")
    idle = Fraction.from_float(float(idle_duration))
    requirements, durations = [], []
    for option in options:
        required = tuple(option.requirements)
        if (len(required) != len(queue) or not any(required)
                or any(r < 0 or int(r) != r for r in required)):
            raise ValueError("nonzero integer option requirements must match the queue")
        duration = float(option.duration)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("positive finite option durations required")
        requirements.append(tuple(int(r) for r in required))
        durations.append(max(idle, Fraction.from_float(duration)))
    return tuple(int(q) for q in queue), int(lanes), idle, requirements, durations


def certified_score_from_selection(options, queues, lanes, selection, *, idle_duration=1.0):
    """Reconstruct the exact feasible score; the supplied score proof is external.

    ``score_exact`` is the existing search's score certificate, not a new proof
    of optimality by this helper.  Its counts, service and score must agree.
    """
    queue, lanes, idle, requirements, durations = _inputs(options, queues, lanes, idle_duration)
    if not selection["score_exact"]:
        raise ValueError("an external completed score certificate is required")
    counts = tuple(selection["option_counts"])
    if (len(counts) != len(options) or any(c < 0 or int(c) != c for c in counts)
            or sum(counts) > lanes):
        raise ValueError("score-certified incumbent has invalid counts")
    service = tuple(sum(c * r[j] for c, r in zip(counts, requirements)) for j in range(len(queue)))
    duration = max((d for c, d in zip(counts, durations) if c), default=idle)
    score = Fraction(sum(q * s for q, s in zip(queue, service)), 1) / duration
    indices = tuple(i for i, c in enumerate(counts) for _ in range(c)) + (None,) * (lanes - sum(counts))
    if (any(s > q for s, q in zip(service, queue)) or tuple(selection["service"]) != service
            or tuple(selection["option_indices"]) != indices
            or selection["duration"] != float(duration) or selection["score"] != float(score)):
        raise ValueError("score-certified incumbent disagrees with its feasible action")
    return score


def canonical_at_certified_score(options, queues, lanes, proven_score, *,
                                 state_visit_budget, maximum_duration, idle_duration=1.0):
    """Return an exact conditional canonical proof, or a preserved budget failure.

    ``proven_score`` must be an exact Fraction reconstructed from a feasible,
    externally score-certified incumbent.  ``maximum_duration`` is that
    incumbent's duration.  A completed infeasibility result exposes inconsistent
    inputs; it does not assert canonical optimality.  The state budget counts
    newly visited DFS states, including those rejected by exact bounds.
    """
    queue, lanes, idle, requirements, durations = _inputs(options, queues, lanes, idle_duration)
    if not isinstance(proven_score, Fraction) or proven_score < 0:
        raise ValueError("nonnegative exact Fraction score required")
    if int(state_visit_budget) != state_visit_budget or state_visit_budget < 0:
        raise ValueError("nonnegative integer state visit budget required")
    if not math.isfinite(maximum_duration) or maximum_duration < float(idle):
        raise ValueError("finite incumbent duration at least idle duration required")
    maximum = Fraction.from_float(float(maximum_duration))
    weights = [sum(q * r for q, r in zip(queue, required)) for required in requirements]
    work = dict(states_visited=0, memo_hits=0, count_branches=0, bound_prunes=0,
                gcd_prunes=0, duration_prunes=0, failed_states_cached=0,
                thresholds_checked=0, noninteger_thresholds=0)
    base = {"conditional_on_external_score_certificate": True,
            "state_visit_budget": int(state_visit_budget),
            "score_fraction": [proven_score.numerator, proven_score.denominator], "work": work}

    def finish(counts, duration):
        indices = tuple(i for i, c in enumerate(counts) for _ in range(c)) + (None,) * (lanes - sum(counts))
        service = tuple(sum(c * r[j] for c, r in zip(counts, requirements)) for j in range(len(queue)))
        return {**base, "complete": True, "canonical_exact": True, "status": "COMPLETE",
                "option_counts": list(counts), "option_indices": list(indices),
                "service": list(service), "duration": float(duration), "score": float(proven_score),
                "duration_fraction": [duration.numerator, duration.denominator],
                "canonical_key": [float(duration), [len(options) if i is None else i for i in indices]]}

    if not proven_score:
        return finish((0,) * len(options), idle)

    class BudgetExhausted(Exception):
        pass

    thresholds = sorted({d for d in durations if d <= maximum})
    for threshold in thresholds:
        work["thresholds_checked"] += 1
        target = proven_score * threshold
        if target.denominator != 1:
            work["noninteger_thresholds"] += 1
            continue
        eligible = tuple(i for i, d in enumerate(durations) if d <= threshold)
        size = len(eligible)
        suffix_gcd = [0] * (size + 1)
        suffix_weight = [0] * (size + 1)
        suffix_requirements = [(0,) * len(queue) for _ in range(size + 1)]
        suffix_threshold = [False] * (size + 1)
        for pos in range(size - 1, -1, -1):
            index = eligible[pos]
            suffix_gcd[pos] = math.gcd(weights[index], suffix_gcd[pos + 1])
            suffix_weight[pos] = max(weights[index], suffix_weight[pos + 1])
            suffix_requirements[pos] = tuple(max(r, later) for r, later in
                                             zip(requirements[index], suffix_requirements[pos + 1]))
            suffix_threshold[pos] = durations[index] == threshold or suffix_threshold[pos + 1]
        failed = set()

        def search(pos, remaining_lanes, capacity, remaining_target, attained):
            key = (pos, remaining_lanes, capacity, remaining_target, attained)
            if key in failed:
                work["memo_hits"] += 1
                return None
            if work["states_visited"] >= state_visit_budget:
                raise BudgetExhausted
            work["states_visited"] += 1

            def reject(counter):
                work[counter] += 1
                failed.add(key)
                work["failed_states_cached"] += 1
                return None

            if not remaining_target:
                return (0,) * (size - pos) if attained else reject("duration_prunes")
            if not remaining_lanes or pos == size:
                return reject("bound_prunes")
            if not attained and not suffix_threshold[pos]:
                return reject("duration_prunes")
            resource_bound = sum(q * min(c, remaining_lanes * r) for q, c, r in
                                 zip(queue, capacity, suffix_requirements[pos]))
            if remaining_target > min(resource_bound, remaining_lanes * suffix_weight[pos]):
                return reject("bound_prunes")
            if not suffix_gcd[pos] or remaining_target % suffix_gcd[pos]:
                return reject("gcd_prunes")
            index = eligible[pos]
            required = requirements[index]
            limit = min(remaining_lanes, *(c // r for c, r in zip(capacity, required) if r))
            if weights[index]:
                limit = min(limit, remaining_target // weights[index])
            for count in range(limit, -1, -1):
                work["count_branches"] += 1
                tail = search(pos + 1, remaining_lanes - count,
                              tuple(c - count * r for c, r in zip(capacity, required)),
                              remaining_target - count * weights[index],
                              attained or (count > 0 and durations[index] == threshold))
                if tail is not None:
                    return (count,) + tail
            failed.add(key)
            work["failed_states_cached"] += 1
            return None

        try:
            local = search(0, lanes, queue, target.numerator, False)
        except BudgetExhausted:
            return {**base, "complete": False, "canonical_exact": False, "status": "STATE_BUDGET_EXHAUSTED"}
        if local is not None:
            counts = [0] * len(options)
            for index, count in zip(eligible, local):
                counts[index] = count
            return finish(counts, threshold)
    return {**base, "complete": True, "canonical_exact": False, "status": "NO_ACTION_AT_CERTIFIED_SCORE"}
