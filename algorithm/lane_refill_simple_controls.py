"""Plain single-lane scheduling controls on the complete-job catalogue.

The catalogue is scanned once per call. The shortest-mean-completion rule is a
batch SJF heuristic, not a mean-flow oracle; all loaded resident offsets count.
Support certificates always refer to the full one-lane catalogue, including
when a heuristic or fixed p1 policy chooses a lower-support action.
"""
from dataclasses import dataclass
from fractions import Fraction


POLICIES = ("enum_maxweight", "delay", "fixed_p1")


@dataclass(frozen=True)
class SingleLaneSelection:
    option_counts: tuple
    option_indices: tuple
    service: tuple
    duration: float
    score: float
    upper_bound: float
    gap: float
    exact: bool
    score_exact: bool
    canonical_exact: bool
    lp_node_budget: int
    candidate_evaluations: int
    option_probes: int
    lp_calls: int = 0
    lp_matrix_coefficients: int = 0
    duration_root_count: int = 0
    dual_column_checks: int = 0
    branch_splits: int = 0
    bound_pruned_regions: int = 0
    safe_fallback_lp_calls: int = 0
    residual_fill_option_probes: int = 0


def select_single_lane_control(options, queues, lanes, budget, *, policy="enum_maxweight"):
    """Choose from current feasible options with no future arrivals or LP calls."""
    if lanes != 1 or policy not in POLICIES:
        raise ValueError("simple controls require one lane and a registered policy")
    options, queue = tuple(options), tuple(queues)
    # (index, service, effective duration, exact support, mean job offset)
    candidates = [(None, (0,) * len(queue), 1.0, Fraction(0), Fraction(0))]
    for index, option in enumerate(options):
        if any(need > count for need, count in zip(option.requirements, queue, strict=True)):
            continue
        duration = max(1.0, float(option.duration))
        support = Fraction(sum(q * d for q, d in zip(queue, option.requirements, strict=True))) / Fraction.from_float(duration)
        offsets = tuple(value for group in option.completion_offsets for value in group)
        mean_offset = sum((Fraction.from_float(float(value)) for value in offsets), Fraction(0)) / len(offsets)
        candidates.append((index, tuple(option.requirements), duration, support, mean_offset))

    def canonical(row):
        return row[2], len(options) if row[0] is None else row[0]

    best = min(candidates, key=lambda row: (-row[3], *canonical(row)))
    chosen = best
    feasible = candidates[1:]
    if policy == "delay" and feasible:
        chosen = min(feasible, key=lambda row: (row[4], -row[3], *canonical(row)))
    elif policy == "fixed_p1":
        fixed = [row for row in feasible if not options[row[0]].loaded and sum(row[1]) == 1]
        if feasible and not fixed:
            raise ValueError("fixed p1 control needs a dedicated singleton for each arrival class")
        chosen = min(fixed, key=lambda row: (-row[3], *canonical(row))) if fixed else candidates[0]
    index, service, duration, support, _ = chosen
    counts = tuple(int(i == index) for i in range(len(options)))
    score, upper = float(support), float(best[3])
    # A rounded-down inexact upper bound is not a valid certificate.
    if support != best[3] and Fraction.from_float(upper) < best[3]:
        from math import inf, nextafter
        upper = nextafter(upper, inf)
    score_exact = support == best[3]
    canonical_exact = chosen[0] == best[0]
    return SingleLaneSelection(counts, (index,), service, duration, score, upper,
                               max(0.0, upper - score), canonical_exact,
                               score_exact, canonical_exact, budget, len(candidates), len(options))
