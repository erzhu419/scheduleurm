"""Full-catalogue canonical search using threshold-specific requirement groups.

For group g let m_g be its shortest effective duration. At threshold T the
representative is the earliest original index with duration <= T. A proxy
catalogue orders these representatives by original index and assigns every
proxy duration T. Thus the existing B&B's candidate tie keys AND canonical
box bounds use the required threshold order, with only one variable per group.

For any nonempty feasible group counts c, A = Q.service is positive and their
best duration is D = max(m_g: c_g > 0). The representatives at D attain this
duration and the smallest original-index tuple. Consequently one of the
group-minimum thresholds contains the full-catalogue canonical optimum. The
maximum of the proxy upper bounds certifies the original catalogue.

Each threshold receives a static share of the total LP budget and runs once.
No unused budget is transferred and no common incumbent is shared. All search
work counters are summed over these independent subproblems, except peak
frontier size and maximum dual repair, which take maxima. Represented-space
and remaining-region counters describe proxy subproblems, whose domains can
overlap; they are not counts of distinct full-catalogue actions. Full-width
output counts and indices always address the caller's original options.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from fractions import Fraction
from typing import Any, Sequence

from algorithm.canonical_count_space_bnb_frame_candidates import (
    CanonicalCountSpaceBnbSelection, _float_up, select_canonical_count_space_bnb_frame,
)
from algorithm.requirement_dominance_frame_candidates import reduce_requirement_dominated_options


@dataclass(frozen=True)
class ThresholdCatalogueWork:
    """Cold-call catalogue construction and returned-candidate lifting work."""

    original_option_count: int
    requirement_group_count: int
    duration_threshold_count: int
    reduction_option_probes: int
    reduction_requirement_coordinate_reads: int
    reduction_duration_comparisons: int
    grouping_option_probes: int
    representative_option_probes: int
    proxy_option_count: int
    lift_option_probes: int


@dataclass(frozen=True)
class ThresholdCanonicalRequirementSelection:
    selection: CanonicalCountSpaceBnbSelection
    preprocessing: ThresholdCatalogueWork
    thresholds: tuple[float, ...]
    threshold_budgets: tuple[int, ...]


@dataclass(frozen=True)
class _ProxyOption:
    requirements: tuple[int, ...]
    duration: float


def select_threshold_canonical_requirement_dominance_frame(
    options: Sequence[Any],
    queues: Sequence[int],
    lanes: int,
    lp_node_budget: int,
    idle_duration: float = 1.0,
) -> ThresholdCanonicalRequirementSelection:
    """Return a feasible original action and conservative full certificates.

    Budget remainders go to earlier thresholds. At least one LP call per
    threshold is required. Global canonical exactness requires every proxy
    subproblem to close its canonical search, even if a sharper joint test
    could discard some unresolved subproblems.
    """
    options, queue = tuple(options), tuple(queues)
    reduction = reduce_requirement_dominated_options(options, idle_duration)
    if int(lp_node_budget) != lp_node_budget or lp_node_budget < 0:
        raise ValueError("lp_node_budget must be a nonnegative integer")
    budget = int(lp_node_budget)
    if int(lanes) != lanes or lanes < 1:
        raise ValueError("lanes must be a positive integer")
    lanes = int(lanes)
    if any(int(value) != value or value < 0 for value in queue):
        raise ValueError("queues must contain nonnegative integer job counts")
    queue = tuple(int(value) for value in queue)
    if options and len(options[0].requirements) != len(queue):
        raise ValueError("option requirements must match queue coordinates")
    idle = Fraction.from_float(float(idle_duration))
    durations = tuple(Fraction.from_float(max(float(idle_duration), float(option.duration)))
                      for option in options)
    grouped: dict[tuple[int, ...], list[int]] = {}
    for index, option in enumerate(options):
        grouped.setdefault(tuple(int(value) for value in option.requirements), []).append(index)
    groups = tuple((requirement, tuple(members), min(durations[i] for i in members))
                   for requirement, members in grouped.items())
    thresholds = tuple(sorted({minimum for _, _, minimum in groups}))
    if budget < len(thresholds):
        raise ValueError(f"LP budget must cover {len(thresholds)} duration thresholds")
    representative_probes = proxy_count = lift_probes = 0

    def representative(members, threshold, *, lifting=False):
        nonlocal representative_probes, lift_probes
        for index in members:
            if lifting:
                lift_probes += 1
            else:
                representative_probes += 1
            if durations[index] <= threshold:
                return index
        raise AssertionError("eligible requirement group has no representative")

    def work():
        return ThresholdCatalogueWork(
            len(options), len(groups), len(thresholds),
            reduction.reduction_option_probes,
            reduction.reduction_requirement_coordinate_reads,
            reduction.reduction_duration_comparisons,
            len(options), representative_probes, proxy_count, lift_probes,
        )

    if not thresholds:
        selected = select_canonical_count_space_bnb_frame(options, queue, lanes, budget, idle_duration)
        return ThresholdCanonicalRequirementSelection(selected, work(), (), ())
    share, remainder = divmod(budget, len(thresholds))
    allocations = tuple(share + int(i < remainder) for i in range(len(thresholds)))
    children = []
    upper_bounds = []
    best_key = None
    best_payload = None
    for threshold, allocation in zip(thresholds, allocations, strict=True):
        ordered = sorted(
            (representative(members, threshold), group_index)
            for group_index, (_, members, minimum) in enumerate(groups)
            if minimum <= threshold
        )
        proxies = tuple(_ProxyOption(groups[group_index][0], float(threshold))
                        for _, group_index in ordered)
        proxy_count += len(proxies)
        child = select_canonical_count_space_bnb_frame(proxies, queue, lanes, allocation, idle_duration)
        children.append(child)
        numerator = sum(waiting * done for waiting, done in zip(queue, child.service, strict=True))
        upper_bounds.append(Fraction(numerator) / threshold if child.score_exact
                            else Fraction.from_float(child.upper_bound))

        # A residual candidate can use only groups shorter than this root.
        # Re-lift its counts at their own minimum duration, not the root T.
        actual_duration = max((idle, *(groups[group_index][2]
            for count, (_, group_index) in zip(child.option_counts, ordered, strict=True) if count)))
        full_counts = [0] * len(options)
        for count, (_, group_index) in zip(child.option_counts, ordered, strict=True):
            if count:
                index = representative(groups[group_index][1], actual_duration, lifting=True)
                full_counts[index] = count
        expanded = tuple(index for index, count in enumerate(full_counts) for _ in range(count))
        indices = expanded + (None,) * (lanes - len(expanded))
        score = Fraction(numerator) / actual_duration
        key = (-score, actual_duration, expanded + (len(options),) * (lanes - len(expanded)))
        if best_key is None or key < best_key:
            best_key = key
            best_payload = (tuple(full_counts), indices, child.service)

    score_exact_value = -best_key[0]
    upper_exact = max(upper_bounds)
    if upper_exact < score_exact_value:
        raise ArithmeticError("proxy certificates do not cover the lifted candidate")
    score_exact = upper_exact == score_exact_value
    canonical_exact = score_exact and all(child.canonical_exact for child in children)
    score = float(score_exact_value)
    upper = score if score_exact else _float_up(upper_exact)
    gap = max(0.0, upper - score)
    payload = {
        "option_counts": best_payload[0], "option_indices": best_payload[1],
        "service": best_payload[2], "duration": float(best_key[1]),
        "score": score, "upper_bound": upper, "gap": gap,
        "epsilon": gap / sum(queue) if sum(queue) else 0.0,
        "exact": canonical_exact, "score_exact": score_exact, "canonical_exact": canonical_exact,
    }
    for field in fields(CanonicalCountSpaceBnbSelection):
        if field.name not in payload:
            values = [getattr(child, field.name) for child in children]
            payload[field.name] = (max(values) if field.name in ("frontier_peak", "maximum_dual_repair")
                                   else sum(values))
    return ThresholdCanonicalRequirementSelection(
        CanonicalCountSpaceBnbSelection(**payload), work(),
        tuple(float(threshold) for threshold in thresholds), allocations,
    )
