"""Try canonical completion at the paid root score before branch LP work."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Sequence

from algorithm.canonical_count_space_bnb_frame_candidates import CanonicalCountSpaceBnbSelection
from algorithm.frontier_score_proof_certificate_routed_frame_candidates import PaidLpScoreFrontier, capture_strict_frontier
from algorithm.residual_capacity_frontier_frame_candidates import _residual_frontier_probe
from algorithm.resumable_probe_frame_candidates import _probe
from algorithm.strict_score_proof_canonical_completion_frame_candidates import StrictScoreProofCanonicalCompletionSelection
from algorithm.zero_delta_affine_dual_reuse_integer_capacity_frame_candidates import (
    ZeroDeltaAffineDualReuseIntegerCapacitySearch, ZeroDeltaAffineDualReuseIntegerCapacityThresholdSelection,
)


@dataclass(frozen=True)
class EarlyRootRoutingWork:
    option_visits: int
    requirement_coordinate_reads: int
    duration_reads: int
    duration_comparisons: int
    group_lookups: int
    group_writes: int
    threshold_value_reads: int
    threshold_set_insertions: int
    root_snapshot_calls: int
    root_snapshot_heap_entries_read: int
    root_snapshot_retained_entries_read: int


@dataclass(frozen=True)
class EarlyRootCanonicalSelection:
    selection: CanonicalCountSpaceBnbSelection
    root_lp: ZeroDeltaAffineDualReuseIntegerCapacityThresholdSelection
    root_lp_budget: int
    initial_lp: ZeroDeltaAffineDualReuseIntegerCapacityThresholdSelection
    final_lp: ZeroDeltaAffineDualReuseIntegerCapacityThresholdSelection
    probe: StrictScoreProofCanonicalCompletionSelection | None
    resumed: bool
    initial_lp_budget: int
    lp_node_budget: int
    integer_state_budget: int
    status: str
    probe_stage: str
    frontier_evidence: PaidLpScoreFrontier | None
    routing_preparation: EarlyRootRoutingWork


def _root_budget(options, idle_duration):
    """Read the physical duration strata once, charging this cold routing scan."""
    minima = {}
    coordinate_reads = duration_comparisons = 0
    for option in options:
        requirement = tuple(int(value) for value in option.requirements)
        coordinate_reads += len(requirement)
        duration = max(float(idle_duration), float(option.duration))
        duration_comparisons += 1
        previous = minima.get(requirement)
        if previous is not None:
            duration = min(previous, duration)
            duration_comparisons += 1
        minima[requirement] = duration
    groups = len(minima)
    root_budget = len(set(minima.values()))
    return root_budget, EarlyRootRoutingWork(
        len(options), coordinate_reads, len(options), duration_comparisons,
        len(options), len(options), groups, groups, 0, 0, 0)


def select_early_root_canonical_frame(
    options: Sequence[Any], queues: Sequence[int], lanes: int,
    initial_lp_budget: int, lp_node_budget: int, integer_state_budget: int,
    idle_duration: float = 1.0,
) -> EarlyRootCanonicalSelection:
    """Keep one LP frontier and at most one shared-budget integer probe.

    A certified root score enters the existing canonical helper directly. An
    exhausted root completion retains all its work and leaves no integer
    budget for another probe; only the same LP search continues. Open root
    scores retain the original initial/resumed/residual routing rule.
    """
    for value, name in ((initial_lp_budget, "initial_lp_budget"),
                        (lp_node_budget, "lp_node_budget"),
                        (integer_state_budget, "integer_state_budget")):
        if int(value) != value or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    initial_budget, total_budget, probe_budget = map(int, (initial_lp_budget, lp_node_budget, integer_state_budget))
    if total_budget < initial_budget:
        raise ValueError("total LP budget must cover the initial LP budget")
    options, queue = tuple(options), tuple(queues)
    root_budget, preparation = _root_budget(options, idle_duration)
    if root_budget > initial_budget:
        raise ValueError("initial LP budget must cover all catalogue duration roots")
    search = ZeroDeltaAffineDualReuseIntegerCapacitySearch(options, queue, lanes, idle_duration=idle_duration)
    root = search.advance_to(root_budget)
    raw_heap_count = len(search._search.gi_frame.f_locals["frontier"])
    # The frozen snapshot scans the raw heap for retention and maximum bound,
    # then the retained entries for ties and inherited-unsolved counts. The
    # call count records its fixed construction work; wall time includes all
    # snapshot computation, including canonical-key evaluation in retention.
    preparation = replace(preparation, root_snapshot_calls=1,
        root_snapshot_heap_entries_read=2 * raw_heap_count,
        root_snapshot_retained_entries_read=2 * root.selection.frontier_regions_remaining)
    initial, probe, probe_stage, evidence = root, None, "none", None

    def finish(selected, final_lp, resumed, status):
        return EarlyRootCanonicalSelection(
            selected, root, root_budget, initial, final_lp, probe, resumed,
            initial_budget, total_budget, probe_budget, status, probe_stage, evidence, preparation)

    if root.selection.canonical_exact:
        return finish(root.selection, root, False, "LP_ROOT_COMPLETE")
    if root.selection.score_exact:
        probe = _probe(options, queue, lanes, root, probe_budget, idle_duration)
        probe_stage = "root"
        if probe.selection.canonical_exact:
            return finish(probe.selection, root, False, "INTEGER_PROBE_COMPLETE")

    initial = search.advance_to(initial_budget)
    if initial.selection.canonical_exact:
        return finish(initial.selection, initial, False, "LP_INITIAL_COMPLETE")
    if initial.selection.score_exact and probe is None:
        probe = _probe(options, queue, lanes, initial, probe_budget, idle_duration)
        probe_stage = "initial"
        if probe.selection.canonical_exact:
            return finish(probe.selection, initial, False, "INTEGER_PROBE_COMPLETE")

    final_lp = search.advance_to(total_budget)
    selected = final_lp.selection
    if selected.canonical_exact:
        return finish(selected, final_lp, True, "LP_RESUME_COMPLETE")
    if selected.score_exact and probe is None:
        probe = _probe(options, queue, lanes, final_lp, probe_budget, idle_duration)
        probe_stage = "resumed"
        selected = probe.selection
        if selected.canonical_exact:
            return finish(selected, final_lp, True, "INTEGER_PROBE_COMPLETE")
    if not selected.score_exact and probe is None:
        evidence = capture_strict_frontier(search)
        probe = _residual_frontier_probe(options, queue, lanes, final_lp, probe_budget, idle_duration, evidence)
        probe_stage = "frontier_late"
        return finish(probe.selection, final_lp, True,
            "FRONTIER_PROBE_COMPLETE" if probe.selection.canonical_exact else "FRONTIER_PROBE_INCOMPLETE")
    return finish(selected, final_lp, True, "LP_RESUME_INCOMPLETE")
