"""Try a short unresolved LP prefix before one certified canonical probe."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from algorithm.branch_reserved_integer_capacity_frame_candidates import BranchReservedIntegerCapacitySearch
from algorithm.certificate_routed_frame_candidates import CertificateRoutedSelection
from algorithm.root_pruned_integer_capacity_frame_candidates import RootPrunedIntegerCapacityThresholdSelection
from algorithm.resumable_probe_frame_candidates import _probe


@dataclass(frozen=True)
class ShortLookaheadRoutedSelection(CertificateRoutedSelection):
    lookahead: tuple[RootPrunedIntegerCapacityThresholdSelection, ...]


def select_short_lookahead_certificate_routed_frame(
    options: Sequence[Any], queues: Sequence[int], lanes: int,
    initial_lp_budget: int, lp_node_budget: int, integer_state_budget: int,
    idle_duration: float = 1.0,
) -> ShortLookaheadRoutedSelection:
    """Reuse one frontier and at most one certificate-only integer probe.

    A certified score with an entirely unresolved retained frontier permits
    at most its initial number of one-slot LP advances. Canonical closure
    ends the search immediately; a retained solved region ends the prefix.
    The current certified snapshot then supplies the single integer probe,
    followed by full LP continuation only if needed. ``final_lp`` owns all
    cumulative work; ``lookahead`` records prefix snapshots without adding
    their work again. Root slots remain reserved by the unchanged kernel.
    """
    for value, name in ((initial_lp_budget, "initial_lp_budget"),
                        (lp_node_budget, "lp_node_budget"),
                        (integer_state_budget, "integer_state_budget")):
        if int(value) != value or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    initial_budget, total_budget, probe_budget = map(
        int, (initial_lp_budget, lp_node_budget, integer_state_budget))
    if total_budget < initial_budget:
        raise ValueError("total LP budget must cover the initial LP budget")

    options, queue = tuple(options), tuple(queues)
    search = BranchReservedIntegerCapacitySearch(options, queue, lanes, idle_duration=idle_duration)
    initial = search.advance_to(initial_budget)
    current, current_budget = initial, initial_budget
    lookahead = []

    def finish(selected, final_lp, probe, status, probe_stage):
        return ShortLookaheadRoutedSelection(
            selected, initial, final_lp, probe, current_budget > initial_budget,
            initial_budget, total_budget, probe_budget, status, probe_stage,
            tuple(lookahead))

    if initial.selection.canonical_exact:
        return finish(initial.selection, initial, None, "LP_INITIAL_COMPLETE", "none")

    initial_frontier = initial.selection.frontier_regions_remaining
    if (initial.selection.score_exact and initial_frontier > 0
            and initial.selection.inherited_unsolved_regions == initial_frontier):
        for _ in range(min(initial_frontier, total_budget - initial_budget)):
            current_budget += 1
            current = search.advance_to(current_budget)
            lookahead.append(current)
            if current.selection.canonical_exact:
                return finish(current.selection, current, None, "LP_LOOKAHEAD_COMPLETE", "none")
            if (current.selection.inherited_unsolved_regions
                    < current.selection.frontier_regions_remaining):
                break

    probe, probe_stage = None, "none"
    if current.selection.score_exact:
        probe = _probe(options, queue, lanes, current, probe_budget, idle_duration)
        probe_stage = "lookahead" if lookahead else "initial"
        if probe.selection.canonical_exact:
            return finish(probe.selection, current, probe, "INTEGER_PROBE_COMPLETE", probe_stage)

    if current_budget < total_budget:
        current_budget = total_budget
        current = search.advance_to(current_budget)
    selected = current.selection
    if not selected.canonical_exact and selected.score_exact and probe is None:
        probe = _probe(options, queue, lanes, current, probe_budget, idle_duration)
        probe_stage = "resumed"
        selected = probe.selection
        if selected.canonical_exact:
            return finish(selected, current, probe, "INTEGER_PROBE_COMPLETE", probe_stage)

    return finish(selected, current, probe,
                  "LP_RESUME_COMPLETE" if selected.canonical_exact else "LP_RESUME_INCOMPLETE",
                  probe_stage)
