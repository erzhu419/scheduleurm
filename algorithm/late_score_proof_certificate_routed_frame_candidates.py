"""Permit one shared-budget integer proof after the final LP score stays open."""
from __future__ import annotations

from typing import Any, Sequence

from algorithm.branch_reserved_integer_capacity_frame_candidates import BranchReservedIntegerCapacitySearch
from algorithm.certificate_routed_frame_candidates import CertificateRoutedSelection
from algorithm.dual_reuse_integer_capacity_frame_candidates import DualReuseIntegerCapacitySearch
from algorithm.resumable_probe_frame_candidates import _probe


def select_branch_reserved_late_score_proof_frame(
    options: Sequence[Any], queues: Sequence[int], lanes: int,
    initial_lp_budget: int, lp_node_budget: int, integer_state_budget: int,
    idle_duration: float = 1.0,
) -> CertificateRoutedSelection:
    """Use reserved branch slots and attempt a late proof if LP score stays open."""
    return _select_late_score_proof_frame(
        BranchReservedIntegerCapacitySearch, options, queues, lanes,
        initial_lp_budget, lp_node_budget, integer_state_budget, idle_duration)


def select_dual_reuse_late_score_proof_frame(
    options: Sequence[Any], queues: Sequence[int], lanes: int,
    initial_lp_budget: int, lp_node_budget: int, integer_state_budget: int,
    idle_duration: float = 1.0,
) -> CertificateRoutedSelection:
    """Reuse child dual bounds and apply the same conditional late proof rule."""
    return _select_late_score_proof_frame(
        DualReuseIntegerCapacitySearch, options, queues, lanes,
        initial_lp_budget, lp_node_budget, integer_state_budget, idle_duration)


def _select_late_score_proof_frame(
    search_type, options, queues, lanes, initial_lp_budget, lp_node_budget,
    integer_state_budget, idle_duration,
) -> CertificateRoutedSelection:
    """Preserve ordinary routing, adding a first probe only after an open score.

    The unchanged probe shares its integer budget between strict score proof
    and canonical completion. It reuses the final paid LP snapshot and adds
    no LP calls. Earlier probes are never repeated; raw LP snapshots remain
    distinct from any score or canonical certificate completed by integers.
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
    search = search_type(options, queue, lanes, idle_duration=idle_duration)
    initial = search.advance_to(initial_budget)

    def finish(selected, final_lp, probe, resumed, status, probe_stage):
        return CertificateRoutedSelection(
            selected, initial, final_lp, probe, resumed, initial_budget,
            total_budget, probe_budget, status, probe_stage)

    if initial.selection.canonical_exact:
        return finish(initial.selection, initial, None, False, "LP_INITIAL_COMPLETE", "none")

    probe, probe_stage = None, "none"
    if initial.selection.score_exact:
        probe = _probe(options, queue, lanes, initial, probe_budget, idle_duration)
        probe_stage = "initial"
        if probe.selection.canonical_exact:
            return finish(probe.selection, initial, probe, False, "INTEGER_PROBE_COMPLETE", probe_stage)

    final_lp = search.advance_to(total_budget)
    selected = final_lp.selection
    if not selected.canonical_exact and selected.score_exact and probe is None:
        probe = _probe(options, queue, lanes, final_lp, probe_budget, idle_duration)
        probe_stage = "resumed"
        selected = probe.selection
        if selected.canonical_exact:
            return finish(selected, final_lp, probe, True, "INTEGER_PROBE_COMPLETE", probe_stage)

    if not selected.canonical_exact and not selected.score_exact and probe is None:
        probe = _probe(options, queue, lanes, final_lp, probe_budget, idle_duration)
        selected = probe.selection
        return finish(selected, final_lp, probe, True,
                      "LATE_PROBE_COMPLETE" if selected.canonical_exact else "LATE_PROBE_INCOMPLETE",
                      "late")

    return finish(selected, final_lp, probe, True,
                  "LP_RESUME_COMPLETE" if selected.canonical_exact else "LP_RESUME_INCOMPLETE",
                  probe_stage)
