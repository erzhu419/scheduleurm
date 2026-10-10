"""Use exact native own-tight faces with the frozen fused probe controller."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from algorithm.dual_face_root_canonical_frame_candidates import (
    DUAL_FACE_COUNTERS, DualFaceRootCanonicalSelection, DualFaceWork,
    _capture_root_dual_faces, _dual_face_root_probe,
)
from algorithm.early_root_canonical_frame_candidates import EarlyRootRoutingWork
from algorithm.frontier_score_proof_certificate_routed_frame_candidates import capture_strict_frontier
from algorithm.own_tight_dual_native_kernel import (
    RootSettleWork, OwnTightDualNativeSearch,
    OwnTightDualNativeThresholdSelection,
)
from algorithm.residual_capacity_frontier_frame_candidates import _residual_frontier_probe
from algorithm.resumable_probe_frame_candidates import _probe


@dataclass(frozen=True)
class OwnTightDualNativeSelection(DualFaceRootCanonicalSelection):
    """Preserve genuine raw and settled records built in the same native call."""
    settled_lp: OwnTightDualNativeThresholdSelection | None
    root_settle_work: RootSettleWork | None


def select_own_tight_dual_native_frame(
    options: Sequence[Any], queues: Sequence[int], lanes: int,
    initial_lp_budget: int, lp_node_budget: int, integer_state_budget: int,
    idle_duration: float = 1.0,
) -> OwnTightDualNativeSelection:
    """Pay one native frontier, its root-only settlement, and at most one probe."""
    for value, name in ((initial_lp_budget, "initial_lp_budget"), (lp_node_budget, "lp_node_budget"),
                        (integer_state_budget, "integer_state_budget")):
        if int(value) != value or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    initial_budget, total_budget, probe_budget = map(int, (initial_lp_budget, lp_node_budget, integer_state_budget))
    if total_budget < initial_budget:
        raise ValueError("total LP budget must cover the initial LP budget")
    options, queue = tuple(options), tuple(queues)
    search = OwnTightDualNativeSearch(options, queue, lanes, idle_duration=idle_duration)
    root, settled, settle_work = search.advance_roots_and_settle(initial_budget)
    root_budget = root.selection.duration_root_count
    raw_heap_count = (settle_work.frontier_regions_before if settle_work is not None
                      else len(search._search.gi_frame.f_locals["frontier"]))
    preparation = EarlyRootRoutingWork(0, 0, 0, 0, 0, 0, 0, 0, 1,
        raw_heap_count, 0)
    initial, probe, probe_stage, evidence, dual_evidence = root, None, "none", None, None
    fees = {name: 0 for name in DUAL_FACE_COUNTERS}

    def finish(selected, final_lp, resumed, status):
        return OwnTightDualNativeSelection(
            selected, root, root_budget, initial, final_lp, probe, resumed, initial_budget, total_budget,
            probe_budget, status, probe_stage, evidence, preparation, dual_evidence, DualFaceWork(**fees),
            settled, settle_work)

    if root.selection.canonical_exact:
        return finish(root.selection, root, False, "LP_ROOT_COMPLETE")
    if settled is not None:
        initial = settled
        if settled.selection.canonical_exact:
            return finish(settled.selection, settled, False, "LP_ROOT_SETTLE_COMPLETE")
        dual_evidence = _capture_root_dual_faces(search, fees)
        probe = _dual_face_root_probe(options, queue, lanes, settled, probe_budget, idle_duration, dual_evidence, fees)
        probe_stage = "root"
        if probe.selection.canonical_exact:
            return finish(probe.selection, settled, False, "INTEGER_PROBE_COMPLETE")
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
