"""Residual-capacity routing with exact zero-delta dual short circuits."""
from __future__ import annotations

from typing import Any, Sequence

from algorithm.frontier_score_proof_certificate_routed_frame_candidates import FrontierScoreProofRoutedSelection
from algorithm.zero_delta_affine_dual_reuse_integer_capacity_frame_candidates import ZeroDeltaAffineDualReuseIntegerCapacitySearch
from algorithm.residual_capacity_frontier_frame_candidates import _select_residual_capacity_frontier_frame


def select_zero_delta_affine_dual_residual_capacity_frontier_frame(
    options: Sequence[Any], queues: Sequence[int], lanes: int,
    initial_lp_budget: int, lp_node_budget: int, integer_state_budget: int,
    idle_duration: float = 1.0,
) -> FrontierScoreProofRoutedSelection:
    """Use the existing proof/completion router on one exact zero-delta affine-dual LP search."""
    return _select_residual_capacity_frontier_frame(
        ZeroDeltaAffineDualReuseIntegerCapacitySearch, options, queues, lanes,
        initial_lp_budget, lp_node_budget, integer_state_budget, idle_duration)
