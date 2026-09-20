"""Activate the frozen mixed successor only when its second step is necessary.

The catalogue-derived capacity bounds weighted service in one complete global
frame.  If visible weighted backlog exceeds twice that capacity, the current
frame and first successor cannot clear it, so the second successor necessarily
sees nonempty backlog.  The strict gate routes that state to the unchanged
mixed-successor selector; every other state uses the unchanged v4 selector.
"""
from __future__ import annotations

from typing import Any, Sequence

from algorithm.backlog_gated_frame_candidates import BacklogGate, build_backlog_gate
from algorithm.budgeted_frame_candidates import BudgetedFrameSelection, select_budgeted_frame
from algorithm.mixed_successor_rollout_frame_candidates import (
    MixedSuccessorRolloutFrameSelection,
    select_mixed_successor_rollout_frame,
)


__all__ = (
    "BacklogGate",
    "build_backlog_gate",
    "select_rollout_depth_gated_frame",
)


def select_rollout_depth_gated_frame(
    options: Sequence[Any],
    queues: Sequence[int],
    lanes: int,
    budget: int,
    idle_duration: float = 1.0,
    *,
    gate: BacklogGate,
    arrival_rates: Sequence[float],
    terminal_service_rates: Sequence[float],
    arrival_rngs: Sequence[Any],
) -> BudgetedFrameSelection | MixedSuccessorRolloutFrameSelection:
    """Run exactly one frozen selector from the current visible queue.

    ``gate.capacity`` already covers all lanes in one complete frame.  The
    factor two represents the current frame plus the first successor.  Gate
    evaluation consumes no planning randomness.
    """
    if gate.load(queues) > 2 * gate.capacity:
        return select_mixed_successor_rollout_frame(
            options,
            queues,
            lanes,
            budget,
            idle_duration,
            arrival_rates=arrival_rates,
            terminal_service_rates=terminal_service_rates,
            arrival_rngs=arrival_rngs,
        )
    return select_budgeted_frame(options, queues, lanes, budget, idle_duration)
