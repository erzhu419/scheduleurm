"""Use the frozen lookahead rule only above a catalogue-derived backlog gate.

Each class weight is inversely proportional to its largest single-lane batch.
The gate capacity bounds the weighted service of every complete global frame.
A weighted backlog strictly above this capacity therefore cannot be cleared
in one frame. This condition is sufficient, not necessary; the gate does not
provide a flow-time guarantee. Each decision calls exactly one frozen selector.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import lcm
from typing import Any, Sequence

from algorithm.budgeted_frame_candidates import (
    BudgetedFrameSelection,
    select_budgeted_frame,
)
from algorithm.lookahead_frame_candidates import (
    LookaheadFrameSelection,
    select_lookahead_frame,
)


@dataclass(frozen=True)
class BacklogGate:
    weights: tuple[int, ...]
    capacity: int
    setup_requirement_reads: int

    def load(self, queues: Sequence[int]) -> int:
        """Return the weighted visible backlog, with one read per coordinate."""
        if len(queues) != len(self.weights):
            raise ValueError("queues must match the gate coordinates")
        return sum(weight * count for weight, count in zip(self.weights, queues))


def build_backlog_gate(options: Sequence[Any], lanes: int) -> BacklogGate:
    """Scan requirements twice to construct integer weights and frame capacity."""
    if int(lanes) != lanes or lanes < 1:
        raise ValueError("lanes must be a positive integer")
    requirements = tuple(option.requirements for option in options)
    if not requirements or not requirements[0]:
        raise ValueError("the gate requires a nonempty catalogue and coordinates")
    largest_batches = [0] * len(requirements[0])
    reads = 0
    for counts in requirements:
        if len(counts) != len(largest_batches):
            raise ValueError("option requirements must share queue coordinates")
        for coordinate, count in enumerate(counts):
            reads += 1
            if int(count) != count or count < 0:
                raise ValueError("requirements must contain nonnegative integer job counts")
            largest_batches[coordinate] = max(largest_batches[coordinate], int(count))
    if any(batch <= 0 for batch in largest_batches):
        raise ValueError("every gate coordinate must have positive catalogue service")
    scale = lcm(*largest_batches)
    weights = tuple(scale // batch for batch in largest_batches)
    largest_weighted_service = 0
    for counts in requirements:
        weighted_service = 0
        for weight, count in zip(weights, counts):
            reads += 1
            weighted_service += weight * int(count)
        largest_weighted_service = max(largest_weighted_service, weighted_service)
    return BacklogGate(weights, int(lanes) * largest_weighted_service, reads)


def select_backlog_gated_frame(
    options: Sequence[Any],
    queues: Sequence[int],
    lanes: int,
    budget: int,
    idle_duration: float = 1.0,
    *,
    gate: BacklogGate,
    arrival_rates: Sequence[float],
    terminal_service_rates: Sequence[float],
) -> BudgetedFrameSelection | LookaheadFrameSelection:
    """Return the chosen frozen selector's native result without another search."""
    if gate.load(queues) > gate.capacity:
        return select_lookahead_frame(
            options, queues, lanes, budget, idle_duration,
            arrival_rates=arrival_rates,
            terminal_service_rates=terminal_service_rates,
            terminal_weight=1.0,
            max_relative_support_loss=0.01,
        )
    return select_budgeted_frame(options, queues, lanes, budget, idle_duration)
