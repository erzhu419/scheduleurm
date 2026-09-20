"""Support-triggered cold-budget rescue around the canonical bulk-tie selector.

The primary and optional rescue are independent calls to the existing
canonical score-tie bulk-substitution selector.  The rescue is triggered only
when the primary certificate does not leave a strictly positive support margin
beyond the shared score tolerance.  No reference action, random state, or work
from the primary call is reused by the rescue.

``total_work_row`` charges every field from every selector call that actually
ran.  A rescue must preserve or improve the primary score and must preserve or
tighten its certified upper bound.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from algorithm.canonical_count_space_bnb_frame_candidates import (
    CanonicalCountSpaceBnbSelection,
)
from algorithm.canonical_score_tie_bulk_substitution_frame_candidates import (
    CanonicalScoreTieBulkSelection,
    select_canonical_score_tie_bulk_substitution_frame,
    total_work_row as bulk_tie_total_work_row,
)


SCORE_TOLERANCE = 1e-10


@dataclass(frozen=True)
class CanonicalSupportRescueBulkTieSelection:
    primary_result: CanonicalScoreTieBulkSelection
    rescue_result: CanonicalScoreTieBulkSelection | None
    selection: CanonicalCountSpaceBnbSelection
    rescue_triggered: bool


def select_canonical_support_rescue_bulk_tie_frame(
    options: Sequence[Any],
    queues: Sequence[int],
    lanes: int,
    primary_lp_node_budget: int,
    rescue_lp_node_budget: int,
    support_epsilon_limit: float,
    idle_duration: float = 1.0,
) -> CanonicalSupportRescueBulkTieSelection:
    """Run a cold primary selector and, when required, a cold rescue selector."""

    primary = select_canonical_score_tie_bulk_substitution_frame(
        options,
        queues,
        lanes,
        primary_lp_node_budget,
        idle_duration,
    )
    triggered = (
        float(support_epsilon_limit) - primary.selection.epsilon
        <= SCORE_TOLERANCE
    )
    if not triggered:
        return CanonicalSupportRescueBulkTieSelection(
            primary_result=primary,
            rescue_result=None,
            selection=primary.selection,
            rescue_triggered=False,
        )

    rescue = select_canonical_score_tie_bulk_substitution_frame(
        options,
        queues,
        lanes,
        rescue_lp_node_budget,
        idle_duration,
    )
    if rescue.selection.score + SCORE_TOLERANCE < primary.selection.score:
        raise ArithmeticError("support rescue reduced the selected score")
    if (
        rescue.selection.upper_bound
        > primary.selection.upper_bound + SCORE_TOLERANCE
    ):
        raise ArithmeticError("support rescue loosened the certified upper bound")
    return CanonicalSupportRescueBulkTieSelection(
        primary_result=primary,
        rescue_result=rescue,
        selection=rescue.selection,
        rescue_triggered=True,
    )


def total_work_row(
    result: CanonicalSupportRescueBulkTieSelection,
) -> dict[str, int]:
    """Return the exact sum of all primary and executed rescue work fields."""

    primary = bulk_tie_total_work_row(result.primary_result)
    if result.rescue_result is None:
        return primary
    rescue = bulk_tie_total_work_row(result.rescue_result)
    return {field: primary[field] + rescue[field] for field in primary}


__all__ = [
    "CanonicalSupportRescueBulkTieSelection",
    "SCORE_TOLERANCE",
    "select_canonical_support_rescue_bulk_tie_frame",
    "total_work_row",
]
