"""Regression tests for the residual-capacity continuous search trace."""
from __future__ import annotations

from dataclasses import dataclass, fields

import pytest

from algorithm.residual_capacity_frame_candidates import (
    BudgetedFrameSelection,
    select_residual_capacity_frame,
)
from algorithm.experiments.residual_capacity_frame_search_trace import (
    ResidualCapacityFrameSearchTrace,
    trace_residual_capacity_frame_search,
)
from algorithm.experiments.aligned_hardware_frame_model import load_model


@dataclass(frozen=True)
class TraceOption:
    requirements: tuple[int, ...]
    duration: float
    completion_offsets: tuple[tuple[float, ...], ...]


def _assert_selection_fields_equal(actual, expected):
    for field in fields(BudgetedFrameSelection):
        assert getattr(actual, field.name) == getattr(expected, field.name), field.name


@pytest.mark.parametrize("queues", ((6, 6, 10), (4, 3, 5)))
def test_trace_checkpoints_match_independent_selector_runs(queues):
    model = load_model()
    traced = trace_residual_capacity_frame_search(
        model.options,
        queues,
        model.lanes,
        small_budget=26,
        exact_budget=120,
    )

    assert isinstance(traced, ResidualCapacityFrameSearchTrace)
    _assert_selection_fields_equal(
        traced.small,
        select_residual_capacity_frame(model.options, queues, model.lanes, 26),
    )
    _assert_selection_fields_equal(
        traced.exact,
        select_residual_capacity_frame(model.options, queues, model.lanes, 120),
    )
    assert traced.exact.exact
    assert traced.small_gap == traced.small.upper_bound - traced.small.score
    assert traced.score_regret == traced.exact.score - traced.small.score
    assert traced.bound_slack == traced.small.upper_bound - traced.exact.score
    assert traced.gap_identity_error == pytest.approx(0.0, abs=1e-15)
    assert traced.physical_signature_evaluations == (
        traced.exact.feasible_candidate_count + 1
    )


def test_exact_flag_certifies_support_without_unique_physical_tie():
    options = (
        TraceOption((2,), 2.0, ((1.0, 2.0),)),
        TraceOption((1,), 1.0, ((1.0,),)),
    )
    traced = trace_residual_capacity_frame_search(options, (2,), lanes=1)

    assert traced.exact.exact
    assert traced.exact.score == 2.0
    assert traced.exact.option_indices == (1,)
    assert traced.first_exact_score_attempt == 2
    assert traced.first_k120_selected_physical_signature_attempt == 3


def test_trace_work_counts_obey_binary_mask_identities():
    model = load_model()
    traced = trace_residual_capacity_frame_search(
        model.options, (4, 3, 5), model.lanes,
    )
    selected = traced.exact
    masks = 1 << len(model.workloads)
    positions = len(model.options) + 1

    assert selected.bound_price_vector_count == masks
    assert selected.bound_dual_mask_evaluations == (
        selected.bound_threshold_evaluations * masks
    )
    assert selected.bound_dual_mask_sums == selected.upper_bound_calls * masks
    assert selected.bound_dual_coordinate_products == len(model.workloads) * (
        masks * positions + selected.upper_bound_calls
    )
