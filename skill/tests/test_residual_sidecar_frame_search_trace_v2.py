"""Regression tests for corrected sidecar trace-level certificate work."""
from __future__ import annotations

from dataclasses import dataclass, fields

import pytest

from algorithm.residual_sidecar_frame_candidates import (
    ResidualSidecarFrameSelection,
    select_residual_sidecar_frame,
)
from algorithm.experiments.residual_sidecar_frame_search_trace_v2 import (
    ResidualSidecarFrameSearchTraceV2,
    trace_residual_sidecar_frame_search_v2,
)
from algorithm.experiments.aligned_hardware_frame_model import load_model


@dataclass(frozen=True)
class TraceOption:
    requirements: tuple[int, ...]
    duration: float
    completion_offsets: tuple[tuple[float, ...], ...]


def _assert_selection_fields_equal(actual, expected):
    for field in fields(ResidualSidecarFrameSelection):
        assert getattr(actual, field.name) == getattr(expected, field.name), field.name


def _assert_work_identities(selected, option_count, dimensions, threshold_count):
    masks = 1 << dimensions
    positions = option_count + 1
    assert selected.bound_price_vector_count == masks
    assert selected.bound_option_probes == threshold_count * positions
    assert selected.bound_dual_table_option_probes == threshold_count * masks * positions
    assert selected.bound_dual_mask_evaluations == (
        selected.bound_threshold_evaluations * masks
    )
    assert selected.bound_dual_mask_sums == selected.upper_bound_calls * masks
    assert selected.bound_dual_coordinate_products == dimensions * (
        masks * positions + selected.upper_bound_calls
    )
    assert selected.upper_bound_calls == (
        selected.legacy_rejected_prefixes
        + selected.virtual_legacy_serial_allocations
    )
    assert selected.virtual_legacy_serial_allocations == (
        selected.safe_pruned_prefixes_on_generation + selected.frontier_pushes
    )
    assert selected.frontier_pushes == (
        selected.paid_search_pops
        + selected.safe_pruned_prefixes_on_pop
        + selected.exact_stop_frontier_entries_cleared
        + selected.certificate_frontier_entries
    )
    assert selected.attempts == selected.seed_attempts + selected.paid_search_pops
    assert selected.certificate_frontier_scans == 1


def _assert_trace_certificate_work(traced):
    snapshots = 1 if traced.small is traced.exact else 2
    entries = traced.exact.certificate_frontier_entries
    if snapshots == 2:
        entries += traced.small.certificate_frontier_entries
    assert traced.certificate_frontier_scans == snapshots
    assert traced.certificate_frontier_entries_scanned == entries


@pytest.mark.parametrize("queues", ((6, 6, 10), (4, 3, 5)))
def test_continuous_checkpoints_match_independent_selector_runs(queues):
    model = load_model()
    traced = trace_residual_sidecar_frame_search_v2(
        model.options,
        queues,
        model.lanes,
        small_budget=26,
        exact_budget=120,
    )

    assert isinstance(traced, ResidualSidecarFrameSearchTraceV2)
    _assert_selection_fields_equal(
        traced.small,
        select_residual_sidecar_frame(model.options, queues, model.lanes, 26),
    )
    _assert_selection_fields_equal(
        traced.exact,
        select_residual_sidecar_frame(model.options, queues, model.lanes, 120),
    )
    assert traced.exact.exact
    assert traced.small_gap == traced.small.upper_bound - traced.small.score
    assert traced.score_regret == traced.exact.score - traced.small.score
    assert traced.bound_slack == traced.small.upper_bound - traced.exact.score
    assert traced.gap_identity_error == pytest.approx(0.0, abs=1e-15)
    assert traced.physical_signature_evaluations == (
        traced.exact.feasible_candidate_count + 1
    )
    _assert_trace_certificate_work(traced)

    threshold_count = len({
        1.0, *(max(1.0, float(option.duration)) for option in model.options),
    })
    _assert_work_identities(
        traced.small, len(model.options), len(model.workloads), threshold_count,
    )
    _assert_work_identities(
        traced.exact, len(model.options), len(model.workloads), threshold_count,
    )


def test_small_snapshot_does_not_charge_or_mutate_exact_search():
    options = (
        TraceOption((0, 2, 3), 7.0, ((), (7.0, 7.0), (7.0, 7.0, 7.0))),
        TraceOption((3, 0, 2), 2.0, ((2.0, 2.0, 2.0), (), (2.0, 2.0))),
        TraceOption((0, 3, 3), 3.0, ((), (3.0, 3.0, 3.0), (3.0, 3.0, 3.0))),
        TraceOption((3, 3, 0), 2.0, ((2.0, 2.0, 2.0), (2.0, 2.0, 2.0), ())),
        TraceOption((2, 3, 2), 3.0, ((3.0, 3.0), (3.0, 3.0, 3.0), (3.0, 3.0))),
    )
    traced = trace_residual_sidecar_frame_search_v2(
        options, (3, 5, 9), lanes=2, small_budget=12, exact_budget=24,
    )

    _assert_selection_fields_equal(
        traced.small,
        select_residual_sidecar_frame(options, (3, 5, 9), 2, 12),
    )
    _assert_selection_fields_equal(
        traced.exact,
        select_residual_sidecar_frame(options, (3, 5, 9), 2, 24),
    )
    assert traced.exact.exact
    assert (
        traced.exact.safe_pruned_prefixes_on_generation
        + traced.exact.safe_pruned_prefixes_on_pop
    ) > 0
    _assert_trace_certificate_work(traced)


def test_exact_flag_certifies_support_without_unique_physical_tie():
    options = (
        TraceOption((2,), 2.0, ((1.0, 2.0),)),
        TraceOption((1,), 1.0, ((1.0,),)),
    )
    traced = trace_residual_sidecar_frame_search_v2(options, (2,), lanes=1)

    assert traced.exact.exact
    assert traced.exact.score == 2.0
    assert traced.exact.option_indices == (1,)
    assert traced.first_exact_score_attempt == 2
    assert traced.first_k120_selected_physical_signature_attempt == 3
    _assert_trace_certificate_work(traced)
