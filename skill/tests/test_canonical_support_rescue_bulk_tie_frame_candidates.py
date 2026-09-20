import random

import pytest

from algorithm import canonical_score_tie_bulk_substitution_frame_candidates as bulk
from algorithm import canonical_support_rescue_bulk_tie_frame_candidates as rescue
from algorithm import canonical_count_space_bnb_frame_candidates as canonical
from algorithm.canonical_count_space_bnb_frame_candidates import (
    select_canonical_count_space_bnb_frame,
)
from algorithm.experiments import (
    aligned_hardware_joint95_v5_envelope_r16_bulk_tie_confirmation as confirmation,
)


SUPPORT_EPSILON_LIMIT = 0.044725253635528484


@pytest.fixture(scope="module")
def model():
    protocol = confirmation._read_object(confirmation.DEFAULT_REGISTRATION)
    snapshot = protocol["development_contract"][
        "physical_complete_frame_envelope"
    ]["model"]
    return (
        confirmation.previous.development.source.capacity.base
        ._model_from_snapshot(snapshot)
    )


def test_unsafe_failure_witness_runs_cold_k144_rescue_without_rng(
    model, monkeypatch
):
    budgets = []
    base_selector = bulk.select_canonical_count_space_bnb_frame

    def recording_base_selector(options, queues, lanes, lp_node_budget, idle_duration):
        budgets.append(lp_node_budget)
        return base_selector(options, queues, lanes, lp_node_budget, idle_duration)

    def forbidden_rng(*args, **kwargs):
        raise AssertionError("support rescue must not use random state")

    monkeypatch.setattr(
        bulk, "select_canonical_count_space_bnb_frame", recording_base_selector
    )
    monkeypatch.setattr(
        canonical, "select_canonical_count_space_bnb_frame", recording_base_selector
    )
    for name in ("random", "Random", "randint", "randrange"):
        if hasattr(rescue, name):
            monkeypatch.setattr(rescue, name, forbidden_rng)
    monkeypatch.setattr(random, "Random", forbidden_rng)
    monkeypatch.setattr(random, "random", forbidden_rng)

    result = rescue.select_canonical_support_rescue_bulk_tie_frame(
        model.options,
        (0, 0, 80),
        16,
        9,
        144,
        SUPPORT_EPSILON_LIMIT,
    )
    reference = select_canonical_count_space_bnb_frame(
        model.options, (0, 0, 80), 16, 4096
    )

    assert budgets == [9, 144]
    assert result.primary_result.selection.epsilon > SUPPORT_EPSILON_LIMIT
    assert result.primary_result.selection.score_exact is False
    assert result.primary_result.selection.canonical_exact is False
    assert result.rescue_triggered is True
    assert result.rescue_result is not None
    assert result.selection.score_exact is True
    assert result.selection.canonical_exact is True
    assert result.selection.option_indices == reference.option_indices
    assert result.selection.score == pytest.approx(reference.score)


def test_rescue_work_is_exact_fieldwise_sum_of_k9_and_k144(model):
    result = rescue.select_canonical_support_rescue_bulk_tie_frame(
        model.options,
        (0, 0, 80),
        16,
        9,
        144,
        SUPPORT_EPSILON_LIMIT,
    )
    assert result.rescue_result is not None

    primary_work = bulk.total_work_row(result.primary_result)
    rescue_work = bulk.total_work_row(result.rescue_result)
    expected = {
        field: primary_work[field] + rescue_work[field]
        for field in primary_work
    }

    assert rescue.total_work_row(result) == expected
    assert expected["lp_calls"] == 139
    assert expected["lp_matrix_coefficients"] == 15087
    assert expected["candidate_evaluations"] == 141


def test_safe_nonexact_seed13041_witness_skips_rescue(model, monkeypatch):
    budgets = []
    selector = rescue.select_canonical_score_tie_bulk_substitution_frame

    def recording_selector(*args, **kwargs):
        budgets.append(args[3])
        return selector(*args, **kwargs)

    monkeypatch.setattr(
        rescue,
        "select_canonical_score_tie_bulk_substitution_frame",
        recording_selector,
    )
    result = rescue.select_canonical_support_rescue_bulk_tie_frame(
        model.options,
        (8, 8, 88),
        16,
        9,
        144,
        SUPPORT_EPSILON_LIMIT,
    )

    assert budgets == [9]
    assert result.primary_result.selection.score_exact is False
    assert result.primary_result.selection.canonical_exact is False
    assert result.primary_result.selection.epsilon < SUPPORT_EPSILON_LIMIT
    assert result.rescue_triggered is False
    assert result.rescue_result is None
    assert result.selection == result.primary_result.selection
    assert rescue.total_work_row(result) == bulk.total_work_row(result.primary_result)


def test_score_exact_tie_uses_primary_bulk_polish_without_rescue(model):
    result = rescue.select_canonical_support_rescue_bulk_tie_frame(
        model.options,
        (88, 88, 96),
        16,
        9,
        144,
        SUPPORT_EPSILON_LIMIT,
    )
    reference = select_canonical_count_space_bnb_frame(
        model.options, (88, 88, 96), 16, 4096
    )

    assert result.primary_result.base_selection.score_exact is True
    assert result.primary_result.base_selection.canonical_exact is False
    assert result.primary_result.polish_triggered is True
    assert result.primary_result.incumbent_changed is True
    assert result.rescue_triggered is False
    assert result.rescue_result is None
    assert result.selection == result.primary_result.selection
    assert result.selection.option_indices == reference.option_indices


def test_support_threshold_requires_margin_beyond_shared_tolerance(
    model, monkeypatch
):
    primary = bulk.select_canonical_score_tie_bulk_substitution_frame(
        model.options, (8, 8, 88), 16, 9
    )
    epsilon = primary.selection.epsilon
    budgets = []

    def fixed_selector(options, queues, lanes, lp_node_budget, idle_duration=1.0):
        budgets.append(lp_node_budget)
        return primary

    monkeypatch.setattr(
        rescue,
        "select_canonical_score_tie_bulk_substitution_frame",
        fixed_selector,
    )

    at_boundary = rescue.select_canonical_support_rescue_bulk_tie_frame(
        model.options, (8, 8, 88), 16, 9, 144, epsilon
    )
    assert at_boundary.rescue_triggered is True
    assert budgets == [9, 144]

    budgets.clear()
    beyond_tolerance = rescue.select_canonical_support_rescue_bulk_tie_frame(
        model.options,
        (8, 8, 88),
        16,
        9,
        144,
        epsilon + 2.0 * rescue.SCORE_TOLERANCE,
    )
    assert beyond_tolerance.rescue_triggered is False
    assert budgets == [9]
