import json

import pytest

from algorithm.canonical_count_space_bnb_frame_candidates import (
    select_canonical_count_space_bnb_frame,
)
from algorithm.canonical_count_space_bnb_single_exchange_frame_candidates import (
    select_canonical_count_space_bnb_single_exchange_frame,
    total_work_row,
)
from algorithm.experiments import (
    aligned_hardware_joint95_v5_envelope_budget_pair_confirmation as confirmation,
)


@pytest.fixture(scope="module")
def v5_model():
    result = json.loads(confirmation.DEFAULT_RESULT.read_text(encoding="utf-8"))
    snapshot = result["protocol"]["development_contract"][
        "physical_complete_frame_envelope"
    ]["model"]
    return confirmation.development.source.capacity.base._model_from_snapshot(snapshot)


def test_single_exchange_repairs_consumed_confirmation_witness(v5_model):
    queue = (12, 4, 44)
    result = select_canonical_count_space_bnb_single_exchange_frame(
        v5_model.options, queue, 8, 48
    )
    reference = select_canonical_count_space_bnb_frame(
        v5_model.options, queue, 8, 4096
    )

    assert result.base_selection.option_indices == (3, 3, 7, 8, 8, 8, 8, 8)
    assert result.selection.option_indices == reference.option_indices
    assert result.selection.score == pytest.approx(reference.score)
    assert result.selection.epsilon < result.base_selection.epsilon
    assert result.exchange_attempts == 27
    assert result.feasible_exchange_count == 24
    assert result.incumbent_changed is True


def test_exchange_work_is_charged_without_changing_lp_or_matrix_work(v5_model):
    result = select_canonical_count_space_bnb_single_exchange_frame(
        v5_model.options, (12, 4, 44), 8, 48
    )
    work = total_work_row(result)

    assert work["lp_calls"] == result.base_selection.lp_calls
    assert work["lp_matrix_coefficients"] == (
        result.base_selection.lp_matrix_coefficients
    )
    assert work["candidate_evaluations"] == (
        result.base_selection.candidate_evaluations + result.exchange_attempts
    )
    assert work["residual_fill_option_probes"] == (
        result.base_selection.residual_fill_option_probes + result.exchange_attempts
    )


def test_exact_base_selection_skips_polish(v5_model):
    result = select_canonical_count_space_bnb_single_exchange_frame(
        v5_model.options, (208, 48, 104), 16, 4096
    )

    assert result.base_selection.score_exact is True
    assert result.polish_triggered is False
    assert result.exchange_attempts == 0
    assert result.selection == result.base_selection
