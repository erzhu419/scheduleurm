import pytest

from algorithm.canonical_count_space_bnb_frame_candidates import (
    select_canonical_count_space_bnb_frame,
)
from algorithm.canonical_score_tie_bulk_substitution_frame_candidates import (
    select_canonical_score_tie_bulk_substitution_frame,
    total_work_row,
)
from algorithm.experiments import (
    aligned_hardware_joint95_v5_envelope_single_exchange_confirmation as confirmation,
)


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


@pytest.mark.parametrize(
    "queue",
    [
        (88, 88, 96),
        (56, 56, 112),
        (56, 56, 80),
    ],
)
def test_bulk_substitution_repairs_consumed_tie_witnesses(model, queue):
    result = select_canonical_score_tie_bulk_substitution_frame(
        model.options, queue, 16, 9
    )
    reference = select_canonical_count_space_bnb_frame(
        model.options, queue, 16, 4096
    )

    assert result.polish_triggered is True
    assert result.pair_checks == 7
    assert result.bulk_candidate_attempts == 1
    assert result.feasible_bulk_candidate_count == 1
    assert result.accepted_bulk_substitution_count == 1
    assert result.incumbent_changed is True
    assert result.selection.option_indices == reference.option_indices
    assert result.selection.score == result.base_selection.score == reference.score
    assert result.selection.upper_bound == result.base_selection.upper_bound
    assert result.selection.score_exact is True
    assert result.selection.canonical_exact is False
    work = total_work_row(result)
    assert work["candidate_evaluations"] == (
        result.base_selection.candidate_evaluations + 7
    )
    assert work["residual_fill_option_probes"] == (
        result.base_selection.residual_fill_option_probes + 7
    )


def test_canonical_exact_base_skips_bulk_substitution(model):
    result = select_canonical_score_tie_bulk_substitution_frame(
        model.options, (88, 88, 96), 16, 57
    )

    assert result.base_selection.canonical_exact is True
    assert result.selection == result.base_selection
    assert result.polish_triggered is False
    assert result.pair_checks == 0
    assert result.bulk_candidate_attempts == 0
    assert result.incumbent_changed is False
