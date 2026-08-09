from __future__ import annotations

import pytest

from algorithm.experiments.fjsp_family_holdout_freeze import stratified_indices


@pytest.mark.parametrize(
    ("count", "expected"),
    (
        (5, (0, 1, 2, 3, 4)),
        (18, (0, 4, 8, 12, 17)),
        (21, (0, 5, 10, 15, 20)),
        (60, (0, 14, 29, 44, 59)),
    ),
)
def test_stratified_indices_are_deterministic_and_span_family(count, expected):
    assert stratified_indices(count) == expected


def test_stratified_selection_rejects_tiny_families():
    with pytest.raises(ValueError, match="at least five"):
        stratified_indices(4)
