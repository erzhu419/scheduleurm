from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import itertools
import random

import pytest

import algorithm.canonical_count_space_bnb_frame_candidates as canonical
from algorithm.canonical_count_space_bnb_frame_candidates import (
    select_canonical_count_space_bnb_frame,
    select_canonical_count_space_bnb_score_exact_tie_continuation_frame,
    trace_canonical_count_space_bnb_frame,
)
from algorithm.experiments.aligned_hardware_frame_model import load_model


@dataclass(frozen=True)
class _Option:
    requirements: tuple[int, ...]
    duration: float


def _frozen_upper_envelope_options():
    """Nine-option physical envelope used by the fixed-cap trajectory run."""

    return (
        _Option((1, 0, 1), 25.33384071330409),
        _Option((1, 0, 1), 28.494309588582176),
        _Option((1, 0, 0), 16.117803063026795),
        _Option((3, 0, 0), 29.161806321283994),
        _Option((0, 1, 0), 5.483864962033319),
        _Option((0, 3, 0), 13.383258160659276),
        _Option((0, 0, 1), 14.23047969470213),
        _Option((0, 0, 3), 18.491416285122614),
        _Option((0, 0, 8), 29.642385286046192),
    )


def _exhaustive_action(options, queues, lanes, idle_duration=1.0):
    """Small-test oracle for the full score then canonical-key order."""
    option_count = len(options)
    best = None
    for ordered in itertools.product((None, *range(option_count)), repeat=lanes):
        indices = tuple(sorted(
            ordered,
            key=lambda index: option_count if index is None else index,
        ))
        service = tuple(
            sum(
                options[index].requirements[coordinate]
                for index in indices
                if index is not None
            )
            for coordinate in range(len(queues))
        )
        if any(
            done > waiting
            for done, waiting in zip(service, queues, strict=True)
        ):
            continue
        duration = max((
            Fraction.from_float(float(idle_duration)),
            *(
                Fraction.from_float(float(options[index].duration))
                for index in indices
                if index is not None
            ),
        ))
        numerator = sum(
            waiting * done
            for waiting, done in zip(queues, service, strict=True)
        )
        score = Fraction(numerator) / duration
        key = (
            duration,
            tuple(option_count if index is None else index for index in indices),
        )
        candidate = (score, key, indices)
        if (
            best is None
            or candidate[0] > best[0]
            or (candidate[0] == best[0] and candidate[1] < best[1])
        ):
            best = candidate
    return best


@pytest.mark.parametrize(
    ("queue", "expected_counts"),
    (
        (
            (11, 6, 10),
            (2, 0, 0, 3, 0, 2, 0, 0, 1),
        ),
        (
            (15, 1, 0),
            (0, 0, 3, 4, 1, 0, 0, 0, 0),
        ),
    ),
)
def test_formal_cold_run_tie_queues_return_the_canonical_action(
    queue,
    expected_counts,
):
    model = load_model()
    selected = select_canonical_count_space_bnb_frame(
        model.options, queue, lanes=8, lp_node_budget=4096
    )

    assert selected.option_counts == expected_counts
    assert selected.score_exact
    assert selected.canonical_exact
    assert selected.exact
    assert selected.canonical_key_bound_evaluations > 0
    assert selected.canonical_tie_pruned_regions > 0
    assert selected.canonical_tie_regions_remaining == 0


def test_synthetic_equal_score_region_resolves_to_smaller_option_indices():
    # Options 2 and 3 have identical service and duration.  The B&B must keep
    # score-equal regions live until it proves that four copies of option 2 are
    # the smallest canonical maximizer.
    options = (
        _Option((2,), 4.0),
        _Option((1,), 4.0),
        _Option((1,), 2.0),
        _Option((1,), 2.0),
    )
    selected = select_canonical_count_space_bnb_frame(
        options, (6,), lanes=4, lp_node_budget=2000
    )

    assert selected.option_indices == (2, 2, 2, 2)
    assert selected.score == pytest.approx(12.0)
    assert selected.score_exact and selected.canonical_exact and selected.exact
    assert selected.canonical_tie_incumbent_updates > 0
    assert selected.canonical_key_bound_evaluations > 0


def test_zero_score_gap_does_not_claim_canonical_exactness_early():
    # The optimistic key bound deliberately ignores resources.  At K2 the
    # score is closed but one equal-bound box can still appear to contain a
    # smaller key.  One free continuation resolves it without another LP call.
    options = (
        _Option((2, 1), 2.0),
        _Option((1, 0), 2.0),
    )
    early, resolved = trace_canonical_count_space_bnb_frame(
        options, (1, 1), lanes=2, lp_node_budgets=(2, 3)
    )

    assert early.gap == 0.0
    assert early.score_exact
    assert not early.canonical_exact
    assert not early.exact
    assert early.canonical_tie_regions_remaining == 1
    assert resolved.option_indices == (1, None)
    assert resolved.score_exact and resolved.canonical_exact and resolved.exact
    assert resolved.lp_calls == early.lp_calls == 2


@pytest.mark.parametrize(
    (
        "queue",
        "lanes",
        "initial_budget",
        "continuation_budget",
        "primary_work",
        "final_work",
        "primary_action",
        "final_action",
    ),
    (
        (
            (12, 2, 8),
            8,
            288,
            576,
            (288, 30_840, 289),
            (343, 37_275, 344),
            (3, 3, 3, 3, 4, 4, 8, None),
            (3, 3, 3, 3, 4, 4, 8, None),
        ),
        (
            (25, 3, 23),
            16,
            576,
            1152,
            (576, 65_577, 577),
            (1063, 122_556, 1064),
            (0, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 5, 7, 7, 8, 8),
            (0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 5, 8, 8),
        ),
    ),
)
def test_score_exact_tie_continuation_reuses_one_live_frontier(
    monkeypatch,
    queue,
    lanes,
    initial_budget,
    continuation_budget,
    primary_work,
    final_work,
    primary_action,
    final_action,
):
    actual_linprog = canonical.linprog
    linprog_calls = 0

    def counted_linprog(*args, **kwargs):
        nonlocal linprog_calls
        linprog_calls += 1
        return actual_linprog(*args, **kwargs)

    monkeypatch.setattr(canonical, "linprog", counted_linprog)
    result = select_canonical_count_space_bnb_score_exact_tie_continuation_frame(
        _frozen_upper_envelope_options(),
        queue,
        lanes,
        initial_budget,
        continuation_budget,
    )

    primary = result.primary_selection
    selected = result.selection
    assert result.continuation_triggered
    assert result.initial_budget == initial_budget
    assert result.continuation_budget == continuation_budget
    assert primary.score_exact and not primary.canonical_exact
    assert selected.score_exact and selected.canonical_exact
    assert primary.option_indices == primary_action
    assert selected.option_indices == final_action
    assert (
        primary.lp_calls,
        primary.lp_matrix_coefficients,
        primary.candidate_evaluations,
    ) == primary_work
    assert (
        selected.lp_calls,
        selected.lp_matrix_coefficients,
        selected.candidate_evaluations,
    ) == final_work
    assert linprog_calls == selected.lp_calls
    assert linprog_calls != primary.lp_calls + selected.lp_calls


@pytest.mark.parametrize(
    ("queue", "lanes", "initial_budget", "continuation_budget", "exact", "lp_calls"),
    (
        ((1, 1, 1), 8, 20, 100, True, 14),
        ((25, 3, 23), 16, 9, 576, False, 9),
    ),
)
def test_score_exact_tie_continuation_stops_for_exact_or_score_inexact_primary(
    monkeypatch,
    queue,
    lanes,
    initial_budget,
    continuation_budget,
    exact,
    lp_calls,
):
    actual_linprog = canonical.linprog
    observed_calls = 0

    def counted_linprog(*args, **kwargs):
        nonlocal observed_calls
        observed_calls += 1
        return actual_linprog(*args, **kwargs)

    monkeypatch.setattr(canonical, "linprog", counted_linprog)
    result = select_canonical_count_space_bnb_score_exact_tie_continuation_frame(
        _frozen_upper_envelope_options(),
        queue,
        lanes,
        initial_budget,
        continuation_budget,
    )

    assert not result.continuation_triggered
    assert result.selection is result.primary_selection
    assert result.selection.canonical_exact is exact
    assert result.selection.score_exact is exact
    assert result.selection.lp_node_budget == initial_budget
    assert result.selection.lp_calls == observed_calls == lp_calls


def test_public_trace_still_visits_all_requested_score_inexact_checkpoints():
    first, second = trace_canonical_count_space_bnb_frame(
        _frozen_upper_envelope_options(),
        (25, 3, 23),
        lanes=16,
        lp_node_budgets=(9, 20),
    )

    assert not first.score_exact and not second.score_exact
    assert first.lp_node_budget == first.lp_calls == 9
    assert second.lp_node_budget == second.lp_calls == 20


def test_random_small_canonical_actions_match_complete_ordered_oracle():
    rng = random.Random(260913)
    for _ in range(24):
        dimension = rng.randint(1, 3)
        option_count = rng.randint(1, 4)
        lanes = rng.randint(1, 3)
        options = []
        for _ in range(option_count):
            requirements = tuple(
                rng.randint(0, 3) for _ in range(dimension)
            )
            if not any(requirements):
                requirements = (1, *requirements[1:])
            options.append(_Option(
                requirements,
                rng.choice((0.5, 1.0, 2.0, 4.0)),
            ))
        options = tuple(options)
        queues = tuple(rng.randint(0, 7) for _ in range(dimension))

        oracle_score, _, oracle_indices = _exhaustive_action(
            options, queues, lanes
        )
        selected = select_canonical_count_space_bnb_frame(
            options, queues, lanes, lp_node_budget=5000
        )

        assert selected.score_exact and selected.canonical_exact and selected.exact
        assert Fraction.from_float(selected.score) == oracle_score
        assert selected.option_indices == oracle_indices
        assert selected.lp_calls < 5000


def test_empty_catalogue_is_score_and_canonical_exact():
    selected = select_canonical_count_space_bnb_frame(
        (), (3, 2), lanes=4, lp_node_budget=0
    )

    assert selected.option_indices == (None,) * 4
    assert selected.score == selected.upper_bound == selected.gap == 0.0
    assert selected.score_exact and selected.canonical_exact and selected.exact
    assert selected.canonical_key_bound_evaluations == 0
    assert selected.canonical_tie_regions_remaining == 0
