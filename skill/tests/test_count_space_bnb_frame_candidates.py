from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
import random
from types import SimpleNamespace

import pytest

import algorithm.count_space_bnb_frame_candidates as count_bnb
from algorithm.count_space_bnb_frame_candidates import (
    select_count_space_bnb_frame,
    trace_count_space_bnb_frame,
)


@dataclass(frozen=True)
class Option:
    requirements: tuple[int, ...]
    duration: float


def _oracle(options, queues, lanes, idle_duration=1.0):
    best = 0.0
    for indices in itertools.product(
        (None, *range(len(options))), repeat=lanes
    ):
        service = tuple(
            sum(
                options[index].requirements[coordinate]
                for index in indices
                if index is not None
            )
            for coordinate in range(len(queues))
        )
        if any(done > waiting for done, waiting in zip(service, queues)):
            continue
        duration = max((
            idle_duration,
            *(
                options[index].duration
                for index in indices
                if index is not None
            ),
        ))
        best = max(
            best,
            sum(q * done for q, done in zip(queues, service)) / duration,
        )
    return best


def _assert_realizable(selected, options, queues, lanes, idle_duration=1.0):
    assert len(selected.option_counts) == len(options)
    assert len(selected.option_indices) == lanes
    reconstructed_counts = tuple(
        selected.option_indices.count(index) for index in range(len(options))
    )
    assert selected.option_counts == reconstructed_counts
    assert tuple(
        index for index in selected.option_indices if index is not None
    ) == tuple(sorted(
        index for index in selected.option_indices if index is not None
    ))
    service = tuple(
        sum(
            selected.option_counts[index]
            * options[index].requirements[coordinate]
            for index in range(len(options))
        )
        for coordinate in range(len(queues))
    )
    assert selected.service == service
    assert all(done <= waiting for done, waiting in zip(service, queues))
    duration = max((
        idle_duration,
        *(
            options[index].duration
            for index, count in enumerate(selected.option_counts)
            if count
        ),
    ))
    assert selected.duration == duration
    assert selected.score == pytest.approx(
        sum(q * done for q, done in zip(queues, service)) / duration
    )


def _assert_work(selected, option_count, dimension):
    assert selected.lp_calls <= selected.lp_node_budget
    assert selected.lp_calls >= selected.duration_root_count
    assert selected.candidate_evaluations == (
        selected.lp_calls + selected.singleton_candidate_evaluations + 1
    )
    assert selected.dual_certified_lp_calls + selected.safe_fallback_lp_calls == (
        selected.lp_calls
    )
    assert selected.successful_lp_calls <= selected.lp_calls
    assert selected.dual_column_checks <= option_count * selected.lp_calls
    assert selected.residual_fill_option_probes <= option_count * selected.lp_calls
    assert selected.lp_matrix_coefficients <= (
        option_count * (1 + dimension + option_count) * selected.lp_calls
    )
    assert selected.inherited_unsolved_regions <= (
        selected.frontier_regions_remaining
    )
    assert selected.represented_count_vectors >= 1
    assert selected.gap == pytest.approx(
        selected.upper_bound - selected.score
    )


def test_random_small_oracles_are_covered_at_every_budget_prefix():
    rng = random.Random(260912)
    for _ in range(18):
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
            options.append(Option(
                requirements,
                rng.choice((0.5, 1.0, 2.0, 4.0)),
            ))
        options = tuple(options)
        queues = tuple(rng.randint(0, 7) for _ in range(dimension))
        optimum = _oracle(options, queues, lanes)
        roots = len(set(max(1.0, option.duration) for option in options))
        previous_score = -1.0
        previous_upper = math.inf
        for budget in range(roots, roots + 9):
            selected = select_count_space_bnb_frame(
                options, queues, lanes, budget
            )
            _assert_realizable(selected, options, queues, lanes)
            _assert_work(selected, option_count, dimension)
            assert selected.score <= optimum + 1e-12
            assert optimum <= selected.upper_bound + 1e-12
            assert selected.score + 1e-12 >= previous_score
            assert selected.upper_bound <= previous_upper + 1e-12
            previous_score = selected.score
            previous_upper = selected.upper_bound


def test_large_node_budget_closes_small_count_spaces_exactly():
    options = (
        Option((2, 0), 1.0),
        Option((0, 3), 2.0),
        Option((1, 1), 4.0),
        Option((3, 2), 2.0),
    )
    queues = (7, 8)
    lanes = 4
    selected = select_count_space_bnb_frame(
        options, queues, lanes, lp_node_budget=2000
    )
    optimum = _oracle(options, queues, lanes)

    assert selected.exact
    assert selected.score == pytest.approx(optimum)
    assert selected.upper_bound == selected.score
    assert selected.lp_calls < 2000
    assert selected.represented_ordered_configurations == 5 ** 4
    assert selected.represented_count_vectors == math.comb(8, 4)


def test_checkpoint_trace_matches_independent_prefixes_without_repaying_roots(
    monkeypatch,
):
    options = (
        Option((2, 0), 1.0),
        Option((0, 3), 2.0),
        Option((1, 1), 4.0),
        Option((3, 2), 2.0),
    )
    queues = (7, 8)
    budgets = (3, 6, 12)
    original = count_bnb.linprog
    trace_calls = 0

    def counted(*args, **kwargs):
        nonlocal trace_calls
        trace_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(count_bnb, "linprog", counted)
    traced = trace_count_space_bnb_frame(options, queues, 4, budgets)
    assert trace_calls == traced[-1].lp_calls

    monkeypatch.setattr(count_bnb, "linprog", original)
    independent = tuple(
        select_count_space_bnb_frame(options, queues, 4, budget)
        for budget in budgets
    )
    assert traced == independent
    assert trace_calls < sum(row.lp_calls for row in independent)


def test_checkpoint_reads_do_not_reduce_later_frontier_peak():
    # This instance leaves dominated entries in the live heap at K=4.  A
    # checkpoint must report them as terminally pruned without removing them
    # from the cumulative trace, or the later frontier-peak accounting differs
    # from an uninterrupted one-budget run.
    options = (
        Option((0, 3, 3), 4.0),
        Option((0, 1, 2), 1.0),
        Option((2, 3, 2), 2.0),
        Option((2, 3, 1), 8.0),
    )
    queues = (8, 3, 3)
    budgets = (4, 5, 7, 11)
    traced = trace_count_space_bnb_frame(options, queues, 2, budgets)
    independent = tuple(
        select_count_space_bnb_frame(options, queues, 2, budget)
        for budget in budgets
    )
    assert traced == independent
    assert independent[1].frontier_peak == 3


def test_budget_exhaustion_keeps_the_unsolved_sibling_parent_bound():
    options = (
        Option((1, 0), 1.0),
        Option((0, 3), 1.0),
        Option((2, 0), 3.0),
    )
    queues = (3, 7)
    lanes = 4
    selected = select_count_space_bnb_frame(
        options, queues, lanes, lp_node_budget=3
    )
    optimum = _oracle(options, queues, lanes)

    assert selected.duration_root_count == 2
    assert selected.lp_calls == 3
    assert selected.branch_splits == 1
    assert selected.inherited_unsolved_regions == 1
    assert selected.frontier_regions_remaining >= 1
    assert not selected.exact
    assert selected.score <= optimum <= selected.upper_bound


def test_zero_dual_candidate_is_repaired_into_a_safe_exactly_checked_upper(
    monkeypatch,
):
    original = count_bnb.linprog

    def zero_dual(*args, **kwargs):
        result = original(*args, **kwargs)
        return SimpleNamespace(
            success=result.success,
            x=result.x,
            ineqlin=SimpleNamespace(
                marginals=[0.0] * len(kwargs["A_ub"])
            ),
        )

    monkeypatch.setattr(count_bnb, "linprog", zero_dual)
    options = (
        Option((2, 0), 1.0),
        Option((0, 1), 1.0),
    )
    queues = (3, 3)
    selected = select_count_space_bnb_frame(
        options, queues, lanes=3, lp_node_budget=1
    )

    assert selected.dual_certified_lp_calls == 1
    assert selected.safe_fallback_lp_calls == 0
    assert selected.dual_repair_count == 1
    assert selected.maximum_dual_repair > 0.0
    assert _oracle(options, queues, 3) <= selected.upper_bound


def test_solver_failure_retains_the_safe_analytic_bound(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic solver failure")

    monkeypatch.setattr(count_bnb, "linprog", fail)
    options = (
        Option((2, 0), 1.0),
        Option((0, 1), 2.0),
    )
    queues = (3, 3)
    selected = select_count_space_bnb_frame(
        options, queues, lanes=3, lp_node_budget=2
    )

    assert selected.lp_calls == selected.duration_root_count == 2
    assert selected.successful_lp_calls == 0
    assert selected.dual_certified_lp_calls == 0
    assert selected.safe_fallback_lp_calls == 2
    assert _oracle(options, queues, 3) <= selected.upper_bound


def test_empty_catalogue_and_validation_contract():
    empty = select_count_space_bnb_frame((), (3, 2), lanes=4, lp_node_budget=0)
    assert empty.option_indices == (None,) * 4
    assert empty.lp_calls == empty.duration_root_count == 0
    assert empty.score == empty.upper_bound == empty.gap == empty.epsilon == 0.0
    assert empty.exact

    options = (Option((1, 0), 1.0), Option((0, 1), 2.0))
    with pytest.raises(ValueError, match="positive integer"):
        select_count_space_bnb_frame(options, (1, 1), 0, 2)
    with pytest.raises(ValueError, match="nonnegative integer"):
        select_count_space_bnb_frame(options, (1, 1), 2, 1.5)
    with pytest.raises(ValueError, match="cover 2 duration roots"):
        select_count_space_bnb_frame(options, (1, 1), 2, 1)
    with pytest.raises(ValueError, match="positive and finite"):
        select_count_space_bnb_frame(options, (1, 1), 2, 2, idle_duration=0)
    with pytest.raises(ValueError, match="nonnegative integer job counts"):
        select_count_space_bnb_frame(options, (1, -1), 2, 2)
    with pytest.raises(ValueError):
        select_count_space_bnb_frame((Option((0, 0), 1.0),), (1, 1), 2, 1)
    with pytest.raises(ValueError):
        select_count_space_bnb_frame((Option((1,), 1.0),), (1, 1), 2, 1)
    with pytest.raises(ValueError):
        select_count_space_bnb_frame((Option((1, 0), math.inf),), (1, 1), 2, 1)
    with pytest.raises(ValueError, match="at least one checkpoint"):
        trace_count_space_bnb_frame(options, (1, 1), 2, ())
    with pytest.raises(ValueError, match="strictly increasing"):
        trace_count_space_bnb_frame(options, (1, 1), 2, (2, 2))
