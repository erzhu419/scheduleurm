"""Contracts for two-lane anchor-star frame candidate construction."""
from __future__ import annotations

import ast
from dataclasses import dataclass
import inspect
import math

import pytest

import algorithm.anchor_star_frame_candidates as anchor_star
from algorithm.anchor_star_frame_candidates import select_anchor_star_frame
from algorithm.experiments.aligned_hardware_frame_model import load_model


@dataclass(frozen=True)
class Option:
    requirements: tuple[int, ...]
    duration: float


def _cartesian_optimum(options, queues, idle_duration=1.0):
    """Independent two-lane oracle; the selector is not reused here."""
    best = 0.0
    choices = (None, *range(len(options)))
    for first in choices:
        for second in choices:
            indices = (first, second)
            service = tuple(
                sum(
                    options[index].requirements[coordinate]
                    for index in indices
                    if index is not None
                )
                for coordinate in range(len(queues))
            )
            if any(
                served > available
                for served, available in zip(service, queues, strict=True)
            ):
                continue
            duration = max((
                idle_duration,
                *(
                    options[index].duration
                    for index in indices
                    if index is not None
                ),
            ))
            score = sum(
                queue * served
                for queue, served in zip(queues, service, strict=True)
            ) / duration
            best = max(best, score)
    return best


def _assert_realizable(selected, options, queues, idle_duration=1.0):
    assert len(selected.option_indices) == 2
    assert all(
        index is None or 0 <= index < len(options)
        for index in selected.option_indices
    )
    service = tuple(
        sum(
            options[index].requirements[coordinate]
            for index in selected.option_indices
            if index is not None
        )
        for coordinate in range(len(queues))
    )
    duration = max((
        idle_duration,
        *(
            options[index].duration
            for index in selected.option_indices
            if index is not None
        ),
    ))
    assert selected.service == service
    assert all(
        served <= available
        for served, available in zip(service, queues, strict=True)
    )
    assert selected.duration == duration
    assert selected.score == pytest.approx(
        sum(
            queue * served
            for queue, served in zip(queues, service, strict=True)
        ) / duration
    )


def _assert_work_identities(selected, options, queues, budget):
    option_count = len(options)
    active = sum(
        min(
            2,
            min(
                queue // count
                for queue, count in zip(
                    queues, option.requirements, strict=True
                )
                if count
            ),
        ) > 0
        for option in options
    )

    assert selected.seed_attempts == option_count + 1
    assert selected.attempts == selected.seed_attempts + selected.pair_attempts
    assert selected.active_anchor_count == active
    assert selected.pair_attempts <= active * max(0, active - 1)
    assert selected.feasible_pair_count <= selected.pair_attempts
    assert selected.free_self_skips <= active
    assert selected.free_locally_infeasible_skips <= active * (
        option_count - active
    )
    assert selected.anchor_node_pops == (
        selected.pair_attempts
        + selected.free_self_skips
        + selected.free_locally_infeasible_skips
    )
    assert selected.candidate_score_evaluations == selected.feasible_candidate_count
    assert selected.feasible_candidate_count == (
        1 + active + selected.feasible_pair_count
    )
    assert selected.local_score_evaluations == option_count
    assert selected.upper_bound_calls == (
        selected.frontier_pushes + selected.bound_rejected_anchor_nodes
    )
    assert selected.upper_bound_calls == (
        selected.active_anchor_count
        + selected.anchor_node_pops
        - selected.exhausted_anchor_cursors
    )
    assert selected.bound_queue_coordinate_evaluations == (
        len(queues) * selected.bound_threshold_evaluations
    )
    assert selected.frontier_pushes - selected.anchor_node_pops == (
        selected.exact_stop_frontier_entries_cleared
        + selected.frontier_remaining
    )
    assert selected.budget_blocked_pair_lookups in (0, 1)
    if selected.budget_blocked_pair_lookups:
        assert selected.attempts == budget
        assert selected.frontier_remaining > 0
    assert selected.represented_ordered_action_count == (option_count + 1) ** 2
    assert selected.exhaustive_support_candidate_cap == (
        option_count + 1 + option_count * (option_count - 1)
    )
    assert sorted(selected.ordered_option_indices) == list(range(option_count))
    assert selected.gap == pytest.approx(
        selected.upper_bound - selected.score
    )
    assert selected.epsilon == pytest.approx(
        selected.gap / sum(queues) if sum(queues) else 0.0
    )
    assert selected.exact == (selected.gap == 0.0)
    assert selected.exact == (selected.frontier_remaining == 0)


def test_nine_option_small_catalogue_matches_cartesian_oracle_at_k82():
    options = (
        Option((1, 0), 1.0),
        Option((0, 1), 1.5),
        Option((1, 1), 2.0),
        Option((2, 0), 2.5),
        Option((0, 2), 3.0),
        Option((2, 1), 4.0),
        Option((1, 2), 4.5),
        Option((3, 0), 5.0),
        Option((0, 3), 5.5),
    )
    for queues in ((0, 0), (1, 1), (2, 3), (5, 2), (8, 8)):
        selected = select_anchor_star_frame(
            options, queues, lanes=2, budget=82
        )

        _assert_realizable(selected, options, queues)
        _assert_work_identities(selected, options, queues, budget=82)
        assert selected.exact
        assert selected.score == pytest.approx(
            _cartesian_optimum(options, queues)
        )
        assert selected.upper_bound == pytest.approx(selected.score)
        assert selected.frontier_remaining == 0
        assert selected.attempts <= 82


def test_budget_prefixes_improve_score_and_tighten_the_valid_upper_bound():
    options = (
        Option((1, 0), 1.0),
        Option((0, 1), 1.25),
        Option((1, 1), 2.5),
        Option((2, 1), 3.0),
    )
    queues = (3, 2)
    optimum = _cartesian_optimum(options, queues)
    seed_budget = len(options) + 1
    exact_budget = seed_budget + len(options) * (len(options) - 1)
    previous_score = -1.0
    previous_upper = math.inf
    frozen_order = None

    for budget in range(seed_budget, exact_budget + 1):
        selected = select_anchor_star_frame(
            options, queues, lanes=2, budget=budget
        )

        _assert_realizable(selected, options, queues)
        _assert_work_identities(selected, options, queues, budget=budget)
        assert selected.attempts <= budget
        assert selected.score <= optimum + 1e-12
        assert selected.upper_bound + 1e-12 >= optimum
        assert selected.score + 1e-12 >= previous_score
        assert selected.upper_bound <= previous_upper + 1e-12
        if frozen_order is None:
            frozen_order = selected.ordered_option_indices
        assert selected.ordered_option_indices == frozen_order
        previous_score = selected.score
        previous_upper = selected.upper_bound

    assert selected.exact
    assert selected.score == pytest.approx(optimum)
    assert selected.upper_bound == pytest.approx(optimum)


def test_first_post_seed_attempt_is_a_complete_pair():
    options = (
        Option((1, 0), 1.0),
        Option((0, 1), 1.0),
    )
    queues = (1, 1)

    one_pair = select_anchor_star_frame(
        options, queues, lanes=2, budget=4
    )
    assert one_pair.seed_attempts == 3
    assert one_pair.pair_attempts == 1
    assert None not in one_pair.option_indices
    assert set(one_pair.option_indices) == {0, 1}
    assert one_pair.service == (1, 1)
    assert one_pair.score == 2.0


def test_orientations_are_paid_raw_tie_and_free_skips_close():
    options = (
        Option((1, 0), 1.0),
        Option((0, 2), 2.0),
        Option((2, 1), 1.0),
    )
    queues = (1, 3)
    selected = select_anchor_star_frame(
        options, queues, lanes=2, budget=10
    )

    _assert_realizable(selected, options, queues)
    _assert_work_identities(selected, options, queues, budget=10)
    assert selected.active_anchor_count == 2
    # Options 0 and 1 are the only individually feasible heterogeneous pair,
    # so two paid feasible attempts prove that both raw orientations ran.
    assert selected.pair_attempts == 2
    assert selected.feasible_pair_count == 2
    assert selected.option_indices == (0, 1)
    assert selected.service == (1, 2)
    assert selected.duration == 2.0
    assert selected.score == 3.5
    assert selected.free_self_skips == 2
    assert selected.free_locally_infeasible_skips == 2
    assert selected.attempts == selected.seed_attempts + 2 == 6
    assert selected.anchor_node_pops == 6
    assert selected.exact


def test_exact_certificate_is_support_only_not_a_global_action_tie():
    options = (
        Option((1, 1), 4.0 / 3.0),
        Option((2, 0), 1.0),
        Option((0, 1), 1.0),
    )
    selected = select_anchor_star_frame(
        options, (2, 2), lanes=2, budget=10
    )

    # The homogeneous seed and heterogeneous pair both attain support 6.  The
    # seed closes the support certificate before the shorter pair is paid.
    assert selected.exact
    assert selected.attempts == selected.seed_attempts == 4
    assert selected.option_indices == (0, 0)
    assert selected.duration == pytest.approx(4.0 / 3.0)
    assert selected.score == pytest.approx(6.0)
    assert _cartesian_optimum(options, (2, 2)) == pytest.approx(6.0)


def test_registered_nine_option_model_is_support_exact_by_cap_82():
    model = load_model()
    queues = (100, 100, 100)
    selected = select_anchor_star_frame(
        model.options, queues, lanes=model.lanes, budget=82
    )

    _assert_realizable(selected, model.options, queues)
    _assert_work_identities(selected, model.options, queues, budget=82)
    assert model.lanes == 2
    assert len(model.options) == 9
    assert selected.seed_attempts == 10
    assert selected.pair_attempts <= 72
    assert selected.attempts <= 82
    assert selected.free_self_skips <= 9
    assert selected.free_locally_infeasible_skips == 0
    assert selected.exact
    assert selected.score == pytest.approx(
        _cartesian_optimum(model.options, queues)
    )
    assert selected.upper_bound == pytest.approx(selected.score)


def test_selector_does_not_use_itertools_product(monkeypatch):
    tree = ast.parse(inspect.getsource(anchor_star))
    imported_product_names = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "itertools"
        for alias in node.names
        if alias.name == "product"
    }
    product_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "itertools"
                and node.func.attr == "product"
            )
            or (
                isinstance(node.func, ast.Name)
                and node.func.id in imported_product_names
            )
        )
    ]
    assert product_calls == []

    def forbidden_product(*args, **kwargs):
        raise AssertionError("selector must construct anchor pairs lazily")

    import itertools

    monkeypatch.setattr(itertools, "product", forbidden_product)
    options = tuple(Option((index + 1,), 1.0) for index in range(9))
    selected = select_anchor_star_frame(
        options, (100,), lanes=2, budget=26
    )
    _assert_realizable(selected, options, (100,))
    assert selected.attempts <= 26


def test_only_two_lanes_are_accepted():
    options = (Option((1,), 1.0),)

    for lanes in (1, 3, 1.5):
        with pytest.raises(ValueError, match="two|2"):
            select_anchor_star_frame(options, (2,), lanes=lanes, budget=2)


def test_invalid_inputs_are_rejected_before_free_pair_skips():
    options = (Option((1, 0), 1.0), Option((0, 1), 2.0))

    with pytest.raises(ValueError, match="budget"):
        select_anchor_star_frame(options, (1, 1), lanes=2, budget=2)
    with pytest.raises(ValueError, match="budget"):
        select_anchor_star_frame(options, (1, 1), lanes=2, budget=3.5)
    for idle_duration in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(ValueError, match="idle_duration"):
            select_anchor_star_frame(
                options,
                (1, 1),
                lanes=2,
                budget=3,
                idle_duration=idle_duration,
            )
    for queues in ((1, -1), (1, 0.5)):
        with pytest.raises(ValueError, match="queues"):
            select_anchor_star_frame(options, queues, lanes=2, budget=3)

    invalid_options = (
        Option((1,), 1.0),
        Option((-1, 0), 1.0),
        Option((0.5, 0), 1.0),
        Option((0, 0), 1.0),
        Option((2, 0), 0.0),
        Option((2, 0), math.inf),
        Option((2, 0), math.nan),
    )
    for invalid in invalid_options:
        # Queue (1, 1) makes the positive (2, 0) variants locally
        # infeasible.  Validation must still reject their invalid duration.
        with pytest.raises(ValueError):
            select_anchor_star_frame(
                (options[0], invalid), (1, 1), lanes=2, budget=3
            )
