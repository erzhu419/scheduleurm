"""Tests for the one-frontier legacy-v4 budget curve."""
from __future__ import annotations

from dataclasses import dataclass, fields
import itertools
import random

import pytest

from algorithm.budgeted_frame_candidates import (
    BudgetedFrameSelection,
    select_budgeted_frame,
)
from algorithm.experiments.aligned_budgeted_frame_search_curve import (
    DEFAULT_CHECKPOINT_BUDGETS,
    trace_budgeted_frame_search_curve,
)
from algorithm.experiments.aligned_hardware_frame_model import load_model


@dataclass(frozen=True)
class Option:
    requirements: tuple[int, ...]
    duration: float


def _assert_selection_equal(actual, expected):
    for field in fields(BudgetedFrameSelection):
        assert getattr(actual, field.name) == getattr(expected, field.name), field.name


def _cartesian_oracle_score(options, queues, lanes, idle_duration=1.0):
    best = 0.0
    for indices in itertools.product((None, *range(len(options))), repeat=lanes):
        service = tuple(
            sum(
                options[index].requirements[coordinate]
                for index in indices
                if index is not None
            )
            for coordinate in range(len(queues))
        )
        if any(count > available for count, available in zip(service, queues)):
            continue
        duration = max((
            idle_duration,
            *(options[index].duration for index in indices if index is not None),
        ))
        best = max(
            best,
            sum(q * count for q, count in zip(queues, service)) / duration,
        )
    return best


def _exact_budget(option_count, lanes):
    width = option_count + 1
    return width + sum(width**depth for depth in range(1, lanes + 1))


def _assert_work_identities(curve):
    work = curve.work
    last = curve.checkpoints[-1].selection
    assert work.trace_calls == 1
    assert work.checkpoint_reads == len(curve.checkpoint_budgets)
    assert work.carried_exact_checkpoint_reads == sum(
        checkpoint.carried_after_exact for checkpoint in curve.checkpoints
    )
    assert work.paid_candidate_attempts == work.seed_attempts + work.expanded_prefixes
    assert (
        work.feasible_candidate_evaluations + work.infeasible_candidate_attempts
        == work.paid_candidate_attempts
    )
    assert work.upper_bound_calls == work.frontier_push_attempts
    assert (
        work.frontier_pushes + work.bound_rejected_pushes
        == work.frontier_push_attempts
    )
    assert (
        work.frontier_pops
        == work.expanded_prefixes + work.certificate_closure_pops
    )
    assert work.frontier_pushes == (
        work.frontier_pops
        + work.frontier_entries_discarded_on_exact
        + work.frontier_remaining
    )
    assert work.paid_candidate_attempts == last.attempts
    assert work.seed_attempts == last.seed_attempts
    assert work.feasible_candidate_evaluations == last.feasible_candidate_count
    assert work.local_score_evaluations == last.local_score_evaluations
    assert work.bound_option_probes == last.bound_option_probes
    assert work.bound_threshold_evaluations == last.bound_threshold_evaluations
    assert work.frontier_peak == last.frontier_peak


def _assert_curve_properties(curve, oracle_score, queue_norm):
    exact_seen = False
    previous = None
    for checkpoint in curve.checkpoints:
        selection = checkpoint.selection
        assert selection.score <= oracle_score + 1e-12
        assert oracle_score <= selection.upper_bound + 1e-12
        regret = oracle_score - selection.score
        slack = selection.upper_bound - oracle_score
        assert regret >= -1e-12
        assert slack >= -1e-12
        assert selection.gap == pytest.approx(regret + slack, abs=1e-12)
        expected_epsilon = selection.gap / queue_norm if queue_norm else 0.0
        assert selection.epsilon == pytest.approx(expected_epsilon, abs=1e-12)
        if previous is not None:
            assert selection.score + 1e-12 >= previous.score
            assert selection.upper_bound <= previous.upper_bound + 1e-12
            assert selection.gap <= previous.gap + 1e-12
            assert selection.epsilon <= previous.epsilon + 1e-12
            assert selection.attempts >= previous.attempts
            assert (
                selection.feasible_candidate_count
                >= previous.feasible_candidate_count
            )
            assert (
                selection.bound_threshold_evaluations
                >= previous.bound_threshold_evaluations
            )
            if exact_seen:
                _assert_selection_equal(selection, previous)
        exact_seen = exact_seen or selection.exact
        previous = selection
    _assert_work_identities(curve)


def test_real_model_every_integer_checkpoint_matches_fresh_selector():
    model = load_model()
    queues = (4, 3, 5)
    curve = trace_budgeted_frame_search_curve(
        model.options,
        queues,
        model.lanes,
    )

    assert curve.checkpoint_budgets == DEFAULT_CHECKPOINT_BUDGETS
    assert tuple(checkpoint.budget for checkpoint in curve.checkpoints) == tuple(
        range(10, 121)
    )
    for checkpoint in curve.checkpoints:
        _assert_selection_equal(
            checkpoint.selection,
            select_budgeted_frame(
                model.options,
                queues,
                model.lanes,
                checkpoint.budget,
            ),
        )

    oracle_score = _cartesian_oracle_score(model.options, queues, model.lanes)
    _assert_curve_properties(curve, oracle_score, sum(queues))
    assert not curve.selection_at(26).exact
    assert curve.selection_at(120).exact
    assert curve.selection_at(120).score == pytest.approx(oracle_score)
    assert curve.first_exact_budget is not None
    assert curve.first_exact_paid_attempt == curve.first_exact_budget


def test_random_small_catalogues_match_fresh_selector_and_oracle_at_every_k():
    rng = random.Random(20260912)
    for _ in range(12):
        options = []
        for _ in range(3):
            requirements = (rng.randint(0, 3), rng.randint(0, 3))
            while not any(requirements):
                requirements = (rng.randint(0, 3), rng.randint(0, 3))
            options.append(
                Option(
                    requirements=requirements,
                    duration=rng.choice((1.0, 1.5, 2.0, 3.5)),
                )
            )
        options = tuple(options)
        queues = (rng.randint(0, 7), rng.randint(0, 7))
        lanes = 2
        seed_budget = len(options) + 1
        final_budget = _exact_budget(len(options), lanes)
        budgets = tuple(range(seed_budget, final_budget + 1))
        curve = trace_budgeted_frame_search_curve(
            options,
            queues,
            lanes,
            checkpoint_budgets=budgets,
        )

        for checkpoint in curve.checkpoints:
            _assert_selection_equal(
                checkpoint.selection,
                select_budgeted_frame(
                    options,
                    queues,
                    lanes,
                    checkpoint.budget,
                ),
            )
        oracle_score = _cartesian_oracle_score(options, queues, lanes)
        _assert_curve_properties(curve, oracle_score, sum(queues))
        assert curve.checkpoints[-1].selection.exact
        assert curve.checkpoints[-1].selection.score == pytest.approx(oracle_score)


def test_early_exact_checkpoints_persist_without_more_selector_work():
    model = load_model()
    curve = trace_budgeted_frame_search_curve(
        model.options,
        (0, 0, 0),
        model.lanes,
    )
    first = curve.checkpoints[0].selection

    assert curve.first_exact_budget == 10
    assert curve.first_exact_paid_attempt == 10
    assert first.exact
    assert first.attempts == 10
    for checkpoint in curve.checkpoints:
        _assert_selection_equal(checkpoint.selection, first)
        assert checkpoint.carried_after_exact == (checkpoint.budget > 10)
    assert curve.work.paid_candidate_attempts == 10
    assert curve.work.carried_exact_checkpoint_reads == 110
    assert curve.work.expanded_prefixes == 0


def test_exact_between_sparse_checkpoints_is_carried_to_the_next_budget():
    options = (
        Option((1, 0), 1.0),
        Option((0, 1), 1.0),
    )
    queues = (1, 1)
    budgets = (3, 6, 8, 15)
    curve = trace_budgeted_frame_search_curve(
        options,
        queues,
        lanes=2,
        checkpoint_budgets=budgets,
    )

    assert curve.first_exact_paid_attempt == 7
    assert curve.first_exact_budget == 8
    assert not curve.selection_at(6).exact
    assert curve.checkpoints[2].carried_after_exact
    assert curve.checkpoints[3].carried_after_exact
    for checkpoint in curve.checkpoints:
        _assert_selection_equal(
            checkpoint.selection,
            select_budgeted_frame(options, queues, 2, checkpoint.budget),
        )
    _assert_selection_equal(curve.selection_at(8), curve.selection_at(15))
    _assert_curve_properties(curve, 2.0, sum(queues))


def test_checkpoint_contract_rejects_ambiguous_budgets_and_allows_partial_curve():
    options = (Option((1,), 1.0), Option((2,), 3.0))

    for budgets in ((), (2,), (3, 3), (4, 3), (3, 3.5)):
        with pytest.raises(ValueError):
            trace_budgeted_frame_search_curve(
                options,
                (4,),
                lanes=2,
                checkpoint_budgets=budgets,
            )

    partial = trace_budgeted_frame_search_curve(
        options,
        (4,),
        lanes=2,
        checkpoint_budgets=(3,),
    )
    _assert_selection_equal(
        partial.selection_at(3),
        select_budgeted_frame(options, (4,), 2, 3),
    )
    assert partial.first_exact_budget == (
        3 if partial.selection_at(3).exact else None
    )
    with pytest.raises(KeyError):
        partial.selection_at(4)
