from __future__ import annotations

from dataclasses import dataclass
import itertools
import random

import pytest

from algorithm.temporal_incumbent_count_space_bnb_frame_candidates import (
    project_temporal_incumbent,
    select_count_space_bnb_frame,
)


@dataclass(frozen=True)
class Option:
    requirements: tuple[int, ...]
    duration: float


def _oracle(options, queues, lanes, idle_duration=1.0):
    candidates = []
    for indices in itertools.product((None, *range(len(options))), repeat=lanes):
        ordered = tuple(sorted(index for index in indices if index is not None))
        counts = tuple(ordered.count(index) for index in range(len(options)))
        service = tuple(
            sum(
                counts[index] * options[index].requirements[coordinate]
                for index in range(len(options))
            )
            for coordinate in range(len(queues))
        )
        if any(done > waiting for done, waiting in zip(service, queues, strict=True)):
            continue
        duration = max(
            (idle_duration, *(options[index].duration for index in ordered))
        )
        score = sum(
            waiting * done
            for waiting, done in zip(queues, service, strict=True)
        ) / duration
        padded = ordered + (None,) * (lanes - len(ordered))
        key = (duration, tuple(len(options) if i is None else i for i in padded))
        candidates.append((score, key, counts, service, padded))
    score = max(row[0] for row in candidates)
    return min((row for row in candidates if row[0] == score), key=lambda row: row[1])


def test_projection_clips_previous_counts_and_greedily_adds_only_feasible_copies():
    options = (
        Option((2, 0), 1.0),
        Option((0, 2), 1.0),
        Option((1, 1), 2.0),
    )
    projected = project_temporal_incumbent(
        options, queues=(3, 4), lanes=4, previous_option_counts=(3, 2, 1)
    )

    assert projected.option_counts == (1, 2, 0)
    assert projected.service == (2, 4)
    assert sum(projected.option_counts) <= 4
    assert projected.retained_copies == 3
    assert projected.greedy_added_copies == 0
    assert projected.option_probes == 3


def test_warm_incumbent_is_only_a_lower_bound_and_closes_to_cold_oracle():
    rng = random.Random(260913)
    for _ in range(20):
        dimension = rng.randint(1, 3)
        option_count = rng.randint(1, 4)
        lanes = rng.randint(1, 3)
        options = []
        for _ in range(option_count):
            requirements = tuple(rng.randint(0, 2) for _ in range(dimension))
            if not any(requirements):
                requirements = (1, *requirements[1:])
            options.append(Option(requirements, rng.choice((1.0, 2.0, 4.0))))
        options = tuple(options)
        previous = tuple(rng.randint(0, lanes + 1) for _ in options)
        queue = tuple(rng.randint(0, 6) for _ in range(dimension))
        projected = project_temporal_incumbent(options, queue, lanes, previous)
        warm = select_count_space_bnb_frame(
            options,
            queue,
            lanes,
            4096,
            initial_option_counts=projected.option_counts,
        )
        cold = select_count_space_bnb_frame(options, queue, lanes, 4096)
        oracle = _oracle(options, queue, lanes)

        assert all(
            done <= waiting
            for done, waiting in zip(projected.service, queue, strict=True)
        )
        assert warm.exact and cold.exact
        assert warm.score == cold.score == pytest.approx(oracle[0])
        assert warm.option_counts == cold.option_counts == oracle[2]
        assert warm.service == cold.service == oracle[3]
        assert warm.option_indices == cold.option_indices == oracle[4]
        assert warm.candidate_evaluations == (
            warm.lp_calls + warm.singleton_candidate_evaluations + 2
        )
        assert warm.initial_incumbent_candidate_evaluations == 1
        assert cold.initial_incumbent_candidate_evaluations == 0


def test_invalid_incumbent_cannot_enter_the_search():
    options = (Option((1, 0), 1.0), Option((0, 1), 2.0))
    with pytest.raises(ValueError, match="match the nonnegative integer catalogue"):
        project_temporal_incumbent(options, (1, 1), 2, (1,))
    with pytest.raises(ArithmeticError, match="construction is invalid"):
        select_count_space_bnb_frame(
            options, (4, 4), 2, 4096, initial_option_counts=(2, 2)
        )
    with pytest.raises(ArithmeticError, match="exceeds the visible queue"):
        select_count_space_bnb_frame(
            options, (0, 0), 2, 4096, initial_option_counts=(1, 0)
        )


def test_equal_score_warm_action_cannot_hide_the_cold_preferred_tie():
    options = (
        Option((1,), 1.0),
        Option((1,), 1.0),
    )
    cold = select_count_space_bnb_frame(options, (1,), 1, 4096)
    warm = select_count_space_bnb_frame(
        options, (1,), 1, 4096, initial_option_counts=(1, 0)
    )

    assert cold.exact and warm.exact
    assert cold.score == warm.score == 1.0
    # The cold implementation's root residual candidate is option 1 here.
    # The external equal-score option 0 must not change that reference trace.
    assert cold.option_counts == warm.option_counts == (0, 1)
    assert cold.option_indices == warm.option_indices == (1,)
