"""Exact S1 headroom uses the original risk set and genuine Fraction ties."""
from dataclasses import fields
from fractions import Fraction

from algorithm.background_competition_continuous_lane_rollout import BackgroundRootScores
from algorithm.experiments.aligned_hardware_fifo_s1_headroom import root_screen


def scores(first, second):
    values = {field.name: 0 for field in fields(BackgroundRootScores)}
    values.update(costs={root: Fraction(value) for root, value in first.items()},
        squared_costs={root: Fraction(value) for root, value in second.items()},
        candidate_count=len(first), sample_count=8, background_rollout_overflow=False)
    return BackgroundRootScores(**values)


def test_original_risk_filters_low_s1_and_preserves_exact_legal_ties():
    scored = scores({0: 0, 1: 1, 2: 1, 3: 2}, {0: 0, 1: 101, 2: 90, 3: 200})
    original = scores({0: 0, 1: 1, 2: 2, 3: 3}, {0: 11, 1: 5, 2: 10, 3: 8})
    row = root_screen(scored, original, parent_root=1, guard_root=3, bg_root=3, mw_root=2)
    assert row["original_risk_roots"] == [1, 2, 3]
    assert row["argmin_roots"] == [1, 2] and row["exact_tie_count"] == 2
    assert row["min_S1_exact"] == [1, 1] and row["min_S1_per_path"] == .125
    assert row["BG_headroom_exact"] == [1, 1]
    assert row["guard_gap_exact"] == [1, 1] and row["runner_up_gap_exact"] == [1, 1]
    assert {root["root"]: root["risk_allowed"] for root in row["roots"]} == {
        0: False, 1: True, 2: True, 3: True}
    assert row["diagnostic_work"] == {"native_root_score_reads": 12, "recorded_root_rows": 4}


def test_float_neighbors_are_not_ties_and_cached_parent_must_be_exact_minimum():
    epsilon = Fraction(1, 2**60)
    first = {0: Fraction(1), 1: 1 + epsilon, 2: 1 + 2 * epsilon}
    assert len({float(value) for value in first.values()}) == 1
    scored, original = scores(first, {0: 1, 1: 2, 2: 3}), scores(first, {0: 1, 1: 2, 2: 3})
    row = root_screen(scored, original, parent_root=0, guard_root=2, bg_root=1, mw_root=2)
    assert row["argmin_roots"] == [0] and row["exact_tie_count"] == 1
    assert row["BG_headroom_exact"] == [epsilon.numerator, epsilon.denominator]
    assert row["runner_up_gap_exact"] == [epsilon.numerator, epsilon.denominator]
    assert row["guard_gap_exact"] == [(2 * epsilon).numerator, (2 * epsilon).denominator]
    try:
        root_screen(scored, original, parent_root=1, guard_root=2, bg_root=1, mw_root=2)
    except ValueError:
        pass
    else:
        raise AssertionError("a float-equal but strictly higher cached parent passed")


if __name__ == "__main__":
    test_original_risk_filters_low_s1_and_preserves_exact_legal_ties()
    test_float_neighbors_are_not_ties_and_cached_parent_must_be_exact_minimum()
    print("2 passed", flush=True)
