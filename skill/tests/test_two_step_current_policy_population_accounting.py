"""Admission labels reconstruct exact native scores without changing execution."""
from collections import Counter
from fractions import Fraction
from functools import lru_cache

from algorithm.arrival_aware_lane_rollout import PlanningArrivals
from algorithm.experiments.aligned_hardware_frame_model import FrameOption
from algorithm.experiments.aligned_hardware_two_step_current_policy_population_accounting import (
    AccountingPolicy, SOURCES, exact_reconstruction, population_cells, population_total,
)
from algorithm.two_step_current_policy_bg_incumbent_lane_rollout import TwoStepCurrentPolicyBGIncumbentLaneRollout


OPTIONS = (
    FrameOption("a1", (1, 0, 0), 1., ((1.,), (), ())),
    FrameOption("b1", (0, 1, 0), 1.25, ((), (1.25,), ())),
    FrameOption("c1", (0, 0, 1), 1.5, ((), (), (1.5,))),
    FrameOption("loaded-ab", (1, 1, 0), 1.75, ((.5,), (1.75,), ()), loaded=True),
    FrameOption("a2", (2, 0, 0), 1.5, ((1.5, 1.5), (), ())),
)
SAMPLES = 8


def semantic(planner):
    return {"rng": planner.rng.getstate(), "counter": planner.decision_counter,
            "rates": planner.arrival_rates,
            "counts": getattr(planner, "_outer_observed_counts", None),
            "child": semantic(planner.inner_planner) if hasattr(planner, "inner_planner") else None}


def make_pair(seed, cap, rates, counts):
    native = TwoStepCurrentPolicyBGIncumbentLaneRollout(OPTIONS, seed, SAMPLES, cap)
    accounting = AccountingPolicy(OPTIONS, seed, SAMPLES, cap)
    for planner in (native, accounting):
        planner.arrival_rates = rates
        planner._outer_observed_counts = counts
        planner.decision_counter = 7
    return native, accounting


def context(paths):
    return ((2, 1, 1), (1000.1, 1005.3), 1000.1, 0,
            ((2.1, 0.), (1.2,), (0.,)), paths)


def assert_native_equivalence(native, accounting, expected, actual):
    assert actual == expected
    assert semantic(accounting) == semantic(native)
    for name in ("_recourse_work", "_two_step_work", "_inner_current_work", "_inner_current_fallbacks"):
        assert Counter(getattr(accounting, name)) == Counter(getattr(native, name)), name
    assert accounting.recorded_scores == actual
    assert accounting.recorded_post_prefix_state == {
        "rng_state": native.rng.getstate(), "decision_counter": native.decision_counter}
    assert accounting.recording_work == Counter(
        recorded_post_prefix_rng_words_copied=len(native.rng.getstate()[1]),
        recorded_post_prefix_counter_reads=1)


@lru_cache(maxsize=1)
def complete_accounting():
    native, accounting = make_pair("parallel-normal", 16, (.12, .07, .08), (120, 70, 80))
    paths = tuple(PlanningArrivals(events, False, 0, 0) for events in (
        ((0., 0), (.9, 1)), ((.1, 0), (.7, 2)), ((.6, 1),), ()) * 2)
    expected = native.score_background_roots(*context(paths))
    actual = accounting.score_background_roots(*context(paths))
    return native, accounting, paths, expected, actual


def test_zero_time_prefix_birth_has_explicit_source_even_with_zero_age_visible_jobs():
    _, accounting, _, _, actual = complete_accounting()
    assert not actual.background_rollout_overflow
    for root in actual.costs:
        assert accounting.population[root, 0, "tagged_prefix", 0]["n"] == 1
        assert accounting.population[root, 0, "visible_fifo", 0]["n"] == 2
        assert population_total(accounting, root, "visible_fifo", 0)[0] == 2 * SAMPLES
        assert population_total(accounting, root, "tagged_prefix", 0)[0] == 4
    flow = Fraction.from_float(1.2) + Fraction(7, 4)
    for sample in range(SAMPLES):
        assert accounting.population[3, sample, "visible_fifo", 1] == {
            "n": 1, "S1": flow, "S2": flow * flow}


def test_population_decomposition_preserves_all_native_scores_fees_state_and_sample_normalization():
    native, accounting, paths, expected, actual = complete_accounting()
    assert_native_equivalence(native, accounting, expected, actual)
    assert exact_reconstruction(accounting) == {
        "all_roots_exact_Fraction_reconstruction_passed": True, "complete_population": True}
    assert actual.background_admissions > 0
    scored_admissions = background_admissions = 0
    for cell in accounting.population.values():
        assert isinstance(cell["n"], int)
        assert isinstance(cell["S1"], Fraction) and isinstance(cell["S2"], Fraction)
    for root in actual.costs:
        scored = [cell for (r, _, source, _), cell in accounting.population.items()
                  if r == root and source != "unscored_background"]
        assert sum((cell["S1"] for cell in scored), Fraction(0)) == actual.costs[root]
        assert sum((cell["S2"] for cell in scored), Fraction(0)) == actual.squared_costs[root]
        scored_admissions += sum(cell["n"] for cell in scored)
        for coordinate, visible in enumerate((2, 1, 1)):
            assert population_total(accounting, root, "visible_fifo", coordinate)[0] == visible * SAMPLES
            prefix = sum(event[1] == coordinate for path in paths for event in path.events)
            assert population_total(accounting, root, "tagged_prefix", coordinate)[0] == prefix
        cells = population_cells(accounting, root)
        assert len(cells) == len(SOURCES) * 3
        for cell in cells:
            n, first, second = population_total(accounting, root, cell["source"], cell["coordinate"])
            assert cell["scored"] == (cell["source"] != "unscored_background")
            assert cell["mean_jobs_per_sample"] == n / SAMPLES
            assert cell["S1_sum_flow_s_per_sample"] == float(first / SAMPLES)
            assert cell["S2_sum_squared_flow_s2_per_sample"] == float(second / SAMPLES)
            assert cell["mean_admitted_flow_s"] == (float(first / n) if n else None)
            if not cell["scored"]:
                background_admissions += n
    assert scored_admissions == actual.flow_evaluations
    assert background_admissions == actual.background_admissions
    assert scored_admissions + background_admissions == actual.admitted_job_reads


def test_overflow_reconstructs_only_native_partial_admissions_without_complete_population_claim():
    native, accounting = make_pair("parallel-overflow", 2, (.5, .4, .3), (500, 400, 300))
    paths = (PlanningArrivals(((.2, 2), (.9, 1)), False, 0, 0),) * SAMPLES
    expected = native.score_background_roots(*context(paths))
    actual = accounting.score_background_roots(*context(paths))
    assert expected.background_rollout_overflow
    assert_native_equivalence(native, accounting, expected, actual)
    assert exact_reconstruction(accounting) == {
        "all_roots_exact_Fraction_reconstruction_passed": True, "complete_population": False}
    counted = sum(cell["n"] for (_, _, source, _), cell in accounting.population.items()
                  if source != "unscored_background")
    full_population = len(actual.costs) * (4 * SAMPLES + sum(len(path.events) for path in paths))
    assert 0 < counted == actual.flow_evaluations < full_population
    assert sum(cell["n"] for cell in accounting.population.values()) == actual.admitted_job_reads
    assert any(not any(key[0] == root for key in accounting.population) for root in actual.costs)


if __name__ == "__main__":
    test_zero_time_prefix_birth_has_explicit_source_even_with_zero_age_visible_jobs()
    test_population_decomposition_preserves_all_native_scores_fees_state_and_sample_normalization()
    test_overflow_reconstructs_only_native_partial_admissions_without_complete_population_claim()
    print("3 passed", flush=True)
