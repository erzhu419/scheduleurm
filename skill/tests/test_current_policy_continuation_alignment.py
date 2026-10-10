"""Persistent old-current continuation preserves causal prefixes and accounting."""
from dataclasses import replace
from fractions import Fraction
from functools import lru_cache

from algorithm.arrival_aware_lane_rollout import PlanningArrivals
from algorithm.experiments import aligned_hardware_current_policy_continuation_alignment as alignment
from algorithm.experiments import aligned_hardware_two_step_current_policy_population_accounting as population
from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption
from algorithm.two_step_bg_incumbent_original_risk_lane_rollout import TwoStepBGIncumbentOriginalRiskLaneRollout


SAMPLES = 8
OPTIONS = (
    FrameOption("a1", (1, 0, 0), 1., ((1.,), (), ())),
    FrameOption("b1", (0, 1, 0), 1.25, ((), (1.25,), ())),
    FrameOption("c1", (0, 0, 1), 1.5, ((), (), (1.5,))),
    FrameOption("loaded-ab", (1, 1, 0), 1.75, ((.5,), (1.75,), ()), loaded=True),
    FrameOption("a2", (2, 0, 0), 1.5, ((1.5, 1.5), (), ())),
)


class RecordingDeterministicCurrent(TwoStepBGIncumbentOriginalRiskLaneRollout):
    """Choose native MW, advancing real child state with one charged RNG draw."""
    def __init__(self, *args):
        super().__init__(*args)
        self.records = []

    def __call__(self, options, queue, available, now, lane, ages, counts):
        self._outer_observed_counts = tuple(counts)
        self.arrival_rates = tuple(count / now if now else 0. for count in counts)
        record = {"queue": tuple(queue), "available": tuple(available), "now": now,
                  "lane": lane, "ages": tuple(tuple(group) for group in ages),
                  "counts": tuple(counts), "rates": self.arrival_rates,
                  "counter_before": self.decision_counter, "rng_before": self.rng.getstate()}
        decision = self.maxweight(options, queue, available, now, lane, ages)
        self.rng.random()
        self.decision_counter += 1
        record.update(counter_after=self.decision_counter, rng_after=self.rng.getstate())
        self.records.append(record)
        return replace(decision, planning_random_draws=1,
            estimated_arrival_rates=self.arrival_rates, observed_arrival_counts=tuple(counts),
            rate_exposure_s=float(now), rate_estimation_coordinate_reads=len(counts),
            rate_estimation_divisions=len(counts) if now else 0)


def planners(options, seed, cap, counts, now):
    finite = population.AccountingPolicy(options, seed, SAMPLES, cap)
    matched = alignment.MatchedContinuationPolicy(options, seed, SAMPLES, cap)
    for planner in (finite, matched):
        planner.arrival_rates = tuple(count / now for count in counts)
        planner._outer_observed_counts = counts
        planner.decision_counter = 7
        planner.inner_planner = RecordingDeterministicCurrent(options, seed, SAMPLES, cap)
    return finite, matched


@lru_cache(maxsize=1)
def clock_case():
    options = (
        FrameOption("a1", (1, 0), 1., ((1.,), ())),
        FrameOption("b1", (0, 1), 1., ((), (1.,))),
        FrameOption("long-loaded-ab", (1, 1), 10., ((.5,), (10.,)), loaded=True),
    )
    finite, matched = planners(options, "continuation-clock", 64, (1, 0), 1000.)
    paths = (PlanningArrivals(((0., 0), (.2, 0), (3.2, 0), (6.2, 0), (9.2, 0)), False, 0, 0),) * SAMPLES
    context = ((1, 0), (1000., 1015.), 1000., 0, ((0.,), ()), paths)
    before = matched.rng.getstate()
    original_scores = finite.score_background_roots(*context)
    matched_scores = matched.score_background_roots(*context)
    return finite, matched, original_scores, matched_scores, before


def test_original_two_call_prefix_and_later_released_counts_idle_rng_counter_and_lane_clock():
    finite, matched, original_scores, matched_scores, before = clock_case()
    assert matched_scores == original_scores and not matched_scores.background_rollout_overflow
    assert matched.rng.getstate() == finite.rng.getstate() == before
    assert matched.decision_counter == finite.decision_counter == 7
    assert len(finite.inner_planner.records) == 2 * SAMPLES
    assert len(matched.inner_planner.records) == 5 * SAMPLES
    for sample in range(SAMPLES):
        original = finite.inner_planner.records[2 * sample:2 * sample + 2]
        continuing = matched.inner_planner.records[5 * sample:5 * sample + 5]
        assert original == continuing[:2]
        assert [record["now"] for record in continuing] == [1001., 1002., 1004., 1007., 1010.]
        assert [record["counts"] for record in continuing] == [(3, 0), (3, 0), (4, 0), (5, 0), (6, 0)]
        assert [record["counter_before"] for record in continuing] == [8, 9, 11, 14, 17]
        assert [record["available"] for record in continuing] == [
            (1001., 1015.), (1002., 1015.), (1004., 1015.), (1007., 1015.), (1010., 1015.)]
        assert all(record["lane"] == 0 for record in continuing)
        expected_ages = (((1., .8), ()), ((1.8,), ()),
                         ((4. - 3.2,), ()), ((7. - 6.2,), ()), ((10. - 9.2,), ()))
        assert [record["ages"] for record in continuing] == list(expected_ages)
        for index, record in enumerate(continuing):
            assert record["counter_after"] == record["counter_before"] + 1
            assert record["rng_after"] != record["rng_before"]
            assert record["rates"] == tuple(count / record["now"] for count in record["counts"])
            if index:
                assert record["rng_before"] == continuing[index - 1]["rng_after"]
            else:
                assert record["rng_before"] == before
    work, steps, late = matched._recourse_work, matched._two_step_work, matched.continuation_work
    assert work["recourse_calls"] == 5 * SAMPLES
    assert steps["recourse_first_calls"] == steps["recourse_second_calls"] == SAMPLES
    assert late["late_current_calls"] == 3 * SAMPLES
    assert late["late_virtual_idle_counter_updates"] == steps["recourse_virtual_idle_counter_updates"] == 5 * SAMPLES
    assert work["recourse_arrival_counter_updates"] == 5 * SAMPLES
    assert finite._recourse_work["recourse_arrival_counter_updates"] == 2 * SAMPLES
    assert work["recourse_rng_words_restored"] == SAMPLES * len(before[1])


@lru_cache(maxsize=1)
def mixed_case():
    finite, matched = planners(OPTIONS, "parallel-normal", 64, (120, 70, 80), 1000.1)
    paths = (PlanningArrivals(((0., 0), (.2, 2), (.9, 1)), False, 0, 0),) * SAMPLES
    context = ((2, 1, 1), (1000.1, 1005.3), 1000.1, 0, ((2.1, 0.), (1.2,), (0.,)), paths)
    before = matched.rng.getstate()
    original_scores = finite.score_background_roots(*context)
    matched_scores = matched.score_background_roots(*context)
    return finite, matched, original_scores, matched_scores, before


def test_all_root_tagged_populations_fraction_costs_and_late_work_preserve_outer_rng():
    finite, matched, original_scores, matched_scores, before = mixed_case()
    assert matched_scores == original_scores and not matched_scores.background_rollout_overflow
    assert matched.rng.getstate() == finite.rng.getstate() == before
    assert matched.decision_counter == finite.decision_counter == 7
    assert matched.population == finite.population
    assert population.exact_reconstruction(matched) == {
        "all_roots_exact_Fraction_reconstruction_passed": True, "complete_population": True}
    assert matched_scores.background_admissions > 0
    for root in matched_scores.costs:
        first = second = Fraction(0)
        for coordinate, visible in enumerate((2, 1, 1)):
            assert population.population_total(matched, root, "visible_fifo", coordinate)[0] == visible * SAMPLES
            assert population.population_total(matched, root, "tagged_prefix", coordinate)[0] == SAMPLES
            for source in population.SOURCES[:2]:
                n, s1, s2 = population.population_total(matched, root, source, coordinate)
                assert n > 0 and isinstance(s1, Fraction) and isinstance(s2, Fraction)
                first += s1
                second += s2
        assert first == matched_scores.costs[root] and second == matched_scores.squared_costs[root]
    flow = Fraction.from_float(1.2) + Fraction(7, 4)
    for sample in range(SAMPLES):
        assert matched.population[3, sample, "visible_fifo", 1] == {"n": 1, "S1": flow, "S2": flow * flow}
    calls = len(matched.inner_planner.records)
    assert calls == matched._recourse_work["recourse_calls"]
    assert matched.continuation_work["late_current_calls"] > 0
    assert calls == (matched._two_step_work["recourse_first_calls"]
        + matched._two_step_work["recourse_second_calls"] + matched.continuation_work["late_current_calls"])
    assert matched._recourse_work["recourse_root_option_probes"] == calls * len(OPTIONS)
    assert matched._recourse_work["recourse_planning_random_draws"] == calls
    assert matched._recourse_work["recourse_rate_estimation_coordinate_reads"] == calls * 3
    assert calls > finite._recourse_work["recourse_calls"]


def test_cap_overflow_partial_population_remains_unavailable_and_cannot_borrow_pair_labels():
    _, matched = planners(OPTIONS, "parallel-overflow", 2, (500, 400, 300), 1000.1)
    paths = (PlanningArrivals(((.2, 2), (.9, 1)), False, 0, 0),) * SAMPLES
    scored = matched.score_background_roots(
        (2, 1, 1), (1000.1, 1005.3), 1000.1, 0, ((2.1, 0.), (1.2,), (0.,)), paths)
    assert scored.background_rollout_overflow
    checks = population.exact_reconstruction(matched)
    assert checks == {"all_roots_exact_Fraction_reconstruction_passed": True, "complete_population": False}
    counted = sum(cell["n"] for (_, _, source, _), cell in matched.population.items()
                  if source != "unscored_background")
    assert 0 < counted == scored.flow_evaluations < len(scored.costs) * 6 * SAMPLES
    native = {
        "tape_id": "toy", "frame_index": 0, "node": "toy", "load": .8, "seed": 41,
        "original_stage_source_match_passed": True, "post_prefix_state_match_passed": True,
        "new_native_option_index": 0, "native_choice_kind": "A", "pair_available": False,
        "new_background_overflow": True, "inner_current_fallbacks": {},
        "new_pair_scores": {"pair_preference": "A", "pair_choice_correct": True},
    }
    baseline = {"group": "kept_beneficial_A", "native_pair_scores": {"pair_choice_correct": True},
                "old_current_T_mean_A_minus_BG_s": -1., "classes": []}
    model = FrameModel("toy", "partial-continuation", 2, ("a", "b", "c"), (1., 1., 1.), OPTIONS)
    record = alignment.fixed_pair_record(matched, native, {}, baseline, model)
    assert not record["pair_available"] and record["new_background_overflow"]
    assert record["accounting_checks"] == checks
    for key in ("new_pair_scores", "new_classes", "population_cells", "fixed_pair_choice_correct",
                "fixed_pair_preference", "tagged_sample_counts_match_passed"):
        assert record[key] is None


if __name__ == "__main__":
    test_original_two_call_prefix_and_later_released_counts_idle_rng_counter_and_lane_clock()
    test_all_root_tagged_populations_fraction_costs_and_late_work_preserve_outer_rng()
    test_cap_overflow_partial_population_remains_unavailable_and_cannot_borrow_pair_labels()
    print("3 passed", flush=True)
