"""Exact reuse preserves native FIFO scores, actions, random order, and fees."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, replace
from random import Random

import pytest

import algorithm.background_competition_continuous_lane_rollout as background_module
from algorithm.arrival_aware_lane_rollout import PlanningArrivals
from algorithm.background_competition_continuous_lane_rollout import BackgroundCompetitionContinuousLaneRollout
from algorithm.exact_reuse_inner_decision import InnerDecisionReuse
from algorithm.exact_reuse_lane_rollout import (
    ExactReuseBackgroundCompetitionContinuousLaneRollout,
    ExactReuseTwoStepCurrentPolicyBGIncumbentLaneRollout,
)
from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption
from algorithm.experiments.observed_rate_contextual_lane_refill_replay import simulate_observed_rate_contextual_trajectory
from algorithm.two_step_bg_incumbent_original_risk_lane_rollout import TwoStepBGIncumbentOriginalRiskLaneRollout
from algorithm.two_step_current_policy_bg_incumbent_lane_rollout import TwoStepCurrentPolicyBGIncumbentLaneRollout


OPTIONS = (FrameOption("a1", (1, 0, 0), 1., ((1.,), (), ())),
           FrameOption("b1", (0, 1, 0), 1.25, ((), (1.25,), ())),
           FrameOption("c1", (0, 0, 1), 1.5, ((), (), (1.5,))),
           FrameOption("loaded-ab", (1, 1, 0), 1.75, ((.5,), (1.75,), ()), loaded=True),
           FrameOption("a2", (2, 0, 0), 1.5, ((1.5, 1.5), (), ())))


def semantic(planner):
    return {"rng": planner.rng.getstate(), "counter": planner.decision_counter,
            "rates": planner.arrival_rates,
            "counts": getattr(planner, "_outer_observed_counts", None),
            "child": semantic(planner.inner_planner) if hasattr(planner, "inner_planner") else None}


def restore(planner, state):
    planner.rng.setstate(state["rng"])
    planner.decision_counter = state["counter"]
    planner.arrival_rates = state["rates"]
    if state["counts"] is not None:
        planner._outer_observed_counts = state["counts"]
    if state["child"] is not None:
        restore(planner.inner_planner, state["child"])


def fixed_suffix_gaps(monkeypatch, gap):
    class ScriptedRandom:
        def __init__(self, seed):
            pass

        def expovariate(self, rate):
            return gap

    monkeypatch.setattr(background_module, "Random", ScriptedRandom)


def test_mw_reuse_uses_full_queue_catalogue_and_exact_ties_with_real_scan_fees():
    options = OPTIONS[:-1] + (replace(OPTIONS[-1], duration=2., completion_offsets=((2., 2.), (), ())),)
    native = BackgroundCompetitionContinuousLaneRollout(options, "mw-exact", sample_count=1, event_cap=2)
    fast = ExactReuseBackgroundCompetitionContinuousLaneRollout(options, "mw-exact", sample_count=1, event_cap=2)
    states = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (2, 0, 0), (0, 2, 0), (1, 1, 1))
    before = semantic(fast)
    for queue in states + states:
        assert fast._best_index(queue) == native._best_index(queue)
    assert fast._best_index((2, 0, 0))[0] == 0  # Exact support tie keeps the shorter period.
    work = fast._mw_reuse.work
    assert work["cache_misses"] == len(states) and work["cache_hits"] == len(states) + 1
    assert work["actual_option_probes"] == len(states) * len(options)
    assert work["logical_option_probes"] == (2 * len(states) + 1) * len(options)
    assert work["cache_key_coordinate_reads"] == 3 * (2 * len(states) + 1)
    assert semantic(fast) == before
    other_options = options[:-1] + (replace(options[-1], duration=1., completion_offsets=((1., 1.), (), ())),)
    other = ExactReuseBackgroundCompetitionContinuousLaneRollout(other_options, "mw-exact", sample_count=1, event_cap=2)
    assert other._best_index((2, 0, 0))[0] == 4
    fast._mw_reuse.finish_outer_decision()
    assert not fast._mw_reuse.cache
    assert fast._best_index((2, 0, 0)) == native._best_index((2, 0, 0))
    assert fast._mw_reuse.work["cache_misses"] == len(states) + 1
    for count in range(20):
        assert fast._best_index((count, 0, 0)) == native._best_index((count, 0, 0))
        assert len(fast._mw_reuse.cache) <= fast._mw_reuse.capacity


@pytest.mark.parametrize("rates,cap,gap,overflow", [
    ((0., 0., 0.), 16, 1., False),
    ((.5, .4, .3), 128, .4, False),
    ((.5, .4, .3), 2, .1, True),
])
def test_dyadic_leaf_matches_exact_fraction_fifo_competition_and_partial_fees(monkeypatch, rates, cap, gap, overflow):
    fixed_suffix_gaps(monkeypatch, gap)
    native = BackgroundCompetitionContinuousLaneRollout(OPTIONS, "dyadic-live", sample_count=2, event_cap=cap)
    fast = ExactReuseBackgroundCompetitionContinuousLaneRollout(OPTIONS, "dyadic-live", sample_count=2, event_cap=cap)
    for planner in (native, fast):
        planner.arrival_rates = rates
        planner.decision_counter = 11
    args = ((2, 1, 1), (10.1, 12.3, 15.5), 10.1, 0,
            ((2.1, .3), (1.2,), (.1,)),
            (PlanningArrivals(((.2, 2), (.9, 1)), False, 0, 0),) * 2)
    before = semantic(fast)
    expected = native.score_background_roots(*args)
    actual = fast.score_background_roots(*args)
    assert actual == expected
    assert actual.background_rollout_overflow == overflow
    assert semantic(fast) == semantic(native) == before
    for field in ("background_planning_events", "background_planning_random_draws",
                  "background_planning_rate_reads", "background_seed_initializations",
                  "flow_evaluations", "rollout_calls", "admitted_job_reads"):
        assert fast.kernel_work[field] == getattr(actual, field)
    assert fast.kernel_work["scorer_calls"] == 1
    if rates[0] and not overflow:
        assert actual.background_admissions > 0 and actual.background_planning_events > 0
    if overflow:
        assert 0 < actual.flow_evaluations < actual.candidate_count * 2 * 6


def test_complete_inner_choose_hits_only_same_causal_input_rng_and_counter():
    native = TwoStepBGIncumbentOriginalRiskLaneRollout(OPTIONS, "inner-key", sample_count=2, event_cap=8)
    cached = TwoStepBGIncumbentOriginalRiskLaneRollout(OPTIONS, "inner-key", sample_count=2, event_cap=8)
    reuse = InnerDecisionReuse(cached)
    initial = semantic(native)
    base = (OPTIONS, (2, 1, 1), (10.1, 11.3), 10.1, 0,
            ((2.1, .3), (1.2,), (.1,)), (4, 2, 2))
    variants = [base,
        (*base[:5], ((3.1, .3), (1.2,), (.1,)), base[6]),
        (base[0], base[1], (10.1, 11.5), *base[3:]),
        (*base[:6], (5, 2, 2)),
        (base[0], base[1], (10.3, 11.3), 10.3, *base[4:]),
        (base[0], (1, 2, 1), *base[2:5], ((.3,), (1.2, .2), (.1,)), base[6]),
        (base[0], base[1], (11.3, 10.1), base[3], 1, *base[5:])]
    first = None
    for index, context in enumerate(variants + [base, base]):
        state = deepcopy(initial)
        if index == len(variants) + 1:
            state["counter"] += 1
        restore(native, state)
        restore(cached, state)
        expected, actual = native(*context), cached(*context)
        assert asdict(actual) == asdict(expected) and semantic(cached) == semantic(native)
        if index == 0:
            first = actual
    state = deepcopy(initial)
    rng = Random()
    rng.setstate(state["rng"])
    rng.random()
    state["rng"] = rng.getstate()
    restore(native, state)
    restore(cached, state)
    assert asdict(cached(*base)) == asdict(native(*base))
    assert semantic(cached) == semantic(native)
    assert reuse.work["current_choose_calls"] == len(variants) + 3
    assert reuse.work["current_cache_hits"] == 1
    assert reuse.work["current_native_choose_calls"] == len(variants) + 2
    assert reuse.saved_work["current"]["root_option_probes"] == first.root_option_probes
    assert reuse.saved_work["current"]["rate_estimation_coordinate_reads"] == 0
    reuse.begin_outer_decision()
    assert not reuse.caches["current"] and not reuse.caches["BG"]


def test_no_recourse_hit_preserves_the_calling_branch_child_state():
    options = (FrameOption("single", (1,), 1., ((1.,),)),)
    native = TwoStepBGIncumbentOriginalRiskLaneRollout(options, "untouched-child", sample_count=1)
    cached = TwoStepBGIncumbentOriginalRiskLaneRollout(options, "untouched-child", sample_count=1)
    reuse = InnerDecisionReuse(cached)
    context = (options, (1,), (1000.,), 1000., 0, ((1.,),), (1,))
    initial = semantic(native)
    first = native(*context)
    assert first.recourse_calls == 0 and cached(*context) == first
    for planner in (native, cached):
        planner.inner_planner(options, (2,), (1001., 1002.), 1001., 0, ((1., 0.),), (2,))
        child = semantic(planner.inner_planner)
        restore(planner, {**initial, "child": child})
    child_before = semantic(cached.inner_planner)
    assert cached(*context) == native(*context)
    assert semantic(cached) == semantic(native) and semantic(cached.inner_planner) == child_before
    assert reuse.work["current_cache_hits"] == 1


def test_nested_first_second_calls_and_fifo_scores_match_native_without_changing_outer_rng():
    native = TwoStepCurrentPolicyBGIncumbentLaneRollout(OPTIONS, "nested-exact", sample_count=2, event_cap=8)
    fast = ExactReuseTwoStepCurrentPolicyBGIncumbentLaneRollout(OPTIONS, "nested-exact", sample_count=2, event_cap=8)
    for planner in (native, fast):
        planner.arrival_rates = (.004, .002, .002)
        planner._outer_observed_counts = (4, 2, 2)
        planner.decision_counter = 7
    args = ((2, 1, 1), (1000.1, 1005.3), 1000.1, 0,
            ((2.1, .3), (1.2,), (.1,)),
            (PlanningArrivals(((.2, 2), (.9, 1)), False, 0, 0),) * 2)
    before = semantic(fast)
    expected = native.score_background_roots(*args)
    actual = fast.score_background_roots(*args)
    assert actual == expected and semantic(fast) == semantic(native)
    assert fast.rng.getstate() == before["rng"] and fast.decision_counter == 7
    assert fast._two_step_work["recourse_first_calls"] > 0 and fast._two_step_work["recourse_second_calls"] > 0
    assert fast._inner_decision_reuse.work["current_cache_hits"] > 0
    assert fast._recourse_work == native._recourse_work and fast._inner_current_work == native._inner_current_work


def test_complete_inner_overflow_cache_retains_bg_and_exact_partial_logical_work(monkeypatch):
    fixed_suffix_gaps(monkeypatch, .1)
    options = (FrameOption("single", (1,), 1., ((1.,),)),
               FrameOption("pair", (2,), 1.5, ((1.5, 1.5),)),
               FrameOption("triple", (3,), 2., ((2., 2., 2.),)),
               FrameOption("loaded-pair", (2,), 2., ((.1, .1),), loaded=True))
    native = TwoStepBGIncumbentOriginalRiskLaneRollout(options, "cached-overflow", sample_count=1, event_cap=2)
    cached = TwoStepBGIncumbentOriginalRiskLaneRollout(options, "cached-overflow", sample_count=1, event_cap=2)
    reuse = InnerDecisionReuse(cached)
    initial = semantic(native)
    context = (options, (4,), (1000., 1005.), 1000., 0, ((10., 0., 0., 0.),), (4,))
    first = None
    for _ in range(2):
        restore(native, initial)
        restore(cached, initial)
        expected, actual = native(*context), cached(*context)
        assert actual == expected and semantic(cached) == semantic(native)
        assert actual.background_rollout_overflow and not actual.incumbent_background_rollout_overflow
        assert actual.selection.option_indices == (actual.incumbent_root_index,) == (3,)
        assert actual.incumbent_maxweight_root_index == 2 and actual.flow_evaluations > 0
        first = actual
    assert reuse.work["current_cache_hits"] == 1
    assert reuse.saved_work["current"]["background_planning_events"] == first.background_planning_events
    assert reuse.saved_work["current"]["recourse_rollout_option_probes"] == first.recourse_rollout_option_probes > 0


def test_small_native_multiclass_loaded_trajectory_and_cumulative_actual_ledgers_are_exact():
    model = FrameModel("toy", "exact-full", 2, ("a", "b", "c"), (1., 1., 1.), OPTIONS)
    tape = [(0., 0, 0), (0., 0, 1), (0., 1, 0), (0., 2, 0),
            (3., 0, 2), (3., 1, 1), (3., 2, 1), (6., 0, 3), (8., 2, 2)]
    native = TwoStepCurrentPolicyBGIncumbentLaneRollout(OPTIONS, "full-exact", sample_count=2, event_cap=8)
    fast = ExactReuseTwoStepCurrentPolicyBGIncumbentLaneRollout(OPTIONS, "full-exact", sample_count=2, event_cap=8)
    deltas = {"mw_reuse_work": Counter(), "inner_decision_reuse_work": Counter(), "kernel_work": Counter()}

    def select(*context):
        decision = fast(*context)
        for group, values in fast.last_reuse_work.items():
            for name, value in values.items():
                if name.endswith("cache_peak_entries"):
                    deltas[group][name] = max(deltas[group][name], value)
                else:
                    deltas[group][name] += value
        return decision

    expected = simulate_observed_rate_contextual_trajectory(model, tape, 12., 16., native, policy="exact")
    actual = simulate_observed_rate_contextual_trajectory(model, tape, 12., 16., select, policy="exact")
    assert actual == expected and semantic(fast) == semantic(native)
    assert any(3 in frame["option_indices"] for frame in actual["frames"])
    for group, values in fast.reuse_work.items():
        assert deltas[group] == Counter(values)
    mw, inner = fast.reuse_work["mw_reuse_work"], fast.reuse_work["inner_decision_reuse_work"]
    assert mw["cache_hits"] > 0 and mw["actual_option_probes"] < mw["logical_option_probes"]
    assert inner["current_cache_hits"] > 0 and inner["BG_cache_hits"] > 0
    assert inner["current_choose_calls"] == inner["current_native_choose_calls"] + inner["current_cache_hits"]
    assert inner["BG_choose_calls"] == inner["BG_native_choose_calls"] + inner["BG_cache_hits"]
    assert inner["current_cache_peak_entries"] <= 2 * fast.sample_count * len(OPTIONS)
    assert inner["BG_cache_peak_entries"] <= 2 * fast.sample_count * len(OPTIONS)
    assert not fast._mw_reuse.cache
