"""Tick suffix clocks must preserve native populations, RNG, scores, and fees."""
from collections import Counter
from fractions import Fraction
from functools import lru_cache
import gzip
import json
from pathlib import Path
import pickle
from types import MethodType
from unittest.mock import patch

from algorithm.arrival_aware_lane_rollout import PlanningArrivals, sample_planning_arrivals
import algorithm.background_competition_continuous_lane_rollout as background
from algorithm.exact_reuse_lane_rollout import ExactReuseBackgroundCompetitionContinuousLaneRollout
from algorithm.experiments.aligned_hardware_frame_model import FrameOption
from algorithm.experiments.aligned_hardware_performance_necessity import load_models
import algorithm.tick_background_rollout as tick


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
STEM = "aligned_hardware_persistent_current_policy_development_20261002"
OPTIONS = (
    FrameOption("a1", (1, 0, 0), 1., ((1.,), (), ())),
    FrameOption("b1", (0, 1, 0), 1.25, ((), (1.25,), ())),
    FrameOption("c1", (0, 0, 1), 1.5, ((), (), (1.5,))),
    FrameOption("loaded-ab", (1, 1, 0), 1.75, ((.5,), (1.75,), ()), loaded=True),
    FrameOption("a2", (2, 0, 0), 1.5, ((1.5, 1.5), (), ())),
)


def semantic(planner):
    return (planner.rng.getstate(), planner.decision_counter, planner.arrival_rates,
        getattr(planner, "_outer_observed_counts", None),
        semantic(planner.inner_planner) if hasattr(planner, "inner_planner") else None)


class FixedGapRandom:
    def __init__(self, seed):
        pass

    def expovariate(self, rate):
        return .4


def test_shared_suffix_releases_keep_float_recurrence_rng_and_nonmonotone_root_clock():
    scale = 2**55
    native = background._LazyBackgroundArrivals((.5, .4, 0.), 1.75, "tick-shared", 64, 2)
    work = Counter()
    fast = tick._TickBackgroundArrivals((.5, .4, 0.), 1.75, 7 * scale // 4,
        "tick-shared", 64, 2, scale, work)
    # A later root/sample may revisit an earlier release clock after events exist.
    for start in (1.75, 2.5, 6., 2., 3., 8., 4.):
        assert native.release_through(Fraction.from_float(start)) == fast.release_through(int(start * scale))
        assert [(Fraction(birth, scale), c) for birth, c in fast.events] == native.events
        for name in ("random_draws", "rate_reads", "seed_initializations", "overflow"):
            assert getattr(fast, name) == getattr(native, name), name
        assert (fast.rng.getstate() if fast.rng else None) == (native.rng.getstate() if native.rng else None)
        assert fast.next_times == native.next_times
    assert fast.events and work["background_tick_conversions"] == fast.random_draws
    # Equal class clocks retain the native coordinate tie before the next draw.
    with patch.object(background, "Random", FixedGapRandom), patch.object(tick, "Random", FixedGapRandom):
        native = background._LazyBackgroundArrivals((.5, .4), 1.75, "tie", 64, 0)
        fast = tick._TickBackgroundArrivals((.5, .4), 1.75, 7 * scale // 4,
            "tie", 64, 0, scale, Counter())
        assert native.release_through(Fraction(3)) == fast.release_through(3 * scale)
        assert [(Fraction(birth, scale), c) for birth, c in fast.events] == native.events
        assert [c for _, c in fast.events] == [0, 1, 0, 1, 0, 1]


def compare_leaf(options, seed, cap, rates, counter, context):
    native = background.BackgroundCompetitionContinuousLaneRollout(options, seed, 8, cap)
    dyadic = ExactReuseBackgroundCompetitionContinuousLaneRollout(options, seed, 8, cap)
    fast = ExactReuseBackgroundCompetitionContinuousLaneRollout(options, seed, 8, cap)
    fast.score_background_roots = MethodType(tick.score_background_roots, fast)
    for planner in (native, dyadic, fast):
        planner.arrival_rates, planner.decision_counter = rates, counter
    before = semantic(native)
    expected = native.score_background_roots(*context)
    assert dyadic.score_background_roots(*context) == expected
    actual = fast.score_background_roots(*context)
    assert actual == expected
    assert semantic(native) == semantic(dyadic) == semantic(fast) == before
    for name in ("rollout_calls", "flow_evaluations", "admitted_job_reads", "background_planning_events",
                 "background_planning_random_draws", "background_planning_rate_reads", "background_seed_initializations"):
        assert fast.kernel_work[name] == getattr(actual, name), name
    assert fast.kernel_work["background_tick_conversions"] == actual.background_planning_random_draws
    assert fast.kernel_work["window_tick_conversions"] == 1
    for name, value in dyadic.kernel_work.items():
        if name != "background_tick_conversions":
            assert fast.kernel_work[name] == value, name
    return actual


def test_no_arrivals_and_delayed_tagged_prefix_keep_idle_fee_and_completion_offsets():
    context = ((2, 1, 1), (10.1, 12.3, 15.5), 10.1, 0,
        ((2.1, .3), (1.2,), (.1,)), (PlanningArrivals((), False, 0, 0),) * 8)
    actual = compare_leaf(OPTIONS, "tick-no-background", 128, (0., 0., 0.), 11, context)
    assert not actual.background_rollout_overflow and actual.background_planning_random_draws == 0
    options = (
        FrameOption("a1", (1, 0), 1., ((1.,), ())),
        FrameOption("b1", (0, 1), 1., ((), (1.,))),
        FrameOption("loaded-ab", (1, 1), 10., ((.5,), (10.,)), loaded=True),
    )
    context = ((1, 0), (1000., 1015.), 1000., 0, ((.75,), ()),
        (PlanningArrivals(((9.2, 0),), False, 0, 0),) * 8)
    actual = compare_leaf(options, "tick-idle", 128, (.001, 0.), 7, context)
    assert actual.rollout_idle_calls > 0 and actual.background_planning_events == 0
    assert actual.costs[0] == 8 * (Fraction(7, 4) + Fraction(11) - Fraction.from_float(9.2))


def test_prefix_and_suffix_cap_keep_partial_s1_s2_and_every_native_fee():
    class DenseRandom(FixedGapRandom):
        def expovariate(self, rate):
            return .1

    context = ((2, 1, 1), (10.1, 12.3, 15.5), 10.1, 0,
        ((2.1, .3), (1.2,), (.1,)),
        (PlanningArrivals(((.2, 2), (.9, 1)), False, 0, 0),) * 8)
    with patch.object(background, "Random", DenseRandom), patch.object(tick, "Random", DenseRandom):
        actual = compare_leaf(OPTIONS, "tick-cap-overflow", 2, (.5, .4, .3), 11, context)
    assert actual.background_rollout_overflow
    assert actual.background_planning_events == 1
    assert 0 < actual.flow_evaluations < actual.candidate_count * 8 * 6
    assert any(value > 0 for value in actual.costs.values())
    assert any(value == 0 for value in actual.costs.values())
    assert all(isinstance(value, Fraction) for value in (*actual.costs.values(), *actual.squared_costs.values()))


@lru_cache(maxsize=1)
def frozen_protocol():
    return json.loads((ARTIFACTS / (STEM + "_protocol.json")).read_text())


def trajectory_for(index):
    document = json.loads((ARTIFACTS / (STEM + f"_shard_{index:02d}.json")).read_text())
    with gzip.open(document["trajectory_file"], "rt") as source:
        return json.load(source)["runs"][0]["policies"]["arrival_rollout"]


def prefix_for_frame(model, tape, frames, wanted):
    from random import Random
    rng = Random(f"arrival-plan-v1:{model.node}:{tape['load']}:online")
    protocol = frozen_protocol()
    for index, frame in enumerate(frames[:wanted + 1]):
        rates = tuple(value / frame["start"] if frame["start"] else 0.
            for value in frame["observed_arrival_counts"])
        paths = []
        if frame["rollout"]["eligible"]:
            for _ in range(protocol["planning_samples"]):
                paths.append(sample_planning_arrivals(rates, max(option.duration for option in model.options),
                    rng, protocol["planning_event_cap"]))
                if paths[-1].overflow:
                    break
        if index == wanted:
            assert sum(len(path.events) for path in paths) == frame["rollout"]["tagged_planning_events"]
            return tuple(paths), rng.getstate(), rates


def test_four_real_catalogues_keep_actual_frame_populations_and_budget128():
    protocol = frozen_protocol()
    models = load_models(protocol["models"])
    assert len(models) == 4 and protocol["planning_samples"] == 8 and protocol["planning_event_cap"] == 128
    for model in models:
        index, tape = next((i, tape) for i, tape in enumerate(protocol["tapes"])
            if tape["node"] == model.node and (ARTIFACTS / (STEM + f"_shard_{i:02d}.json")).is_file())
        frames = trajectory_for(index)["frames"]
        wanted = next(i for i, frame in enumerate(frames) if frame["start"] > 0.
            and frame["rollout"]["candidate_count"] > 1 and not frame["rollout"]["planning_overflow"])
        frame = frames[wanted]
        paths, _, rates = prefix_for_frame(model, tape, frames, wanted)
        assert len(paths) == 8 and not any(path.overflow for path in paths)
        context = (tuple(frame["queue"]), tuple(frame["available_before"]), frame["start"],
            frame["lane_ids"][0], tuple(tuple(group) for group in frame["waiting_ages"]), paths)
        compare_leaf(model.options, f"arrival-plan-v1:{model.node}:{tape['load']}:online",
            128, rates, wanted, context)


def test_profiled_heavy_nested_chunk_keeps_full_native_work_costs_and_controller_states():
    from algorithm import parallel_persistent_current_policy_lane_rollout as original
    from algorithm import tick_persistent_current_policy_lane_rollout as optimized
    with (ARTIFACTS / "aligned_hardware_tick_background_rollout_20261002_heavy_payload.pkl").open("rb") as source:
        fixture = pickle.load(source)
    assert fixture["samples"] == 8 and fixture["cap"] == 128
    assert fixture["payload"]["root"] == 4 and fixture["payload"]["sample_start"] == 0
    assert len(fixture["payload"]["paths"]) == 2
    outputs, states = {}, {}
    for label, module in (("original", original), ("optimized", optimized)):
        module._initialize_worker(fixture["options"], fixture["seed"], fixture["samples"], fixture["cap"])
        outputs[label] = module._run_root(fixture["payload"])
        states[label] = semantic(module._run_root.__globals__["_worker_planner"])
    ignored = {"actual_reuse_work", "worker_cpu_s", "worker_wall_s", "constructor_cpu_s", "constructor_wall_s"}
    assert {key: value for key, value in outputs["original"].items() if key not in ignored} == {
        key: value for key, value in outputs["optimized"].items() if key not in ignored}
    assert states["original"] == states["optimized"]
    actual = outputs["optimized"]
    assert isinstance(actual["cost"], Fraction) and isinstance(actual["squared"], Fraction)
    assert actual["recourse_work"]["recourse_calls"] == 35
    assert actual["continuation_work"]["late_current_calls"] == 31
    original_kernel = outputs["original"]["actual_reuse_work"]["kernel_work"]
    optimized_kernel = actual["actual_reuse_work"]["kernel_work"]
    # Equal scores also need to demonstrate that the new leaf was executed.
    assert optimized_kernel["window_tick_conversions"] == optimized_kernel["scorer_calls"] > 0
    assert "window_tick_conversions" not in original_kernel
    assert optimized_kernel["background_tick_conversions"] == optimized_kernel["background_planning_random_draws"]


if __name__ == "__main__":
    test_shared_suffix_releases_keep_float_recurrence_rng_and_nonmonotone_root_clock()
    test_no_arrivals_and_delayed_tagged_prefix_keep_idle_fee_and_completion_offsets()
    test_prefix_and_suffix_cap_keep_partial_s1_s2_and_every_native_fee()
    test_four_real_catalogues_keep_actual_frame_populations_and_budget128()
    test_profiled_heavy_nested_chunk_keeps_full_native_work_costs_and_controller_states()
    print("5 passed", flush=True)
