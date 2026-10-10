"""Direct one-lane controls: feasible service, honest support claims and FIFO."""
from fractions import Fraction
from functools import partial

import pytest

from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption
from algorithm.experiments.aligned_hardware_shared_threshold_scaling_screen import (
    COUNTERS, check_selection,
)
from algorithm.experiments.lane_refill_performance_replay import (
    simulate_lane_refill_trajectory,
)
from algorithm.lane_refill_simple_controls import select_single_lane_control


BUDGET = 64


def choose(options, queue, *, policy="enum_maxweight", expected=None):
    selected = select_single_lane_control(options, queue, 1, BUDGET, policy=policy)
    assert not any(check_selection(selected, options, queue, 1, BUDGET,
                                   expected=expected).values())
    return selected


def test_complete_loaded_and_batch_options_require_all_admitted_jobs():
    options = (
        FrameOption("single", (1, 0), 3., ((3.,), ())),
        FrameOption("batch", (2, 0), 4., ((4., 4.), ())),
        FrameOption("loaded", (1, 1), 2., ((1.,), (2.,)), loaded=True),
    )
    loaded = choose(options, (2, 1), expected=(-Fraction(3, 2), Fraction(2), (2,)))
    batch = choose(options, (2, 0), expected=(-Fraction(1), Fraction(4), (1,)))
    assert loaded.option_indices == (2,) and loaded.service == (1, 1)
    assert batch.option_indices == (1,) and batch.service == (2, 0)
    assert loaded.exact and batch.exact


def test_delay_counts_resident_completion_instead_of_only_early_target():
    options = (
        FrameOption("loaded", (1, 1), 10., ((.2,), (10.,)), loaded=True),
        FrameOption("single", (1, 0), 3., ((3.,), ())),
    )
    selected = choose(options, (1, 1), policy="delay")
    # Loaded target finishes at .2, but the two admitted jobs average 5.1.
    assert selected.option_indices == (1,)
    assert selected.service == (1, 0)
    assert selected.canonical_exact


def test_short_mean_completion_and_maximum_support_are_distinct_objectives():
    options = (
        FrameOption("long_batch", (4, 0), 3., ((3.,) * 4, ())),
        FrameOption("short", (0, 1), 1., ((), (1.,))),
    )
    expected = (-Fraction(16, 3), Fraction(3), (0,))
    maximum = choose(options, (4, 1), expected=expected)
    delay = choose(options, (4, 1), policy="delay", expected=expected)
    assert maximum.option_indices == (0,)
    assert delay.option_indices == (1,)
    assert not delay.score_exact and not delay.canonical_exact and not delay.exact
    assert delay.upper_bound >= float(Fraction(16, 3))
    assert delay.gap > 0


def test_fixed_p1_excludes_batches_and_loaded_service_without_false_full_oracle():
    options = (
        FrameOption("p1", (1, 0), 4., ((4.,), ())),
        FrameOption("batch", (3, 0), 6., ((6.,) * 3, ())),
        FrameOption("loaded", (1, 1), 2., ((1.,), (2.,)), loaded=True),
    )
    expected = (-Fraction(2), Fraction(2), (2,))
    restricted = choose(options, (3, 1), policy="fixed_p1", expected=expected)
    assert restricted.option_indices == (0,)
    assert not restricted.score_exact and not restricted.canonical_exact
    assert choose(options, (1, 0), policy="fixed_p1").canonical_exact


def test_idle_and_canonical_ties_use_effective_duration_floor_and_index():
    options = (
        FrameOption("first", (1,), .8, ((.8,),)),
        FrameOption("second", (1,), .2, ((.2,),)),
    )
    for policy in ("enum_maxweight", "delay", "fixed_p1"):
        idle = choose(options, (0,), policy=policy,
                      expected=(Fraction(0), Fraction(1), (2,)))
        assert idle.option_indices == (None,) and idle.duration == 1.
        assert idle.service == (0,) and idle.exact
    maximum = choose(options, (2,), expected=(-Fraction(2), Fraction(1), (0,)))
    assert maximum.option_indices == (0,) and maximum.duration == 1.
    # Delay chooses the .2 completion even though canonical support ties use index.
    delay = choose(options, (2,), policy="delay")
    assert delay.option_indices == (1,)
    assert delay.score_exact and not delay.canonical_exact and not delay.exact


def test_equal_delay_uses_support_then_duration_then_original_index():
    options = (
        FrameOption("batch", (2, 0), 4., ((1., 3.), ())),
        FrameOption("shorter", (1, 0), 2., ((2.,), ())),
        FrameOption("less_support", (0, 1), 2., ((), (2.,))),
        FrameOption("duplicate", (1, 0), 2., ((2.,), ())),
    )
    selected = choose(options, (2, 1), policy="delay")
    assert selected.option_indices == (1,)
    stronger = options + (FrameOption("higher_support", (2, 0), 2.,
                                     ((2., 2.), ())),)
    assert choose(stronger, (2, 1), policy="delay").option_indices == (4,)


def test_direct_controls_scan_once_and_use_no_lp_or_integer_search():
    class OnePassCatalogue:
        def __init__(self, values):
            self.values, self.iterations = values, 0

        def __len__(self):
            return len(self.values)

        def __iter__(self):
            self.iterations += 1
            assert self.iterations == 1, "control scanned catalogue more than once"
            return iter(self.values)

    values = (FrameOption("p1", (1,), 2., ((2.,),)),
              FrameOption("batch", (2,), 3., ((3., 3.),)))
    for policy in ("enum_maxweight", "delay", "fixed_p1"):
        for waiting, candidates in ((1, 2), (2, 3)):
            catalogue = OnePassCatalogue(values)
            selected = select_single_lane_control(catalogue, (waiting,), 1, BUDGET,
                                                  policy=policy)
            assert catalogue.iterations == 1
            assert selected.option_probes == len(values)
            assert selected.candidate_evaluations == candidates  # feasible + idle
            for counter in COUNTERS:
                if counter != "candidate_evaluations":
                    assert getattr(selected, counter) == 0
            assert getattr(selected, "integer_states_visited", 0) == 0
        with pytest.raises(ValueError, match="one|single|1"):
            select_single_lane_control(values, (2,), 2, BUDGET, policy=policy)


def test_real_fifo_replay_uses_only_released_prefix_and_keeps_censored_tails():
    option = FrameOption("batch", (2,), 10., ((8., 2.),))
    model = FrameModel("small", "controls", 1, ("a",), (1.,), (option,))
    selector = partial(select_single_lane_control, policy="enum_maxweight")

    def run(tape):
        return simulate_lane_refill_trajectory(
            model, tape, 2., 4., BUDGET, selector,
            policy="enum_maxweight", dispatch_rule="single")[0]

    first = run(((0., 0, 0), (0., 0, 1), (0., 0, 2), (1., 0, 3)))
    second = run(((0., 0, 0), (0., 0, 1), (0., 0, 2), (1.5, 0, 3)))
    first_action = lambda result: (result["frames"][0]["queue"],
                                   result["frames"][0]["option_indices"])
    assert first_action(first) == first_action(second) == ((3,), (0,))
    assert [job["completion"] for job in first["jobs"]] == [2., 8., None, None]
    assert first["jobs"][2]["start"] is None
    snapshot = first["snapshots"]["observation_end"]
    assert (snapshot["completed_jobs"], snapshot["inflight_jobs"],
            snapshot["queued_jobs"], snapshot["unfinished_jobs"]) == (1, 1, 2, 3)
    assert snapshot["restricted_mean_flow_s"] == pytest.approx(13 / 4)
    assert first["occupied_lane_time_s"] == 4.
    assert first["waiting_area_s"] == 7.
