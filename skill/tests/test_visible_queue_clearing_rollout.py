"""Visible clearing costs, lane reservations, bounded work and honest support."""
from fractions import Fraction
from math import nextafter

import pytest

from algorithm.experiments.aligned_hardware_frame_model import FrameOption
from algorithm.lane_refill_simple_controls import select_single_lane_control
from algorithm.visible_queue_clearing_rollout import (
    select_context_maxweight, select_visible_queue_clearing,
)


def ordinary_options():
    return (
        FrameOption("a1", (1, 0), 2., ((2.,), ())),
        FrameOption("a2", (2, 0), 3., ((3., 3.), ())),
        FrameOption("b1", (0, 1), 1., ((), (1.,))),
    )


def test_clearing_counts_remaining_jobs_and_improves_on_batch_root():
    decision = select_visible_queue_clearing(ordinary_options(), (2, 1), (0., 3.), 0., 0)
    # MW a2: two completions at3 and b at4 ->10.
    # Root b1: b at1, then a2 finishes both at4 ->9.
    assert decision.selection.option_indices == (2,)
    assert decision.baseline_predicted_completion_s == 10.
    assert decision.chosen_predicted_completion_s == 9.
    assert decision.changed and decision.eligible
    assert decision.candidate_count == 3 and decision.queue_limit == 4
    assert decision.selection.service == (0, 1)
    assert decision.selection.score == 1.
    assert decision.selection.upper_bound >= float(Fraction(4, 3))
    assert decision.selection.gap > 0
    assert not decision.selection.score_exact
    assert not decision.selection.canonical_exact and not decision.selection.exact


def test_busy_lane_availability_changes_the_cost_and_root():
    early = select_visible_queue_clearing(ordinary_options(), (2, 1), (0., 1.), 0., 0)
    late = select_visible_queue_clearing(ordinary_options(), (2, 1), (0., 3.), 0., 0)
    assert early.selection.option_indices == (1,)
    assert not early.changed and early.baseline_predicted_completion_s == 8.
    assert late.selection.option_indices == (2,) and late.changed


def test_absolute_now_is_removed_from_waiting_completion_objective():
    origin = select_visible_queue_clearing(ordinary_options(), (2, 1), (0., 3.), 0., 0)
    shifted = select_visible_queue_clearing(ordinary_options(), (2, 1), (7., 10.), 7., 0)
    assert shifted == origin


def test_other_free_lane_runs_continuation_at_the_same_time():
    decision = select_visible_queue_clearing(ordinary_options(), (2, 1), (0., 0.), 0., 0)
    assert decision.baseline_predicted_completion_s == 7.
    # Root a1 finishes at2; the other lane serves b at1, then a at3.
    assert decision.chosen_predicted_completion_s == 6.
    assert decision.changed and decision.selection.option_indices == (0,)


def test_loaded_early_target_counts_resident_and_reserves_complete_lane():
    options = (
        FrameOption("loaded", (1, 1), 10., ((.2,), (10.,)), loaded=True),
        FrameOption("a1", (1, 0), 8., ((8.,), ())),
        FrameOption("b1", (0, 1), 8., ((), (8.,))),
    )
    decision = select_visible_queue_clearing(options, (2, 1), (0., 100.), 0., 0)
    # Loaded cost .2+10, and next a cannot start before10; it finishes18.
    assert decision.baseline_predicted_completion_s == pytest.approx(28.2)
    assert not decision.changed and decision.selection.option_indices == (0,)


def test_high_queue_keeps_exact_maxweight_without_rollouts():
    options = ordinary_options()
    decision = select_visible_queue_clearing(options, (4, 1), (0., 3.), 0., 0)
    expected = select_single_lane_control(options, (4, 1), 1, 0)
    assert decision.selection == expected
    assert not decision.eligible and not decision.changed
    assert decision.candidate_count == decision.rollout_calls == decision.rollout_option_probes == 0
    assert decision.root_option_probes == len(options)
    assert decision.baseline_predicted_completion_s is None
    assert decision.chosen_predicted_completion_s is None


def test_empty_queue_returns_baseline_idle_without_prediction():
    decision = select_visible_queue_clearing(ordinary_options(), (0, 0), (0., 3.), 0., 0)
    assert decision.selection.option_indices == (None,)
    assert not decision.eligible and not decision.changed
    assert decision.candidate_count == 0 and decision.rollout_calls == 0


def test_infeasible_root_is_not_a_candidate():
    options = ordinary_options()
    decision = select_visible_queue_clearing(options, (1, 0), (0., 3.), 0., 0)
    assert decision.candidate_count == 1
    assert decision.selection.option_indices == (0,)
    assert decision.baseline_predicted_completion_s == 2.
    assert decision.rollout_calls == 0


def test_actual_calls_and_probes_obey_derived_queue_bound():
    options, queue = ordinary_options(), (2, 1)
    decision = select_visible_queue_clearing(options, queue, (0., 3.), 0., 0)
    assert decision.rollout_calls == 4  # a1 takes2 more calls, a2/b1 take1 each.
    assert decision.rollout_calls <= len(options) * (sum(queue) - 1)
    assert decision.rollout_option_probes == decision.rollout_calls * len(options)
    # Baseline, shared preprocessing and feasible-root enumeration are paid.
    assert decision.root_option_probes == 3 * len(options)


def test_equal_clearing_cost_keeps_the_maxweight_root():
    options = (FrameOption("first", (1,), 1., ((.5,),)),
               FrameOption("second", (1,), 1., ((.5,),)))
    decision = select_visible_queue_clearing(options, (1,), (0.,), 0., 0)
    assert decision.selection.option_indices == (0,) and not decision.changed
    assert decision.selection.score_exact and decision.selection.canonical_exact


def test_fraction_cost_detects_strict_improvement_without_tuned_epsilon():
    options = (FrameOption("first", (1,), 1., ((.5,),)),
               FrameOption("earlier", (1,), 1., ((nextafter(.5, 0.),),)))
    decision = select_visible_queue_clearing(options, (1,), (0.,), 0., 0)
    assert decision.changed and decision.selection.option_indices == (1,)
    assert decision.selection.score_exact
    assert not decision.selection.canonical_exact and not decision.selection.exact
    assert decision.selection.gap == 0.
    assert decision.chosen_predicted_completion_s < decision.baseline_predicted_completion_s


def test_changed_nonbinary_equal_support_keeps_zero_gap_exact_score_certificate():
    options = (FrameOption("first", (1,), 3., ((3.,),)),
               FrameOption("earlier", (1,), 3., ((1.,),)))
    decision = select_visible_queue_clearing(options, (1,), (0.,), 0., 0)
    selected = decision.selection
    assert decision.changed and selected.option_indices == (1,)
    assert selected.score == float(Fraction(1, 3))
    assert selected.score_exact and selected.upper_bound == selected.score
    assert selected.gap == 0.
    assert not selected.canonical_exact and not selected.exact


def test_completion_offsets_do_not_replace_the_one_second_lane_floor():
    options = (FrameOption("short", (1,), .2, ((.2,),)),
               FrameOption("batch", (2,), 3., ((3., 3.),)))
    decision = select_visible_queue_clearing(options, (2,), (0.,), 0., 0)
    assert decision.baseline_predicted_completion_s == pytest.approx(1.4)
    assert decision.selection.duration == 1.
    assert not decision.changed


def test_context_maxweight_uses_one_scan_and_has_no_rollout_claims():
    options = ordinary_options()
    decision = select_context_maxweight(options, (2, 1), (0., 3.), 0., 0)
    assert decision.selection == select_single_lane_control(options, (2, 1), 1, 0)
    assert not decision.eligible and not decision.changed
    assert decision.candidate_count == decision.rollout_calls == decision.rollout_option_probes == 0
    assert decision.root_option_probes == len(options)
    assert decision.queue_limit == 4
    assert decision.baseline_predicted_completion_s is None


def test_dispatch_rejects_a_lane_that_is_still_reserved():
    with pytest.raises(ValueError, match="already be available"):
        select_visible_queue_clearing(ordinary_options(), (2, 1), (0., 3.), 0., 1)
