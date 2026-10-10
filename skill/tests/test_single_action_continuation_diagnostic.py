from algorithm.continuous_arrival_age_tail_lane_rollout import ContinuousArrivalAgeTailLaneRollout
from algorithm.experiments.aligned_hardware_frame_model import FrameOption
from algorithm.experiments.single_action_continuation_diagnostic import MWContinuation

OPTIONS = (FrameOption("a1", (1, 0), 2., ((2.,), ())),
           FrameOption("a2", (2, 0), 3., ((3., 3.), ())),
           FrameOption("b1", (0, 1), 1., ((), (1.,))))


def test_original_first_then_MW_never_replans_later():
    selector = MWContinuation(ContinuousArrivalAgeTailLaneRollout(OPTIONS, "same"), True)
    state = (OPTIONS, (2, 1), (1000., 1003.), 1000., 0, ((0., 0.), (0.,)), (100, 50))
    first = selector(*state)
    assert first.eligible and first.planning_random_draws > 0
    after_first = selector.planner.rng.getstate()
    second = selector(*state)
    assert second == selector.planner.maxweight(*state)
    assert selector.planner.rng.getstate() == after_first
    assert selector.work["proposal_calls"] == 1 and selector.work["MW_calls"] == 1


def test_MW_first_consumes_identical_proposal_draws_and_is_scanned_separately():
    original = MWContinuation(ContinuousArrivalAgeTailLaneRollout(OPTIONS, "same"), True)
    replacement = MWContinuation(ContinuousArrivalAgeTailLaneRollout(OPTIONS, "same"), False)
    state = (OPTIONS, (2, 1), (1000., 1003.), 1000., 0, ((0., 0.), (0.,)), (100, 50))
    original(*state)
    forced = replacement(*state)
    assert original.planner.rng.getstate() == replacement.planner.rng.getstate()
    assert forced == replacement.planner.maxweight(*state)
    assert replacement.work["root_option_probes"] == original.work["root_option_probes"] + len(OPTIONS)
