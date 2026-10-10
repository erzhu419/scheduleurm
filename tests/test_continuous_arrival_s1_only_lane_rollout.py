"""Small real-scorer fixtures for removing only the S2 selection constraint."""
import unittest

from algorithm.continuous_arrival_age_tail_lane_rollout import ContinuousArrivalAgeTailLaneRollout
from algorithm.continuous_arrival_s1_only_lane_rollout import ContinuousArrivalS1OnlyLaneRollout
from algorithm.experiments.aligned_hardware_frame_model import FrameOption


OPTIONS = (
    FrameOption("A", (1, 0), 2.0, ((2.0,), ())),
    FrameOption("B", (0, 1), 6.0, ((), (6.0,))),
    FrameOption("loaded", (1, 1), 10.0, ((1.0,), (8.5,)), loaded=True),
)
CONTEXT = (OPTIONS, (1, 1), (1000.0,), 1000.0, 0, ((0.0,), (100.0,)), (1, 1))


class ContinuousArrivalS1OnlyTest(unittest.TestCase):
    def test_s1_best_excluded_by_s2_is_selected_and_charged(self):
        old = ContinuousArrivalAgeTailLaneRollout(OPTIONS, 41)
        new = ContinuousArrivalS1OnlyLaneRollout(OPTIONS, 41)
        before, after = old(*CONTEXT), new(*CONTEXT)
        self.assertEqual(before.selection.option_indices, (0,))
        self.assertEqual(before.unconstrained_root_index, 2)
        self.assertEqual(after.selection.option_indices, (2,))
        self.assertLess(after.chosen_predicted_completion_s, after.baseline_predicted_completion_s)
        self.assertTrue(after.chosen_exceeds_MW_S2)
        self.assertEqual(after.diagnostic_S2_comparisons, 1)
        self.assertEqual(after.root_ordering_comparisons, after.candidate_count - 1)
        self.assertEqual(after.strict_improvement_comparisons, 1)
        self.assertEqual(after.risk_filtered_candidates, 0)
        self.assertEqual(after.risk_feasible_candidates, after.candidate_count)
        self.assertGreater(after.selection.gap, 0)
        for name in ("sample_count", "planning_events", "planning_random_draws", "planning_rate_reads",
                     "rollout_calls", "rollout_option_probes", "waiting_age_reads", "flow_evaluations"):
            self.assertEqual(getattr(after, name), getattr(before, name))
        self.assertEqual(new.rng.getstate(), old.rng.getstate())
        tied = (OPTIONS[0], FrameOption("A_tie", (1, 0), 2.0, ((2.0,), ())), OPTIONS[1])
        tie = ContinuousArrivalS1OnlyLaneRollout(tied, 41)(
            tied, (1, 0), (1000.0,), 1000.0, 0, ((10.0,), ()), (1, 0))
        self.assertEqual(tie.selection.option_indices, (0,))
        self.assertFalse(tie.changed)
        self.assertFalse(tie.chosen_exceeds_MW_S2)

    def test_constant_visible_age_shift_preserves_all_s1_rankings(self):
        planner = ContinuousArrivalS1OnlyLaneRollout(OPTIONS, 41)
        paths = ((),) * 8
        original = planner.score_age_roots(*CONTEXT[1:6], paths)
        shifted_context = (*CONTEXT[:5], ((25.0,), (125.0,)), CONTEXT[-1])
        shifted = planner.score_age_roots(*shifted_context[1:6], paths)
        self.assertEqual(original.candidate_count, shifted.candidate_count)
        self.assertEqual(shifted.costs, {root: value + 400 for root, value in original.costs.items()})
        first = ContinuousArrivalS1OnlyLaneRollout(OPTIONS, 41)(*CONTEXT)
        second = ContinuousArrivalS1OnlyLaneRollout(OPTIONS, 41)(*shifted_context)
        self.assertEqual(first.selection.option_indices, second.selection.option_indices)
        self.assertEqual(second.chosen_predicted_completion_s - first.chosen_predicted_completion_s, 50.0)

    def test_overflow_falls_back_whole_decision_with_identical_rng(self):
        old = ContinuousArrivalAgeTailLaneRollout(OPTIONS, 41, event_cap=1)
        new = ContinuousArrivalS1OnlyLaneRollout(OPTIONS, 41, event_cap=1)
        context = (*CONTEXT[:-1], (100000, 100000))
        before, after = old(*context), new(*context)
        self.assertTrue(after.planning_overflow)
        self.assertEqual(after.selection, before.selection)
        self.assertFalse(after.changed)
        self.assertEqual(after.rollout_calls, 0)
        self.assertEqual(after.diagnostic_S2_comparisons, 0)
        self.assertEqual(after.root_ordering_comparisons, 0)
        self.assertEqual(after.strict_improvement_comparisons, 0)
        self.assertEqual(after.planning_random_draws, before.planning_random_draws)
        self.assertEqual(new.rng.getstate(), old.rng.getstate())
        outside = (OPTIONS, (3, 0), (1000.0,), 1000.0, 0, ((30.0, 20.0, 10.0), ()), (3, 0))
        old = ContinuousArrivalAgeTailLaneRollout(OPTIONS, 41)
        new = ContinuousArrivalS1OnlyLaneRollout(OPTIONS, 41)
        initial_rng = new.rng.getstate()
        before, after = old(*outside), new(*outside)
        self.assertFalse(after.eligible)
        self.assertEqual(after.selection, before.selection)
        self.assertEqual(new.rng.getstate(), initial_rng)


if __name__ == "__main__":
    unittest.main()
