"""The actual unit462/frame16 strict Fraction comparison rounds to a float tie."""
from copy import deepcopy
import unittest

from algorithm.experiments.bg_planning_validation import exact_certificate_matches


class ExactPlanningValidationTest(unittest.TestCase):
    def test_real_serialized_tie_needs_proved_strict_improvement(self):
        certificate = dict(frame_index=16, start=19.79592588210709, baseline_root=7, actual_root=2,
            baseline_S1_sum=dict(numerator=1966927551922117576979, denominator=1152921504606846976),
            actual_S1_sum=dict(numerator=1966927551922117568787, denominator=1152921504606846976),
            strict_S1_improvement_sum=dict(numerator=1, denominator=140737488355328),
            baseline_S2_sum=dict(numerator=46863925164538981369930195025569020047913,
                                 denominator=1329227995784915872903807060280344576),
            actual_S2_sum=dict(numerator=46863925164538979940797904426982637343273,
                               denominator=1329227995784915872903807060280344576),
            S2_nonincrease_sum=dict(numerator=174454625317210251795,
                                    denominator=162259276829213363391578010288128),
            rounded_scores=dict(baseline_predicted_completion_s=213.25471249155547,
                chosen_predicted_completion_s=213.25471249155547,
                baseline_predicted_squared_flow_s2=4407.062343061921,
                chosen_predicted_squared_flow_s2=4407.062343061921),
            exact_S1_strict_improvement=True, exact_S2_nonincrease=True,
            actual_exact_rule_root_matched=True, target_work_metadata_matched=True)
        frame = dict(start=certificate["start"], option_indices=[2],
            rollout=dict(sample_count=8, changed=True, **certificate["rounded_scores"]))
        info = frame["rollout"]
        self.assertFalse(info["chosen_predicted_completion_s"] < info["baseline_predicted_completion_s"])
        self.assertTrue(exact_certificate_matches(certificate, frame))
        nonimprovement = deepcopy(certificate)
        nonimprovement["actual_S1_sum"] = nonimprovement["baseline_S1_sum"]
        nonimprovement["strict_S1_improvement_sum"] = dict(numerator=0, denominator=1)
        self.assertFalse(exact_certificate_matches(nonimprovement, frame))


if __name__ == "__main__":
    unittest.main()
