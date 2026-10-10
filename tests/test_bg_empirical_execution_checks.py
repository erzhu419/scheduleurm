"""Actual float-tie evidence and measured-service execution checks."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import unittest

from algorithm.background_competition_continuous_lane_rollout import BackgroundCompetitionContinuousLaneRollout
from algorithm.experiments.aligned_hardware_performance_necessity import load_models
from algorithm.experiments.bg_empirical_execution_checks import (
    ExactRecordedBG, check_empirical_execution, check_recorded_bg, exact_scores_match,
)
from algorithm.experiments.empirical_service_lane_replay import (
    EmpiricalServiceTape, MeanService, simulate_empirical_service_trajectory,
)


ART = Path(__file__).resolve().parents[1] / "md/experiment_artifacts"


class EmpiricalExecutionChecksTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bank = json.loads((ART / "aligned_hardware_BG_empirical_service_samples_20261004.json").read_text())
        cls.models = load_models(bank["mean_model_snapshots"])
        cls.catalogues = {row["node"]: row for row in bank["catalogues"]}
        original = json.loads((ART / "aligned_hardware_background_competition_continuous_confirmation_20260930_protocol.json").read_text())
        cls.tapes = {model.node: tuple(tuple(birth) for birth in next(tape for tape in original["tapes"]
            if tape["node"] == model.node and tape["load"] == .8 and tape["seed"] == 100001)["arrivals"] if birth[0] < 40)
            for model in cls.models}
        cls.protocol = dict(environment_arrival_end=40, observation_end=80, planning_event_cap=128, planning_samples=8)

    def test_exact_telemetry_preserves_actual_bg_decisions_and_rng(self):
        for model in self.models:
            with self.subTest(node=model.node):
                before = BackgroundCompetitionContinuousLaneRollout(model.options, "exact-trace-check", 8, 128)
                after = ExactRecordedBG(model.options, "exact-trace-check", 8, 128)
                old = simulate_empirical_service_trajectory(model, self.tapes[model.node], 40, 80, before, MeanService(), policy="BG")
                new = simulate_empirical_service_trajectory(model, self.tapes[model.node], 40, 80, after, MeanService(), policy="BG")
                self.assertEqual(old["jobs"], new["jobs"])
                for original, actual in zip(old["frames"], new["frames"], strict=True):
                    record = deepcopy(actual)
                    record["rollout"].pop("exact_root_scores")
                    self.assertEqual(original, record)
                self.assertEqual(before.rng.getstate(), after.rng.getstate())
                self.assertTrue(check_recorded_bg(new, model, self.protocol, after))

    def test_real_unit462_float_tie_requires_strict_rational_improvement(self):
        proof = json.loads((ART / "aligned_hardware_BG_planning_streams_20261004_runner_unit462_exact_fraction_proof.json").read_text())
        certificate, = proof["certificates"]
        scores = {key: [certificate[key]["numerator"], certificate[key]["denominator"]]
            for key in ("baseline_S1_sum", "actual_S1_sum", "baseline_S2_sum", "actual_S2_sum")}
        scores.update(baseline_root=certificate["baseline_root"], actual_root=certificate["actual_root"])
        frame = dict(option_indices=[certificate["actual_root"]], rollout=dict(
            **certificate["rounded_scores"], sample_count=8, changed=True, exact_root_scores=scores))
        self.assertEqual(frame["rollout"]["baseline_predicted_completion_s"], frame["rollout"]["chosen_predicted_completion_s"])
        self.assertTrue(exact_scores_match(frame))
        frame["rollout"]["exact_root_scores"]["actual_S1_sum"] = scores["baseline_S1_sum"]
        self.assertFalse(exact_scores_match(frame))

    def test_saved_clock_samples_and_causal_forecasts_are_reconstructed(self):
        for model in self.models:
            for condition in ("mean", "empirical"):
                with self.subTest(node=model.node, condition=condition):
                    samples = {action["action_id"]: action["training"] for action in self.catalogues[model.node]["actions"]}
                    service = EmpiricalServiceTape(samples, "execution-check") if condition == "empirical" else MeanService()
                    planner = ExactRecordedBG(model.options, "execution-check-plan", 8, 128)
                    result = simulate_empirical_service_trajectory(model, self.tapes[model.node], 40, 80,
                        planner, service, policy="BG")
                    result.update(condition=condition, service_seed="execution-check")
                    # Check JSON-roundtripped records, as used in the server merge.
                    saved = json.loads(json.dumps(result))
                    self.assertTrue(check_empirical_execution(saved, model, self.protocol, service))
                    self.assertTrue(check_recorded_bg(saved, model, self.protocol, planner))
                    saved["frames"][0]["available_before"][0] += 1
                    self.assertFalse(check_empirical_execution(saved, model, self.protocol, service))


if __name__ == "__main__":
    unittest.main()
