"""Exercise frozen registration and saved-arm composition without simulations."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

from algorithm.experiments import aligned_hardware_bg_empirical_service_development as runner
from algorithm.experiments.integer_capacity_trajectory_replay import trajectory_snapshot


class EmpiricalDevelopmentRunnerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        artifacts = runner.ROOT / "md/experiment_artifacts"
        names = (runner.STEM + "_design.json", runner.STEM + "_loads.json",
            "aligned_hardware_BG_empirical_service_samples_20261004.json",
            runner.ORIGINAL + "_protocol.json")
        for name in names:
            copyfile(artifacts / name, self.directory / name)
        with redirect_stdout(StringIO()):
            runner.register(self.directory)
        self.protocol = runner.load_protocol(self.directory)

    def tearDown(self):
        self.temporary.cleanup()

    def test_actual_frozen_inputs_register_complete_shared_fixture_roster(self):
        protocol = self.protocol
        self.assertEqual((len(protocol["tapes"]), len(protocol["units"])), (24, 96))
        self.assertEqual(len(list(self.directory.glob(runner.STEM + "_fixture_*.json.gz"))), 24)
        for offset in range(0, 96, 4):
            units = protocol["units"][offset:offset + 4]
            self.assertEqual([(unit["condition"], unit["policy"]) for unit in units],
                [("mean", "maxweight"), ("mean", "BG"), ("empirical", "maxweight"), ("empirical", "BG")])
            self.assertEqual(len({unit["fixture_file"] for unit in units}), 1)
            fixture = runner.read(self.directory / units[0]["fixture_file"])
            self.assertEqual(fixture["registered_at_utc"], protocol["registered_at_utc"])
            self.assertEqual(fixture["tape"]["service_seed"] - fixture["tape"]["seed"], 100000)
            self.assertEqual(set(fixture["samples_by_action"]),
                {option["action_id"] for option in fixture["model"]["options"]})
        self.assertNotEqual(protocol["registered_at_utc"], protocol["source_BG_execution_epoch"])
        for name in ("bg_empirical_execution_checks.py", "bg_empirical_service_statistics.py"):
            self.assertIn("algorithm/experiments/" + name, protocol["execution_sources"])

    def test_postcheck_failure_preserves_trajectory_and_environment_snapshot_horizon(self):
        unit = self.protocol["units"][0]
        fixture = runner.read(self.directory / unit["fixture_file"])
        result = dict(jobs=[dict(coordinate=c, index=i, arrival=float(t), start=None, completion=None)
            for t, c, i in fixture["tape"]["arrivals"]], frames=[])
        with patch.object(runner, "simulate_empirical_service_trajectory", return_value=result) as execute, \
                patch.object(runner, "check_empirical_execution", return_value=False):
            with self.assertRaisesRegex(ValueError, "empirical execution"):
                runner.run_unit(self.directory, 0)
        self.assertEqual(execute.call_args.args[2:4], (1200., 1200.))
        saved = runner.read(runner.path(self.directory, "_unit_000_trajectory.json.gz"))
        self.assertEqual(saved["result"]["condition"], "mean")
        self.assertEqual(saved["result"]["service_seed"], unit["service_seed_key"])
        self.assertFalse(runner.path(self.directory, "_unit_000.json").exists())
        runner.path(self.directory, "_unit_000_trajectory.json.gz").unlink()

    def test_actual_paired_outcomes_and_statistics_merge_serialized_jobs_and_snapshots(self):
        # These are synthetic saved-arm shape fixtures, not scheduling runs.
        tapes = {tape["tape_id"]: tape for tape in self.protocol["tapes"]}
        for unit in self.protocol["units"]:
            flow = {("mean", "maxweight"): 20., ("mean", "BG"): 15.,
                ("empirical", "maxweight"): 40., ("empirical", "BG"): 38.}[unit["condition"], unit["policy"]]
            jobs = [dict(coordinate=c, index=i, arrival=float(t), start=float(t), completion=float(t) + flow)
                for t, c, i in tapes[unit["tape_id"]]["arrivals"]]
            snapshots = {name: trajectory_snapshot(jobs, 1200., 3)
                for name in ("arrival_horizon", "observation_end")}
            trajectory = runner.path(self.directory, f"_unit_{unit['unit_index']:03d}_trajectory.json.gz")
            runner.save(trajectory, dict(registered_at_utc=self.protocol["registered_at_utc"], unit=unit,
                result=dict(jobs=jobs, snapshots=snapshots, condition=unit["condition"],
                    service_seed=unit["service_seed_key"])))
            runner.save(runner.path(self.directory, f"_unit_{unit['unit_index']:03d}.json"),
                dict(**unit, registered_at_utc=self.protocol["registered_at_utc"], execution_checks_passed=True,
                    trajectory_file=trajectory.name, work={"max_rollout_calls": 0},
                    service_environment_work={"support_record_draws": 0},
                    exact_score_diagnostic_work={"exact_score_records": 0, "retained_fraction_pairs": 0}))
        with redirect_stdout(StringIO()):
            runner.merge(self.directory)
        summary = runner.read(runner.path(self.directory, "_summary.json"))
        self.assertEqual(summary["completed_new_trajectories"], 96)
        self.assertEqual(summary["merge_work"]["separate_condition_policy_pairs"], 48)
        for condition in runner.CONDITIONS:
            statistics = summary["statistics"]["condition_summaries"][condition]
            self.assertEqual([len(statistics[key]) for key in
                ("groups", "class_results", "seed_blocks", "paired_results")], [8, 24, 3, 24])
            self.assertIn("decision", summary["performance_screens"][condition])
            flow = 20. if condition == "mean" else 40.
            expected_unfinished = sum(300. <= time <= 900. and time + flow > 900.
                for tape in tapes.values() for time, _, _ in tape["arrivals"])
            self.assertGreater(expected_unfinished, 0)
            self.assertEqual(statistics["pooled"]["absolute_snapshots"]["arrival_horizon"]
                ["factual"]["unfinished_jobs"], expected_unfinished)
        self.assertFalse(summary["statistics"]["confidence_intervals_computed"])


if __name__ == "__main__":
    unittest.main()
