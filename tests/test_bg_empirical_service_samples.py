"""Checks on admitted physical sources and their native BG compositions."""
from copy import deepcopy
import json
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory
import unittest

from algorithm.experiments.aligned_hardware_empirical_service_samples import (
    OUTPUT, ROOT, loaded_phase, read, source_components,
)
from algorithm.experiments.aligned_hardware_physical_catalogue_models import load_catalogues


class EmpiricalServiceSamplesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bank = read(OUTPUT)

    def test_native_roster_and_training_means_keep_holdout_separate(self):
        models = {model.node: model for model in load_catalogues()}
        self.assertEqual(self.bank["coverage"]["actions"], 35)
        self.assertEqual(self.bank["coverage"]["training_samples"], 1476)
        self.assertGreater(self.bank["coverage"]["minimum_raw_duration"], 1)
        means = {model["node"]: model for model in self.bank["mean_model_snapshots"]}
        for catalogue in self.bank["catalogues"]:
            node = catalogue["node"]
            self.assertEqual(catalogue["native_job_sizes"], [60.0, 40.0, 120.0])
            self.assertEqual({action["action_id"] for action in catalogue["actions"]},
                             {option.action_id for option in models[node].options})
            mean_options = {option["action_id"]: option for option in means[node]["options"]}
            for action in catalogue["actions"]:
                training, holdout = action["training"], action["holdout"]
                self.assertEqual(len(training), 144 if action["loaded"] else 12)
                self.assertEqual(len(holdout), 1)
                self.assertTrue(all(sample["empty_wave"] <= 12 for sample in training))
                self.assertEqual(holdout[0]["empty_wave"], 13)
                if action["loaded"]:
                    self.assertEqual({(sample["empty_wave"], sample["loaded_wave"]) for sample in training},
                                     {(empty, loaded) for empty in range(1, 13) for loaded in range(1, 13)})
                    self.assertEqual(holdout[0]["loaded_wave"], 13)
                option = mean_options[action["action_id"]]
                self.assertEqual(option["requirements"], action["requirements"])
                self.assertAlmostEqual(option["duration"], mean(sample["effective_duration"] for sample in training))
                for coordinate, count in enumerate(action["requirements"]):
                    for slot in range(count):
                        self.assertAlmostEqual(option["completion_offsets"][coordinate][slot],
                            mean(sample["completion_offsets"][coordinate][slot] for sample in training))

    def test_training_capacity_exclusion_cannot_be_readmitted_from_successful_wave(self):
        gate_path = ROOT / "md/experiment_artifacts/critical_gpu_stochastic_lcb_gate_v5_cpu_a_20260803.json"
        gate = read(gate_path)
        excluded = next(row for row in gate["training_capacity_excluded_cells"]
                        if row["workload_key"] == "gpu_cnn_torch_resnet50" and row["profile"] == 2)
        with self.assertRaisesRegex(ValueError, "not admitted by final gate"):
            source_components(gate_path, excluded)
        catalogue = next(model for model in self.bank["catalogues"] if model["node"] == "cpu_a")
        self.assertFalse(any(action["requirements"][0] == 2 for action in catalogue["actions"]))

    def test_node007_p10_requires_all_registered_transport_corrected_artifacts(self):
        catalogue = next(model for model in self.bank["catalogues"] if model["node"] == "node007")
        action = next(action for action in catalogue["actions"] if action["requirements"] == [0, 0, 10])
        self.assertTrue(all(component["transport_corrected"]
                            and "critical_gpu_p10_transport_v2_node007" in component["source_path"]
                            for component in action["components"]["empty"]))
        gate_path = ROOT / "md/experiment_artifacts/critical_gpu_stochastic_lcb_gate_v9_node007_20260808.json"
        gate = read(gate_path)
        certificate = next(row for row in gate["certificate"]["rows"]
                           if row["workload_key"] == "gpu_llm_distilgpt2" and row["profile"] == 10)
        altered = deepcopy(gate)
        altered["correction_artifacts"] = altered["correction_artifacts"][:-1]
        with TemporaryDirectory(prefix="empirical_gate_fixture_") as temporary:
            path = Path(temporary) / "gate.json"
            path.write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError, "thirteen gate-listed"):
                source_components(path, certificate)

    def test_loaded_environment_mismatch_is_rejected_and_pair_remains_same_row(self):
        gate_path = ROOT / "md/experiment_artifacts/critical_gpu_loaded_stochastic_lcb_gate_cpu_a_20260809.json"
        gate = read(gate_path)
        certificate = next(row for row in gate["certificate"]["rows"] if row["scenario_id"] == "cnn_after_llm")
        payload = read(gate["source_artifacts"][0]["path"])
        paired = loaded_phase(payload, certificate)
        original = next(row for row in payload["rows"] if row["scenario_id"] == "cnn_after_llm")
        self.assertEqual(paired["target_phase"]["completion_unit_s"],
                         original["target_completion_model"]["completion_unit_s"])
        self.assertEqual(paired["resident_overlap_service_units_per_s"],
                         original["resident_overlap_service"]["conservative_rate_units_per_s"])
        altered = deepcopy(payload)
        row = next(row for row in altered["rows"] if row["scenario_id"] == "cnn_after_llm")
        row["resident_workload_env"] = "different-native-environment"
        with self.assertRaisesRegex(ValueError, "resident/target pair mismatch"):
            loaded_phase(altered, certificate)


if __name__ == "__main__":
    unittest.main()
