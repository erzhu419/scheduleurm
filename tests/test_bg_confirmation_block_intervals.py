"""The synthetic population cancels within every seed, not within every cell."""
from copy import deepcopy
from random import Random
from statistics import quantiles
import unittest

from algorithm.experiments.bg_confirmation_block_intervals import (
    ARRIVAL_SEEDS, CELLS, block_intervals,
)


def fixture():
    rows = []
    for seed_index, seed in enumerate(ARRIVAL_SEEDS):
        for cell_index, (node, lanes, load) in enumerate(CELLS):
            baseline = 200 + seed_index * 10 + cell_index * 10
            deviation = (2 * cell_index - 7) * 10
            pair = {}
            for label, value in (("base", baseline), ("new", baseline + deviation)):
                pair[label] = {
                    "time": 1200.0, "arrivals": 100, "common_jobs": 100,
                    "restricted_mean_flow_s": value, "common_p90_flow_s": 2 * value,
                    "unfinished_jobs": 20 + (0 if label == "base" else 2 * cell_index - 7),
                }
            rows.append({"tape_id": f"{node}:{load}:{seed}", "node": node,
                         "lanes": lanes, "load": load, "seed": seed,
                         "snapshots": {"observation_end": pair}})
    return rows


class PairedBlockIntervalsTest(unittest.TestCase):
    def test_whole_seed_resampling_and_ratio_of_means(self):
        result = block_intervals(fixture(), labels=("base", "new"), bootstrap_replicates=200)
        for name, metric in result["metrics"].items():
            self.assertEqual(metric["paired_difference"], {"estimate": 0.0, "ci95": [0.0, 0.0]})
            if name != "unfinished_jobs":
                self.assertEqual(metric["relative_change"]["estimate"], 0.0)
                self.assertEqual(metric["relative_change"]["ci95"], [0.0, 0.0])
            else:
                self.assertNotIn("relative_change", metric)

    def test_relative_interval_uses_resampled_means_not_seed_ratios(self):
        rows = fixture()
        for row in rows:
            pair = row["snapshots"]["observation_end"]
            pair["new"]["restricted_mean_flow_s"] = pair["base"]["restricted_mean_flow_s"] + 40
        result = block_intervals(rows, labels=("base", "new"), bootstrap_replicates=200)
        relative = result["metrics"]["restricted_mean_flow_s"]["relative_change"]
        self.assertAlmostEqual(relative["estimate"], 40 / 290)
        rng = Random(20261003)
        values = [40 / (sum(235 + 10 * rng.randrange(12) for _ in range(12)) / 12)
                  for _ in range(200)]
        expected = quantiles(values, n=40, method="inclusive")
        self.assertAlmostEqual(relative["ci95"][0], expected[0])
        self.assertAlmostEqual(relative["ci95"][1], expected[-1])

    def test_missing_common_metric_does_not_drop_a_seed(self):
        rows = deepcopy(fixture())
        for label in ("base", "new"):
            rows[0]["snapshots"]["observation_end"][label]["common_p90_flow_s"] = None
            rows[0]["snapshots"]["observation_end"][label]["common_jobs"] = 0
        result = block_intervals(rows, labels=("base", "new"), bootstrap_replicates=20)
        self.assertFalse(result["metrics"]["common_p90_flow_s"]["estimable"])
        self.assertEqual(result["arrival_seed_blocks"], 12)
        self.assertEqual(result["paired_tapes"], 96)


if __name__ == "__main__":
    unittest.main()
