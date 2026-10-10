"""Different paired completion sets must not share a single candidate P90."""
import unittest

from algorithm.experiments.bg_confirmation_block_intervals import ARRIVAL_SEEDS, CELLS
from algorithm.experiments.s1_only_supplemental_statistics import supplemental_statistics


class S1OnlySupplementalStatisticsTest(unittest.TestCase):
    def test_pair_specific_common_p90_orientation_and_unfinished_totals(self):
        rows = []
        for seed_index, seed in enumerate(ARRIVAL_SEEDS):
            for cell_index, (node, lanes, load) in enumerate(CELLS):
                contrasts = {}
                for label, base_p90, candidate_p90, base_flow, base_tail in (
                    ("no_background", 70., 80., 30., 3),
                    ("BG", 95., 90., 31., 1),
                    ("maxweight", 50., 70., 40., 4),
                ):
                    offset = seed_index + cell_index
                    snapshots = {}
                    for cutoff, time in (("arrival_horizon", 900.), ("observation_end", 1200.)):
                        pair = {}
                        for arm, flow, p90, tail in (
                            ("factual", base_flow, base_p90, base_tail),
                            ("replacement", 32., candidate_p90, 2),
                        ):
                            pair[arm] = dict(time=time, arrivals=100,
                                common_jobs={"no_background": 93, "BG": 95, "maxweight": 91}[label],
                                restricted_mean_flow_s=flow + offset,
                                common_p90_flow_s=p90 + offset,
                                unfinished_jobs=tail if time == 900 else 0)
                        snapshots[cutoff] = pair
                    contrasts[label] = {"snapshots": snapshots}
                rows.append(dict(tape_id=f"{node}:{load}:{seed}", node=node,
                    lanes=lanes, load=load, seed=seed, contrasts=contrasts))

        result = supplemental_statistics(rows)
        self.assertFalse(result["automatic_adoption"])
        self.assertTrue(result["BG_main_result_preserved"])
        self.assertEqual(result["primary_comparator"], "no_background")
        for label, flow_delta, p90_delta, tail_delta in (
            ("no_background", 2., 10., -96.),
            ("BG", 1., -5., 96.),
            ("maxweight", -8., 20., -192.),
        ):
            comparison = result["comparisons"][label]
            self.assertEqual(comparison["role"], "primary" if label == "no_background" else "descriptive")
            for cutoff, intervals in comparison["block_intervals"].items():
                self.assertEqual(intervals["labels"], (label, "S1_only"))
                self.assertEqual(intervals["arrival_seed_blocks"], 12)
                self.assertEqual(intervals["cells_per_block"], 8)
                self.assertEqual(intervals["bootstrap_replicates"], 20_000)
                self.assertEqual(intervals["bootstrap_seed"], 20261003)
                for metric, difference in (("restricted_mean_flow_s", flow_delta),
                    ("common_p90_flow_s", p90_delta),
                    ("unfinished_jobs", tail_delta if cutoff == "arrival_horizon" else 0.)):
                    paired = intervals["metrics"][metric]["paired_difference"]
                    self.assertAlmostEqual(paired["estimate"], difference)
                    for bound in paired["ci95"]:
                        self.assertAlmostEqual(bound, difference)
                mean_metric = intervals["metrics"]["restricted_mean_flow_s"]
                self.assertAlmostEqual(mean_metric["relative_change"]["estimate"],
                    flow_delta / mean_metric["baseline"])


if __name__ == "__main__":
    unittest.main()
