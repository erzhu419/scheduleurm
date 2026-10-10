"""Detect broken block/profile pairing, count scaling and shared-P90 reuse."""
from copy import deepcopy
from random import Random
from statistics import quantiles
import unittest

from algorithm.experiments import bg_bursty_statistics as stats


def fixture():
    populations = {profile: {} for profile in stats.PROFILES}
    for profile in stats.PROFILES:
        rows = []
        for seed_index, seed in enumerate(stats.ARRIVAL_SEEDS):
            for cell_index, (node, lanes, load) in enumerate(stats.CELLS):
                baseline = (200 if profile == "steady" else 400) + 10 * seed_index + 5 * cell_index
                cell_effect = 2 * cell_index - 7
                mean_effect = 3 * seed_index + cell_effect + (7 if profile == "bursty" else 0)
                p90_effect = 2 * (3 * seed_index + cell_effect) + (11 if profile == "bursty" else 0)
                count_effect = seed_index + cell_effect + (2 if profile == "bursty" else 0)
                snapshots = {}
                for cutoff, time in stats.CUTOFF_TIMES.items():
                    pair = {}
                    for arm in ("factual", "replacement"):
                        replacement = arm == "replacement"
                        unfinished = 50 + (count_effect if replacement else 0)
                        classes = []
                        for coordinate in range(3):
                            class_unfinished = 20 + seed_index + (cell_index if replacement else 0)
                            classes.append(dict(arrivals=60, completed_jobs=60 - class_unfinished,
                                common_class_jobs=40,
                                queued_jobs=class_unfinished - 3, inflight_jobs=3,
                                unfinished_jobs=class_unfinished,
                                restricted_mean_flow_s=baseline + (mean_effect if replacement else 0),
                                completed_p90_flow_s=2 * baseline + 13 + (p90_effect if replacement else 0),
                                common_class_mean_flow_s=baseline + (mean_effect if replacement else 0),
                                common_class_p90_flow_s=2 * baseline + (p90_effect if replacement else 0)))
                        pair[arm] = dict(time=time, arrivals=150, common_jobs=100,
                            completed_jobs=150 - unfinished, queued_jobs=unfinished - 5,
                            inflight_jobs=5, unfinished_jobs=unfinished,
                            restricted_mean_flow_s=baseline + (mean_effect if replacement else 0),
                            common_p90_flow_s=2 * baseline + (p90_effect if replacement else 0),
                            per_coordinate=classes)
                    snapshots[cutoff] = pair
                effects = dict(delta_T_mean=mean_effect, delta_T_common_p90=p90_effect,
                               delta_900_unfinished=count_effect, delta_1200_unfinished=count_effect)
                classes = [dict(coordinate=coordinate, workload=f"class{coordinate}",
                    **{**effects, "delta_900_unfinished": cell_index,
                       "delta_1200_unfinished": cell_index}) for coordinate in range(3)]
                rows.append(dict(tape_id=f"{profile}:{node}:{load}:{seed}", node=node,
                    lanes=lanes, load=load, seed=seed,
                    outcomes=dict(snapshots=snapshots, per_coordinate_deltas=classes, **effects)))
        for contrast in stats.CONTRASTS:
            populations[profile][contrast] = deepcopy(rows)
    return populations


class BurstyStatisticsTest(unittest.TestCase):
    def test_matched_profiles_keep_whole_seed_blocks_and_total_counts(self):
        data = fixture()
        for profile in stats.PROFILES:
            for row in data[profile]["S1_only_minus_no_background"]:
                for cutoff in stats.CUTOFF_TIMES:
                    for arm in ("factual", "replacement"):
                        row["outcomes"]["snapshots"][cutoff][arm]["common_p90_flow_s"] += 100
        result = stats.analyse_bursty(data)
        for contrast in stats.CONTRASTS:
            for cutoff in stats.CUTOFF_TIMES:
                metrics = result["difference_of_differences"][contrast]["snapshots"][cutoff]["metrics"]
                for metric, expected in (("restricted_mean_flow_s", 7.0),
                                         ("common_p90_flow_s", 11.0), ("unfinished_jobs", 192.0)):
                    self.assertEqual(metrics[metric]["paired_difference"],
                                     dict(estimate=expected, ci95=[expected, expected]))
            profile = result["profile_summaries"]["bursty"][contrast]
            self.assertEqual([len(profile[key]) for key in
                              ("groups", "seed_aggregates", "class_results", "paired_results")],
                             [8, 12, 24, 96])
            class_result = profile["class_results"][0]
            for cutoff in stats.CUTOFF_TIMES:
                for arm in ("factual", "replacement"):
                    tail = class_result["absolute_snapshots"][cutoff][arm]
                    self.assertEqual(tail["queued_jobs"] + tail["inflight_jobs"], tail["unfinished_jobs"])
                    self.assertEqual(tail["inflight_jobs"], 12 * 3)
                    self.assertEqual(tail["common_class_jobs"], 12 * 40)
                    self.assertEqual(tail["completed_p90_flow_s"] - tail["common_class_p90_flow_s"], 13)
            self.assertNotIn("snapshots", profile["paired_results"][0])
        summaries = result["profile_summaries"]
        for profile, baseline in (("steady", 545.0), ("bursty", 945.0)):
            for contrast in stats.CONTRASTS:
                p90 = summaries[profile][contrast]["block_intervals"]["observation_end"]["metrics"]["common_p90_flow_s"]
                self.assertEqual(p90["baseline"], baseline + (100 if contrast == "S1_only_minus_no_background" else 0))

    def test_relative_change_resamples_aggregate_denominator(self):
        rows = fixture()["steady"]["BG_minus_maxweight"]
        blocks = stats._blocks(rows)
        rng = Random(stats.BOOTSTRAP_SEED)
        draws = [tuple(rng.randrange(12) for _ in range(12)) for _ in range(200)]
        result = stats._profile_intervals(blocks, "observation_end", draws)
        expected = [sum(3 * index for index in draw) / sum(217.5 + 10 * index for index in draw)
                    for draw in draws]
        relative = result["metrics"]["restricted_mean_flow_s"]["relative_change"]
        self.assertAlmostEqual(relative["estimate"], 16.5 / 272.5)
        interval = quantiles(expected, n=40, method="inclusive")
        self.assertAlmostEqual(relative["ci95"][0], interval[0])
        self.assertAlmostEqual(relative["ci95"][1], interval[-1])
        p90 = result["metrics"]["common_p90_flow_s"]
        self.assertAlmostEqual(p90["baseline"], 545.0)
        self.assertAlmostEqual(p90["paired_difference"]["estimate"], 33.0)

    def test_incomplete_roster_or_mismatched_contrast_tape_is_rejected(self):
        data = fixture()
        with self.assertRaisesRegex(ValueError, "twelve new seed blocks"):
            stats._blocks(data["steady"]["BG_minus_maxweight"][:-1])
        data["steady"]["S1_only_minus_no_background"][0]["tape_id"] += ":different"
        with self.assertRaisesRegex(ValueError, "identical literal tapes"):
            stats.analyse_bursty(data)


if __name__ == "__main__":
    unittest.main()
