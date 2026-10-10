"""Verify replication averaging, shared-P90 populations and arrival-block CIs."""
from copy import deepcopy
from random import Random
from statistics import quantiles
import unittest

from algorithm.experiments import bg_planning_stream_statistics as stats
from algorithm.experiments.bg_bursty_statistics import CELLS


def fixture():
    data = {profile: [] for profile in stats.PROFILES}
    for profile in stats.PROFILES:
        for replication, stream_effect in zip(stats.PLANNING_REPLICATIONS, (-2, 0, 5)):
            for seed_index, seed in enumerate(stats.ARRIVAL_SEEDS):
                for cell_index, (node, lanes, load) in enumerate(CELLS):
                    baseline = (200 if profile == "steady" else 400) + 10 * seed_index + 5 * cell_index
                    cell_effect = 2 * cell_index - 7
                    stream_noise = (8 if replication == 1 else -8 if replication == 2 else 0) * seed_index
                    mean_effect = 3 * seed_index + cell_effect + stream_effect + stream_noise + (7 if profile == "bursty" else 0)
                    p90_effect = 2 * (3 * seed_index + cell_effect + stream_effect + stream_noise) + (11 if profile == "bursty" else 0)
                    count_effect = seed_index + cell_effect + stream_effect + (2 if profile == "bursty" else 0)
                    bonus = int(seed_index == cell_index == 0 and replication == 3)
                    snapshots = {}
                    for cutoff, time in stats.CUTOFF_TIMES.items():
                        pair = {}
                        for arm in ("factual", "replacement"):
                            replacement = arm == "replacement"
                            classes = []
                            for coordinate in range(3):
                                unfinished = 20 + seed_index + coordinate
                                if replacement:
                                    unfinished += count_effect + (bonus if coordinate == 0 else 0)
                                classes.append(dict(arrivals=160, completed_jobs=160 - unfinished,
                                    queued_jobs=unfinished - 3, inflight_jobs=3, unfinished_jobs=unfinished,
                                    common_class_jobs=40 + replication + coordinate,
                                    restricted_mean_flow_s=baseline + coordinate + (mean_effect if replacement else 0),
                                    completed_p90_flow_s=2 * (baseline + coordinate) + 13 + (p90_effect if replacement else 0),
                                    common_class_mean_flow_s=baseline + coordinate + (mean_effect if replacement else 0),
                                    common_class_p90_flow_s=2 * (baseline + coordinate) + 20 * replication + (p90_effect if replacement else 0)))
                            pair[arm] = dict(time=time, per_coordinate=classes,
                                **{field: sum(item[field] for item in classes) for field in stats.COUNTS},
                                common_jobs=sum(item["common_class_jobs"] for item in classes),
                                restricted_mean_flow_s=baseline + 1 + (mean_effect if replacement else 0),
                                common_p90_flow_s=2 * (baseline + 1) + 20 * replication + (p90_effect if replacement else 0))
                        snapshots[cutoff] = pair
                    effects = dict(delta_T_mean=mean_effect, delta_T_common_p90=p90_effect,
                        delta_900_unfinished=3 * count_effect + bonus,
                        delta_1200_unfinished=3 * count_effect + bonus)
                    classes = [dict(coordinate=coordinate, workload=f"class{coordinate}",
                        delta_T_mean=mean_effect, delta_T_common_p90=p90_effect,
                        delta_900_unfinished=count_effect + (bonus if coordinate == 0 else 0),
                        delta_1200_unfinished=count_effect + (bonus if coordinate == 0 else 0))
                        for coordinate in range(3)]
                    data[profile].append(dict(tape_id=f"{profile}:{node}:{load}:{seed}",
                        node=node, lanes=lanes, load=load, seed=seed, profile=profile,
                        planning_replication=replication, planning_seed=f"plan:{node}:{load}:{replication}",
                        outcomes=dict(snapshots=snapshots, per_coordinate_deltas=classes, **effects)))
    return data


class PlanningStreamStatisticsTest(unittest.TestCase):
    def test_average_inside_cell_and_retain_every_stream_and_profile(self):
        result = stats.analyse_planning_streams(fixture())
        steady = result["profile_summaries"]["steady"]
        averaged = steady["averaged"]
        self.assertEqual(set(steady["replications"]), {"1", "2", "3"})
        for summary in (averaged, *steady["replications"].values()):
            self.assertEqual([len(summary[key]) for key in
                              ("groups", "class_results", "seed_aggregates", "paired_results")],
                             [8, 24, 12, 96])
        metrics = averaged["block_intervals"]["observation_end"]["metrics"]
        self.assertAlmostEqual(metrics["restricted_mean_flow_s"]["paired_difference"]["estimate"], 17.5)
        self.assertAlmostEqual(metrics["restricted_mean_flow_s"]["relative_change"]["estimate"], 17.5 / 273.5)
        rng = Random(stats.BOOTSTRAP_SEED)
        block_means = [sum(rng.randrange(12) for _ in range(12)) / 12
                       for _ in range(stats.BOOTSTRAP_REPLICATES)]
        differences = quantiles([1 + 3 * value for value in block_means], n=40, method="inclusive")
        relatives = quantiles([(1 + 3 * value) / (218.5 + 10 * value) for value in block_means],
                              n=40, method="inclusive")
        for field, expected in (("paired_difference", differences), ("relative_change", relatives)):
            self.assertAlmostEqual(metrics["restricted_mean_flow_s"][field]["ci95"][0], expected[0])
            self.assertAlmostEqual(metrics["restricted_mean_flow_s"][field]["ci95"][1], expected[-1])
        self.assertAlmostEqual(metrics["common_p90_flow_s"]["baseline"], 2 * 273.5 + 40)
        self.assertAlmostEqual(metrics["common_p90_flow_s"]["paired_difference"]["estimate"], 35.0)
        counts = metrics["unfinished_jobs"]
        self.assertEqual(counts["baseline"], 7632)
        self.assertAlmostEqual(counts["candidate"], 7632 + 1872 + 1 / 3)
        self.assertNotIn("relative_change", counts)
        self.assertEqual(counts["aggregation"], "mean_total_jobs_over_three_planning_replications")
        individual = [steady["replications"][str(rep)]["block_intervals"]["observation_end"]["metrics"]
                      for rep in stats.PLANNING_REPLICATIONS]
        self.assertAlmostEqual(counts["candidate"], sum(metric["unfinished_jobs"]["candidate"] for metric in individual) / 3)
        self.assertEqual([metric["common_p90_flow_s"]["baseline"] for metric in individual], [567, 587, 607])
        for cutoffs in result["difference_of_differences"].values():
            entries = (cutoffs.values() if "snapshots" not in cutoffs else (cutoffs,))
            for entry in entries:
                for interval in entry["snapshots"].values():
                    for metric, expected in (("restricted_mean_flow_s", 7),
                                             ("common_p90_flow_s", 11), ("unfinished_jobs", 576)):
                        actual = interval["metrics"][metric]["paired_difference"]
                        self.assertAlmostEqual(actual["estimate"], expected)
                        self.assertAlmostEqual(actual["ci95"][0], expected)
                        self.assertAlmostEqual(actual["ci95"][1], expected)

    def test_reused_MW_counts_must_match_but_shared_completion_P90_may_differ(self):
        rows = fixture()["steady"]
        identity = stats._identity(rows[0])
        cell = [row for row in rows if stats._identity(row) == identity]
        self.assertEqual(stats._average_cell(cell)["outcomes"]["snapshots"]["observation_end"]["factual"]["common_jobs"], 129)
        cell = deepcopy(cell)
        cell[2]["outcomes"]["snapshots"]["observation_end"]["factual"]["unfinished_jobs"] += 1
        with self.assertRaisesRegex(ValueError, "reused MW full-cohort"):
            stats._average_cell(cell)

    def test_missing_stream_or_changed_literal_tape_is_rejected(self):
        data = fixture()
        with self.assertRaisesRegex(ValueError, "all288 paired rows"):
            stats.analyse_planning_streams({**data, "steady": data["steady"][:-1]})
        data["steady"][100]["tape_id"] += ":different"
        with self.assertRaisesRegex(ValueError, "same literal tapes"):
            stats.analyse_planning_streams(data)


if __name__ == "__main__":
    unittest.main()
