"""Detect lost condition pairing, wrong task-count scaling and hidden HOLDs."""
from copy import deepcopy
import unittest

from algorithm.experiments import bg_empirical_service_statistics as stats


def fixture():
    data = {condition: [] for condition in stats.CONDITIONS}
    for condition in stats.CONDITIONS:
        empirical = condition == "empirical"
        for seed_index, (seed, service_seed) in enumerate(stats.SEED_PAIRS.items()):
            for cell_index, (node, lanes, load) in enumerate(stats.CELLS):
                baseline = (300 if empirical else 100) + 10 * seed_index + cell_index
                mean_effect, p90_effect = (-7, -12) if empirical else (-10, -20)
                unfinished_effect = -2 if empirical else -1
                snapshots = {}
                for cutoff, time in stats.CUTOFF_TIMES.items():
                    unfinished = 20 if cutoff == "arrival_horizon" else 10
                    common = (33 if empirical else 35) if unfinished == 20 else (45 if empirical else 47)
                    pair = {}
                    for arm in ("factual", "replacement"):
                        replacement = arm == "replacement"
                        classes = []
                        for coordinate in range(3):
                            class_effect = (9 if empirical else 6) if coordinate == 0 else (-15 if empirical else -18)
                            tail = unfinished + (unfinished_effect if replacement else 0)
                            inflight = 4 if replacement else 3
                            classes.append(dict(arrivals=60, completed_jobs=60 - tail,
                                queued_jobs=tail - inflight, inflight_jobs=inflight, unfinished_jobs=tail,
                                common_class_jobs=common,
                                restricted_mean_flow_s=baseline + (class_effect if replacement else 0),
                                completed_p90_flow_s=2 * baseline + 11 + (p90_effect if replacement else 0),
                                common_class_mean_flow_s=baseline + (class_effect if replacement else 0),
                                common_class_p90_flow_s=2 * baseline + (p90_effect if replacement else 0)))
                        pair[arm] = dict(time=time, arrivals=180,
                            completed_jobs=sum(group["completed_jobs"] for group in classes),
                            queued_jobs=sum(group["queued_jobs"] for group in classes),
                            inflight_jobs=sum(group["inflight_jobs"] for group in classes),
                            unfinished_jobs=sum(group["unfinished_jobs"] for group in classes),
                            common_jobs=3 * common,
                            restricted_mean_flow_s=baseline + (mean_effect if replacement else 0),
                            common_p90_flow_s=2 * baseline + (p90_effect if replacement else 0),
                            per_coordinate=classes)
                    snapshots[cutoff] = pair
                effects = dict(delta_T_mean=mean_effect, delta_T_common_p90=p90_effect,
                    delta_900_unfinished=3 * unfinished_effect, delta_1200_unfinished=3 * unfinished_effect)
                classes = [dict(coordinate=coordinate, workload=f"class{coordinate}",
                    **{**effects, "delta_T_mean": (9 if empirical else 6) if coordinate == 0 else (-15 if empirical else -18),
                       "delta_900_unfinished": unfinished_effect, "delta_1200_unfinished": unfinished_effect})
                    for coordinate in range(3)]
                data[condition].append(dict(tape_id=f"{node}:{load}:{seed}", node=node, lanes=lanes,
                    load=load, seed=seed, service_seed=service_seed, condition=condition,
                    outcomes=dict(snapshots=snapshots, per_coordinate_deltas=classes, **effects)))
    return data


class EmpiricalServiceStatisticsTest(unittest.TestCase):
    def test_condition_specific_p90_matched_changes_and_whole_block_counts(self):
        result = stats.analyse_empirical_development(fixture())
        self.assertFalse(result["confidence_intervals_computed"])
        for condition, baseline, unfinished in (("mean", 113.5, 1368), ("empirical", 313.5, 1296)):
            summary = result["condition_summaries"][condition]
            self.assertEqual([len(summary[key]) for key in
                              ("groups", "class_results", "seed_blocks", "paired_results")], [8, 24, 3, 24])
            final = summary["pooled"]["absolute_snapshots"]["observation_end"]
            self.assertEqual(final["factual"]["restricted_mean_flow_s"], baseline)
            self.assertEqual(final["factual"]["common_p90_flow_s"], 2 * baseline)
            self.assertEqual(summary["pooled"]["absolute_snapshots"]["arrival_horizon"]
                             ["replacement"]["unfinished_jobs"], unfinished)
            self.assertEqual([block["service_seed"] for block in summary["seed_blocks"]], list(stats.SERVICE_SEEDS))
            self.assertTrue(all(block["paired_cells"] == 8 for block in summary["seed_blocks"]))
            self.assertEqual(sum(item["delta_T_mean"]["mean"] > 0 for item in summary["class_results"]), 8)
            for item in summary["class_results"]:
                for pair in item["absolute_snapshots"].values():
                    for arm in pair.values():
                        self.assertEqual(arm["queued_jobs"] + arm["inflight_jobs"], arm["unfinished_jobs"])
            self.assertNotIn("block_intervals", summary)
        differences = result["matched_effect_changes"]
        self.assertEqual([len(differences[key]) for key in ("groups", "seed_blocks", "paired_results")], [8, 3, 24])
        self.assertEqual(differences["pooled"]["delta_T_mean"]["mean"], 3)
        self.assertEqual(differences["pooled"]["delta_T_common_p90"]["mean"], 8)
        self.assertEqual(differences["pooled"]["delta_900_unfinished"]["total"], -72)
        for cutoff in stats.CUTOFF_TIMES:
            metrics = differences["pooled"]["snapshots"][cutoff]["metrics"]
            self.assertEqual(metrics["common_p90_flow_s"]["mean"], 8)
            self.assertEqual(metrics["unfinished_jobs"]["total"], -72)
            self.assertTrue(all(block["snapshots"][cutoff]["metrics"]["unfinished_jobs"]["total"] == -24
                                for block in differences["seed_blocks"]))

    def test_one_condition_group_regression_stays_hold_despite_pooled_gains(self):
        data = fixture()
        for row in data["empirical"]:
            if row["node"] != "node007" or row["load"] != 0.8:
                continue
            pair = row["outcomes"]["snapshots"]["arrival_horizon"]
            candidate = pair["replacement"]
            candidate.update(unfinished_jobs=72, completed_jobs=108, queued_jobs=60, inflight_jobs=12)
            for coordinate in candidate["per_coordinate"]:
                coordinate.update(unfinished_jobs=24, completed_jobs=36, queued_jobs=20, inflight_jobs=4)
            row["outcomes"]["delta_900_unfinished"] = 12
            for coordinate in row["outcomes"]["per_coordinate_deltas"]:
                coordinate["delta_900_unfinished"] = 4
        summaries = stats.analyse_empirical_development(data)["condition_summaries"]
        gates = dict(min_gain=0.01, group_secondary_tolerance=0.05)
        mean_screen = stats.performance_screen(summaries["mean"], gates)
        empirical_screen = stats.performance_screen(summaries["empirical"], gates)
        self.assertTrue(mean_screen["numerical_gates_passed"])
        self.assertTrue(empirical_screen["flow_gain_threshold_passed"])
        self.assertTrue(all(empirical_screen["pooled_unfinished_checks"].values()))
        self.assertFalse(empirical_screen["numerical_gates_passed"])
        self.assertEqual(empirical_screen["failed_secondary_groups"], [dict(node="node007", lanes=4,
            load=0.8, cutoff="arrival_horizon", failed_metrics=["unfinished_jobs"])])
        self.assertFalse(empirical_screen["independent_confirmation_performed"])
        zero = deepcopy(summaries["mean"])
        pair = zero["groups"][0]["absolute_snapshots"]["observation_end"]
        pair["factual"]["unfinished_jobs"], pair["replacement"]["unfinished_jobs"] = 0, 1
        self.assertFalse(stats.performance_screen(zero, gates)["numerical_gates_passed"])

    def test_service_seed_or_literal_tape_mismatch_is_not_repaired_by_position(self):
        data = fixture()
        data["empirical"][0]["service_seed"] = stats.SERVICE_SEEDS[1]
        with self.assertRaisesRegex(ValueError, "zipped arrival/service seed blocks"):
            stats.analyse_empirical_development(data)
        data = fixture()
        data["empirical"][0]["tape_id"] += ":different"
        with self.assertRaisesRegex(ValueError, "identical literal tapes"):
            stats.analyse_empirical_development(data)


if __name__ == "__main__":
    unittest.main()
