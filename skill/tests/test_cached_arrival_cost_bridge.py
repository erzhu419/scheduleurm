from fractions import Fraction as F
import unittest

from algorithm.experiments.cached_arrival_cost_bridge import FIELDS, birth_stats, compare, pair_stats


class CachedArrivalCostBridgeTests(unittest.TestCase):
    def test_preanchor_commitment_and_H_waiting_have_hand_computed_costs(self):
        # Omit elapsed pre-anchor time; keep the committed completion and a newborn.
        parent = [dict(birth=F(-4), start=F(-2), completion=F(1), source="committed"),
            dict(birth=F(1), start=F(2), completion=F(5), source="prefix")]
        native = [dict(birth=F(-4), start=F(-2), completion=F(2), source="committed"),
            dict(birth=F(1), start=F(3), completion=F(6), source="prefix")]
        result = pair_stats(parent, native, F(3))
        self.assertEqual(result["parent"]["cost"], 5)
        self.assertEqual(result["native"]["cost"], 7)
        self.assertEqual(result["delta"], dict(cost=F(-2), common_area=F(-1), waiting_area=F(-1),
            inflight_area=F(0), terminal_cost=F(-1), full_waiting_cost=F(-1), full_inflight_cost=F(-1)))
        self.assertEqual(result["at_H"]["native"], dict(waiting=1, inflight=0, done=1, waiting_age_s=F(2)))
        self.assertEqual(birth_stats(parent, F(3)), dict(newborn_jobs=1, quarter_H_counts=[0, 1, 0, 0],
            birth_time_sum_s=F(1), birth_exposure_s=F(2)))


    def test_per_job_normalization_pools_unequal_path_counts(self):
        def stats(count, delta):
            return dict(jobs=count, delta={field: F(delta) for field in FIELDS})
        # Two forecast paths have 1 and 3 jobs, and costs 2 and 10: pooled delta/job=3.
        result = compare(stats(2, 4), [stats(1, 2), stats(3, 10)])
        cost = result["count_per_job_decomposition"]["cost"]
        self.assertEqual(cost, dict(actual_delta_per_job=F(2), forecast_pooled_delta_per_job=F(3),
            forecast_standardized_to_actual_count=F(6), gap=F(2),
            count_term_at_actual_per_job_delta=F(0), per_job_term_at_forecast_count=F(2)))
