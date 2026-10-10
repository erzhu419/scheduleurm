"""Arrival-clock checks; no scheduler trajectories or fitted statistics."""
from collections import Counter
import unittest

from algorithm.experiments.aligned_hardware_frame_experiment import poisson_arrivals
from algorithm.experiments.matched_bursty_arrivals import (
    cumulative_modulated_time, inverse_modulated_time, matched_bursty_arrivals,
    modulate_arrival_tape, modulation_high_first,
)


# Actual published gpu_b/.95 rates, not an invented numerical fixture.
RATES = (0.12690001827996245, 0.27857287626960897, 0.3412059828785984)
SEEDS = tuple(range(200001, 200013))


class MatchedBurstyArrivalsTest(unittest.TestCase):
    def test_both_clock_phases_have_registered_intensities_and_exact_endpoints(self):
        for high_first in (False, True):
            first_exposure = 75.0 if high_first else 25.0
            second_exposure = 100.0 - first_exposure
            self.assertEqual(cumulative_modulated_time(50.0, high_first), first_exposure)
            self.assertEqual(cumulative_modulated_time(100.0, high_first)
                             - cumulative_modulated_time(50.0, high_first), second_exposure)
            self.assertEqual(inverse_modulated_time(first_exposure, high_first), 50.0)
            for endpoint in range(0, 1201, 100):
                self.assertEqual(cumulative_modulated_time(endpoint, high_first), endpoint)
                self.assertEqual(inverse_modulated_time(endpoint, high_first), endpoint)
            for time in (0.0, 1.0, 25.0, 49.0, 50.0, 51.0, 75.0, 99.0,
                         100.0, 125.0, 599.75, 900.0, 1199.999):
                mapped = cumulative_modulated_time(time, high_first)
                self.assertAlmostEqual(inverse_modulated_time(mapped, high_first), time)

    def test_real_rate_tapes_preserve_cohort_counts_and_class_fifo_ids(self):
        phases = set()
        for seed in SEEDS:
            phases.add(modulation_high_first(seed))
            steady = poisson_arrivals(RATES, 1200.0, seed)
            bursty = modulate_arrival_tape(steady, seed)
            self.assertEqual(bursty, tuple(sorted(bursty)))
            self.assertEqual(len(steady), len(bursty))
            self.assertEqual([(coordinate, index) for _, coordinate, index in steady],
                             [(coordinate, index) for _, coordinate, index in bursty])
            for cutoff in (300.0, 900.0, 1200.0):
                self.assertEqual(Counter(coordinate for time, coordinate, _ in steady if time <= cutoff),
                                 Counter(coordinate for time, coordinate, _ in bursty if time <= cutoff))
            steady_cohort = {(coordinate, index) for time, coordinate, index in steady
                             if 300.0 <= time <= 900.0}
            bursty_cohort = {(coordinate, index) for time, coordinate, index in bursty
                             if 300.0 <= time <= 900.0}
            self.assertEqual(steady_cohort, bursty_cohort)
            self.assertNotEqual([time for time, _, _ in steady], [time for time, _, _ in bursty])
            self.assertEqual(matched_bursty_arrivals(RATES, 1200.0, seed), bursty)
        self.assertEqual(phases, {False, True})

    def test_jobs_at_cohort_and_cycle_boundaries_keep_their_exact_timestamps(self):
        steady = ((0.0, 0, 0), (50.0, 1, 0), (100.0, 0, 1),
                  (300.0, 0, 2), (900.0, 0, 3), (1200.0, 0, 4))
        for seed in SEEDS:
            bursty = modulate_arrival_tape(steady, seed)
            birth_by_id = {(coordinate, index): time for time, coordinate, index in bursty}
            for time, coordinate, index in steady:
                if time % 100.0 == 0.0:
                    self.assertEqual(birth_by_id[(coordinate, index)], time)
            self.assertEqual(modulate_arrival_tape(steady, seed), bursty)


if __name__ == "__main__":
    unittest.main()
