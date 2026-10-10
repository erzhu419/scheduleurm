"""Observed-rate age-tail planning without an arrival stopping-time input."""
from algorithm.observed_rate_age_tail_lane_rollout import ObservedRateAgeTailLaneRollout


class ContinuousArrivalAgeTailLaneRollout(ObservedRateAgeTailLaneRollout):
    """Unbounded observation exposure; every eligible window is tau_max.

    Infinity represents no announced stopping time internally. It never enters
    the finite planning population, recorded decision or evaluation protocol.
    The constructor receives neither evaluation end nor true arrival rates.
    """
    def __init__(self, options, planning_seed, sample_count=8, event_cap=128):
        super().__init__(options, float("inf"), planning_seed, sample_count, event_cap)
