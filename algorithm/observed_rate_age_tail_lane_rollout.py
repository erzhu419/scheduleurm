"""Age-tail planning with arrival rates estimated from observed births only.

The constructor has no arrival-rate or load input. At each decision the
homogeneous Poisson MLE uses cumulative released arrivals, including jobs
already admitted or completed, divided by elapsed observation exposure. The
registered public arrival stopping time bounds exposure and planning window.
"""
from dataclasses import dataclass, fields, replace

from algorithm.age_tail_constrained_lane_rollout import (
    AgeTailConstrainedLaneRollout, AgeTailDecision,
)


@dataclass(frozen=True)
class ObservedRateDecision(AgeTailDecision):
    estimated_arrival_rates: tuple = ()
    observed_arrival_counts: tuple = ()
    rate_exposure_s: float = 0.0
    rate_estimation_coordinate_reads: int = 0
    rate_estimation_divisions: int = 0


class ObservedRateAgeTailLaneRollout(AgeTailConstrainedLaneRollout):
    """Unsmoothed cumulative-count MLE plus the frozen eight-sample S2 rule."""
    def __init__(self, options, arrival_horizon, planning_seed,
                 sample_count=8, event_cap=128):
        options = tuple(options)
        dimension = len(options[0].requirements)
        super().__init__(options, (0.0,) * dimension, arrival_horizon, planning_seed,
                         sample_count, event_cap)
        self.preprocessing_work["rate_dimension_option_probes"] = 1

    def maxweight(self, options, queues, available, now, lane, waiting_ages,
                  observed_arrival_counts=None):
        """Plain MW ignores counts; the parent selector also calls it with six args."""
        baseline = super().maxweight(options, queues, available, now, lane, waiting_ages)
        return ObservedRateDecision(**{field.name: getattr(baseline, field.name)
                                       for field in fields(AgeTailDecision)})

    def __call__(self, options, queues, available, now, lane, waiting_ages,
                 observed_arrival_counts):
        exposure = min(float(now), self.arrival_horizon)
        counts, estimated = [], []
        for count in observed_arrival_counts:
            counts.append(count)
            estimated.append(count / exposure if exposure > 0 else 0.0)
        self.arrival_rates = tuple(estimated)
        selected = super().__call__(options, queues, available, now, lane, waiting_ages)
        return replace(selected, estimated_arrival_rates=self.arrival_rates,
            observed_arrival_counts=tuple(counts), rate_exposure_s=exposure,
            rate_estimation_coordinate_reads=len(counts),
            rate_estimation_divisions=len(counts) if exposure > 0 else 0)
