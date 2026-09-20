"""Sample one planning path from caller-owned, independent per-class RNGs.

The first exponential draw beyond the shared horizon is charged but is not a
stored arrival. Zero-rate classes consume no draws. The caller retains RNG
state between active decisions; this module neither creates nor seeds RNGs.
"""
from __future__ import annotations

from dataclasses import dataclass

from algorithm.arrival_rollout_terminal import ArrivalPath


@dataclass(frozen=True)
class PoissonArrivalPath(ArrivalPath):
    random_draws: int


def build_poisson_arrival_path(arrival_rates, horizon, arrival_rngs) -> PoissonArrivalPath:
    """Accumulate class-specific exponential gaps and retain arrivals at H."""
    if len(arrival_rates) != len(arrival_rngs):
        raise ValueError("arrival_rngs must match the registered arrival-rate coordinates")
    times = []
    generated = rate_reads = random_draws = 0
    for rate, rng in zip(arrival_rates, arrival_rngs):
        rate_reads += 1
        events = []
        if rate > 0.0:
            time = 0.0
            while True:
                time += rng.expovariate(rate)
                random_draws += 1
                if time > horizon:
                    break
                events.append(time)
                generated += 1
        times.append(tuple(events))
    return PoissonArrivalPath(horizon, tuple(times), generated, rate_reads, random_draws)
