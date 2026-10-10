"""Matched periodic rate modulation of the original Poisson arrival tape.

The physical intensity alternates between 1.5 and 0.5 times each class's
original rate for 50 seconds each. Its integrated clock is the identity at
every 100-second endpoint. Inverting that clock changes only arrival times:
class/job identities, their FIFO order and the jobs born in [300, 900] remain
exactly matched. A separate seed stream chooses the common high/low-first
phase; this tests synchronized periodic nonstationarity, not random burst
sizes or overdispersion in full-period arrival counts.

The modulation is an environment-only transformation. Its phase, true rates
and clock are never planner inputs.
"""
from __future__ import annotations

import math
import random
from typing import Sequence

from algorithm.experiments.aligned_hardware_frame_experiment import poisson_arrivals


PERIOD_S = 100.0
HALF_PERIOD_S = 50.0
HIGH_RATE_MULTIPLIER = 1.5
LOW_RATE_MULTIPLIER = 0.5
PHASE_SEED_RULE = "arrival-modulation-v1:{seed}"
Arrival = tuple[float, int, int]


def modulation_high_first(seed: int) -> bool:
    """One independent phase bit shared by all classes and cells of a seed."""
    return bool(random.Random(PHASE_SEED_RULE.format(seed=seed)).randrange(2))


def cumulative_modulated_time(time: float, high_first: bool) -> float:
    """Integrated intensity multiplier from physical time zero to ``time``."""
    cycle = math.floor(time / PERIOD_S)
    within = time - cycle * PERIOD_S
    first = HIGH_RATE_MULTIPLIER if high_first else LOW_RATE_MULTIPLIER
    second = LOW_RATE_MULTIPLIER if high_first else HIGH_RATE_MULTIPLIER
    return (cycle * PERIOD_S + first * min(within, HALF_PERIOD_S)
            + second * max(0.0, within - HALF_PERIOD_S))


def inverse_modulated_time(time: float, high_first: bool) -> float:
    """Convert an original Poisson operational-clock timestamp to physical time."""
    cycle = math.floor(time / PERIOD_S)
    within = time - cycle * PERIOD_S
    first = HIGH_RATE_MULTIPLIER if high_first else LOW_RATE_MULTIPLIER
    second = LOW_RATE_MULTIPLIER if high_first else HIGH_RATE_MULTIPLIER
    first_exposure = HALF_PERIOD_S * first
    physical = (within / first if within <= first_exposure else
                HALF_PERIOD_S + (within - first_exposure) / second)
    return cycle * PERIOD_S + physical


def modulate_arrival_tape(arrivals: Sequence[Arrival], seed: int) -> tuple[Arrival, ...]:
    """Return the chronological matched tape, retaining every original job ID."""
    high_first = modulation_high_first(seed)
    return tuple(sorted((inverse_modulated_time(time, high_first), coordinate, index)
                        for time, coordinate, index in arrivals))


def matched_bursty_arrivals(
    rates: Sequence[float], horizon: float, seed: int,
) -> tuple[Arrival, ...]:
    """Modulate ``poisson_arrivals``; use the registered 1200-second endpoint."""
    return modulate_arrival_tape(poisson_arrivals(rates, horizon, seed), seed)
