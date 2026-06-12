"""Active-bucket sampler and bounded regime-change detector.

These classes are deployment-ready algorithm components, but they are not wired
into ``skill/scheduler.py`` by default.  They make the stochastic objects used by
the learning/regime certificate concrete enough to test and audit before a live
A/B hook is enabled.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class BucketCandidate:
    bucket: str
    score: float
    payload: Any = None


@dataclass(frozen=True)
class SampleDecision:
    selected: BucketCandidate
    bucket_counts_before: dict[str, int]
    forced_exploration: bool


class RoundRobinActiveBucketSampler:
    """Deterministic forced sampler over the finite active bucket set."""

    def __init__(self, *, min_samples_per_bucket: int = 128) -> None:
        self.min_samples_per_bucket = max(1, int(min_samples_per_bucket))
        self._counts: defaultdict[str, int] = defaultdict(int)

    def choose(self, candidates: Iterable[BucketCandidate | Mapping[str, Any]]) -> SampleDecision:
        rows = [_as_candidate(candidate) for candidate in candidates]
        if not rows:
            raise ValueError("at least one candidate is required")
        by_bucket: dict[str, list[BucketCandidate]] = {}
        for row in rows:
            by_bucket.setdefault(row.bucket, []).append(row)
        active = sorted(by_bucket)
        counts_before = {bucket: int(self._counts[bucket]) for bucket in active}
        under_sampled = [
            bucket for bucket in active
            if self._counts[bucket] < self.min_samples_per_bucket
        ]
        if under_sampled:
            chosen_bucket = min(under_sampled, key=lambda bucket: (self._counts[bucket], bucket))
            forced = True
        else:
            chosen_bucket = max(
                active,
                key=lambda bucket: (
                    max(candidate.score for candidate in by_bucket[bucket]),
                    -self._counts[bucket],
                    bucket,
                ),
            )
            forced = False
        selected = max(by_bucket[chosen_bucket], key=lambda candidate: (candidate.score, str(candidate.payload)))
        self._counts[selected.bucket] += 1
        return SampleDecision(
            selected=selected,
            bucket_counts_before=counts_before,
            forced_exploration=forced,
        )

    def counts(self) -> dict[str, int]:
        return dict(self._counts)

    def coverage_certificate(self, active_buckets: Iterable[str]) -> dict[str, Any]:
        buckets = sorted({str(bucket) for bucket in active_buckets})
        min_count = min((self._counts[bucket] for bucket in buckets), default=0)
        return {
            "model": "deterministic_round_robin_forced_active_bucket_sampler",
            "active_bucket_count": len(buckets),
            "min_observed_samples_per_bucket": int(min_count),
            "required_min_samples_per_bucket": self.min_samples_per_bucket,
            "coverage_probability_lower_bound": 1.0 if min_count >= self.min_samples_per_bucket else 0.0,
            "coverage_certified": bool(buckets) and min_count >= self.min_samples_per_bucket,
        }


@dataclass(frozen=True)
class ChangeEvent:
    regime_key: str
    index: int
    previous_mean: float
    current_mean: float
    shift: float


class TwoWindowMeanShiftDetector:
    """Bounded two-window detector with Hoeffding tail accounting."""

    def __init__(
        self,
        *,
        window_size: int = 512,
        threshold: float = 0.20,
        min_shift: float = 0.40,
    ) -> None:
        self.window_size = max(1, int(window_size))
        self.threshold = max(0.0, float(threshold))
        self.min_shift = max(0.0, float(min_shift))
        self._windows: dict[str, deque[float]] = {}
        self._indices: defaultdict[str, int] = defaultdict(int)
        self._last_event_index: defaultdict[str, int] = defaultdict(lambda: -10**18)

    def update(self, regime_key: str, value: float) -> ChangeEvent | None:
        key = str(regime_key)
        bounded = min(1.0, max(0.0, float(value)))
        window = self._windows.setdefault(key, deque(maxlen=2 * self.window_size))
        window.append(bounded)
        self._indices[key] += 1
        if len(window) < 2 * self.window_size:
            return None
        previous = list(window)[: self.window_size]
        current = list(window)[self.window_size :]
        previous_mean = sum(previous) / float(self.window_size)
        current_mean = sum(current) / float(self.window_size)
        shift = current_mean - previous_mean
        index = self._indices[key]
        if abs(shift) < self.threshold:
            return None
        if index - self._last_event_index[key] < self.window_size:
            return None
        self._last_event_index[key] = index
        return ChangeEvent(
            regime_key=key,
            index=index,
            previous_mean=previous_mean,
            current_mean=current_mean,
            shift=shift,
        )

    def tail_certificate(self, *, switching_count: int, failure_budget: float = 0.025) -> dict[str, Any]:
        miss = detector_miss_bound(
            window_size=self.window_size,
            threshold=self.threshold,
            min_shift=self.min_shift,
        )
        false_alarm = detector_false_alarm_bound(
            window_size=self.window_size,
            threshold=self.threshold,
        )
        switches = max(0, int(switching_count))
        union = min(1.0, switches * miss + max(1, switches + 1) * false_alarm)
        return {
            "model": "bounded_two_window_mean_shift_detector",
            "window_size": self.window_size,
            "threshold": self.threshold,
            "min_shift": self.min_shift,
            "switching_count": switches,
            "per_change_miss_bound": miss,
            "per_window_false_alarm_bound": false_alarm,
            "union_tail_bound": union,
            "allocated_failure_budget": float(failure_budget),
            "detection_delay_bound_decisions": 2 * self.window_size,
            "change_point_detector_tail_certified": (
                switches > 0
                and self.min_shift > self.threshold
                and union <= float(failure_budget)
            ),
        }


def sampler_hoeffding_radius(
    *,
    sample_count: int,
    active_bucket_count: int,
    failure_budget: float,
) -> float:
    n = max(1, int(sample_count))
    buckets = max(1, int(active_bucket_count))
    delta = max(1e-12, float(failure_budget) / buckets)
    return math.sqrt(math.log(2.0 / delta) / (2.0 * n))


def detector_miss_bound(*, window_size: int, threshold: float, min_shift: float) -> float:
    m = max(1, int(window_size))
    tau = max(0.0, float(threshold))
    gap = max(0.0, float(min_shift))
    if gap <= tau:
        return 1.0
    eps = 0.5 * (gap - tau)
    return min(1.0, 2.0 * math.exp(-2.0 * m * eps * eps))


def detector_false_alarm_bound(*, window_size: int, threshold: float) -> float:
    m = max(1, int(window_size))
    tau = max(0.0, float(threshold))
    return min(1.0, 2.0 * math.exp(-2.0 * m * (0.5 * tau) * (0.5 * tau)))


def _as_candidate(candidate: BucketCandidate | Mapping[str, Any]) -> BucketCandidate:
    if isinstance(candidate, BucketCandidate):
        return candidate
    bucket = str(candidate.get("candidate_bucket") or candidate.get("bucket") or "")
    if not bucket:
        raise ValueError("candidate bucket is required")
    return BucketCandidate(
        bucket=bucket,
        score=float(candidate.get("score") or candidate.get("robust_score") or 0.0),
        payload=candidate,
    )
