"""Online ETA and lower-confidence service updates."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class ServiceSample:
    key: str
    units_done: float
    elapsed_s: float
    source: str = ""

    @property
    def rate(self) -> float:
        if self.elapsed_s <= 0.0:
            return 0.0
        return max(0.0, float(self.units_done) / float(self.elapsed_s))

    def snapshot(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "units_done": float(self.units_done),
            "elapsed_s": float(self.elapsed_s),
            "rate": self.rate,
            "source": self.source,
        }


@dataclass(frozen=True)
class ServiceEstimate:
    key: str
    sample_count: int
    mean_rate: float
    lcb_rate: float
    radius: float
    eta_s: float | None
    certified: bool
    confidence: float
    reused_from_cache: bool = False
    probe_required: bool = False

    def snapshot(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "sample_count": int(self.sample_count),
            "mean_rate": float(self.mean_rate),
            "lcb_rate": float(self.lcb_rate),
            "radius": float(self.radius),
            "eta_s": None if self.eta_s is None else float(self.eta_s),
            "certified": bool(self.certified),
            "confidence": float(self.confidence),
            "reused_from_cache": bool(self.reused_from_cache),
            "probe_required": bool(self.probe_required),
        }


@dataclass
class OnlineServiceEstimator:
    confidence: float = 0.95
    min_samples: int = 2
    min_observed_range: float = 1e-9
    _samples: dict[str, list[ServiceSample]] = field(default_factory=dict)

    def add_sample(self, sample: ServiceSample) -> None:
        if sample.rate <= 0.0 or not math.isfinite(sample.rate):
            return
        self._samples.setdefault(sample.key, []).append(sample)

    def add_progress(
        self,
        *,
        key: str,
        units_done: float,
        elapsed_s: float,
        source: str = "",
    ) -> None:
        self.add_sample(ServiceSample(key=key, units_done=units_done, elapsed_s=elapsed_s, source=source))

    def estimate(self, key: str, *, remaining_units: float | None = None) -> ServiceEstimate:
        samples = self._samples.get(key, [])
        rates = [sample.rate for sample in samples if sample.rate > 0.0 and math.isfinite(sample.rate)]
        if not rates:
            return ServiceEstimate(
                key=key,
                sample_count=0,
                mean_rate=0.0,
                lcb_rate=0.0,
                radius=0.0,
                eta_s=None,
                certified=False,
                confidence=float(self.confidence),
                reused_from_cache=False,
                probe_required=True,
            )
        mean = sum(rates) / float(len(rates))
        radius = empirical_bernstein_radius(
            rates,
            confidence=float(self.confidence),
            min_range=float(self.min_observed_range),
        )
        lcb = max(0.0, mean - radius)
        eta = None
        if remaining_units is not None and lcb > 0.0:
            eta = max(0.0, float(remaining_units)) / lcb
        return ServiceEstimate(
            key=key,
            sample_count=len(rates),
            mean_rate=mean,
            lcb_rate=lcb,
            radius=radius,
            eta_s=eta,
            certified=len(rates) >= int(self.min_samples) and lcb > 0.0,
            confidence=float(self.confidence),
            reused_from_cache=False,
            probe_required=not (len(rates) >= int(self.min_samples) and lcb > 0.0),
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "confidence": float(self.confidence),
            "min_samples": int(self.min_samples),
            "keys": {
                key: self.estimate(key).snapshot()
                for key in sorted(self._samples)
            },
        }


def service_observation_key(
    *,
    workload_key: str,
    profile: int,
    load_state: str,
    node_bucket: str = "",
    command_fingerprint: str = "",
    algorithm_mode: str = "",
) -> str:
    return "|".join(
        str(part).strip()
        for part in (
            workload_key,
            int(profile),
            load_state or "unknown",
            node_bucket or "",
            command_fingerprint or "",
            algorithm_mode or "",
        )
    )


def estimator_from_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    key_field: str = "key",
    units_field: str = "units_done",
    elapsed_field: str = "elapsed_s",
    confidence: float = 0.95,
    min_samples: int = 2,
) -> OnlineServiceEstimator:
    estimator = OnlineServiceEstimator(confidence=confidence, min_samples=min_samples)
    for row in rows:
        key = str(row.get(key_field) or "")
        if not key:
            continue
        estimator.add_progress(
            key=key,
            units_done=float(row.get(units_field) or 0.0),
            elapsed_s=float(row.get(elapsed_field) or 0.0),
            source=str(row.get("source") or ""),
        )
    return estimator


@dataclass
class ReusableEtaCache:
    """Thin ETA/probe cache over ``OnlineServiceEstimator``.

    A key is reusable only when it matches the workload, profile, load state,
    node bucket, command fingerprint, and algorithm mode.  This intentionally
    mirrors the theorem-facing finite-state admission rule: new co-location
    states must probe first, while repeated states use the lower-confidence
    estimate without relaunching a warm-up measurement.
    """

    estimator: OnlineServiceEstimator = field(default_factory=OnlineServiceEstimator)
    _last_certified: dict[str, ServiceEstimate] = field(default_factory=dict)
    query_count: int = 0
    reuse_count: int = 0
    probe_required_count: int = 0

    def observe(
        self,
        *,
        key: str,
        units_done: float,
        elapsed_s: float,
        remaining_units: float | None = None,
        source: str = "",
    ) -> ServiceEstimate:
        self.estimator.add_progress(
            key=key,
            units_done=units_done,
            elapsed_s=elapsed_s,
            source=source,
        )
        estimate = self.estimator.estimate(key, remaining_units=remaining_units)
        if estimate.certified:
            self._last_certified[key] = estimate
        return estimate

    def lookup(self, key: str, *, remaining_units: float | None = None) -> ServiceEstimate:
        self.query_count += 1
        estimate = self.estimator.estimate(key, remaining_units=remaining_units)
        if estimate.certified:
            self.reuse_count += 1
            estimate = _with_cache_flags(estimate, reused=True, probe=False)
            self._last_certified[key] = estimate
            return estimate
        cached = self._last_certified.get(key)
        if cached is not None and cached.certified:
            self.reuse_count += 1
            eta = None
            if remaining_units is not None and cached.lcb_rate > 0.0:
                eta = max(0.0, float(remaining_units)) / float(cached.lcb_rate)
            return ServiceEstimate(
                key=cached.key,
                sample_count=cached.sample_count,
                mean_rate=cached.mean_rate,
                lcb_rate=cached.lcb_rate,
                radius=cached.radius,
                eta_s=eta,
                certified=True,
                confidence=cached.confidence,
                reused_from_cache=True,
                probe_required=False,
            )
        self.probe_required_count += 1
        return _with_cache_flags(estimate, reused=False, probe=True)

    def snapshot(self) -> dict[str, Any]:
        return {
            "query_count": int(self.query_count),
            "reuse_count": int(self.reuse_count),
            "probe_required_count": int(self.probe_required_count),
            "reuse_fraction": (
                float(self.reuse_count) / float(self.query_count)
                if self.query_count else 0.0
            ),
            "estimator": self.estimator.snapshot(),
        }


def empirical_bernstein_radius(
    values: list[float],
    *,
    confidence: float = 0.95,
    min_range: float = 1e-9,
) -> float:
    n = len(values)
    if n <= 1:
        return float("inf")
    mean = sum(values) / float(n)
    variance = sum((value - mean) ** 2 for value in values) / float(n - 1)
    delta = max(1e-12, 1.0 - float(confidence))
    log_term = math.log(3.0 / delta)
    observed_range = max(max(values) - min(values), float(min_range))
    return (
        math.sqrt(2.0 * variance * log_term / float(n))
        + 3.0 * observed_range * log_term / float(n - 1)
    )


def _with_cache_flags(
    estimate: ServiceEstimate,
    *,
    reused: bool,
    probe: bool,
) -> ServiceEstimate:
    return ServiceEstimate(
        key=estimate.key,
        sample_count=estimate.sample_count,
        mean_rate=estimate.mean_rate,
        lcb_rate=estimate.lcb_rate,
        radius=estimate.radius,
        eta_s=estimate.eta_s,
        certified=estimate.certified,
        confidence=estimate.confidence,
        reused_from_cache=bool(reused),
        probe_required=bool(probe),
    )
