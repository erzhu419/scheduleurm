"""Service-rate cache built from real Scheduleurm experiment traces."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable


_PROFILE_RE = re.compile(r"profile_(\d+)_(?:per_gpu|per_resource)")


@dataclass(frozen=True)
class ProfileRecord:
    workload_key: str
    command_fingerprint: str
    resource_kind: str
    node_bucket: str
    profile: int
    unit: str
    total_units: float
    aggregate_rate: float
    per_task_rates: tuple[float, ...]
    source: str
    capacity_boundary: bool = False
    gpu_count_observed: int = 1

    @property
    def mean_rate(self) -> float:
        if self.profile <= 0:
            return 0.0
        return float(self.aggregate_rate) / float(self.profile)

    def snapshot(self) -> dict[str, Any]:
        return {
            "workload_key": self.workload_key,
            "command_fingerprint": self.command_fingerprint,
            "resource_kind": self.resource_kind,
            "node_bucket": self.node_bucket,
            "profile": self.profile,
            "unit": self.unit,
            "total_units": self.total_units,
            "aggregate_rate": self.aggregate_rate,
            "per_task_rates": list(self.per_task_rates),
            "source": self.source,
            "capacity_boundary": self.capacity_boundary,
            "gpu_count_observed": self.gpu_count_observed,
        }

    @staticmethod
    def from_snapshot(data: dict[str, Any]) -> "ProfileRecord":
        return ProfileRecord(
            workload_key=str(data["workload_key"]),
            command_fingerprint=str(data.get("command_fingerprint") or ""),
            resource_kind=str(data.get("resource_kind") or "hybrid"),
            node_bucket=str(data.get("node_bucket") or ""),
            profile=int(data["profile"]),
            unit=str(data.get("unit") or "unit"),
            total_units=float(data.get("total_units") or 1.0),
            aggregate_rate=float(data.get("aggregate_rate") or 0.0),
            per_task_rates=tuple(float(x) for x in (data.get("per_task_rates") or [])),
            source=str(data.get("source") or ""),
            capacity_boundary=bool(data.get("capacity_boundary")),
            gpu_count_observed=max(1, int(data.get("gpu_count_observed") or 1)),
        )


class ServiceRateCache:
    """Small in-memory cache keyed by workload and co-location profile."""

    def __init__(self, records: Iterable[ProfileRecord] = ()):
        self._records: dict[tuple[str, int], ProfileRecord] = {}
        for record in records:
            self.add(record)

    def add(self, record: ProfileRecord) -> None:
        key = (record.workload_key, int(record.profile))
        old = self._records.get(key)
        if old is None or _record_quality(record) >= _record_quality(old):
            self._records[key] = record

    def get(self, workload_key: str, profile: int) -> ProfileRecord | None:
        return self._records.get((workload_key, int(profile)))

    def profiles(self, workload_key: str, *, include_boundaries: bool = False) -> list[ProfileRecord]:
        records = [
            record for (key, _), record in self._records.items()
            if key == workload_key and (include_boundaries or not record.capacity_boundary)
        ]
        return sorted(records, key=lambda record: record.profile)

    def available_workloads(self) -> list[str]:
        return sorted({key for key, _ in self._records})

    def best_profile_for_makespan(
        self,
        workload_key: str,
        *,
        task_count: int,
        total_units: float,
        resource_count: int = 1,
    ) -> ProfileRecord:
        candidates = self.profiles(workload_key)
        if not candidates:
            raise KeyError(f"no service cache entries for workload {workload_key!r}")
        return min(
            candidates,
            key=lambda record: (
                deterministic_makespan_s(
                    task_count=task_count,
                    total_units=total_units,
                    resource_count=resource_count,
                    profile=record.profile,
                    aggregate_rate=record.aggregate_rate,
                ),
                record.profile,
            ),
        )

    def best_profile_for_guarded_mean_flow(
        self,
        workload_key: str,
        *,
        task_count: int,
        total_units: float,
        resource_count: int = 1,
        max_makespan_regret: float = 0.02,
    ) -> ProfileRecord:
        """Choose a delay-friendly profile inside a makespan-regret guard.

        The guard keeps the policy close to the support/max-throughput action,
        while the mean-flow tie-break is the bounded scheduling penalty used by
        the robust MaxWeight route. Missing profiles are never interpolated.
        """

        candidates = self.profiles(workload_key)
        if not candidates:
            raise KeyError(f"no service cache entries for workload {workload_key!r}")
        rows = []
        for record in candidates:
            makespan = deterministic_makespan_s(
                task_count=task_count,
                total_units=total_units,
                resource_count=resource_count,
                profile=record.profile,
                aggregate_rate=record.aggregate_rate,
            )
            rows.append(
                (
                    record,
                    makespan,
                    deterministic_mean_flow_s(
                        task_count=task_count,
                        total_units=total_units,
                        resource_count=resource_count,
                        profile=record.profile,
                        aggregate_rate=record.aggregate_rate,
                    ),
                )
            )
        best_makespan = min(row[1] for row in rows)
        guard = best_makespan * (1.0 + max(0.0, float(max_makespan_regret)))
        admissible = [row for row in rows if row[1] <= guard]
        if not admissible:
            admissible = rows
        return min(admissible, key=lambda row: (row[2], row[1], row[0].profile))[0]

    def snapshot(self) -> dict[str, Any]:
        return {"records": [record.snapshot() for record in sorted(self._records.values(), key=lambda r: (r.workload_key, r.profile))]}

    @staticmethod
    def from_snapshot(data: dict[str, Any]) -> "ServiceRateCache":
        return ServiceRateCache(ProfileRecord.from_snapshot(row) for row in data.get("records") or [])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.snapshot(), indent=2, sort_keys=True), encoding="utf-8")

    @staticmethod
    def load(path: Path) -> "ServiceRateCache":
        return ServiceRateCache.from_snapshot(json.loads(path.read_text(encoding="utf-8")))


def deterministic_makespan_s(
    *,
    task_count: int,
    total_units: float,
    resource_count: int,
    profile: int,
    aggregate_rate: float,
) -> float:
    slots = max(1, int(resource_count)) * max(1, int(profile))
    waves = (max(0, int(task_count)) + slots - 1) // slots
    mean_rate = float(aggregate_rate) / float(max(1, int(profile)))
    if mean_rate <= 0:
        return float("inf")
    return float(waves) * float(total_units) / mean_rate


def deterministic_mean_flow_s(
    *,
    task_count: int,
    total_units: float,
    resource_count: int,
    profile: int,
    aggregate_rate: float,
) -> float:
    n = max(0, int(task_count))
    if n <= 0:
        return 0.0
    slots = max(1, int(resource_count)) * max(1, int(profile))
    mean_rate = float(aggregate_rate) / float(max(1, int(profile)))
    if mean_rate <= 0:
        return float("inf")
    wave_seconds = float(total_units) / mean_rate
    remaining = n
    wave = 0
    total_completion = 0.0
    while remaining > 0:
        wave += 1
        done = min(slots, remaining)
        total_completion += float(done) * float(wave) * wave_seconds
        remaining -= done
    return total_completion / float(n)


def cache_needs_probe(cache: ServiceRateCache, workload_key: str, profile: int) -> bool:
    record = cache.get(workload_key, profile)
    return record is None or record.capacity_boundary or record.aggregate_rate <= 0


def missing_exact_profiles(cache: ServiceRateCache, workload_key: str, profiles: Iterable[int]) -> list[int]:
    """Profiles that still need real measurement before replay can use them."""

    return [
        int(profile)
        for profile in sorted({int(x) for x in profiles})
        if cache_needs_probe(cache, workload_key, int(profile))
    ]


def records_from_summary_file(
    path: Path,
    *,
    workload_key: str,
    command_fingerprint: str,
    resource_kind: str,
    total_units: float,
    node_bucket: str = "",
) -> list[ProfileRecord]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    profile = _profile_from_summary(path, summary)
    aggregate = _aggregate_rate(summary)
    rates = _per_task_rates(summary)
    gpu_count = _gpu_count(summary)
    aggregate_per_resource = aggregate / float(gpu_count)
    rates_per_resource = tuple(float(x) for x in rates)
    unit = _unit(summary)
    unusable = _summary_is_unusable_for_exact_replay(summary)
    return [
        ProfileRecord(
            workload_key=workload_key,
            command_fingerprint=command_fingerprint,
            resource_kind=resource_kind,
            node_bucket=node_bucket,
            profile=profile,
            unit=unit,
            total_units=float(total_units),
            aggregate_rate=float(aggregate_per_resource),
            per_task_rates=rates_per_resource,
            source=str(path),
            capacity_boundary=unusable,
            gpu_count_observed=gpu_count,
        )
    ]


def add_summary_glob(
    cache: ServiceRateCache,
    glob_pattern: str,
    *,
    workload_key: str,
    command_fingerprint: str,
    resource_kind: str,
    total_units: float,
    node_bucket: str = "",
) -> None:
    for path in sorted(Path().glob(glob_pattern) if not glob_pattern.startswith("/") else Path("/").glob(glob_pattern[1:])):
        for record in records_from_summary_file(
            path,
            workload_key=workload_key,
            command_fingerprint=command_fingerprint,
            resource_kind=resource_kind,
            total_units=total_units,
            node_bucket=node_bucket,
        ):
            cache.add(record)


def add_protocol_cpu_curve(
    cache: ServiceRateCache,
    *,
    workload_key: str = "cpu_heavy",
    total_units: float = 3600.0,
    max_workers: int = 32,
    per_worker_rate: float = 1.0,
    saturation_workers: int = 16,
) -> None:
    for workers in range(1, max_workers + 1):
        effective = min(workers, saturation_workers)
        aggregate = float(effective) * float(per_worker_rate)
        if workers > saturation_workers:
            aggregate *= max(0.4, 1.0 - 0.015 * (workers - saturation_workers))
        cache.add(
            ProfileRecord(
                workload_key=workload_key,
                command_fingerprint="protocol_cpu_linear_saturation_v1",
                resource_kind="cpu_heavy",
                node_bucket="cpu_pool",
                profile=workers,
                unit="cpu_unit",
                total_units=float(total_units),
                aggregate_rate=max(0.001, aggregate),
                per_task_rates=tuple([max(0.001, aggregate / workers)] * workers),
                source="protocol:cpu_linear_saturation_v1",
            )
        )


def _record_quality(record: ProfileRecord) -> tuple[int, int, float]:
    return (
        0 if record.capacity_boundary else 1,
        len(record.per_task_rates),
        float(record.aggregate_rate),
    )


def _profile_from_summary(path: Path, summary: dict[str, Any]) -> int:
    text = str(summary.get("phase") or path.name)
    match = _PROFILE_RE.search(text)
    if not match:
        raise ValueError(f"cannot infer profile count from {path}")
    return int(match.group(1))


def _aggregate_rate(summary: dict[str, Any]) -> float:
    for key in (
        "measurement_aggregate_rate_unit_s_median",
        "measurement_aggregate_rate_step_s_median",
        "aggregate_active_rate_unit_s",
        "aggregate_active_rate_step_s",
    ):
        if summary.get(key) is not None:
            return float(summary.get(key) or 0.0)
    return 0.0


def _per_task_rates(summary: dict[str, Any]) -> tuple[float, ...]:
    for key in ("rates_unit_s", "rates_step_s"):
        rates = [float(x) for x in (summary.get(key) or []) if float(x) > 0]
        if rates:
            return tuple(rates)
    running = max(1, int(summary.get("running_count") or 1))
    aggregate = _aggregate_rate(summary)
    return tuple([aggregate / running] * running) if aggregate > 0 else tuple()


def _gpu_count(summary: dict[str, Any]) -> int:
    per_gpu = summary.get("expected_per_gpu_running") or summary.get("per_gpu_running") or {}
    if isinstance(per_gpu, dict) and per_gpu:
        return max(1, len(per_gpu))
    return 1


def _unit(summary: dict[str, Any]) -> str:
    units = [str(x) for x in (summary.get("rate_units") or []) if str(x)]
    if units:
        return sorted(set(units))[0]
    return "unit"


def _summary_is_unusable_for_exact_replay(summary: dict[str, Any]) -> bool:
    """True when a measured profile should not be used as a valid service curve."""

    if bool(summary.get("capacity_boundary")):
        return True
    if summary.get("placement_valid") is False:
        return True
    if int(summary.get("blocked_count") or 0) > 0:
        return True
    statuses = summary.get("status_counts") or {}
    if isinstance(statuses, dict):
        for key in ("failed", "cancelled", "killed"):
            if int(statuses.get(key) or 0) > 0:
                return True
    return False
