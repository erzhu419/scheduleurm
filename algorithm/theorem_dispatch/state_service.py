"""State-dependent marginal service cache.

The base Scheduleurm replay cache is keyed by workload and co-location profile.
This module adds an explicit load-state dimension for measured marginal rows
such as empty GPU, high-VRAM-resident GPU, empty CPU, and CPU-resident nodes.
Unknown states fall back to the base cache only when the caller asks for a
fallback; theorem-facing gates can keep the fallback flag visible.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from simulation.service_cache import ProfileRecord, ServiceRateCache


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_CACHE_PATH = REPO_ROOT / "md" / "experiment_artifacts" / "corner_case_service_cache_20260613.json"


@dataclass(frozen=True)
class StateServiceLookup:
    workload_key: str
    profile: int
    load_state: str
    record: ProfileRecord | None
    certified: bool
    fallback_used: bool
    reason: str

    def snapshot(self) -> dict[str, Any]:
        return {
            "workload_key": self.workload_key,
            "profile": int(self.profile),
            "load_state": self.load_state,
            "certified": bool(self.certified),
            "fallback_used": bool(self.fallback_used),
            "reason": self.reason,
            "record": self.record.snapshot() if self.record is not None else None,
        }


class StateDependentServiceCache:
    def __init__(
        self,
        *,
        base_cache: ServiceRateCache | None = None,
        state_records: Iterable[ProfileRecord] = (),
    ):
        self.base_cache = base_cache
        self._state_records: dict[tuple[str, int, str], ProfileRecord] = {}
        for record in state_records:
            self.add(record)

    @staticmethod
    def load(
        path: str | Path = DEFAULT_STATE_CACHE_PATH,
        *,
        base_cache: ServiceRateCache | None = None,
    ) -> "StateDependentServiceCache":
        p = Path(path).expanduser()
        records: list[ProfileRecord] = []
        if p.exists():
            records = _snapshot_records(ServiceRateCache.load(p))
        return StateDependentServiceCache(base_cache=base_cache, state_records=records)

    @staticmethod
    def from_cache(
        cache: ServiceRateCache,
        *,
        base_cache: ServiceRateCache | None = None,
    ) -> "StateDependentServiceCache":
        return StateDependentServiceCache(
            base_cache=base_cache,
            state_records=_snapshot_records(cache),
        )

    def add(self, record: ProfileRecord) -> None:
        state = load_state_from_record(record)
        semantic = semantic_workload_from_record(record)
        key = (semantic, int(record.profile), state)
        old = self._state_records.get(key)
        if old is None or _record_quality(record) >= _record_quality(old):
            self._state_records[key] = record

    def lookup(
        self,
        workload_key: str,
        profile: int,
        load_state: str,
        *,
        allow_base_fallback: bool = True,
    ) -> StateServiceLookup:
        semantic = semantic_workload_key(workload_key)
        state = str(load_state or "unknown")
        key = (semantic, int(profile), state)
        record = self._state_records.get(key)
        if record is not None and not record.capacity_boundary and record.aggregate_rate > 0.0:
            return StateServiceLookup(
                workload_key=semantic,
                profile=int(profile),
                load_state=state,
                record=record,
                certified=True,
                fallback_used=False,
                reason="state_exact",
            )
        if allow_base_fallback and self.base_cache is not None:
            base = self.base_cache.get(workload_key, int(profile))
            if base is not None and not base.capacity_boundary and base.aggregate_rate > 0.0:
                return StateServiceLookup(
                    workload_key=str(workload_key),
                    profile=int(profile),
                    load_state=state,
                    record=base,
                    certified=True,
                    fallback_used=True,
                    reason="base_profile_fallback",
                )
        return StateServiceLookup(
            workload_key=semantic,
            profile=int(profile),
            load_state=state,
            record=None,
            certified=False,
            fallback_used=False,
            reason="state_profile_missing",
        )

    def available_states(self) -> dict[str, list[str]]:
        out: dict[str, set[str]] = {}
        for workload, _profile, state in self._state_records:
            out.setdefault(workload, set()).add(state)
        return {key: sorted(value) for key, value in sorted(out.items())}

    def marginal_opportunity_rows(
        self,
        workload_key: str,
        *,
        profile: int = 1,
        reference_state: str = "empty",
    ) -> list[dict[str, Any]]:
        semantic = semantic_workload_key(workload_key)
        reference = self.lookup(
            semantic,
            profile,
            reference_state,
            allow_base_fallback=False,
        )
        rows = []
        for state in self.available_states().get(semantic, []):
            lookup = self.lookup(semantic, profile, state, allow_base_fallback=False)
            ratio = 0.0
            if (
                lookup.certified
                and reference.certified
                and reference.record is not None
                and lookup.record is not None
                and reference.record.aggregate_rate > 0.0
            ):
                ratio = float(lookup.record.aggregate_rate) / float(reference.record.aggregate_rate)
            rows.append(
                {
                    "workload_key": semantic,
                    "profile": int(profile),
                    "load_state": state,
                    "certified": bool(lookup.certified),
                    "fallback_used": bool(lookup.fallback_used),
                    "aggregate_rate": 0.0 if lookup.record is None else float(lookup.record.aggregate_rate),
                    "relative_to_reference": float(ratio),
                    "reference_state": reference_state,
                }
            )
        return rows


def classify_load_state(
    *,
    resource_kind: str,
    running_task_count: int = 0,
    used_mb: int = 0,
    total_mb: int = 0,
    cpu_workers: int = 0,
    total_cpu: int = 0,
) -> str:
    kind = str(resource_kind or "").lower()
    if kind.startswith("gpu"):
        frac = float(max(0, used_mb)) / float(max(1, total_mb))
        if as_boolish_busy(running_task_count, frac, util_threshold=80):
            return "gpu_util_resident"
        if frac >= 0.85:
            return "vram_saturated_resident"
        if frac >= 0.70:
            return "high_vram_resident"
        if running_task_count >= 4 or frac >= 0.50:
            return "full_loaded"
        if running_task_count >= 2 or frac >= 0.20:
            return "half_loaded"
        return "empty"
    if kind.startswith("cpu"):
        denom = max(1, int(total_cpu))
        frac = float(max(0, cpu_workers)) / float(denom)
        if frac >= 0.70:
            return "cpu_full_resident"
        if frac >= 0.35:
            return "cpu_half_resident"
        return "empty"
    return "unknown"


def as_boolish_busy(running_task_count: int, frac: float, *, util_threshold: int = 80) -> bool:
    return int(running_task_count) >= 6 and float(frac) >= float(util_threshold) / 100.0


def semantic_workload_key(workload_key: str) -> str:
    key = str(workload_key or "")
    if "marginal_cuda" in key or key.startswith("gpu_") or key.startswith("corner_gpu"):
        return "marginal_cuda"
    if "cpu" in key:
        return "marginal_cpu"
    return key


def semantic_workload_from_record(record: ProfileRecord) -> str:
    return semantic_workload_key(record.workload_key)


def load_state_from_record(record: ProfileRecord) -> str:
    text = f"{record.workload_key} {record.resource_kind} {record.node_bucket}".lower()
    if "saturated" in text:
        return "vram_saturated_resident"
    if "30gb" in text or "vram" in text:
        return "high_vram_resident"
    if "full_loaded" in text or "full-resident" in text:
        return "full_loaded"
    if "half_loaded" in text or "half-resident" in text:
        return "half_loaded"
    if "180worker" in text:
        return "cpu_full_resident"
    if "96worker" in text:
        return "cpu_half_resident"
    if "empty" in text:
        return "empty"
    return "unknown"


def _record_quality(record: ProfileRecord) -> tuple[int, int, float]:
    return (
        0 if record.capacity_boundary else 1,
        len(record.per_task_rates),
        float(record.aggregate_rate),
    )


def _snapshot_records(cache: ServiceRateCache) -> list[ProfileRecord]:
    return [ProfileRecord.from_snapshot(row) for row in cache.snapshot().get("records") or []]
