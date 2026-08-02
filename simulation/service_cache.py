"""Service-rate cache built from real Scheduleurm experiment traces."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable


_PROFILE_RE = re.compile(r"profile_(\d+)_(?:per_gpu|per_resource)")

STATEWISE_LOOKUP_EXACT = "exact"
STATEWISE_LOOKUP_MISSING = "missing"
STATEWISE_LOOKUP_SCOPED_BOUNDARY = "scoped_boundary"
_STATEWISE_LOOKUP_STATUSES = frozenset(
    {
        STATEWISE_LOOKUP_EXACT,
        STATEWISE_LOOKUP_MISSING,
        STATEWISE_LOOKUP_SCOPED_BOUNDARY,
    }
)


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
    workload_env: str = ""
    resource_state: str = "unspecified"
    resident_mix: str = ""
    eta_source: str = ""
    stable_rate_ready: bool = False
    hardware_class: str = ""
    completion_model_ready: bool = False
    completion_model_sample_count: int = 0
    startup_overhead_s: float = 0.0
    completion_unit_s: float = 0.0
    finalization_overhead_s: float = 0.0
    checkpoint_observed_s: float = 0.0
    save_observed_s: float = 0.0
    completion_total_wall_s: float = 0.0
    completion_model_relative_error: float = 0.0
    allocation_workers: int = 1
    colocation_count: int = 0

    def __post_init__(self) -> None:
        """Normalize the explicit CPU allocation and co-location dimensions.

        ``profile`` remains the compatibility alias for co-location count.
        Historical records omit both new fields, so their old profile value
        continues to mean one worker per colocated task.
        """

        workers = max(1, int(self.allocation_workers or 1))
        colocation = max(1, int(self.colocation_count or self.profile or 1))
        object.__setattr__(self, "allocation_workers", workers)
        object.__setattr__(self, "colocation_count", colocation)
        object.__setattr__(self, "profile", colocation)

    @property
    def mean_rate(self) -> float:
        if self.profile <= 0:
            return 0.0
        return float(self.aggregate_rate) / float(self.profile)

    def completion_eta_s(
        self,
        *,
        completed_units: float = 0.0,
        include_startup: bool = False,
    ) -> float:
        """Return phase-aware remaining time for a naturally completed run.

        Service-only stable probes intentionally leave this unavailable.  This
        prevents steady-state throughput from being presented as end-to-end
        completion time when initialization or terminal writes are material.
        """

        if not self.completion_model_ready:
            raise ValueError("profile has no natural-completion ETA model")
        remaining_units = max(0.0, float(self.total_units) - float(completed_units))
        eta = remaining_units * max(0.0, float(self.completion_unit_s))
        eta += max(0.0, float(self.finalization_overhead_s))
        if include_startup:
            eta += max(0.0, float(self.startup_overhead_s))
        return eta

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
            "allocation_workers": self.allocation_workers,
            "colocation_count": self.colocation_count,
            "workload_env": self.workload_env,
            "resource_state": self.resource_state,
            "resident_mix": self.resident_mix,
            "eta_source": self.eta_source,
            "stable_rate_ready": self.stable_rate_ready,
            "hardware_class": self.hardware_class,
            "completion_model_ready": self.completion_model_ready,
            "completion_model_sample_count": self.completion_model_sample_count,
            "startup_overhead_s": self.startup_overhead_s,
            "completion_unit_s": self.completion_unit_s,
            "finalization_overhead_s": self.finalization_overhead_s,
            "checkpoint_observed_s": self.checkpoint_observed_s,
            "save_observed_s": self.save_observed_s,
            "completion_total_wall_s": self.completion_total_wall_s,
            "completion_model_relative_error": self.completion_model_relative_error,
        }

    @staticmethod
    def from_snapshot(data: dict[str, Any]) -> "ProfileRecord":
        legacy_profile = int(
            data.get("profile") or data.get("colocation_count") or 1
        )
        return ProfileRecord(
            workload_key=str(data["workload_key"]),
            command_fingerprint=str(data.get("command_fingerprint") or ""),
            resource_kind=str(data.get("resource_kind") or "hybrid"),
            node_bucket=str(data.get("node_bucket") or ""),
            profile=legacy_profile,
            unit=str(data.get("unit") or "unit"),
            total_units=float(data.get("total_units") or 1.0),
            aggregate_rate=float(data.get("aggregate_rate") or 0.0),
            per_task_rates=tuple(float(x) for x in (data.get("per_task_rates") or [])),
            source=str(data.get("source") or ""),
            capacity_boundary=bool(data.get("capacity_boundary")),
            gpu_count_observed=max(1, int(data.get("gpu_count_observed") or 1)),
            allocation_workers=max(1, int(data.get("allocation_workers") or 1)),
            colocation_count=max(
                1, int(data.get("colocation_count") or legacy_profile)
            ),
            workload_env=str(data.get("workload_env") or ""),
            resource_state=str(data.get("resource_state") or "unspecified"),
            resident_mix=str(data.get("resident_mix") or _infer_resident_mix_from_snapshot(data)),
            eta_source=str(data.get("eta_source") or ""),
            stable_rate_ready=bool(data.get("stable_rate_ready")),
            hardware_class=str(data.get("hardware_class") or ""),
            completion_model_ready=bool(data.get("completion_model_ready")),
            completion_model_sample_count=max(
                0, int(data.get("completion_model_sample_count") or 0)
            ),
            startup_overhead_s=max(0.0, float(data.get("startup_overhead_s") or 0.0)),
            completion_unit_s=max(0.0, float(data.get("completion_unit_s") or 0.0)),
            finalization_overhead_s=max(
                0.0, float(data.get("finalization_overhead_s") or 0.0)
            ),
            checkpoint_observed_s=max(
                0.0, float(data.get("checkpoint_observed_s") or 0.0)
            ),
            save_observed_s=max(0.0, float(data.get("save_observed_s") or 0.0)),
            completion_total_wall_s=max(
                0.0, float(data.get("completion_total_wall_s") or 0.0)
            ),
            completion_model_relative_error=max(
                0.0, float(data.get("completion_model_relative_error") or 0.0)
            ),
        )


@dataclass(frozen=True)
class StatewiseLookupResult:
    """Auditable result from a fail-closed full-tuple statewise lookup."""

    status: str
    workload_key: str
    workload_env: str
    node_bucket: str
    resource_state: str
    profile: int
    allocation_workers: int
    colocation_count: int
    resident_mix: str
    record: ProfileRecord | None = None
    boundary: ProfileRecord | None = None

    def __post_init__(self) -> None:
        if self.status not in _STATEWISE_LOOKUP_STATUSES:
            raise ValueError(f"unsupported statewise lookup status: {self.status!r}")
        if self.status == STATEWISE_LOOKUP_EXACT:
            valid = self.record is not None and self.boundary is None
        elif self.status == STATEWISE_LOOKUP_SCOPED_BOUNDARY:
            valid = self.record is None and self.boundary is not None
        else:
            valid = self.record is None and self.boundary is None
        if not valid:
            raise ValueError(
                f"statewise lookup payload does not match status {self.status!r}"
            )

    @property
    def is_exact(self) -> bool:
        return self.status == STATEWISE_LOOKUP_EXACT

    @property
    def is_missing(self) -> bool:
        return self.status == STATEWISE_LOOKUP_MISSING

    @property
    def is_scoped_boundary(self) -> bool:
        return self.status == STATEWISE_LOOKUP_SCOPED_BOUNDARY

    @property
    def request_tuple(self) -> tuple[Any, ...]:
        return (
            self.workload_key,
            self.workload_env,
            self.node_bucket,
            self.resource_state,
            int(self.profile),
            int(self.allocation_workers),
            int(self.colocation_count),
            self.resident_mix,
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "workload_key": self.workload_key,
            "workload_env": self.workload_env,
            "node_bucket": self.node_bucket,
            "resource_state": self.resource_state,
            "profile": int(self.profile),
            "allocation_workers": int(self.allocation_workers),
            "colocation_count": int(self.colocation_count),
            "resident_mix": self.resident_mix,
            "record": self.record.snapshot() if self.record is not None else None,
            "boundary": (
                self.boundary.snapshot() if self.boundary is not None else None
            ),
        }


class ServiceRateCache:
    """Small in-memory cache keyed by workload and co-location profile."""

    def __init__(self, records: Iterable[ProfileRecord] = ()):
        self._records: dict[tuple[str, int], ProfileRecord] = {}
        self._capacity_boundaries: dict[tuple[str, int], ProfileRecord] = {}
        self._scoped_capacity_boundaries: dict[tuple[Any, ...], ProfileRecord] = {}
        self._capacity_caps: dict[str, int] = {}
        self._state_records: dict[tuple[str, int, int, str, str], ProfileRecord] = {}
        self._env_state_records: dict[tuple[str, str, int, int, str, str], ProfileRecord] = {}
        self._hardware_state_records: dict[tuple[str, str, int, int, str, str], ProfileRecord] = {}
        self._mix_state_records: dict[tuple[str, int, int, str, str, str], ProfileRecord] = {}
        self._env_mix_state_records: dict[tuple[str, str, int, int, str, str, str], ProfileRecord] = {}
        self._hardware_mix_state_records: dict[tuple[str, str, int, int, str, str, str], ProfileRecord] = {}
        self._exact_state_records: dict[tuple[Any, ...], ProfileRecord] = {}
        for record in records:
            self.add(record)

    def add(self, record: ProfileRecord, *, force_replace: bool = False) -> None:
        key = (record.workload_key, int(record.profile))
        if record.capacity_boundary:
            if _record_has_capacity_scope(record):
                old_scoped = self._scoped_capacity_boundaries.get(_snapshot_identity(record))
                if force_replace or old_scoped is None or _record_quality(record) >= _record_quality(old_scoped):
                    self._scoped_capacity_boundaries[_snapshot_identity(record)] = record
                return
            old_boundary = self._capacity_boundaries.get(key)
            if force_replace or old_boundary is None or _record_quality(record) >= _record_quality(old_boundary):
                self._capacity_boundaries[key] = record
            old_cap = self._capacity_caps.get(record.workload_key)
            if old_cap is None or int(record.profile) < old_cap:
                self._capacity_caps[record.workload_key] = int(record.profile)
            return
        if _record_can_update_legacy_index(record):
            old = self._records.get(key)
            if force_replace or old is None or _record_quality(record) >= _record_quality(old):
                self._records[key] = record
        self._add_state_indexes(record, force_replace=force_replace)

    def clear_capacity_boundaries(self, workload_key: str) -> None:
        """Drop obsolete capacity caps before installing a fresher certificate.

        Capacity boundaries are deliberately conservative, but they are also
        measurement-certificate scoped.  When a later task-native probe remeasures
        the same workload under a corrected progress/ETA protocol, old boundary
        rows must not keep blocking the newer feasible rows.
        """

        key = str(workload_key)
        self._capacity_boundaries = {
            boundary_key: record
            for boundary_key, record in self._capacity_boundaries.items()
            if boundary_key[0] != key
        }
        self._scoped_capacity_boundaries = {
            boundary_key: record
            for boundary_key, record in self._scoped_capacity_boundaries.items()
            if record.workload_key != key
        }
        self._capacity_caps.pop(key, None)

    def get(self, workload_key: str, profile: int) -> ProfileRecord | None:
        key = (workload_key, int(profile))
        cap = self._capacity_caps.get(workload_key)
        if cap is not None and int(profile) >= cap:
            return self._capacity_boundaries.get(key)
        return self._records.get(key)

    def get_statewise(
        self,
        workload_key: str,
        profile: int | None = None,
        *,
        allocation_workers: int | None = None,
        colocation_count: int | None = None,
        node_bucket: str = "",
        resource_state: str = "",
        resident_mix: str = "",
        workload_env: str = "",
        hardware_class: str = "",
        strict: bool = False,
    ) -> ProfileRecord | None:
        """Return the best measured v2 row for a concrete node/load state.

        The legacy cache remains keyed by ``(workload_key, profile)`` where
        profile means co-location count and one worker is allocated per task.
        Explicit worker-allocation rows use the two-dimensional statewise key
        and cannot replace that fallback.  Historical callers can keep passing
        ``profile``; new callers may name ``colocation_count`` explicitly.
        """

        key = str(workload_key or "")
        if colocation_count is None:
            if profile is None:
                raise TypeError("profile or colocation_count is required")
            prof = int(profile)
        else:
            prof = int(colocation_count)
            if profile is not None and int(profile) != prof:
                raise ValueError("profile and colocation_count disagree")
        workers = max(1, int(allocation_workers or 1))
        env = _canon_key(workload_env)
        node = _canon_key(node_bucket)
        state = _canon_state(resource_state)
        mix = _canon_key(resident_mix)
        hw = _canon_key(hardware_class) or hardware_class_for_node_bucket(node)
        state_candidates = tuple(_state_candidates(state))
        mix_candidates = tuple(_mix_candidates(mix))
        node_candidates = tuple(_node_candidates(node))
        env_candidates = tuple(_env_candidates(env))
        hw_candidates = tuple(_hardware_candidates(hw))

        if mix:
            for env_key in env_candidates:
                if not env_key:
                    continue
                for node_key in node_candidates:
                    if not node_key:
                        continue
                    for state_key in state_candidates:
                        for mix_key in mix_candidates:
                            record = self._env_mix_state_records.get(
                                (key, env_key, workers, prof, node_key, state_key, mix_key)
                            )
                            if record is not None:
                                return record
            for node_key in node_candidates:
                if not node_key:
                    continue
                for state_key in state_candidates:
                    for mix_key in mix_candidates:
                        record = self._mix_state_records.get(
                            (key, workers, prof, node_key, state_key, mix_key)
                        )
                        if record is not None:
                            return record
            for env_key in env_candidates:
                if not env_key:
                    continue
                for hw_key in hw_candidates:
                    if not hw_key:
                        continue
                    for state_key in state_candidates:
                        for mix_key in mix_candidates:
                            record = self._hardware_mix_state_records.get(
                                (key, env_key, workers, prof, hw_key, state_key, mix_key)
                            )
                            if record is not None:
                                return record
            if strict:
                return None

        for env_key in env_candidates:
            if not env_key:
                continue
            for node_key in node_candidates:
                if not node_key:
                    continue
                for state_key in state_candidates:
                    record = self._env_state_records.get(
                        (key, env_key, workers, prof, node_key, state_key)
                    )
                    if record is not None:
                        return record
        for node_key in node_candidates:
            if not node_key:
                continue
            for state_key in state_candidates:
                record = self._state_records.get(
                    (key, workers, prof, node_key, state_key)
                )
                if record is not None:
                    return record
        for env_key in env_candidates:
            if not env_key:
                continue
            for hw_key in hw_candidates:
                if not hw_key:
                    continue
                for state_key in state_candidates:
                    record = self._hardware_state_records.get(
                        (key, env_key, workers, prof, hw_key, state_key)
                    )
                    if record is not None:
                        return record
        constrained = bool(node or env or state != "unspecified" or hw)
        if strict or constrained or workers != 1:
            return None
        return self.get(key, prof)

    def lookup_statewise_exact(
        self,
        workload_key: str,
        profile: int | None = None,
        *,
        workload_env: str,
        node_bucket: str,
        resource_state: str,
        allocation_workers: int | None = None,
        colocation_count: int | None = None,
        resident_mix: str = "",
    ) -> StatewiseLookupResult:
        """Look up one concrete statewise request without any fallback.

        Matching uses the complete request tuple. It does not broaden resource
        state to ``unspecified``, shorten node buckets, use hardware-equivalent
        nodes, or consult the legacy ``(workload_key, profile)`` index.
        A matching scoped v2 capacity boundary takes precedence over positive
        rows at and above the boundary on the same allocation/profile axis.
        """

        prof, workers, colocation = _statewise_profile_dimensions(
            profile=profile,
            allocation_workers=allocation_workers,
            colocation_count=colocation_count,
        )
        request_key = _exact_statewise_key(
            workload_key=workload_key,
            workload_env=workload_env,
            node_bucket=node_bucket,
            resource_state=resource_state,
            profile=prof,
            allocation_workers=workers,
            colocation_count=colocation,
            resident_mix=resident_mix,
        )
        result_fields = {
            "workload_key": request_key[0],
            "workload_env": request_key[1],
            "node_bucket": request_key[2],
            "resource_state": request_key[3],
            "profile": prof,
            "allocation_workers": workers,
            "colocation_count": colocation,
            "resident_mix": request_key[4],
        }
        boundary = self._exact_scoped_capacity_boundary(request_key)
        if boundary is not None:
            return StatewiseLookupResult(
                status=STATEWISE_LOOKUP_SCOPED_BOUNDARY,
                boundary=boundary,
                **result_fields,
            )
        record = self._exact_state_records.get(request_key)
        if record is not None:
            return StatewiseLookupResult(
                status=STATEWISE_LOOKUP_EXACT,
                record=record,
                **result_fields,
            )
        return StatewiseLookupResult(
            status=STATEWISE_LOOKUP_MISSING,
            **result_fields,
        )

    def statewise_profiles(
        self,
        workload_key: str,
        *,
        allocation_workers: int | None = None,
        colocation_count: int | None = None,
        node_bucket: str = "",
        resource_state: str = "",
        workload_env: str = "",
    ) -> list[ProfileRecord]:
        seen: set[tuple[str, int, int, str, str, str, str]] = set()
        candidates = (
            list(self._state_records.values())
            + list(self._env_state_records.values())
            + list(self._mix_state_records.values())
            + list(self._env_mix_state_records.values())
        )
        out = []
        for record in candidates:
            if record.workload_key != workload_key:
                continue
            ident = (
                record.workload_key,
                int(record.allocation_workers),
                int(record.colocation_count),
                str(record.node_bucket),
                str(record.resource_state),
                str(record.resident_mix),
                str(record.workload_env),
            )
            if ident in seen:
                continue
            seen.add(ident)
            if node_bucket and _canon_key(record.node_bucket) not in set(_node_candidates(node_bucket)):
                continue
            if resource_state and _canon_state(record.resource_state) not in set(_state_candidates(resource_state)):
                continue
            if workload_env and _canon_key(record.workload_env) not in set(_env_candidates(workload_env)):
                continue
            if allocation_workers is not None and int(record.allocation_workers) != int(allocation_workers):
                continue
            if colocation_count is not None and int(record.colocation_count) != int(colocation_count):
                continue
            out.append(record)
        return sorted(
            out,
            key=lambda record: (
                record.colocation_count,
                record.allocation_workers,
                record.node_bucket,
                record.resource_state,
                record.resident_mix,
            ),
        )

    def profiles(self, workload_key: str, *, include_boundaries: bool = False) -> list[ProfileRecord]:
        records = [
            record for (key, _), record in self._records.items()
            if key == workload_key and not self._blocked_by_capacity_cap(record)
        ]
        if include_boundaries:
            records.extend(
                record for (key, _), record in self._capacity_boundaries.items()
                if key == workload_key
            )
        return sorted(records, key=lambda record: record.profile)

    def available_workloads(self) -> list[str]:
        statewise = (
            list(self._state_records.values())
            + list(self._env_state_records.values())
            + list(self._hardware_state_records.values())
            + list(self._mix_state_records.values())
            + list(self._env_mix_state_records.values())
            + list(self._hardware_mix_state_records.values())
            + list(self._exact_state_records.values())
        )
        return sorted(
            {key for key, _ in self._records}
            | {key for key, _ in self._capacity_boundaries}
            | {record.workload_key for record in self._scoped_capacity_boundaries.values()}
            | {record.workload_key for record in statewise}
        )

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

    def best_profile_for_mean_flow(
        self,
        workload_key: str,
        *,
        task_count: int,
        total_units: float,
        resource_count: int = 1,
    ) -> ProfileRecord:
        """Choose the profile minimizing deterministic mean completion time."""

        candidates = self.profiles(workload_key)
        if not candidates:
            raise KeyError(f"no service cache entries for workload {workload_key!r}")
        return min(
            candidates,
            key=lambda record: (
                deterministic_mean_flow_s(
                    task_count=task_count,
                    total_units=total_units,
                    resource_count=resource_count,
                    profile=record.profile,
                    aggregate_rate=record.aggregate_rate,
                ),
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

    def snapshot(self) -> dict[str, Any]:
        records: dict[tuple[Any, ...], ProfileRecord] = {}
        boundaries: dict[tuple[Any, ...], ProfileRecord] = {}
        for record in self._exact_state_records.values():
            if not self._blocked_by_capacity_cap(record):
                records[_snapshot_identity(record)] = record
        for record in self._records.values():
            if not self._blocked_by_capacity_cap(record):
                records[_snapshot_identity(record)] = record
        for record in self._state_records.values():
            if not self._blocked_by_capacity_cap(record):
                records[_snapshot_identity(record)] = record
        for record in self._env_state_records.values():
            if not self._blocked_by_capacity_cap(record):
                records[_snapshot_identity(record)] = record
        for record in self._hardware_state_records.values():
            if not self._blocked_by_capacity_cap(record):
                records[_snapshot_identity(record)] = record
        for record in self._mix_state_records.values():
            if not self._blocked_by_capacity_cap(record):
                records[_snapshot_identity(record)] = record
        for record in self._env_mix_state_records.values():
            if not self._blocked_by_capacity_cap(record):
                records[_snapshot_identity(record)] = record
        for record in self._hardware_mix_state_records.values():
            if not self._blocked_by_capacity_cap(record):
                records[_snapshot_identity(record)] = record
        for record in self._capacity_boundaries.values():
            boundaries[_snapshot_identity(record)] = record
        for record in self._scoped_capacity_boundaries.values():
            boundaries[_snapshot_identity(record)] = record
        return {
            "records": [
                record.snapshot()
                for record in sorted(
                    records.values(),
                    key=lambda r: (
                        r.workload_key,
                        r.colocation_count,
                        r.allocation_workers,
                        r.workload_env,
                        r.node_bucket,
                        r.resource_state,
                        r.resident_mix,
                        r.capacity_boundary,
                        r.source,
                    ),
                )
            ],
            "capacity_boundaries": [
                record.snapshot()
                for record in sorted(
                    boundaries.values(),
                    key=lambda r: (
                        r.workload_key,
                        r.colocation_count,
                        r.allocation_workers,
                        r.workload_env,
                        r.node_bucket,
                        r.resource_state,
                        r.resident_mix,
                        r.source,
                    ),
                )
            ],
        }

    @staticmethod
    def from_snapshot(data: dict[str, Any]) -> "ServiceRateCache":
        rows = list(data.get("records") or []) + list(data.get("capacity_boundaries") or [])
        return ServiceRateCache(ProfileRecord.from_snapshot(row) for row in rows)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.snapshot(), indent=2, sort_keys=True), encoding="utf-8")

    @staticmethod
    def load(path: Path) -> "ServiceRateCache":
        return ServiceRateCache.from_snapshot(json.loads(path.read_text(encoding="utf-8")))

    def _blocked_by_capacity_cap(self, record: ProfileRecord) -> bool:
        cap = self._capacity_caps.get(record.workload_key)
        return cap is not None and int(record.profile) >= cap

    def _add_state_indexes(self, record: ProfileRecord, *, force_replace: bool = False) -> None:
        if record.capacity_boundary:
            return
        exact_key = _exact_statewise_key_for_record(record)
        self._put_index(
            self._exact_state_records,
            exact_key,
            record,
            force_replace=force_replace,
        )
        node = _canon_key(record.node_bucket)
        state = _canon_state(record.resource_state)
        mix = _canon_key(record.resident_mix)
        env = _canon_key(record.workload_env)
        hw = _canon_key(record.hardware_class) or hardware_class_for_node_bucket(node)
        workers = int(record.allocation_workers)
        colocation = int(record.colocation_count)
        if mix:
            if node:
                key = (record.workload_key, workers, colocation, node, state, mix)
                self._put_index(self._mix_state_records, key, record, force_replace=force_replace)
            if env and node:
                key = (
                    record.workload_key,
                    env,
                    workers,
                    colocation,
                    node,
                    state,
                    mix,
                )
                self._put_index(self._env_mix_state_records, key, record, force_replace=force_replace)
            if env and hw:
                key = (
                    record.workload_key,
                    env,
                    workers,
                    colocation,
                    hw,
                    state,
                    mix,
                )
                self._put_index(self._hardware_mix_state_records, key, record, force_replace=force_replace)
            return
        if node:
            key = (record.workload_key, workers, colocation, node, state)
            self._put_index(self._state_records, key, record, force_replace=force_replace)
        if env and node:
            key = (record.workload_key, env, workers, colocation, node, state)
            self._put_index(self._env_state_records, key, record, force_replace=force_replace)
        if env and hw:
            key = (record.workload_key, env, workers, colocation, hw, state)
            self._put_index(self._hardware_state_records, key, record, force_replace=force_replace)

    def _exact_scoped_capacity_boundary(
        self,
        request_key: tuple[Any, ...],
    ) -> ProfileRecord | None:
        candidates = [
            boundary
            for boundary in self._scoped_capacity_boundaries.values()
            if _scoped_capacity_boundary_applies(boundary, request_key)
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda record: (
                max(
                    int(record.allocation_workers),
                    int(record.colocation_count),
                ),
                _snapshot_identity(record),
            ),
        )

    @staticmethod
    def _put_index(
        target: dict[tuple[Any, ...], ProfileRecord],
        key: tuple[Any, ...],
        record: ProfileRecord,
        *,
        force_replace: bool,
    ) -> None:
        old = target.get(key)
        if force_replace or old is None or _record_quality(record) >= _record_quality(old):
            target[key] = record


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
    workload_env: str = "",
    resource_state: str = "unspecified",
    resident_mix: str = "",
    eta_source: str = "",
    hardware_class: str = "",
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
    stable_rate_ready = _summary_has_stable_progress_rate(summary)
    completion = _completion_model_fields(summary)
    resolved_eta_source = eta_source or ("tqdm/progress" if stable_rate_ready else "unknown")
    resolved_hardware_class = hardware_class or hardware_class_for_node_bucket(node_bucket)
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
            allocation_workers=_allocation_workers_from_summary(summary),
            colocation_count=profile,
            workload_env=workload_env,
            resource_state=resource_state,
            resident_mix=resident_mix,
            eta_source=resolved_eta_source,
            stable_rate_ready=stable_rate_ready,
            hardware_class=resolved_hardware_class,
            **completion,
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
    workload_env: str = "",
    resource_state: str = "unspecified",
    resident_mix: str = "",
    eta_source: str = "",
    hardware_class: str = "",
) -> None:
    for path in sorted(Path().glob(glob_pattern) if not glob_pattern.startswith("/") else Path("/").glob(glob_pattern[1:])):
        for record in records_from_summary_file(
            path,
            workload_key=workload_key,
            command_fingerprint=command_fingerprint,
            resource_kind=resource_kind,
            total_units=total_units,
            node_bucket=node_bucket,
            workload_env=workload_env,
            resource_state=resource_state,
            resident_mix=resident_mix,
            eta_source=eta_source,
            hardware_class=hardware_class,
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
                workload_env="cpu_synthetic",
                resource_state="unspecified",
                resident_mix="",
                eta_source="protocol",
                stable_rate_ready=True,
                hardware_class="cpu_pool",
            )
        )


def _record_quality(record: ProfileRecord) -> tuple[int, int, float]:
    return (
        0 if record.capacity_boundary else 1,
        1 if record.stable_rate_ready or "progress" in str(record.eta_source).lower() or "tqdm" in str(record.eta_source).lower() else 0,
        len(record.per_task_rates),
        -float(record.aggregate_rate),
    )


def _snapshot_identity(record: ProfileRecord) -> tuple[Any, ...]:
    return (
        str(record.workload_key),
        int(record.allocation_workers),
        int(record.colocation_count),
        str(record.workload_env),
        str(record.node_bucket),
        str(record.resource_state),
        str(record.resident_mix),
        str(record.hardware_class),
        bool(record.capacity_boundary),
        str(record.command_fingerprint),
        str(record.source),
    )


def _record_has_capacity_scope(record: ProfileRecord) -> bool:
    """Whether a capacity boundary is tied to a concrete fabric slice.

    Historical/default-cache boundaries are theorem-population caps even when
    they carry node metadata from the run that discovered the cap.  The live-v2
    cross-node ETA protocol is different: a failed/unstable profile on one
    concrete server must not block a valid row for another server.  We therefore
    scope only boundaries emitted by the live-v2 service-cache protocol.
    """

    fingerprint = _canon_key(record.command_fingerprint)
    explicit_allocation = int(record.allocation_workers) != 1
    return bool(
        _canon_key(record.node_bucket)
        and (explicit_allocation or fingerprint.startswith("live_tqdm_service_cache_v2"))
    )


def _record_can_update_legacy_index(record: ProfileRecord) -> bool:
    """Whether a v2 row is safe as an unconstrained legacy fallback.

    The legacy cache key is only ``(workload_key, profile)``.  It cannot encode
    load state, environment, hardware class, or node family.  Rows measured by
    the live-v2 / full-factorial ETA pipeline must therefore stay in the
    statewise indexes even when their state is ``empty``; otherwise a row from
    one server or one co-location experiment can silently replace the ordinary
    service curve and poison q01/q11 replay comparisons.
    """

    if int(record.allocation_workers) != 1:
        return False
    if _canon_state(record.resource_state) not in {"", "unspecified", "empty"}:
        return False
    if _record_is_v2_scoped(record):
        return False
    return True


def _record_is_v2_scoped(record: ProfileRecord) -> bool:
    fingerprint = _canon_key(record.command_fingerprint)
    source = _canon_key(record.source)
    if fingerprint.startswith("live_tqdm_service_cache_v2"):
        return True
    if any(
        marker in source
        for marker in (
            "service_cache_v2",
            "full_factorial_eta",
            "full_factorial_cpu_eta",
            "live_marginal_under_load",
            "cross_node_eta_matrix",
            "live_extra_eta",
        )
    ):
        return True
    node = _canon_key(record.node_bucket)
    environment = _canon_key(record.workload_env)
    explicit_hardware = _canon_key(record.hardware_class)
    inferred_hardware = hardware_class_for_node_bucket(node)
    concrete_fabric = bool(
        ":" in node
        or explicit_hardware
        or inferred_hardware.startswith(("cpu_", "gpu_"))
    )
    return bool(node and environment and concrete_fabric)


def hardware_class_for_node_bucket(node_bucket: str) -> str:
    text = _canon_key(node_bucket)
    if not text:
        return ""
    if text.startswith("jtl110gpu") or "jtl110gpu" in text or "jtl110gpu2" in text:
        return "gpu_3080ti_12gb_dual"
    if text.startswith("jtl311linux") or "jtl311linux" in text:
        return "gpu_rtx2080_8gb_dual_cpu_fast"
    if text.startswith("node007") or "node007-direct" in text:
        return "gpu_node007_4x12gb"
    if re.search(r"node00[1-6](?:\b|[_:.-])", text) or text in {f"node00{i}" for i in range(1, 7)}:
        return "cpu_hpc_192c"
    if text.startswith("jtl110cpu"):
        return "cpu_jtl110_128c"
    return text.split(":")[0]


def _canon_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    return re.sub(r"[^a-z0-9_.:+-]+", "_", text).strip("_")


def _canon_state(value: Any) -> str:
    text = _canon_key(value)
    return text or "unspecified"


def _statewise_profile_dimensions(
    *,
    profile: int | None,
    allocation_workers: int | None,
    colocation_count: int | None,
) -> tuple[int, int, int]:
    if colocation_count is None:
        if profile is None:
            raise TypeError("profile or colocation_count is required")
        colocation = int(profile)
    else:
        colocation = int(colocation_count)
        if profile is not None and int(profile) != colocation:
            raise ValueError("profile and colocation_count disagree")
    workers = 1 if allocation_workers is None else int(allocation_workers)
    if workers <= 0:
        raise ValueError("allocation_workers must be positive")
    if colocation <= 0:
        raise ValueError("colocation_count must be positive")
    return colocation, workers, colocation


def _exact_statewise_key(
    *,
    workload_key: str,
    workload_env: str,
    node_bucket: str,
    resource_state: str,
    profile: int,
    allocation_workers: int,
    colocation_count: int,
    resident_mix: str,
) -> tuple[Any, ...]:
    return (
        _canon_key(workload_key),
        _canon_key(workload_env),
        _canon_key(node_bucket),
        _canon_key(resource_state),
        _canon_key(resident_mix),
        int(profile),
        int(allocation_workers),
        int(colocation_count),
    )


def _exact_statewise_key_for_record(record: ProfileRecord) -> tuple[Any, ...]:
    return _exact_statewise_key(
        workload_key=record.workload_key,
        workload_env=record.workload_env,
        node_bucket=record.node_bucket,
        resource_state=record.resource_state,
        profile=record.profile,
        allocation_workers=record.allocation_workers,
        colocation_count=record.colocation_count,
        resident_mix=record.resident_mix,
    )


def _scoped_capacity_boundary_applies(
    boundary: ProfileRecord,
    request_key: tuple[Any, ...],
) -> bool:
    boundary_key = _exact_statewise_key_for_record(boundary)
    if request_key[:5] != boundary_key[:5]:
        return False

    request_workers = int(request_key[6])
    request_colocation = int(request_key[7])
    boundary_workers = int(boundary.allocation_workers)
    boundary_colocation = int(boundary.colocation_count)

    if boundary_workers > 1 and boundary_colocation == 1:
        return (
            request_colocation == boundary_colocation
            and request_workers >= boundary_workers
        )
    if boundary_workers == 1 and boundary_colocation == 1:
        return (
            request_colocation == 1
            or request_workers == 1
        )
    return (
        request_workers == boundary_workers
        and request_colocation >= boundary_colocation
    )


def _infer_resident_mix_from_snapshot(data: dict[str, Any]) -> str:
    state = _canon_state(data.get("resource_state"))
    if state == "half_loaded":
        return "same_workload_half_capacity"
    if state == "full_loaded":
        return "same_workload_to_capacity_boundary"
    text = _canon_key(
        " ".join(
            str(data.get(key) or "")
            for key in ("source", "command_fingerprint", "scenario_id", "background_kind")
        )
    )
    if "live_marginal_under_load" not in text and "marginal_under_load" not in text:
        return ""
    if state == "high_vram_resident":
        return "llm_or_memory_resident"
    if state == "cpu_resident":
        return "cpu_worker_resident"
    if "llm_rl" in text or ("llm" in text and "rl" in text):
        return "cnn_plus_llm_plus_hybrid_rl"
    if "rl_resident" in text or "hybrid_rl" in text:
        return "cnn_plus_hybrid_rl"
    if "llm_resident" in text:
        return "cnn_plus_llm"
    if "cpu_" in text:
        return "cpu_plus_gpu_target"
    return ""


def _state_candidates(resource_state: str) -> list[str]:
    state = _canon_state(resource_state)
    out = [state]
    if state != "unspecified":
        out.append("unspecified")
    return out


def _mix_candidates(resident_mix: str) -> list[str]:
    mix = _canon_key(resident_mix)
    if not mix:
        return [""]
    return [mix]


def _node_candidates(node_bucket: str) -> list[str]:
    node = _canon_key(node_bucket)
    if not node:
        return [""]
    out = [node]
    if ":" in node:
        out.append(node.split(":", 1)[0])
    return list(dict.fromkeys(out))


def _env_candidates(workload_env: str) -> list[str]:
    env = _canon_key(workload_env)
    if not env:
        return [""]
    aliases = {
        "half_cheetah": "halfcheetah",
        "halfcheetah-v2": "halfcheetah",
        "halfcheetah-v3": "halfcheetah",
        "ant-v2": "ant",
        "ant-v3": "ant",
        "hopper-v2": "hopper",
        "hopper-v3": "hopper",
        "walker2d-v2": "walker2d",
        "walker2d-v3": "walker2d",
    }
    canonical = aliases.get(env, env)
    return list(dict.fromkeys([canonical, env, ""]))


def _hardware_candidates(hardware_class: str) -> list[str]:
    hw = _canon_key(hardware_class)
    return [hw, ""] if hw else [""]


def _profile_from_summary(path: Path, summary: dict[str, Any]) -> int:
    explicit = summary.get("colocation_count")
    if explicit is not None:
        return max(1, int(explicit))
    text = str(summary.get("phase") or path.name)
    match = _PROFILE_RE.search(text)
    if not match:
        raise ValueError(f"cannot infer profile count from {path}")
    return int(match.group(1))


def _allocation_workers_from_summary(summary: dict[str, Any]) -> int:
    return max(1, int(summary.get("allocation_workers") or 1))


def _aggregate_rate(summary: dict[str, Any]) -> float:
    for key in (
        "aggregate_stable_rate_unit_s",
        "aggregate_stable_rate_step_s",
        "measurement_aggregate_rate_unit_s_median",
        "measurement_aggregate_rate_step_s_median",
        "aggregate_active_rate_unit_s",
        "aggregate_active_rate_step_s",
    ):
        if summary.get(key) is not None:
            return float(summary.get(key) or 0.0)
    return 0.0


def _per_task_rates(summary: dict[str, Any]) -> tuple[float, ...]:
    for key in ("stable_rates_unit_s", "stable_rates_step_s", "rates_unit_s", "rates_step_s"):
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


def _completion_model_fields(summary: dict[str, Any]) -> dict[str, Any]:
    """Extract a conservative per-task natural-completion model.

    A profile is completion-ready only when every launched child finished
    naturally and emitted a ready model.  Maxima across colocated children are
    used because ETA admission must not hide the slowest task behind an average.
    """

    rows = summary.get("rows") or []
    if not isinstance(rows, list):
        rows = []
    models = [
        row.get("completion_model")
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("completion_model"), dict)
    ]
    running_count = max(0, int(summary.get("running_count") or len(rows)))
    ready_models = [
        model for model in models if bool(model.get("completion_model_ready"))
    ]
    ready = bool(
        running_count > 0
        and len(ready_models) == running_count
        and len(models) == running_count
    )
    if not ready:
        return {
            "completion_model_ready": False,
            "completion_model_sample_count": len(ready_models),
        }

    def maximum(*keys: str) -> float:
        values = []
        for model in ready_models:
            for key in keys:
                if key in model:
                    try:
                        values.append(max(0.0, float(model.get(key) or 0.0)))
                    except (TypeError, ValueError):
                        pass
                    break
        return max(values, default=0.0)

    return {
        "completion_model_ready": True,
        "completion_model_sample_count": len(ready_models),
        "startup_overhead_s": maximum(
            "startup_overhead_s", "startup_to_first_progress_s"
        ),
        "completion_unit_s": maximum("completion_unit_s", "amortized_unit_s"),
        "finalization_overhead_s": maximum(
            "terminal_overhead_s", "finalization_after_last_progress_s"
        ),
        "checkpoint_observed_s": maximum("checkpoint_observed_s"),
        "save_observed_s": maximum("save_observed_s"),
        "completion_total_wall_s": maximum("total_wall_s"),
        "completion_model_relative_error": maximum("model_relative_error"),
    }


def _summary_is_unusable_for_exact_replay(summary: dict[str, Any]) -> bool:
    """True when a measured profile should not be used as a valid service curve."""

    if bool(summary.get("capacity_boundary")):
        return True
    stable_timeout_accepted = _summary_accepts_stable_timeout(summary)
    if summary.get("measurement_valid") is False and not stable_timeout_accepted:
        return True
    if summary.get("placement_valid") is False:
        return True
    running_count = int(summary.get("running_count") or 0)
    if running_count > 0:
        accepted_count = int(summary.get("returncode_accepted_count") or 0)
        strict_count = (
            int(summary.get("returncode_valid_count") or 0)
            if "returncode_valid_count" in summary
            else running_count
        )
        if max(accepted_count, strict_count) < running_count and not stable_timeout_accepted:
            return True
        if summary.get("terminate_on_stable") is True:
            if summary.get("all_stable_rate_ready") is False:
                return True
            if int(summary.get("stable_rate_ready_count") or 0) < running_count:
                return True
    if int(summary.get("blocked_count") or 0) > 0:
        return True
    statuses = summary.get("status_counts") or {}
    if isinstance(statuses, dict):
        for key in ("failed", "cancelled", "killed"):
            if int(statuses.get(key) or 0) > 0:
                return True
    return False


def _summary_accepts_stable_timeout(summary: dict[str, Any]) -> bool:
    if summary.get("require_stable_rate") is not True:
        return False
    running_count = int(summary.get("running_count") or 0)
    if running_count <= 0:
        return False
    if summary.get("all_stable_rate_ready") is not True:
        return False
    if int(summary.get("stable_rate_ready_count") or 0) < running_count:
        return False
    if summary.get("placement_valid") is False:
        return False
    return float(summary.get("aggregate_stable_rate_unit_s") or summary.get("aggregate_stable_rate_step_s") or 0.0) > 0.0


def _summary_has_stable_progress_rate(summary: dict[str, Any]) -> bool:
    if summary.get("all_stable_rate_ready") is True:
        return True
    if int(summary.get("stable_rate_ready_count") or 0) > 0:
        return True
    if summary.get("terminate_on_stable") is True and summary.get("all_stable_rate_ready") is not False:
        return True
    for key in (
        "aggregate_stable_rate_unit_s",
        "aggregate_stable_rate_step_s",
        "stable_rates_unit_s",
        "stable_rates_step_s",
    ):
        value = summary.get(key)
        if isinstance(value, list) and any(float(x or 0.0) > 0.0 for x in value):
            return True
        try:
            if value is not None and float(value) > 0.0:
                return True
        except (TypeError, ValueError):
            pass
    source = " ".join(str(summary.get(key) or "") for key in ("eta_source", "progress_source", "rate_source"))
    return "tqdm" in source.lower() or "progress" in source.lower()
