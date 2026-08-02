"""Fail-closed statewise views for theorem-facing completion replay.

The ordinary replay cache intentionally keeps a compatibility index keyed by
``(workload_key, profile)``.  That index is useful for historical experiments,
but it cannot distinguish hardware, load state, workload environment, worker
allocation, or co-location count.  This module projects an explicitly declared
finite set of full-tuple records into an isolated replay cache.  No source-cache
fallback is permitted.

Two views are produced:

* ``lower_service`` retains the calibrated lower service used for robust action
  selection and theorem slack;
* ``completion_point`` converts a natural-completion model into an end-to-end
  effective aggregate rate used for JCT/makespan replay.

Keeping the views separate prevents a point ETA estimate from being presented
as a lower service bound, or a conservative lower bound from being presented as
the measured point completion time.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

from .fast_forward import ReplayPolicy, WorkloadSpec
from .service_cache import ProfileRecord, ServiceRateCache


@dataclass(frozen=True)
class ExactReplayProfile:
    """One admitted full-tuple service row for replay."""

    workload_key: str
    workload_env: str
    node_bucket: str
    resource_state: str
    colocation_count: int
    allocation_workers: int = 1
    resident_mix: str = ""

    @property
    def profile(self) -> int:
        return int(self.colocation_count)

    def snapshot(self) -> dict[str, Any]:
        return {
            "workload_key": self.workload_key,
            "workload_env": self.workload_env,
            "node_bucket": self.node_bucket,
            "resource_state": self.resource_state,
            "allocation_workers": int(self.allocation_workers),
            "colocation_count": int(self.colocation_count),
            "resident_mix": self.resident_mix,
        }


@dataclass(frozen=True)
class ExactReplayViews:
    """Auditable lower-service and point-completion cache projections."""

    lower_service: ServiceRateCache
    completion_point: ServiceRateCache
    rows: tuple[dict[str, Any], ...]

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "projection_semantics": (
                "full_tuple_source_lookup_with_separate_lower_service_and_"
                "natural_completion_point_views"
            ),
            "row_count": len(self.rows),
            "rows": list(self.rows),
        }


def build_exact_replay_views(
    source: ServiceRateCache,
    requests: Iterable[ExactReplayProfile],
) -> ExactReplayViews:
    """Build isolated legacy-shaped views from exact statewise records.

    Every request must resolve through ``lookup_statewise_exact`` and must have
    both a non-history stable lower-service record and a natural-completion
    model.  Duplicate workload/profile requests are rejected because the
    projected cache deliberately has only that compatibility key.
    """

    selected = tuple(requests)
    if not selected:
        raise ValueError("at least one exact replay profile is required")
    projection_keys: set[tuple[str, int]] = set()
    lower_records: list[ProfileRecord] = []
    point_records: list[ProfileRecord] = []
    audit_rows: list[dict[str, Any]] = []

    for request in selected:
        _validate_request(request)
        projection_key = (request.workload_key, request.profile)
        if projection_key in projection_keys:
            raise ValueError(
                "projected replay key is ambiguous: "
                f"{request.workload_key}/p{request.profile}"
            )
        projection_keys.add(projection_key)

        lookup = source.lookup_statewise_exact(
            request.workload_key,
            workload_env=request.workload_env,
            node_bucket=request.node_bucket,
            resource_state=request.resource_state,
            allocation_workers=request.allocation_workers,
            colocation_count=request.colocation_count,
            resident_mix=request.resident_mix,
        )
        if not lookup.is_exact or lookup.record is None:
            raise KeyError(
                "missing exact replay service tuple: "
                f"{request.snapshot()} status={lookup.status}"
            )
        source_record = lookup.record
        _validate_source_record(source_record, request=request)

        point_rate = _completion_point_aggregate_rate(source_record)
        lower_records.append(
            _project_record(
                source_record,
                aggregate_rate=float(source_record.aggregate_rate),
                view="lower_service",
            )
        )
        point_records.append(
            _project_record(
                source_record,
                aggregate_rate=point_rate,
                view="completion_point",
            )
        )
        audit_rows.append(
            {
                **request.snapshot(),
                "source_command_fingerprint": source_record.command_fingerprint,
                "source_path": source_record.source,
                "eta_source": source_record.eta_source,
                "completion_model_sample_count": int(
                    source_record.completion_model_sample_count
                ),
                "completion_total_wall_s": float(
                    source_record.completion_total_wall_s
                ),
                "source_group_total_units": float(source_record.total_units),
                "lower_service_aggregate_rate": float(
                    source_record.aggregate_rate
                ),
                "completion_point_aggregate_rate": point_rate,
                "completion_point_rate_formula": (
                    "source_group_total_units/completion_total_wall_s"
                ),
            }
        )

    return ExactReplayViews(
        lower_service=ServiceRateCache(lower_records),
        completion_point=ServiceRateCache(point_records),
        rows=tuple(audit_rows),
    )


@dataclass(frozen=True)
class FrozenReplayAction:
    workload_key: str
    profile: int
    family_name: str
    delegate: ReplayPolicy

    def snapshot(self) -> dict[str, Any]:
        return {
            "workload_key": self.workload_key,
            "profile": int(self.profile),
            "family_name": self.family_name,
            "delegate": self.delegate.snapshot(),
        }


@dataclass(frozen=True)
class FrozenActionUnionPolicy(ReplayPolicy):
    """Replay fixed actions selected on a different, lower-service view.

    The base profile is fixed per workload.  The selected family's trajectory
    semantics are still delegated, so admission/drain behavior is retained
    while service time is evaluated on the completion-point cache.
    """

    frozen_actions: tuple[FrozenReplayAction, ...] = ()

    def _action(self, workload_key: str) -> FrozenReplayAction:
        rows = [
            action
            for action in self.frozen_actions
            if action.workload_key == workload_key
        ]
        if len(rows) != 1:
            raise KeyError(
                f"expected one frozen action for {workload_key!r}, found {len(rows)}"
            )
        return rows[0]

    def select_profile(
        self,
        cache: ServiceRateCache,
        spec: WorkloadSpec,
    ) -> ProfileRecord:
        action = self._action(spec.workload_key)
        record = cache.get(spec.workload_key, action.profile)
        if record is None or record.capacity_boundary or record.aggregate_rate <= 0:
            raise KeyError(
                f"missing completion-point profile {spec.workload_key!r}/p{action.profile}"
            )
        return record

    def trajectory_policy_for(
        self,
        cache: ServiceRateCache,
        spec: WorkloadSpec,
        *,
        base_target_profile: int | None = None,
    ) -> ReplayPolicy | None:
        return self._action(spec.workload_key).delegate

    def trace_action_policy_for(
        self,
        cache: ServiceRateCache,
        spec: WorkloadSpec,
        *,
        arrival_mode: str,
        base_target_profile: int | None = None,
    ) -> ReplayPolicy | None:
        # Returning self preserves the frozen base profile.  The event-level
        # trajectory hook above delegates after the base profile is resolved.
        return self

    def snapshot(self) -> dict[str, Any]:
        return {
            **super().snapshot(),
            "selection_service_view": "lower_service",
            "replay_service_view": "natural_completion_point",
            "frozen_actions": [
                action.snapshot() for action in self.frozen_actions
            ],
        }


def _validate_request(request: ExactReplayProfile) -> None:
    if not request.workload_key.strip():
        raise ValueError("workload_key is required")
    if not request.workload_env.strip():
        raise ValueError("workload_env is required for exact replay")
    if not request.node_bucket.strip():
        raise ValueError("node_bucket is required for exact replay")
    if not request.resource_state.strip():
        raise ValueError("resource_state is required for exact replay")
    if int(request.allocation_workers) <= 0:
        raise ValueError("allocation_workers must be positive")
    if int(request.colocation_count) <= 0:
        raise ValueError("colocation_count must be positive")


def _validate_source_record(
    record: ProfileRecord,
    *,
    request: ExactReplayProfile,
) -> None:
    errors = []
    if record.capacity_boundary:
        errors.append("capacity_boundary")
    if not record.stable_rate_ready or float(record.aggregate_rate) <= 0.0:
        errors.append("lower_service_not_ready")
    if not record.completion_model_ready:
        errors.append("completion_model_not_ready")
    if int(record.completion_model_sample_count) <= 0:
        errors.append("completion_model_sample_count_zero")
    if float(record.total_units) <= 0.0:
        errors.append("group_total_units_nonpositive")
    if float(record.completion_total_wall_s) <= 0.0:
        errors.append("completion_total_wall_nonpositive")
    eta_source = str(record.eta_source or "").lower()
    if "history" in eta_source or "hist" in eta_source:
        errors.append("history_eta_forbidden")
    if not any(token in eta_source for token in ("progress", "tqdm")):
        errors.append("task_native_progress_eta_required")
    if int(record.allocation_workers) != int(request.allocation_workers):
        errors.append("allocation_worker_axis_mismatch")
    if int(record.colocation_count) != int(request.colocation_count):
        errors.append("colocation_axis_mismatch")
    if errors:
        raise ValueError(
            f"unusable exact replay row {request.snapshot()}: "
            + ",".join(errors)
        )


def _completion_point_aggregate_rate(record: ProfileRecord) -> float:
    rate = float(record.total_units) / float(record.completion_total_wall_s)
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError("natural-completion point rate must be finite and positive")
    return rate


def _project_record(
    record: ProfileRecord,
    *,
    aggregate_rate: float,
    view: str,
) -> ProfileRecord:
    profile = int(record.colocation_count)
    per_task = float(aggregate_rate) / float(profile)
    return ProfileRecord(
        workload_key=record.workload_key,
        command_fingerprint=(
            f"exact_statewise_replay_projection_{view}:"
            f"{record.command_fingerprint}"
        ),
        resource_kind=record.resource_kind,
        node_bucket="",
        profile=profile,
        unit=record.unit,
        total_units=float(record.total_units),
        aggregate_rate=float(aggregate_rate),
        per_task_rates=tuple(per_task for _ in range(profile)),
        source=(
            f"statewise-replay-view:{view};source-node={record.node_bucket};"
            f"source-state={record.resource_state};source={record.source}"
        ),
        capacity_boundary=False,
        gpu_count_observed=record.gpu_count_observed,
        workload_env="",
        resource_state="unspecified",
        resident_mix="",
        eta_source=record.eta_source,
        stable_rate_ready=True,
        hardware_class="",
        completion_model_ready=record.completion_model_ready,
        completion_model_sample_count=record.completion_model_sample_count,
        startup_overhead_s=record.startup_overhead_s,
        completion_unit_s=record.completion_unit_s,
        finalization_overhead_s=record.finalization_overhead_s,
        checkpoint_observed_s=record.checkpoint_observed_s,
        save_observed_s=record.save_observed_s,
        completion_total_wall_s=record.completion_total_wall_s,
        completion_model_relative_error=record.completion_model_relative_error,
        allocation_workers=1,
        colocation_count=profile,
    )
