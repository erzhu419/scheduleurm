"""Recalculate controlled-migration decisions from the final service cache.

The live checkpoint artifacts are used only for their measured migration cost
``K``.  Their historical rates, remaining work, and beneficial flags are
explicitly stale inputs.  Current and destination service rates come from
exact, hardware-local rows in a caller-supplied final unified
``ServiceRateCache``.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from algorithm.theorem_dispatch.migration import MigrationCost
from simulation.service_cache import ProfileRecord, ServiceRateCache


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_FINAL_CACHE = ARTIFACT_ROOT / "service_cache_v2_cpu_gpu_loaded_final_20260809.json"
DEFAULT_JSON_OUTPUT = ARTIFACT_ROOT / "migration_rate_recalibration_gate_20260809.json"
DEFAULT_MARKDOWN_OUTPUT = ARTIFACT_ROOT / "migration_rate_recalibration_gate_20260809.md"
PROGRESS_POINTS = (0.25, 0.50, 0.75)


@dataclass(frozen=True)
class ExactRateState:
    workload_key: str
    workload_env: str
    node: str
    node_bucket: str
    resource_state: str
    profile: int = 1
    allocation_workers: int = 1
    colocation_count: int = 1
    resident_mix: str = ""

    def snapshot(self) -> dict[str, Any]:
        return {
            "workload_key": self.workload_key,
            "workload_env": self.workload_env,
            "node": self.node,
            "node_bucket": self.node_bucket,
            "resource_state": self.resource_state,
            "profile": int(self.profile),
            "allocation_workers": int(self.allocation_workers),
            "colocation_count": int(self.colocation_count),
            "resident_mix": self.resident_mix,
        }


@dataclass(frozen=True)
class CostSignature:
    source_node: str
    dest_node: str
    checkpoint_mib: int
    checkpoint_policy: str
    sync_policy: str
    resume_policy: str

    def snapshot(self) -> dict[str, Any]:
        return {
            "source_node": self.source_node,
            "dest_node": self.dest_node,
            "checkpoint_mib": int(self.checkpoint_mib),
            "checkpoint_policy": self.checkpoint_policy,
            "sync_policy": self.sync_policy,
            "resume_policy": self.resume_policy,
        }


@dataclass(frozen=True)
class MigrationRateSpec:
    migration_id: str
    family: str
    source: ExactRateState
    target: ExactRateState
    cost_signature: CostSignature
    controlled_benchmark: bool = True
    ordinary_user_task: bool = False

    def __post_init__(self) -> None:
        if self.source.workload_key != self.target.workload_key:
            raise ValueError("source and target workload keys must match")
        if self.source.workload_env != self.target.workload_env:
            raise ValueError("source and target workload environments must match")
        if self.source.node != self.cost_signature.source_node:
            raise ValueError("source node and cost signature disagree")
        if self.target.node != self.cost_signature.dest_node:
            raise ValueError("target node and cost signature disagree")


CNN_COST = CostSignature(
    source_node="jtl110gpu",
    dest_node="jtl311linux",
    checkpoint_mib=96,
    checkpoint_policy="controlled_cnn_state_payload",
    sync_policy="tar_ssh",
    resume_policy="hash_verify_resume",
)
RL_COST = CostSignature(
    source_node="jtl110gpu",
    dest_node="jtl311linux",
    checkpoint_mib=48,
    checkpoint_policy="controlled_resac_cycle_state_payload",
    sync_policy="tar_ssh",
    resume_policy="hash_verify_resume",
)
CPU_COST = CostSignature(
    source_node="node003",
    dest_node="node005",
    checkpoint_mib=32,
    checkpoint_policy="controlled_cpu_state_payload",
    sync_policy="tar_ssh",
    resume_policy="hash_verify_resume",
)


DEFAULT_SPECS = (
    MigrationRateSpec(
        migration_id="cnn_jtl110gpu_to_jtl311linux",
        family="pure_gpu",
        source=ExactRateState(
            workload_key="gpu_cnn_torch_resnet50",
            workload_env="resnet50_synthetic_train_amp_b32_224",
            node="jtl110gpu",
            node_bucket="gpu_3080ti_12gb_dual:jtl110gpu",
            resource_state="empty",
        ),
        target=ExactRateState(
            workload_key="gpu_cnn_torch_resnet50",
            workload_env="resnet50_synthetic_train_amp_b32_224",
            node="jtl311linux",
            node_bucket="gpu_2080_8gb_dual:jtl311linux",
            resource_state="empty",
        ),
        cost_signature=CNN_COST,
    ),
    MigrationRateSpec(
        migration_id="resac_ant_jtl110gpu_to_jtl311linux",
        family="hybrid_rl",
        source=ExactRateState(
            workload_key="hybrid_rl_resac_ant",
            workload_env="Ant-v2",
            node="jtl110gpu",
            node_bucket="gpu_3080ti_12gb_dual:jtl110gpu",
            resource_state="empty",
            profile=2,
            colocation_count=2,
        ),
        target=ExactRateState(
            workload_key="hybrid_rl_resac_ant",
            workload_env="Ant-v2",
            node="jtl311linux",
            node_bucket="gpu_2080_8gb_dual:jtl311linux",
            resource_state="empty",
            profile=2,
            colocation_count=2,
        ),
        cost_signature=RL_COST,
    ),
    MigrationRateSpec(
        migration_id="cpu_node003_half_to_node005_empty",
        family="pure_cpu",
        source=ExactRateState(
            workload_key="cpu_heavy_local_bench",
            workload_env="cpu",
            node="node003",
            node_bucket="node003:cpu_hpc_192c",
            resource_state="half_loaded",
            resident_mix="controlled_resident_workers=96",
        ),
        target=ExactRateState(
            workload_key="cpu_heavy_local_bench",
            workload_env="cpu",
            node="node005",
            node_bucket="node005:cpu_hpc_192c",
            resource_state="empty",
        ),
        cost_signature=CPU_COST,
    ),
    MigrationRateSpec(
        migration_id="cpu_node003_full_to_node005_empty",
        family="pure_cpu",
        source=ExactRateState(
            workload_key="cpu_heavy_local_bench",
            workload_env="cpu",
            node="node003",
            node_bucket="node003:cpu_hpc_192c",
            resource_state="full_loaded",
            resident_mix="controlled_resident_workers=180",
        ),
        target=ExactRateState(
            workload_key="cpu_heavy_local_bench",
            workload_env="cpu",
            node="node005",
            node_bucket="node005:cpu_hpc_192c",
            resource_state="empty",
        ),
        cost_signature=CPU_COST,
    ),
)


@dataclass(frozen=True)
class _MeasuredCost:
    cost: MigrationCost
    path: Path
    artifact_sha256: str
    row_index: int
    row: Mapping[str, Any]


def build_migration_rate_recalibration_gate(
    *,
    final_cache_path: Path,
    cost_artifact_paths: Iterable[Path],
    specs: Sequence[MigrationRateSpec] = DEFAULT_SPECS,
) -> dict[str, Any]:
    """Build a fail-closed migration-rate recalibration certificate."""

    cache_path = Path(final_cache_path)
    cost_paths = tuple(sorted({Path(path) for path in cost_artifact_paths}, key=str))
    common = {
        "gate": "migration_rate_recalibration_gate",
        "schema_version": 1,
        "rate_equation": "W/mu_old - W/mu_new > K",
        "progress_points": list(PROGRESS_POINTS),
        "final_unified_cache_path": str(cache_path),
        "cost_artifact_paths": [str(path) for path in cost_paths],
        "required_migration_ids": [spec.migration_id for spec in specs],
        "required_row_count": len(specs) * len(PROGRESS_POINTS),
        "ordinary_user_tasks_excluded": True,
        "legacy_or_state_fallback_allowed": False,
        "old_cost_artifact_rates_used": False,
        "old_cost_artifact_beneficial_flags_inherited": False,
        "claim_boundary": (
            "This certificate recalculates controlled benchmark migration actions "
            "for the declared exact hardware/load states. It does not authorize "
            "migration of ordinary user tasks or extrapolate to missing states."
        ),
    }
    if not cache_path.is_file():
        return {
            **common,
            "status": "WAIT_FINAL_UNIFIED_CACHE",
            "pass": False,
            "rows": [],
            "ready_row_count": 0,
            "blocked_row_count": 0,
            "wait_reasons": [
                {
                    "code": "FINAL_UNIFIED_CACHE_MISSING",
                    "path": str(cache_path),
                }
            ],
            "validation_errors": [],
        }
    missing_cost_paths = [str(path) for path in cost_paths if not path.is_file()]
    if not cost_paths or missing_cost_paths:
        return {
            **common,
            "status": "WAIT_MIGRATION_COST_ARTIFACT",
            "pass": False,
            "rows": [],
            "ready_row_count": 0,
            "blocked_row_count": 0,
            "wait_reasons": [
                {
                    "code": "MIGRATION_COST_INPUT_MISSING",
                    "paths": missing_cost_paths or ["no cost artifact supplied"],
                }
            ],
            "validation_errors": [],
        }

    try:
        cache_bytes = cache_path.read_bytes()
        cache_payload = json.loads(cache_bytes.decode("utf-8"))
        if not isinstance(cache_payload, dict):
            raise ValueError("cache JSON root must be an object")
        snapshot = cache_payload.get("service_cache_snapshot", cache_payload)
        if not isinstance(snapshot, dict):
            raise ValueError("service_cache_snapshot must be an object")
        cache = ServiceRateCache.from_snapshot(snapshot)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return {
            **common,
            "status": "FAIL_INVALID_FINAL_UNIFIED_CACHE",
            "pass": False,
            "rows": [],
            "ready_row_count": 0,
            "blocked_row_count": 0,
            "wait_reasons": [],
            "validation_errors": [
                {"code": "INVALID_FINAL_UNIFIED_CACHE", "detail": f"{type(exc).__name__}: {exc}"}
            ],
        }

    cost_rows, cost_errors = _load_measured_costs(cost_paths)
    if cost_errors:
        return {
            **common,
            "status": "FAIL_INVALID_MIGRATION_COST_ARTIFACT",
            "pass": False,
            "final_unified_cache_sha256": hashlib.sha256(cache_bytes).hexdigest(),
            "rows": [],
            "ready_row_count": 0,
            "blocked_row_count": 0,
            "wait_reasons": [],
            "validation_errors": cost_errors,
        }

    rows: list[dict[str, Any]] = []
    for spec in specs:
        rows.extend(_evaluate_spec(cache=cache, costs=cost_rows, spec=spec))
    ready = [row for row in rows if row.get("recalculation_ready") is True]
    blocked = [row for row in rows if row.get("recalculation_ready") is not True]
    all_expected_rows = len(rows) == len(specs) * len(PROGRESS_POINTS)
    certificate_ready = all_expected_rows and not blocked
    return {
        **common,
        "status": "PASS" if certificate_ready else "BLOCKED",
        "pass": bool(certificate_ready),
        "certificate_ready": bool(certificate_ready),
        "final_unified_cache_sha256": hashlib.sha256(cache_bytes).hexdigest(),
        "cost_artifact_count": len(cost_paths),
        "valid_measured_cost_row_count": len(cost_rows),
        "rows": rows,
        "ready_row_count": len(ready),
        "blocked_row_count": len(blocked),
        "beneficial_row_count": sum(
            row.get("beneficial_by_recalculated_threshold") is True for row in ready
        ),
        "nonbeneficial_row_count": sum(
            row.get("beneficial_by_recalculated_threshold") is False for row in ready
        ),
        "wait_reasons": [],
        "validation_errors": [],
    }


def _evaluate_spec(
    *,
    cache: ServiceRateCache,
    costs: Sequence[_MeasuredCost],
    spec: MigrationRateSpec,
) -> list[dict[str, Any]]:
    base = {
        "migration_id": spec.migration_id,
        "family": spec.family,
        "workload_key": spec.source.workload_key,
        "workload_env": spec.source.workload_env,
        "source_state_request": spec.source.snapshot(),
        "target_state_request": spec.target.snapshot(),
        "required_cost_signature": spec.cost_signature.snapshot(),
    }
    if not spec.controlled_benchmark or spec.ordinary_user_task:
        return [
            _blocked_row(
                base,
                point,
                "ORDINARY_USER_TASK_EXCLUDED",
                "migration certificates are restricted to controlled benchmark tasks",
            )
            for point in PROGRESS_POINTS
        ]

    source_record, source_error = _lookup_rate_record(cache, spec.source)
    target_record, target_error = _lookup_rate_record(cache, spec.target)
    if source_error or target_error:
        reason = source_error or target_error or "RATE_CERTIFICATE_MISSING"
        detail = "; ".join(
            text
            for text in (
                f"source: {source_error}" if source_error else "",
                f"target: {target_error}" if target_error else "",
            )
            if text
        )
        return [
            _blocked_row(base, point, reason, detail)
            for point in PROGRESS_POINTS
        ]
    assert source_record is not None and target_record is not None

    if source_record.unit != target_record.unit:
        return [
            _blocked_row(
                base,
                point,
                "SERVICE_UNIT_MISMATCH",
                f"source unit {source_record.unit!r} != target unit {target_record.unit!r}",
            )
            for point in PROGRESS_POINTS
        ]
    if not math.isclose(
        source_record.total_units,
        target_record.total_units,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        return [
            _blocked_row(
                base,
                point,
                "TOTAL_WORK_UNIT_MISMATCH",
                (
                    f"source total_units={source_record.total_units}, "
                    f"target total_units={target_record.total_units}"
                ),
            )
            for point in PROGRESS_POINTS
        ]

    current_rate = _per_task_lower_service(source_record)
    target_rate = _per_task_lower_service(target_record)
    total_work = float(source_record.total_units)
    out = []
    for point in PROGRESS_POINTS:
        matches = _matching_costs(costs, signature=spec.cost_signature, point=point)
        if not matches:
            out.append(
                _blocked_row(
                    base,
                    point,
                    "EXACT_COST_SIGNATURE_MISSING",
                    "no measured K matches source/destination/checkpoint/sync/resume signature and progress point",
                )
            )
            continue
        measured = max(matches, key=lambda item: item.cost.total_penalty_units)
        remaining = total_work * (1.0 - point)
        keep_s = remaining / current_rate
        migrate_service_s = remaining / target_rate
        migrate_s = measured.cost.total_penalty_units + migrate_service_s
        net_saving = keep_s - migrate_s
        old_flag = measured.row.get("beneficial_by_threshold")
        out.append(
            {
                **base,
                "progress_fraction": point,
                "recalculation_ready": True,
                "block_code": "",
                "block_detail": "",
                "service_unit": source_record.unit,
                "total_work_units": total_work,
                "remaining_work_units": remaining,
                "current_lower_service_units_per_s": current_rate,
                "target_lower_service_units_per_s": target_rate,
                "keep_remaining_completion_s": keep_s,
                "migrate_service_completion_s": migrate_service_s,
                "migration_total_penalty_s": measured.cost.total_penalty_units,
                "migrate_remaining_completion_s": migrate_s,
                "migration_net_saving_s": net_saving,
                "beneficial_by_recalculated_threshold": net_saving > 0.0,
                "migration_action_theorem_ready": net_saving > 0.0,
                "source_rate_record": _rate_record_audit(source_record),
                "target_rate_record": _rate_record_audit(target_record),
                "migration_cost": measured.cost.snapshot(),
                "cost_source": {
                    "path": str(measured.path),
                    "sha256": measured.artifact_sha256,
                    "row_index": measured.row_index,
                    "cost_artifact_workload_key": str(measured.row.get("workload_key") or ""),
                    "signature_reused_across_workload_key": (
                        str(measured.row.get("workload_key") or "")
                        != spec.source.workload_key
                    ),
                },
                "stale_cost_artifact_fields_audit": {
                    "old_current_rate": measured.row.get("current_rate"),
                    "old_target_rate": measured.row.get("target_rate"),
                    "old_remaining_work": measured.row.get("remaining_work"),
                    "old_beneficial_by_threshold": old_flag,
                    "old_theorem_ready": measured.row.get("theorem_ready"),
                    "old_flag_matches_recalculation": (
                        bool(old_flag) == (net_saving > 0.0)
                        if isinstance(old_flag, bool)
                        else None
                    ),
                    "inherited": False,
                },
            }
        )
    return out


def _lookup_rate_record(
    cache: ServiceRateCache,
    request: ExactRateState,
) -> tuple[ProfileRecord | None, str | None]:
    result = cache.lookup_statewise_exact(
        request.workload_key,
        workload_env=request.workload_env,
        node_bucket=request.node_bucket,
        resource_state=request.resource_state,
        profile=request.profile,
        allocation_workers=request.allocation_workers,
        colocation_count=request.colocation_count,
        resident_mix=request.resident_mix,
    )
    if result.is_scoped_boundary:
        return None, "EXACT_STATE_CAPACITY_BOUNDARY"
    if not result.is_exact or result.record is None:
        return None, "EXACT_STATE_RATE_MISSING"
    record = result.record
    if record.capacity_boundary:
        return None, "EXACT_STATE_CAPACITY_BOUNDARY"
    if not record.stable_rate_ready:
        return None, "STABLE_RATE_CERTIFICATE_MISSING"
    if not record.completion_model_ready:
        return None, "COMPLETION_MODEL_CERTIFICATE_MISSING"
    if int(record.completion_model_sample_count) <= 0:
        return None, "COMPLETION_MODEL_SAMPLE_COUNT_MISSING"
    if not _node_bucket_is_hardware_local(record.node_bucket, request.node):
        return None, "RATE_ROW_NOT_HARDWARE_LOCAL"
    if not str(record.hardware_class or "").strip():
        return None, "HARDWARE_CLASS_MISSING"
    if not str(record.source or "").strip():
        return None, "RATE_PROVENANCE_MISSING"
    eta_source = str(record.eta_source or "").strip().lower()
    if not eta_source or "hist" in eta_source:
        return None, "TASK_NATIVE_ETA_PROVENANCE_MISSING"
    if "lcb" not in eta_source and "lower_service" not in eta_source:
        return None, "LOWER_SERVICE_CERTIFICATE_MISSING"
    if not math.isfinite(float(record.total_units)) or float(record.total_units) <= 0.0:
        return None, "TOTAL_WORK_UNITS_INVALID"
    if (
        not math.isfinite(float(record.completion_unit_s))
        or float(record.completion_unit_s) <= 0.0
        or not math.isfinite(float(record.completion_total_wall_s))
        or float(record.completion_total_wall_s) <= 0.0
    ):
        return None, "COMPLETION_MODEL_PARAMETERS_INVALID"
    try:
        _per_task_lower_service(record)
    except ValueError as exc:
        return None, f"LOWER_SERVICE_INVALID:{exc}"
    return record, None


def _per_task_lower_service(record: ProfileRecord) -> float:
    rates = tuple(float(value) for value in record.per_task_rates)
    if len(rates) != int(record.colocation_count):
        raise ValueError("per_task_rates length does not match colocation_count")
    if not rates or any(not math.isfinite(value) or value <= 0.0 for value in rates):
        raise ValueError("per-task lower service must be finite and positive")
    aggregate = float(record.aggregate_rate)
    if not math.isfinite(aggregate) or aggregate <= 0.0:
        raise ValueError("aggregate lower service must be finite and positive")
    if not math.isclose(sum(rates), aggregate, rel_tol=1e-7, abs_tol=1e-10):
        raise ValueError("per-task and aggregate lower service disagree")
    return min(rates)


def _node_bucket_is_hardware_local(node_bucket: str, node: str) -> bool:
    bucket = str(node_bucket or "")
    physical = str(node or "")
    return bool(
        physical
        and (
            bucket == physical
            or bucket.startswith(f"{physical}:")
            or bucket.endswith(f":{physical}")
        )
    )


def _load_measured_costs(
    paths: Sequence[Path],
) -> tuple[list[_MeasuredCost], list[dict[str, Any]]]:
    measured: list[_MeasuredCost] = []
    errors: list[dict[str, Any]] = []
    for path in paths:
        try:
            raw = path.read_bytes()
            payload = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append(
                {"code": "COST_ARTIFACT_UNREADABLE", "path": str(path), "detail": f"{type(exc).__name__}: {exc}"}
            )
            continue
        if not isinstance(payload, dict):
            errors.append({"code": "COST_ARTIFACT_ROOT_INVALID", "path": str(path)})
            continue
        if not (
            payload.get("gate") == "live_checkpoint_migration_cost_gate"
            and payload.get("status") == "LIVE_CHECKPOINT_MIGRATION_COST_READY"
            and payload.get("pass") is True
            and int(payload.get("pending_count") or 0) == 0
            and int(payload.get("blocked_count") or 0) == 0
        ):
            errors.append({"code": "COST_ARTIFACT_NOT_READY", "path": str(path)})
            continue
        rows = payload.get("rows")
        if not isinstance(rows, list):
            rows = payload.get("measured_rows")
        if not isinstance(rows, list):
            errors.append({"code": "COST_ARTIFACT_ROWS_INVALID", "path": str(path)})
            continue
        digest = hashlib.sha256(raw).hexdigest()
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                errors.append({"code": "COST_ROW_INVALID", "path": str(path), "row_index": index})
                continue
            try:
                cost = _validated_measured_cost(row)
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(
                    {
                        "code": "COST_ROW_INVALID",
                        "path": str(path),
                        "row_index": index,
                        "detail": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            measured.append(
                _MeasuredCost(
                    cost=cost,
                    path=path,
                    artifact_sha256=digest,
                    row_index=index,
                    row=row,
                )
            )
    return measured, errors


def _validated_measured_cost(row: Mapping[str, Any]) -> MigrationCost:
    if not (
        row.get("launched") is True
        and row.get("measurement_valid") is True
        and row.get("status") == "MEASURED"
    ):
        raise ValueError("row is not a valid live measurement")
    point = _finite_float(row.get("progress_fraction"), "progress_fraction")
    if not any(math.isclose(point, expected, rel_tol=0.0, abs_tol=1e-9) for expected in PROGRESS_POINTS):
        raise ValueError("unexpected progress point")
    checkpoint_mib = int(row["checkpoint_mib"])
    if checkpoint_mib <= 0:
        raise ValueError("checkpoint_mib must be positive")
    for key in ("source_node", "dest_node", "checkpoint_policy", "sync_policy", "resume_policy"):
        if not str(row.get(key) or "").strip():
            raise ValueError(f"{key} missing")
    checkpoint = row.get("checkpoint_meta")
    resume = row.get("resume_meta")
    if not isinstance(checkpoint, Mapping) or not isinstance(resume, Mapping):
        raise ValueError("checkpoint/resume metadata missing")
    payload_bytes = checkpoint_mib * 1024 * 1024
    if int(checkpoint.get("checkpoint_mib") or 0) != checkpoint_mib:
        raise ValueError("checkpoint size signature mismatch")
    if int(checkpoint.get("payload_bytes") or 0) != payload_bytes:
        raise ValueError("checkpoint payload size mismatch")
    if int(resume.get("payload_bytes") or 0) != payload_bytes:
        raise ValueError("resume payload size mismatch")
    if resume.get("hash_ok") is not True:
        raise ValueError("resume hash verification missing")
    if not str(checkpoint.get("sha256") or "") or checkpoint.get("sha256") != resume.get("sha256"):
        raise ValueError("checkpoint/resume hash mismatch")
    if resume.get("expected_sha256") != checkpoint.get("sha256"):
        raise ValueError("resume expected hash mismatch")
    checkpoint_flush = _nonnegative_float(
        row.get("checkpoint_flush_s"), "checkpoint_flush_s"
    )
    resume_warmup = _nonnegative_float(
        row.get("resume_warmup_s"), "resume_warmup_s"
    )
    if not math.isclose(
        checkpoint_flush,
        _nonnegative_float(
            checkpoint.get("checkpoint_flush_s"),
            "checkpoint_meta.checkpoint_flush_s",
        ),
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ValueError("checkpoint flush time disagrees with checkpoint metadata")
    if not math.isclose(
        resume_warmup,
        _nonnegative_float(
            resume.get("resume_warmup_s"),
            "resume_meta.resume_warmup_s",
        ),
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ValueError("resume warmup time disagrees with resume metadata")
    cost = MigrationCost(
        checkpoint_flush_s=checkpoint_flush,
        sync_s=_nonnegative_float(row.get("sync_s"), "sync_s"),
        environment_staging_s=_nonnegative_float(row.get("environment_staging_s"), "environment_staging_s"),
        resume_warmup_s=resume_warmup,
        lost_work_s=_nonnegative_float(row.get("lost_work_s"), "lost_work_s"),
        risk_penalty_units=_nonnegative_float(row.get("risk_penalty_units"), "risk_penalty_units"),
    )
    for key, recomputed in (
        ("total_time_s", cost.total_time_s),
        ("total_penalty_units", cost.total_penalty_units),
    ):
        reported = _finite_float(row.get(key), key)
        if not math.isclose(reported, recomputed, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"reported {key} disagrees with cost components")
    return cost


def _matching_costs(
    costs: Sequence[_MeasuredCost],
    *,
    signature: CostSignature,
    point: float,
) -> list[_MeasuredCost]:
    expected = signature.snapshot()
    out = []
    for measured in costs:
        row = measured.row
        if any(row.get(key) != value for key, value in expected.items()):
            continue
        if not math.isclose(
            float(row.get("progress_fraction")), point, rel_tol=0.0, abs_tol=1e-9
        ):
            continue
        out.append(measured)
    return out


def _rate_record_audit(record: ProfileRecord) -> dict[str, Any]:
    return {
        "workload_key": record.workload_key,
        "workload_env": record.workload_env,
        "node_bucket": record.node_bucket,
        "hardware_class": record.hardware_class,
        "resource_state": record.resource_state,
        "resident_mix": record.resident_mix,
        "profile": record.profile,
        "allocation_workers": record.allocation_workers,
        "colocation_count": record.colocation_count,
        "unit": record.unit,
        "total_units": record.total_units,
        "aggregate_lower_service_units_per_s": record.aggregate_rate,
        "per_task_lower_service_units_per_s": list(record.per_task_rates),
        "eta_source": record.eta_source,
        "stable_rate_ready": record.stable_rate_ready,
        "completion_model_ready": record.completion_model_ready,
        "completion_model_sample_count": record.completion_model_sample_count,
        "source": record.source,
    }


def _blocked_row(
    base: Mapping[str, Any],
    point: float,
    code: str,
    detail: str,
) -> dict[str, Any]:
    return {
        **base,
        "progress_fraction": float(point),
        "recalculation_ready": False,
        "beneficial_by_recalculated_threshold": None,
        "migration_action_theorem_ready": False,
        "block_code": code,
        "block_detail": detail,
    }


def _finite_float(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _nonnegative_float(value: Any, label: str) -> float:
    number = _finite_float(value, label)
    if number < 0.0:
        raise ValueError(f"{label} must be nonnegative")
    return number


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Migration Rate Recalibration Gate",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Pass: `{str(bool(report.get('pass'))).lower()}`",
        f"- Final cache: `{report.get('final_unified_cache_path')}`",
        f"- Ready rows: `{report.get('ready_row_count', 0)}` / `{report.get('required_row_count', 0)}`",
        f"- Ordinary user tasks excluded: `{str(bool(report.get('ordinary_user_tasks_excluded'))).lower()}`",
        f"- Old rates/flags inherited: `false`",
        "",
    ]
    wait_reasons = report.get("wait_reasons") or []
    errors = report.get("validation_errors") or []
    if wait_reasons:
        lines.extend(["## Wait Reasons", ""])
        lines.extend(f"- `{row.get('code')}`: {row}" for row in wait_reasons)
        lines.append("")
    if errors:
        lines.extend(["## Validation Errors", ""])
        lines.extend(f"- `{row.get('code')}`: {row.get('detail') or row.get('path') or ''}" for row in errors)
        lines.append("")
    rows = report.get("rows") or []
    if rows:
        lines.extend(
            [
                "| Migration | Progress | Ready | Old rate | New rate | Keep (s) | Migrate (s) | Net (s) | Beneficial | Block |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in rows:
            lines.append(
                "| `{migration}` | {progress:.2f} | {ready} | {old_rate} | {new_rate} | {keep} | {migrate} | {net} | {beneficial} | `{block}` |".format(
                    migration=row.get("migration_id"),
                    progress=float(row.get("progress_fraction") or 0.0),
                    ready=str(bool(row.get("recalculation_ready"))).lower(),
                    old_rate=_fmt(row.get("current_lower_service_units_per_s")),
                    new_rate=_fmt(row.get("target_lower_service_units_per_s")),
                    keep=_fmt(row.get("keep_remaining_completion_s")),
                    migrate=_fmt(row.get("migrate_remaining_completion_s")),
                    net=_fmt(row.get("migration_net_saving_s")),
                    beneficial=(
                        "n/a"
                        if row.get("beneficial_by_recalculated_threshold") is None
                        else str(bool(row.get("beneficial_by_recalculated_threshold"))).lower()
                    ),
                    block=row.get("block_code") or "",
                )
            )
        lines.append("")
    lines.extend([str(report.get("claim_boundary") or ""), ""])
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def write_outputs(
    report: Mapping[str, Any],
    *,
    json_output: Path,
    markdown_output: Path,
) -> None:
    _atomic_write(
        Path(json_output),
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    _atomic_write(Path(markdown_output), markdown_report(report))


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _default_cost_artifacts() -> tuple[Path, ...]:
    return tuple(sorted(ARTIFACT_ROOT.glob("live_checkpoint_migration_cost*.json")))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-cache", type=Path, default=DEFAULT_FINAL_CACHE)
    parser.add_argument("--cost-artifact", type=Path, action="append")
    parser.add_argument("--output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN_OUTPUT)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    costs = tuple(args.cost_artifact or _default_cost_artifacts())
    report = build_migration_rate_recalibration_gate(
        final_cache_path=args.final_cache,
        cost_artifact_paths=costs,
    )
    write_outputs(
        report,
        json_output=args.output,
        markdown_output=args.markdown_output,
    )
    print(json.dumps({"status": report["status"], "pass": report["pass"]}, sort_keys=True))
    if report["pass"]:
        return 0
    if str(report["status"]).startswith("WAIT"):
        return 3
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
