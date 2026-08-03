"""Fail-closed merge of critical GPU LCB certificates into service cache v2.

The merge is deliberately atomic at the evidence level: all three declared
representative GPU hardware gates must be PASS under the v5 measurement
protocol before any final cache snapshot is exposed or written.  Every added
row is exact-state-only and is audited against the legacy two-key index.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping

from simulation.service_cache import ProfileRecord, ServiceRateCache

from .critical_gpu_completion_campaign import (
    CALIBRATION_WAVES,
    HOLDOUT_WAVE,
    NODE_SPECS,
    PROTOCOL,
    TRAINING_WAVES,
    campaign_cells,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
CAMPAIGN_DATE = "20260803"
RESOURCE_STATE = "empty"
REPRESENTATIVE_NODES = ("jtl110gpu", "node007", "jtl311linux")
EXPECTED_CELL_COUNT_PER_NODE = 9
EXPECTED_NATURAL_COMPLETION_SAMPLES = len(TRAINING_WAVES) + len(
    CALIBRATION_WAVES
)
DEFAULT_BASE = ARTIFACT_ROOT / "service_cache_v2_critical_phase_20260802.json"
DEFAULT_REPORT = ARTIFACT_ROOT / "critical_gpu_lcb_cache_merge_gate_20260803.json"
DEFAULT_MARKDOWN = DEFAULT_REPORT.with_suffix(".md")
DEFAULT_CACHE_OUTPUT = (
    ARTIFACT_ROOT / "service_cache_v2_critical_cpu_gpu_phase_20260803.json"
)
ETA_SOURCE = (
    "task_native_tqdm_progress_natural_completion_"
    "operational_pre_holdout_mean_jct_wave_max_split_conformal_lcb"
)
COMMAND_FINGERPRINT = (
    "live_tqdm_service_cache_v2_critical_gpu_completion_lcb_v1"
)


def default_gate_paths(
    *, artifact_root: Path = ARTIFACT_ROOT
) -> dict[str, Path]:
    return {
        node: Path(artifact_root)
        / f"critical_gpu_stochastic_lcb_gate_v5_{node}_{CAMPAIGN_DATE}.json"
        for node in REPRESENTATIVE_NODES
    }


def build_critical_gpu_lcb_cache_merge(
    *,
    base_cache_path: Path = DEFAULT_BASE,
    gate_paths: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    """Return a complete cache merge or a snapshot-free WAIT/FAIL report."""

    base_path = Path(base_cache_path)
    supplied = (
        {str(node): Path(path) for node, path in gate_paths.items()}
        if gate_paths is not None
        else default_gate_paths()
    )
    validation_errors: list[dict[str, Any]] = []
    wait_reasons: list[dict[str, Any]] = []
    unexpected_nodes = sorted(set(supplied) - set(REPRESENTATIVE_NODES))
    if unexpected_nodes:
        validation_errors.append(
            _issue(
                "UNREGISTERED_GATE_NODE",
                detail=f"unexpected gate nodes: {unexpected_nodes!r}",
            )
        )

    gate_payloads: dict[str, tuple[dict[str, Any], Path]] = {}
    gate_audits: list[dict[str, Any]] = []
    for node in REPRESENTATIVE_NODES:
        path = supplied.get(node)
        if path is None or not path.is_file():
            wait_reasons.append(
                _issue(
                    "MISSING_GPU_GATE",
                    node=node,
                    detail=str(path) if path is not None else "path not supplied",
                )
            )
            gate_audits.append(
                {
                    "node": node,
                    "path": str(path) if path is not None else None,
                    "status": "MISSING",
                    "protocol_pass": False,
                }
            )
            continue
        payload, load_error, digest = _load_json_object(path)
        if load_error is not None:
            validation_errors.append(
                _issue("UNREADABLE_GPU_GATE", node=node, detail=load_error)
            )
            gate_audits.append(
                {
                    "node": node,
                    "path": str(path),
                    "status": "UNREADABLE",
                    "sha256": digest,
                    "protocol_pass": False,
                }
            )
            continue
        assert payload is not None
        source_status = str(payload.get("status") or "")
        audit = {
            "node": node,
            "path": str(path),
            "sha256": digest,
            "status": source_status,
            "pass": bool(payload.get("pass")),
            "schema_version": payload.get("schema_version"),
            "measurement_protocol": payload.get("measurement_protocol"),
            "protocol_pass": bool(
                source_status == "PASS"
                and payload.get("pass") is True
                and payload.get("measurement_protocol") == PROTOCOL
            ),
        }
        gate_audits.append(audit)
        if source_status.startswith("WAIT"):
            wait_reasons.append(
                _issue(
                    "GPU_GATE_NOT_READY",
                    node=node,
                    detail=f"source status={source_status!r}",
                )
            )
            continue
        if source_status != "PASS" or payload.get("pass") is not True:
            validation_errors.append(
                _issue(
                    "GPU_GATE_NOT_PASSING",
                    node=node,
                    detail=(
                        f"source status={source_status!r}, "
                        f"pass={payload.get('pass')!r}"
                    ),
                )
            )
            continue
        gate_payloads[node] = (payload, path)

    common = {
        "gate": "critical_gpu_lcb_cache_merge",
        "schema_version": 1,
        "base_cache_path": str(base_path),
        "required_representative_nodes": list(REPRESENTATIVE_NODES),
        "required_measurement_protocol": PROTOCOL,
        "source_gate_contract": "v5_PASS",
        "resource_state": RESOURCE_STATE,
        "expected_cells_per_node": EXPECTED_CELL_COUNT_PER_NODE,
        "expected_total_inserted_rows": (
            len(REPRESENTATIVE_NODES) * EXPECTED_CELL_COUNT_PER_NODE
        ),
        "source_gate_audits": gate_audits,
        "wait_reasons": wait_reasons,
        "validation_errors": validation_errors,
        "legacy_index_unchanged": False,
        "complete_cache_ready": False,
        "final_cache_snapshot_included": False,
        "inserted_count": 0,
        "inserted_rows": [],
        "claim_boundary": (
            "The final cache exists only when jtl110gpu, node007, and "
            "jtl311linux each provide a hardware-local v5 PASS certificate for "
            "all nine pre-registered empty-state GPU actions. Rows are exact "
            "workload_env x node_bucket x resource_state x profile records; "
            "they never update the legacy workload/profile fallback."
        ),
    }
    if validation_errors:
        return {**common, "status": "FAIL_SOURCE_GATES", "pass": False}
    if wait_reasons:
        return {**common, "status": "WAIT_GPU_GATES", "pass": False}

    if not base_path.is_file():
        common["validation_errors"].append(
            _issue("MISSING_BASE_CACHE", detail=str(base_path))
        )
        return {**common, "status": "FAIL_BASE_CACHE", "pass": False}
    try:
        base_bytes = base_path.read_bytes()
        cache = ServiceRateCache.from_snapshot(json.loads(base_bytes.decode("utf-8")))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        common["validation_errors"].append(
            _issue(
                "INVALID_BASE_CACHE",
                detail=f"{type(exc).__name__}: {exc}",
            )
        )
        return {**common, "status": "FAIL_BASE_CACHE", "pass": False}

    legacy_before = _legacy_index_snapshot(cache)
    inserted: list[dict[str, Any]] = []
    exact_keys: set[tuple[Any, ...]] = set()
    try:
        for node in REPRESENTATIVE_NODES:
            payload, source_path = gate_payloads[node]
            rows = _validated_gate_rows(payload, node=node, path=source_path)
            for row in rows:
                record = _profile_record(row, source=source_path)
                exact_key = (
                    record.workload_key,
                    record.workload_env,
                    record.node_bucket,
                    record.resource_state,
                    record.profile,
                )
                if exact_key in exact_keys:
                    raise ValueError(f"duplicate exact GPU cache key {exact_key!r}")
                exact_keys.add(exact_key)
                cache.add(record, force_replace=True)
                lookup = cache.lookup_statewise_exact(
                    record.workload_key,
                    workload_env=record.workload_env,
                    node_bucket=record.node_bucket,
                    resource_state=record.resource_state,
                    allocation_workers=record.allocation_workers,
                    colocation_count=record.colocation_count,
                    resident_mix=record.resident_mix,
                )
                if not lookup.is_exact or lookup.record != record:
                    raise ValueError(
                        f"exact statewise lookup did not round-trip {exact_key!r}"
                    )
                inserted.append(record.snapshot())
    except (KeyError, TypeError, ValueError) as exc:
        common["validation_errors"].append(
            _issue(
                "GPU_ROW_VALIDATION_FAILED",
                detail=f"{type(exc).__name__}: {exc}",
            )
        )
        return {**common, "status": "FAIL_VALIDATION", "pass": False}

    expected_total = len(REPRESENTATIVE_NODES) * EXPECTED_CELL_COUNT_PER_NODE
    if len(inserted) != expected_total:
        common["validation_errors"].append(
            _issue(
                "INSERTED_ROW_COUNT_MISMATCH",
                detail=f"expected {expected_total}, observed {len(inserted)}",
            )
        )
        return {**common, "status": "FAIL_VALIDATION", "pass": False}

    legacy_after = _legacy_index_snapshot(cache)
    if legacy_after != legacy_before:
        common["validation_errors"].append(
            _issue(
                "LEGACY_INDEX_CHANGED",
                detail="legacy records, boundaries, or capacity caps changed",
            )
        )
        return {**common, "status": "FAIL_LEGACY_INDEX", "pass": False}

    snapshot = cache.snapshot()
    return {
        **common,
        "status": "PASS",
        "pass": True,
        "base_cache_sha256": hashlib.sha256(base_bytes).hexdigest(),
        "legacy_index_unchanged": True,
        "legacy_index_sha256_before": _json_sha256(legacy_before),
        "legacy_index_sha256_after": _json_sha256(legacy_after),
        "complete_cache_ready": True,
        "final_cache_snapshot_included": True,
        "inserted_count": len(inserted),
        "inserted_rows": inserted,
        "service_cache_snapshot": snapshot,
    }


def _load_json_object(
    path: Path,
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, f"{type(exc).__name__}: {exc}", None
    digest = hashlib.sha256(raw).hexdigest()
    if not isinstance(payload, dict):
        return None, "JSON root must be an object", digest
    return payload, None, digest


def _validated_gate_rows(
    payload: Mapping[str, Any],
    *,
    node: str,
    path: Path,
) -> list[Mapping[str, Any]]:
    spec = NODE_SPECS[node]
    required_scalars = {
        "gate": "critical_gpu_stochastic_lcb_gate",
        "schema_version": 1,
        "status": "PASS",
        "pass": True,
        "certificate_ready": True,
        "node": node,
        "node_bucket": spec.node_bucket,
        "hardware_class": spec.hardware_class,
        "resource_state": RESOURCE_STATE,
        "measurement_protocol": PROTOCOL,
        "same_measurement_protocol_all_waves": True,
        "all_measurements_ready": True,
        "same_cell_set_all_waves": True,
        "hardware_local_only": True,
        "smoke_wave_excluded": True,
        "expected_cell_count": EXPECTED_CELL_COUNT_PER_NODE,
        "expected_wave_count": (
            len(TRAINING_WAVES) + len(CALIBRATION_WAVES) + 1
        ),
    }
    for field, expected in required_scalars.items():
        if payload.get(field) != expected:
            raise ValueError(
                f"{path}: {field} expected {expected!r}, got {payload.get(field)!r}"
            )
    if payload.get("validation_errors"):
        raise ValueError(f"{path}: PASS gate contains validation errors")
    if payload.get("wait_reasons"):
        raise ValueError(f"{path}: PASS gate contains wait reasons")
    certificate = payload.get("certificate") or {}
    for field in (
        "constructed",
        "finite_sample_rank_ready",
        "all_holdout_bounds_valid",
    ):
        if certificate.get(field) is not True:
            raise ValueError(f"{path}: certificate.{field} is not true")
    margin = _finite_float(
        certificate.get("simultaneous_ratio_margin"),
        f"{path}: simultaneous_ratio_margin",
    )
    if margin < 0.0:
        raise ValueError(f"{path}: simultaneous ratio margin is negative")

    rows = certificate.get("rows") or []
    if not isinstance(rows, list):
        raise ValueError(f"{path}: certificate rows must be a list")
    expected = {
        _expected_row_key(cell): cell
        for cell in campaign_cells(node=node, wave=TRAINING_WAVES[0])
    }
    if len(expected) != EXPECTED_CELL_COUNT_PER_NODE:
        raise RuntimeError(f"campaign declares {len(expected)} cells for {node}")
    observed: dict[tuple[str, str, str, int], Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"{path}: certificate row must be an object")
        key = _row_key(row)
        if key in observed:
            raise ValueError(f"{path}: duplicate certificate row {key!r}")
        observed[key] = row
    if set(observed) != set(expected):
        raise ValueError(
            f"{path}: certificate cell set mismatch; "
            f"missing={sorted(set(expected) - set(observed))!r}, "
            f"unexpected={sorted(set(observed) - set(expected))!r}"
        )
    for key, row in observed.items():
        _validate_certificate_row(
            row,
            expected=expected[key],
            node=node,
            path=path,
        )
    return [observed[key] for key in sorted(observed)]


def _validate_certificate_row(
    row: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
    node: str,
    path: Path,
) -> None:
    spec = NODE_SPECS[node]
    profile = int(expected["profile"])
    gpu_count = len(spec.gpus)
    task_count = profile * gpu_count
    expected_scalars = {
        "workload_key": expected["workload_key"],
        "workload_env": expected["workload_env"],
        "node": node,
        "node_bucket": spec.node_bucket,
        "hardware_class": spec.hardware_class,
        "resource_state": RESOURCE_STATE,
        "profile": profile,
        "profile_axis": "tasks_per_gpu",
        "gpu_count": gpu_count,
        "total_task_count": task_count,
        "training_sample_count": len(TRAINING_WAVES),
        "calibration_wave_count": len(CALIBRATION_WAVES),
        "holdout_wave": HOLDOUT_WAVE,
    }
    for field, value in expected_scalars.items():
        if row.get(field) != value:
            raise ValueError(
                f"{path}: {node} row {field} expected {value!r}, "
                f"got {row.get(field)!r}"
            )
    for flag in (
        "drain_completion_holdout_covered",
        "mean_task_jct_holdout_covered",
        "lower_service_valid_on_holdout",
    ):
        if row.get(flag) is not True:
            raise ValueError(f"{path}: {node} row {flag} is not true")
    if row.get("lower_service_bound_kind") != (
        "phase_aware_wave_max_simultaneous_split_conformal"
    ):
        raise ValueError(f"{path}: unexpected lower-service bound kind")

    lower = _positive_float(
        row.get("lower_service_units_per_s"),
        f"{path}: lower_service_units_per_s",
    )
    realized = _positive_float(
        row.get("realized_service_units_per_s"),
        f"{path}: realized_service_units_per_s",
    )
    if lower > realized * (1.0 + 1e-12):
        raise ValueError(f"{path}: lower service exceeds holdout realized service")
    total_per_task = _positive_float(
        row.get("total_units_per_task"), f"{path}: total_units_per_task"
    )
    aggregate_units = _positive_float(
        row.get("aggregate_total_units"), f"{path}: aggregate_total_units"
    )
    if not math.isclose(
        aggregate_units,
        total_per_task * task_count,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ValueError(f"{path}: aggregate units do not match task count")

    operational = row.get("operational_pre_holdout_phase_model") or {}
    if int(operational.get("sample_count") or 0) != (
        EXPECTED_NATURAL_COMPLETION_SAMPLES
    ):
        raise ValueError(f"{path}: operational model sample count is not 12")
    mean_phase = operational.get("mean_jct_phase") or {}
    phase_total_units = _positive_float(
        mean_phase.get("total_units_per_task"),
        f"{path}: operational mean-JCT total units",
    )
    if not math.isclose(phase_total_units, total_per_task, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"{path}: operational model uses different task units")
    startup = _nonnegative_float(
        mean_phase.get("startup_overhead_s"), f"{path}: startup"
    )
    unit_s = _positive_float(
        mean_phase.get("completion_unit_s"), f"{path}: completion unit"
    )
    terminal = _nonnegative_float(
        mean_phase.get("terminal_overhead_s"), f"{path}: terminal"
    )
    predicted = startup + total_per_task * unit_s + terminal
    point = _positive_float(
        row.get("operational_point_mean_task_jct_s"),
        f"{path}: operational mean-JCT point",
    )
    if not math.isclose(predicted, point, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"{path}: operational mean-JCT point is inconsistent")
    _nonnegative_float(
        row.get("mean_jct_point_relative_error"),
        f"{path}: mean-JCT point error",
    )
    _positive_float(
        row.get("simultaneous_upper_drain_completion_s"),
        f"{path}: upper drain completion",
    )
    _positive_float(
        row.get("simultaneous_upper_mean_task_jct_s"),
        f"{path}: upper mean JCT",
    )


def _profile_record(row: Mapping[str, Any], *, source: Path) -> ProfileRecord:
    profile = int(row["profile"])
    gpu_count = int(row["gpu_count"])
    node_lower_service = float(row["lower_service_units_per_s"])
    per_gpu_aggregate_rate = node_lower_service / gpu_count
    total_units_per_task = float(row["total_units_per_task"])
    operational = row["operational_pre_holdout_phase_model"]
    mean_phase = operational["mean_jct_phase"]
    point_mean_jct = float(row["operational_point_mean_task_jct_s"])
    return ProfileRecord(
        workload_key=str(row["workload_key"]),
        command_fingerprint=COMMAND_FINGERPRINT,
        resource_kind=(
            "hybrid"
            if str(row["workload_key"]).startswith("hybrid_")
            else "gpu"
        ),
        node_bucket=str(row["node_bucket"]),
        profile=profile,
        unit=str(row["service_unit"]),
        total_units=total_units_per_task,
        aggregate_rate=per_gpu_aggregate_rate,
        per_task_rates=tuple(
            per_gpu_aggregate_rate / profile for _ in range(profile)
        ),
        source=str(source),
        capacity_boundary=False,
        gpu_count_observed=gpu_count,
        allocation_workers=1,
        colocation_count=profile,
        workload_env=str(row["workload_env"]),
        resource_state=RESOURCE_STATE,
        resident_mix="",
        eta_source=ETA_SOURCE,
        stable_rate_ready=True,
        hardware_class=str(row["hardware_class"]),
        completion_model_ready=True,
        completion_model_sample_count=int(operational["sample_count"]),
        startup_overhead_s=float(mean_phase["startup_overhead_s"]),
        completion_unit_s=float(mean_phase["completion_unit_s"]),
        finalization_overhead_s=float(mean_phase["terminal_overhead_s"]),
        checkpoint_observed_s=0.0,
        save_observed_s=0.0,
        completion_total_wall_s=point_mean_jct,
        completion_model_relative_error=float(
            row["mean_jct_point_relative_error"]
        ),
    )


def _expected_row_key(row: Mapping[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(row["workload_key"]),
        str(row["workload_env"]),
        str(row["node_bucket"]),
        int(row["profile"]),
    )


def _row_key(row: Mapping[str, Any]) -> tuple[str, str, str, int]:
    try:
        key = _expected_row_key(row)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid certificate row identity: {exc}") from exc
    if not key[0] or not key[1] or not key[2] or key[3] <= 0:
        raise ValueError(f"invalid certificate row identity {key!r}")
    return key


def _legacy_index_snapshot(cache: ServiceRateCache) -> dict[str, Any]:
    """Capture every data structure consulted by legacy two-key lookup."""

    records = getattr(cache, "_records")
    boundaries = getattr(cache, "_capacity_boundaries")
    caps = getattr(cache, "_capacity_caps")
    return {
        "records": [
            {
                "key": [str(key[0]), int(key[1])],
                "record": record.snapshot(),
            }
            for key, record in sorted(records.items())
        ],
        "capacity_boundaries": [
            {
                "key": [str(key[0]), int(key[1])],
                "record": record.snapshot(),
            }
            for key, record in sorted(boundaries.items())
        ],
        "capacity_caps": [
            {"workload_key": str(key), "profile": int(value)}
            for key, value in sorted(caps.items())
        ],
    }


def _positive_float(value: Any, label: str) -> float:
    result = _finite_float(value, label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _nonnegative_float(value: Any, label: str) -> float:
    result = _finite_float(value, label)
    if result < 0.0:
        raise ValueError(f"{label} must be nonnegative")
    return result


def _finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _json_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _issue(
    code: str,
    *,
    node: str | None = None,
    detail: str = "",
) -> dict[str, Any]:
    return {"code": code, "node": node, "detail": detail}


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_merge_outputs(
    report: Mapping[str, Any],
    *,
    report_path: Path,
    markdown_path: Path,
    cache_output_path: Path,
) -> dict[str, Any]:
    """Write an audit report and write the final cache only for PASS."""

    passed = bool(report.get("pass")) and report.get("status") == "PASS"
    snapshot = report.get("service_cache_snapshot")
    cache_written = bool(passed and isinstance(snapshot, Mapping))
    if cache_written:
        _atomic_write(
            cache_output_path,
            json.dumps(snapshot, indent=2, sort_keys=True, allow_nan=False) + "\n",
        )
    disk_report = {
        key: value for key, value in report.items() if key != "service_cache_snapshot"
    }
    disk_report.update(
        {
            "report_path": str(report_path),
            "markdown_path": str(markdown_path),
            "cache_output_path": str(cache_output_path),
            "cache_output_written": cache_written,
            "preexisting_cache_output_not_modified": bool(
                not cache_written and cache_output_path.exists()
            ),
        }
    )
    _atomic_write(
        report_path,
        json.dumps(disk_report, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    _atomic_write(markdown_path, _markdown(disk_report))
    return disk_report


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Critical GPU LCB to service-cache-v2 merge gate",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Complete cache ready: `{bool(report.get('complete_cache_ready'))}`",
        f"- Final cache written: `{bool(report.get('cache_output_written'))}`",
        f"- Base cache: `{report.get('base_cache_path')}`",
        f"- Required protocol: `{report.get('required_measurement_protocol')}`",
        f"- Inserted exact rows: `{report.get('inserted_count')}` / `{report.get('expected_total_inserted_rows')}`",
        f"- Legacy two-key index unchanged: `{bool(report.get('legacy_index_unchanged'))}`",
        "",
        "## Representative hardware gates",
        "",
        "| node | status | protocol PASS | source |",
        "|---|---|:---:|---|",
    ]
    for audit in report.get("source_gate_audits") or []:
        lines.append(
            f"| `{audit.get('node')}` | `{audit.get('status')}` | "
            f"{'yes' if audit.get('protocol_pass') else 'no'} | `{audit.get('path')}` |"
        )
    if report.get("wait_reasons"):
        lines.extend(["", "## Wait reasons", ""])
        for issue in report["wait_reasons"]:
            lines.append(
                f"- `{issue.get('code')}` node={issue.get('node')}: "
                f"{issue.get('detail')}"
            )
    if report.get("validation_errors"):
        lines.extend(["", "## Validation errors", ""])
        for issue in report["validation_errors"]:
            lines.append(
                f"- `{issue.get('code')}` node={issue.get('node')}: "
                f"{issue.get('detail')}"
            )
    lines.extend(["", str(report.get("claim_boundary") or ""), ""])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    defaults = default_gate_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-cache", type=Path, default=DEFAULT_BASE)
    parser.add_argument(
        "--jtl110gpu-gate", type=Path, default=defaults["jtl110gpu"]
    )
    parser.add_argument("--node007-gate", type=Path, default=defaults["node007"])
    parser.add_argument(
        "--jtl311linux-gate", type=Path, default=defaults["jtl311linux"]
    )
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--cache-output", type=Path, default=DEFAULT_CACHE_OUTPUT)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    report = build_critical_gpu_lcb_cache_merge(
        base_cache_path=args.base_cache,
        gate_paths={
            "jtl110gpu": args.jtl110gpu_gate,
            "node007": args.node007_gate,
            "jtl311linux": args.jtl311linux_gate,
        },
    )
    disk_report = write_merge_outputs(
        report,
        report_path=args.report_output,
        markdown_path=args.markdown_output,
        cache_output_path=args.cache_output,
    )
    print(
        json.dumps(
            {
                "status": disk_report["status"],
                "pass": disk_report["pass"],
                "report_output": str(args.report_output),
                "markdown_output": str(args.markdown_output),
                "cache_output": str(args.cache_output),
                "cache_output_written": disk_report["cache_output_written"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if disk_report["pass"]:
        return 0
    return 3 if str(disk_report["status"]).startswith("WAIT") else 2


if __name__ == "__main__":
    raise SystemExit(main())
