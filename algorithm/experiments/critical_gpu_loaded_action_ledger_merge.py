"""Merge certified GPU co-location trajectories without scalarizing actions.

The statewise service cache stores one workload coordinate per record, whereas
the theorem-facing candidate is a vector-valued action.  This gate therefore
emits both representations atomically:

* an action ledger preserving the complete resident/target lower-service
  vector; and
* exact-state coordinate records keyed by physical execution class, load state,
  and direction-sensitive resident mix.

No loaded-state record is allowed to update the legacy ``(workload, profile)``
index.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping

from simulation.service_cache import ProfileRecord, ServiceRateCache

from .critical_gpu_all_hardware_lcb_cache_merge import (
    DEFAULT_CACHE_OUTPUT as ALL_HARDWARE_CACHE,
    DEFAULT_REPORT as ALL_HARDWARE_REPORT,
)
from .critical_gpu_completion_campaign import NODE_SPECS
from .critical_gpu_lcb_cache_merge import _legacy_index_snapshot
from .critical_gpu_loaded_completion_campaign import (
    CAMPAIGN_DATE,
    PROTOCOL,
    SCENARIOS,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
LOADED_SCOPE_NODES = ("jtl110gpu", "node007", "jtl311linux")
DEFAULT_REPORT = ARTIFACT_ROOT / "critical_gpu_loaded_action_ledger_merge_20260809.json"
DEFAULT_MARKDOWN = DEFAULT_REPORT.with_suffix(".md")
DEFAULT_LEDGER = ARTIFACT_ROOT / "critical_gpu_loaded_action_ledger_20260809.json"
DEFAULT_CACHE_OUTPUT = ARTIFACT_ROOT / "service_cache_v2_cpu_gpu_loaded_final_20260809.json"
ETA_SOURCE = (
    "task_native_tqdm_natural_completion_loaded_trajectory_"
    "wave_max_joint_split_conformal_lcb"
)
COMMAND_FINGERPRINT = "critical_gpu_loaded_trajectory_lcb_v1"


def default_gate_paths(*, artifact_root: Path = ARTIFACT_ROOT) -> dict[str, Path]:
    return {
        node: Path(artifact_root)
        / f"critical_gpu_loaded_stochastic_lcb_gate_{node}_{CAMPAIGN_DATE}.json"
        for node in LOADED_SCOPE_NODES
    }


def build_critical_gpu_loaded_action_ledger_merge(
    *,
    base_cache_path: Path = ALL_HARDWARE_CACHE,
    base_report_path: Path = ALL_HARDWARE_REPORT,
    gate_paths: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    supplied = (
        {str(node): Path(path) for node, path in gate_paths.items()}
        if gate_paths is not None
        else default_gate_paths()
    )
    waits: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for label, path in (
        ("base_cache", Path(base_cache_path)),
        ("base_report", Path(base_report_path)),
    ):
        if not path.is_file():
            waits.append(_issue("MISSING_SOURCE", source=label, detail=str(path)))
    unexpected = sorted(set(supplied) - set(LOADED_SCOPE_NODES))
    if unexpected:
        errors.append(
            _issue("UNREGISTERED_LOADED_NODE", detail=repr(unexpected))
        )
    for node in LOADED_SCOPE_NODES:
        path = supplied.get(node)
        if path is None or not path.is_file():
            waits.append(
                _issue(
                    "MISSING_LOADED_GATE",
                    node=node,
                    detail=str(path) if path is not None else "not supplied",
                )
            )

    common = {
        "gate": "critical_gpu_loaded_action_ledger_merge",
        "schema_version": 1,
        "status": "WAIT_SOURCES",
        "pass": False,
        "base_cache_path": str(base_cache_path),
        "base_report_path": str(base_report_path),
        "loaded_scope_nodes": list(LOADED_SCOPE_NODES),
        "required_measurement_protocol": PROTOCOL,
        "expected_action_count": len(LOADED_SCOPE_NODES) * len(SCENARIOS),
        "expected_coordinate_record_count": (
            2 * len(LOADED_SCOPE_NODES) * len(SCENARIOS)
        ),
        "action_count": 0,
        "coordinate_record_count": 0,
        "legacy_index_unchanged": False,
        "registered_loaded_action_ledger_ready": False,
        "unified_cache_ready": False,
        "wait_reasons": waits,
        "validation_errors": errors,
        "claim_boundary": (
            "The ledger certifies only the four registered, direction-sensitive "
            "two-workload trajectories on the representative 3080Ti host "
            "jtl110gpu, node007, and jtl311linux. The homogeneous jtl110gpu2 "
            "host remains an equivalence audit and is not counted as an "
            "independent loaded-state sample. Each action "
            "retains its complete two-coordinate lower-service vector. Exact-state "
            "cache rows are coordinate views, not independent-action claims, and "
            "cannot update the legacy workload/profile index. No claim is made for "
            "unmeasured mixtures, hardware, or future workloads."
        ),
    }
    if errors:
        return {**common, "status": "FAIL_SOURCE_SET"}
    if waits:
        return common

    try:
        cache_bytes = Path(base_cache_path).read_bytes()
        cache = ServiceRateCache.from_snapshot(json.loads(cache_bytes.decode("utf-8")))
        report_bytes = Path(base_report_path).read_bytes()
        base_report = json.loads(report_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        errors.append(_issue("INVALID_BASE", detail=f"{type(exc).__name__}: {exc}"))
        return {**common, "status": "FAIL_BASE", "validation_errors": errors}

    if not (
        base_report.get("gate") == "critical_gpu_all_hardware_lcb_cache_merge"
        and base_report.get("status") == "PASS"
        and base_report.get("pass") is True
        and base_report.get("all_four_gpu_nodes_ready") is True
        and base_report.get("legacy_index_unchanged") is True
    ):
        errors.append(_issue("BASE_GATE_NOT_PASSING"))
        return {**common, "status": "FAIL_BASE", "validation_errors": errors}
    expected_cache_hash = str(base_report.get("cache_output_sha256") or "")
    actual_cache_hash = hashlib.sha256(cache_bytes).hexdigest()
    if expected_cache_hash and expected_cache_hash != actual_cache_hash:
        errors.append(_issue("BASE_CACHE_HASH_MISMATCH"))
        return {**common, "status": "FAIL_BASE", "validation_errors": errors}

    validated: dict[str, tuple[list[Mapping[str, Any]], Path, bytes]] = {}
    for node in LOADED_SCOPE_NODES:
        source = supplied[node]
        try:
            raw = source.read_bytes()
            payload = json.loads(raw.decode("utf-8"))
            rows = _validated_loaded_rows(payload, node=node, path=source)
            validated[node] = (rows, source, raw)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            errors.append(
                _issue(
                    "LOADED_GATE_INVALID",
                    node=node,
                    detail=f"{type(exc).__name__}: {exc}",
                )
            )
    if errors:
        return {
            **common,
            "status": "FAIL_LOADED_GATES",
            "validation_errors": errors,
        }

    legacy_before = _legacy_index_snapshot(cache)
    actions: list[dict[str, Any]] = []
    coordinates: list[dict[str, Any]] = []
    exact_keys: set[tuple[Any, ...]] = set()
    try:
        for node in LOADED_SCOPE_NODES:
            rows, source, _ = validated[node]
            effective_bucket = _effective_node_bucket(node)
            for row in rows:
                action = _action_row(
                    row,
                    node=node,
                    node_bucket=effective_bucket,
                    source=source,
                )
                actions.append(action)
                for role in ("resident", "target"):
                    record = _coordinate_record(
                        row,
                        role=role,
                        node_bucket=effective_bucket,
                        source=source,
                    )
                    key = (
                        record.workload_key,
                        record.workload_env,
                        record.node_bucket,
                        record.resource_state,
                        record.allocation_workers,
                        record.colocation_count,
                        record.resident_mix,
                    )
                    if key in exact_keys:
                        raise ValueError(f"duplicate loaded coordinate {key!r}")
                    exact_keys.add(key)
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
                        raise ValueError(f"exact lookup failed for {key!r}")
                    coordinates.append(
                        {
                            "action_id": action["action_id"],
                            "coordinate_role": role,
                            **record.snapshot(),
                        }
                    )
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(_issue("LOADED_INSERT_FAILED", detail=f"{type(exc).__name__}: {exc}"))
        return {
            **common,
            "status": "FAIL_INSERTION",
            "validation_errors": errors,
        }

    if len(actions) != common["expected_action_count"]:
        errors.append(_issue("ACTION_COUNT_MISMATCH", detail=str(len(actions))))
    if len(coordinates) != common["expected_coordinate_record_count"]:
        errors.append(
            _issue("COORDINATE_COUNT_MISMATCH", detail=str(len(coordinates)))
        )
    legacy_after = _legacy_index_snapshot(cache)
    if legacy_after != legacy_before:
        errors.append(_issue("LEGACY_INDEX_CHANGED"))
    if errors:
        return {
            **common,
            "status": "FAIL_FINAL_AUDIT",
            "validation_errors": errors,
        }

    ledger = {
        "schema_version": 1,
        "ledger": "critical_gpu_loaded_action_ledger",
        "measurement_protocol": PROTOCOL,
        "action_count": len(actions),
        "actions": sorted(actions, key=lambda row: row["action_id"]),
        "claim_boundary": common["claim_boundary"],
    }
    return {
        **common,
        "status": "PASS",
        "pass": True,
        "base_cache_sha256": actual_cache_hash,
        "base_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "source_gate_sha256": {
            node: hashlib.sha256(validated[node][2]).hexdigest()
            for node in LOADED_SCOPE_NODES
        },
        "action_count": len(actions),
        "coordinate_record_count": len(coordinates),
        "actions": ledger["actions"],
        "coordinate_records": coordinates,
        "legacy_index_unchanged": True,
        "registered_loaded_action_ledger_ready": True,
        "unified_cache_ready": True,
        "action_ledger": ledger,
        "service_cache_snapshot": cache.snapshot(),
    }


def _validated_loaded_rows(
    payload: Mapping[str, Any], *, node: str, path: Path
) -> list[Mapping[str, Any]]:
    spec = NODE_SPECS[node]
    required = {
        "gate": "critical_gpu_loaded_stochastic_lcb_gate",
        "schema_version": 1,
        "status": "PASS",
        "pass": True,
        "certificate_ready": True,
        "node": node,
        "node_bucket": spec.node_bucket,
        "hardware_class": spec.hardware_class,
        "resource_state": "mixed_colocation",
        "measurement_protocol": PROTOCOL,
        "same_measurement_code_all_waves": True,
        "all_measurements_ready": True,
        "expected_scenario_count": len(SCENARIOS),
        "expected_wave_count": 13,
    }
    for field, value in required.items():
        if payload.get(field) != value:
            raise ValueError(
                f"{path}: {field} expected {value!r}, got {payload.get(field)!r}"
            )
    if payload.get("wait_reasons") or payload.get("validation_errors"):
        raise ValueError(f"{path}: PASS gate contains blockers")
    certificate = payload.get("certificate") or {}
    for field in ("constructed", "finite_sample_rank_ready", "all_holdout_bounds_valid"):
        if certificate.get(field) is not True:
            raise ValueError(f"{path}: certificate.{field} is not true")
    margin = _finite(certificate.get("simultaneous_ratio_margin"), "margin")
    if margin < 0.0:
        raise ValueError(f"{path}: negative conformal margin")
    rows = certificate.get("rows") or []
    if not isinstance(rows, list):
        raise ValueError(f"{path}: certificate rows must be a list")
    expected = {row.scenario_id: row for row in SCENARIOS}
    observed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        scenario_id = str(row.get("scenario_id") or "")
        if scenario_id in observed:
            raise ValueError(f"{path}: duplicate scenario {scenario_id!r}")
        observed[scenario_id] = row
    if set(observed) != set(expected):
        raise ValueError(f"{path}: scenario set mismatch")
    for scenario_id, row in observed.items():
        scenario = expected[scenario_id]
        expected_scalars = {
            "node": node,
            "node_bucket": spec.node_bucket,
            "hardware_class": spec.hardware_class,
            "resource_state": scenario.resource_state,
            "resident_mix": scenario.resident_mix,
            "resident_workload_key": scenario.resident_workload,
            "target_workload_key": scenario.target_workload,
            "profile": 2,
            "profile_axis": "total_tasks_per_gpu",
            "training_sample_count": 3,
            "calibration_wave_count": 9,
            "holdout_wave": 13,
            "lower_service_bound_kind": (
                "wave_max_joint_target_time_resident_service_split_conformal"
            ),
        }
        for field, value in expected_scalars.items():
            if row.get(field) != value:
                raise ValueError(
                    f"{path}: {scenario_id}.{field} expected {value!r}, "
                    f"got {row.get(field)!r}"
                )
        for field in (
            "target_completion_holdout_covered",
            "resident_service_holdout_covered",
        ):
            if row.get(field) is not True:
                raise ValueError(f"{path}: {scenario_id}.{field} is not true")
        lower = row.get("lower_service_vector") or {}
        expected_keys = {scenario.resident_workload, scenario.target_workload}
        if set(lower) != expected_keys:
            raise ValueError(f"{path}: {scenario_id} lower-service vector mismatch")
        for workload, value in lower.items():
            if _positive(value, f"{scenario_id}:{workload}") <= 0.0:
                raise AssertionError("unreachable")
        target_lower = _positive(
            row.get("target_lower_service_units_per_s"), "target lower service"
        )
        resident_lower = _positive(
            row.get("simultaneous_lower_resident_overlap_service_units_per_s"),
            "resident lower service",
        )
        if not math.isclose(
            target_lower,
            float(lower[scenario.target_workload]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{path}: {scenario_id} target vector mismatch")
        if not math.isclose(
            resident_lower,
            float(lower[scenario.resident_workload]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{path}: {scenario_id} resident vector mismatch")
        target_realized = _positive(
            row.get("target_holdout_realized_service_units_per_s"),
            "target realized service",
        )
        resident_realized = _positive(
            row.get("holdout_actual_resident_overlap_service_units_per_s"),
            "resident realized service",
        )
        if target_lower > target_realized * (1.0 + 1e-12):
            raise ValueError(f"{path}: target LCB exceeds holdout")
        if resident_lower > resident_realized * (1.0 + 1e-12):
            raise ValueError(f"{path}: resident LCB exceeds holdout")
        operational = row.get("operational_pre_holdout_model") or {}
        if int(operational.get("sample_count") or 0) != 12:
            raise ValueError(f"{path}: operational sample count is not 12")
        phase = operational.get("target_phase") or {}
        predicted = (
            _nonnegative(phase.get("startup_overhead_s"), "startup")
            + _positive(phase.get("completion_unit_s"), "unit time")
            * _positive(row.get("target_canonical_total_units"), "target units")
            + _nonnegative(phase.get("terminal_overhead_s"), "terminal")
        )
        if not math.isclose(
            predicted,
            _positive(row.get("operational_target_completion_s"), "target point"),
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError(f"{path}: inconsistent operational target model")
    return [observed[key] for key in sorted(observed)]


def _effective_node_bucket(node: str) -> str:
    return f"{NODE_SPECS[node].node_bucket}:{node}"


def _action_row(
    row: Mapping[str, Any], *, node: str, node_bucket: str, source: Path
) -> dict[str, Any]:
    scenario_id = str(row["scenario_id"])
    return {
        "action_id": f"gpu_loaded:{node}:{scenario_id}",
        "action_kind": "directional_gpu_colocation_trajectory",
        "physical_node": node,
        "node_bucket": node_bucket,
        "hardware_class": str(row["hardware_class"]),
        "resource_state": str(row["resource_state"]),
        "resident_mix": str(row["resident_mix"]),
        "profile": 2,
        "profile_axis": "total_tasks_per_gpu",
        "resident_workload_key": str(row["resident_workload_key"]),
        "resident_workload_env": str(row["resident_workload_env"]),
        "target_workload_key": str(row["target_workload_key"]),
        "target_workload_env": str(row["target_workload_env"]),
        "lower_service_vector": {
            str(key): float(value)
            for key, value in (row["lower_service_vector"] or {}).items()
        },
        "bounded_action_penalty_units": 0.0,
        "penalty_scope": (
            "no migration or reconfiguration; measured interference is already "
            "inside the simultaneous lower-service vector"
        ),
        "simultaneous_upper_target_completion_s": float(
            row["simultaneous_upper_target_completion_s"]
        ),
        "target_completion_holdout_covered": True,
        "resident_service_holdout_covered": True,
        "source": str(source),
    }


def _coordinate_record(
    row: Mapping[str, Any], *, role: str, node_bucket: str, source: Path
) -> ProfileRecord:
    if role not in {"resident", "target"}:
        raise ValueError(f"unsupported coordinate role {role!r}")
    workload_key = str(row[f"{role}_workload_key"])
    workload_env = str(row[f"{role}_workload_env"])
    lower = _positive(
        row["lower_service_vector"][workload_key], f"{role} lower service"
    )
    is_target = role == "target"
    total_units = _positive(row[f"{role}_canonical_total_units"], f"{role} units")
    phase = (
        (row.get("operational_pre_holdout_model") or {}).get("target_phase") or {}
        if is_target
        else {}
    )
    completion_point = (
        _positive(row.get("operational_target_completion_s"), "target completion")
        if is_target
        else 0.0
    )
    return ProfileRecord(
        workload_key=workload_key,
        command_fingerprint=COMMAND_FINGERPRINT,
        resource_kind="hybrid" if workload_key.startswith("hybrid_") else "gpu",
        node_bucket=node_bucket,
        profile=2,
        unit=str(row[f"{role}_service_unit"]),
        total_units=total_units,
        aggregate_rate=lower,
        per_task_rates=(lower,),
        source=f"{source};coordinate_role={role}",
        capacity_boundary=False,
        gpu_count_observed=1,
        workload_env=workload_env,
        resource_state=str(row["resource_state"]),
        resident_mix=str(row["resident_mix"]),
        eta_source=ETA_SOURCE,
        stable_rate_ready=True,
        hardware_class=str(row["hardware_class"]),
        completion_model_ready=is_target,
        completion_model_sample_count=12 if is_target else 0,
        startup_overhead_s=(
            _nonnegative(phase.get("startup_overhead_s"), "startup")
            if is_target
            else 0.0
        ),
        completion_unit_s=(
            _positive(phase.get("completion_unit_s"), "unit time")
            if is_target
            else 0.0
        ),
        finalization_overhead_s=(
            _nonnegative(phase.get("terminal_overhead_s"), "terminal")
            if is_target
            else 0.0
        ),
        completion_total_wall_s=completion_point,
        completion_group_total_units=total_units if is_target else 0.0,
        completion_model_relative_error=(
            _nonnegative(row.get("target_point_relative_error"), "target error")
            if is_target
            else 0.0
        ),
        allocation_workers=1,
        colocation_count=2,
    )


def write_outputs(
    report: Mapping[str, Any], *, report_path: Path, markdown_path: Path,
    ledger_path: Path, cache_path: Path
) -> dict[str, Any]:
    disk = dict(report)
    snapshot = disk.pop("service_cache_snapshot", None)
    ledger = disk.pop("action_ledger", None)
    cache_written = False
    ledger_written = False
    if report.get("pass") is True:
        if snapshot is None or ledger is None:
            raise ValueError("PASS merge is missing cache or ledger payload")
        cache_text = json.dumps(snapshot, indent=2, sort_keys=True) + "\n"
        ledger_text = json.dumps(ledger, indent=2, sort_keys=True) + "\n"
        _atomic_write(cache_path, cache_text)
        _atomic_write(ledger_path, ledger_text)
        disk["cache_output_path"] = str(cache_path)
        disk["cache_output_sha256"] = hashlib.sha256(cache_text.encode()).hexdigest()
        disk["ledger_output_path"] = str(ledger_path)
        disk["ledger_output_sha256"] = hashlib.sha256(ledger_text.encode()).hexdigest()
        cache_written = ledger_written = True
    disk["cache_output_written"] = cache_written
    disk["ledger_output_written"] = ledger_written
    _atomic_write(report_path, json.dumps(disk, indent=2, sort_keys=True) + "\n")
    _atomic_write(markdown_path, markdown_report(disk))
    return disk


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Critical GPU Loaded Action Ledger Merge",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Actions: `{report.get('action_count')}` / `{report.get('expected_action_count')}`",
        f"- Coordinate rows: `{report.get('coordinate_record_count')}` / `{report.get('expected_coordinate_record_count')}`",
        f"- Legacy index unchanged: `{str(bool(report.get('legacy_index_unchanged'))).lower()}`",
        "",
        "| Action | Node | Resident -> target | Lower-service vector |",
        "|---|---|---|---|",
    ]
    for row in report.get("actions") or []:
        lines.append(
            "| `{}` | `{}` | `{}` -> `{}` | `{}` |".format(
                row["action_id"],
                row["physical_node"],
                row["resident_workload_key"],
                row["target_workload_key"],
                json.dumps(row["lower_service_vector"], sort_keys=True),
            )
        )
    lines.extend(["", str(report.get("claim_boundary") or ""), ""])
    return "\n".join(lines)


def _positive(value: Any, label: str) -> float:
    result = _finite(value, label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _nonnegative(value: Any, label: str) -> float:
    result = _finite(value, label)
    if result < 0.0:
        raise ValueError(f"{label} must be nonnegative")
    return result


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _issue(
    code: str, *, node: str | None = None, source: str | None = None,
    detail: str = ""
) -> dict[str, Any]:
    return {"code": code, "node": node, "source": source, "detail": detail}


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-cache", type=Path, default=ALL_HARDWARE_CACHE)
    parser.add_argument("--base-report", type=Path, default=ALL_HARDWARE_REPORT)
    for node in LOADED_SCOPE_NODES:
        parser.add_argument(
            f"--{node}-gate",
            type=Path,
            default=default_gate_paths()[node],
        )
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--ledger-output", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--cache-output", type=Path, default=DEFAULT_CACHE_OUTPUT)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    report = build_critical_gpu_loaded_action_ledger_merge(
        base_cache_path=args.base_cache,
        base_report_path=args.base_report,
        gate_paths={node: getattr(args, f"{node}_gate") for node in LOADED_SCOPE_NODES},
    )
    disk = write_outputs(
        report,
        report_path=args.report_output,
        markdown_path=args.markdown_output,
        ledger_path=args.ledger_output,
        cache_path=args.cache_output,
    )
    print(
        json.dumps(
            {
                "status": disk["status"],
                "pass": disk["pass"],
                "cache_output_written": disk["cache_output_written"],
                "ledger_output_written": disk["ledger_output_written"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if disk["pass"] else (3 if str(disk["status"]).startswith("WAIT") else 2)


if __name__ == "__main__":
    raise SystemExit(main())
