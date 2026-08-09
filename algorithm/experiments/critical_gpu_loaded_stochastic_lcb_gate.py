"""Simultaneous lower-service certificate for measured GPU co-location actions.

Each campaign wave contains the four finite, direction-sensitive mixed actions
registered by :mod:`critical_gpu_loaded_completion_campaign`.  Training waves
freeze a phase-aware target completion model and a resident overlap-rate model.
Calibration waves contribute one maximum ratio score over both functionals and
all actions.  The untouched holdout must satisfy every target-time upper bound
and every resident-service lower bound.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

from .critical_gpu_completion_campaign import (
    GPU_IDLE_MEMORY_LIMIT_MB,
    GPU_IDLE_UTIL_LIMIT_PCT,
    NODE_SPECS,
    WORKLOAD_SPECS,
)
from .critical_gpu_loaded_completion_campaign import (
    ALL_WAVES,
    ARTIFACT_ROOT,
    CALIBRATION_WAVES,
    CAMPAIGN_DATE,
    CAMPAIGN_PREFIX,
    HOLDOUT_WAVE,
    PROTOCOL,
    SCENARIOS,
    TRAINING_WAVES,
    _split_role,
    measurement_code_manifest,
    _resident_total_units,
)


EXPECTED_WAVES = tuple(ALL_WAVES)
RESOURCE_STATE = "mixed_colocation"
MIS_COVERAGE_ALPHA = 0.10
REPO_ROOT = Path(__file__).resolve().parents[2]
MAX_NORMALIZED_LOAD1 = 0.50
MIN_AVAILABLE_MEMORY_FRACTION = 0.10
MAX_UNAPPROVED_USER_CPU_FRACTION = 0.05
# Conservative upper bound on undeclared CPU-time as a fraction of the full
# host capacity over the target interval.  Snapshot spikes remain visible in
# the artifact; only isolated, exposure-negligible bursts may pass this gate.
MAX_UNAPPROVED_USER_CPU_EXPOSURE_FRACTION = 0.001
HOST_MONITOR_ALLOWED_COMMANDS = frozenset(
    {
        "awk",
        "bash",
        "date",
        "grep",
        "nvidia-smi",
        "ps",
        "sed",
        "sleep",
        "ssh",
        "sshd",
        "timeout",
        "tr",
        "wc",
    }
)


def discover_campaign_paths(
    *, node: str, artifact_root: Path = ARTIFACT_ROOT
) -> dict[int, Path]:
    paths: dict[int, Path] = {}
    code_prefix = measurement_code_manifest()["sha256"][:12]
    for wave in EXPECTED_WAVES:
        canonical = Path(artifact_root) / (
            f"{CAMPAIGN_PREFIX}_{node}_r{wave:02d}_{CAMPAIGN_DATE}_code"
            f"{code_prefix}.json"
        )
        if canonical.is_file():
            paths[wave] = canonical
    return paths


def build_critical_gpu_loaded_stochastic_lcb_gate(
    *,
    node: str,
    campaign_paths: Mapping[int, Path] | None = None,
    artifact_root: Path = ARTIFACT_ROOT,
    miscoverage_alpha: float = MIS_COVERAGE_ALPHA,
) -> dict[str, Any]:
    if node not in NODE_SPECS:
        raise ValueError(f"unsupported node {node!r}")
    alpha = float(miscoverage_alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("miscoverage_alpha must lie in (0, 1)")
    paths = (
        {int(wave): Path(path) for wave, path in campaign_paths.items()}
        if campaign_paths is not None
        else discover_campaign_paths(node=node, artifact_root=artifact_root)
    )
    errors: list[dict[str, Any]] = []
    waits: list[dict[str, Any]] = []
    unexpected = sorted(set(paths) - set(EXPECTED_WAVES))
    if unexpected:
        errors.append(_issue("UNREGISTERED_WAVES", detail=repr(unexpected)))

    expected = _expected_scenarios(node)
    observations: dict[tuple[int, str], dict[str, Any]] = {}
    sources = []
    code_hashes = set()
    wave_audits = []
    for wave in EXPECTED_WAVES:
        path = paths.get(wave)
        if path is None or not path.is_file():
            waits.append(_issue("MISSING_WAVE", wave=wave, detail=str(path)))
            continue
        try:
            raw = path.read_bytes()
            payload = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append(
                _issue(
                    "UNREADABLE_WAVE",
                    wave=wave,
                    detail=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        if not isinstance(payload, dict):
            errors.append(_issue("INVALID_WAVE_ROOT", wave=wave))
            continue
        code_hash = str(
            (payload.get("measurement_code_manifest") or {}).get("sha256") or ""
        )
        code_hashes.add(code_hash)
        sources.append(
            {
                "wave": wave,
                "path": str(path),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "measurement_code_sha256": code_hash,
                "status": payload.get("status"),
            }
        )
        audit, wave_waits, wave_errors, rows = _audit_wave(
            node=node,
            wave=wave,
            payload=payload,
            expected=expected,
        )
        wave_audits.append(audit)
        waits.extend(wave_waits)
        errors.extend(wave_errors)
        observations.update(
            {(wave, scenario_id): row for scenario_id, row in rows.items()}
        )

    if code_hashes and ("" in code_hashes or len(code_hashes) != 1):
        errors.append(
            _issue(
                "MEASUREMENT_CODE_IDENTITY_MISMATCH",
                detail=f"observed={sorted(code_hashes)!r}",
            )
        )
    expected_count = len(EXPECTED_WAVES) * len(expected)
    measurements_ready = bool(
        not waits and not errors and len(observations) == expected_count
    )
    certificate: dict[str, Any] = {
        "constructed": False,
        "finite_sample_rank_ready": False,
        "all_holdout_bounds_valid": False,
        "rows": [],
    }
    if measurements_ready:
        certificate = _build_certificate(
            observations=observations,
            expected=expected,
            alpha=alpha,
        )
        if not certificate["finite_sample_rank_ready"]:
            errors.append(_issue("CONFORMAL_RANK_UNAVAILABLE"))
        elif not certificate["all_holdout_bounds_valid"]:
            errors.append(_issue("HOLDOUT_NOT_COVERED"))

    if errors:
        status = "FAIL_VALIDATION"
    elif waits:
        status = "WAIT_MISSING_OR_INCOMPLETE_WAVES"
    elif certificate["constructed"] and certificate["all_holdout_bounds_valid"]:
        status = "PASS"
    else:
        status = "FAIL_CERTIFICATE"
    return {
        "gate": "critical_gpu_loaded_stochastic_lcb_gate",
        "schema_version": 1,
        "status": status,
        "pass": status == "PASS",
        "certificate_ready": status == "PASS",
        "node": node,
        "node_bucket": NODE_SPECS[node].node_bucket,
        "hardware_class": NODE_SPECS[node].hardware_class,
        "resource_state": RESOURCE_STATE,
        "measurement_protocol": PROTOCOL,
        "expected_scenario_count": len(expected),
        "expected_wave_count": len(EXPECTED_WAVES),
        "expected_observation_count": expected_count,
        "ready_observation_count": len(observations),
        "training_waves": list(TRAINING_WAVES),
        "calibration_waves": list(CALIBRATION_WAVES),
        "holdout_wave": HOLDOUT_WAVE,
        "miscoverage_alpha": alpha,
        "nominal_simultaneous_coverage": 1.0 - alpha,
        "same_measurement_code_all_waves": bool(
            len(code_hashes) == 1 and "" not in code_hashes
        ),
        "measurement_code_sha256": (
            next(iter(code_hashes))
            if len(code_hashes) == 1 and "" not in code_hashes
            else None
        ),
        "all_measurements_ready": measurements_ready,
        "expected_scenarios": [expected[key] for key in sorted(expected)],
        "source_artifacts": sources,
        "wave_audits": wave_audits,
        "wait_reasons": waits,
        "validation_errors": errors,
        "certificate": certificate,
        "claim_boundary": (
            "This is a hardware-local certificate for four registered, "
            "direction-sensitive two-workload GPU co-location trajectories. "
            "Target completion includes initialization, the canonical outer-loop "
            "work, checkpoints/final save, and natural exit. Resident service is "
            "the counter increment strictly inside the target start/end markers "
            "divided by the full overlap duration. One wave-maximum split-"
            "conformal margin jointly covers both functionals for all four "
            "actions. Every row also hash-verifies the pre-launch nvidia-smi "
            "snapshot and rejects undeclared load on any registered GPU, so "
            "a hash-bound one-second sidecar covers the complete target interval, "
            "bounds aggregate host load and available memory, and rejects high-CPU "
            "same-user processes outside the two controlled process groups. Thus "
            "same-GPU co-location rows do not silently absorb undeclared host/PCIe "
            "contention. It neither populates the legacy "
            "workload/profile index nor "
            "extrapolates to unmeasured mixtures, hardware, or future workloads."
        ),
    }


def _expected_scenarios(node: str) -> dict[str, dict[str, Any]]:
    workload = {row.workload_key: row for row in WORKLOAD_SPECS}
    result = {}
    for scenario in SCENARIOS:
        resident = workload[scenario.resident_workload]
        target = workload[scenario.target_workload]
        result[scenario.scenario_id] = {
            "scenario_id": scenario.scenario_id,
            "node": node,
            "node_bucket": NODE_SPECS[node].node_bucket,
            "hardware_class": NODE_SPECS[node].hardware_class,
            "resource_state": scenario.resource_state,
            "resident_mix": scenario.resident_mix,
            "resident_workload_key": resident.workload_key,
            "resident_workload_env": resident.workload_env,
            "target_workload_key": target.workload_key,
            "target_workload_env": target.workload_env,
            "resident_measurement_total_units": _resident_total_units(
                node, scenario
            ),
            "target_measurement_total_units": int(scenario.target_total_units),
            "resident_canonical_total_units": int(resident.max_iters),
            "target_canonical_total_units": int(target.max_iters),
            "resident_service_unit": resident.unit,
            "target_service_unit": target.unit,
            "profile": 2,
            "profile_axis": "total_tasks_per_gpu",
        }
    return result


def _audit_wave(
    *,
    node: str,
    wave: int,
    payload: Mapping[str, Any],
    expected: Mapping[str, Mapping[str, Any]],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    waits: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    observations: dict[str, dict[str, Any]] = {}
    scalar = {
        "gate": "critical_gpu_loaded_completion_campaign",
        "schema_version": 1,
        "protocol": PROTOCOL,
        "node": node,
        "node_bucket": NODE_SPECS[node].node_bucket,
        "hardware_class": NODE_SPECS[node].hardware_class,
        "wave": wave,
        "split_role": _split_role(wave),
        "allow_launch": True,
        "legacy_scheduler_limits_bypassed": True,
        "natural_completion_required": True,
        "task_native_progress_required": True,
        "full_resident_target_overlap_required": True,
        "terminate_on_stable": False,
        "algorithm_path": "controlled_theorem_measurement_direct",
        "ordinary_running_tasks_touched": False,
    }
    for field, value in scalar.items():
        if payload.get(field) != value:
            errors.append(
                _issue(
                    "CAMPAIGN_CONTRACT_MISMATCH",
                    wave=wave,
                    detail=f"{field}: expected {value!r}, got {payload.get(field)!r}",
                )
            )
    if (payload.get("selection") or {}).get("filtered"):
        errors.append(_issue("FILTERED_WAVE_FORBIDDEN", wave=wave))
    if (payload.get("pre_registered_split") or {}) != {
        "training": list(TRAINING_WAVES),
        "calibration": list(CALIBRATION_WAVES),
        "holdout": HOLDOUT_WAVE,
    }:
        errors.append(_issue("SPLIT_CONTRACT_MISMATCH", wave=wave))
    initial = payload.get("measurement_code_manifest") or {}
    final = payload.get("final_measurement_code_manifest") or {}
    if not (
        payload.get("measurement_code_unchanged") is True
        and initial.get("sha256")
        and initial.get("sha256") == final.get("sha256")
    ):
        errors.append(_issue("MEASUREMENT_CODE_IDENTITY_INVALID", wave=wave))

    rows = payload.get("rows") or []
    indexed = {}
    for row in rows:
        scenario_id = str(row.get("scenario_id") or "")
        if scenario_id in indexed:
            errors.append(
                _issue("DUPLICATE_SCENARIO", wave=wave, scenario=scenario_id)
            )
        indexed[scenario_id] = row
    missing = sorted(set(expected) - set(indexed))
    extra = sorted(set(indexed) - set(expected))
    if missing:
        waits.append(_issue("MISSING_SCENARIOS", wave=wave, detail=repr(missing)))
    if extra:
        errors.append(_issue("UNEXPECTED_SCENARIOS", wave=wave, detail=repr(extra)))
    campaign_pass = payload.get("status") == "PASS" and payload.get("pass") is True
    if not campaign_pass:
        issue = _issue(
            "CAMPAIGN_NOT_PASSING",
            wave=wave,
            detail=f"status={payload.get('status')!r}, pass={payload.get('pass')!r}",
        )
        (waits if missing or str(payload.get("status") or "").startswith("WAIT") else errors).append(issue)

    for scenario_id in sorted(set(expected) & set(indexed)):
        row_errors, observation = _audit_row(
            row=indexed[scenario_id], expected=expected[scenario_id], wave=wave
        )
        errors.extend(row_errors)
        if not row_errors and observation is not None:
            observations[scenario_id] = observation
    return (
        {
            "wave": wave,
            "split_role": _split_role(wave),
            "campaign_pass": campaign_pass,
            "observed_scenario_count": len(indexed),
            "ready_scenario_count": len(observations),
            "missing_scenarios": missing,
            "unexpected_scenarios": extra,
            "wait_count": len(waits),
            "validation_error_count": len(errors),
        },
        waits,
        errors,
        observations,
    )


def _audit_row(
    *, row: Mapping[str, Any], expected: Mapping[str, Any], wave: int
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    errors: list[dict[str, Any]] = []
    scenario_id = str(expected["scenario_id"])

    def require(condition: bool, code: str, detail: str = "") -> None:
        if not condition:
            errors.append(
                _issue(code, wave=wave, scenario=scenario_id, detail=detail)
            )

    for field in (
        "scenario_id",
        "node",
        "node_bucket",
        "hardware_class",
        "resource_state",
        "resident_mix",
        "resident_workload_key",
        "resident_workload_env",
        "target_workload_key",
        "target_workload_env",
    ):
        require(
            row.get(field) == expected[field],
            "ROW_METADATA_MISMATCH",
            f"{field}: expected {expected[field]!r}, got {row.get(field)!r}",
        )
    require(int(row.get("wave") or -1) == wave, "ROW_WAVE_MISMATCH")
    require(row.get("split_role") == _split_role(wave), "ROW_SPLIT_MISMATCH")
    require(row.get("ready") is True, "ROW_NOT_READY")
    require(not bool(row.get("capacity_boundary")), "CAPACITY_BOUNDARY")
    require(row.get("eta_source") == "task_native_tqdm_natural_completion_loaded_trajectory", "ETA_SOURCE_INVALID")
    require(row.get("legacy_scheduler_limits_bypassed") is True, "LEGACY_LIMITS_NOT_BYPASSED")
    require(row.get("ordinary_running_tasks_touched") is False, "NO_TOUCH_VIOLATION")
    prelaunch_gpu = _prelaunch_assigned_gpu_audit(row=row, node=str(expected["node"]))
    require(
        prelaunch_gpu.get("audit_ready") is True,
        str(prelaunch_gpu.get("error_code") or "PRELAUNCH_GPU_AUDIT_INVALID"),
        str(prelaunch_gpu.get("detail") or ""),
    )
    require(
        prelaunch_gpu.get("assigned_gpu_idle") is True,
        "ASSIGNED_GPU_NOT_IDLE_AT_RESIDENT_START",
        (
            f"gpu={prelaunch_gpu.get('gpu')}, "
            f"used_mb={prelaunch_gpu.get('used_mb')}, "
            f"util_pct={prelaunch_gpu.get('util_pct')}, "
            f"limits=({GPU_IDLE_MEMORY_LIMIT_MB} MB, {GPU_IDLE_UTIL_LIMIT_PCT}%)"
        ),
    )
    require(
        prelaunch_gpu.get("all_registered_gpus_idle") is True,
        "UNDECLARED_NODE_GPU_LOAD_AT_RESIDENT_START",
        (
            f"nonidle_registered_gpus={prelaunch_gpu.get('nonidle_registered_gpus')!r}, "
            f"limits=({GPU_IDLE_MEMORY_LIMIT_MB} MB, {GPU_IDLE_UTIL_LIMIT_PCT}%)"
        ),
    )
    continuous_gpu = _continuous_other_gpu_audit(
        row=row,
        node=str(expected["node"]),
    )
    require(
        continuous_gpu.get("audit_ready") is True,
        str(continuous_gpu.get("error_code") or "GPU_MONITOR_AUDIT_INVALID"),
        str(continuous_gpu.get("detail") or ""),
    )
    require(
        continuous_gpu.get("target_interval_covered") is True,
        "GPU_MONITOR_TARGET_INTERVAL_NOT_COVERED",
        (
            f"first_sample_ns={continuous_gpu.get('first_sample_ns')}, "
            f"last_sample_ns={continuous_gpu.get('last_sample_ns')}, "
            f"target=({continuous_gpu.get('target_start_ns')}, "
            f"{continuous_gpu.get('target_end_ns')})"
        ),
    )
    require(
        continuous_gpu.get("other_registered_gpus_idle") is True,
        "UNDECLARED_CROSS_GPU_LOAD_DURING_TARGET",
        f"violations={continuous_gpu.get('violations')!r}",
    )
    require(
        continuous_gpu.get("controlled_process_groups_observed") is True,
        "CONTROLLED_GPU_PROCESS_GROUP_NOT_OBSERVED",
        f"observed={continuous_gpu.get('observed_process_group_ids')!r}",
    )
    require(
        continuous_gpu.get("no_unapproved_compute_processes") is True,
        "UNDECLARED_GPU_COMPUTE_PROCESS_DURING_TARGET",
        f"unapproved={continuous_gpu.get('unapproved_compute_processes')!r}",
    )
    require(
        continuous_gpu.get("host_interval_covered") is True,
        "HOST_MONITOR_TARGET_INTERVAL_NOT_COVERED",
        f"missing={continuous_gpu.get('missing_host_sample_ns')!r}",
    )
    require(
        continuous_gpu.get("host_process_interval_sampled") is True,
        "HOST_PROCESS_MONITOR_TARGET_INTERVAL_NOT_COVERED",
        f"samples={continuous_gpu.get('host_process_sample_count')!r}",
    )
    require(
        continuous_gpu.get("host_load_within_bound") is True,
        "UNDECLARED_HOST_LOAD_DURING_TARGET",
        f"violations={continuous_gpu.get('host_load_violations')!r}",
    )
    require(
        continuous_gpu.get("host_memory_within_bound") is True,
        "INSUFFICIENT_HOST_MEMORY_DURING_TARGET",
        f"violations={continuous_gpu.get('host_memory_violations')!r}",
    )
    require(
        continuous_gpu.get("unapproved_cpu_exposure_within_bound") is True,
        "UNDECLARED_HOST_PROCESS_DURING_TARGET",
        (
            "violations="
            f"{continuous_gpu.get('unapproved_cpu_processes')!r}, "
            "conservative_exposure="
            f"{continuous_gpu.get('conservative_unapproved_cpu_exposure_fraction')!r}, "
            "limit="
            f"{continuous_gpu.get('max_unapproved_cpu_exposure_fraction')!r}"
        ),
    )
    require(row.get("resident_alive_at_target_start") is True, "OVERLAP_INVALID")
    require(row.get("resident_alive_at_target_end") is True, "OVERLAP_INVALID")
    require(_int_or(row.get("resident_returncode"), -1) == 0, "NATURAL_EXIT_INVALID")
    require(_int_or(row.get("target_returncode"), -1) == 0, "NATURAL_EXIT_INVALID")
    require(
        int(row.get("resident_total_units") or 0)
        == int(expected["resident_measurement_total_units"]),
        "RESIDENT_UNITS_MISMATCH",
    )
    require(
        int(row.get("target_total_units") or 0)
        == int(expected["target_measurement_total_units"]),
        "TARGET_UNITS_MISMATCH",
    )

    resident_model = row.get("resident_completion_model") or {}
    target_model = row.get("target_completion_model") or {}
    overlap = row.get("resident_overlap_service") or {}
    _require_completion_model(
        resident_model,
        expected_units=int(expected["resident_measurement_total_units"]),
        expected_unit=str(expected["resident_service_unit"]),
        require=require,
        prefix="resident",
    )
    _require_completion_model(
        target_model,
        expected_units=int(expected["target_measurement_total_units"]),
        expected_unit=str(expected["target_service_unit"]),
        require=require,
        prefix="target",
    )
    require(overlap.get("ready") is True, "OVERLAP_SERVICE_INVALID")
    require(float(overlap.get("duration_s") or 0.0) > 0.0, "OVERLAP_SERVICE_INVALID")
    require(int(overlap.get("progress_observation_count") or 0) >= 2, "OVERLAP_SERVICE_INVALID")
    require(int(overlap.get("completed_units_lower") or 0) > 0, "OVERLAP_SERVICE_INVALID")
    require(float(overlap.get("conservative_rate_units_per_s") or 0.0) > 0.0, "OVERLAP_SERVICE_INVALID")
    require(overlap.get("unit") == expected["resident_service_unit"], "OVERLAP_SERVICE_UNIT_INVALID")
    if errors:
        return errors, None

    target_phase = _phase(target_model)
    canonical_units = float(expected["target_canonical_total_units"])
    actual_target_canonical = _predict_phase(target_phase, canonical_units)
    return errors, {
        **dict(expected),
        "wave": wave,
        "split_role": _split_role(wave),
        "target_phase": target_phase,
        "target_canonical_completion_s": actual_target_canonical,
        "resident_overlap_service_units_per_s": float(
            overlap["conservative_rate_units_per_s"]
        ),
        "resident_overlap_duration_s": float(overlap["duration_s"]),
        "resident_overlap_progress_observations": int(
            overlap["progress_observation_count"]
        ),
        "prelaunch_assigned_gpu_audit": prelaunch_gpu,
        "continuous_other_gpu_audit": continuous_gpu,
    }


def _prelaunch_assigned_gpu_audit(
    *, row: Mapping[str, Any], node: str
) -> dict[str, Any]:
    """Verify that only the declared resident creates the loaded state.

    The completion campaign records an ``nvidia-smi`` snapshot before launching
    the resident and another after both tasks exit.  A row labelled with only a
    controlled resident is invalid when either the selected physical GPU or a
    different registered GPU was already loaded by an undeclared process in the
    first snapshot.  The node-wide check prevents host/PCIe contention from
    entering a row whose declared state contains only same-GPU co-location.
    """

    path_text = str(row.get("diagnostics_log_path") or "").strip()
    expected_sha256 = str(row.get("diagnostics_sha256") or "").strip().lower()
    try:
        gpu = int(row.get("gpu"))
    except (TypeError, ValueError):
        return {
            "audit_ready": False,
            "assigned_gpu_idle": False,
            "error_code": "PRELAUNCH_GPU_INDEX_INVALID",
            "detail": f"gpu={row.get('gpu')!r}",
        }
    if gpu not in tuple(int(value) for value in NODE_SPECS[node].gpus):
        return {
            "audit_ready": False,
            "assigned_gpu_idle": False,
            "error_code": "PRELAUNCH_GPU_INDEX_INVALID",
            "detail": f"gpu={gpu} is outside registered node GPUs",
        }
    if not path_text or len(expected_sha256) != 64:
        return {
            "audit_ready": False,
            "assigned_gpu_idle": False,
            "error_code": "PRELAUNCH_GPU_DIAGNOSTICS_MISSING",
            "detail": f"path={path_text!r}, sha256={expected_sha256!r}",
        }
    path = Path(path_text)
    try:
        payload = path.read_bytes()
    except OSError as exc:
        return {
            "audit_ready": False,
            "assigned_gpu_idle": False,
            "error_code": "PRELAUNCH_GPU_DIAGNOSTICS_UNREADABLE",
            "detail": f"{path}: {type(exc).__name__}: {exc}",
        }
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if observed_sha256 != expected_sha256:
        return {
            "audit_ready": False,
            "assigned_gpu_idle": False,
            "error_code": "PRELAUNCH_GPU_DIAGNOSTICS_HASH_MISMATCH",
            "detail": (
                f"path={path}, expected={expected_sha256}, "
                f"observed={observed_sha256}"
            ),
        }
    expected_indices = {int(value) for value in NODE_SPECS[node].gpus}
    first_snapshot: dict[int, dict[str, Any]] = {}
    for line in payload.decode("utf-8", errors="replace").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            index = int(parts[0])
            total_mb = int(parts[2])
            used_mb = int(parts[3])
            free_mb = int(parts[4])
            util_pct = int(parts[5])
        except ValueError:
            continue
        if index in expected_indices and index not in first_snapshot:
            first_snapshot[index] = {
                "gpu": index,
                "name": parts[1],
                "total_mb": total_mb,
                "used_mb": used_mb,
                "free_mb": free_mb,
                "util_pct": util_pct,
            }
        if set(first_snapshot) == expected_indices:
            break
    if set(first_snapshot) != expected_indices or gpu not in first_snapshot:
        return {
            "audit_ready": False,
            "assigned_gpu_idle": False,
            "error_code": "PRELAUNCH_GPU_SNAPSHOT_INCOMPLETE",
            "detail": (
                f"expected={sorted(expected_indices)!r}, "
                f"observed={sorted(first_snapshot)!r}"
            ),
        }
    selected = first_snapshot[gpu]
    assigned_idle = bool(
        selected["used_mb"] <= GPU_IDLE_MEMORY_LIMIT_MB
        and selected["util_pct"] <= GPU_IDLE_UTIL_LIMIT_PCT
    )
    nonidle = [
        dict(first_snapshot[index])
        for index in sorted(first_snapshot)
        if not (
            first_snapshot[index]["used_mb"] <= GPU_IDLE_MEMORY_LIMIT_MB
            and first_snapshot[index]["util_pct"] <= GPU_IDLE_UTIL_LIMIT_PCT
        )
    ]
    return {
        "audit_ready": True,
        "assigned_gpu_idle": assigned_idle,
        "all_registered_gpus_idle": not nonidle,
        "nonidle_registered_gpus": nonidle,
        "registered_gpu_snapshot": [
            dict(first_snapshot[index]) for index in sorted(first_snapshot)
        ],
        "gpu": gpu,
        "used_mb": selected["used_mb"],
        "util_pct": selected["util_pct"],
        "idle_memory_limit_mb": GPU_IDLE_MEMORY_LIMIT_MB,
        "idle_util_limit_pct": GPU_IDLE_UTIL_LIMIT_PCT,
        "diagnostics_path": str(path),
        "diagnostics_sha256": observed_sha256,
    }


def _conservative_unapproved_cpu_exposure(
    *,
    host_process_samples: Sequence[Mapping[str, Any]],
    high_snapshot_totals: Sequence[Mapping[str, Any]],
    target_start_ns: int,
    target_end_ns: int,
) -> dict[str, Any]:
    """Upper-bound transient CPU exposure using neighboring process samples.

    A process present in one snapshot is conservatively treated as present from
    the preceding process snapshot through the following one.  Consecutive high
    snapshots form one burst, so overlapping brackets are never double-counted.
    """

    duration_ns = int(target_end_ns) - int(target_start_ns)
    if duration_ns <= 0:
        return {"conservative_exposure_fraction": math.inf, "bursts": []}
    sample_times = sorted(
        {
            int(sample["sample_ns"])
            for sample in host_process_samples
            if target_start_ns <= int(sample["sample_ns"]) <= target_end_ns
        }
    )
    high_by_ns = {
        int(row["sample_ns"]): float(row["total_unapproved_cpu_fraction"])
        for row in high_snapshot_totals
        if target_start_ns <= int(row["sample_ns"]) <= target_end_ns
    }
    if not high_by_ns:
        return {"conservative_exposure_fraction": 0.0, "bursts": []}
    position = {sample_ns: index for index, sample_ns in enumerate(sample_times)}
    high_positions = sorted(
        position[sample_ns]
        for sample_ns in high_by_ns
        if sample_ns in position
    )
    if len(high_positions) != len(high_by_ns):
        return {"conservative_exposure_fraction": math.inf, "bursts": []}

    groups: list[tuple[int, int]] = []
    start = previous = high_positions[0]
    for current in high_positions[1:]:
        if current == previous + 1:
            previous = current
            continue
        groups.append((start, previous))
        start = previous = current
    groups.append((start, previous))

    bursts = []
    total_exposure = 0.0
    for start_index, end_index in groups:
        left_ns = (
            int(target_start_ns)
            if start_index == 0
            else max(int(target_start_ns), sample_times[start_index - 1])
        )
        right_ns = (
            int(target_end_ns)
            if end_index + 1 >= len(sample_times)
            else min(int(target_end_ns), sample_times[end_index + 1])
        )
        peak_fraction = max(
            high_by_ns[sample_times[index]]
            for index in range(start_index, end_index + 1)
        )
        exposure_fraction = peak_fraction * max(right_ns - left_ns, 0) / duration_ns
        total_exposure += exposure_fraction
        bursts.append(
            {
                "first_high_sample_ns": sample_times[start_index],
                "last_high_sample_ns": sample_times[end_index],
                "conservative_start_ns": left_ns,
                "conservative_end_ns": right_ns,
                "conservative_duration_s": max(right_ns - left_ns, 0) / 1e9,
                "peak_host_cpu_fraction": peak_fraction,
                "conservative_exposure_fraction": exposure_fraction,
                "high_snapshot_count": end_index - start_index + 1,
            }
        )
    return {
        "conservative_exposure_fraction": total_exposure,
        "bursts": bursts,
    }


def _continuous_other_gpu_audit(
    *, row: Mapping[str, Any], node: str
) -> dict[str, Any]:
    """Audit undeclared load on non-assigned GPUs throughout the target interval."""

    path_text = str(row.get("gpu_monitor_log_path") or "").strip()
    expected_sha256 = str(row.get("gpu_monitor_sha256") or "").strip().lower()
    try:
        assigned_gpu = int(row.get("gpu"))
        resident_pgid = int(row.get("resident_process_group_id"))
        target_pgid = int(row.get("target_process_group_id"))
        target_start_ns = int(row.get("target_start_ns"))
        target_end_ns = int(row.get("target_end_ns"))
    except (TypeError, ValueError):
        return {
            "audit_ready": False,
            "target_interval_covered": False,
            "other_registered_gpus_idle": False,
            "error_code": "GPU_MONITOR_METADATA_INVALID",
                "detail": (
                    f"gpu={row.get('gpu')!r}, start={row.get('target_start_ns')!r}, "
                    f"end={row.get('target_end_ns')!r}, "
                    f"pgids=({row.get('resident_process_group_id')!r}, "
                    f"{row.get('target_process_group_id')!r})"
                ),
            }
    expected_indices = {int(value) for value in NODE_SPECS[node].gpus}
    if (
        assigned_gpu not in expected_indices
        or resident_pgid <= 0
        or target_pgid <= 0
        or resident_pgid == target_pgid
        or target_start_ns <= 0
        or target_end_ns <= target_start_ns
    ):
        return {
            "audit_ready": False,
            "target_interval_covered": False,
            "other_registered_gpus_idle": False,
            "error_code": "GPU_MONITOR_METADATA_INVALID",
                "detail": (
                    f"gpu={assigned_gpu}, pgids=({resident_pgid}, {target_pgid}), "
                    f"target=({target_start_ns}, {target_end_ns})"
                ),
            }
    if not path_text or len(expected_sha256) != 64:
        return {
            "audit_ready": False,
            "target_interval_covered": False,
            "other_registered_gpus_idle": False,
            "error_code": "GPU_MONITOR_MISSING",
            "detail": f"path={path_text!r}, sha256={expected_sha256!r}",
        }
    path = Path(path_text)
    try:
        payload = path.read_bytes()
    except OSError as exc:
        return {
            "audit_ready": False,
            "target_interval_covered": False,
            "other_registered_gpus_idle": False,
            "error_code": "GPU_MONITOR_UNREADABLE",
            "detail": f"{path}: {type(exc).__name__}: {exc}",
        }
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if observed_sha256 != expected_sha256:
        return {
            "audit_ready": False,
            "target_interval_covered": False,
            "other_registered_gpus_idle": False,
            "error_code": "GPU_MONITOR_HASH_MISMATCH",
            "detail": (
                f"path={path}, expected={expected_sha256}, observed={observed_sha256}"
            ),
        }

    samples: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in payload.decode("utf-8", errors="replace").splitlines():
        marker = re.fullmatch(r"__GPU_SAMPLE_NS__\s+(\d+)", line.strip())
        if marker:
            if current is not None:
                samples.append(current)
            current = {
                "sample_ns": int(marker.group(1)),
                "gpus": {},
                "processes": [],
                "host": None,
                "user_process_sampled": False,
                "user_processes": [],
            }
            continue
        if current is None:
            continue
        host = re.fullmatch(
            r"__HOST_SAMPLE__\s+(\d+)\s+([-+0-9.eE]+)\s+"
            r"([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+(\d+)\s+(\d+)",
            line.strip(),
        )
        if host:
            current["host"] = {
                "nproc": int(host.group(1)),
                "load1": float(host.group(2)),
                "load5": float(host.group(3)),
                "load15": float(host.group(4)),
                "mem_available_kb": int(host.group(5)),
                "mem_total_kb": int(host.group(6)),
            }
            continue
        process_snapshot = re.fullmatch(
            r"__USER_PROC_SNAPSHOT__\s+(\d+)", line.strip()
        )
        if process_snapshot:
            current["user_process_sampled"] = True
            current["user_process_sample_index"] = int(process_snapshot.group(1))
            continue
        user_process = re.fullmatch(
            r"__USER_PROC__\s+(\d+)\s+(-?\d+)\s+([-+0-9.eE]+)\s+"
            r"(\d+)\s+(\S+)",
            line.strip(),
        )
        if user_process:
            current["user_processes"].append(
                {
                    "pid": int(user_process.group(1)),
                    "pgid": int(user_process.group(2)),
                    "cpu_pct": float(user_process.group(3)),
                    "rss_kb": int(user_process.group(4)),
                    "command": user_process.group(5),
                }
            )
            continue
        process = re.fullmatch(
            r"__GPU_PROC__\s+(\d+)\s+(-?\d+)\s+(\S+)\s+(\S+)",
            line.strip(),
        )
        if process:
            current["processes"].append(
                {
                    "pid": int(process.group(1)),
                    "pgid": int(process.group(2)),
                    "gpu_uuid": process.group(3),
                    "used_memory_mb": process.group(4),
                }
            )
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            index = int(parts[0])
            used_mb = int(parts[3])
            util_pct = int(parts[5])
        except ValueError:
            continue
        if index in expected_indices:
            current["gpus"][index] = {
                "gpu": index,
                "used_mb": used_mb,
                "util_pct": util_pct,
            }
    if current is not None:
        samples.append(current)
    incomplete = [
        int(sample["sample_ns"])
        for sample in samples
        if set(sample["gpus"]) != expected_indices
    ]
    if len(samples) < 2 or incomplete:
        return {
            "audit_ready": False,
            "target_interval_covered": False,
            "other_registered_gpus_idle": False,
            "error_code": "GPU_MONITOR_SAMPLES_INVALID",
            "detail": f"samples={len(samples)}, incomplete={incomplete[:10]!r}",
        }

    samples.sort(key=lambda sample: int(sample["sample_ns"]))
    first_sample_ns = int(samples[0]["sample_ns"])
    last_sample_ns = int(samples[-1]["sample_ns"])
    target_interval_covered = bool(
        first_sample_ns <= target_start_ns and last_sample_ns >= target_end_ns
    )
    violations = []
    allowed_pgids = {resident_pgid, target_pgid}
    observed_pgids: set[int] = set()
    unapproved_processes = []
    pid_pgid_candidates: dict[int, set[int]] = {}
    for sample in samples:
        for process in sample["processes"]:
            process_pgid = int(process["pgid"])
            if process_pgid > 0:
                pid_pgid_candidates.setdefault(int(process["pid"]), set()).add(
                    process_pgid
                )
    resolved_transient_pgids = []
    target_samples = [
        sample
        for sample in samples
        if target_start_ns <= int(sample["sample_ns"]) <= target_end_ns
    ]
    missing_host_sample_ns = [
        int(sample["sample_ns"])
        for sample in target_samples
        if not isinstance(sample.get("host"), Mapping)
    ]
    host_interval_covered = bool(target_samples) and not missing_host_sample_ns
    host_process_samples = [
        sample
        for sample in target_samples
        if sample.get("user_process_sampled") is True
    ]
    host_load_violations = []
    host_memory_violations = []
    host_normalized_load1_max = 0.0
    host_available_memory_fraction_min = 1.0
    for sample in target_samples:
        host = sample.get("host")
        if not isinstance(host, Mapping):
            continue
        nproc = int(host.get("nproc") or 0)
        load1 = float(host.get("load1") or 0.0)
        mem_available_kb = int(host.get("mem_available_kb") or 0)
        mem_total_kb = int(host.get("mem_total_kb") or 0)
        normalized_load1 = load1 / nproc if nproc > 0 else math.inf
        available_fraction = (
            mem_available_kb / mem_total_kb if mem_total_kb > 0 else 0.0
        )
        host_normalized_load1_max = max(
            host_normalized_load1_max, normalized_load1
        )
        host_available_memory_fraction_min = min(
            host_available_memory_fraction_min, available_fraction
        )
        if normalized_load1 > MAX_NORMALIZED_LOAD1:
            host_load_violations.append(
                {
                    "sample_ns": int(sample["sample_ns"]),
                    "load1": load1,
                    "nproc": nproc,
                    "normalized_load1": normalized_load1,
                }
            )
        if available_fraction < MIN_AVAILABLE_MEMORY_FRACTION:
            host_memory_violations.append(
                {
                    "sample_ns": int(sample["sample_ns"]),
                    "mem_available_kb": mem_available_kb,
                    "mem_total_kb": mem_total_kb,
                    "available_fraction": available_fraction,
                }
            )
    unapproved_cpu_processes = []
    unapproved_cpu_snapshot_totals = []
    for sample in host_process_samples:
        sample_ns = int(sample["sample_ns"])
        host = sample.get("host") or {}
        nproc = int(host.get("nproc") or 0)
        host_cpu_capacity_pct = 100.0 * nproc
        total_unapproved_cpu_pct = 0.0
        for process in sample["user_processes"]:
            pgid = int(process["pgid"])
            command = str(process["command"])
            if pgid in allowed_pgids or command in HOST_MONITOR_ALLOWED_COMMANDS:
                continue
            cpu_pct = float(process["cpu_pct"])
            total_unapproved_cpu_pct += max(cpu_pct, 0.0)
            host_cpu_fraction = (
                max(cpu_pct, 0.0) / host_cpu_capacity_pct
                if host_cpu_capacity_pct > 0.0
                else math.inf
            )
            if host_cpu_fraction > MAX_UNAPPROVED_USER_CPU_FRACTION:
                unapproved_cpu_processes.append(
                    {
                        "sample_ns": sample_ns,
                        "host_nproc": nproc,
                        "host_cpu_fraction": host_cpu_fraction,
                        **process,
                    }
                )
        total_unapproved_cpu_fraction = (
            total_unapproved_cpu_pct / host_cpu_capacity_pct
            if host_cpu_capacity_pct > 0.0
            else math.inf
        )
        if total_unapproved_cpu_fraction > MAX_UNAPPROVED_USER_CPU_FRACTION:
            unapproved_cpu_snapshot_totals.append(
                {
                    "sample_ns": sample_ns,
                    "host_nproc": nproc,
                    "total_unapproved_cpu_pct": total_unapproved_cpu_pct,
                    "total_unapproved_cpu_fraction": total_unapproved_cpu_fraction,
                }
            )
    cpu_exposure = _conservative_unapproved_cpu_exposure(
        host_process_samples=host_process_samples,
        high_snapshot_totals=unapproved_cpu_snapshot_totals,
        target_start_ns=target_start_ns,
        target_end_ns=target_end_ns,
    )
    maxima = {
        index: {"gpu": index, "max_used_mb": 0, "max_util_pct": 0}
        for index in sorted(expected_indices - {assigned_gpu})
    }
    for sample in samples:
        sample_ns = int(sample["sample_ns"])
        if target_start_ns <= sample_ns <= target_end_ns:
            for process in sample["processes"]:
                pgid = int(process["pgid"])
                if pgid <= 0:
                    candidates = pid_pgid_candidates.get(int(process["pid"]), set())
                    if len(candidates) == 1:
                        resolved_pgid = next(iter(candidates))
                        resolved_transient_pgids.append(
                            {
                                "sample_ns": sample_ns,
                                "pid": int(process["pid"]),
                                "reported_pgid": pgid,
                                "resolved_pgid": resolved_pgid,
                                "resolution_rule": "same_pid_unique_positive_pgid_in_hash_bound_monitor",
                            }
                        )
                        pgid = resolved_pgid
                observed_pgids.add(pgid)
                if pgid not in allowed_pgids:
                    unapproved_processes.append(
                        {"sample_ns": sample_ns, **process, "effective_pgid": pgid}
                    )
        for index, maximum in maxima.items():
            gpu_row = sample["gpus"][index]
            maximum["max_used_mb"] = max(
                int(maximum["max_used_mb"]), int(gpu_row["used_mb"])
            )
            maximum["max_util_pct"] = max(
                int(maximum["max_util_pct"]), int(gpu_row["util_pct"])
            )
            if (
                int(gpu_row["used_mb"]) > GPU_IDLE_MEMORY_LIMIT_MB
                or int(gpu_row["util_pct"]) > GPU_IDLE_UTIL_LIMIT_PCT
            ):
                violations.append({"sample_ns": sample_ns, **gpu_row})
    return {
        "audit_ready": True,
        "target_interval_covered": target_interval_covered,
        "other_registered_gpus_idle": not violations,
        "controlled_process_groups_observed": allowed_pgids <= observed_pgids,
        "no_unapproved_compute_processes": not unapproved_processes,
        "host_interval_covered": host_interval_covered,
        "host_process_interval_sampled": bool(host_process_samples),
        "host_load_within_bound": bool(
            host_interval_covered and not host_load_violations
        ),
        "host_memory_within_bound": bool(
            host_interval_covered and not host_memory_violations
        ),
        "no_unapproved_cpu_processes": bool(
            host_process_samples
            and not unapproved_cpu_processes
            and not unapproved_cpu_snapshot_totals
        ),
        "unapproved_cpu_exposure_within_bound": bool(
            host_process_samples
            and float(cpu_exposure["conservative_exposure_fraction"])
            <= MAX_UNAPPROVED_USER_CPU_EXPOSURE_FRACTION
        ),
        "missing_host_sample_ns": missing_host_sample_ns[:50],
        "host_process_sample_count": len(host_process_samples),
        "host_normalized_load1_max": host_normalized_load1_max,
        "host_available_memory_fraction_min": (
            host_available_memory_fraction_min if target_samples else None
        ),
        "host_load_violations": host_load_violations[:50],
        "host_memory_violations": host_memory_violations[:50],
        "unapproved_cpu_processes": unapproved_cpu_processes[:50],
        "unapproved_cpu_process_count": len(unapproved_cpu_processes),
        "unapproved_cpu_snapshot_totals": unapproved_cpu_snapshot_totals[:50],
        "unapproved_cpu_transient_bursts": cpu_exposure["bursts"][:50],
        "unapproved_cpu_transient_burst_count": len(cpu_exposure["bursts"]),
        "conservative_unapproved_cpu_exposure_fraction": cpu_exposure[
            "conservative_exposure_fraction"
        ],
        "max_normalized_load1": MAX_NORMALIZED_LOAD1,
        "min_available_memory_fraction": MIN_AVAILABLE_MEMORY_FRACTION,
        "max_unapproved_user_cpu_fraction": MAX_UNAPPROVED_USER_CPU_FRACTION,
        "max_unapproved_cpu_exposure_fraction": (
            MAX_UNAPPROVED_USER_CPU_EXPOSURE_FRACTION
        ),
        "allowed_process_group_ids": sorted(allowed_pgids),
        "observed_process_group_ids": sorted(observed_pgids),
        "unapproved_compute_processes": unapproved_processes[:50],
        "unapproved_compute_process_count": len(unapproved_processes),
        "resolved_transient_process_group_ids": resolved_transient_pgids[:50],
        "resolved_transient_process_group_id_count": len(resolved_transient_pgids),
        "assigned_gpu": assigned_gpu,
        "target_start_ns": target_start_ns,
        "target_end_ns": target_end_ns,
        "first_sample_ns": first_sample_ns,
        "last_sample_ns": last_sample_ns,
        "sample_count": len(samples),
        "sample_interval_s": 1,
        "other_gpu_maxima": [maxima[index] for index in sorted(maxima)],
        "violations": violations[:50],
        "violation_count": len(violations),
        "idle_memory_limit_mb": GPU_IDLE_MEMORY_LIMIT_MB,
        "idle_util_limit_pct": GPU_IDLE_UTIL_LIMIT_PCT,
        "gpu_monitor_path": str(path),
        "gpu_monitor_sha256": observed_sha256,
    }


def _require_completion_model(
    model: Mapping[str, Any],
    *,
    expected_units: int,
    expected_unit: str,
    require: Any,
    prefix: str,
) -> None:
    require(model.get("completion_model_ready") is True, "NATURAL_EXIT_INVALID", prefix)
    require(model.get("natural_exit") is True, "NATURAL_EXIT_INVALID", prefix)
    require(model.get("stopped_on_stable") is False, "NATURAL_EXIT_INVALID", prefix)
    require(_int_or(model.get("child_returncode"), -1) == 0, "NATURAL_EXIT_INVALID", prefix)
    require(int(model.get("total_units") or 0) == expected_units, "COMPLETION_UNITS_INVALID", prefix)
    require(model.get("unit") == expected_unit, "COMPLETION_UNIT_INVALID", prefix)
    startup = float(model.get("startup_overhead_s") or 0.0)
    unit_s = float(model.get("completion_unit_s") or 0.0)
    terminal = float(model.get("terminal_overhead_s") or 0.0)
    total = float(model.get("total_wall_s") or 0.0)
    require(startup >= 0.0 and unit_s > 0.0 and terminal >= 0.0, "PHASE_MODEL_INVALID", prefix)
    require(total > 0.0, "PHASE_MODEL_INVALID", prefix)
    require(
        math.isclose(
            startup + expected_units * unit_s + terminal,
            total,
            rel_tol=1e-8,
            abs_tol=1e-8,
        ),
        "PHASE_MODEL_INVALID",
        prefix,
    )


def _build_certificate(
    *,
    observations: Mapping[tuple[int, str], Mapping[str, Any]],
    expected: Mapping[str, Mapping[str, Any]],
    alpha: float,
) -> dict[str, Any]:
    frozen = {}
    operational = {}
    for scenario_id in sorted(expected):
        frozen[scenario_id] = _fit_models(
            [observations[(wave, scenario_id)] for wave in TRAINING_WAVES]
        )
        operational[scenario_id] = _fit_models(
            [
                observations[(wave, scenario_id)]
                for wave in (*TRAINING_WAVES, *CALIBRATION_WAVES)
            ]
        )

    wave_scores = []
    for wave in CALIBRATION_WAVES:
        scores = []
        for scenario_id in sorted(expected):
            obs = observations[(wave, scenario_id)]
            model = frozen[scenario_id]
            predicted_target = _predict_phase(
                model["target_phase"],
                float(expected[scenario_id]["target_canonical_total_units"]),
            )
            actual_target = float(obs["target_canonical_completion_s"])
            predicted_resident = float(model["resident_overlap_service_units_per_s"])
            actual_resident = float(obs["resident_overlap_service_units_per_s"])
            scores.extend(
                (
                    {
                        "scenario_id": scenario_id,
                        "functional": "target_completion_time",
                        "score": _upper_ratio_score(actual_target, predicted_target),
                    },
                    {
                        "scenario_id": scenario_id,
                        "functional": "resident_inverse_service",
                        "score": _upper_ratio_score(predicted_resident, actual_resident),
                    },
                )
            )
        maximizer = max(scores, key=lambda row: float(row["score"]))
        wave_scores.append(
            {
                "wave": wave,
                "maximum_ratio_score": float(maximizer["score"]),
                "maximizing_scenario": maximizer["scenario_id"],
                "maximizing_functional": maximizer["functional"],
                "scored_functional_count": len(scores),
            }
        )

    rank = int(math.ceil((len(wave_scores) + 1) * (1.0 - alpha)))
    rank_ready = bool(wave_scores and 1 <= rank <= len(wave_scores))
    margin = (
        max(
            0.0,
            sorted(float(row["maximum_ratio_score"]) for row in wave_scores)[rank - 1],
        )
        if rank_ready
        else math.inf
    )
    rows = []
    for scenario_id in sorted(expected):
        spec = expected[scenario_id]
        holdout = observations[(HOLDOUT_WAVE, scenario_id)]
        frozen_model = frozen[scenario_id]
        point_model = operational[scenario_id]
        canonical_units = float(spec["target_canonical_total_units"])
        frozen_target = _predict_phase(frozen_model["target_phase"], canonical_units)
        point_target = _predict_phase(point_model["target_phase"], canonical_units)
        upper_target = frozen_target * (1.0 + margin)
        point_resident = float(point_model["resident_overlap_service_units_per_s"])
        frozen_resident = float(frozen_model["resident_overlap_service_units_per_s"])
        lower_resident = frozen_resident / (1.0 + margin)
        actual_target = float(holdout["target_canonical_completion_s"])
        actual_resident = float(holdout["resident_overlap_service_units_per_s"])
        lower_target = canonical_units / upper_target
        target_covered = actual_target <= upper_target * (1.0 + 1e-12)
        resident_covered = lower_resident <= actual_resident * (1.0 + 1e-12)
        lower_service_vector = {
            str(spec["resident_workload_key"]): lower_resident,
            str(spec["target_workload_key"]): lower_target,
        }
        rows.append(
            {
                **dict(spec),
                "training_sample_count": len(TRAINING_WAVES),
                "calibration_wave_count": len(CALIBRATION_WAVES),
                "holdout_wave": HOLDOUT_WAVE,
                "frozen_training_model": frozen_model,
                "operational_pre_holdout_model": point_model,
                "frozen_target_completion_s": frozen_target,
                "operational_target_completion_s": point_target,
                "holdout_actual_target_completion_s": actual_target,
                "simultaneous_upper_target_completion_s": upper_target,
                "target_completion_holdout_covered": target_covered,
                "target_lower_service_units_per_s": lower_target,
                "target_holdout_realized_service_units_per_s": canonical_units / actual_target,
                "frozen_resident_overlap_service_units_per_s": frozen_resident,
                "operational_resident_overlap_service_units_per_s": point_resident,
                "holdout_actual_resident_overlap_service_units_per_s": actual_resident,
                "simultaneous_lower_resident_overlap_service_units_per_s": lower_resident,
                "resident_service_holdout_covered": resident_covered,
                "lower_service_vector": lower_service_vector,
                "target_point_relative_error": abs(point_target - actual_target) / actual_target,
                "resident_point_relative_error": abs(point_resident - actual_resident) / actual_resident,
                "lower_service_bound_kind": "wave_max_joint_target_time_resident_service_split_conformal",
            }
        )
    all_covered = bool(rows) and all(
        row["target_completion_holdout_covered"]
        and row["resident_service_holdout_covered"]
        and all(float(value) > 0.0 for value in row["lower_service_vector"].values())
        for row in rows
    )
    return {
        "constructed": rank_ready,
        "finite_sample_rank_ready": rank_ready,
        "conformal_rank": rank,
        "calibration_wave_count": len(wave_scores),
        "simultaneous_ratio_margin": margin if math.isfinite(margin) else None,
        "wave_max_scores": wave_scores,
        "simultaneous_scope": {
            "scenario_count": len(expected),
            "functionals_per_scenario": 2,
            "functionals": [
                "target_completion_time",
                "resident_inverse_service",
            ],
            "joint_functional_count_per_wave": 2 * len(expected),
        },
        "all_holdout_bounds_valid": all_covered,
        "rows": rows,
    }


def _fit_models(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    phases = [row["target_phase"] for row in rows]
    return {
        "fit": "componentwise_median",
        "sample_count": len(rows),
        "target_phase": {
            key: float(median(float(phase[key]) for phase in phases))
            for key in (
                "startup_overhead_s",
                "completion_unit_s",
                "terminal_overhead_s",
            )
        },
        "resident_overlap_service_units_per_s": float(
            median(float(row["resident_overlap_service_units_per_s"]) for row in rows)
        ),
    }


def _phase(model: Mapping[str, Any]) -> dict[str, float]:
    return {
        "startup_overhead_s": float(model["startup_overhead_s"]),
        "completion_unit_s": float(model["completion_unit_s"]),
        "terminal_overhead_s": float(model["terminal_overhead_s"]),
    }


def _predict_phase(model: Mapping[str, Any], total_units: float) -> float:
    return (
        float(model["startup_overhead_s"])
        + float(total_units) * float(model["completion_unit_s"])
        + float(model["terminal_overhead_s"])
    )


def _upper_ratio_score(actual: float, predicted: float) -> float:
    if actual <= 0.0 or predicted <= 0.0:
        return math.inf
    return actual / predicted - 1.0


def _int_or(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(fallback)


def _issue(
    code: str,
    *,
    wave: int | None = None,
    scenario: str | None = None,
    detail: str = "",
) -> dict[str, Any]:
    return {"code": code, "wave": wave, "scenario": scenario, "detail": detail}


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Critical GPU Loaded-State Stochastic LCB Gate",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Node: `{report.get('node')}`",
        f"- Ready observations: `{report.get('ready_observation_count')}` / `{report.get('expected_observation_count')}`",
        "",
        "| Scenario | Resident -> target | Upper target s | Lower resident | Lower target | Holdout covered |",
        "|---|---|---:|---:|---:|:---:|",
    ]
    for row in (report.get("certificate") or {}).get("rows") or []:
        lines.append(
            "| `{}` | `{}` -> `{}` | {:.6g} | {:.6g} | {:.6g} | {} |".format(
                row.get("scenario_id"),
                row.get("resident_workload_key"),
                row.get("target_workload_key"),
                float(row.get("simultaneous_upper_target_completion_s") or 0.0),
                float(row.get("simultaneous_lower_resident_overlap_service_units_per_s") or 0.0),
                float(row.get("target_lower_service_units_per_s") or 0.0),
                str(
                    bool(row.get("target_completion_holdout_covered"))
                    and bool(row.get("resident_service_holdout_covered"))
                ).lower(),
            )
        )
    lines.extend(["", str(report.get("claim_boundary") or ""), ""])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", required=True, choices=tuple(NODE_SPECS))
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--alpha", type=float, default=MIS_COVERAGE_ALPHA)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    report = build_critical_gpu_loaded_stochastic_lcb_gate(
        node=args.node,
        artifact_root=args.artifact_root,
        miscoverage_alpha=args.alpha,
    )
    output = args.output or (
        ARTIFACT_ROOT
        / f"critical_gpu_loaded_stochastic_lcb_gate_{args.node}_{CAMPAIGN_DATE}.json"
    )
    markdown = args.markdown_output or output.with_suffix(".md")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "pass": report["pass"]}, indent=2, sort_keys=True))
    return 0 if report["pass"] else (3 if report["status"].startswith("WAIT") else 2)


if __name__ == "__main__":
    raise SystemExit(main())
