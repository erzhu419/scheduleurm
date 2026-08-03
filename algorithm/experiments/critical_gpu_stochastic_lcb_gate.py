"""Simultaneous completion-time and lower-service gate for critical GPU cells.

The gate consumes one hardware node at a time.  Waves 1--3 freeze a
phase-aware completion model, waves 4--12 calibrate a single wave-maximum
one-sided conformal margin over every declared cell, and wave 13 is an
untouched holdout audit.  Missing or not-yet-run waves are a WAIT condition;
completed measurements that violate the campaign contract are a FAIL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import fmean, median
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from .critical_gpu_completion_campaign import (
    ARTIFACT_ROOT,
    CALIBRATION_WAVES,
    CAMPAIGN_PREFIX,
    GPU_IDLE_MEMORY_LIMIT_MB,
    GPU_IDLE_UTIL_LIMIT_PCT,
    HOLDOUT_WAVE,
    NODE_SPECS,
    PROTOCOL,
    STAGED_TRAJECTORY_EVIDENCE,
    STATIONARY_RATE_EVIDENCE,
    TRAINING_WAVES,
    WORKLOAD_SPECS,
    campaign_cells,
    wave_role,
)


CAMPAIGN_DATE = "20260803"
RESOURCE_STATE = "empty"
EXPECTED_WAVES = (*TRAINING_WAVES, *CALIBRATION_WAVES, HOLDOUT_WAVE)
EXPECTED_CELL_COUNT = 9


def default_campaign_paths(
    *,
    node: str,
    artifact_root: Path = ARTIFACT_ROOT,
) -> dict[int, Path]:
    """Return the pre-registered r01--r13 source paths for one node."""

    return {
        wave: Path(artifact_root)
        / _campaign_artifact_name(node=node, wave=wave)
        for wave in EXPECTED_WAVES
    }


def build_critical_gpu_stochastic_lcb_gate(
    *,
    node: str,
    campaign_paths: Mapping[int, Path] | None = None,
    artifact_root: Path = ARTIFACT_ROOT,
    miscoverage_alpha: float = 0.10,
) -> dict[str, Any]:
    """Build one hardware-local simultaneous split-conformal certificate."""

    if node not in NODE_SPECS:
        raise ValueError(f"unsupported node {node!r}")
    alpha = float(miscoverage_alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("miscoverage_alpha must lie in (0, 1)")

    paths = (
        {int(wave): Path(path) for wave, path in campaign_paths.items()}
        if campaign_paths is not None
        else default_campaign_paths(node=node, artifact_root=artifact_root)
    )
    expected_wave_set = set(EXPECTED_WAVES)
    supplied_wave_set = set(paths)
    validation_errors: list[dict[str, Any]] = []
    wait_reasons: list[dict[str, Any]] = []
    if 0 in supplied_wave_set:
        validation_errors.append(
            _issue("SMOKE_WAVE_FORBIDDEN", wave=0, detail="wave 0 is excluded")
        )
    for wave in sorted(supplied_wave_set - expected_wave_set):
        validation_errors.append(
            _issue(
                "UNREGISTERED_WAVE",
                wave=wave,
                detail="only pre-registered waves 1--13 are admissible",
            )
        )

    node_spec = NODE_SPECS[node]
    expected_cells = _expected_cells(node)
    if len(expected_cells) != EXPECTED_CELL_COUNT:
        raise RuntimeError(
            f"campaign definition exposes {len(expected_cells)} cells, expected 9"
        )

    payloads: dict[int, dict[str, Any]] = {}
    source_artifacts: list[dict[str, Any]] = []
    missing_waves: list[int] = []
    for wave in EXPECTED_WAVES:
        path = paths.get(wave)
        if path is None or not path.is_file():
            missing_waves.append(wave)
            wait_reasons.append(
                _issue(
                    "MISSING_WAVE",
                    wave=wave,
                    detail=str(path) if path is not None else "path not supplied",
                )
            )
            continue
        expected_name = _campaign_artifact_name(node=node, wave=wave)
        if path.name != expected_name:
            validation_errors.append(
                _issue(
                    "ARTIFACT_PATH_RULE_MISMATCH",
                    wave=wave,
                    detail=f"expected {expected_name!r}, got {path.name!r}",
                )
            )
        try:
            raw = path.read_bytes()
            payload = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            validation_errors.append(
                _issue(
                    "UNREADABLE_ARTIFACT",
                    wave=wave,
                    detail=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        if not isinstance(payload, dict):
            validation_errors.append(
                _issue(
                    "INVALID_ARTIFACT_ROOT",
                    wave=wave,
                    detail="JSON root must be an object",
                )
            )
            continue
        payloads[wave] = payload
        source_artifacts.append(
            {
                "wave": wave,
                "path": str(path),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "status": payload.get("status"),
                "pass": bool(payload.get("pass")),
                "measurement_code_sha256": (
                    (payload.get("measurement_code_manifest") or {}).get("sha256")
                ),
            }
        )

    code_hashes = {
        str((payload.get("measurement_code_manifest") or {}).get("sha256") or "")
        for payload in payloads.values()
    }
    if payloads and ("" in code_hashes or len(code_hashes) != 1):
        validation_errors.append(
            _issue(
                "MEASUREMENT_CODE_IDENTITY_MISMATCH",
                detail=f"expected one nonempty code hash, observed {sorted(code_hashes)!r}",
            )
        )

    observations: dict[tuple[int, str], dict[str, Any]] = {}
    wave_audits: list[dict[str, Any]] = []
    for wave in EXPECTED_WAVES:
        payload = payloads.get(wave)
        if payload is None:
            continue
        audit, wave_waits, wave_errors, wave_observations = _audit_wave(
            node=node,
            wave=wave,
            payload=payload,
            expected_cells=expected_cells,
        )
        wave_audits.append(audit)
        wait_reasons.extend(wave_waits)
        validation_errors.extend(wave_errors)
        observations.update(
            {
                (wave, cell_id): observation
                for cell_id, observation in wave_observations.items()
            }
        )

    expected_observation_count = len(EXPECTED_WAVES) * len(expected_cells)
    all_measurements_ready = bool(
        not missing_waves
        and not wait_reasons
        and not validation_errors
        and len(observations) == expected_observation_count
    )

    certificate: dict[str, Any] = {
        "constructed": False,
        "finite_sample_rank_ready": False,
        "rows": [],
    }
    if all_measurements_ready:
        certificate = _build_certificate(
            observations=observations,
            expected_cells=expected_cells,
            alpha=alpha,
        )
        if not certificate.get("finite_sample_rank_ready"):
            validation_errors.append(
                _issue(
                    "CONFORMAL_RANK_UNAVAILABLE",
                    detail=(
                        "the requested alpha cannot be certified with the "
                        "pre-registered calibration-wave count"
                    ),
                )
            )
        elif not certificate.get("all_holdout_bounds_valid"):
            validation_errors.append(
                _issue(
                    "HOLDOUT_NOT_COVERED",
                    detail="at least one simultaneous holdout bound was violated",
                )
            )

    if validation_errors:
        status = "FAIL_VALIDATION"
    elif wait_reasons:
        status = (
            "WAIT_MISSING_WAVES"
            if missing_waves
            else "WAIT_INCOMPLETE_WAVES"
        )
    elif certificate.get("constructed") and certificate.get(
        "all_holdout_bounds_valid"
    ):
        status = "PASS"
    else:
        status = "FAIL_CERTIFICATE"
    passed = status == "PASS"

    return {
        "gate": "critical_gpu_stochastic_lcb_gate",
        "schema_version": 1,
        "status": status,
        "pass": passed,
        "certificate_ready": passed,
        "node": node,
        "node_bucket": node_spec.node_bucket,
        "hardware_class": node_spec.hardware_class,
        "resource_state": RESOURCE_STATE,
        "measurement_protocol": PROTOCOL,
        "campaign_prefix": CAMPAIGN_PREFIX,
        "same_measurement_protocol_all_waves": all(
            payload.get("measurement_protocol") == PROTOCOL
            for payload in payloads.values()
        ),
        "same_measurement_code_all_waves": bool(
            len(payloads) == len(EXPECTED_WAVES)
            and len(code_hashes) == 1
            and "" not in code_hashes
        ),
        "measurement_code_sha256": (
            next(iter(code_hashes)) if len(code_hashes) == 1 and "" not in code_hashes else None
        ),
        "expected_cell_count": len(expected_cells),
        "expected_wave_count": len(EXPECTED_WAVES),
        "expected_observation_count": expected_observation_count,
        "ready_observation_count": len(observations),
        "training_waves": list(TRAINING_WAVES),
        "calibration_waves": list(CALIBRATION_WAVES),
        "holdout_wave": HOLDOUT_WAVE,
        "smoke_wave_excluded": True,
        "miscoverage_alpha": alpha,
        "nominal_simultaneous_coverage": 1.0 - alpha,
        "missing_waves": missing_waves,
        "wait_reasons": wait_reasons,
        "validation_errors": validation_errors,
        "all_measurements_ready": all_measurements_ready,
        "same_cell_set_all_waves": bool(
            all_measurements_ready
            and all(audit.get("same_expected_cell_set") for audit in wave_audits)
        ),
        "hardware_local_only": True,
        "expected_cells": [expected_cells[key] for key in sorted(expected_cells)],
        "wave_audits": wave_audits,
        "source_artifacts": source_artifacts,
        "certificate": certificate,
        "claim_boundary": (
            "The certificate is hardware-local and statewise for the declared "
            "workload_env x node_bucket x empty resource_state x profile cells. "
            "Stationary actions require every child to expose a stable task-native "
            "rate and a natural completion model. The staged RE-SAC p5 action is "
            "instead certified as a finite admission-and-drain trajectory: every "
            "child supplies all 40 outer-loop observations and naturally completes, "
            "while a static within-trajectory rate remains diagnostic rather than "
            "an assumed service constant. Both modes include allocated checkpoint "
            "evidence, and their completion-time observations feed the same "
            "one-sided split-conformal lower-service construction. Training waves freeze the "
            "phase model; calibration-wave maximum scores provide one-sided joint "
            "coverage over both drain completion time and mean task JCT for the "
            "nine declared actions. It neither uses smoke measurements nor pools "
            "hardware classes, and it does not extrapolate to loaded states, "
            "unmeasured profiles, or arbitrary future workload environments."
        ),
    }


def _expected_cells(node: str) -> dict[str, dict[str, Any]]:
    spec_by_key = {spec.workload_key: spec for spec in WORKLOAD_SPECS}
    result: dict[str, dict[str, Any]] = {}
    for raw in campaign_cells(node=node, wave=TRAINING_WAVES[0]):
        key = _service_cell_id(raw, resource_state=RESOURCE_STATE)
        if key in result:
            raise RuntimeError(f"duplicate expected service cell {key!r}")
        spec = spec_by_key[str(raw["workload_key"])]
        evidence_mode = str(raw["service_evidence_mode"])
        trajectory_mode = evidence_mode == STAGED_TRAJECTORY_EVIDENCE
        result[key] = {
            "service_cell_id": key,
            "workload_key": raw["workload_key"],
            "workload_env": raw["workload_env"],
            "quadrant": raw["quadrant"],
            "node": raw["node"],
            "node_bucket": raw["node_bucket"],
            "hardware_class": raw["hardware_class"],
            "resource_state": RESOURCE_STATE,
            "profile": int(raw["profile"]),
            "profile_axis": raw["profile_axis"],
            "gpu_count": len(raw["gpus"]),
            "total_task_count": int(raw["total_task_count"]),
            "total_units_per_task": int(spec.max_iters),
            "service_unit": spec.unit,
            "min_progress_observations": (
                int(spec.max_iters)
                if trajectory_mode
                else int(spec.min_rate_samples)
            ),
            "min_interval_samples": (
                max(1, int(spec.max_iters) - 1)
                if trajectory_mode
                else int(spec.min_rate_samples)
            ),
            "trajectory_cycle_units": int(spec.stable_cycle_units),
            "trajectory_complete_cycles": (
                int(spec.max_iters) // max(1, int(spec.stable_cycle_units))
            ),
            "service_evidence_mode": evidence_mode,
        }
    return result


def _audit_wave(
    *,
    node: str,
    wave: int,
    payload: Mapping[str, Any],
    expected_cells: Mapping[str, Mapping[str, Any]],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    waits: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    observations: dict[str, dict[str, Any]] = {}
    expected_role = wave_role(wave)
    node_spec = NODE_SPECS[node]

    scalar_contract = {
        "gate": "critical_gpu_completion_campaign",
        "campaign_id": (
            f"{CAMPAIGN_PREFIX}_{node}_r{wave:02d}_{CAMPAIGN_DATE}"
        ),
        "measurement_protocol": PROTOCOL,
        "node": node,
        "wave": wave,
        "split_role": expected_role,
        "legacy_scheduler_limits_bypassed": True,
        "algorithm_path": "controlled_theorem_measurement_direct",
        "terminate_on_stable": False,
        "natural_completion_required": True,
        "task_native_progress_required": True,
        "coordinated_post_warmup_start_required_for_builtin_gpu_workloads": True,
        "admission_delay_included_in_completion_jct": True,
        "staged_trajectory_completion_evidence_required": True,
        "real_checkpoint_allocation_required": True,
    }
    for field, expected in scalar_contract.items():
        if payload.get(field) != expected:
            errors.append(
                _issue(
                    "CAMPAIGN_CONTRACT_MISMATCH",
                    wave=wave,
                    detail=f"{field}: expected {expected!r}, got {payload.get(field)!r}",
                )
            )
    split = payload.get("pre_registered_split") or {}
    if split != {
        "training": list(TRAINING_WAVES),
        "calibration": list(CALIBRATION_WAVES),
        "holdout": HOLDOUT_WAVE,
        "smoke_excluded": 0,
    }:
        errors.append(
            _issue(
                "SPLIT_CONTRACT_MISMATCH",
                wave=wave,
                detail="training/calibration/holdout declaration changed",
            )
        )
    selection = payload.get("selection") or {}
    if selection.get("filtered"):
        errors.append(
            _issue(
                "FILTERED_CAMPAIGN_FORBIDDEN",
                wave=wave,
                detail="the simultaneous gate requires all nine cells in one wave",
            )
        )
    code_manifest = payload.get("measurement_code_manifest") or {}
    final_code_manifest = payload.get("final_measurement_code_manifest") or {}
    if not (
        payload.get("measurement_code_unchanged") is True
        and code_manifest.get("sha256")
        and code_manifest.get("sha256") == final_code_manifest.get("sha256")
    ):
        errors.append(
            _issue(
                "MEASUREMENT_CODE_IDENTITY_INVALID",
                wave=wave,
                detail="campaign code manifest is missing or changed during the wave",
            )
        )
    source_node_spec = payload.get("node_spec") or {}
    for field, expected in (
        ("node", node),
        ("node_bucket", node_spec.node_bucket),
        ("hardware_class", node_spec.hardware_class),
    ):
        if source_node_spec.get(field) != expected:
            errors.append(
                _issue(
                    "CROSS_HARDWARE_OR_NODE_SOURCE",
                    wave=wave,
                    detail=(
                        f"node_spec.{field}: expected {expected!r}, "
                        f"got {source_node_spec.get(field)!r}"
                    ),
                )
            )

    expected_set = set(expected_cells)
    planned_index, planned_duplicates, planned_invalid = _index_cells(
        payload.get("planned_cells") or []
    )
    row_index, row_duplicates, row_invalid = _index_cells(payload.get("rows") or [])
    errors.extend(
        _issue("INVALID_CELL_IDENTITY", wave=wave, detail=detail)
        for detail in [*planned_invalid, *row_invalid]
    )
    errors.extend(
        _issue("DUPLICATE_CELL", wave=wave, cell=cell, detail=kind)
        for kind, duplicates in (
            ("planned_cells", planned_duplicates),
            ("rows", row_duplicates),
        )
        for cell in duplicates
    )
    if set(planned_index) != expected_set:
        errors.append(
            _issue(
                "PLANNED_CELL_SET_MISMATCH",
                wave=wave,
                detail=_set_delta(expected_set, set(planned_index)),
            )
        )
    missing_rows = sorted(expected_set - set(row_index))
    unexpected_rows = sorted(set(row_index) - expected_set)
    if missing_rows:
        waits.append(
            _issue(
                "MISSING_CELLS",
                wave=wave,
                detail=", ".join(missing_rows),
            )
        )
    if unexpected_rows:
        errors.append(
            _issue(
                "UNEXPECTED_CELLS",
                wave=wave,
                detail=", ".join(unexpected_rows),
            )
        )

    campaign_status = str(payload.get("status") or "")
    campaign_pass = bool(payload.get("pass"))
    if campaign_status != "PASS" or not campaign_pass:
        if campaign_status in {
            "MANIFEST_ONLY",
            "RUNNING",
            "WAIT_ENVIRONMENT",
            "WAIT_RESOURCE",
        } or missing_rows:
            waits.append(
                _issue(
                    "CAMPAIGN_NOT_COMPLETE",
                    wave=wave,
                    detail=f"status={campaign_status!r}, pass={campaign_pass}",
                )
            )
        else:
            errors.append(
                _issue(
                    "COMPLETED_CAMPAIGN_NOT_PASSING",
                    wave=wave,
                    detail=f"status={campaign_status!r}, pass={campaign_pass}",
                )
            )

    campaign_terminal_pass = campaign_status == "PASS" and campaign_pass
    if campaign_terminal_pass:
        for cell_id in sorted(expected_set & set(row_index)):
            row = row_index[cell_id]
            row_errors, observation = _audit_measurement_row(
                row=row,
                expected=expected_cells[cell_id],
                wave=wave,
            )
            errors.extend(row_errors)
            if observation is not None and not row_errors:
                observations[cell_id] = observation

    audit = {
        "wave": wave,
        "split_role": expected_role,
        "campaign_status": campaign_status,
        "campaign_pass": campaign_pass,
        "planned_cell_count": len(planned_index),
        "observed_cell_count": len(row_index),
        "ready_cell_count": len(observations),
        "terminal_pass": campaign_terminal_pass,
        "missing_cells": missing_rows,
        "unexpected_cells": unexpected_rows,
        "same_expected_cell_set": bool(
            not planned_duplicates
            and not row_duplicates
            and set(planned_index) == expected_set
            and set(row_index) == expected_set
        ),
        "wait_count": len(waits),
        "validation_error_count": len(errors),
    }
    return audit, waits, errors, observations


def _audit_measurement_row(
    *,
    row: Mapping[str, Any],
    expected: Mapping[str, Any],
    wave: int,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    cell_id = str(expected["service_cell_id"])
    errors: list[dict[str, Any]] = []

    def require(condition: bool, code: str, detail: str) -> None:
        if not condition:
            errors.append(_issue(code, wave=wave, cell=cell_id, detail=detail))

    for field in (
        "node",
        "node_bucket",
        "hardware_class",
        "workload_key",
        "workload_env",
        "profile_axis",
        "total_task_count",
        "service_evidence_mode",
    ):
        require(
            row.get(field) == expected.get(field),
            "CELL_METADATA_MISMATCH",
            f"{field}: expected {expected.get(field)!r}, got {row.get(field)!r}",
        )
    require(
        int(row.get("profile") or 0) == int(expected["profile"]),
        "CELL_METADATA_MISMATCH",
        f"profile: expected {expected['profile']}, got {row.get('profile')!r}",
    )
    require(_int_or(row.get("wave"), -1) == wave, "WAVE_MISMATCH", "row wave differs")
    require(row.get("split_role") == wave_role(wave), "SPLIT_ROLE_MISMATCH", "row role differs")
    require(
        row.get("measurement_protocol") == PROTOCOL,
        "PROTOCOL_MISMATCH",
        f"row protocol must be {PROTOCOL!r}",
    )
    require(bool(row.get("ready")), "ROW_NOT_READY", "ready is not true")
    require(row.get("status") == "READY", "ROW_NOT_READY", f"status={row.get('status')!r}")
    require(not bool(row.get("capacity_boundary")), "CAPACITY_BOUNDARY", "cell did not naturally complete")

    expected_tasks = int(expected["total_task_count"])
    expected_gpus = int(expected["gpu_count"])
    profile = int(expected["profile"])
    evidence_mode = str(expected["service_evidence_mode"])
    trajectory_mode = evidence_mode == STAGED_TRAJECTORY_EVIDENCE
    require(
        evidence_mode in {
            STATIONARY_RATE_EVIDENCE,
            STAGED_TRAJECTORY_EVIDENCE,
        },
        "SERVICE_EVIDENCE_MODE_INVALID",
        f"unregistered evidence mode {evidence_mode!r}",
    )
    preflight = row.get("preflight") or {}
    selected_gpus = preflight.get("selected_gpus") or []
    require(bool(preflight.get("ready")), "EMPTY_PREFLIGHT_INVALID", "preflight is not ready")
    require(_int_or(preflight.get("returncode"), -1) == 0, "EMPTY_PREFLIGHT_INVALID", "preflight returncode is nonzero")
    require(not (preflight.get("compute_process_rows") or []), "EMPTY_PREFLIGHT_INVALID", "compute process existed before launch")
    require(len(selected_gpus) == expected_gpus, "EMPTY_PREFLIGHT_INVALID", "selected GPU count differs")
    require(
        all(
            float(gpu.get("util_pct") or 0.0) <= GPU_IDLE_UTIL_LIMIT_PCT
            and float(gpu.get("used_mb") or 0.0) <= GPU_IDLE_MEMORY_LIMIT_MB
            for gpu in selected_gpus
        ),
        "EMPTY_PREFLIGHT_INVALID",
        "GPU utilization or memory exceeds the registered empty-state boundary",
    )

    summary = row.get("summary") or {}
    require(summary.get("probe") == "remote_workload_selected_profile_probe", "PROGRESS_SOURCE_INVALID", "unexpected probe source")
    require(bool(summary.get("measurement_valid")), "SUMMARY_NOT_READY", "measurement_valid is false")
    require(bool(row.get("service_evidence_ready")), "PROGRESS_SOURCE_INVALID", "service evidence is not ready")
    if trajectory_mode:
        require(summary.get("require_stable_rate") is False, "SERVICE_EVIDENCE_MODE_INVALID", "trajectory action incorrectly requires a stationary rate")
        trajectory = row.get("trajectory_progress_audit") or {}
        require(bool(trajectory.get("required")), "TRAJECTORY_EVIDENCE_INVALID", "trajectory audit is not required")
        require(bool(trajectory.get("ready")), "TRAJECTORY_EVIDENCE_INVALID", "trajectory audit is not ready")
        require(trajectory.get("service_evidence_mode") == evidence_mode, "TRAJECTORY_EVIDENCE_INVALID", "trajectory mode differs")
        require(bool(trajectory.get("completion_trajectory_includes_admission_and_drain")), "TRAJECTORY_EVIDENCE_INVALID", "trajectory omits admission or drain")
        require(int(trajectory.get("expected_task_count") or 0) == expected_tasks, "TASK_COUNT_MISMATCH", "trajectory task count differs")
        require(int(trajectory.get("observed_task_count") or 0) == expected_tasks, "TASK_COUNT_MISMATCH", "trajectory observed task count differs")
        require(int(trajectory.get("expected_progress_observations_per_task") or 0) == int(expected["total_units_per_task"]), "TRAJECTORY_EVIDENCE_INVALID", "trajectory progress support differs")
        require(int(trajectory.get("expected_interval_samples_per_task") or 0) == int(expected["min_interval_samples"]), "TRAJECTORY_EVIDENCE_INVALID", "trajectory interval support differs")
        require(int(trajectory.get("cycle_units") or 0) == int(expected["trajectory_cycle_units"]), "TRAJECTORY_EVIDENCE_INVALID", "trajectory cycle width differs")
        require(int(trajectory.get("expected_complete_cycles_per_task") or 0) == int(expected["trajectory_complete_cycles"]), "TRAJECTORY_EVIDENCE_INVALID", "trajectory cycle support differs")
        trajectory_children = trajectory.get("children") or []
        require(len(trajectory_children) == expected_tasks, "TASK_COUNT_MISMATCH", "trajectory child audit count differs")
        require(
            all(
                bool(child.get("ready"))
                and int(child.get("progress_observation_count") or 0)
                >= int(expected["total_units_per_task"])
                and int(child.get("interval_sample_count") or 0)
                >= int(expected["min_interval_samples"])
                and int(child.get("complete_cycle_count") or 0)
                >= int(expected["trajectory_complete_cycles"])
                for child in trajectory_children
            ),
            "TRAJECTORY_EVIDENCE_INVALID",
            "a trajectory child lacks full progress, interval, or cycle support",
        )
    else:
        require(summary.get("require_stable_rate") is True, "SERVICE_EVIDENCE_MODE_INVALID", "stationary action does not require a stable rate")
        require(bool(summary.get("all_stable_rate_ready")), "PROGRESS_SOURCE_INVALID", "stable task-native rate missing")
    require(bool(summary.get("all_completion_models_ready")), "NATURAL_COMPLETION_INVALID", "completion model missing")
    require(bool(summary.get("coordinated_profile_launch")), "SUMMARY_NOT_READY", "profile launch was not coordinated")
    require(summary.get("terminate_on_stable") is False, "NATURAL_COMPLETION_INVALID", "measurement terminated on stable ETA")
    require(not bool(summary.get("capacity_boundary")), "CAPACITY_BOUNDARY", "summary reports capacity boundary")
    require(bool(summary.get("placement_valid")), "PLACEMENT_INVALID", "placement is invalid")
    require(int(summary.get("running_count") or 0) == expected_tasks, "TASK_COUNT_MISMATCH", "running task count differs")
    require(int(summary.get("running_with_rate_count") or 0) == expected_tasks, "PROGRESS_SOURCE_INVALID", "not every child has a rate")
    if not trajectory_mode:
        require(int(summary.get("stable_rate_ready_count") or 0) == expected_tasks, "PROGRESS_SOURCE_INVALID", "not every child has a stable rate")
    require(int(summary.get("completion_model_ready_count") or 0) == expected_tasks, "NATURAL_COMPLETION_INVALID", "not every child has a completion model")
    require(int(summary.get("returncode_accepted_count") or 0) == expected_tasks, "NATURAL_COMPLETION_INVALID", "not every child returncode was accepted")
    require(int(summary.get("profile") or 0) == profile, "CELL_METADATA_MISMATCH", "summary profile differs")
    require(str(summary.get("node") or "") == str(expected["node"]), "CELL_METADATA_MISMATCH", "summary node differs")
    require(
        all(int(value) == profile for value in (summary.get("per_gpu_running") or {}).values()),
        "PLACEMENT_INVALID",
        "per-GPU running count differs from profile",
    )
    if str(summary.get("eta_source") or "").lower() == "history":
        require(False, "HISTORY_ETA_FORBIDDEN", "summary used history ETA")

    probe = row.get("probe_result") or {}
    require(bool(probe.get("pass")), "SUMMARY_NOT_READY", "nested probe did not pass")
    if not trajectory_mode:
        require(int(probe.get("stable_rate_ready_count") or 0) == expected_tasks, "PROGRESS_SOURCE_INVALID", "probe lacks stable rows")
    require(int(probe.get("completion_model_ready_count") or 0) == expected_tasks, "NATURAL_COMPLETION_INVALID", "probe lacks completion rows")
    backend = row.get("backend_audit") or {}
    require(bool(backend.get("ready")), "BACKEND_INVALID", "workload backend audit failed")
    require(int(backend.get("accepted_task_count") or 0) == expected_tasks, "BACKEND_INVALID", "backend task count differs")
    completion = row.get("completion_audit") or {}
    require(bool(completion.get("ready")), "NATURAL_COMPLETION_INVALID", "completion audit failed")
    require(bool(completion.get("all_natural_exit")), "NATURAL_COMPLETION_INVALID", "a child did not exit naturally")
    require(bool(completion.get("all_completion_models_ready")), "NATURAL_COMPLETION_INVALID", "a child model is not ready")
    require(bool(completion.get("checkpoint_or_finalization_observed")), "CHECKPOINT_INVALID", "checkpoint/finalization phase missing")
    require(int(completion.get("task_count") or 0) == expected_tasks, "TASK_COUNT_MISMATCH", "completion audit task count differs")
    coordination = row.get("coordination_audit") or {}
    coordination_required = str(expected["workload_key"]) != "hybrid_rl_resac_ant"
    require(bool(coordination.get("ready")), "COORDINATION_BARRIER_INVALID", "post-warmup start barrier audit failed")
    require(bool(coordination.get("required")) == coordination_required, "COORDINATION_BARRIER_INVALID", "barrier requirement differs")
    if coordination_required:
        require(int(coordination.get("start_marker_count") or 0) == expected_tasks, "COORDINATION_BARRIER_INVALID", "barrier start marker count differs")
        require(int(coordination.get("end_marker_count") or 0) == expected_tasks, "COORDINATION_BARRIER_INVALID", "barrier end marker count differs")
    require(bool(row.get("measurement_code_identity_ready")), "MEASUREMENT_CODE_IDENTITY_INVALID", "row code identity is not ready")
    admission = row.get("admission_audit") or {}
    require(bool(admission.get("ready")), "ADMISSION_DELAY_INVALID", "staged admission audit failed")
    require(bool(admission.get("delay_counted_in_startup_and_jct")), "ADMISSION_DELAY_INVALID", "admission delay is not counted in JCT")

    artifact = row.get("artifact_audit") or {}
    checkpoint_files = [
        item
        for item in artifact.get("files") or []
        if "/checkpoints/" in str(item.get("path") or "").replace("\\", "/")
        and int(item.get("size_bytes") or 0) > 0
        and int(item.get("allocated_blocks_512") or 0) > 0
    ]
    require(bool(row.get("checkpoint_allocation_ready")), "CHECKPOINT_INVALID", "checkpoint allocation flag is false")
    require(bool(artifact.get("ready")), "CHECKPOINT_INVALID", "artifact audit is not ready")
    require(_int_or(artifact.get("returncode"), -1) == 0, "CHECKPOINT_INVALID", "artifact audit returncode is nonzero")
    require(int(artifact.get("total_allocated_bytes") or 0) > 0, "CHECKPOINT_INVALID", "no allocated checkpoint bytes")
    require(len(checkpoint_files) >= expected_tasks, "CHECKPOINT_INVALID", "fewer real checkpoint files than tasks")

    children = summary.get("rows") or []
    require(len(children) == expected_tasks, "TASK_COUNT_MISMATCH", "child row count differs")
    child_run_names = [str(child.get("run_name") or "").strip() for child in children]
    checkpoint_paths = [str(item.get("path") or "") for item in checkpoint_files]
    require(
        all(child_run_names),
        "CHECKPOINT_INVALID",
        "a child run name is missing, so checkpoint ownership cannot be audited",
    )
    require(
        all(
            any(run_name in checkpoint_path for checkpoint_path in checkpoint_paths)
            for run_name in child_run_names
        ),
        "CHECKPOINT_INVALID",
        "at least one child lacks an allocated checkpoint file",
    )
    child_models: list[dict[str, float]] = []
    min_progress = int(expected["min_progress_observations"])
    min_intervals = int(expected["min_interval_samples"])
    for index, child in enumerate(children):
        model = child.get("completion_model") or {}
        child_prefix = f"child {index}"
        require(_int_or(child.get("returncode"), -1) == 0, "NATURAL_COMPLETION_INVALID", f"{child_prefix} returncode is nonzero")
        if not trajectory_mode:
            require(bool(child.get("stable_rate_ready")), "PROGRESS_SOURCE_INVALID", f"{child_prefix} stable rate missing")
            require(float(child.get("stable_rate") or 0.0) > 0.0, "PROGRESS_SOURCE_INVALID", f"{child_prefix} stable rate is nonpositive")
        require(bool(child.get("completion_model_ready")), "NATURAL_COMPLETION_INVALID", f"{child_prefix} model flag is false")
        require(str(child.get("unit") or "") == str(expected["service_unit"]), "PROGRESS_SOURCE_INVALID", f"{child_prefix} progress unit differs")
        require(str(child.get("eta_source") or "").lower() != "history", "HISTORY_ETA_FORBIDDEN", f"{child_prefix} used history ETA")
        require(bool(model.get("completion_model_ready")), "NATURAL_COMPLETION_INVALID", f"{child_prefix} phase model is not ready")
        require(_int_or(model.get("child_returncode"), -1) == 0, "NATURAL_COMPLETION_INVALID", f"{child_prefix} model returncode is nonzero")
        require(bool(model.get("natural_exit")), "NATURAL_COMPLETION_INVALID", f"{child_prefix} was not a natural exit")
        require(not bool(model.get("stopped_on_stable")), "NATURAL_COMPLETION_INVALID", f"{child_prefix} stopped on stable ETA")
        require(str(model.get("readiness_reason") or "") == "ready", "NATURAL_COMPLETION_INVALID", f"{child_prefix} readiness reason differs")
        require(str(model.get("unit") or "") == str(expected["service_unit"]), "PROGRESS_SOURCE_INVALID", f"{child_prefix} model unit differs")
        require(int(model.get("progress_observation_count") or 0) >= min_progress, "PROGRESS_SOURCE_INVALID", f"{child_prefix} has too few progress observations")
        require(int(model.get("interval_sample_count") or 0) >= min_intervals, "PROGRESS_SOURCE_INVALID", f"{child_prefix} has too few outer-loop intervals")
        require(int(model.get("total_units") or 0) == int(expected["total_units_per_task"]), "TASK_UNITS_MISMATCH", f"{child_prefix} total units differ")
        startup = _nonnegative_float(model.get("startup_overhead_s"))
        unit_s = _positive_float(model.get("completion_unit_s"))
        terminal = _nonnegative_float(model.get("terminal_overhead_s"))
        total_units = _positive_float(model.get("total_units"))
        total_wall = _positive_float(model.get("total_wall_s"))
        require(unit_s > 0.0 and total_units > 0.0 and total_wall > 0.0, "PHASE_MODEL_INVALID", f"{child_prefix} phase values are nonpositive")
        phase_total = startup + total_units * unit_s + terminal
        require(_close(phase_total, total_wall), "PHASE_MODEL_INVALID", f"{child_prefix} phase sum does not equal natural JCT")
        if str(expected["workload_key"]) != "hybrid_rl_resac_ant":
            require(float(model.get("checkpoint_observed_s") or 0.0) > 0.0, "CHECKPOINT_INVALID", f"{child_prefix} checkpoint phase missing")
            require(float(model.get("save_observed_s") or 0.0) > 0.0, "CHECKPOINT_INVALID", f"{child_prefix} save phase missing")
        else:
            require(float(model.get("finalization_after_last_progress_s") or 0.0) > 0.0, "CHECKPOINT_INVALID", f"{child_prefix} finalization phase missing")
        child_models.append(
            {
                "startup_overhead_s": startup,
                "completion_unit_s": unit_s,
                "terminal_overhead_s": terminal,
                "total_units": total_units,
                "total_wall_s": total_wall,
            }
        )

    if errors or len(child_models) != expected_tasks:
        return errors, None
    critical = max(child_models, key=lambda item: item["total_wall_s"])
    aggregate_units = sum(item["total_units"] for item in child_models)
    return errors, {
        "wave": wave,
        "split_role": wave_role(wave),
        "service_cell_id": cell_id,
        "workload_key": expected["workload_key"],
        "workload_env": expected["workload_env"],
        "node": expected["node"],
        "node_bucket": expected["node_bucket"],
        "hardware_class": expected["hardware_class"],
        "resource_state": RESOURCE_STATE,
        "profile": profile,
        "profile_axis": expected["profile_axis"],
        "task_count": expected_tasks,
        "service_unit": expected["service_unit"],
        "service_evidence_mode": evidence_mode,
        "aggregate_total_units": aggregate_units,
        "drain_completion_s": critical["total_wall_s"],
        "mean_task_jct_s": fmean(item["total_wall_s"] for item in child_models),
        "drain_phase": {
            "startup_overhead_s": critical["startup_overhead_s"],
            "completion_unit_s": critical["completion_unit_s"],
            "terminal_overhead_s": critical["terminal_overhead_s"],
            "total_units_per_task": critical["total_units"],
        },
        "mean_jct_phase": {
            "startup_overhead_s": fmean(item["startup_overhead_s"] for item in child_models),
            "completion_unit_s": fmean(item["completion_unit_s"] for item in child_models),
            "terminal_overhead_s": fmean(item["terminal_overhead_s"] for item in child_models),
            "total_units_per_task": fmean(item["total_units"] for item in child_models),
        },
        "task_native_progress_ready": True,
        "stationary_stable_rate_ready": not trajectory_mode,
        "trajectory_completion_support_ready": trajectory_mode,
        "natural_completion_ready": True,
        "real_checkpoint_ready": True,
        "empty_preflight_ready": True,
    }


def _build_certificate(
    *,
    observations: Mapping[tuple[int, str], Mapping[str, Any]],
    expected_cells: Mapping[str, Mapping[str, Any]],
    alpha: float,
) -> dict[str, Any]:
    frozen_models: dict[str, dict[str, Any]] = {}
    operational_models: dict[str, dict[str, Any]] = {}
    for cell_id in sorted(expected_cells):
        training = [observations[(wave, cell_id)] for wave in TRAINING_WAVES]
        pre_holdout = [
            observations[(wave, cell_id)]
            for wave in (*TRAINING_WAVES, *CALIBRATION_WAVES)
        ]
        frozen_models[cell_id] = _fit_phase_models(training)
        operational_models[cell_id] = _fit_phase_models(pre_holdout)

    wave_scores: list[dict[str, Any]] = []
    for wave in CALIBRATION_WAVES:
        scores: list[dict[str, Any]] = []
        for cell_id in sorted(expected_cells):
            observation = observations[(wave, cell_id)]
            model = frozen_models[cell_id]
            drain_base = _predict_phase(model["drain_phase"])
            mean_base = _predict_phase(model["mean_jct_phase"])
            scores.extend(
                (
                    {
                        "cell": cell_id,
                        "functional": "drain_completion",
                        "score": _ratio_score(
                            actual=float(observation["drain_completion_s"]),
                            predicted=drain_base,
                        ),
                    },
                    {
                        "cell": cell_id,
                        "functional": "mean_task_jct",
                        "score": _ratio_score(
                            actual=float(observation["mean_task_jct_s"]),
                            predicted=mean_base,
                        ),
                    },
                )
            )
        maximizer = max(scores, key=lambda item: float(item["score"]))
        wave_scores.append(
            {
                "wave": wave,
                "maximum_ratio_score": float(maximizer["score"]),
                "maximizing_cell": maximizer["cell"],
                "maximizing_functional": maximizer["functional"],
                "scored_functional_count": len(scores),
            }
        )

    raw_rank = int(math.ceil((len(wave_scores) + 1) * (1.0 - alpha)))
    rank_ready = bool(wave_scores and 1 <= raw_rank <= len(wave_scores))
    margin = (
        max(
            0.0,
            sorted(float(row["maximum_ratio_score"]) for row in wave_scores)[
                raw_rank - 1
            ],
        )
        if rank_ready
        else math.inf
    )

    rows: list[dict[str, Any]] = []
    for cell_id in sorted(expected_cells):
        expected = expected_cells[cell_id]
        holdout = observations[(HOLDOUT_WAVE, cell_id)]
        frozen = frozen_models[cell_id]
        operational = operational_models[cell_id]
        frozen_drain = _predict_phase(frozen["drain_phase"])
        frozen_mean = _predict_phase(frozen["mean_jct_phase"])
        point_drain = _predict_phase(operational["drain_phase"])
        point_mean = _predict_phase(operational["mean_jct_phase"])
        upper_drain = frozen_drain * (1.0 + margin) if math.isfinite(margin) else math.inf
        upper_mean = frozen_mean * (1.0 + margin) if math.isfinite(margin) else math.inf
        actual_drain = float(holdout["drain_completion_s"])
        actual_mean = float(holdout["mean_task_jct_s"])
        aggregate_units = float(holdout["aggregate_total_units"])
        drain_covered = bool(
            math.isfinite(upper_drain)
            and actual_drain <= upper_drain * (1.0 + 1e-12)
        )
        mean_covered = bool(
            math.isfinite(upper_mean)
            and actual_mean <= upper_mean * (1.0 + 1e-12)
        )
        lower_service = (
            aggregate_units / upper_drain
            if aggregate_units > 0.0 and math.isfinite(upper_drain) and upper_drain > 0.0
            else 0.0
        )
        realized_service = (
            aggregate_units / actual_drain
            if aggregate_units > 0.0 and actual_drain > 0.0
            else 0.0
        )
        lower_valid = bool(
            drain_covered
            and lower_service > 0.0
            and lower_service <= realized_service * (1.0 + 1e-12)
        )
        rows.append(
            {
                **dict(expected),
                "training_sample_count": len(TRAINING_WAVES),
                "calibration_wave_count": len(CALIBRATION_WAVES),
                "holdout_wave": HOLDOUT_WAVE,
                "frozen_training_phase_model": frozen,
                "operational_pre_holdout_phase_model": operational,
                "frozen_base_drain_completion_s": frozen_drain,
                "frozen_base_mean_task_jct_s": frozen_mean,
                "operational_point_drain_completion_s": point_drain,
                "operational_point_mean_task_jct_s": point_mean,
                "holdout_actual_drain_completion_s": actual_drain,
                "holdout_actual_mean_task_jct_s": actual_mean,
                "drain_point_relative_error": abs(point_drain - actual_drain) / actual_drain,
                "mean_jct_point_relative_error": abs(point_mean - actual_mean) / actual_mean,
                "simultaneous_upper_drain_completion_s": (
                    upper_drain if math.isfinite(upper_drain) else None
                ),
                "simultaneous_upper_mean_task_jct_s": (
                    upper_mean if math.isfinite(upper_mean) else None
                ),
                "drain_completion_holdout_covered": drain_covered,
                "mean_task_jct_holdout_covered": mean_covered,
                "aggregate_total_units": aggregate_units,
                "lower_service_units_per_s": lower_service,
                "realized_service_units_per_s": realized_service,
                "lower_service_valid_on_holdout": lower_valid,
                "lower_service_bound_kind": (
                    "phase_aware_wave_max_simultaneous_split_conformal"
                ),
            }
        )

    all_covered = bool(rows) and all(
        row["drain_completion_holdout_covered"]
        and row["mean_task_jct_holdout_covered"]
        and row["lower_service_valid_on_holdout"]
        for row in rows
    )
    return {
        "constructed": rank_ready,
        "finite_sample_rank_ready": rank_ready,
        "conformal_rank": raw_rank,
        "calibration_wave_count": len(wave_scores),
        "simultaneous_ratio_margin": margin if math.isfinite(margin) else None,
        "wave_max_scores": wave_scores,
        "simultaneous_scope": {
            "cell_count": len(expected_cells),
            "functionals_per_cell": 2,
            "functionals": ["drain_completion", "mean_task_jct"],
            "joint_functional_count_per_wave": 2 * len(expected_cells),
        },
        "all_holdout_bounds_valid": all_covered,
        "rows": rows,
    }


def _fit_phase_models(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "fit": "componentwise_median",
        "sample_count": len(rows),
        "drain_phase": _median_phase(rows, "drain_phase"),
        "mean_jct_phase": _median_phase(rows, "mean_jct_phase"),
    }


def _median_phase(
    rows: Sequence[Mapping[str, Any]],
    field: str,
) -> dict[str, float]:
    phases = [row[field] for row in rows]
    return {
        name: float(median(float(phase[name]) for phase in phases))
        for name in (
            "startup_overhead_s",
            "completion_unit_s",
            "terminal_overhead_s",
            "total_units_per_task",
        )
    }


def _predict_phase(model: Mapping[str, Any]) -> float:
    return (
        float(model["startup_overhead_s"])
        + float(model["total_units_per_task"])
        * float(model["completion_unit_s"])
        + float(model["terminal_overhead_s"])
    )


def _ratio_score(*, actual: float, predicted: float) -> float:
    if actual <= 0.0 or predicted <= 0.0:
        return math.inf
    return actual / predicted - 1.0


def _index_cells(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[str], list[str]]:
    indexed: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    invalid: list[str] = []
    for index, row in enumerate(rows):
        try:
            key = _service_cell_id(row, resource_state=RESOURCE_STATE)
        except (KeyError, TypeError, ValueError) as exc:
            invalid.append(f"row {index}: {type(exc).__name__}: {exc}")
            continue
        if key in indexed:
            duplicates.append(key)
            continue
        indexed[key] = dict(row)
    return indexed, duplicates, invalid


def _service_cell_id(
    row: Mapping[str, Any],
    *,
    resource_state: str,
) -> str:
    workload_env = str(row["workload_env"]).strip()
    node_bucket = str(row["node_bucket"]).strip()
    profile = int(row["profile"])
    if not workload_env or not node_bucket or profile <= 0:
        raise ValueError("workload_env, node_bucket, and positive profile are required")
    return (
        f"workload_env={workload_env}|node_bucket={node_bucket}|"
        f"resource_state={resource_state}|profile={profile}"
    )


def _campaign_artifact_name(*, node: str, wave: int) -> str:
    return f"{CAMPAIGN_PREFIX}_{node}_r{int(wave):02d}_{CAMPAIGN_DATE}.json"


def _set_delta(expected: set[str], observed: set[str]) -> str:
    return (
        f"missing={sorted(expected - observed)!r}, "
        f"unexpected={sorted(observed - expected)!r}"
    )


def _issue(
    code: str,
    *,
    wave: int | None = None,
    cell: str | None = None,
    detail: str = "",
) -> dict[str, Any]:
    return {
        "code": code,
        "wave": wave,
        "cell": cell,
        "detail": detail,
    }


def _positive_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) and result > 0.0 else 0.0


def _int_or(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(fallback)


def _nonnegative_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) and result >= 0.0 else 0.0


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-7, abs_tol=1e-7)


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


def write_gate_outputs(
    report: Mapping[str, Any],
    *,
    json_path: Path,
    markdown_path: Path,
) -> None:
    _atomic_write(
        json_path,
        json.dumps(dict(report), indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    _atomic_write(markdown_path, _markdown(report, json_path))


def _markdown(report: Mapping[str, Any], json_path: Path) -> str:
    certificate = report.get("certificate") or {}
    margin = certificate.get("simultaneous_ratio_margin")
    lines = [
        "# Critical GPU stochastic completion/LCB gate",
        "",
        f"- JSON artifact: `{json_path}`",
        f"- Status: `{report.get('status')}`",
        f"- Node: `{report.get('node')}`",
        f"- Hardware-local bucket: `{report.get('node_bucket')}`",
        f"- Resource state: `{report.get('resource_state')}`",
        f"- Ready observations: `{report.get('ready_observation_count')}` / `{report.get('expected_observation_count')}`",
        f"- Missing waves: `{report.get('missing_waves')}`",
        f"- Smoke excluded: `{bool(report.get('smoke_wave_excluded'))}`",
        (
            "- Simultaneous conformal margin: "
            + (f"`{float(margin):.6f}`" if margin is not None else "`not available`")
        ),
        "",
    ]
    if report.get("wait_reasons"):
        lines.extend(["## Wait reasons", ""])
        for issue in report["wait_reasons"]:
            lines.append(
                f"- `{issue.get('code')}` wave={issue.get('wave')}: "
                f"{issue.get('detail')}"
            )
        lines.append("")
    if report.get("validation_errors"):
        lines.extend(["## Validation errors", ""])
        for issue in report["validation_errors"]:
            lines.append(
                f"- `{issue.get('code')}` wave={issue.get('wave')} "
                f"cell={issue.get('cell')}: {issue.get('detail')}"
            )
        lines.append("")
    rows = certificate.get("rows") or []
    if rows:
        lines.extend(
            [
                "## Hardware-local certificate",
                "",
                "| workload environment | profile | frozen drain (s) | upper drain (s) | holdout drain (s) | upper mean JCT (s) | lower service | covered |",
                "|---|---:|---:|---:|---:|---:|---:|:---:|",
            ]
        )
        for row in rows:
            lines.append(
                "| `{workload_env}` | {profile} | {base:.3f} | {upper:.3f} | "
                "{actual:.3f} | {mean_upper:.3f} | {lower:.6f} | {covered} |".format(
                    workload_env=row["workload_env"],
                    profile=row["profile"],
                    base=row["frozen_base_drain_completion_s"],
                    upper=row["simultaneous_upper_drain_completion_s"],
                    actual=row["holdout_actual_drain_completion_s"],
                    mean_upper=row["simultaneous_upper_mean_task_jct_s"],
                    lower=row["lower_service_units_per_s"],
                    covered=(
                        "yes"
                        if row["lower_service_valid_on_holdout"]
                        and row["mean_task_jct_holdout_covered"]
                        else "no"
                    ),
                )
            )
        lines.append("")
    lines.extend([str(report.get("claim_boundary") or ""), ""])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", choices=sorted(NODE_SPECS), required=True)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--alpha", type=float, default=0.10)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    output = args.output or (
        args.artifact_root
        / f"critical_gpu_stochastic_lcb_gate_v5_{args.node}_{CAMPAIGN_DATE}.json"
    )
    markdown = args.markdown_output or output.with_suffix(".md")
    report = build_critical_gpu_stochastic_lcb_gate(
        node=args.node,
        artifact_root=args.artifact_root,
        miscoverage_alpha=args.alpha,
    )
    write_gate_outputs(report, json_path=output, markdown_path=markdown)
    print(
        json.dumps(
            {
                "status": report["status"],
                "pass": report["pass"],
                "output": str(output),
                "markdown_output": str(markdown),
                "missing_waves": report["missing_waves"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if report["pass"]:
        return 0
    return 3 if str(report["status"]).startswith("WAIT") else 2


if __name__ == "__main__":
    raise SystemExit(main())
