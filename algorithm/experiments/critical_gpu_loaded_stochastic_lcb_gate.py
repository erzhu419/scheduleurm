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
)


EXPECTED_WAVES = tuple(ALL_WAVES)
RESOURCE_STATE = "mixed_colocation"
MIS_COVERAGE_ALPHA = 0.10
REPO_ROOT = Path(__file__).resolve().parents[2]


def discover_campaign_paths(
    *, node: str, artifact_root: Path = ARTIFACT_ROOT
) -> dict[int, Path]:
    paths: dict[int, Path] = {}
    for wave in EXPECTED_WAVES:
        prefix = f"{CAMPAIGN_PREFIX}_{node}_r{wave:02d}_{CAMPAIGN_DATE}_code"
        matches = sorted(
            path
            for path in Path(artifact_root).glob(f"{prefix}*.json")
            if "_sel" not in path.name
        )
        if len(matches) == 1:
            paths[wave] = matches[0]
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
            "snapshot and rejects an assigned GPU already carrying undeclared "
            "load. It neither populates the legacy workload/profile index nor "
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
            "resident_measurement_total_units": int(scenario.resident_total_units),
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
    }


def _prelaunch_assigned_gpu_audit(
    *, row: Mapping[str, Any], node: str
) -> dict[str, Any]:
    """Verify that only the declared resident creates the loaded state.

    The completion campaign records an ``nvidia-smi`` snapshot before launching
    the resident and another after both tasks exit.  A row labelled with only a
    controlled resident is invalid when the selected physical GPU was already
    loaded by an undeclared process in the first snapshot.
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
    idle = bool(
        selected["used_mb"] <= GPU_IDLE_MEMORY_LIMIT_MB
        and selected["util_pct"] <= GPU_IDLE_UTIL_LIMIT_PCT
    )
    return {
        "audit_ready": True,
        "assigned_gpu_idle": idle,
        "gpu": gpu,
        "used_mb": selected["used_mb"],
        "util_pct": selected["util_pct"],
        "idle_memory_limit_mb": GPU_IDLE_MEMORY_LIMIT_MB,
        "idle_util_limit_pct": GPU_IDLE_UTIL_LIMIT_PCT,
        "diagnostics_path": str(path),
        "diagnostics_sha256": observed_sha256,
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
