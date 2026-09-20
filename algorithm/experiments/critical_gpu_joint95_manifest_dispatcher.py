"""Consume and dispatch the registered joint95 GPU collection manifest.

The registration manifest is immutable input.  This module never builds or
rewrites it.  ``submit-next`` adds at most one local control-plane task to the
Scheduleurm queue; that task acquires the existing measurement reservation and
then direct-launches exactly one registered visit on ``jtl110gpu2``.  The
holdout remains unavailable until ``release-holdout`` has observed nineteen
valid calibration terminals and frozen their registered rank-19 margin.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Iterator, Mapping, Sequence

from algorithm.experiments import critical_gpu_completion_campaign as empty_suite
from algorithm.experiments import critical_gpu_loaded_completion_campaign as loaded_suite
from algorithm.experiments import critical_gpu_loaded_stochastic_lcb_gate as loaded_gate
from algorithm.experiments import critical_gpu_stochastic_lcb_gate as empty_gate
from algorithm.experiments.critical_gpu_joint95_collection_runner import (
    CALIBRATION_WAVES,
    HOLDOUT_WAVE,
    PROTOCOL,
    local_readiness_snapshot,
    validate_manifest,
)
from algorithm.experiments.critical_gpu_reserved_campaign import (
    _active_scheduler_tasks,
    _wait_for_scheduler_ack,
)
from algorithm.experiments.theorem_measurement_reservation import (
    MeasurementReservation,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEDULER = REPO_ROOT / "skill" / "scheduler.py"
DEFAULT_NODE_STATUS = Path.home() / ".claude" / "scheduler" / "node_probe_cache.json"
DEFAULT_QUEUE_STATUS = Path.home() / ".claude" / "scheduler" / "queue.json"
RESULT_GATE = "critical_gpu_joint95_registered_visit"
RELEASE_GATE = "critical_gpu_joint95_holdout_release"
FAIL_STOP_FILENAME = "campaign_fail_stop.json"
ACTIVE_TASK_STATES = frozenset({"queued", "launching", "running"})
COLLECTION_SIGNATURE_PREFIX = "ScheduleurmBench/critical_gpu_joint95/"


class DispatchRefusal(RuntimeError):
    """A registered ordering, completion, or launch condition is not met."""


def _host_cpu_preflight_snapshot(node: str) -> dict[str, Any]:
    """Sample the same-user CPU load forbidden by the loaded interval audit."""

    shell = (
        'set -e; NPROC=$(getconf _NPROCESSORS_ONLN); '
        'printf "__NPROC__ %s\\n" "$NPROC"; '
        'ps -U "$(id -u)" -o pid=,pgid=,pcpu=,rss=,comm='
    )
    prefix = empty_suite.ARTIFACT_ROOT / (
        f"critical_gpu_joint95_host_preflight_{node}_{time.time_ns()}"
    )
    try:
        rc, out, err = empty_suite._run_remote_capture(
            node, shell, prefix, timeout_s=90
        )
    except Exception as exc:
        return {"ready": False, "error": f"{type(exc).__name__}: {exc}"}
    match = re.search(r"(?m)^__NPROC__\s+(\d+)\s*$", out)
    nproc = int(match.group(1)) if match else 0
    processes: list[dict[str, Any]] = []
    for line in out.splitlines():
        row = re.fullmatch(
            r"\s*(\d+)\s+(-?\d+)\s+([-+0-9.eE]+)\s+(\d+)\s+(\S+)\s*",
            line,
        )
        if row:
            processes.append({
                "pid": int(row.group(1)),
                "pgid": int(row.group(2)),
                "cpu_pct": float(row.group(3)),
                "rss_kb": int(row.group(4)),
                "command": row.group(5),
            })
    unapproved = [
        process for process in processes
        if process["command"] not in loaded_gate.HOST_MONITOR_ALLOWED_COMMANDS
    ]
    total_pct = sum(max(float(row["cpu_pct"]), 0.0) for row in unapproved)
    fraction = total_pct / (100.0 * nproc) if nproc > 0 else math.inf
    threshold = loaded_gate.MAX_UNAPPROVED_USER_CPU_FRACTION
    return {
        "ready": bool(rc == 0 and nproc > 0 and processes and fraction <= threshold),
        "returncode": int(rc),
        "host_nproc": nproc,
        "unapproved_cpu_fraction": fraction,
        "max_unapproved_cpu_fraction": threshold,
        "unapproved_cpu_processes": [
            row for row in unapproved
            if float(row["cpu_pct"]) / (100.0 * nproc) > threshold
        ] if nproc > 0 else [],
        "unapproved_cpu_process_count": len(unapproved),
        "stderr_tail": err[-1000:],
    }


def _wait_for_clean_node(
    *,
    node: str,
    lease: MeasurementReservation,
    poll_s: float,
    consecutive_samples: int,
) -> dict[str, Any]:
    """Require consecutive GPU, scheduler, and host-CPU idle samples."""

    required = max(1, int(consecutive_samples))
    clean = 0
    blocked_samples = 0
    last_blocking_host: dict[str, Any] | None = None
    latest: dict[str, Any] = {}
    while clean < required:
        lease.ensure_healthy()
        scheduler_tasks = _active_scheduler_tasks(node)
        try:
            idle = empty_suite._gpu_idle_snapshot(empty_suite.NODE_SPECS[node])
        except Exception as exc:
            idle = {"ready": False, "error": f"{type(exc).__name__}: {exc}"}
        host = _host_cpu_preflight_snapshot(node)
        latest = {
            "sampled_at": time.time(),
            "gpu_idle": idle,
            "active_scheduler_tasks": scheduler_tasks,
            "host_cpu": host,
        }
        if idle.get("ready") is True and not scheduler_tasks and host.get("ready") is True:
            clean += 1
        else:
            clean = 0
            blocked_samples += 1
            if host.get("ready") is not True:
                last_blocking_host = host
        print(
            f"[{node}] clean_guard={clean}/{required} "
            f"gpu_ready={idle.get('ready') is True} "
            f"active_tasks={len(scheduler_tasks)} "
            f"host_cpu_ready={host.get('ready') is True} "
            f"unapproved_cpu_fraction={host.get('unapproved_cpu_fraction')!r}",
            flush=True,
        )
        if clean < required:
            time.sleep(max(1.0, float(poll_s)))
    latest["blocked_sample_count"] = blocked_samples
    latest["last_blocking_host_cpu"] = last_blocking_host
    return latest


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DispatchRefusal(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DispatchRefusal(f"JSON root must be an object: {path}")
    return value


def load_registered_manifest(path: Path) -> dict[str, Any]:
    """Load a formal registration without invoking the manifest builder."""

    manifest = _read_object(Path(path))
    if manifest.get("schema_version") != 1:
        raise DispatchRefusal("registered manifest schema version differs")
    if manifest.get("gate") != "critical_gpu_joint95_collection_runner":
        raise DispatchRefusal("registered manifest gate differs")
    if manifest.get("registered") is not True:
        raise DispatchRefusal("manifest is not formally registered")
    if manifest.get("status") not in {
        "REGISTERED_READY_FOR_SEPARATE_LAUNCH",
        "REGISTERED_WAIT_READINESS",
    }:
        raise DispatchRefusal(
            f"manifest status is not registered: {manifest.get('status')!r}"
        )
    stored_validation = manifest.get("validation") or {}
    if not isinstance(stored_validation, Mapping) or stored_validation.get("pass") is not True:
        raise DispatchRefusal("registered manifest does not carry a passing validation")
    current = validate_manifest(manifest)
    if current.get("pass") is not True:
        raise DispatchRefusal(
            "registered manifest no longer validates: "
            + "; ".join(str(value) for value in current.get("errors") or [])
        )
    return manifest


def _work_item(manifest: Mapping[str, Any], wave: int) -> dict[str, Any]:
    rows = [
        dict(row)
        for row in manifest.get("joint_work_items") or []
        if int(row.get("joint_wave") or -1) == int(wave)
    ]
    if len(rows) != 1:
        raise DispatchRefusal(f"expected one registered work item for wave {wave}")
    return rows[0]


def _attempt_number(path: Path, joint_unit_id: str) -> int | None:
    match = re.fullmatch(
        re.escape(joint_unit_id) + r"_attempt(\d+)\.json",
        path.name,
    )
    return int(match.group(1)) if match else None


def _attempt_paths(results_dir: Path, joint_unit_id: str) -> list[Path]:
    root = Path(results_dir)
    if not root.is_dir():
        return []
    rows = []
    for path in root.glob(f"{joint_unit_id}_attempt*.json"):
        attempt = _attempt_number(path, joint_unit_id)
        if attempt is not None:
            rows.append((attempt, path))
    return [path for _, path in sorted(rows)]


def next_attempt_path(results_dir: Path, item: Mapping[str, Any]) -> Path:
    joint_unit_id = str(item["joint_unit_id"])
    attempts = [
        value
        for path in _attempt_paths(results_dir, joint_unit_id)
        if (value := _attempt_number(path, joint_unit_id)) is not None
    ]
    return Path(results_dir) / f"{joint_unit_id}_attempt{max(attempts, default=0) + 1:02d}.json"


def validate_success_result(
    payload: Mapping[str, Any],
    item: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the terminal and its nine scalar outcomes field by field."""

    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    expected_ids = list(item.get("scored_functional_ids") or [])
    observations = list(payload.get("observations") or [])
    empty_rows = list(payload.get("empty_rows") or [])
    loaded_rows = list((payload.get("loaded_campaign") or {}).get("rows") or [])
    require(payload.get("gate") == RESULT_GATE, "result gate differs")
    require(payload.get("schema_version") == 1, "result schema version differs")
    require(payload.get("protocol") == PROTOCOL, "result protocol differs")
    require(payload.get("manifest_registered") is True, "result did not consume a registered manifest")
    require(
        payload.get("joint_unit_id") == item.get("joint_unit_id"),
        "joint unit identity differs",
    )
    require(
        int(payload.get("joint_wave") or -1) == int(item.get("joint_wave") or -2),
        "joint wave differs",
    )
    require(payload.get("split_role") == item.get("split_role"), "split role differs")
    require(payload.get("status") == "PASS", "terminal status is not PASS")
    require(payload.get("pass") is True, "terminal pass flag is not true")
    require(payload.get("terminal") is True, "result is not terminal")
    require(payload.get("suite_order") == item.get("suite_order"), "suite order differs")
    require(
        payload.get("measurement_bundles_unchanged") is True,
        "measurement bundle changed during the visit",
    )
    interval_audits = payload.get("scheduler_interval_audits")
    if interval_audits is not None:
        require(
            isinstance(interval_audits, list)
            and len(interval_audits) == 9
            and all(
                isinstance(audit, Mapping)
                and audit.get("status") == "PASS"
                and int(audit.get("overlap_count") or 0) == 0
                for audit in interval_audits
            ),
            "all nine scheduler interval audits must pass",
        )
    require(len(observations) == 9, "result must contain nine observations")
    expected_empty_ids = list((item.get("empty_suite") or {}).get("service_cell_ids") or [])
    observed_empty_ids = [str(row.get("registered_service_cell_id") or "") for row in empty_rows]
    require(observed_empty_ids == expected_empty_ids, "empty terminal identities or order differ")
    require(
        len(empty_rows) == 7
        and all(
            row.get("ready") is True
            and row.get("status") == "READY"
            and not bool(row.get("capacity_boundary"))
            and int(row.get("wave") or -1) == int(item.get("joint_wave") or -2)
            and row.get("split_role") == item.get("split_role")
            for row in empty_rows
        ),
        "all seven empty rows must be successful terminals",
    )
    expected_loaded_ids = list((item.get("loaded_suite") or {}).get("scenario_ids") or [])
    observed_loaded_ids = [str(row.get("scenario_id") or "") for row in loaded_rows]
    require(observed_loaded_ids == expected_loaded_ids, "loaded terminal identities or order differ")
    require(
        len(loaded_rows) == 2
        and all(
            row.get("ready") is True
            and not bool(row.get("capacity_boundary"))
            and int(row.get("wave") or -1) == int(item.get("joint_wave") or -2)
            and row.get("split_role") == item.get("split_role")
            for row in loaded_rows
        ),
        "both loaded rows must be successful terminals",
    )
    observed_ids = [str(row.get("functional_id") or "") for row in observations]
    require(observed_ids == expected_ids, "functional identities or order differ")
    scores: list[float] = []
    for index, row in enumerate(observations):
        try:
            actual = float(row.get("actual_time_s"))
            predicted = float(row.get("frozen_predicted_time_s"))
            score = float(row.get("ratio_score"))
        except (TypeError, ValueError):
            errors.append(f"observation {index} has nonnumeric values")
            continue
        require(math.isfinite(actual) and actual > 0.0, f"observation {index} actual time is invalid")
        require(
            math.isfinite(predicted) and predicted > 0.0,
            f"observation {index} predictor is invalid",
        )
        expected_score = actual / predicted - 1.0 if predicted > 0.0 else math.nan
        require(
            math.isfinite(score)
            and math.isclose(score, expected_score, rel_tol=1e-12, abs_tol=1e-12),
            f"observation {index} ratio score differs",
        )
        if math.isfinite(score):
            scores.append(score)
    if len(scores) == 9:
        maximum = max(scores)
        try:
            reported = float(payload.get("joint_wave_score"))
        except (TypeError, ValueError):
            reported = math.nan
        require(
            math.isfinite(reported)
            and math.isclose(reported, maximum, rel_tol=1e-12, abs_tol=1e-12),
            "joint wave score is not the maximum of the nine ratio scores",
        )
    return {"pass": not errors, "errors": errors}


def _scan_results(
    manifest: Mapping[str, Any],
    results_dir: Path,
) -> dict[int, dict[str, Any]]:
    scans: dict[int, dict[str, Any]] = {}
    for item in manifest.get("joint_work_items") or []:
        wave = int(item["joint_wave"])
        attempts = []
        successes = []
        invalid_passes = []
        for path in _attempt_paths(results_dir, str(item["joint_unit_id"])):
            try:
                payload = _read_object(path)
            except DispatchRefusal as exc:
                attempts.append({"path": str(path), "status": "UNREADABLE", "error": str(exc)})
                continue
            audit = validate_success_result(payload, item)
            row = {
                "path": str(path),
                "status": payload.get("status"),
                "terminal": payload.get("terminal"),
                "pass": payload.get("pass"),
                "success_audit": audit,
            }
            attempts.append(row)
            if audit["pass"]:
                successes.append({"path": path, "payload": payload})
            elif payload.get("status") == "PASS" or payload.get("pass") is True:
                invalid_passes.append({"path": str(path), "errors": audit["errors"]})
        scans[wave] = {
            "item": dict(item),
            "attempts": attempts,
            "successes": successes,
            "invalid_passes": invalid_passes,
        }
    return scans


def _release_path(results_dir: Path) -> Path:
    return Path(results_dir) / "holdout_release.json"


def _fail_stop_path(results_dir: Path) -> Path:
    return Path(results_dir) / FAIL_STOP_FILENAME


def _campaign_fail_stop(results_dir: Path) -> dict[str, Any] | None:
    path = _fail_stop_path(results_dir)
    if not path.is_file():
        return None
    payload = _read_object(path)
    if payload.get("status") != "FAIL_STOP":
        return None
    return {"path": str(path), "payload": payload}


def _successful_calibrations(
    scans: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for wave in CALIBRATION_WAVES:
        successes = list((scans.get(wave) or {}).get("successes") or [])
        if len(successes) == 1:
            rows.append(successes[0])
    return rows


def _result_predictor_vector(payload: Mapping[str, Any]) -> list[tuple[str, float]]:
    return [
        (str(row.get("functional_id") or ""), float(row.get("frozen_predicted_time_s")))
        for row in payload.get("observations") or []
    ]


def _registered_predictor_vector(
    manifest: Mapping[str, Any],
    predictors: tuple[Mapping[str, float], Mapping[str, float]],
) -> list[tuple[str, float]]:
    empty_predictors, loaded_predictors = predictors
    rows = []
    for functional in manifest.get("scored_functionals") or []:
        source = str(functional["source"])
        identity = str(functional["identity"])
        value = (
            empty_predictors[identity]
            if source == "empty"
            else loaded_predictors[identity]
        )
        rows.append((str(functional["functional_id"]), float(value)))
    return rows


def validate_holdout_release(
    payload: Mapping[str, Any],
    manifest: Mapping[str, Any],
    scans: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    successes = _successful_calibrations(scans)
    expected_ids = [
        str(_work_item(manifest, wave)["joint_unit_id"])
        for wave in CALIBRATION_WAVES
    ]
    scores = [float(row["payload"]["joint_wave_score"]) for row in successes]
    rank = int((manifest.get("coverage_contract") or {}).get("finite_sample_rank") or 0)
    expected_margin = (
        max(0.0, sorted(scores)[rank - 1])
        if len(scores) == len(CALIBRATION_WAVES) and 1 <= rank <= len(scores)
        else math.nan
    )
    require(payload.get("schema_version") == 1, "release schema version differs")
    require(payload.get("gate") == RELEASE_GATE, "release gate differs")
    require(payload.get("protocol") == PROTOCOL, "release protocol differs")
    require(payload.get("status") == "HOLDOUT_RELEASED", "holdout is not released")
    require(payload.get("holdout_wave") == HOLDOUT_WAVE, "release holdout wave differs")
    require(payload.get("calibration_waves") == list(CALIBRATION_WAVES), "release waves differ")
    require(payload.get("calibration_joint_unit_ids") == expected_ids, "release unit identities differ")
    require(
        payload.get("calibration_result_paths")
        == [str(row["path"]) for row in successes],
        "release result paths differ",
    )
    require(
        payload.get("calibration_joint_wave_scores") == scores,
        "release wave scores differ",
    )
    require(len(successes) == len(CALIBRATION_WAVES), "nineteen successful calibration terminals are required")
    require(payload.get("finite_sample_rank") == rank == 19, "release rank must be 19")
    require(payload.get("calibration_margin_frozen") is True, "calibration margin is not frozen")
    try:
        margin = float(payload.get("frozen_margin"))
    except (TypeError, ValueError):
        margin = math.nan
    require(
        math.isfinite(margin)
        and math.isfinite(expected_margin)
        and math.isclose(margin, expected_margin, rel_tol=1e-12, abs_tol=1e-12),
        "frozen margin differs from the rank-19 calibration margin",
    )
    return {"pass": not errors, "errors": errors, "frozen_margin": margin}


def collection_status(
    manifest: Mapping[str, Any],
    results_dir: Path,
) -> dict[str, Any]:
    results_dir = Path(results_dir).resolve()
    scans = _scan_results(manifest, results_dir)
    fail_stop = _campaign_fail_stop(results_dir)
    if fail_stop is not None:
        reason = str(fail_stop["payload"].get("reason") or "campaign fail-stop sentinel is active")
        return {
            "status": "FAIL_STOP",
            "terminal": True,
            "errors": [f"campaign fail-stop sentinel is active: {reason}"],
            "fail_stop": fail_stop,
            "successful_calibration_waves": [
                wave for wave in CALIBRATION_WAVES if len(scans[wave]["successes"]) == 1
            ],
            "successful_calibration_count": sum(
                len(scans[wave]["successes"]) == 1 for wave in CALIBRATION_WAVES
            ),
            "holdout_success": len(scans[HOLDOUT_WAVE]["successes"]) == 1,
            "holdout_released": False,
            "predictor_anchor_wave": None,
            "release_path": str(_release_path(results_dir)),
            "release_audit": {"pass": False, "errors": ["campaign is fail-stopped"]},
            "next_wave": None,
            "waves": {
                str(wave): {
                    "attempt_count": len(scan["attempts"]),
                    "success_count": len(scan["successes"]),
                    "invalid_passes": scan["invalid_passes"],
                }
                for wave, scan in scans.items()
            },
        }
    errors: list[str] = []
    success_waves = [wave for wave in (*CALIBRATION_WAVES, HOLDOUT_WAVE) if len(scans[wave]["successes"]) == 1]
    for wave, scan in scans.items():
        if len(scan["successes"]) > 1:
            errors.append(f"wave {wave} has more than one successful terminal")
        if scan["invalid_passes"]:
            errors.append(f"wave {wave} has a PASS artifact that fails result validation")
    successful_rows = [
        scans[wave]["successes"][0]
        for wave in (*CALIBRATION_WAVES, HOLDOUT_WAVE)
        if len(scans[wave]["successes"]) == 1
    ]
    predictor_anchor = (
        _result_predictor_vector(successful_rows[0]["payload"])
        if successful_rows
        else []
    )
    for row in successful_rows[1:]:
        if _result_predictor_vector(row["payload"]) != predictor_anchor:
            errors.append("successful visits do not share one frozen predictor vector")
            break

    first_incomplete = next(
        (wave for wave in CALIBRATION_WAVES if wave not in success_waves),
        None,
    )
    if first_incomplete is not None:
        later = [wave for wave in CALIBRATION_WAVES if wave > first_incomplete and wave in success_waves]
        if later:
            errors.append(
                f"out-of-order calibration success after wave {first_incomplete}: {later}"
            )

    release_file = _release_path(results_dir)
    release_audit: dict[str, Any] = {"pass": False, "errors": ["release artifact is absent"]}
    if release_file.is_file():
        try:
            release_audit = validate_holdout_release(
                _read_object(release_file), manifest, scans
            )
        except DispatchRefusal as exc:
            release_audit = {"pass": False, "errors": [str(exc)]}
        if not release_audit["pass"]:
            errors.append("holdout release artifact fails validation")
    if HOLDOUT_WAVE in success_waves and not release_audit["pass"]:
        errors.append("holdout terminal exists without a valid explicit release")

    next_wave: int | None
    state: str
    if errors:
        next_wave = None
        state = "FAIL_RESULT_STATE"
    elif first_incomplete is not None:
        next_wave = first_incomplete
        state = "READY_CALIBRATION"
    elif not release_audit["pass"]:
        next_wave = None
        state = "HOLDOUT_LOCKED"
    elif HOLDOUT_WAVE not in success_waves:
        next_wave = HOLDOUT_WAVE
        state = "READY_HOLDOUT"
    else:
        next_wave = None
        state = "COMPLETE"
    return {
        "status": state,
        "errors": errors,
        "successful_calibration_waves": [wave for wave in CALIBRATION_WAVES if wave in success_waves],
        "successful_calibration_count": sum(wave in success_waves for wave in CALIBRATION_WAVES),
        "holdout_success": HOLDOUT_WAVE in success_waves,
        "holdout_released": bool(release_audit["pass"]),
        "predictor_anchor_wave": (
            int(successful_rows[0]["payload"]["joint_wave"])
            if successful_rows
            else None
        ),
        "release_path": str(release_file),
        "release_audit": release_audit,
        "next_wave": next_wave,
        "waves": {
            str(wave): {
                "attempt_count": len(scan["attempts"]),
                "success_count": len(scan["successes"]),
                "invalid_passes": scan["invalid_passes"],
            }
            for wave, scan in scans.items()
        },
    }


def release_holdout(
    *,
    manifest_path: Path,
    results_dir: Path,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    results_dir = Path(results_dir).resolve()
    manifest = load_registered_manifest(manifest_path)
    fail_stop = _campaign_fail_stop(results_dir)
    if fail_stop is not None:
        raise DispatchRefusal(
            f"campaign is fail-stopped by {fail_stop['path']}; holdout release is forbidden"
        )
    scans = _scan_results(manifest, results_dir)
    if any(scan["invalid_passes"] for scan in scans.values()):
        raise DispatchRefusal("a passing result artifact fails validation")
    successes = _successful_calibrations(scans)
    if len(successes) != len(CALIBRATION_WAVES):
        raise DispatchRefusal(
            "holdout remains locked: expected 19 successful calibration terminals, "
            f"found {len(successes)}"
        )
    if any(len(scans[wave]["successes"]) != 1 for wave in CALIBRATION_WAVES):
        raise DispatchRefusal("each calibration wave must have exactly one successful terminal")
    predictor_vectors = {
        tuple(_result_predictor_vector(row["payload"]))
        for row in successes
    }
    if len(predictor_vectors) != 1:
        raise DispatchRefusal("calibration terminals do not share one frozen predictor vector")
    if scans[HOLDOUT_WAVE]["attempts"]:
        raise DispatchRefusal("holdout attempts exist before explicit release")
    rank = int((manifest.get("coverage_contract") or {}).get("finite_sample_rank") or 0)
    scores = [float(row["payload"]["joint_wave_score"]) for row in successes]
    if rank != 19 or not 1 <= rank <= len(scores):
        raise DispatchRefusal("registered finite-sample rank is unavailable")
    release = {
        "schema_version": 1,
        "gate": RELEASE_GATE,
        "protocol": PROTOCOL,
        "status": "HOLDOUT_RELEASED",
        "released_at": time.time(),
        "holdout_wave": HOLDOUT_WAVE,
        "calibration_waves": list(CALIBRATION_WAVES),
        "calibration_joint_unit_ids": [
            str(_work_item(manifest, wave)["joint_unit_id"])
            for wave in CALIBRATION_WAVES
        ],
        "calibration_result_paths": [str(row["path"]) for row in successes],
        "calibration_joint_wave_scores": scores,
        "finite_sample_rank": rank,
        "calibration_margin_frozen": True,
        "frozen_margin": max(0.0, sorted(scores)[rank - 1]),
    }
    audit = validate_holdout_release(release, manifest, scans)
    if not audit["pass"]:
        raise DispatchRefusal("refusing invalid holdout release: " + "; ".join(audit["errors"]))
    path = _release_path(results_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(release, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return release


def _active_collection_tasks(queue_path: Path) -> list[dict[str, Any]]:
    if not Path(queue_path).is_file():
        return []
    queue = _read_object(queue_path)
    tasks: Any = queue.get("tasks") or []
    if isinstance(tasks, Mapping):
        tasks = list(tasks.values())
    return [
        {
            "id": row.get("id") or row.get("task_id"),
            "status": str(row.get("status") or row.get("state") or "").lower(),
            "signature": row.get("signature"),
            "description": row.get("description"),
            "node": row.get("node") or row.get("assigned_node"),
        }
        for row in tasks if isinstance(tasks, list)
        if isinstance(row, Mapping)
        and str(row.get("status") or row.get("state") or "").lower() in ACTIVE_TASK_STATES
        and str(row.get("signature") or "").startswith(COLLECTION_SIGNATURE_PREFIX)
    ]


def _finite_timestamp(value: Any) -> float | None:
    try:
        timestamp = float(value)
    except (TypeError, ValueError):
        return None
    return timestamp if math.isfinite(timestamp) and timestamp >= 0.0 else None


def _completed_measurement_child_labels(
    rows: Sequence[Mapping[str, Any]],
    *,
    node: str,
    measurement_run_id: str | None,
    measurement_started_at: float,
) -> dict[str, tuple[str, float, int | None]]:
    """Use completed measurement rows to identify their exact child labels.

    The watcher timestamps an auto-adopted process when it discovers it, which
    can happen after the direct-launch wrapper and cleanup have returned.  A
    prior row is eligible only when its immediate audit ended before this
    measurement began; the current row is eligible after its wrapper returns.
    """

    labels: dict[str, tuple[str, float, int | None]] = {}
    for row in rows:
        prior = row.get("joint95_scheduler_interval_audit")
        completed_at = (
            _finite_timestamp(prior.get("measurement_finished_at"))
            if isinstance(prior, Mapping)
            else None
        )
        transport = row.get("durable_transport")
        if isinstance(transport, Mapping) and row.get("scenario_id"):
            scenario_id = row.get("scenario_id")
            scenario = next(
                (item for item in loaded_suite.SCENARIOS if item.scenario_id == scenario_id),
                None,
            )
            transport_run_id = transport.get("run_id")
            resident_model = row.get("resident_completion_model")
            target_model = row.get("target_completion_model")
            try:
                wave = int(row.get("wave"))
                resident_pgid = int(row.get("resident_process_group_id"))
                target_pgid = int(row.get("target_process_group_id"))
                target_started = int(row.get("target_start_ns"))
                target_finished = int(row.get("target_end_ns"))
            except (TypeError, ValueError):
                continue
            if (
                scenario is None
                or row.get("node") != node
                or wave <= 0
                or resident_pgid <= 0
                or target_pgid <= 0
                or target_started <= 0
                or target_finished <= target_started
                or not isinstance(transport_run_id, str)
                or not transport_run_id.endswith(f"_loaded_{scenario_id}")
                or row.get("ready") is not True
                or row.get("remote_returncode") != 0
                or row.get("resident_returncode") != 0
                or row.get("target_returncode") != 0
                or transport.get("state") != "DONE"
                or transport.get("remote_returncode") != 0
                or not isinstance(resident_model, Mapping)
                or not isinstance(target_model, Mapping)
                or resident_model.get("natural_exit") is not True
                or target_model.get("natural_exit") is not True
            ):
                continue
            if transport_run_id != measurement_run_id and (
                completed_at is None or completed_at > measurement_started_at
            ):
                continue
            run_base = f"critical_gpu_loaded_{node}_{scenario_id}_r{wave:02d}"
            for role, index, pgid in (
                ("resident", 0, resident_pgid),
                ("target", 1, target_pgid),
            ):
                label = f"{run_base}_{role}_gpu{int(scenario.gpu)}_{index}"
                labels[label] = (
                    transport_run_id,
                    completed_at if completed_at is not None else measurement_started_at,
                    pgid,
                )
            continue
        run_id = row.get("run_id")
        summary = row.get("summary")
        probe = row.get("probe_result")
        completion = row.get("completion_audit")
        cleanup = row.get("remote_cleanup")
        if not isinstance(run_id, str) or not run_id:
            continue
        if not all(isinstance(value, Mapping) for value in (summary, probe, completion, cleanup)):
            continue
        if (
            row.get("ready") is not True
            or summary.get("run_id") != run_id
            or summary.get("measurement_valid") is not True
            or probe.get("run_id") != run_id
            or probe.get("pass") is not True
            or completion.get("ready") is not True
            or cleanup.get("ready") is not True
            or cleanup.get("returncode") != 0
        ):
            continue
        if run_id != measurement_run_id and (
            completed_at is None or completed_at > measurement_started_at
        ):
            continue
        children = summary.get("rows")
        if not isinstance(children, list) or not children:
            continue
        if (
            summary.get("returncode_valid_count") != len(children)
            or summary.get("returncode_accepted_count") != len(children)
        ):
            continue
        child_labels: list[str] = []
        for child in children:
            if not isinstance(child, Mapping):
                break
            label = child.get("run_name")
            model = child.get("completion_model")
            if (
                not isinstance(label, str)
                or not label.startswith(run_id + "_")
                or child.get("returncode") != 0
                or not isinstance(model, Mapping)
                or model.get("natural_exit") is not True
            ):
                break
            child_labels.append(label)
        else:
            if len(set(child_labels)) == len(child_labels):
                for label in child_labels:
                    labels[label] = (
                        run_id,
                        completed_at if completed_at is not None else measurement_started_at,
                        None,
                    )
    return labels


def _captured_child_label(command: Any) -> str | None:
    if not isinstance(command, str):
        return None
    try:
        argv = shlex.split(command)
    except ValueError:
        return None
    positions = [
        index for index, token in enumerate(argv)
        if token in {"--label", "--run_name"}
    ]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        return None
    return argv[positions[0] + 1]


def _scheduler_history_interval_audit(
    *,
    queue_path: Path,
    node: str,
    measurement_kind: str,
    measurement_id: str,
    started_at: float,
    finished_at: float,
    completed_child_rows: Sequence[Mapping[str, Any]] = (),
    measurement_run_id: str | None = None,
) -> dict[str, Any]:
    """Find scheduler tasks that ran on the target during one measurement.

    Direct-launched measurement children can later appear as auto-adopted queue
    tasks.  Only exact children proven complete by their measurement row are
    recognized; all other target-node queue tasks remain overlap candidates.
    """

    child_labels = _completed_measurement_child_labels(
        completed_child_rows,
        node=node,
        measurement_run_id=measurement_run_id,
        measurement_started_at=started_at,
    )
    queue = _read_object(Path(queue_path))
    tasks: Any = queue.get("tasks") or []
    if isinstance(tasks, Mapping):
        tasks = list(tasks.values())
    if not isinstance(tasks, list):
        raise DispatchRefusal(f"scheduler queue tasks must be a list or object: {queue_path}")
    overlaps: list[dict[str, Any]] = []
    recognized_children: list[dict[str, Any]] = []
    for row in tasks:
        if not isinstance(row, Mapping):
            continue
        task_node = str(
            row.get("node") or row.get("assigned_node") or row.get("last_node") or ""
        )
        if task_node != node:
            continue
        task_started = _finite_timestamp(
            row.get("started_at")
            or row.get("actual_started_at")
            or row.get("launching_started_at")
        )
        if task_started is None:
            continue
        task_finished = _finite_timestamp(
            row.get("finished_at") or row.get("backend_finished_at")
        )
        status = str(row.get("status") or row.get("state") or "").lower()
        if task_finished is None:
            if status not in ACTIVE_TASK_STATES:
                continue
            interval_end = finished_at
        else:
            interval_end = task_finished
        if task_started < finished_at and interval_end > started_at:
            label = _captured_child_label(row.get("cmd")) if row.get("auto_adopted") is True else None
            if label in child_labels:
                run_id, completed_at, child_pgid = child_labels[label]
                if child_pgid is not None:
                    try:
                        task_pgid = int(row.get("process_group"))
                    except (TypeError, ValueError):
                        task_pgid = None
                    if task_pgid != child_pgid:
                        label = None
            if label in child_labels:
                run_id, completed_at, _ = child_labels[label]
                recognized_children.append(
                    {
                        "id": row.get("id") or row.get("task_id"),
                        "run_id": run_id,
                        "child_label": label,
                        "measurement_completed_at": completed_at,
                    }
                )
                continue
            overlaps.append(
                {
                    "id": row.get("id") or row.get("task_id"),
                    "status": status,
                    "node": task_node,
                    "project": row.get("project"),
                    "signature": row.get("signature"),
                    "description": row.get("description"),
                    "started_at": task_started,
                    "finished_at": task_finished,
                    "active_at_audit": task_finished is None,
                }
            )
    return {
        "status": "FAIL_OVERLAP" if overlaps else "PASS",
        "target_node": node,
        "measurement_kind": measurement_kind,
        "measurement_id": measurement_id,
        "measurement_started_at": started_at,
        "measurement_finished_at": finished_at,
        "queue_path": str(Path(queue_path)),
        "overlap_count": len(overlaps),
        "overlapping_tasks": overlaps,
        "recognized_measurement_children": recognized_children,
        "target_queue_tasks_are_external": not bool(recognized_children),
    }


def build_submit_plan(
    *,
    manifest_path: Path,
    results_dir: Path,
    queue_path: Path = DEFAULT_QUEUE_STATUS,
    node_status_path: Path | None = DEFAULT_NODE_STATUS,
    scheduler_path: Path = SCHEDULER,
    python_bin: str = sys.executable,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    results_dir = Path(results_dir).resolve()
    manifest = load_registered_manifest(manifest_path)
    state = collection_status(manifest, results_dir)
    if state["errors"]:
        raise DispatchRefusal("result state is invalid: " + "; ".join(state["errors"]))
    active = _active_collection_tasks(queue_path)
    if active:
        raise DispatchRefusal(
            "one joint95 control task is already active: "
            + ", ".join(str(row.get("id")) for row in active)
        )
    wave = state.get("next_wave")
    readiness = local_readiness_snapshot(
        node=str(manifest["node"]),
        node_status_path=node_status_path if node_status_path and Path(node_status_path).is_file() else None,
        queue_status_path=queue_path if Path(queue_path).is_file() else None,
    )
    if wave is None:
        return {
            "status": state["status"],
            "submitted": False,
            "collection": state,
            "target_readiness": readiness,
            "scheduler_argv": None,
        }
    item = _work_item(manifest, int(wave))
    result_path = next_attempt_path(results_dir, item)
    worker_argv = [
        python_bin,
        "-m",
        "algorithm.experiments.critical_gpu_joint95_manifest_dispatcher",
        "execute-wave",
        "--manifest",
        str(manifest_path),
        "--results-dir",
        str(results_dir),
        "--wave",
        str(int(wave)),
        "--result",
        str(result_path),
        "--queue-status",
        str(Path(queue_path).resolve()),
        "--allow-launch",
    ]
    command = shlex.join(worker_argv) + " && echo DONE"
    signature = COLLECTION_SIGNATURE_PREFIX + str(item["joint_unit_id"])
    scheduler_argv = [
        python_bin,
        str(Path(scheduler_path).resolve()),
        "submit",
        "--description",
        f"joint95 registered {item['split_role']} visit wave {int(wave)}",
        "--cmd",
        command,
        "--cwd",
        str(REPO_ROOT),
        "--signature",
        signature,
        "--project",
        "ScheduleurmBench",
        "--vram",
        "0",
        "--ram-mb",
        "1024",
        "--cpu",
        "1",
        "--priority",
        "high",
        "--require-node",
        "local",
        "--wait-for-file",
        str(manifest_path),
    ]
    return {
        "status": "READY_TO_SUBMIT_ONE",
        "submitted": False,
        "single_step": True,
        "joint_wave": int(wave),
        "split_role": item["split_role"],
        "joint_unit_id": item["joint_unit_id"],
        "result_path": str(result_path),
        "collection": state,
        "target_readiness": readiness,
        "controller_placement": {
            "require_node": "local",
            "vram_mb": 0,
            "purpose": "hold target-node measurement reservation without occupying target GPU",
        },
        "worker_argv": worker_argv,
        "scheduler_argv": scheduler_argv,
    }


def submit_next(
    *,
    manifest_path: Path,
    results_dir: Path,
    queue_path: Path = DEFAULT_QUEUE_STATUS,
    node_status_path: Path | None = DEFAULT_NODE_STATUS,
    scheduler_path: Path = SCHEDULER,
    python_bin: str = sys.executable,
    dry_run: bool,
) -> dict[str, Any]:
    plan = build_submit_plan(
        manifest_path=manifest_path,
        results_dir=results_dir,
        queue_path=queue_path,
        node_status_path=node_status_path,
        scheduler_path=scheduler_path,
        python_bin=python_bin,
    )
    if plan.get("scheduler_argv") is None:
        return {**plan, "dry_run": bool(dry_run)}
    if dry_run:
        return {**plan, "status": "DRY_RUN", "dry_run": True}
    completed = subprocess.run(
        plan["scheduler_argv"],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    if completed.returncode != 0:
        raise DispatchRefusal(
            f"scheduler submit failed rc={completed.returncode}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    match = re.search(r"\bsubmitted\s+(t\d+)\b", completed.stdout or "")
    if match is None:
        raise DispatchRefusal(
            f"scheduler did not return a task id: {completed.stdout!r}"
        )
    return {
        **plan,
        "status": "SUBMITTED_ONE",
        "submitted": True,
        "dry_run": False,
        "task_id": match.group(1),
        "scheduler_stdout": completed.stdout,
    }


def _write_new_result(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(dict(report), stream, indent=2, sort_keys=True)
        stream.write("\n")


def _replace_result(path: Path, report: Mapping[str, Any]) -> None:
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    tmp = Path(raw_tmp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(dict(report), stream, indent=2, sort_keys=True)
            stream.write("\n")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def _registered_predictors(manifest: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
    history = manifest.get("historical_training_inputs") or {}
    empty_payload = _read_object(Path(str(history.get("empty_gate") or "")))
    loaded_payload = _read_object(Path(str(history.get("loaded_gate") or "")))
    empty = {
        str(row["service_cell_id"]): float(row["frozen_base_drain_completion_s"])
        for row in (empty_payload.get("certificate") or {}).get("rows") or []
        if row.get("service_cell_id")
    }
    loaded = {
        str(row["scenario_id"]): float(row["frozen_target_completion_s"])
        for row in (loaded_payload.get("certificate") or {}).get("rows") or []
        if row.get("scenario_id")
    }
    expected_empty = {str(row["service_cell_id"]) for row in manifest.get("empty_contract") or []}
    expected_loaded = {str(row["scenario_id"]) for row in manifest.get("loaded_contract") or []}
    if not expected_empty <= set(empty) or any(empty[key] <= 0.0 for key in expected_empty):
        raise DispatchRefusal("registered empty predictors are unavailable")
    if not expected_loaded <= set(loaded) or any(loaded[key] <= 0.0 for key in expected_loaded):
        raise DispatchRefusal("registered loaded predictors are unavailable")
    return (
        {key: empty[key] for key in expected_empty},
        {key: loaded[key] for key in expected_loaded},
    )


@contextmanager
def _continuation_wave_role(wave: int, role: str) -> Iterator[None]:
    """Permit one registered continuation wave without changing frozen files."""

    old_empty_gate_role = empty_gate.wave_role
    old_loaded_role = loaded_suite._split_role
    old_loaded_gate_role = loaded_gate._split_role

    def empty_role(value: int) -> str:
        return role if int(value) == int(wave) else old_empty_gate_role(value)

    def loaded_role(value: int) -> str:
        return role if int(value) == int(wave) else old_loaded_role(value)

    empty_gate.wave_role = empty_role
    loaded_suite._split_role = loaded_role
    loaded_gate._split_role = loaded_role
    try:
        yield
    finally:
        empty_gate.wave_role = old_empty_gate_role
        loaded_suite._split_role = old_loaded_role
        loaded_gate._split_role = old_loaded_gate_role


def _bundle_matches(manifest: Mapping[str, Any], suite_name: str) -> bool:
    registered = ((manifest.get("measurement_bundles") or {}).get(suite_name) or {}).get("manifest")
    current = (
        empty_suite.measurement_code_manifest()
        if suite_name == "empty"
        else loaded_suite.measurement_code_manifest()
    )
    return registered == current


def _run_empty_suite(
    *,
    manifest: Mapping[str, Any],
    item: Mapping[str, Any],
    lease: MeasurementReservation,
    poll_s: float,
    clean_samples: int,
    report: dict[str, Any],
    output: Path,
    queue_path: Path = DEFAULT_QUEUE_STATUS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    node = str(item["node"])
    wave = int(item["joint_wave"])
    role = str(item["split_role"])
    node_spec = empty_suite.NODE_SPECS[node]
    lease.ensure_healthy()
    environment = empty_suite._node_environment_gate(node_spec)
    report["empty_environment_gate"] = environment
    _replace_result(output, report)
    if environment.get("ready") is not True:
        raise DispatchRefusal("empty suite environment gate is not ready")

    contracts = list(manifest.get("empty_contract") or [])
    wanted = list((item.get("empty_suite") or {}).get("service_cell_ids") or [])
    by_id = {str(row["service_cell_id"]): dict(row) for row in contracts}
    if wanted != [str(row["service_cell_id"]) for row in contracts]:
        contracts = [by_id[key] for key in wanted]
    specs = {spec.workload_key: spec for spec in empty_suite.WORKLOAD_SPECS}
    base_cells = {
        (str(row["workload_key"]), int(row["profile"])): row
        for row in empty_suite.campaign_cells(node=node, wave=1)
    }
    raw_rows: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    for index, contract in enumerate(contracts):
        lease.ensure_healthy()
        clean = _wait_for_clean_node(
            node=node,
            lease=lease,
            poll_s=poll_s,
            consecutive_samples=clean_samples,
        )
        preflight = clean["gpu_idle"]
        key = (str(contract["workload_key"]), int(contract["profile"]))
        if key not in base_cells:
            raise DispatchRefusal(f"empty registered cell has no current suite cell: {key}")
        cell = {
            **dict(base_cells[key]),
            "cell_id": f"{item['joint_unit_id']}_empty_{index + 1:02d}",
            "wave": wave,
            "split_role": role,
        }
        measured_started_at = time.time()
        row = empty_suite._run_cell(
            campaign_id=f"{output.stem}_empty",
            node_spec=node_spec,
            spec=specs[key[0]],
            cell=cell,
            preflight=preflight,
            cleanup_remote_output=True,
            expected_code_sha256=str((item.get("empty_suite") or {})["measurement_code_sha256"]),
            measurement_manifest_builder=empty_suite.measurement_code_manifest,
            measurement_protocol=str((item.get("empty_suite") or {})["measurement_protocol"]),
        )
        measured_finished_at = time.time()
        interval_audit = _scheduler_history_interval_audit(
            queue_path=queue_path,
            node=node,
            measurement_kind="empty_cell",
            measurement_id=str(contract["service_cell_id"]),
            started_at=measured_started_at,
            finished_at=measured_finished_at,
            completed_child_rows=raw_rows + [row],
            measurement_run_id=str(row.get("run_id") or ""),
        )
        row["joint95_clean_preflight"] = clean
        row["joint95_scheduler_interval_audit"] = interval_audit
        row["registered_service_cell_id"] = contract["service_cell_id"]
        raw_rows.append(row)
        report["empty_rows"] = raw_rows
        report.setdefault("scheduler_interval_audits", []).append(interval_audit)
        _replace_result(output, report)
        if interval_audit["overlap_count"]:
            task_ids = [str(task.get("id")) for task in interval_audit["overlapping_tasks"]]
            raise DispatchRefusal(
                f"empty cell {contract['service_cell_id']} overlapped target-node "
                f"scheduler task(s): {', '.join(task_ids)}"
            )
        errors, observation = empty_gate._audit_measurement_row(
            row=row,
            expected=contract,
            wave=wave,
            expected_protocol=str((item.get("empty_suite") or {})["measurement_protocol"]),
        )
        if errors or observation is None:
            raise DispatchRefusal(
                f"empty cell {contract['service_cell_id']} failed terminal audit: {errors!r}"
            )
        observations.append(observation)
    return raw_rows, observations


def _run_loaded_suite(
    *,
    manifest: Mapping[str, Any],
    item: Mapping[str, Any],
    lease: MeasurementReservation,
    poll_s: float,
    clean_samples: int,
    report: dict[str, Any],
    output: Path,
    queue_path: Path = DEFAULT_QUEUE_STATUS,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    node = str(item["node"])
    wave = int(item["joint_wave"])
    scenario_ids = list((item.get("loaded_suite") or {}).get("scenario_ids") or [])
    scenario_index = {
        scenario.scenario_id: scenario for scenario in loaded_suite.SCENARIOS
    }
    if any(scenario_id not in scenario_index for scenario_id in scenario_ids):
        raise DispatchRefusal("a registered loaded scenario is absent from the current suite")
    selected = tuple(scenario_index[scenario_id] for scenario_id in scenario_ids)
    campaign_id = f"{output.stem}_loaded"
    deploy_dir = loaded_suite.RUN_ROOT / campaign_id / "deployment"
    deploy_dir.mkdir(parents=True, exist_ok=True)
    lease.ensure_healthy()
    loaded_suite._deploy_progress_wrapper(
        node,
        deploy_dir,
        cwd=str(REPO_ROOT),
        output_root=loaded_suite.NODE_SPECS[node].output_root,
    )
    rows: list[dict[str, Any]] = []
    for scenario, gpu in loaded_suite._scenario_gpu_assignments(
        selected,
        loaded_suite.NODE_SPECS[node].gpus,
        wave=wave,
    ):
        lease.ensure_healthy()
        clean = _wait_for_clean_node(
            node=node,
            lease=lease,
            poll_s=poll_s,
            consecutive_samples=clean_samples,
        )
        measured_started_at = time.time()
        row = loaded_suite._run_loaded_trajectory(
            node=node,
            wave=wave,
            campaign_id=campaign_id,
            scenario=scenario,
            gpu=gpu,
            cleanup_remote_output=True,
        )
        measured_finished_at = time.time()
        interval_audit = _scheduler_history_interval_audit(
            queue_path=queue_path,
            node=node,
            measurement_kind="loaded_scenario",
            measurement_id=str(scenario.scenario_id),
            started_at=measured_started_at,
            finished_at=measured_finished_at,
            completed_child_rows=list(report.get("empty_rows") or []) + rows + [row],
            measurement_run_id=str(
                row.get("run_id")
                or (row.get("durable_transport") or {}).get("run_id")
                or ""
            ),
        )
        row["joint95_scheduler_interval_audit"] = interval_audit
        row["joint95_clean_preflight"] = clean
        rows.append(row)
        report.setdefault("scheduler_interval_audits", []).append(interval_audit)
        report["loaded_campaign"] = {
            "gate": "critical_gpu_joint95_loaded_continuation",
            "node": node,
            "wave": wave,
            "split_role": item["split_role"],
            "scenario_ids": scenario_ids,
            "rows": rows,
            "status": "RUNNING",
            "pass": False,
        }
        _replace_result(output, report)
        if interval_audit["overlap_count"]:
            task_ids = [str(task.get("id")) for task in interval_audit["overlapping_tasks"]]
            raise DispatchRefusal(
                f"loaded scenario {scenario.scenario_id} overlapped target-node "
                f"scheduler task(s): {', '.join(task_ids)}"
            )
    campaign = {
        "gate": "critical_gpu_joint95_loaded_continuation",
        "node": node,
        "wave": wave,
        "split_role": item["split_role"],
        "scenario_ids": scenario_ids,
        "rows": rows,
        "status": "PASS" if len(rows) == len(scenario_ids) else "INCOMPLETE",
        "pass": len(rows) == len(scenario_ids),
        "legacy_pre_registered_split_not_reused": True,
    }
    report["loaded_campaign"] = campaign
    _replace_result(output, report)
    indexed = {str(row.get("scenario_id") or ""): row for row in rows}
    contracts = {str(row["scenario_id"]): row for row in manifest.get("loaded_contract") or []}
    if list(indexed) != scenario_ids:
        raise DispatchRefusal(
            f"loaded campaign returned identities {list(indexed)!r}, expected {scenario_ids!r}"
        )
    observations = []
    for scenario_id in scenario_ids:
        errors, observation = loaded_gate._audit_row(
            row=indexed[scenario_id],
            expected=contracts[scenario_id],
            wave=wave,
        )
        if errors or observation is None:
            raise DispatchRefusal(
                f"loaded scenario {scenario_id} failed terminal audit: {errors!r}"
            )
        observations.append(observation)
    return campaign, observations


def _scored_observations(
    *,
    manifest: Mapping[str, Any],
    empty_observations: Sequence[Mapping[str, Any]],
    loaded_observations: Sequence[Mapping[str, Any]],
    predictors: tuple[Mapping[str, float], Mapping[str, float]],
) -> list[dict[str, Any]]:
    empty_predictors, loaded_predictors = predictors
    empty_actual = {
        str(row["service_cell_id"]): float(row["drain_completion_s"])
        for row in empty_observations
    }
    loaded_actual = {
        str(row["scenario_id"]): float(row["target_canonical_completion_s"])
        for row in loaded_observations
    }
    rows = []
    for functional in manifest.get("scored_functionals") or []:
        source = str(functional["source"])
        identity = str(functional["identity"])
        if source == "empty":
            actual = empty_actual[identity]
            predicted = empty_predictors[identity]
        elif source == "loaded":
            actual = loaded_actual[identity]
            predicted = loaded_predictors[identity]
        else:
            raise DispatchRefusal(f"unknown registered functional source {source!r}")
        rows.append(
            {
                "functional_id": functional["functional_id"],
                "source": source,
                "identity": identity,
                "functional": functional["functional"],
                "actual_time_s": actual,
                "frozen_predicted_time_s": predicted,
                "ratio_score": actual / predicted - 1.0,
            }
        )
    return rows


def execute_wave(
    *,
    manifest_path: Path,
    results_dir: Path,
    wave: int,
    output: Path,
    allow_launch: bool,
    poll_s: float = 60.0,
    clean_samples: int = 1,
    ttl_s: float = 900.0,
    queue_path: Path = DEFAULT_QUEUE_STATUS,
) -> dict[str, Any]:
    if not allow_launch:
        raise DispatchRefusal("execute-wave requires explicit --allow-launch")
    manifest_path = Path(manifest_path).resolve()
    results_dir = Path(results_dir).resolve()
    output = Path(output).resolve()
    if output.parent != results_dir:
        raise DispatchRefusal("result path must be directly inside results-dir")
    manifest = load_registered_manifest(manifest_path)
    state = collection_status(manifest, results_dir)
    if state["errors"]:
        raise DispatchRefusal("result state is invalid: " + "; ".join(state["errors"]))
    if state.get("next_wave") != int(wave):
        raise DispatchRefusal(
            f"wave {wave} is not the next registered visit; next={state.get('next_wave')!r}, "
            f"status={state['status']}"
        )
    item = _work_item(manifest, int(wave))
    if output != next_attempt_path(results_dir, item).resolve():
        raise DispatchRefusal("result path is not the next attempt path for this joint identity")
    predictors = _registered_predictors(manifest)
    registered_predictor_vector = _registered_predictor_vector(manifest, predictors)
    if int(wave) != CALIBRATION_WAVES[0]:
        prior_scans = _scan_results(manifest, results_dir)
        anchor_successes = prior_scans[CALIBRATION_WAVES[0]]["successes"]
        if len(anchor_successes) != 1:
            raise DispatchRefusal("wave14 predictor anchor is unavailable")
        if _result_predictor_vector(anchor_successes[0]["payload"]) != registered_predictor_vector:
            raise DispatchRefusal(
                "current predictor values differ from the successful wave14 anchor"
            )
    report: dict[str, Any] = {
        "schema_version": 1,
        "gate": RESULT_GATE,
        "protocol": PROTOCOL,
        "status": "STARTING",
        "pass": False,
        "terminal": False,
        "manifest_path": str(manifest_path),
        "manifest_registered": True,
        "joint_unit_id": item["joint_unit_id"],
        "joint_wave": int(wave),
        "split_role": item["split_role"],
        "started_at": time.time(),
        "suite_order": ["empty", "loaded"],
        "empty_rows": [],
        "loaded_campaign": None,
        "observations": [],
        "scheduler_interval_audits": [],
        "measurement_bundles_unchanged": False,
        "continuation_adapter": (
            "empty _run_cell uses manifest-derived current-suite cells; loaded "
            "_run_loaded_trajectory changes only the legacy split-role metadata hook "
            "for this manifest wave"
        ),
        "remote_csv_pulled": False,
        "remote_checkpoint_pulled": False,
    }
    _write_new_result(output, report)
    lease = MeasurementReservation(
        node=str(item["node"]),
        purpose=f"registered joint95 visit {item['joint_unit_id']}",
        ttl_s=ttl_s,
    )
    try:
        with lease as reservation:
            report["reservation"] = dict(reservation)
            report["status"] = "WAIT_SCHEDULER_ACK"
            _replace_result(output, report)
            _wait_for_scheduler_ack(
                node=str(item["node"]),
                reservation=reservation,
                lease=lease,
                poll_s=poll_s,
            )
            report["initial_clean_preflight"] = _wait_for_clean_node(
                node=str(item["node"]),
                lease=lease,
                poll_s=poll_s,
                consecutive_samples=clean_samples,
            )
            report["status"] = "RUNNING_EMPTY"
            _replace_result(output, report)
            with _continuation_wave_role(int(wave), str(item["split_role"])):
                _, empty_observations = _run_empty_suite(
                    manifest=manifest,
                    item=item,
                    lease=lease,
                    poll_s=poll_s,
                    clean_samples=clean_samples,
                    report=report,
                    output=output,
                    queue_path=queue_path,
                )
                report["status"] = "RUNNING_LOADED"
                _replace_result(output, report)
                _, loaded_observations = _run_loaded_suite(
                    manifest=manifest,
                    item=item,
                    lease=lease,
                    poll_s=poll_s,
                    clean_samples=clean_samples,
                    report=report,
                    output=output,
                    queue_path=queue_path,
                )
            lease.ensure_healthy()
            bundles_unchanged = bool(
                _bundle_matches(manifest, "empty")
                and _bundle_matches(manifest, "loaded")
            )
            if not bundles_unchanged:
                raise DispatchRefusal("measurement bundle changed during the registered visit")
            if _registered_predictors(manifest) != predictors:
                raise DispatchRefusal("registered predictor values changed during the visit")
            observations = _scored_observations(
                manifest=manifest,
                empty_observations=empty_observations,
                loaded_observations=loaded_observations,
                predictors=predictors,
            )
            report["observations"] = observations
            report["joint_wave_score"] = max(float(row["ratio_score"]) for row in observations)
            report["measurement_bundles_unchanged"] = True
            report["status"] = "PASS"
            report["pass"] = True
            report["terminal"] = True
            report["finished_at"] = time.time()
            audit = validate_success_result(report, item)
            if not audit["pass"]:
                raise DispatchRefusal("terminal self-audit failed: " + "; ".join(audit["errors"]))
            _replace_result(output, report)
            return report
    except BaseException as exc:
        report["status"] = "FAILED"
        report["pass"] = False
        report["terminal"] = True
        report["finished_at"] = time.time()
        report["error"] = f"{type(exc).__name__}: {exc}"
        _replace_result(output, report)
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m algorithm.experiments.critical_gpu_joint95_manifest_dispatcher"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def common(target: argparse.ArgumentParser) -> None:
        target.add_argument("--manifest", type=Path, required=True)
        target.add_argument("--results-dir", type=Path, required=True)

    status = sub.add_parser("status")
    common(status)
    status.add_argument("--node-status", type=Path, default=DEFAULT_NODE_STATUS)
    status.add_argument("--queue-status", type=Path, default=DEFAULT_QUEUE_STATUS)

    submit = sub.add_parser("submit-next")
    common(submit)
    submit.add_argument("--node-status", type=Path, default=DEFAULT_NODE_STATUS)
    submit.add_argument("--queue-status", type=Path, default=DEFAULT_QUEUE_STATUS)
    submit.add_argument("--scheduler", type=Path, default=SCHEDULER)
    submit.add_argument("--python-bin", default=sys.executable)
    submit.add_argument("--dry-run", action="store_true")

    release = sub.add_parser("release-holdout")
    common(release)

    execute = sub.add_parser("execute-wave")
    common(execute)
    execute.add_argument("--wave", type=int, required=True)
    execute.add_argument("--result", type=Path, required=True)
    execute.add_argument("--allow-launch", action="store_true")
    execute.add_argument("--poll-s", type=float, default=60.0)
    execute.add_argument("--clean-samples", type=int, default=1)
    execute.add_argument("--ttl-s", type=float, default=900.0)
    execute.add_argument("--queue-status", type=Path, default=DEFAULT_QUEUE_STATUS)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "status":
            manifest = load_registered_manifest(args.manifest)
            report = {
                "collection": collection_status(manifest, args.results_dir),
                "active_collection_tasks": _active_collection_tasks(args.queue_status),
                "target_readiness": local_readiness_snapshot(
                    node=str(manifest["node"]),
                    node_status_path=args.node_status if args.node_status.is_file() else None,
                    queue_status_path=args.queue_status if args.queue_status.is_file() else None,
                ),
            }
            code = 0 if not report["collection"]["errors"] else 2
        elif args.command == "submit-next":
            report = submit_next(
                manifest_path=args.manifest,
                results_dir=args.results_dir,
                queue_path=args.queue_status,
                node_status_path=args.node_status,
                scheduler_path=args.scheduler,
                python_bin=args.python_bin,
                dry_run=args.dry_run,
            )
            code = 0
        elif args.command == "release-holdout":
            report = release_holdout(
                manifest_path=args.manifest,
                results_dir=args.results_dir,
            )
            code = 0
        else:
            report = execute_wave(
                manifest_path=args.manifest,
                results_dir=args.results_dir,
                wave=args.wave,
                output=args.result,
                allow_launch=args.allow_launch,
                poll_s=args.poll_s,
                clean_samples=args.clean_samples,
                ttl_s=args.ttl_s,
                queue_path=args.queue_status,
            )
            code = 0 if report.get("pass") is True else 3
    except (DispatchRefusal, FileExistsError) as exc:
        report = {"status": "REFUSED", "error": f"{type(exc).__name__}: {exc}"}
        code = 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
