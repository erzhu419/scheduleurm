"""Dispatch the registered independent joint95 v5 campaign one visit at a time.

Campaign shape comes from the validated v5 manifest.  The measurement calls,
reservation, clean-node preflight, and scheduler-history interval audit reuse
the tested joint95 execution internals.  Result ordering, split-conformal
release, v1--v4 exclusion, first-attempt fail-stop, and scheduler submission are
implemented here for the independent v5 identities.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
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
from typing import Any, Iterable, Mapping, Sequence

from algorithm.experiments import critical_gpu_joint95_manifest_dispatcher as common
from algorithm.experiments import critical_gpu_joint95_v5_registration as registration
from skill.scheduler_node.reservations import active_reservations


REPO_ROOT = registration.REPO_ROOT
SCHEDULER = common.SCHEDULER
DEFAULT_NODE_STATUS = common.DEFAULT_NODE_STATUS
DEFAULT_QUEUE_STATUS = common.DEFAULT_QUEUE_STATUS
RESULT_GATE = "critical_gpu_joint95_v5_registered_visit"
RELEASE_GATE = "critical_gpu_joint95_v5_holdout_release"
FAIL_STOP_GATE = "critical_gpu_joint95_v5_campaign_fail_stop"
FAIL_STOP_FILENAME = "campaign_fail_stop.json"
CONTROLLER_STATE_DIRNAME = "controller_state"
CONTROLLER_STATE_GATE = "critical_gpu_joint95_v5_premeasurement_controller_state"
COLLECTION_SIGNATURE_PREFIX = "ScheduleurmBench/critical_gpu_joint95_v5/"
ACTIVE_TASK_STATES = frozenset({"queued", "launching", "running"})

DispatchRefusal = common.DispatchRefusal
MeasurementReservation = common.MeasurementReservation
_wait_for_scheduler_ack = common._wait_for_scheduler_ack
_wait_for_clean_node = common._wait_for_clean_node
_run_empty_suite = common._run_empty_suite
_run_loaded_suite = common._run_loaded_suite


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DispatchRefusal(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DispatchRefusal(f"JSON root must be an object: {path}")
    return value


def _campaign_shape(manifest: Mapping[str, Any]) -> tuple[tuple[int, ...], int]:
    coverage = manifest.get("coverage_contract") or {}
    calibration = tuple(
        int(value) for value in coverage.get("calibration_joint_waves") or []
    )
    try:
        holdout = int(coverage["holdout_joint_wave"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DispatchRefusal("v5 manifest has no valid holdout wave") from exc
    if not calibration or holdout in calibration:
        raise DispatchRefusal("v5 manifest has an invalid calibration/holdout split")
    item_waves = tuple(
        int(row.get("joint_wave")) for row in manifest.get("joint_work_items") or []
    )
    if item_waves != calibration + (holdout,):
        raise DispatchRefusal("v5 manifest work items differ from its registered split")
    return calibration, holdout


def _v1_manifest_for(report: Mapping[str, Any]) -> dict[str, Any]:
    source = str(
        (report.get("replacement_basis") or {}).get("v1_manifest_path") or ""
    )
    if not source:
        raise DispatchRefusal("v5 registration lacks its frozen v1 contract source")
    return _read_object(Path(source))


def _validate_registration(report: Mapping[str, Any]) -> dict[str, Any]:
    return registration.validate_registration(
        report,
        v1_manifest=_v1_manifest_for(report),
    )


def _load_registration(path: Path, *, require_authorized: bool) -> dict[str, Any]:
    manifest = _read_object(Path(path))
    if manifest.get("schema_version") != registration.SCHEMA_VERSION:
        raise DispatchRefusal("registered v5 manifest schema version differs")
    if manifest.get("gate") != registration.GATE:
        raise DispatchRefusal("registered v5 manifest gate differs")
    if manifest.get("registered") is not True:
        raise DispatchRefusal("v5 manifest is not formally registered")
    validation = _validate_registration(manifest)
    if validation.get("pass") is not True:
        raise DispatchRefusal(
            "registered v5 manifest no longer validates: "
            + "; ".join(str(value) for value in validation.get("errors") or [])
        )
    if require_authorized:
        if manifest.get("status") != "REGISTERED_READY_FOR_SEPARATE_LAUNCH":
            raise DispatchRefusal("v5 registration is not ready for separate launch")
        if manifest.get("launch_authorized") is not True:
            raise DispatchRefusal("v5 executor launch is not authorized")
        executor = manifest.get("executor_validation") or {}
        if (
            executor.get("status") != "PASS"
            or executor.get("module") != registration.V5_EXECUTOR_MODULE
            or executor.get("validated_scope")
            != list(registration.V5_EXECUTOR_TEST_SCOPE)
            or executor.get("requires_nine_scheduler_interval_audits") is not True
        ):
            raise DispatchRefusal("v5 executor validation record differs")
        if not common._bundle_matches(manifest, "empty") or not common._bundle_matches(
            manifest, "loaded"
        ):
            raise DispatchRefusal("v5 measurement bundle changed after registration")
        if (
            common.loaded_gate.MAX_UNAPPROVED_USER_CPU_FRACTION
            != registration.HOST_CPU_PREFLIGHT_CONTRACT["max_unapproved_user_cpu_fraction"]
        ):
            raise DispatchRefusal("v5 host CPU preflight threshold changed after registration")
    _campaign_shape(manifest)
    return manifest


def load_registered_manifest(path: Path) -> dict[str, Any]:
    """Load a validated and explicitly executor-authorized v5 manifest."""

    return _load_registration(path, require_authorized=True)


def _replace_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    tmp = Path(raw_tmp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(dict(payload), stream, indent=2, sort_keys=True)
            stream.write("\n")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def _executor_validation_record() -> dict[str, Any]:
    return {
        "status": "PASS",
        "module": registration.V5_EXECUTOR_MODULE,
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "validated_scope": list(registration.V5_EXECUTOR_TEST_SCOPE),
        "requires_nine_scheduler_interval_audits": True,
        "measurements_executed_before_authorization": False,
        "results_file_count_before_authorization": 0,
    }


def _empty_authorization_snapshot(results_dir: Path) -> dict[str, Any]:
    return {
        "results_directory": str(Path(results_dir).resolve()),
        "results_file_count": 0,
        "measurements_executed": False,
        "completed_joint_units": 0,
        "launch_performed": False,
    }


def authorize_registered_manifest(
    *, manifest_path: Path, results_dir: Path
) -> dict[str, Any]:
    """Authorize this tested adapter while the registered v5 cohort is empty."""

    manifest_path = Path(manifest_path).resolve()
    results_dir = Path(results_dir).resolve()
    manifest = _load_registration(manifest_path, require_authorized=False)
    if manifest.get("status") != "REGISTERED_AWAITING_V5_EXECUTOR_VALIDATION":
        raise DispatchRefusal("v5 manifest is not awaiting executor validation")
    if manifest.get("launch_authorized") is not False:
        raise DispatchRefusal("v5 manifest already carries launch authority")
    _assert_results_dir(manifest, results_dir)
    if results_dir.exists() and any(results_dir.iterdir()):
        raise DispatchRefusal("v5 executor authorization requires an empty results directory")
    if manifest.get("observations") or int(manifest.get("completed_joint_units", -1)) != 0:
        raise DispatchRefusal("v5 empirical state is not empty at executor authorization")

    authorized = copy.deepcopy(manifest)
    authorized["status"] = "REGISTERED_READY_FOR_SEPARATE_LAUNCH"
    authorized["launch_authorized"] = True
    authorized["executor_validation"] = _executor_validation_record()
    authorized["authorization_snapshot"] = _empty_authorization_snapshot(results_dir)
    authorized["validation"] = _validate_registration(authorized)
    if not authorized["validation"]["pass"]:
        raise DispatchRefusal(
            "refusing invalid executor authorization: "
            + "; ".join(authorized["validation"]["errors"])
        )
    _replace_json(manifest_path, authorized)
    return authorized


def refresh_registered_authorization(
    *, manifest_path: Path, results_dir: Path
) -> dict[str, Any]:
    """Refresh executor authorization only while the authorized cohort is empty."""

    manifest_path = Path(manifest_path).resolve()
    results_dir = Path(results_dir).resolve()
    manifest = _read_object(manifest_path)
    if (
        manifest.get("schema_version") != registration.SCHEMA_VERSION
        or manifest.get("gate") != registration.GATE
        or manifest.get("registered") is not True
    ):
        raise DispatchRefusal("v5 manifest is not a registered current-schema manifest")
    if (
        manifest.get("status") != "REGISTERED_READY_FOR_SEPARATE_LAUNCH"
        or manifest.get("launch_authorized") is not True
    ):
        raise DispatchRefusal("v5 manifest does not already carry executor authorization")
    _assert_results_dir(manifest, results_dir)
    if not results_dir.is_dir() or any(results_dir.iterdir()):
        raise DispatchRefusal(
            "v5 authorization refresh requires the existing results directory to be empty"
        )
    if (
        manifest.get("observations")
        or int(manifest.get("completed_joint_units", -1)) != 0
        or manifest.get("measurements_executed") is not False
    ):
        raise DispatchRefusal("v5 empirical state is not empty at authorization refresh")

    refreshed = copy.deepcopy(manifest)
    refreshed["scheduler_interval_guard"] = (
        registration._expected_interval_guard_contract()
    )
    refreshed["host_cpu_preflight"] = copy.deepcopy(
        registration.HOST_CPU_PREFLIGHT_CONTRACT
    )
    refreshed["executor_validation"] = _executor_validation_record()
    refreshed["authorization_snapshot"] = _empty_authorization_snapshot(results_dir)
    refreshed["validation"] = _validate_registration(refreshed)
    if not refreshed["validation"]["pass"]:
        raise DispatchRefusal(
            "refusing invalid refreshed executor authorization: "
            + "; ".join(refreshed["validation"]["errors"])
        )
    _replace_json(manifest_path, refreshed)
    return refreshed


def _assert_results_dir(manifest: Mapping[str, Any], results_dir: Path) -> Path:
    expected = Path(
        str((manifest.get("results_contract") or {}).get("results_directory") or "")
    ).resolve()
    actual = Path(results_dir).resolve()
    if actual != expected:
        raise DispatchRefusal(
            f"results directory differs from v5 registration: {actual} != {expected}"
        )
    prior_dirs = {
        Path(registration.DEFAULT_V1_FAIL_STOP).resolve().parent,
        Path(registration.DEFAULT_V2_FAIL_STOP).resolve().parent,
        Path(registration.DEFAULT_V3_FAIL_STOP).resolve().parent,
        Path(registration.DEFAULT_V4_RESULTS_DIR).resolve(),
    }
    if actual in prior_dirs:
        raise DispatchRefusal("v5 results cannot use a v1/v2/v3/v4 results directory")
    return actual


def _work_item(manifest: Mapping[str, Any], wave: int) -> dict[str, Any]:
    rows = [
        dict(row)
        for row in manifest.get("joint_work_items") or []
        if int(row.get("joint_wave") or -1) == int(wave)
    ]
    if len(rows) != 1:
        raise DispatchRefusal(f"expected one registered v5 work item for wave {wave}")
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
    paths = _attempt_paths(results_dir, str(item["joint_unit_id"]))
    if paths:
        raise DispatchRefusal("v5 is first-attempt-only; a visit artifact already exists")
    return Path(results_dir) / f"{item['joint_unit_id']}_attempt01.json"


def _audit_pass(value: Any) -> bool:
    return bool(
        isinstance(value, Mapping)
        and value.get("status") == "PASS"
        and int(value.get("overlap_count") or 0) == 0
    )


def validate_success_result(
    payload: Mapping[str, Any], item: Mapping[str, Any]
) -> dict[str, Any]:
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    expected_ids = list(item.get("scored_functional_ids") or [])
    observations = list(payload.get("observations") or [])
    empty_rows = list(payload.get("empty_rows") or [])
    loaded_rows = list((payload.get("loaded_campaign") or {}).get("rows") or [])
    interval_audits = payload.get("scheduler_interval_audits")
    row_audits = [
        *(row.get("joint95_scheduler_interval_audit") for row in empty_rows),
        *(row.get("joint95_scheduler_interval_audit") for row in loaded_rows),
    ]

    require(payload.get("gate") == RESULT_GATE, "result gate differs")
    require(payload.get("schema_version") == 1, "result schema version differs")
    require(payload.get("protocol") == registration.PROTOCOL, "result protocol differs")
    require(payload.get("manifest_registered") is True, "result did not consume a registered manifest")
    require(payload.get("joint_unit_id") == item.get("joint_unit_id"), "joint unit identity differs")
    require(int(payload.get("joint_wave") or -1) == int(item.get("joint_wave") or -2), "joint wave differs")
    require(payload.get("split_role") == item.get("split_role"), "split role differs")
    require(payload.get("v1_joint_units_used") == 0, "v1 units entered the v5 result")
    require(payload.get("v2_joint_units_used") == 0, "v2 units entered the v5 result")
    require(payload.get("v3_joint_units_used") == 0, "v3 units entered the v5 result")
    require(payload.get("v4_joint_units_used") == 0, "v4 units entered the v5 result")
    require(payload.get("status") == "PASS", "terminal status is not PASS")
    require(payload.get("pass") is True, "terminal pass flag is not true")
    require(payload.get("terminal") is True, "result is not terminal")
    require(payload.get("suite_order") == item.get("suite_order"), "suite order differs")
    require(payload.get("measurement_bundles_unchanged") is True, "measurement bundle changed during the visit")
    require(
        isinstance(interval_audits, list)
        and len(interval_audits) == 9
        and all(_audit_pass(audit) for audit in interval_audits),
        "all nine scheduler interval audits must pass",
    )
    require(
        len(row_audits) == 9
        and all(_audit_pass(audit) for audit in row_audits)
        and row_audits == interval_audits,
        "row-level scheduler interval audits differ from the nine top-level audits",
    )
    if len(row_audits) == 9:
        require(
            [audit.get("measurement_kind") for audit in row_audits]
            == ["empty_cell"] * 7 + ["loaded_scenario"] * 2,
            "scheduler interval audit measurement kinds differ",
        )
        require(
            all(audit.get("target_node") == item.get("node") for audit in row_audits),
            "scheduler interval audit target node differs",
        )

    require(len(observations) == 9, "result must contain nine observations")
    expected_empty = list((item.get("empty_suite") or {}).get("service_cell_ids") or [])
    observed_empty = [
        str(row.get("registered_service_cell_id") or "") for row in empty_rows
    ]
    require(observed_empty == expected_empty, "empty terminal identities or order differ")
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
    expected_loaded = list((item.get("loaded_suite") or {}).get("scenario_ids") or [])
    observed_loaded = [str(row.get("scenario_id") or "") for row in loaded_rows]
    require(observed_loaded == expected_loaded, "loaded terminal identities or order differ")
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
        require(math.isfinite(predicted) and predicted > 0.0, f"observation {index} predictor is invalid")
        expected_score = actual / predicted - 1.0 if predicted > 0.0 else math.nan
        require(
            math.isfinite(score)
            and math.isclose(score, expected_score, rel_tol=1e-12, abs_tol=1e-12),
            f"observation {index} ratio score differs",
        )
        if math.isfinite(score):
            scores.append(score)
    if len(scores) == 9:
        try:
            reported = float(payload.get("joint_wave_score"))
        except (TypeError, ValueError):
            reported = math.nan
        require(
            math.isfinite(reported)
            and math.isclose(reported, max(scores), rel_tol=1e-12, abs_tol=1e-12),
            "joint wave score is not the maximum of the nine ratio scores",
        )
    return {"pass": not errors, "errors": errors}


def _refresh_scheduler_interval_audits(
    *,
    report: dict[str, Any],
    item: Mapping[str, Any],
    queue_path: Path,
) -> list[dict[str, Any]]:
    """Recheck all completed measurement intervals against the latest queue history.

    The immediate per-call audit remains the first guard.  This final refresh catches
    an external process that the scheduler watcher records only after a short
    measurement call has returned.
    """

    empty_rows = list(report.get("empty_rows") or [])
    loaded_rows = list((report.get("loaded_campaign") or {}).get("rows") or [])
    rows = [
        *(
            (row, "empty_cell", str(row.get("registered_service_cell_id") or ""))
            for row in empty_rows
        ),
        *(
            (row, "loaded_scenario", str(row.get("scenario_id") or ""))
            for row in loaded_rows
        ),
    ]
    if len(rows) != 9:
        raise DispatchRefusal("cannot refresh scheduler intervals before all nine measurements")

    completed_rows = [row for row, _, _ in rows]
    refreshed: list[dict[str, Any]] = []
    for row, kind, identity in rows:
        prior = row.get("joint95_scheduler_interval_audit")
        if not isinstance(prior, Mapping):
            raise DispatchRefusal(f"{kind} {identity} lacks its immediate interval audit")
        started_at = common._finite_timestamp(prior.get("measurement_started_at"))
        finished_at = common._finite_timestamp(prior.get("measurement_finished_at"))
        if started_at is None or finished_at is None or finished_at < started_at:
            raise DispatchRefusal(f"{kind} {identity} has an invalid measurement interval")
        audit = common._scheduler_history_interval_audit(
            queue_path=queue_path,
            node=str(item["node"]),
            measurement_kind=kind,
            measurement_id=identity,
            started_at=started_at,
            finished_at=finished_at,
            completed_child_rows=completed_rows,
            measurement_run_id=str(
                row.get("run_id") or (row.get("durable_transport") or {}).get("run_id") or ""
            ),
        )
        row["joint95_scheduler_interval_audit"] = audit
        refreshed.append(audit)
    report["scheduler_interval_audits"] = refreshed
    return refreshed


def _scan_results(
    manifest: Mapping[str, Any], results_dir: Path
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
                attempts.append(
                    {"path": str(path), "status": "UNREADABLE", "error": str(exc)}
                )
                continue
            audit = validate_success_result(payload, item)
            expected_name = f"{item['joint_unit_id']}_attempt01.json"
            if path.name != expected_name:
                audit["pass"] = False
                audit["errors"].append(
                    f"first-attempt-only result must be named {expected_name}"
                )
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


def _foreign_result_paths(manifest: Mapping[str, Any], results_dir: Path) -> list[str]:
    root = Path(results_dir)
    if not root.is_dir():
        return []
    registered_ids = {
        str(row["joint_unit_id"]) for row in manifest.get("joint_work_items") or []
    }
    foreign = []
    for path in root.glob("*.json"):
        if path.name in {FAIL_STOP_FILENAME, "holdout_release.json"}:
            continue
        if not any(_attempt_number(path, identity) is not None for identity in registered_ids):
            foreign.append(str(path.resolve()))
    return sorted(foreign)


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
    return {"path": str(path.resolve()), "payload": payload}


def _result_predictor_vector(payload: Mapping[str, Any]) -> list[tuple[str, float]]:
    return [
        (str(row.get("functional_id") or ""), float(row.get("frozen_predicted_time_s")))
        for row in payload.get("observations") or []
    ]


def _successful_calibrations(
    scans: Mapping[int, Mapping[str, Any]], calibration_waves: Sequence[int]
) -> list[dict[str, Any]]:
    rows = []
    for wave in calibration_waves:
        successes = list((scans.get(int(wave)) or {}).get("successes") or [])
        if len(successes) == 1:
            rows.append(successes[0])
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

    calibration, holdout = _campaign_shape(manifest)
    successes = _successful_calibrations(scans, calibration)
    expected_ids = [str(_work_item(manifest, wave)["joint_unit_id"]) for wave in calibration]
    scores = [float(row["payload"]["joint_wave_score"]) for row in successes]
    rank = int((manifest.get("coverage_contract") or {}).get("finite_sample_rank") or 0)
    expected_margin = (
        max(0.0, sorted(scores)[rank - 1])
        if len(scores) == len(calibration) and 1 <= rank <= len(scores)
        else math.nan
    )
    require(payload.get("schema_version") == 1, "release schema version differs")
    require(payload.get("gate") == RELEASE_GATE, "release gate differs")
    require(payload.get("protocol") == registration.PROTOCOL, "release protocol differs")
    require(payload.get("status") == "HOLDOUT_RELEASED", "holdout is not released")
    require(payload.get("holdout_wave") == holdout, "release holdout wave differs")
    require(payload.get("calibration_waves") == list(calibration), "release waves differ")
    require(payload.get("calibration_joint_unit_ids") == expected_ids, "release unit identities differ")
    require(payload.get("calibration_result_paths") == [str(row["path"]) for row in successes], "release result paths differ")
    require(payload.get("calibration_joint_wave_scores") == scores, "release wave scores differ")
    require(len(successes) == len(calibration) == 19, "nineteen successful v5 calibration terminals are required")
    require(payload.get("finite_sample_rank") == rank == 19, "release rank must be 19")
    require(payload.get("calibration_margin_frozen") is True, "calibration margin is not frozen")
    require(payload.get("v1_joint_units_used") == 0, "v1 units entered the v5 release")
    require(payload.get("v2_joint_units_used") == 0, "v2 units entered the v5 release")
    require(payload.get("v3_joint_units_used") == 0, "v3 units entered the v5 release")
    require(payload.get("v4_joint_units_used") == 0, "v4 units entered the v5 release")
    try:
        margin = float(payload.get("frozen_margin"))
    except (TypeError, ValueError):
        margin = math.nan
    require(
        math.isfinite(margin)
        and math.isfinite(expected_margin)
        and math.isclose(margin, expected_margin, rel_tol=1e-12, abs_tol=1e-12),
        "frozen margin differs from the rank-19 v5 calibration margin",
    )
    return {"pass": not errors, "errors": errors, "frozen_margin": margin}


def collection_status(
    manifest: Mapping[str, Any], results_dir: Path
) -> dict[str, Any]:
    results_dir = _assert_results_dir(manifest, results_dir)
    calibration, holdout = _campaign_shape(manifest)
    scans = _scan_results(manifest, results_dir)
    fail_stop = _campaign_fail_stop(results_dir)
    if fail_stop is not None:
        reason = str(
            fail_stop["payload"].get("reason")
            or "v5 campaign fail-stop sentinel is active"
        )
        return {
            "status": "FAIL_STOP",
            "terminal": True,
            "errors": [f"v5 campaign fail-stop sentinel is active: {reason}"],
            "fail_stop": fail_stop,
            "successful_calibration_waves": [
                wave for wave in calibration if len(scans[wave]["successes"]) == 1
            ],
            "successful_calibration_count": sum(
                len(scans[wave]["successes"]) == 1 for wave in calibration
            ),
            "holdout_success": len(scans[holdout]["successes"]) == 1,
            "holdout_released": False,
            "predictor_anchor_wave": None,
            "release_path": str(_release_path(results_dir)),
            "release_audit": {"pass": False, "errors": ["campaign is fail-stopped"]},
            "next_wave": None,
            "waves": _scan_summary(scans),
        }

    errors: list[str] = []
    foreign = _foreign_result_paths(manifest, results_dir)
    if foreign:
        errors.append("v1/v2/v3/v4 or unregistered result artifacts are present in the v5 directory")
    success_waves = [
        wave
        for wave in (*calibration, holdout)
        if len(scans[wave]["successes"]) == 1
    ]
    for wave, scan in scans.items():
        if len(scan["attempts"]) > 1:
            errors.append(f"wave {wave} has more than one attempt in a first-attempt-only campaign")
        if scan["attempts"] and len(scan["successes"]) != 1:
            errors.append(f"wave {wave} has a non-passing first attempt")
        if len(scan["successes"]) > 1:
            errors.append(f"wave {wave} has more than one successful terminal")
        if scan["invalid_passes"]:
            errors.append(f"wave {wave} has a PASS artifact that fails v5 validation")

    successful_rows = [
        scans[wave]["successes"][0]
        for wave in (*calibration, holdout)
        if len(scans[wave]["successes"]) == 1
    ]
    predictor_anchor = (
        _result_predictor_vector(successful_rows[0]["payload"])
        if successful_rows
        else []
    )
    for row in successful_rows[1:]:
        if _result_predictor_vector(row["payload"]) != predictor_anchor:
            errors.append("successful v5 visits do not share one frozen predictor vector")
            break

    first_incomplete = next(
        (wave for wave in calibration if wave not in success_waves), None
    )
    if first_incomplete is not None:
        later = [
            wave
            for wave in calibration
            if wave > first_incomplete and wave in success_waves
        ]
        if later:
            errors.append(
                f"out-of-order v5 calibration success after wave {first_incomplete}: {later}"
            )

    release_file = _release_path(results_dir)
    release_audit: dict[str, Any] = {
        "pass": False,
        "errors": ["release artifact is absent"],
    }
    if release_file.is_file():
        try:
            release_audit = validate_holdout_release(
                _read_object(release_file), manifest, scans
            )
        except DispatchRefusal as exc:
            release_audit = {"pass": False, "errors": [str(exc)]}
        if not release_audit["pass"]:
            errors.append("v5 holdout release artifact fails validation")
    if holdout in success_waves and not release_audit["pass"]:
        errors.append("v5 holdout terminal exists without a valid explicit release")

    if errors:
        state = "FAIL_RESULT_STATE"
        next_wave = None
    elif first_incomplete is not None:
        state = "READY_CALIBRATION"
        next_wave = first_incomplete
    elif not release_audit["pass"]:
        state = "HOLDOUT_LOCKED"
        next_wave = None
    elif holdout not in success_waves:
        state = "READY_HOLDOUT"
        next_wave = holdout
    else:
        state = "COMPLETE"
        next_wave = None
    return {
        "status": state,
        "terminal": state in {"FAIL_RESULT_STATE", "COMPLETE"},
        "errors": errors,
        "foreign_result_paths": foreign,
        "first_attempt_only": True,
        "successful_calibration_waves": [
            wave for wave in calibration if wave in success_waves
        ],
        "successful_calibration_count": sum(
            wave in success_waves for wave in calibration
        ),
        "holdout_success": holdout in success_waves,
        "holdout_released": bool(release_audit["pass"]),
        "predictor_anchor_wave": (
            int(successful_rows[0]["payload"]["joint_wave"])
            if successful_rows
            else None
        ),
        "release_path": str(release_file),
        "release_audit": release_audit,
        "next_wave": next_wave,
        "waves": _scan_summary(scans),
    }


def _scan_summary(scans: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        str(wave): {
            "attempt_count": len(scan["attempts"]),
            "success_count": len(scan["successes"]),
            "invalid_passes": scan["invalid_passes"],
        }
        for wave, scan in scans.items()
    }


def release_holdout(*, manifest_path: Path, results_dir: Path) -> dict[str, Any]:
    manifest = load_registered_manifest(Path(manifest_path).resolve())
    results_dir = _assert_results_dir(manifest, results_dir)
    calibration, holdout = _campaign_shape(manifest)
    state = collection_status(manifest, results_dir)
    if state["status"] == "FAIL_STOP":
        raise DispatchRefusal("v5 campaign is fail-stopped; holdout release is forbidden")
    if state["errors"]:
        raise DispatchRefusal("v5 result state is invalid: " + "; ".join(state["errors"]))
    scans = _scan_results(manifest, results_dir)
    successes = _successful_calibrations(scans, calibration)
    if len(successes) != len(calibration):
        raise DispatchRefusal(
            "holdout remains locked: expected 19 successful v5 calibration terminals, "
            f"found {len(successes)}"
        )
    if any(len(scans[wave]["attempts"]) != 1 for wave in calibration):
        raise DispatchRefusal("each v5 calibration wave must have exactly one attempt")
    if scans[holdout]["attempts"]:
        raise DispatchRefusal("v5 holdout attempts exist before explicit release")
    predictor_vectors = {
        tuple(_result_predictor_vector(row["payload"])) for row in successes
    }
    if len(predictor_vectors) != 1:
        raise DispatchRefusal("v5 calibration terminals do not share one predictor vector")
    rank = int((manifest.get("coverage_contract") or {}).get("finite_sample_rank") or 0)
    scores = [float(row["payload"]["joint_wave_score"]) for row in successes]
    if rank != 19 or not 1 <= rank <= len(scores):
        raise DispatchRefusal("registered v5 finite-sample rank is unavailable")
    release = {
        "schema_version": 1,
        "gate": RELEASE_GATE,
        "protocol": registration.PROTOCOL,
        "status": "HOLDOUT_RELEASED",
        "released_at": time.time(),
        "holdout_wave": holdout,
        "calibration_waves": list(calibration),
        "calibration_joint_unit_ids": [
            str(_work_item(manifest, wave)["joint_unit_id"]) for wave in calibration
        ],
        "calibration_result_paths": [str(row["path"]) for row in successes],
        "calibration_joint_wave_scores": scores,
        "finite_sample_rank": rank,
        "calibration_margin_frozen": True,
        "frozen_margin": max(0.0, sorted(scores)[rank - 1]),
        "v1_joint_units_used": 0,
        "v2_joint_units_used": 0,
        "v3_joint_units_used": 0,
        "v4_joint_units_used": 0,
    }
    audit = validate_holdout_release(release, manifest, scans)
    if not audit["pass"]:
        raise DispatchRefusal(
            "refusing invalid v5 holdout release: " + "; ".join(audit["errors"])
        )
    path = _release_path(results_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(release, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return release


def _active_collection_tasks(queue_path: Path) -> list[dict[str, Any]]:
    if not Path(queue_path).is_file():
        return []
    queue = _read_object(Path(queue_path))
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
        for row in tasks
        if isinstance(tasks, list)
        if isinstance(row, Mapping)
        and str(row.get("status") or row.get("state") or "").lower()
        in ACTIVE_TASK_STATES
        and str(row.get("signature") or "").startswith(
            "ScheduleurmBench/critical_gpu_joint95"
        )
    ]


def _assert_active_node_reservation(queue_path: Path, results_dir: Path) -> None:
    """Require the campaign-owned scheduler claim before any v5 wave launch."""

    state = _read_object(Path(queue_path))
    expected_marker = str(
        Path(results_dir).resolve() / registration.RESERVATION_RELEASE_FILENAME
    )
    matching = [
        row for row in active_reservations(state)
        if row.get("nodes") == [registration.NODE]
        and row.get("signature_prefix") == COLLECTION_SIGNATURE_PREFIX
        and row.get("until_file") == expected_marker
    ]
    if len(matching) != 1:
        raise DispatchRefusal("joint95 v5 requires its active exclusive-node reservation")
    owner_id = matching[0].get("owner_task_id")
    tasks = state.get("tasks") or []
    if isinstance(tasks, Mapping):
        tasks = list(tasks.values())
    if not any(
        isinstance(task, Mapping)
        and task.get("id") == owner_id
        and task.get("signature") == registration.SUPERVISOR_SCHEDULER_SIGNATURE
        for task in tasks
    ):
        raise DispatchRefusal("joint95 v5 node reservation has no registered supervisor owner")


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
    manifest = load_registered_manifest(manifest_path)
    results_dir = _assert_results_dir(manifest, results_dir)
    state = collection_status(manifest, results_dir)
    if state["errors"]:
        raise DispatchRefusal("v5 result state is invalid: " + "; ".join(state["errors"]))
    active = _active_collection_tasks(queue_path)
    if active:
        raise DispatchRefusal(
            "one joint95 control task is already active: "
            + ", ".join(str(row.get("id")) for row in active)
        )
    readiness = common.local_readiness_snapshot(
        node=str(manifest["node"]),
        node_status_path=(
            node_status_path
            if node_status_path and Path(node_status_path).is_file()
            else None
        ),
        queue_status_path=queue_path if Path(queue_path).is_file() else None,
    )
    wave = state.get("next_wave")
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
        registration.V5_EXECUTOR_MODULE,
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
        f"joint95 v5 registered {item['split_role']} visit wave {int(wave)}",
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
        "--protect-from-auto-eviction",
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
    _assert_active_node_reservation(queue_path, results_dir)
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


def controller_state_path(results_dir: Path, item: Mapping[str, Any]) -> Path:
    return (
        Path(results_dir)
        / CONTROLLER_STATE_DIRNAME
        / f"{item['joint_unit_id']}.json"
    )


def _controller_state_is_restartable(
    payload: Mapping[str, Any], *, item: Mapping[str, Any], output: Path
) -> bool:
    return bool(
        payload.get("gate") == CONTROLLER_STATE_GATE
        and payload.get("protocol") == registration.PROTOCOL
        and payload.get("joint_unit_id") == item["joint_unit_id"]
        and payload.get("joint_wave") == item["joint_wave"]
        and payload.get("status")
        in {"STARTING", "WAIT_SCHEDULER_ACK", "INITIAL_CLEAN_PREFLIGHT"}
        and payload.get("formal_attempt_created") is False
        and payload.get("measurement_started") is False
        and payload.get("empty_rows") == []
        and payload.get("observations") == []
        and payload.get("scheduler_interval_audits") == []
        and not output.exists()
    )


def _start_or_resume_controller_state(
    *, results_dir: Path, item: Mapping[str, Any], manifest_path: Path, output: Path
) -> tuple[Path, dict[str, Any]]:
    """Create or safely resume the non-evidentiary premeasurement state."""

    path = controller_state_path(results_dir, item)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        payload = _read_object(path)
        if not _controller_state_is_restartable(payload, item=item, output=output):
            raise DispatchRefusal(
                "v5 controller cannot restart after a measurement attempt was consumed or became partial"
            )
        payload = dict(payload)
        payload["controller_restart_count"] = int(
            payload.get("controller_restart_count") or 0
        ) + 1
        payload["restarted_at"] = time.time()
        payload["status"] = "STARTING"
        _replace_json(path, payload)
        return path, payload
    payload = {
        "schema_version": 1,
        "gate": CONTROLLER_STATE_GATE,
        "protocol": registration.PROTOCOL,
        "status": "STARTING",
        "terminal": False,
        "pass": False,
        "manifest_path": str(manifest_path),
        "joint_unit_id": item["joint_unit_id"],
        "joint_wave": item["joint_wave"],
        "split_role": item["split_role"],
        "started_at": time.time(),
        "controller_restart_count": 0,
        "formal_attempt_created": False,
        "measurement_started": False,
        "empty_rows": [],
        "observations": [],
        "scheduler_interval_audits": [],
        "result_path": str(output),
        "restart_contract": (
            "Restart is allowed only before attempt01 exists and while rows, "
            "observations, interval audits, and measurement intervals are all empty."
        ),
    }
    _write_new_result(path, payload)
    return path, payload


def _write_premeasurement_fail_stop(
    *,
    results_dir: Path,
    item: Mapping[str, Any],
    control_path: Path,
    control: Mapping[str, Any],
    reason: str,
    terminal_controllers: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Write a validated terminal without fabricating a formal attempt."""

    if not _controller_state_is_restartable(
        control, item=item, output=Path(str(control["result_path"]))
    ):
        raise DispatchRefusal("premeasurement fail-stop requires an unconsumed controller state")
    payload = {
        "schema_version": 1,
        "gate": FAIL_STOP_GATE,
        "protocol": registration.PROTOCOL,
        "status": "FAIL_STOP",
        "terminal": True,
        "pass": False,
        "decision": "JOINT95_V5_PREMEASUREMENT_CONTROLLER_FAIL_STOP",
        "created_at": time.time(),
        "reason": str(reason),
        "failed_wave": item["joint_wave"],
        "failed_joint_unit_id": item["joint_unit_id"],
        "failed_attempt_path": None,
        "controller_state_path": str(control_path),
        "controller_state_status": control.get("status"),
        "terminal_controllers": [dict(row) for row in terminal_controllers],
        "formal_attempt_created": False,
        "measurements_started": False,
        "scheduler_interval_failures": [],
        "v1_joint_units_used": 0,
        "v2_joint_units_used": 0,
        "v3_joint_units_used": 0,
        "v4_joint_units_used": 0,
        "holdout_release_authorized": False,
        "replacement_attempt_authorized": False,
        "controller_restart_authorized_after_terminal": False,
        "claim_boundary": (
            "The controller ended before attempt01 or any measurement interval "
            "existed. The campaign stops with zero contribution from this wave."
        ),
    }
    path = _fail_stop_path(results_dir)
    if path.exists():
        return _read_object(path)
    _write_new_result(path, payload)
    terminal_control = dict(control)
    terminal_control.update(
        {
            "status": "PREMEASUREMENT_FAIL_STOP",
            "terminal": True,
            "failed_at": time.time(),
            "error": str(reason),
        }
    )
    _replace_json(control_path, terminal_control)
    return payload


def _write_fail_stop(
    *, manifest: Mapping[str, Any], results_dir: Path, report: Mapping[str, Any]
) -> dict[str, Any]:
    failed_audits = [
        dict(row)
        for row in report.get("scheduler_interval_audits") or []
        if isinstance(row, Mapping) and row.get("status") == "FAIL_OVERLAP"
    ]
    decision = (
        "JOINT95_V5_INTERVAL_OVERLAP_FAIL_STOP"
        if failed_audits
        else "JOINT95_V5_FIRST_ATTEMPT_FAIL_STOP"
    )
    payload = {
        "schema_version": 1,
        "gate": FAIL_STOP_GATE,
        "protocol": registration.PROTOCOL,
        "status": "FAIL_STOP",
        "terminal": True,
        "pass": False,
        "decision": decision,
        "created_at": time.time(),
        "reason": str(report.get("error") or "v5 first attempt did not pass"),
        "failed_wave": report.get("joint_wave"),
        "failed_joint_unit_id": report.get("joint_unit_id"),
        "failed_attempt_path": report.get("result_path"),
        "controller_state_path": report.get("controller_state_path"),
        "terminal_controllers": list(report.get("terminal_controllers") or []),
        "formal_attempt_created": True,
        "measurements_started": True,
        "scheduler_interval_failures": failed_audits,
        "v1_joint_units_used": 0,
        "v2_joint_units_used": 0,
        "v3_joint_units_used": 0,
        "v4_joint_units_used": 0,
        "holdout_release_authorized": False,
        "replacement_attempt_authorized": False,
        "claim_boundary": (
            "This terminal records a v5 execution failure. It is retained and "
            "does not create a calibration unit or an empirical capacity claim."
        ),
    }
    path = _fail_stop_path(results_dir)
    if path.exists():
        return _read_object(path)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return payload


def finalize_terminal_controller_failure(
    *,
    manifest: Mapping[str, Any],
    results_dir: Path,
    wave: int,
    terminal_controllers: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Convert a dead controller into one auditable campaign fail-stop."""

    results_dir = _assert_results_dir(manifest, results_dir)
    item = _work_item(manifest, int(wave))
    expected_signature = COLLECTION_SIGNATURE_PREFIX + str(item["joint_unit_id"])
    controllers = [
        dict(row)
        for row in terminal_controllers
        if row.get("signature") == expected_signature
        and str(row.get("status") or "").lower() in {"done", "failed", "cancelled"}
    ]
    if not controllers:
        raise DispatchRefusal("terminal controller evidence does not match the expected v5 wave")
    existing = _campaign_fail_stop(results_dir)
    if existing is not None:
        return dict(existing["payload"])
    attempt_path = results_dir / f"{item['joint_unit_id']}_attempt01.json"
    control_path = controller_state_path(results_dir, item)
    if not attempt_path.exists():
        if not control_path.is_file():
            raise DispatchRefusal("terminal controller has neither premeasurement state nor attempt01")
        control = _read_object(control_path)
        return _write_premeasurement_fail_stop(
            results_dir=results_dir,
            item=item,
            control_path=control_path,
            control=control,
            reason="scheduler controller became terminal before attempt01 was created",
            terminal_controllers=controllers,
        )

    report = _read_object(attempt_path)
    if validate_success_result(report, item).get("pass") is True:
        raise DispatchRefusal("successful v5 attempt cannot be finalized as controller failure")
    if (
        report.get("gate") != RESULT_GATE
        or report.get("protocol") != registration.PROTOCOL
        or report.get("joint_unit_id") != item["joint_unit_id"]
        or report.get("joint_wave") != item["joint_wave"]
        or report.get("status") not in {"RUNNING_EMPTY", "RUNNING_LOADED", "FAILED"}
        or report.get("pass") is not False
    ):
        raise DispatchRefusal("terminal controller attempt is not a recognized consumed v5 attempt")
    report = dict(report)
    report.update(
        {
            "status": "FAILED",
            "terminal": True,
            "pass": False,
            "finished_at": time.time(),
            "error": "scheduler controller became terminal after attempt01 was created",
            "terminal_controllers": controllers,
            "replacement_attempt_authorized": False,
        }
    )
    _replace_json(attempt_path, report)
    terminal = _write_fail_stop(
        manifest=manifest, results_dir=results_dir, report=report
    )
    if control_path.is_file():
        control = _read_object(control_path)
        control.update(
            {
                "status": "ATTEMPT_FAIL_STOP",
                "terminal": True,
                "pass": False,
                "finished_at": time.time(),
                "error": report["error"],
            }
        )
        _replace_json(control_path, control)
    return terminal


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
        raise DispatchRefusal("v5 execute-wave requires explicit --allow-launch")
    manifest_path = Path(manifest_path).resolve()
    manifest = load_registered_manifest(manifest_path)
    results_dir = _assert_results_dir(manifest, results_dir)
    _assert_active_node_reservation(queue_path, results_dir)
    output = Path(output).resolve()
    if output.parent != results_dir:
        raise DispatchRefusal("v5 result path must be directly inside results-dir")
    state = collection_status(manifest, results_dir)
    if state["errors"]:
        raise DispatchRefusal("v5 result state is invalid: " + "; ".join(state["errors"]))
    if state.get("next_wave") != int(wave):
        raise DispatchRefusal(
            f"wave {wave} is not the next registered v5 visit; "
            f"next={state.get('next_wave')!r}, status={state['status']}"
        )
    item = _work_item(manifest, int(wave))
    if output != next_attempt_path(results_dir, item).resolve():
        raise DispatchRefusal("v5 result path is not the registered first-attempt path")
    predictors = common._registered_predictors(manifest)
    registered_predictor_vector = common._registered_predictor_vector(
        manifest, predictors
    )
    calibration, _ = _campaign_shape(manifest)
    if int(wave) != calibration[0]:
        prior_scans = _scan_results(manifest, results_dir)
        anchor_successes = prior_scans[calibration[0]]["successes"]
        if len(anchor_successes) != 1:
            raise DispatchRefusal(
                f"wave{calibration[0]} v5 predictor anchor is unavailable"
            )
        if (
            _result_predictor_vector(anchor_successes[0]["payload"])
            != registered_predictor_vector
        ):
            raise DispatchRefusal(
                f"current predictor values differ from wave{calibration[0]} v5 anchor"
            )

    control_path, control = _start_or_resume_controller_state(
        results_dir=results_dir,
        item=item,
        manifest_path=manifest_path,
        output=output,
    )
    report: dict[str, Any] | None = None
    lease = MeasurementReservation(
        node=str(item["node"]),
        purpose=f"registered joint95 v5 visit {item['joint_unit_id']}",
        ttl_s=ttl_s,
    )
    try:
        with lease as reservation:
            control["reservation"] = dict(reservation)
            control["status"] = "WAIT_SCHEDULER_ACK"
            _replace_json(control_path, control)
            _wait_for_scheduler_ack(
                node=str(item["node"]),
                reservation=reservation,
                lease=lease,
                poll_s=poll_s,
            )
            control["status"] = "INITIAL_CLEAN_PREFLIGHT"
            _replace_json(control_path, control)
            initial_clean_preflight = _wait_for_clean_node(
                node=str(item["node"]),
                lease=lease,
                poll_s=poll_s,
                consecutive_samples=clean_samples,
            )
            report = {
                "schema_version": 1,
                "gate": RESULT_GATE,
                "protocol": registration.PROTOCOL,
                "status": "RUNNING_EMPTY",
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
                "v1_joint_units_used": 0,
                "v2_joint_units_used": 0,
                "v3_joint_units_used": 0,
                "v4_joint_units_used": 0,
                "first_attempt_only": True,
                "result_path": str(output),
                "controller_state_path": str(control_path),
                "initial_clean_preflight": initial_clean_preflight,
                "reservation": dict(reservation),
                "remote_csv_pulled": False,
                "remote_checkpoint_pulled": False,
            }
            _write_new_result(output, report)
            control["status"] = "FORMAL_ATTEMPT_CREATED"
            control["formal_attempt_created"] = True
            control["measurement_started"] = True
            control["formal_attempt_created_at"] = time.time()
            _replace_json(control_path, control)
            with common._continuation_wave_role(int(wave), str(item["split_role"])):
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
                common._replace_result(output, report)
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
                common._bundle_matches(manifest, "empty")
                and common._bundle_matches(manifest, "loaded")
            )
            if not bundles_unchanged:
                raise DispatchRefusal("v5 measurement bundle changed during the visit")
            if common._registered_predictors(manifest) != predictors:
                raise DispatchRefusal("v5 registered predictor values changed during the visit")
            observations = common._scored_observations(
                manifest=manifest,
                empty_observations=empty_observations,
                loaded_observations=loaded_observations,
                predictors=predictors,
            )
            report["observations"] = observations
            report["joint_wave_score"] = max(
                float(row["ratio_score"]) for row in observations
            )
            refreshed_audits = _refresh_scheduler_interval_audits(
                report=report,
                item=item,
                queue_path=queue_path,
            )
            overlap_ids = [
                str(task.get("id"))
                for audit_row in refreshed_audits
                for task in audit_row.get("overlapping_tasks") or []
            ]
            if overlap_ids:
                raise DispatchRefusal(
                    "final scheduler-history refresh found target-node overlap: "
                    + ", ".join(overlap_ids)
                )
            report["measurement_bundles_unchanged"] = True
            report["status"] = "PASS"
            report["pass"] = True
            report["terminal"] = True
            report["finished_at"] = time.time()
            audit = validate_success_result(report, item)
            if not audit["pass"]:
                raise DispatchRefusal(
                    "v5 terminal self-audit failed: " + "; ".join(audit["errors"])
                )
            common._replace_result(output, report)
            control["status"] = "COMPLETE"
            control["terminal"] = True
            control["pass"] = True
            control["finished_at"] = time.time()
            _replace_json(control_path, control)
            return report
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        if report is None:
            terminal = _write_premeasurement_fail_stop(
                results_dir=results_dir,
                item=item,
                control_path=control_path,
                control=control,
                reason=error,
            )
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            return terminal
        report["status"] = "FAILED"
        report["pass"] = False
        report["terminal"] = True
        report["finished_at"] = time.time()
        report["error"] = error
        common._replace_result(output, report)
        _write_fail_stop(manifest=manifest, results_dir=results_dir, report=report)
        control["status"] = "ATTEMPT_FAIL_STOP"
        control["terminal"] = True
        control["pass"] = False
        control["finished_at"] = time.time()
        control["error"] = error
        _replace_json(control_path, control)
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m algorithm.experiments.critical_gpu_joint95_v5_dispatcher"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def common_args(target: argparse.ArgumentParser) -> None:
        target.add_argument("--manifest", type=Path, required=True)
        target.add_argument("--results-dir", type=Path, required=True)

    authorize = sub.add_parser("authorize-registration")
    common_args(authorize)

    refresh = sub.add_parser("refresh-authorization")
    common_args(refresh)

    status = sub.add_parser("status")
    common_args(status)
    status.add_argument("--node-status", type=Path, default=DEFAULT_NODE_STATUS)
    status.add_argument("--queue-status", type=Path, default=DEFAULT_QUEUE_STATUS)

    submit = sub.add_parser("submit-next")
    common_args(submit)
    submit.add_argument("--node-status", type=Path, default=DEFAULT_NODE_STATUS)
    submit.add_argument("--queue-status", type=Path, default=DEFAULT_QUEUE_STATUS)
    submit.add_argument("--scheduler", type=Path, default=SCHEDULER)
    submit.add_argument("--python-bin", default=sys.executable)
    submit.add_argument("--dry-run", action="store_true")

    release = sub.add_parser("release-holdout")
    common_args(release)

    execute = sub.add_parser("execute-wave")
    common_args(execute)
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
        if args.command == "authorize-registration":
            report = authorize_registered_manifest(
                manifest_path=args.manifest, results_dir=args.results_dir
            )
            code = 0
        elif args.command == "refresh-authorization":
            report = refresh_registered_authorization(
                manifest_path=args.manifest, results_dir=args.results_dir
            )
            code = 0
        elif args.command == "status":
            manifest = load_registered_manifest(args.manifest)
            report = {
                "collection": collection_status(manifest, args.results_dir),
                "active_collection_tasks": _active_collection_tasks(args.queue_status),
                "target_readiness": common.local_readiness_snapshot(
                    node=str(manifest["node"]),
                    node_status_path=(
                        args.node_status if args.node_status.is_file() else None
                    ),
                    queue_status_path=(
                        args.queue_status if args.queue_status.is_file() else None
                    ),
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
                manifest_path=args.manifest, results_dir=args.results_dir
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
