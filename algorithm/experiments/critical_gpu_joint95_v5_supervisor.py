"""Deterministically supervise the registered joint95-v5 19+1 campaign.

The supervisor is a local control-plane loop.  It never performs a measurement:
it observes the registered collection state, releases the frozen holdout after
nineteen valid calibration units, and calls the existing one-wave dispatcher.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping

from algorithm.experiments import critical_gpu_joint95_v5_dispatcher as dispatcher
from algorithm.experiments import critical_gpu_joint95_v5_registration as campaign
from algorithm.experiments import critical_gpu_completion_campaign as empty_suite
from algorithm.experiments.critical_gpu_reserved_campaign import _active_scheduler_tasks


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
DEFAULT_REGISTRATION = ARTIFACTS / (
    "critical_gpu_joint95_v5_supervisor_registration_v1_20260919.json"
)
GATE = "critical_gpu_joint95_v5_supervisor"
PROTOCOL = "critical_gpu_joint95_v5_deterministic_supervisor_v1"
REGISTRATION_STATUS = "REGISTERED_READY_FOR_LOCAL_CONTROL"
SUPERVISOR_SCHEDULER_SIGNATURE = campaign.SUPERVISOR_SCHEDULER_SIGNATURE
RESERVATION_RELEASE_FILENAME = campaign.RESERVATION_RELEASE_FILENAME
KNOWN_COLLECTION_STATES = frozenset(
    {
        "READY_CALIBRATION",
        "HOLDOUT_LOCKED",
        "READY_HOLDOUT",
        "COMPLETE",
        "FAIL_RESULT_STATE",
        "FAIL_STOP",
    }
)
TERMINAL_CONTROLLER_STATES = frozenset({"done", "failed", "cancelled"})
ARCHIVE_SCAN_BLOCK_BYTES = 64 * 1024
ARCHIVE_SCAN_MAX_BYTES = 32 * 1024 * 1024
ARCHIVE_SCAN_MAX_LINES = 4096
IN_FLIGHT_RESULT_STATES = frozenset(
    {"RUNNING_EMPTY", "RUNNING_LOADED"}
)


class SupervisorRefusal(RuntimeError):
    """The supervisor registration or collection state is unusable."""


def _active_v5_collection_tasks(queue_path: Path) -> list[dict[str, Any]]:
    """Return only active controllers in this campaign's exact v5 namespace."""
    return [
        row
        for row in dispatcher._active_collection_tasks(queue_path)
        if str(row.get("signature") or "").startswith(
            dispatcher.COLLECTION_SIGNATURE_PREFIX
        )
    ]


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SupervisorRefusal(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SupervisorRefusal(f"JSON root must be an object: {path}")
    return value


def _write_new(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(dict(payload), stream, indent=2, sort_keys=True)
        stream.write("\n")


def _scheduler_placement(results_dir: Path) -> dict[str, Any]:
    return {
        "require_node": "local",
        "vram_mb": 0,
        "ram_mb": 512,
        "cpu_cores": 1,
        "priority": "high",
        "signature": SUPERVISOR_SCHEDULER_SIGNATURE,
        "reserve_node": campaign.NODE,
        "reserve_signature_prefix": dispatcher.COLLECTION_SIGNATURE_PREFIX,
        "reserve_until_file": str(Path(results_dir).resolve() / RESERVATION_RELEASE_FILENAME),
        "release_after": ["COMPLETE", "VALIDATED_FAIL_STOP_WITH_QUIESCENT_CHILDREN"],
        "signature_is_outside_joint95_controller_prefix": True,
    }


def _campaign_contract(manifest: Mapping[str, Any]) -> dict[str, Any]:
    coverage = manifest.get("coverage_contract") or {}
    results = manifest.get("results_contract") or {}
    executor = manifest.get("executor_validation") or {}
    replacement = manifest.get("replacement_basis") or {}
    if (
        manifest.get("schema_version") != campaign.SCHEMA_VERSION
        or manifest.get("gate") != campaign.GATE
        or manifest.get("status") != "REGISTERED_READY_FOR_SEPARATE_LAUNCH"
        or manifest.get("registered") is not True
        or manifest.get("launch_authorized") is not True
        or manifest.get("protocol") != campaign.PROTOCOL
        or manifest.get("collection_prefix") != campaign.COLLECTION_PREFIX
        or manifest.get("node") != campaign.NODE
    ):
        raise SupervisorRefusal("joint95-v5 manifest identity or authority differs")
    if (
        coverage.get("calibration_joint_waves")
        != list(campaign.CALIBRATION_WAVES)
        or coverage.get("holdout_joint_wave") != campaign.HOLDOUT_WAVE
        or coverage.get("finite_sample_rank") != 19
    ):
        raise SupervisorRefusal("joint95-v5 19+1 split differs")
    if (
        executor.get("status") != "PASS"
        or executor.get("module") != campaign.V5_EXECUTOR_MODULE
        or executor.get("validated_scope")
        != list(campaign.V5_EXECUTOR_TEST_SCOPE)
    ):
        raise SupervisorRefusal("joint95-v5 executor validation differs")
    if any(
        int(replacement.get(field, -1)) != 0
        for field in (
            "v1_joint_units_eligible_for_v5_calibration",
            "v1_joint_units_eligible_for_v5_holdout",
            "v1_joint_scores_eligible_for_v5_margin",
            "v2_joint_units_eligible_for_v5_calibration",
            "v2_joint_units_eligible_for_v5_holdout",
            "v2_joint_scores_eligible_for_v5_margin",
            "v3_joint_units_eligible_for_v5_calibration",
            "v3_joint_units_eligible_for_v5_holdout",
            "v3_joint_scores_eligible_for_v5_margin",
            "v4_joint_units_eligible_for_v5_calibration",
            "v4_joint_units_eligible_for_v5_holdout",
            "v4_joint_scores_eligible_for_v5_margin",
        )
    ):
        raise SupervisorRefusal("v1/v2/v3/v4 evidence is eligible in the joint95-v5 manifest")
    return {
        "manifest_gate": manifest["gate"],
        "manifest_protocol": manifest["protocol"],
        "collection_prefix": manifest["collection_prefix"],
        "node": manifest["node"],
        "calibration_waves": list(campaign.CALIBRATION_WAVES),
        "holdout_wave": campaign.HOLDOUT_WAVE,
        "finite_sample_rank": 19,
        "results_directory": str(
            Path(str(results.get("results_directory") or "")).resolve()
        ),
        "v1_joint_units_eligible": 0,
        "v1_joint_scores_eligible": 0,
        "v2_joint_units_eligible": 0,
        "v2_joint_scores_eligible": 0,
        "v3_joint_units_eligible": 0,
        "v3_joint_scores_eligible": 0,
        "v4_joint_units_eligible": 0,
        "v4_joint_scores_eligible": 0,
    }


def build_registration(
    *,
    manifest_path: Path = campaign.DEFAULT_OUTPUT,
    results_dir: Path = campaign.DEFAULT_RESULTS_DIR,
    queue_path: Path = dispatcher.DEFAULT_QUEUE_STATUS,
    node_status_path: Path = dispatcher.DEFAULT_NODE_STATUS,
    scheduler_path: Path = dispatcher.SCHEDULER,
    python_bin: str = sys.executable,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    results_dir = Path(results_dir).resolve()
    manifest = dispatcher.load_registered_manifest(manifest_path)
    contract = _campaign_contract(manifest)
    if contract["results_directory"] != str(results_dir):
        raise SupervisorRefusal("supervisor results directory differs from v5 manifest")
    queue_path = Path(queue_path).resolve()
    entries = sorted(results_dir.iterdir()) if results_dir.is_dir() else []
    active = _active_v5_collection_tasks(queue_path)
    if active:
        raise SupervisorRefusal("supervisor registration requires no active v5 controller")
    if entries:
        raise SupervisorRefusal("supervisor registration requires an empty v5 results directory")
    state = dispatcher.collection_status(manifest, results_dir)
    if (
        state.get("status") != "READY_CALIBRATION"
        or state.get("next_wave") != 94
        or state.get("successful_calibration_count") != 0
        or state.get("holdout_success") is not False
        or state.get("errors")
    ):
        raise SupervisorRefusal("joint95-v5 campaign is not at its zero-result start")
    registration_snapshot = {
        "kind": "ZERO_RESULT_START",
        "collection_status_read_while_controller_active": False,
        "results_entry_count": 0,
        "successful_calibration_count": 0,
        "next_wave": 94,
        "scores_read_by_supervisor_registration": 0,
    }
    return {
        "schema_version": 1,
        "gate": GATE,
        "protocol": PROTOCOL,
        "status": REGISTRATION_STATUS,
        "registered": True,
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_paths": {
            "manifest": str(manifest_path),
            "results_directory": str(results_dir),
            "queue_status": str(queue_path),
            "queue_archive": str(queue_path.parent / "queue_archive.jsonl"),
            "node_status": str(Path(node_status_path).resolve()),
            "scheduler": str(Path(scheduler_path).resolve()),
            "python_bin": str(Path(python_bin).resolve()),
        },
        "campaign_contract": contract,
        "registration_snapshot": registration_snapshot,
        "control_policy": {
            "READY_CALIBRATION": "if no active joint95 controller, call existing submit_next once",
            "HOLDOUT_LOCKED": (
                "if exactly 19 calibrations succeeded and no controller is active, "
                "call frozen release_holdout once, then submit_next for wave 113"
            ),
            "READY_HOLDOUT": "if no active joint95 controller, call existing submit_next once",
            "ACTIVE_CONTROLLER": "wait without mutation",
            "REQUEUED_PREMEASUREMENT_CONTROLLER": (
                "the same controller may re-enter only while no attempt file, row, "
                "observation, interval audit, or measurement interval exists"
            ),
            "TERMINAL_CONTROLLER_WITHOUT_SUCCESS": (
                "freeze an unconsumed or partial state as a validated fail-stop, "
                "then release the node claim after children are quiescent"
            ),
            "FAIL_STOP": "terminate and release only after fail-stop validation and child quiescence",
            "FAIL_RESULT_STATE": "terminate without release, submission, or retry",
            "COMPLETE": "terminate successfully",
        },
        "decision_inputs": [
            "active joint95 controller presence",
            "registered collection state only when no joint95 controller is active",
            "successful calibration count only for the frozen 19-unit release boundary",
        ],
        "score_policy": (
            "The supervisor never reads a score for branching. The existing frozen "
            "release_holdout API alone computes registered rank 19."
        ),
        "execution_scope": {
            "measurement_calls": 0,
            "direct_scheduler_subprocess_calls": 0,
            "dispatcher_submit_next_only": True,
            "dispatcher_release_holdout_only": True,
            "terminal_controller_history_checked_before_every_submission": True,
            "active_controller_checked_before_collection_status": True,
            "active_controller_signature_prefix": dispatcher.COLLECTION_SIGNATURE_PREFIX,
            "retry_authorized": False,
            "premeasurement_same_controller_reentry_is_not_attempt_retry": True,
            "partial_or_consumed_attempt_retry_authorized": False,
        },
        "scheduler_placement": _scheduler_placement(results_dir),
        "prohibited": [
            "direct measurement execution",
            "submission while any joint95 controller is active",
            "retry after any failed or invalid attempt",
            "score-dependent selection outside the registered rank-19 release",
            "v1/v2/v3/v4 evidence reuse, server pulls, or paper edits",
        ],
    }


def register(output: Path = DEFAULT_REGISTRATION, **kwargs: Any) -> dict[str, Any]:
    output = Path(output)
    if output.exists():
        raise SupervisorRefusal(f"supervisor registration already exists: {output}")
    protocol = build_registration(**kwargs)
    _write_new(output, protocol)
    return protocol


def _load_registration(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = _read_object(Path(path))
    if (
        protocol.get("schema_version") != 1
        or protocol.get("gate") != GATE
        or protocol.get("protocol") != PROTOCOL
        or protocol.get("status") != REGISTRATION_STATUS
        or protocol.get("registered") is not True
    ):
        raise SupervisorRefusal("supervisor registration is invalid")
    paths = protocol.get("source_paths") or {}
    manifest = dispatcher.load_registered_manifest(Path(paths["manifest"]))
    if _campaign_contract(manifest) != protocol.get("campaign_contract"):
        raise SupervisorRefusal("registered joint95-v5 campaign contract changed")
    if Path(paths["results_directory"]).resolve() != Path(
        protocol["campaign_contract"]["results_directory"]
    ).resolve():
        raise SupervisorRefusal("registered supervisor results directory changed")
    if protocol.get("scheduler_placement") != _scheduler_placement(
        Path(paths["results_directory"])
    ):
        raise SupervisorRefusal("registered v5 scheduler reservation placement changed")
    return protocol, manifest


def _terminal_controller_history(
    queue_path: Path,
    exact_signature: str,
    archive_path: Path | None = None,
    scan_archive: bool = False,
) -> list[dict[str, Any]]:
    queue_path = Path(queue_path)
    queue = dispatcher._read_object(queue_path)
    tasks: Any = queue.get("tasks") or []
    if isinstance(tasks, Mapping):
        tasks = list(tasks.values())
    if not isinstance(tasks, list):
        raise SupervisorRefusal("scheduler queue tasks must be a list or object")
    found: list[dict[str, Any]] = []
    seen: set[tuple[Any, str, Any]] = set()
    for row in tasks:
        if not isinstance(row, Mapping):
            continue
        status = str(row.get("status") or row.get("state") or "").lower()
        task_id = row.get("id") or row.get("task_id")
        signature = row.get("signature")
        key = (task_id, status, signature)
        if (
            signature == exact_signature
            and status in TERMINAL_CONTROLLER_STATES
            and key not in seen
        ):
            seen.add(key)
            found.append(
                {
                    "id": task_id,
                    "status": status,
                    "signature": signature,
                    "source": "queue",
                }
            )
    if found or not scan_archive:
        return found
    archive_path = (
        Path(archive_path)
        if archive_path is not None
        else queue_path.parent / "queue_archive.jsonl"
    )
    archived = _recent_terminal_controller_from_archive(
        archive_path, exact_signature
    )
    if archived is not None:
        found.append(archived)
    return found


def _recent_terminal_controller_from_archive(
    archive_path: Path, exact_signature: str
) -> dict[str, Any] | None:
    """Search only the recent archive tail, newest first, and stop on a match."""

    archive_path = Path(archive_path)
    if not archive_path.is_file():
        return None
    try:
        with archive_path.open("rb") as stream:
            end = stream.seek(0, 2)
            floor = max(0, end - ARCHIVE_SCAN_MAX_BYTES)
            position = end
            carry = b""
            lines_seen = 0
            while position > floor and lines_seen < ARCHIVE_SCAN_MAX_LINES:
                size = min(ARCHIVE_SCAN_BLOCK_BYTES, position - floor)
                position -= size
                stream.seek(position)
                parts = (stream.read(size) + carry).splitlines()
                if position > 0:
                    carry = parts[0] if parts else b""
                    complete = parts[1:]
                else:
                    carry = b""
                    complete = parts
                for raw in reversed(complete):
                    lines_seen += 1
                    try:
                        row = json.loads(raw)
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        if lines_seen >= ARCHIVE_SCAN_MAX_LINES:
                            break
                        continue
                    if not isinstance(row, Mapping):
                        continue
                    status = str(
                        row.get("status") or row.get("state") or ""
                    ).lower()
                    if (
                        row.get("signature") == exact_signature
                        and status in TERMINAL_CONTROLLER_STATES
                    ):
                        return {
                            "id": row.get("id") or row.get("task_id"),
                            "status": status,
                            "signature": row.get("signature"),
                            "source": "archive",
                        }
                    if lines_seen >= ARCHIVE_SCAN_MAX_LINES:
                        break
    except OSError as exc:
        raise SupervisorRefusal(
            f"cannot scan scheduler archive {archive_path}: {exc}"
        ) from exc
    return None


def _expected_wave_from_state(state: Mapping[str, Any]) -> int | None:
    status = str(state.get("status") or "")
    if status == "HOLDOUT_LOCKED":
        return campaign.HOLDOUT_WAVE
    next_wave = state.get("next_wave")
    if next_wave is not None:
        return int(next_wave)
    waves = state.get("waves") or {}
    for wave in campaign.JOINT_WAVES:
        summary = waves.get(str(wave)) or {}
        if int(summary.get("success_count") or 0) != 1:
            return int(wave)
    return None


def supervisor_step(
    registration_path: Path = DEFAULT_REGISTRATION,
    *,
    dry_run: bool,
) -> dict[str, Any]:
    protocol, manifest = _load_registration(registration_path)
    paths = protocol["source_paths"]
    results_dir = Path(paths["results_directory"])
    queue_path = Path(paths["queue_status"])
    active = _active_v5_collection_tasks(queue_path)
    if active:
        return {
            "status": "WAIT_ACTIVE_CONTROLLER",
            "terminal": False,
            "collection_status": "NOT_READ_WHILE_CONTROLLER_ACTIVE",
            "active_controller_ids": [row.get("id") for row in active],
            "mutations": 0,
        }
    state = dispatcher.collection_status(manifest, results_dir)
    status = str(state.get("status") or "")
    if status not in KNOWN_COLLECTION_STATES:
        return {
            "status": "TERMINATE_INVALID_STATE",
            "terminal": True,
            "collection_status": status,
            "errors": ["unknown collection state"],
            "mutations": 0,
        }
    if status == "FAIL_STOP":
        return {
            "status": "TERMINATE_FAIL_STOP",
            "terminal": True,
            "collection_status": status,
            "errors": list(state.get("errors") or []),
            "mutations": 0,
            "retry_attempted": False,
        }
    if status == "COMPLETE":
        return {
            "status": "COMPLETE",
            "terminal": True,
            "collection_status": status,
            "mutations": 0,
        }

    expected_wave = _expected_wave_from_state(state)
    if expected_wave is None:
        raise SupervisorRefusal(f"collection state {status!r} has no expected wave")
    expected_item = dispatcher._work_item(manifest, int(expected_wave))
    exact_signature = (
        dispatcher.COLLECTION_SIGNATURE_PREFIX
        + str(expected_item["joint_unit_id"])
    )
    attempt_path = results_dir / f"{expected_item['joint_unit_id']}_attempt01.json"
    control_path = dispatcher.controller_state_path(results_dir, expected_item)
    terminal_controllers = _terminal_controller_history(
        queue_path,
        exact_signature,
        Path(paths["queue_archive"]),
        control_path.is_file() or attempt_path.is_file(),
    )
    if terminal_controllers:
        if dry_run:
            return {
                "status": "DRY_RUN_FINALIZE_CONTROLLER_FAILURE",
                "terminal": False,
                "collection_status": status,
                "expected_wave": int(expected_wave),
                "controller_signature": exact_signature,
                "terminal_controllers": terminal_controllers,
                "planned_actions": ["write_validated_fail_stop", "release_after_quiescence"],
                "mutations": 0,
                "retry_attempted": False,
            }
        terminal = dispatcher.finalize_terminal_controller_failure(
            manifest=manifest,
            results_dir=results_dir,
            wave=int(expected_wave),
            terminal_controllers=terminal_controllers,
        )
        finalized = dispatcher.collection_status(manifest, results_dir)
        if finalized.get("status") != "FAIL_STOP":
            raise SupervisorRefusal("controller failure did not create a v5 fail-stop")
        return {
            "status": "TERMINATE_FAIL_STOP",
            "terminal": True,
            "collection_status": "FAIL_STOP",
            "expected_wave": int(expected_wave),
            "controller_signature": exact_signature,
            "terminal_controllers": terminal_controllers,
            "fail_stop_decision": terminal.get("decision"),
            "mutations": 2,
            "retry_attempted": False,
        }
    if status == "FAIL_RESULT_STATE" or state.get("errors"):
        return {
            "status": "TERMINATE_INVALID_STATE",
            "terminal": True,
            "collection_status": status,
            "errors": list(state.get("errors") or []),
            "mutations": 0,
            "retry_attempted": False,
        }

    submit_kwargs = {
        "manifest_path": Path(paths["manifest"]),
        "results_dir": results_dir,
        "queue_path": queue_path,
        "node_status_path": Path(paths["node_status"]),
        "scheduler_path": Path(paths["scheduler"]),
        "python_bin": paths["python_bin"],
    }
    if status == "HOLDOUT_LOCKED":
        if (
            state.get("successful_calibration_count") != 19
            or state.get("holdout_success") is not False
            or state.get("holdout_released") is not False
        ):
            return {
                "status": "TERMINATE_INVALID_STATE",
                "terminal": True,
                "collection_status": status,
                "errors": ["holdout lock does not follow exactly 19 calibrations"],
                "mutations": 0,
            }
        if dry_run:
            return {
                "status": "DRY_RUN_RELEASE_THEN_SUBMIT_HOLDOUT",
                "terminal": False,
                "collection_status": status,
                "planned_actions": ["release_holdout", "submit_next_wave113"],
                "mutations": 0,
            }
        dispatcher.release_holdout(
            manifest_path=submit_kwargs["manifest_path"],
            results_dir=results_dir,
        )
        released_state = dispatcher.collection_status(manifest, results_dir)
        if (
            released_state.get("status") != "READY_HOLDOUT"
            or released_state.get("next_wave") != campaign.HOLDOUT_WAVE
            or released_state.get("errors")
        ):
            raise SupervisorRefusal("frozen holdout release did not expose wave 113")
        submitted = dispatcher.submit_next(**submit_kwargs, dry_run=False)
        return {
            "status": "SUBMITTED_HOLDOUT",
            "terminal": False,
            "collection_status": "READY_HOLDOUT",
            "joint_wave": submitted.get("joint_wave"),
            "task_id": submitted.get("task_id"),
            "release_path": str(results_dir / "holdout_release.json"),
            "mutations": 2,
        }

    if status not in {"READY_CALIBRATION", "READY_HOLDOUT"}:
        raise SupervisorRefusal(f"unhandled collection state {status!r}")
    submitted = dispatcher.submit_next(**submit_kwargs, dry_run=dry_run)
    if submitted.get("joint_wave") != expected_wave:
        raise SupervisorRefusal("dispatcher planned a wave different from collection state")
    return {
        "status": (
            "DRY_RUN_SUBMIT_ONE" if dry_run else "SUBMITTED_ONE"
        ),
        "terminal": False,
        "collection_status": status,
        "joint_wave": expected_wave,
        "split_role": submitted.get("split_role"),
        "task_id": submitted.get("task_id"),
        "mutations": 0 if dry_run else 1,
    }


def _validated_fail_stop(manifest: Mapping[str, Any], results_dir: Path) -> bool:
    path = Path(results_dir) / dispatcher.FAIL_STOP_FILENAME
    if not path.is_file():
        return False
    terminal = _read_object(path)
    wave = terminal.get("failed_wave")
    if wave not in campaign.JOINT_WAVES:
        return False
    item = dispatcher._work_item(manifest, int(wave))
    attempt_path = Path(results_dir) / f"{item['joint_unit_id']}_attempt01.json"
    common_valid = bool(
        terminal.get("gate") == dispatcher.FAIL_STOP_GATE
        and terminal.get("protocol") == campaign.PROTOCOL
        and terminal.get("status") == "FAIL_STOP"
        and terminal.get("terminal") is True
        and terminal.get("pass") is False
        and terminal.get("failed_joint_unit_id") == item["joint_unit_id"]
        and terminal.get("replacement_attempt_authorized") is False
        and terminal.get("holdout_release_authorized") is False
        and all(terminal.get(f"v{version}_joint_units_used") == 0 for version in (1, 2, 3, 4))
    )
    if not common_valid:
        return False
    if terminal.get("decision") == "JOINT95_V5_PREMEASUREMENT_CONTROLLER_FAIL_STOP":
        control_path = Path(str(terminal.get("controller_state_path") or ""))
        if not control_path.is_file() or attempt_path.exists():
            return False
        control = _read_object(control_path)
        return bool(
            terminal.get("failed_attempt_path") is None
            and terminal.get("formal_attempt_created") is False
            and terminal.get("measurements_started") is False
            and control.get("gate") == dispatcher.CONTROLLER_STATE_GATE
            and control.get("protocol") == campaign.PROTOCOL
            and control.get("joint_unit_id") == item["joint_unit_id"]
            and control.get("joint_wave") == int(wave)
            and control.get("status") == "PREMEASUREMENT_FAIL_STOP"
            and control.get("terminal") is True
            and control.get("formal_attempt_created") is False
            and control.get("measurement_started") is False
            and control.get("empty_rows") == []
            and control.get("observations") == []
            and control.get("scheduler_interval_audits") == []
        )
    if terminal.get("decision") not in {
        "JOINT95_V5_INTERVAL_OVERLAP_FAIL_STOP",
        "JOINT95_V5_FIRST_ATTEMPT_FAIL_STOP",
    } or not attempt_path.is_file():
        return False
    attempt = _read_object(attempt_path)
    return bool(
        terminal.get("failed_attempt_path") == str(attempt_path)
        and terminal.get("formal_attempt_created") is True
        and terminal.get("measurements_started") is True
        and attempt.get("gate") == dispatcher.RESULT_GATE
        and attempt.get("protocol") == campaign.PROTOCOL
        and attempt.get("joint_unit_id") == item["joint_unit_id"]
        and attempt.get("joint_wave") == int(wave)
        and attempt.get("status") == "FAILED"
        and attempt.get("terminal") is True
        and attempt.get("pass") is False
    )


def release_node_reservation_if_terminal(
    registration_path: Path, status: str
) -> dict[str, Any]:
    """Release the scheduler claim only after a valid terminal and idle children."""

    if status not in {"COMPLETE", "TERMINATE_FAIL_STOP"}:
        raise SupervisorRefusal("nonterminal v5 state cannot release node reservation")
    protocol, manifest = _load_registration(registration_path)
    paths = protocol["source_paths"]
    results_dir = Path(paths["results_directory"])
    queue_path = Path(paths["queue_status"])
    state = dispatcher.collection_status(manifest, results_dir)
    expected_collection_status = (
        "COMPLETE" if status == "COMPLETE" else "FAIL_STOP"
    )
    if state.get("status") != expected_collection_status:
        raise SupervisorRefusal("v5 collection changed before reservation release")
    if status == "TERMINATE_FAIL_STOP" and not _validated_fail_stop(manifest, results_dir):
        raise SupervisorRefusal("v5 fail-stop sentinel is not valid for reservation release")
    if _active_v5_collection_tasks(queue_path):
        raise SupervisorRefusal("v5 wave controller is active at reservation release")
    if _active_scheduler_tasks(campaign.NODE, queue_file=queue_path):
        raise SupervisorRefusal("target-node scheduler work is active at reservation release")
    gpu_idle = empty_suite._gpu_idle_snapshot(empty_suite.NODE_SPECS[campaign.NODE])
    if gpu_idle.get("ready") is not True:
        raise SupervisorRefusal("target GPU work is active at reservation release")
    marker = Path(protocol["scheduler_placement"]["reserve_until_file"])
    if marker.exists():
        return {"reservation_release_marker": str(marker), "already_released": True}
    _write_new(
        marker,
        {
            "gate": "critical_gpu_joint95_v5_node_reservation_release",
            "campaign_status": expected_collection_status,
            "released_at_utc": datetime.now(timezone.utc).isoformat(),
            "active_v5_controllers": 0,
            "active_target_node_tasks": 0,
            "target_gpu_idle": True,
        },
    )
    return {"reservation_release_marker": str(marker), "already_released": False}


def run_supervisor(
    registration_path: Path = DEFAULT_REGISTRATION,
    *,
    dry_run: bool,
    once: bool,
    poll_s: float,
) -> int:
    if poll_s <= 0.0:
        raise SupervisorRefusal("poll interval must be positive")
    if dry_run and not once:
        raise SupervisorRefusal("dry-run requires --once")
    while True:
        report = supervisor_step(registration_path, dry_run=dry_run)
        if not dry_run and report["status"] in {"COMPLETE", "TERMINATE_FAIL_STOP"}:
            report.update(release_node_reservation_if_terminal(registration_path, report["status"]))
        print(json.dumps(report, sort_keys=True), flush=True)
        if once or report["terminal"]:
            return 0 if report["status"] in {"COMPLETE", "WAIT_ACTIVE_CONTROLLER"} or not report["status"].startswith("TERMINATE_") else 2
        time.sleep(min(float(poll_s), 60.0))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("register")
    freeze.add_argument("--output", type=Path, default=DEFAULT_REGISTRATION)
    freeze.add_argument("--manifest", type=Path, default=campaign.DEFAULT_OUTPUT)
    freeze.add_argument("--results-dir", type=Path, default=campaign.DEFAULT_RESULTS_DIR)
    freeze.add_argument("--queue-status", type=Path, default=dispatcher.DEFAULT_QUEUE_STATUS)
    freeze.add_argument("--node-status", type=Path, default=dispatcher.DEFAULT_NODE_STATUS)
    freeze.add_argument("--scheduler", type=Path, default=dispatcher.SCHEDULER)
    freeze.add_argument("--python-bin", default=sys.executable)
    run = sub.add_parser("run")
    run.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    run.add_argument("--once", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--poll-s", type=float, default=30.0)
    args = parser.parse_args()
    try:
        if args.command == "register":
            report = register(
                args.output,
                manifest_path=args.manifest,
                results_dir=args.results_dir,
                queue_path=args.queue_status,
                node_status_path=args.node_status,
                scheduler_path=args.scheduler,
                python_bin=args.python_bin,
            )
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        return run_supervisor(
            args.registration,
            dry_run=args.dry_run,
            once=args.once,
            poll_s=args.poll_s,
        )
    except (SupervisorRefusal, dispatcher.DispatchRefusal) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
