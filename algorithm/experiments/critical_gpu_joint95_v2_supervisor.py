"""Deterministically supervise the registered joint95-v2 19+1 campaign.

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

from algorithm.experiments import critical_gpu_joint95_v2_dispatcher as dispatcher
from algorithm.experiments import critical_gpu_joint95_v2_registration as campaign


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
DEFAULT_REGISTRATION = ARTIFACTS / (
    "critical_gpu_joint95_v2_supervisor_registration_v3_20260913.json"
)
DEFAULT_V2_REGISTRATION = ARTIFACTS / (
    "critical_gpu_joint95_v2_supervisor_registration_v2_20260913.json"
)
DEFAULT_V2_DISPOSITION = ARTIFACTS / (
    "critical_gpu_joint95_v2_supervisor_registration_v2_disposition_20260913.json"
)
DEFAULT_V1_REGISTRATION = ARTIFACTS / (
    "critical_gpu_joint95_v2_supervisor_registration_20260913.json"
)
DEFAULT_V1_DISPOSITION = ARTIFACTS / (
    "critical_gpu_joint95_v2_supervisor_registration_v1_disposition_20260913.json"
)
GATE = "critical_gpu_joint95_v2_supervisor"
PROTOCOL = "critical_gpu_joint95_v2_deterministic_supervisor_v3"
REGISTRATION_STATUS = "REGISTERED_READY_FOR_LOCAL_CONTROL"
SUPERVISOR_SCHEDULER_SIGNATURE = "ScheduleurmBench/joint95_v2_supervisor_v3"
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
IN_FLIGHT_RESULT_STATES = frozenset(
    {"STARTING", "WAIT_SCHEDULER_ACK", "RUNNING_EMPTY", "RUNNING_LOADED"}
)


class SupervisorRefusal(RuntimeError):
    """The supervisor registration or collection state is unusable."""


def _active_v2_collection_tasks(queue_path: Path) -> list[dict[str, Any]]:
    """Return only active controllers in this campaign's exact v2 namespace."""
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
        raise SupervisorRefusal("joint95-v2 manifest identity or authority differs")
    if (
        coverage.get("calibration_joint_waves")
        != list(campaign.CALIBRATION_WAVES)
        or coverage.get("holdout_joint_wave") != campaign.HOLDOUT_WAVE
        or coverage.get("finite_sample_rank") != 19
    ):
        raise SupervisorRefusal("joint95-v2 19+1 split differs")
    if (
        executor.get("status") != "PASS"
        or executor.get("module") != campaign.V2_EXECUTOR_MODULE
        or executor.get("validated_scope")
        != list(campaign.V2_EXECUTOR_TEST_SCOPE)
    ):
        raise SupervisorRefusal("joint95-v2 executor validation differs")
    if any(
        int(replacement.get(field, -1)) != 0
        for field in (
            "v1_joint_units_eligible_for_v2_calibration",
            "v1_joint_units_eligible_for_v2_holdout",
            "v1_joint_scores_eligible_for_v2_margin",
        )
    ):
        raise SupervisorRefusal("v1 evidence is eligible in the joint95-v2 manifest")
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
        raise SupervisorRefusal("supervisor results directory differs from v2 manifest")
    queue_path = Path(queue_path).resolve()
    entries = sorted(results_dir.iterdir()) if results_dir.is_dir() else []
    active = _active_v2_collection_tasks(queue_path)
    first_item = dispatcher._work_item(manifest, campaign.CALIBRATION_WAVES[0])
    first_signature = (
        dispatcher.COLLECTION_SIGNATURE_PREFIX + first_item["joint_unit_id"]
    )
    exact_active = [row for row in active if row.get("signature") == first_signature]
    if active:
        expected_path = results_dir / f"{first_item['joint_unit_id']}_attempt01.json"
        if len(active) != 1 or len(exact_active) != 1 or entries != [expected_path]:
            raise SupervisorRefusal("active repair registration has unexpected control state")
        partial = _read_object(expected_path)
        no_measurements = (
            partial.get("empty_rows") == []
            and partial.get("loaded_campaign") is None
            and partial.get("observations") == []
        )
        if (
            partial.get("status") not in IN_FLIGHT_RESULT_STATES
            or partial.get("terminal") is not False
            or partial.get("pass") is not False
            or partial.get("joint_wave") != 34
            or partial.get("joint_unit_id") != first_item["joint_unit_id"]
            or not no_measurements
        ):
            raise SupervisorRefusal("active wave34 repair is not a zero-measurement partial")
        registration_snapshot = {
            "kind": "ACTIVE_WAVE34_ZERO_MEASUREMENT_CONTROL_REPAIR",
            "collection_status_read_while_controller_active": False,
            "active_controller_id": exact_active[0].get("id"),
            "active_controller_signature": first_signature,
            "attempt_path": str(expected_path),
            "attempt_status": partial["status"],
            "attempt_terminal": False,
            "measurement_count": 0,
            "successful_calibration_count": 0,
            "scores_read_by_supervisor_registration": 0,
        }
    else:
        if entries:
            raise SupervisorRefusal("inactive registration cannot contain v2 attempts")
        state = dispatcher.collection_status(manifest, results_dir)
        if (
            state.get("status") != "READY_CALIBRATION"
            or state.get("next_wave") != 34
            or state.get("successful_calibration_count") != 0
            or state.get("holdout_success") is not False
            or state.get("errors")
        ):
            raise SupervisorRefusal("joint95-v2 campaign is not at its zero-result start")
        registration_snapshot = {
            "kind": "ZERO_RESULT_START",
            "collection_status_read_while_controller_active": False,
            "results_entry_count": 0,
            "successful_calibration_count": 0,
            "next_wave": 34,
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
            "node_status": str(Path(node_status_path).resolve()),
            "scheduler": str(Path(scheduler_path).resolve()),
            "python_bin": str(Path(python_bin).resolve()),
        },
        "campaign_contract": contract,
        "registration_snapshot": registration_snapshot,
        "supersedes": {
            "registration": str(DEFAULT_V2_REGISTRATION.resolve()),
            "disposition": str(DEFAULT_V2_DISPOSITION.resolve()),
            "reason": (
                "v2 trusted the dispatcher's broader legacy active-task match "
                "without filtering to the exact joint95-v2 signature prefix"
            ),
        },
        "control_policy": {
            "READY_CALIBRATION": "if no active joint95 controller, call existing submit_next once",
            "HOLDOUT_LOCKED": (
                "if exactly 19 calibrations succeeded and no controller is active, "
                "call frozen release_holdout once, then submit_next for wave 53"
            ),
            "READY_HOLDOUT": "if no active joint95 controller, call existing submit_next once",
            "ACTIVE_CONTROLLER": "wait without mutation",
            "TERMINAL_CONTROLLER_WITHOUT_SUCCESS": (
                "terminate when the exact expected-wave controller signature is "
                "already terminal but its valid PASS result is absent"
            ),
            "FAIL_STOP": "terminate without release, submission, or retry",
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
        },
        "scheduler_placement": {
            "require_node": "local",
            "vram_mb": 0,
            "signature": SUPERVISOR_SCHEDULER_SIGNATURE,
            "signature_is_outside_joint95_controller_prefix": (
                not SUPERVISOR_SCHEDULER_SIGNATURE.startswith(
                    dispatcher.COLLECTION_SIGNATURE_PREFIX.rsplit("/", 1)[0]
                )
            ),
        },
        "prohibited": [
            "direct measurement execution",
            "submission while any joint95 controller is active",
            "retry after any failed or invalid attempt",
            "score-dependent selection outside the registered rank-19 release",
            "v1 evidence reuse, server pulls, or paper edits",
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
        raise SupervisorRefusal("registered joint95-v2 campaign contract changed")
    if Path(paths["results_directory"]).resolve() != Path(
        protocol["campaign_contract"]["results_directory"]
    ).resolve():
        raise SupervisorRefusal("registered supervisor results directory changed")
    return protocol, manifest


def _terminal_controller_history(
    queue_path: Path, exact_signature: str
) -> list[dict[str, Any]]:
    queue = dispatcher._read_object(Path(queue_path))
    tasks: Any = queue.get("tasks") or []
    if isinstance(tasks, Mapping):
        tasks = list(tasks.values())
    if not isinstance(tasks, list):
        raise SupervisorRefusal("scheduler queue tasks must be a list or object")
    return [
        {
            "id": row.get("id") or row.get("task_id"),
            "status": str(row.get("status") or row.get("state") or "").lower(),
            "signature": row.get("signature"),
        }
        for row in tasks
        if isinstance(row, Mapping)
        and row.get("signature") == exact_signature
        and str(row.get("status") or row.get("state") or "").lower()
        in TERMINAL_CONTROLLER_STATES
    ]


def supervisor_step(
    registration_path: Path = DEFAULT_REGISTRATION,
    *,
    dry_run: bool,
) -> dict[str, Any]:
    protocol, manifest = _load_registration(registration_path)
    paths = protocol["source_paths"]
    results_dir = Path(paths["results_directory"])
    queue_path = Path(paths["queue_status"])
    active = _active_v2_collection_tasks(queue_path)
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
    if status == "FAIL_RESULT_STATE" or state.get("errors"):
        return {
            "status": "TERMINATE_INVALID_STATE",
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

    expected_wave = (
        campaign.HOLDOUT_WAVE
        if status == "HOLDOUT_LOCKED"
        else state.get("next_wave")
    )
    if expected_wave is None:
        raise SupervisorRefusal(f"collection state {status!r} has no expected wave")
    expected_item = dispatcher._work_item(manifest, int(expected_wave))
    exact_signature = (
        dispatcher.COLLECTION_SIGNATURE_PREFIX
        + str(expected_item["joint_unit_id"])
    )
    terminal_controllers = _terminal_controller_history(queue_path, exact_signature)
    if terminal_controllers:
        return {
            "status": "TERMINATE_CONTROLLER_FAILURE",
            "terminal": True,
            "collection_status": status,
            "expected_wave": int(expected_wave),
            "controller_signature": exact_signature,
            "terminal_controllers": terminal_controllers,
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
                "planned_actions": ["release_holdout", "submit_next_wave53"],
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
            raise SupervisorRefusal("frozen holdout release did not expose wave 53")
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
