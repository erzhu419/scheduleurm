"""Run the frozen joint95-v2 analyses once the 19+1 campaign is complete."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from algorithm.experiments import aligned_hardware_joint95_v2_baseline_confirmation_v3 as baseline
from algorithm.experiments import aligned_hardware_joint95_v2_canonical_selector_screen as canonical
from algorithm.experiments import aligned_hardware_joint95_v2_capacity_closure as capacity
from algorithm.experiments import critical_gpu_joint95_v2_dispatcher as dispatcher
from algorithm.experiments import critical_gpu_joint95_v2_registration as campaign


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
DEFAULT_REGISTRATION = ARTIFACTS / (
    "aligned_hardware_joint95_v2_postcampaign_runner_v3_20260913_protocol.json"
)
DEFAULT_V2_REGISTRATION = ARTIFACTS / (
    "aligned_hardware_joint95_v2_postcampaign_runner_v2_20260913_protocol.json"
)
DEFAULT_V2_DISPOSITION = ARTIFACTS / (
    "aligned_hardware_joint95_v2_postcampaign_runner_v2_disposition_20260913.json"
)
DEFAULT_V1_REGISTRATION = ARTIFACTS / (
    "aligned_hardware_joint95_v2_postcampaign_runner_20260913_protocol.json"
)
DEFAULT_V1_DISPOSITION = ARTIFACTS / (
    "aligned_hardware_joint95_v2_postcampaign_runner_v1_disposition_20260913.json"
)
GATE = "aligned_hardware_joint95_v2_postcampaign_runner"
PROTOCOL = "aligned_hardware_joint95_v2_postcampaign_v3"
REGISTRATION_STATUS = "REGISTERED_AWAITING_JOINT95_V2_COMPLETE"
STAGE_ORDER = ("capacity", "canonical", "baseline")
INCOMPLETE_COLLECTION_STATES = frozenset(
    {"READY_CALIBRATION", "HOLDOUT_LOCKED", "READY_HOLDOUT"}
)
CAPACITY_ADVANCE_DECISION = "CAPACITY_ESTABLISHED_CANDIDATE_PENDING"


class PostcampaignRefusal(RuntimeError):
    """The frozen postcampaign contract cannot be used."""


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
        raise PostcampaignRefusal(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PostcampaignRefusal(f"JSON root must be an object: {path}")
    return value


def _write_new(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(dict(payload), stream, indent=2, sort_keys=True)
        stream.write("\n")


def _load_downstream_protocols() -> list[dict[str, Any]]:
    capacity_protocol = _read_object(capacity.DEFAULT_REGISTRATION)
    if (
        capacity_protocol.get("schema_version") != 1
        or capacity_protocol.get("gate") != capacity.GATE
        or capacity_protocol.get("experiment") != capacity.EXPERIMENT
        or capacity_protocol.get("status") != capacity.STATUS
        or capacity_protocol.get("registered") is not True
    ):
        raise PostcampaignRefusal("v2 capacity registration is invalid")

    canonical_protocol = _read_object(canonical.DEFAULT_REGISTRATION)
    try:
        canonical._validate_protocol(canonical_protocol)
    except canonical.V2CanonicalScreenRefusal as exc:
        raise PostcampaignRefusal(str(exc)) from exc
    if canonical_protocol.get("capacity_registration_contract") != capacity_protocol:
        raise PostcampaignRefusal("canonical registration does not embed capacity")

    baseline_protocol = _read_object(baseline.DEFAULT_PROTOCOL)
    try:
        baseline._validate_protocol(baseline_protocol)
    except baseline.V2BaselineRefusal as exc:
        raise PostcampaignRefusal(str(exc)) from exc
    if (
        baseline_protocol.get("capacity_registration_contract") != capacity_protocol
        or baseline_protocol.get("canonical_registration_contract")
        != canonical_protocol
    ):
        raise PostcampaignRefusal("baseline registration does not embed both upstreams")

    return [
        {
            "name": "capacity",
            "module": "algorithm.experiments.aligned_hardware_joint95_v2_capacity_closure",
            "registration_path": str(capacity.DEFAULT_REGISTRATION.resolve()),
            "fixed_sources_path": None,
            "result_path": str(capacity.DEFAULT_RESULT.resolve()),
            "registration_snapshot": capacity_protocol,
            "decision_universe": list(capacity.DECISION_RULE),
            "advance_decisions": [CAPACITY_ADVANCE_DECISION],
        },
        {
            "name": "canonical",
            "module": "algorithm.experiments.aligned_hardware_joint95_v2_canonical_selector_screen",
            "registration_path": str(canonical.DEFAULT_REGISTRATION.resolve()),
            "fixed_sources_path": str(canonical.DEFAULT_FIXED_SOURCES.resolve()),
            "result_path": str(canonical.DEFAULT_RESULT.resolve()),
            "registration_snapshot": canonical_protocol,
            "decision_universe": list(canonical.base.DECISION_RULE),
            "advance_decisions": [canonical.PASS_DECISION],
        },
        {
            "name": "baseline",
            "module": "algorithm.experiments.aligned_hardware_joint95_v2_baseline_confirmation_v3",
            "registration_path": str(baseline.DEFAULT_PROTOCOL.resolve()),
            "fixed_sources_path": str(baseline.DEFAULT_FIXED_SOURCES.resolve()),
            "result_path": str(baseline.DEFAULT_RESULT.resolve()),
            "registration_snapshot": baseline_protocol,
            "decision_universe": list(baseline.base.DECISION_RULE),
            "advance_decisions": [baseline.PASS_DECISION],
        },
    ]


def build_registration(
    *,
    manifest_path: Path = campaign.DEFAULT_OUTPUT,
    results_dir: Path = campaign.DEFAULT_RESULTS_DIR,
    queue_path: Path = dispatcher.DEFAULT_QUEUE_STATUS,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    results_dir = Path(results_dir).resolve()
    manifest = dispatcher.load_registered_manifest(manifest_path)
    registered_results = Path(
        str((manifest.get("results_contract") or {}).get("results_directory") or "")
    ).resolve()
    if registered_results != results_dir:
        raise PostcampaignRefusal("postcampaign results directory differs from v2 manifest")
    queue_path = Path(queue_path).resolve()
    entries = sorted(results_dir.iterdir()) if results_dir.is_dir() else []
    active = _active_v2_collection_tasks(queue_path)
    if active:
        registration_snapshot = {
            "kind": "ACTIVE_COLLECTION_CONTROL_REPAIR",
            "collection_status_read_while_controller_active": False,
            "active_controller_ids": [row.get("id") for row in active],
            "active_controller_signatures": [row.get("signature") for row in active],
            "joint_result_entries": [path.name for path in entries],
        }
    else:
        state = dispatcher.collection_status(manifest, results_dir)
        if (
            entries
            or state.get("status") != "READY_CALIBRATION"
            or state.get("next_wave") != 34
            or state.get("successful_calibration_count") != 0
            or state.get("holdout_success") is not False
        ):
            raise PostcampaignRefusal(
                "inactive postcampaign registration requires zero v2 results"
            )
        registration_snapshot = {
            "kind": "ZERO_RESULT_START",
            "collection_status_read_while_controller_active": False,
            "joint_result_entries": [],
        }
    stages = _load_downstream_protocols()
    existing = [row["name"] for row in stages if Path(row["result_path"]).exists()]
    if existing:
        raise PostcampaignRefusal(
            "postcampaign registration requires no downstream result: "
            + ", ".join(existing)
        )
    return {
        "schema_version": 1,
        "gate": GATE,
        "protocol": PROTOCOL,
        "status": REGISTRATION_STATUS,
        "registered": True,
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_paths": {
            "collection_manifest": str(manifest_path),
            "joint_results_directory": str(results_dir),
            "queue_status": str(queue_path),
        },
        "collection_contract": {
            "protocol": campaign.PROTOCOL,
            "collection_prefix": campaign.COLLECTION_PREFIX,
            "calibration_waves": list(campaign.CALIBRATION_WAVES),
            "holdout_wave": campaign.HOLDOUT_WAVE,
            "required_terminal_status": "COMPLETE",
        },
        "stage_order": list(STAGE_ORDER),
        "stages": stages,
        "registration_snapshot": registration_snapshot,
        "supersedes": {
            "registration": str(DEFAULT_V2_REGISTRATION.resolve()),
            "disposition": str(DEFAULT_V2_DISPOSITION.resolve()),
            "reason": (
                "v2 trusted the dispatcher's broader legacy active-task match "
                "without filtering to the exact joint95-v2 signature prefix"
            ),
        },
        "execution_policy": {
            "incomplete_collection": "wait without running a downstream stage",
            "active_collection_controller": (
                "wait without calling collection_status or a downstream runner "
                "only when its signature starts with the exact joint95-v2 prefix"
            ),
            "failed_or_invalid_collection": "terminate without running a downstream stage",
            "stage_order": "capacity then canonical then baseline",
            "existing_result": "validate and advance; never overwrite or rerun",
            "negative_stage_decision": "retain result and terminate before downstream",
            "capacity_advance_meaning": (
                "CAPACITY_ESTABLISHED_CANDIDATE_PENDING advances because the frozen "
                "canonical stage supplies the missing same-envelope candidate evidence"
            ),
        },
        "prohibited": [
            "scheduler submission or direct measurement",
            "running before collection COMPLETE",
            "overwriting or rerunning an existing result",
            "running a downstream stage after an upstream negative decision",
            "changing stage order or selecting by observed score",
            "server pulls or paper edits",
        ],
        "scheduler_placement": {
            "require_node": "local",
            "vram_mb": 0,
            "signature": "ScheduleurmBench/joint95_v2_postcampaign_v3",
            "active_controller_signature_prefix": dispatcher.COLLECTION_SIGNATURE_PREFIX,
        },
    }


def register(output: Path = DEFAULT_REGISTRATION, **kwargs: Any) -> dict[str, Any]:
    output = Path(output)
    if output.exists():
        raise PostcampaignRefusal(f"postcampaign registration exists: {output}")
    protocol = build_registration(**kwargs)
    _write_new(output, protocol)
    return protocol


def _load_registration(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    protocol = _read_object(path)
    stages = list(protocol.get("stages") or [])
    if (
        protocol.get("schema_version") != 1
        or protocol.get("gate") != GATE
        or protocol.get("protocol") != PROTOCOL
        or protocol.get("status") != REGISTRATION_STATUS
        or protocol.get("registered") is not True
        or protocol.get("stage_order") != list(STAGE_ORDER)
        or [row.get("name") for row in stages] != list(STAGE_ORDER)
    ):
        raise PostcampaignRefusal("postcampaign registration is invalid")
    paths = protocol.get("source_paths") or {}
    manifest = dispatcher.load_registered_manifest(Path(paths["collection_manifest"]))
    registered_results = Path(
        str((manifest.get("results_contract") or {}).get("results_directory") or "")
    ).resolve()
    if registered_results != Path(paths["joint_results_directory"]).resolve():
        raise PostcampaignRefusal("postcampaign collection path changed")
    current = _load_downstream_protocols()
    if current != stages:
        raise PostcampaignRefusal("a downstream registration changed after freeze")
    return protocol, manifest, stages


def _validate_stage_result(
    stage: Mapping[str, Any], payload: Mapping[str, Any]
) -> bool:
    name = str(stage["name"])
    protocol = stage["registration_snapshot"]
    expected = {
        "capacity": (capacity.GATE, capacity.EXPERIMENT),
        "canonical": (canonical.GATE, canonical.EXPERIMENT),
        "baseline": (baseline.GATE, baseline.EXPERIMENT),
    }[name]
    decision = payload.get("scientific_decision")
    if (
        payload.get("schema_version") != 1
        or payload.get("gate") != expected[0]
        or payload.get("experiment") != expected[1]
        or payload.get("status") not in {"PASS", "FAIL"}
        or payload.get("protocol") != protocol
        or decision not in stage["decision_universe"]
    ):
        raise PostcampaignRefusal(f"existing {name} result is invalid")
    advances = decision in stage["advance_decisions"]
    if name == "capacity":
        joint = payload.get("joint95_v2_audit") or {}
        if payload.get("status") != "PASS" or (
            advances
            and (
                payload.get("capacity_established") is not True
                or joint.get("holdout_covered") is not True
            )
        ):
            raise PostcampaignRefusal("capacity result flags contradict its decision")
    elif name == "canonical":
        if (
            payload.get("physical_trajectory_advance") is not advances
            or payload.get("baseline_trajectory_advance") is not advances
        ):
            raise PostcampaignRefusal("canonical result flags contradict its decision")
    elif payload.get("baseline_confirmation_passed") is not advances:
        raise PostcampaignRefusal("baseline result flag contradicts its decision")
    return advances


def _run_stage(stage: Mapping[str, Any]) -> dict[str, Any]:
    name = stage["name"]
    if name == "capacity":
        return capacity.run_registered(
            Path(stage["registration_path"]), Path(stage["result_path"])
        )
    if name == "canonical":
        return canonical.run_registered(
            Path(stage["registration_path"]),
            Path(stage["fixed_sources_path"]),
            Path(stage["result_path"]),
        )
    if name == "baseline":
        return baseline.run_registered(
            Path(stage["registration_path"]),
            Path(stage["fixed_sources_path"]),
            Path(stage["result_path"]),
        )
    raise PostcampaignRefusal(f"unknown postcampaign stage {name!r}")


def postcampaign_step(
    registration_path: Path = DEFAULT_REGISTRATION, *, dry_run: bool
) -> dict[str, Any]:
    protocol, manifest, stages = _load_registration(Path(registration_path))
    results_dir = Path(
        str((manifest.get("results_contract") or {})["results_directory"])
    )
    queue_path = Path(protocol["source_paths"]["queue_status"])
    active = _active_v2_collection_tasks(queue_path)
    if active:
        return {
            "status": "WAIT_COLLECTION_CONTROLLER",
            "terminal": False,
            "collection_status": "NOT_READ_WHILE_CONTROLLER_ACTIVE",
            "active_controller_ids": [row.get("id") for row in active],
            "stages_run": [],
        }
    collection = dispatcher.collection_status(manifest, results_dir)
    state = str(collection.get("status") or "")
    if state in {"FAIL_STOP", "FAIL_RESULT_STATE"} or collection.get("errors"):
        return {
            "status": "TERMINATE_COLLECTION_FAILURE",
            "terminal": True,
            "collection_status": state,
            "errors": list(collection.get("errors") or []),
            "stages_run": [],
        }
    if state in INCOMPLETE_COLLECTION_STATES:
        return {
            "status": "WAIT_COLLECTION",
            "terminal": False,
            "collection_status": state,
            "stages_run": [],
        }
    if state != "COMPLETE":
        return {
            "status": "TERMINATE_COLLECTION_INVALID",
            "terminal": True,
            "collection_status": state,
            "errors": ["unknown collection state"],
            "stages_run": [],
        }

    existing: list[tuple[dict[str, Any], dict[str, Any], bool] | None] = []
    first_missing = None
    for index, stage in enumerate(stages):
        result_path = Path(stage["result_path"])
        if not result_path.is_file():
            if first_missing is None:
                first_missing = index
            existing.append(None)
            continue
        if first_missing is not None:
            return {
                "status": "TERMINATE_DOWNSTREAM_ORDER_INVALID",
                "terminal": True,
                "collection_status": state,
                "errors": [f"{stage['name']} result exists after a missing upstream"],
                "stages_run": [],
            }
        try:
            payload = _read_object(result_path)
            advances = _validate_stage_result(stage, payload)
        except PostcampaignRefusal as exc:
            return {
                "status": "TERMINATE_EXISTING_RESULT_INVALID",
                "terminal": True,
                "collection_status": state,
                "errors": [str(exc)],
                "stages_run": [],
            }
        existing.append((stage, payload, advances))
        if not advances:
            if any(Path(row["result_path"]).exists() for row in stages[index + 1 :]):
                return {
                    "status": "TERMINATE_DOWNSTREAM_ORDER_INVALID",
                    "terminal": True,
                    "collection_status": state,
                    "errors": [f"downstream result exists after negative {stage['name']}"],
                    "stages_run": [],
                }
            return {
                "status": f"TERMINATE_{stage['name'].upper()}_DECISION",
                "terminal": True,
                "collection_status": state,
                "scientific_decision": payload["scientific_decision"],
                "result_path": stage["result_path"],
                "stages_run": [],
                "existing_result_retained": True,
            }

    if first_missing is None:
        return {
            "status": "POSTCAMPAIGN_COMPLETE",
            "terminal": True,
            "collection_status": state,
            "stages_run": [],
            "existing_results_validated": list(STAGE_ORDER),
        }
    if dry_run:
        return {
            "status": "DRY_RUN_POSTCAMPAIGN",
            "terminal": False,
            "collection_status": state,
            "planned_stages": [row["name"] for row in stages[first_missing:]],
            "existing_results_validated": [
                stages[index]["name"] for index in range(first_missing)
            ],
            "stages_run": [],
        }

    ran = []
    for stage in stages[first_missing:]:
        try:
            payload = _run_stage(stage)
            if not Path(stage["result_path"]).is_file():
                raise PostcampaignRefusal(
                    f"{stage['name']} runner returned without its registered result"
                )
            stored = _read_object(Path(stage["result_path"]))
            if stored != payload:
                raise PostcampaignRefusal(
                    f"{stage['name']} returned payload differs from stored result"
                )
            advances = _validate_stage_result(stage, stored)
        except Exception as exc:
            return {
                "status": f"TERMINATE_{stage['name'].upper()}_RUNNER_FAILURE",
                "terminal": True,
                "collection_status": state,
                "errors": [str(exc)],
                "stages_run": ran,
            }
        ran.append(stage["name"])
        if not advances:
            return {
                "status": f"TERMINATE_{stage['name'].upper()}_DECISION",
                "terminal": True,
                "collection_status": state,
                "scientific_decision": stored["scientific_decision"],
                "result_path": stage["result_path"],
                "stages_run": ran,
                "existing_result_retained": True,
            }
    return {
        "status": "POSTCAMPAIGN_COMPLETE",
        "terminal": True,
        "collection_status": state,
        "stages_run": ran,
    }


def run_loop(
    registration_path: Path = DEFAULT_REGISTRATION,
    *,
    dry_run: bool,
    once: bool,
    poll_s: float,
) -> int:
    if poll_s <= 0.0:
        raise PostcampaignRefusal("poll interval must be positive")
    if dry_run and not once:
        raise PostcampaignRefusal("dry-run requires --once")
    while True:
        report = postcampaign_step(registration_path, dry_run=dry_run)
        print(json.dumps(report, sort_keys=True), flush=True)
        if once or report["terminal"]:
            return 2 if report["status"].startswith("TERMINATE_") else 0
        time.sleep(min(float(poll_s), 60.0))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("register")
    freeze.add_argument("--output", type=Path, default=DEFAULT_REGISTRATION)
    freeze.add_argument("--manifest", type=Path, default=campaign.DEFAULT_OUTPUT)
    freeze.add_argument("--results-dir", type=Path, default=campaign.DEFAULT_RESULTS_DIR)
    freeze.add_argument("--queue-status", type=Path, default=dispatcher.DEFAULT_QUEUE_STATUS)
    run = sub.add_parser("run")
    run.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    run.add_argument("--once", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--poll-s", type=float, default=30.0)
    args = parser.parse_args(argv)
    try:
        if args.command == "register":
            payload = register(
                args.output,
                manifest_path=args.manifest,
                results_dir=args.results_dir,
                queue_path=args.queue_status,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        return run_loop(
            args.registration,
            dry_run=args.dry_run,
            once=args.once,
            poll_s=args.poll_s,
        )
    except PostcampaignRefusal as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
