from __future__ import annotations

from pathlib import Path
import json

import pytest

from algorithm.experiments import critical_gpu_joint95_v4_dispatcher as dispatcher
from algorithm.experiments import critical_gpu_joint95_v4_registration as registration
from algorithm.experiments import critical_gpu_joint95_v4_supervisor as supervisor


def _registered_control(tmp_path: Path) -> Path:
    manifest = tmp_path / "manifest.json"
    results = tmp_path / "results"
    registration.write_registration(
        registration.build_registration(results_dir=results),
        output=manifest,
        results_dir=results,
    )
    dispatcher.authorize_registered_manifest(manifest_path=manifest, results_dir=results)
    output = tmp_path / "supervisor.json"
    queue = tmp_path / "queue.json"
    queue.write_text(json.dumps({"tasks": []}), encoding="utf-8")
    supervisor.register(output, manifest_path=manifest, results_dir=results, queue_path=queue)
    return output


def test_supervisor_first_step_plans_only_wave74(tmp_path: Path):
    protocol = _registered_control(tmp_path)
    step = supervisor.supervisor_step(protocol, dry_run=True)
    assert step["status"] == "DRY_RUN_SUBMIT_ONE"
    assert step["joint_wave"] == 74
    assert step["mutations"] == 0


def test_active_controller_masks_partial_result_until_controller_finishes(tmp_path: Path, monkeypatch):
    protocol = _registered_control(tmp_path)
    monkeypatch.setattr(
        supervisor,
        "_active_v4_collection_tasks",
        lambda _: [{"id": "t-current", "signature": "ScheduleurmBench/critical_gpu_joint95_v4/current"}],
    )
    monkeypatch.setattr(
        supervisor.dispatcher,
        "collection_status",
        lambda *args: (_ for _ in ()).throw(AssertionError("collection read during active controller")),
    )
    step = supervisor.supervisor_step(protocol, dry_run=True)
    assert step["status"] == "WAIT_ACTIVE_CONTROLLER"
    assert step["active_controller_ids"] == ["t-current"]
    assert step["mutations"] == 0


def test_terminal_controller_without_success_is_not_retried(tmp_path: Path, monkeypatch):
    protocol = _registered_control(tmp_path)
    monkeypatch.setattr(
        supervisor,
        "_terminal_controller_history",
        lambda *args: [{"id": "t-failed", "status": "failed"}],
    )
    step = supervisor.supervisor_step(protocol, dry_run=True)
    assert step["status"] == "TERMINATE_CONTROLLER_FAILURE"
    assert step["expected_wave"] == 74
    assert step["retry_attempted"] is False
    assert step["mutations"] == 0


def test_valid_fail_stop_releases_node_claim_only_after_children_quiesce(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    protocol_path = _registered_control(tmp_path)
    protocol = json.loads(protocol_path.read_text())
    manifest = dispatcher.load_registered_manifest(Path(protocol["source_paths"]["manifest"]))
    results = Path(protocol["source_paths"]["results_directory"])
    item = manifest["joint_work_items"][0]
    attempt = results / f"{item['joint_unit_id']}_attempt01.json"
    attempt.write_text(json.dumps({
        "gate": dispatcher.RESULT_GATE,
        "protocol": registration.PROTOCOL,
        "joint_unit_id": item["joint_unit_id"],
        "joint_wave": item["joint_wave"],
        "status": "FAILED", "terminal": True, "pass": False,
    }))
    (results / dispatcher.FAIL_STOP_FILENAME).write_text(json.dumps({
        "gate": dispatcher.FAIL_STOP_GATE,
        "protocol": registration.PROTOCOL,
        "status": "FAIL_STOP", "terminal": True, "pass": False,
        "decision": "JOINT95_V4_FIRST_ATTEMPT_FAIL_STOP",
        "failed_wave": item["joint_wave"],
        "failed_joint_unit_id": item["joint_unit_id"],
        "failed_attempt_path": str(attempt),
        "replacement_attempt_authorized": False,
        "holdout_release_authorized": False,
        "v1_joint_units_used": 0, "v2_joint_units_used": 0, "v3_joint_units_used": 0,
    }))
    monkeypatch.setattr(supervisor, "_active_scheduler_tasks", lambda *args, **kwargs: [])
    monkeypatch.setattr(supervisor.empty_suite, "_gpu_idle_snapshot", lambda *args: {"ready": True})
    marker = Path(protocol["scheduler_placement"]["reserve_until_file"])
    monkeypatch.setattr(supervisor, "_active_v4_collection_tasks", lambda *args: [{"id": "t-wave"}])
    with pytest.raises(supervisor.SupervisorRefusal, match="controller is active"):
        supervisor.release_node_reservation_if_terminal(protocol_path, "TERMINATE_FAIL_STOP")
    assert not marker.exists()
    monkeypatch.setattr(supervisor, "_active_v4_collection_tasks", lambda *args: [])
    released = supervisor.release_node_reservation_if_terminal(protocol_path, "TERMINATE_FAIL_STOP")
    assert released["reservation_release_marker"] == str(marker)
    assert marker.stat().st_size > 0


def test_unvalidated_fail_stop_cannot_release_node_claim(tmp_path: Path):
    protocol_path = _registered_control(tmp_path)
    protocol = json.loads(protocol_path.read_text())
    results = Path(protocol["source_paths"]["results_directory"])
    (results / dispatcher.FAIL_STOP_FILENAME).write_text(json.dumps({
        "status": "FAIL_STOP", "reason": "fabricated",
    }))
    with pytest.raises(supervisor.SupervisorRefusal, match="not valid"):
        supervisor.release_node_reservation_if_terminal(protocol_path, "TERMINATE_FAIL_STOP")
    assert not Path(protocol["scheduler_placement"]["reserve_until_file"]).exists()
