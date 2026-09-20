from __future__ import annotations

from pathlib import Path
import json

from algorithm.experiments import critical_gpu_joint95_v3_dispatcher as dispatcher
from algorithm.experiments import critical_gpu_joint95_v3_registration as registration
from algorithm.experiments import critical_gpu_joint95_v3_supervisor as supervisor


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


def test_supervisor_first_step_plans_only_wave54(tmp_path: Path):
    protocol = _registered_control(tmp_path)
    step = supervisor.supervisor_step(protocol, dry_run=True)
    assert step["status"] == "DRY_RUN_SUBMIT_ONE"
    assert step["joint_wave"] == 54
    assert step["mutations"] == 0


def test_active_controller_masks_partial_result_until_controller_finishes(tmp_path: Path, monkeypatch):
    protocol = _registered_control(tmp_path)
    monkeypatch.setattr(
        supervisor,
        "_active_v3_collection_tasks",
        lambda _: [{"id": "t-current", "signature": "ScheduleurmBench/critical_gpu_joint95_v3/current"}],
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
    assert step["expected_wave"] == 54
    assert step["retry_attempted"] is False
    assert step["mutations"] == 0
