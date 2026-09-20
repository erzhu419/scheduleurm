from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess

import pytest

from algorithm.experiments import critical_gpu_joint95_v2_dispatcher as dispatcher
from algorithm.experiments import critical_gpu_joint95_v2_registration as registration


@pytest.fixture()
def authorized_manifest(tmp_path: Path) -> tuple[Path, dict, Path]:
    results = tmp_path / "v2_results"
    path = tmp_path / "v2_manifest.json"
    report = registration.build_registration(results_dir=results)
    registration.write_registration(report, output=path, results_dir=results)
    authorized = dispatcher.authorize_registered_manifest(
        manifest_path=path, results_dir=results
    )
    return path, authorized, results


def _interval_audit(kind: str, identity: str) -> dict:
    return {
        "status": "PASS",
        "target_node": "gpu_b",
        "measurement_kind": kind,
        "measurement_id": identity,
        "measurement_started_at": 100.0,
        "measurement_finished_at": 110.0,
        "queue_path": "/tmp/queue.json",
        "overlap_count": 0,
        "overlapping_tasks": [],
        "target_queue_tasks_are_external": True,
    }


def _success_payload(item: dict, score: float) -> dict:
    observations = []
    for index, functional_id in enumerate(item["scored_functional_ids"]):
        row_score = float(score) - (8 - index) * 0.001
        predicted = 10.0 + index
        observations.append(
            {
                "functional_id": functional_id,
                "source": (
                    "empty" if functional_id.startswith("empty:") else "loaded"
                ),
                "identity": functional_id.split(":", 1)[1].rsplit(":", 1)[0],
                "functional": functional_id.rsplit(":", 1)[1],
                "actual_time_s": predicted * (1.0 + row_score),
                "frozen_predicted_time_s": predicted,
                "ratio_score": row_score,
            }
        )
    empty_rows = []
    for service_cell_id in item["empty_suite"]["service_cell_ids"]:
        empty_rows.append(
            {
                "registered_service_cell_id": service_cell_id,
                "ready": True,
                "status": "READY",
                "capacity_boundary": False,
                "wave": item["joint_wave"],
                "split_role": item["split_role"],
                "joint95_scheduler_interval_audit": _interval_audit(
                    "empty_cell", service_cell_id
                ),
            }
        )
    loaded_rows = []
    for scenario_id in item["loaded_suite"]["scenario_ids"]:
        loaded_rows.append(
            {
                "scenario_id": scenario_id,
                "ready": True,
                "capacity_boundary": False,
                "wave": item["joint_wave"],
                "split_role": item["split_role"],
                "joint95_scheduler_interval_audit": _interval_audit(
                    "loaded_scenario", scenario_id
                ),
            }
        )
    audits = [
        *(row["joint95_scheduler_interval_audit"] for row in empty_rows),
        *(row["joint95_scheduler_interval_audit"] for row in loaded_rows),
    ]
    return {
        "schema_version": 1,
        "gate": dispatcher.RESULT_GATE,
        "protocol": registration.PROTOCOL,
        "status": "PASS",
        "pass": True,
        "terminal": True,
        "manifest_registered": True,
        "joint_unit_id": item["joint_unit_id"],
        "joint_wave": item["joint_wave"],
        "split_role": item["split_role"],
        "suite_order": item["suite_order"],
        "measurement_bundles_unchanged": True,
        "empty_rows": empty_rows,
        "loaded_campaign": {"rows": loaded_rows},
        "scheduler_interval_audits": audits,
        "observations": observations,
        "joint_wave_score": float(score),
        "v1_joint_units_used": 0,
    }


def _write_success(results: Path, item: dict, score: float) -> Path:
    results.mkdir(parents=True, exist_ok=True)
    path = results / f"{item['joint_unit_id']}_attempt01.json"
    path.write_text(json.dumps(_success_payload(item, score)), encoding="utf-8")
    return path


def _ready_snapshots(tmp_path: Path) -> tuple[Path, Path]:
    node = tmp_path / "nodes.json"
    node.write_text(
        json.dumps(
            {
                "ts": 10.0,
                "states": {
                    "gpu_b": {
                        "alive": True,
                        "gpus": [
                            {"idx": 0, "used_mb": 100, "util_pct": 0},
                            {"idx": 1, "used_mb": 10, "util_pct": 0},
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    queue = tmp_path / "queue.json"
    queue.write_text(json.dumps({"tasks": []}), encoding="utf-8")
    return node, queue


def test_authorization_preserves_explicit_no_data_state(tmp_path: Path):
    results = tmp_path / "v2_results"
    path = tmp_path / "v2_manifest.json"
    report = registration.build_registration(results_dir=results)
    waiting = registration.write_registration(
        report, output=path, results_dir=results
    )
    with pytest.raises(dispatcher.DispatchRefusal, match="not ready"):
        dispatcher.load_registered_manifest(path)

    authorized = dispatcher.authorize_registered_manifest(
        manifest_path=path, results_dir=results
    )

    assert waiting["launch_authorized"] is False
    assert authorized["launch_authorized"] is True
    assert authorized["validation"]["pass"] is True
    assert authorized["observations"] == []
    assert authorized["completed_joint_units"] == 0
    assert authorized["measurements_executed"] is False
    assert authorized["authorization_snapshot"]["results_file_count"] == 0
    assert list(results.iterdir()) == []


def test_refresh_authorization_updates_current_contract_only_while_empty(
    authorized_manifest: tuple[Path, dict, Path],
):
    manifest_path, authorized, results = authorized_manifest
    stale = copy.deepcopy(authorized)
    stale["executor_validation"]["validated_at_utc"] = "2000-01-01T00:00:00+00:00"
    stale["executor_validation"]["validated_scope"] = ["superseded scope"]
    stale["scheduler_interval_guard"].pop("terminal_history_refresh")
    manifest_path.write_text(json.dumps(stale), encoding="utf-8")

    refreshed = dispatcher.refresh_registered_authorization(
        manifest_path=manifest_path,
        results_dir=results,
    )

    assert refreshed["launch_authorized"] is True
    assert refreshed["status"] == "REGISTERED_READY_FOR_SEPARATE_LAUNCH"
    assert refreshed["executor_validation"]["validated_at_utc"] != (
        "2000-01-01T00:00:00+00:00"
    )
    assert refreshed["executor_validation"]["validated_scope"] == list(
        registration.V2_EXECUTOR_TEST_SCOPE
    )
    assert refreshed["scheduler_interval_guard"] == (
        registration._expected_interval_guard_contract()
    )
    assert refreshed["authorization_snapshot"]["results_file_count"] == 0
    assert refreshed["validation"]["pass"] is True


def test_refresh_authorization_refuses_any_nonempty_results_directory(
    authorized_manifest: tuple[Path, dict, Path],
):
    manifest_path, _, results = authorized_manifest
    results.mkdir(exist_ok=True)
    (results / "any-result.json").write_text("{}\n", encoding="utf-8")
    before = manifest_path.read_bytes()

    with pytest.raises(dispatcher.DispatchRefusal, match="results directory to be empty"):
        dispatcher.refresh_registered_authorization(
            manifest_path=manifest_path,
            results_dir=results,
        )

    assert manifest_path.read_bytes() == before


@pytest.mark.parametrize(
    ("field", "value"),
    (("observations", [{"ratio_score": 0.1}]), ("completed_joint_units", 1)),
)
def test_refresh_authorization_refuses_nonzero_empirical_manifest_state(
    authorized_manifest: tuple[Path, dict, Path],
    field: str,
    value: object,
):
    manifest_path, authorized, results = authorized_manifest
    changed = copy.deepcopy(authorized)
    changed[field] = value
    manifest_path.write_text(json.dumps(changed), encoding="utf-8")

    with pytest.raises(dispatcher.DispatchRefusal, match="empirical state is not empty"):
        dispatcher.refresh_registered_authorization(
            manifest_path=manifest_path,
            results_dir=results,
        )


def test_manifest_ordering_starts_at_wave34_and_gaps_fail_closed(
    authorized_manifest: tuple[Path, dict, Path],
):
    _, manifest, results = authorized_manifest
    items = {row["joint_wave"]: row for row in manifest["joint_work_items"]}
    status = dispatcher.collection_status(manifest, results)
    assert status["status"] == "READY_CALIBRATION"
    assert status["next_wave"] == 34

    _write_success(results, items[34], 0.10)
    _write_success(results, items[35], 0.11)
    assert dispatcher.collection_status(manifest, results)["next_wave"] == 36

    _write_success(results, items[37], 0.12)
    broken = dispatcher.collection_status(manifest, results)
    assert broken["status"] == "FAIL_RESULT_STATE"
    assert broken["next_wave"] is None
    assert any("out-of-order" in error for error in broken["errors"])


def test_v1_result_is_rejected_and_cannot_anchor_or_calibrate_v2(
    authorized_manifest: tuple[Path, dict, Path],
):
    _, manifest, results = authorized_manifest
    results.mkdir(exist_ok=True)
    v1 = results / "critical_gpu_joint95_v1_gpu_b_r14_attempt01.json"
    v1.write_text(json.dumps({"status": "PASS", "joint_wave_score": -9.0}))

    status = dispatcher.collection_status(manifest, results)

    assert status["status"] == "FAIL_RESULT_STATE"
    assert status["successful_calibration_count"] == 0
    assert status["predictor_anchor_wave"] is None
    assert status["foreign_result_paths"] == [str(v1.resolve())]
    assert status["next_wave"] is None


def test_unknown_json_in_bound_results_directory_fails_closed(
    authorized_manifest: tuple[Path, dict, Path],
):
    _, manifest, results = authorized_manifest
    results.mkdir(exist_ok=True)
    unknown = results / "copied_result.json"
    unknown.write_text(json.dumps({"status": "PASS"}), encoding="utf-8")

    status = dispatcher.collection_status(manifest, results)

    assert status["status"] == "FAIL_RESULT_STATE"
    assert status["foreign_result_paths"] == [str(unknown.resolve())]
    assert status["next_wave"] is None


def test_attempt02_cannot_count_as_the_first_v2_attempt(
    authorized_manifest: tuple[Path, dict, Path],
):
    _, manifest, results = authorized_manifest
    item = manifest["joint_work_items"][0]
    results.mkdir(exist_ok=True)
    path = results / f"{item['joint_unit_id']}_attempt02.json"
    path.write_text(json.dumps(_success_payload(item, 0.1)), encoding="utf-8")

    status = dispatcher.collection_status(manifest, results)

    assert status["status"] == "FAIL_RESULT_STATE"
    assert status["successful_calibration_count"] == 0
    assert status["waves"]["34"]["invalid_passes"]
    assert any(
        "first-attempt-only result" in error
        for error in status["waves"]["34"]["invalid_passes"][0]["errors"]
    )


def test_final_interval_refresh_catches_late_scheduler_adoption(
    authorized_manifest: tuple[Path, dict, Path],
    tmp_path: Path,
):
    _, manifest, _ = authorized_manifest
    item = manifest["joint_work_items"][0]
    report = _success_payload(item, 0.1)
    queue = tmp_path / "queue.json"
    queue.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "id": "t-late",
                        "status": "done",
                        "node": "gpu_b",
                        "started_at": 105.0,
                        "finished_at": 106.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    audits = dispatcher._refresh_scheduler_interval_audits(
        report=report,
        item=item,
        queue_path=queue,
    )

    assert len(audits) == 9
    assert all(audit["status"] == "FAIL_OVERLAP" for audit in audits)
    assert all(audit["overlap_count"] == 1 for audit in audits)
    row_audits = [
        *(row["joint95_scheduler_interval_audit"] for row in report["empty_rows"]),
        *(
            row["joint95_scheduler_interval_audit"]
            for row in report["loaded_campaign"]["rows"]
        ),
    ]
    assert row_audits == report["scheduler_interval_audits"] == audits


def test_all_nine_interval_audits_are_mandatory(
    authorized_manifest: tuple[Path, dict, Path],
):
    _, manifest, results = authorized_manifest
    item = manifest["joint_work_items"][0]
    payload = _success_payload(item, 0.2)
    assert dispatcher.validate_success_result(payload, item)["pass"] is True

    missing = copy.deepcopy(payload)
    missing["scheduler_interval_audits"].pop()
    audit = dispatcher.validate_success_result(missing, item)
    assert audit["pass"] is False
    assert any("nine scheduler interval audits" in error for error in audit["errors"])

    overlap = copy.deepcopy(payload)
    overlap["empty_rows"][0]["joint95_scheduler_interval_audit"][
        "status"
    ] = "FAIL_OVERLAP"
    overlap["empty_rows"][0]["joint95_scheduler_interval_audit"][
        "overlap_count"
    ] = 1
    audit = dispatcher.validate_success_result(overlap, item)
    assert audit["pass"] is False
    assert any("row-level" in error for error in audit["errors"])


def test_holdout_lock_and_release_use_only_19_v2_calibrations(
    authorized_manifest: tuple[Path, dict, Path],
):
    manifest_path, manifest, results = authorized_manifest
    calibration, holdout = dispatcher._campaign_shape(manifest)
    items = {row["joint_wave"]: row for row in manifest["joint_work_items"]}
    for index, wave in enumerate(calibration[:-1], start=1):
        _write_success(results, items[wave], index / 100.0)
    with pytest.raises(dispatcher.DispatchRefusal, match="expected 19"):
        dispatcher.release_holdout(
            manifest_path=manifest_path, results_dir=results
        )

    _write_success(results, items[calibration[-1]], 0.19)
    locked = dispatcher.collection_status(manifest, results)
    assert locked["status"] == "HOLDOUT_LOCKED"
    assert locked["next_wave"] is None

    release = dispatcher.release_holdout(
        manifest_path=manifest_path, results_dir=results
    )
    assert release["calibration_waves"] == list(range(34, 53))
    assert release["holdout_wave"] == 53 == holdout
    assert release["finite_sample_rank"] == 19
    assert release["frozen_margin"] == pytest.approx(0.19)
    assert release["v1_joint_units_used"] == 0
    assert all(
        identity.startswith("critical_gpu_joint95_v2")
        for identity in release["calibration_joint_unit_ids"]
    )
    ready = dispatcher.collection_status(manifest, results)
    assert ready["status"] == "READY_HOLDOUT"
    assert ready["next_wave"] == 53


def test_dry_run_plans_one_wave34_without_scheduler_mutation(
    authorized_manifest: tuple[Path, dict, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    manifest_path, _, results = authorized_manifest
    node, queue = _ready_snapshots(tmp_path)

    def forbidden_run(*args, **kwargs):
        raise AssertionError("dry-run must not call scheduler")

    monkeypatch.setattr(dispatcher.subprocess, "run", forbidden_run)
    report = dispatcher.submit_next(
        manifest_path=manifest_path,
        results_dir=results,
        queue_path=queue,
        node_status_path=node,
        dry_run=True,
    )

    assert report["status"] == "DRY_RUN"
    assert report["joint_wave"] == 34
    assert report["single_step"] is True
    assert report["target_readiness"]["ready"] is True
    worker = report["worker_argv"]
    assert registration.V2_EXECUTOR_MODULE in worker
    assert worker[worker.index("--wave") + 1] == "34"
    assert worker[worker.index("--queue-status") + 1] == str(queue.resolve())
    assert list(results.iterdir()) == []


def test_submit_invokes_scheduler_exactly_once(
    authorized_manifest: tuple[Path, dict, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    manifest_path, _, results = authorized_manifest
    node, queue = _ready_snapshots(tmp_path)
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, "submitted t99991  cpu=1\n", "")

    monkeypatch.setattr(dispatcher.subprocess, "run", fake_run)
    report = dispatcher.submit_next(
        manifest_path=manifest_path,
        results_dir=results,
        queue_path=queue,
        node_status_path=node,
        dry_run=False,
    )

    assert report["status"] == "SUBMITTED_ONE"
    assert report["task_id"] == "t99991"
    assert report["joint_wave"] == 34
    assert len(calls) == 1


def test_fail_stop_sentinel_blocks_submit_and_holdout_release(
    authorized_manifest: tuple[Path, dict, Path],
    tmp_path: Path,
):
    manifest_path, manifest, results = authorized_manifest
    results.mkdir(exist_ok=True)
    sentinel = results / dispatcher.FAIL_STOP_FILENAME
    sentinel.write_text(
        json.dumps(
            {
                "status": "FAIL_STOP",
                "protocol": registration.PROTOCOL,
                "reason": "target-node task overlapped v2 measurement",
                "failed_wave": 34,
            }
        ),
        encoding="utf-8",
    )
    node, queue = _ready_snapshots(tmp_path)

    status = dispatcher.collection_status(manifest, results)
    assert status["status"] == "FAIL_STOP"
    assert status["terminal"] is True
    assert status["next_wave"] is None
    with pytest.raises(dispatcher.DispatchRefusal, match="fail-stop"):
        dispatcher.build_submit_plan(
            manifest_path=manifest_path,
            results_dir=results,
            queue_path=queue,
            node_status_path=node,
        )
    with pytest.raises(dispatcher.DispatchRefusal, match="fail-stopped"):
        dispatcher.release_holdout(
            manifest_path=manifest_path, results_dir=results
        )


def test_second_attempt_is_never_planned(
    authorized_manifest: tuple[Path, dict, Path],
):
    _, manifest, results = authorized_manifest
    item = manifest["joint_work_items"][0]
    results.mkdir(exist_ok=True)
    attempt = results / f"{item['joint_unit_id']}_attempt01.json"
    attempt.write_text(
        json.dumps({"status": "FAILED", "terminal": True, "pass": False}),
        encoding="utf-8",
    )

    status = dispatcher.collection_status(manifest, results)
    assert status["status"] == "FAIL_RESULT_STATE"
    assert status["next_wave"] is None
    with pytest.raises(dispatcher.DispatchRefusal, match="first-attempt-only"):
        dispatcher.next_attempt_path(results, item)


def test_execute_requires_explicit_launch_before_creating_a_result(
    authorized_manifest: tuple[Path, dict, Path],
):
    manifest_path, manifest, results = authorized_manifest
    item = manifest["joint_work_items"][0]
    output = results / f"{item['joint_unit_id']}_attempt01.json"

    with pytest.raises(dispatcher.DispatchRefusal, match="explicit --allow-launch"):
        dispatcher.execute_wave(
            manifest_path=manifest_path,
            results_dir=results,
            wave=34,
            output=output,
            allow_launch=False,
        )
    assert not output.exists()


class _Reservation:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __enter__(self):
        return {"reservation_id": "test-v2-reservation"}

    def __exit__(self, exc_type, exc, traceback):
        return False

    def ensure_healthy(self):
        return None


def test_failed_first_execution_writes_terminal_campaign_sentinel(
    authorized_manifest: tuple[Path, dict, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    manifest_path, manifest, results = authorized_manifest
    item = manifest["joint_work_items"][0]
    output = results / f"{item['joint_unit_id']}_attempt01.json"
    _, queue = _ready_snapshots(tmp_path)

    monkeypatch.setattr(dispatcher, "MeasurementReservation", _Reservation)
    monkeypatch.setattr(dispatcher, "_wait_for_scheduler_ack", lambda **kwargs: None)
    monkeypatch.setattr(
        dispatcher,
        "_wait_for_clean_node",
        lambda **kwargs: {"gpu_idle": {"ready": True}, "active_tasks": []},
    )
    monkeypatch.setattr(
        dispatcher,
        "_run_empty_suite",
        lambda **kwargs: (_ for _ in ()).throw(
            dispatcher.DispatchRefusal("simulated first-attempt failure")
        ),
    )

    report = dispatcher.execute_wave(
        manifest_path=manifest_path,
        results_dir=results,
        wave=34,
        output=output,
        allow_launch=True,
        poll_s=0.0,
        queue_path=queue,
    )

    assert report["status"] == "FAILED"
    sentinel = json.loads(
        (results / dispatcher.FAIL_STOP_FILENAME).read_text(encoding="utf-8")
    )
    assert sentinel["status"] == "FAIL_STOP"
    assert sentinel["decision"] == "JOINT95_V2_FIRST_ATTEMPT_FAIL_STOP"
    assert sentinel["failed_wave"] == 34
    assert sentinel["replacement_attempt_authorized"] is False
    assert dispatcher.collection_status(manifest, results)["status"] == "FAIL_STOP"
