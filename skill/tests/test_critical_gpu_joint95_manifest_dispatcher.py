from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess

import pytest

import algorithm.experiments.critical_gpu_joint95_manifest_dispatcher as dispatcher
from algorithm.experiments.critical_gpu_joint95_collection_runner import (
    CALIBRATION_WAVES,
    HOLDOUT_WAVE,
    build_collection_manifest,
)


@pytest.fixture()
def registered_manifest(tmp_path: Path) -> tuple[Path, dict]:
    payload = build_collection_manifest()
    payload["registered"] = True
    payload["status"] = "REGISTERED_READY_FOR_SEPARATE_LAUNCH"
    path = tmp_path / "registered.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload


def _success_payload(item: dict, score: float) -> dict:
    observations = []
    for index, functional_id in enumerate(item["scored_functional_ids"]):
        row_score = float(score) - (8 - index) * 0.001
        predicted = 10.0 + index
        observations.append(
            {
                "functional_id": functional_id,
                "source": "empty" if functional_id.startswith("empty:") else "loaded",
                "identity": functional_id.split(":", 1)[1].rsplit(":", 1)[0],
                "functional": functional_id.rsplit(":", 1)[1],
                "actual_time_s": predicted * (1.0 + row_score),
                "frozen_predicted_time_s": predicted,
                "ratio_score": row_score,
            }
        )
    return {
        "schema_version": 1,
        "gate": dispatcher.RESULT_GATE,
        "protocol": dispatcher.PROTOCOL,
        "status": "PASS",
        "pass": True,
        "terminal": True,
        "manifest_registered": True,
        "joint_unit_id": item["joint_unit_id"],
        "joint_wave": item["joint_wave"],
        "split_role": item["split_role"],
        "suite_order": item["suite_order"],
        "measurement_bundles_unchanged": True,
        "empty_rows": [
            {
                "registered_service_cell_id": service_cell_id,
                "ready": True,
                "status": "READY",
                "capacity_boundary": False,
                "wave": item["joint_wave"],
                "split_role": item["split_role"],
            }
            for service_cell_id in item["empty_suite"]["service_cell_ids"]
        ],
        "loaded_campaign": {
            "rows": [
                {
                    "scenario_id": scenario_id,
                    "ready": True,
                    "capacity_boundary": False,
                    "wave": item["joint_wave"],
                    "split_role": item["split_role"],
                }
                for scenario_id in item["loaded_suite"]["scenario_ids"]
            ]
        },
        "observations": observations,
        "joint_wave_score": float(score),
    }


def _write_success(results_dir: Path, item: dict, score: float) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{item['joint_unit_id']}_attempt01.json"
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


def test_host_preflight_detects_wave59_style_cpu_process(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        dispatcher.empty_suite,
        "_run_remote_capture",
        lambda *args, **kwargs: (
            0,
            "__NPROC__ 24\n"
            "3762240 3762240 650.0 400000 python\n"
            "3762250 3762250 0.0 1024 ps\n",
            "",
        ),
    )

    snapshot = dispatcher._host_cpu_preflight_snapshot("gpu_b")

    assert snapshot["ready"] is False
    assert snapshot["host_nproc"] == 24
    assert snapshot["unapproved_cpu_fraction"] == pytest.approx(650 / 2400)
    assert snapshot["unapproved_cpu_processes"][0]["pid"] == 3762240


def test_joint_preflight_resets_consecutive_samples_after_busy_host(
    monkeypatch: pytest.MonkeyPatch,
):
    samples = iter((
        {"ready": True, "unapproved_cpu_fraction": 0.0},
        {"ready": False, "unapproved_cpu_fraction": 650 / 2400,
         "unapproved_cpu_processes": [{"pid": 3762240, "cpu_pct": 650.0}]},
        {"ready": True, "unapproved_cpu_fraction": 0.0},
        {"ready": True, "unapproved_cpu_fraction": 0.0},
    ))
    calls = []

    class Lease:
        def ensure_healthy(self):
            calls.append("sample")

    monkeypatch.setattr(dispatcher, "_active_scheduler_tasks", lambda node: [])
    monkeypatch.setattr(
        dispatcher.empty_suite,
        "_gpu_idle_snapshot",
        lambda node_spec: {"ready": True},
    )
    monkeypatch.setattr(
        dispatcher,
        "_host_cpu_preflight_snapshot",
        lambda node: next(samples),
    )
    monkeypatch.setattr(dispatcher.time, "sleep", lambda seconds: None)

    preflight = dispatcher._wait_for_clean_node(
        node="gpu_b",
        lease=Lease(),
        poll_s=0.0,
        consecutive_samples=2,
    )

    assert len(calls) == 4
    assert preflight["host_cpu"]["ready"] is True
    assert preflight["blocked_sample_count"] == 1
    assert preflight["last_blocking_host_cpu"]["unapproved_cpu_processes"][0]["pid"] == 3762240


def test_consumer_rejects_an_unregistered_or_nonregistered_status(
    registered_manifest: tuple[Path, dict],
):
    path, payload = registered_manifest
    unregistered = copy.deepcopy(payload)
    unregistered["registered"] = False
    path.write_text(json.dumps(unregistered), encoding="utf-8")
    with pytest.raises(dispatcher.DispatchRefusal, match="not formally registered"):
        dispatcher.load_registered_manifest(path)

    unregistered["registered"] = True
    unregistered["status"] = "READY_TO_REGISTER"
    path.write_text(json.dumps(unregistered), encoding="utf-8")
    with pytest.raises(dispatcher.DispatchRefusal, match="status is not registered"):
        dispatcher.load_registered_manifest(path)


def test_dry_run_plans_only_wave14_on_the_local_controller(
    registered_manifest: tuple[Path, dict],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    manifest_path, _ = registered_manifest
    node, queue = _ready_snapshots(tmp_path)
    results = tmp_path / "results"

    def forbidden_run(*args, **kwargs):
        raise AssertionError("dry-run must not call the scheduler")

    monkeypatch.setattr(dispatcher.subprocess, "run", forbidden_run)
    report = dispatcher.submit_next(
        manifest_path=manifest_path,
        results_dir=results,
        queue_path=queue,
        node_status_path=node,
        dry_run=True,
    )

    assert report["status"] == "DRY_RUN"
    assert report["joint_wave"] == 14
    assert report["single_step"] is True
    assert report["target_readiness"]["ready"] is True
    argv = report["scheduler_argv"]
    assert argv[argv.index("--require-node") + 1] == "local"
    assert argv[argv.index("--vram") + 1] == "0"
    assert "--wave 14" in argv[argv.index("--cmd") + 1]
    assert not results.exists()


def test_submit_invokes_scheduler_once_and_records_one_task_id(
    registered_manifest: tuple[Path, dict],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    manifest_path, _ = registered_manifest
    node, queue = _ready_snapshots(tmp_path)
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, "submitted t12345  cpu=1\n", "")

    monkeypatch.setattr(dispatcher.subprocess, "run", fake_run)
    report = dispatcher.submit_next(
        manifest_path=manifest_path,
        results_dir=tmp_path / "results",
        queue_path=queue,
        node_status_path=node,
        dry_run=False,
    )

    assert report["status"] == "SUBMITTED_ONE"
    assert report["task_id"] == "t12345"
    assert report["joint_wave"] == 14
    assert len(calls) == 1


def test_first_missing_calibration_controls_order_and_gaps_fail_closed(
    registered_manifest: tuple[Path, dict],
    tmp_path: Path,
):
    _, manifest = registered_manifest
    results = tmp_path / "results"
    items = {row["joint_wave"]: row for row in manifest["joint_work_items"]}
    _write_success(results, items[14], 0.10)
    _write_success(results, items[15], 0.11)

    status = dispatcher.collection_status(manifest, results)
    assert status["status"] == "READY_CALIBRATION"
    assert status["next_wave"] == 16

    _write_success(results, items[17], 0.12)
    status = dispatcher.collection_status(manifest, results)
    assert status["status"] == "FAIL_RESULT_STATE"
    assert any("out-of-order" in error for error in status["errors"])


def test_successful_visits_must_share_wave14_predictor_vector(
    registered_manifest: tuple[Path, dict],
    tmp_path: Path,
):
    _, manifest = registered_manifest
    results = tmp_path / "results"
    items = {row["joint_wave"]: row for row in manifest["joint_work_items"]}
    _write_success(results, items[14], 0.10)
    second = _success_payload(items[15], 0.11)
    second["observations"][0]["frozen_predicted_time_s"] += 1.0
    predicted = second["observations"][0]["frozen_predicted_time_s"]
    score = second["observations"][0]["ratio_score"]
    second["observations"][0]["actual_time_s"] = predicted * (1.0 + score)
    (results / f"{items[15]['joint_unit_id']}_attempt01.json").write_text(
        json.dumps(second), encoding="utf-8"
    )

    status = dispatcher.collection_status(manifest, results)

    assert status["status"] == "FAIL_RESULT_STATE"
    assert status["predictor_anchor_wave"] == 14
    assert any("predictor vector" in error for error in status["errors"])


def test_active_joint_controller_blocks_another_submit(
    registered_manifest: tuple[Path, dict],
    tmp_path: Path,
):
    manifest_path, _ = registered_manifest
    node, queue = _ready_snapshots(tmp_path)
    queue.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "id": "t1",
                        "status": "queued",
                        "signature": dispatcher.COLLECTION_SIGNATURE_PREFIX + "wave14",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(dispatcher.DispatchRefusal, match="already active"):
        dispatcher.build_submit_plan(
            manifest_path=manifest_path,
            results_dir=tmp_path / "results",
            queue_path=queue,
            node_status_path=node,
        )


def test_holdout_release_requires_all_19_successes_and_freezes_rank19_margin(
    registered_manifest: tuple[Path, dict],
    tmp_path: Path,
):
    manifest_path, manifest = registered_manifest
    results = tmp_path / "results"
    items = {row["joint_wave"]: row for row in manifest["joint_work_items"]}
    for wave in CALIBRATION_WAVES[:-1]:
        _write_success(results, items[wave], wave / 100.0)
    with pytest.raises(dispatcher.DispatchRefusal, match="expected 19"):
        dispatcher.release_holdout(
            manifest_path=manifest_path,
            results_dir=results,
        )

    _write_success(results, items[CALIBRATION_WAVES[-1]], CALIBRATION_WAVES[-1] / 100.0)
    locked = dispatcher.collection_status(manifest, results)
    assert locked["status"] == "HOLDOUT_LOCKED"
    assert locked["next_wave"] is None

    release = dispatcher.release_holdout(
        manifest_path=manifest_path,
        results_dir=results,
    )
    assert release["calibration_margin_frozen"] is True
    assert release["finite_sample_rank"] == 19
    assert release["frozen_margin"] == CALIBRATION_WAVES[-1] / 100.0
    ready = dispatcher.collection_status(manifest, results)
    assert ready["status"] == "READY_HOLDOUT"
    assert ready["next_wave"] == HOLDOUT_WAVE
    with pytest.raises(FileExistsError):
        dispatcher.release_holdout(
            manifest_path=manifest_path,
            results_dir=results,
        )


def test_invalid_pass_artifact_never_counts_as_a_terminal(
    registered_manifest: tuple[Path, dict],
    tmp_path: Path,
):
    _, manifest = registered_manifest
    item = manifest["joint_work_items"][0]
    payload = _success_payload(item, 0.2)
    payload["observations"].pop()
    results = tmp_path / "results"
    results.mkdir()
    (results / f"{item['joint_unit_id']}_attempt01.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    status = dispatcher.collection_status(manifest, results)
    assert status["status"] == "FAIL_RESULT_STATE"
    assert status["waves"]["14"]["success_count"] == 0
    assert status["waves"]["14"]["invalid_passes"]


def test_fail_stop_sentinel_is_terminal_and_blocks_submit_and_release(
    registered_manifest: tuple[Path, dict],
    tmp_path: Path,
):
    manifest_path, manifest = registered_manifest
    results = tmp_path / "results"
    results.mkdir()
    sentinel = results / dispatcher.FAIL_STOP_FILENAME
    sentinel.write_text(
        json.dumps(
            {
                "status": "FAIL_STOP",
                "reason": "target-node scheduler task overlapped a measured cell",
                "joint_wave": 23,
            }
        ),
        encoding="utf-8",
    )
    node, queue = _ready_snapshots(tmp_path)

    status = dispatcher.collection_status(manifest, results)

    assert status["status"] == "FAIL_STOP"
    assert status["terminal"] is True
    assert status["next_wave"] is None
    assert status["fail_stop"]["path"] == str(sentinel.resolve())
    assert "overlapped" in status["errors"][0]
    with pytest.raises(dispatcher.DispatchRefusal, match="fail-stop sentinel"):
        dispatcher.build_submit_plan(
            manifest_path=manifest_path,
            results_dir=results,
            queue_path=queue,
            node_status_path=node,
        )
    with pytest.raises(dispatcher.DispatchRefusal, match="fail-stopped"):
        dispatcher.release_holdout(
            manifest_path=manifest_path,
            results_dir=results,
        )


def test_continuation_adapter_is_narrow_and_restores_suite_globals():
    old_waves = dispatcher.loaded_suite.ALL_WAVES
    old_loaded_role = dispatcher.loaded_suite._split_role
    old_empty_role = dispatcher.empty_gate.wave_role

    with dispatcher._continuation_wave_role(14, "calibration"):
        assert dispatcher.loaded_suite.ALL_WAVES == old_waves
        assert 14 not in dispatcher.loaded_suite.ALL_WAVES
        assert dispatcher.loaded_suite._split_role(14) == "calibration"
        assert dispatcher.loaded_gate._split_role(14) == "calibration"
        assert dispatcher.empty_gate.wave_role(14) == "calibration"
        assert dispatcher.loaded_suite._split_role(13) == "holdout"

    assert dispatcher.loaded_suite.ALL_WAVES == old_waves
    assert dispatcher.loaded_suite._split_role is old_loaded_role
    assert dispatcher.empty_gate.wave_role is old_empty_role


class _HealthyLease:
    def ensure_healthy(self):
        return None


class _Reservation(_HealthyLease):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __enter__(self):
        return {"reservation_id": "test-reservation"}

    def __exit__(self, exc_type, exc, traceback):
        return False


def test_execute_waits_for_initial_clean_target_before_empty_suite(
    registered_manifest: tuple[Path, dict],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    manifest_path, manifest = registered_manifest
    item = manifest["joint_work_items"][0]
    results = tmp_path / "results"
    output = results / f"{item['joint_unit_id']}_attempt01.json"
    events = []

    monkeypatch.setattr(dispatcher, "MeasurementReservation", _Reservation)
    monkeypatch.setattr(
        dispatcher,
        "_registered_predictors",
        lambda payload: ({row["service_cell_id"]: 1.0 for row in payload["empty_contract"]},
                         {row["scenario_id"]: 1.0 for row in payload["loaded_contract"]}),
    )

    def fake_ack(**kwargs):
        events.append("ack")

    def fake_clean(**kwargs):
        events.append("clean")
        return {"gpu_idle": {"ready": True}, "active_tasks": []}

    def refuse_empty(**kwargs):
        events.append("empty")
        raise dispatcher.DispatchRefusal("test stop after initial drain")

    monkeypatch.setattr(dispatcher, "_wait_for_scheduler_ack", fake_ack)
    monkeypatch.setattr(dispatcher, "_wait_for_clean_node", fake_clean)
    monkeypatch.setattr(dispatcher, "_run_empty_suite", refuse_empty)

    report = dispatcher.execute_wave(
        manifest_path=manifest_path,
        results_dir=results,
        wave=14,
        output=output,
        allow_launch=True,
        poll_s=0.0,
    )

    assert report["status"] == "FAILED"
    assert events == ["ack", "clean", "empty"]
    assert report["initial_clean_preflight"]["gpu_idle"]["ready"] is True


def test_empty_executor_maps_exact_registered_seven_cells_without_builder_launch(
    registered_manifest: tuple[Path, dict],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _, manifest = registered_manifest
    item = manifest["joint_work_items"][0]
    output = tmp_path / "attempt.json"
    output.write_text("{}", encoding="utf-8")
    _, queue = _ready_snapshots(tmp_path)
    monkeypatch.setattr(
        dispatcher.empty_suite,
        "_node_environment_gate",
        lambda node_spec: {"ready": True},
    )
    monkeypatch.setattr(
        dispatcher,
        "_wait_for_clean_node",
        lambda **kwargs: {"gpu_idle": {"ready": True}},
    )
    calls = []

    def fake_run_cell(**kwargs):
        calls.append(kwargs)
        return {
            **kwargs["cell"],
            "measurement_protocol": kwargs["measurement_protocol"],
            "ready": True,
            "status": "READY",
        }

    monkeypatch.setattr(dispatcher.empty_suite, "_run_cell", fake_run_cell)
    contract_by_key = {
        (row["workload_key"], row["profile"]): row
        for row in manifest["empty_contract"]
    }

    def fake_audit(*, row, expected, wave, expected_protocol):
        assert expected is contract_by_key[(row["workload_key"], row["profile"])]
        return [], {
            "service_cell_id": expected["service_cell_id"],
            "drain_completion_s": 1.0,
            "mean_task_jct_s": 0.9,
        }

    monkeypatch.setattr(dispatcher.empty_gate, "_audit_measurement_row", fake_audit)
    with dispatcher._continuation_wave_role(14, "calibration"):
        _, observations = dispatcher._run_empty_suite(
            manifest=manifest,
            item=item,
            lease=_HealthyLease(),
            poll_s=0.0,
            clean_samples=1,
            report={"empty_rows": []},
            output=output,
            queue_path=queue,
        )

    assert len(calls) == len(observations) == 7
    assert [
        contract_by_key[(call["cell"]["workload_key"], call["cell"]["profile"])][
            "service_cell_id"
        ]
        for call in calls
    ] == item["empty_suite"]["service_cell_ids"]
    assert all(call["cell"]["wave"] == 14 for call in calls)
    assert all(call["cell"]["split_role"] == "calibration" for call in calls)
    assert all(call["cleanup_remote_output"] is True for call in calls)
    assert all(
        row["joint95_scheduler_interval_audit"]["status"] == "PASS"
        for row in json.loads(output.read_text(encoding="utf-8"))["empty_rows"]
    )


def test_empty_executor_refuses_task_started_during_cell_and_preserves_evidence(
    registered_manifest: tuple[Path, dict],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _, manifest = registered_manifest
    item = manifest["joint_work_items"][0]
    output = tmp_path / "attempt.json"
    output.write_text("{}", encoding="utf-8")
    _, queue = _ready_snapshots(tmp_path)
    monkeypatch.setattr(
        dispatcher.empty_suite,
        "_node_environment_gate",
        lambda node_spec: {"ready": True},
    )
    monkeypatch.setattr(
        dispatcher,
        "_wait_for_clean_node",
        lambda **kwargs: {"gpu_idle": {"ready": True}},
    )

    def fake_run_cell(**kwargs):
        queue.write_text(
            json.dumps(
                {
                    "tasks": [
                        {
                            "id": "t-overlap",
                            "status": "running",
                            "node": item["node"],
                            "project": "another-project",
                            "started_at": 105.0,
                            "finished_at": None,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return {
            **kwargs["cell"],
            "measurement_protocol": kwargs["measurement_protocol"],
            "ready": True,
            "status": "READY",
        }

    monkeypatch.setattr(dispatcher.empty_suite, "_run_cell", fake_run_cell)
    clock = iter((100.0, 110.0))
    monkeypatch.setattr(dispatcher.time, "time", lambda: next(clock))
    report = {"empty_rows": [], "scheduler_interval_audits": []}

    with dispatcher._continuation_wave_role(14, "calibration"):
        with pytest.raises(dispatcher.DispatchRefusal, match="t-overlap"):
            dispatcher._run_empty_suite(
                manifest=manifest,
                item=item,
                lease=_HealthyLease(),
                poll_s=0.0,
                clean_samples=1,
                report=report,
                output=output,
                queue_path=queue,
            )

    saved = json.loads(output.read_text(encoding="utf-8"))
    audit = saved["scheduler_interval_audits"][0]
    assert audit["status"] == "FAIL_OVERLAP"
    assert audit["measurement_started_at"] == 100.0
    assert audit["measurement_finished_at"] == 110.0
    assert audit["overlapping_tasks"][0]["id"] == "t-overlap"
    assert (
        saved["empty_rows"][0]["joint95_scheduler_interval_audit"]
        == audit
    )


def test_scheduler_interval_audit_allows_nonoverlapping_target_history(tmp_path: Path):
    queue = tmp_path / "queue.json"
    queue.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "id": "t-before",
                        "status": "done",
                        "node": "gpu_b",
                        "started_at": 80.0,
                        "finished_at": 90.0,
                    },
                    {
                        "id": "t-other-node",
                        "status": "running",
                        "node": "local",
                        "started_at": 105.0,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    audit = dispatcher._scheduler_history_interval_audit(
        queue_path=queue,
        node="gpu_b",
        measurement_kind="empty_cell",
        measurement_id="cell-1",
        started_at=100.0,
        finished_at=110.0,
    )

    assert audit["status"] == "PASS"
    assert audit["overlap_count"] == 0
    assert audit["overlapping_tasks"] == []


def test_scheduler_interval_audit_detects_recently_completed_overlap(tmp_path: Path):
    queue = tmp_path / "queue.json"
    queue.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "id": "t-completed-overlap",
                        "status": "done",
                        "last_node": "gpu_b",
                        "started_at": 105.0,
                        "finished_at": 108.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    audit = dispatcher._scheduler_history_interval_audit(
        queue_path=queue,
        node="gpu_b",
        measurement_kind="loaded_scenario",
        measurement_id="scenario-1",
        started_at=100.0,
        finished_at=110.0,
    )

    assert audit["status"] == "FAIL_OVERLAP"
    assert audit["overlap_count"] == 1
    assert audit["overlapping_tasks"][0]["id"] == "t-completed-overlap"
    assert audit["overlapping_tasks"][0]["active_at_audit"] is False


def test_scheduler_interval_audit_recognizes_only_completed_direct_child(tmp_path: Path):
    run_id = "joint95_r34_empty_resnet50_p1"
    label = f"{run_id}_profile_1_per_gpu_gpu0_0"
    completed_row = {
        "run_id": run_id,
        "ready": True,
        "probe_result": {"run_id": run_id, "pass": True},
        "summary": {
            "run_id": run_id,
            "measurement_valid": True,
            "returncode_valid_count": 1,
            "returncode_accepted_count": 1,
            "rows": [
                {
                    "run_name": label,
                    "returncode": 0,
                    "completion_model": {"natural_exit": True},
                }
            ],
        },
        "completion_audit": {"ready": True},
        "remote_cleanup": {"ready": True, "returncode": 0},
        "joint95_scheduler_interval_audit": {"measurement_finished_at": 99.0},
    }
    queue = tmp_path / "queue.json"
    queue.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "id": "t-own-late-adopt",
                        "status": "done",
                        "node": "gpu_b",
                        "started_at": 101.0,
                        "finished_at": 120.0,
                        "auto_adopted": True,
                        "cmd": f"python benchmark.py --label {label} --steps 60",
                    },
                    {
                        "id": "t-other-adopted",
                        "status": "done",
                        "node": "gpu_b",
                        "started_at": 102.0,
                        "finished_at": 108.0,
                        "auto_adopted": True,
                        "cmd": "python benchmark.py --label unrelated_run_profile_1_per_gpu_gpu0_0",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    def audit(row: dict) -> dict:
        return dispatcher._scheduler_history_interval_audit(
            queue_path=queue,
            node="gpu_b",
            measurement_kind="empty_cell",
            measurement_id="resnet50-p3",
            started_at=100.0,
            finished_at=110.0,
            completed_child_rows=[row],
            measurement_run_id="joint95_r34_empty_resnet50_p3",
        )

    result = audit(completed_row)
    assert result["status"] == "FAIL_OVERLAP"
    assert [row["id"] for row in result["overlapping_tasks"]] == ["t-other-adopted"]
    assert result["recognized_measurement_children"][0]["id"] == "t-own-late-adopt"
    assert result["target_queue_tasks_are_external"] is False

    only_child = json.loads(queue.read_text(encoding="utf-8"))
    only_child["tasks"].pop()
    queue.write_text(json.dumps(only_child), encoding="utf-8")
    assert audit(completed_row)["status"] == "PASS"
    only_child["tasks"][0]["auto_adopted"] = False
    queue.write_text(json.dumps(only_child), encoding="utf-8")
    assert audit(completed_row)["overlap_count"] == 1
    only_child["tasks"][0]["auto_adopted"] = True
    only_child["tasks"].append(
        {
            "id": "t-other-adopted",
            "status": "done",
            "node": "gpu_b",
            "started_at": 102.0,
            "finished_at": 108.0,
            "auto_adopted": True,
            "cmd": "python benchmark.py --label unrelated_run_profile_1_per_gpu_gpu0_0",
        }
    )
    queue.write_text(json.dumps(only_child), encoding="utf-8")

    incomplete = copy.deepcopy(completed_row)
    incomplete["summary"]["rows"][0]["returncode"] = 1
    result = audit(incomplete)
    assert result["overlap_count"] == 2
    assert result["recognized_measurement_children"] == []
    assert result["target_queue_tasks_are_external"] is True

    too_late = copy.deepcopy(completed_row)
    too_late["joint95_scheduler_interval_audit"]["measurement_finished_at"] = 101.0
    assert audit(too_late)["overlap_count"] == 2


def test_scheduler_interval_audit_loaded_child_requires_exact_label_and_process_group(
    tmp_path: Path,
):
    queue = tmp_path / "queue.json"
    queue.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "id": "t-loaded-resident",
                        "status": "done",
                        "node": "gpu_b",
                        "started_at": 101.0,
                        "finished_at": 120.0,
                        "auto_adopted": True,
                        "process_group": 451,
                        "cmd": (
                            "python train.py --run_name "
                            "critical_gpu_loaded_gpu_b_cnn_after_rl_r34_resident_gpu0_0"
                        ),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    loaded_row = {
        "scenario_id": "cnn_after_rl",
        "node": "gpu_b",
        "wave": 34,
        "ready": True,
        "remote_returncode": 0,
        "resident_returncode": 0,
        "target_returncode": 0,
        "resident_process_group_id": 451,
        "target_process_group_id": 452,
        "target_start_ns": 1_000,
        "target_end_ns": 2_000,
        "durable_transport": {
            "run_id": "joint95_r34_attempt01_loaded_cnn_after_rl",
            "state": "DONE",
            "remote_returncode": 0,
        },
        "resident_completion_model": {"natural_exit": True},
        "target_completion_model": {"natural_exit": True},
        "joint95_scheduler_interval_audit": {"measurement_finished_at": 99.0},
    }

    def audit(row: dict) -> dict:
        return dispatcher._scheduler_history_interval_audit(
            queue_path=queue,
            node="gpu_b",
            measurement_kind="empty_cell",
            measurement_id="next-cell",
            started_at=100.0,
            finished_at=110.0,
            completed_child_rows=[row],
            measurement_run_id="next-run",
        )

    assert audit(loaded_row)["status"] == "PASS"
    wrong_group = copy.deepcopy(loaded_row)
    wrong_group["resident_process_group_id"] = 999
    assert audit(wrong_group)["overlap_count"] == 1
    incomplete = copy.deepcopy(loaded_row)
    incomplete["resident_returncode"] = 1
    assert audit(incomplete)["overlap_count"] == 1


def test_loaded_executor_uses_two_registered_scenarios_and_wave_rotation(
    registered_manifest: tuple[Path, dict],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _, manifest = registered_manifest
    item = manifest["joint_work_items"][0]
    output = tmp_path / "attempt.json"
    output.write_text("{}", encoding="utf-8")
    _, queue = _ready_snapshots(tmp_path)
    monkeypatch.setattr(
        dispatcher.loaded_suite,
        "_deploy_progress_wrapper",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        dispatcher,
        "_wait_for_clean_node",
        lambda **kwargs: {"gpu_idle": {"ready": True}},
    )
    calls = []

    def fake_run_loaded(**kwargs):
        calls.append(kwargs)
        return {
            "scenario_id": kwargs["scenario"].scenario_id,
            "wave": kwargs["wave"],
            "split_role": dispatcher.loaded_suite._split_role(kwargs["wave"]),
            "gpu": kwargs["gpu"],
            "ready": True,
            "capacity_boundary": False,
        }

    monkeypatch.setattr(dispatcher.loaded_suite, "_run_loaded_trajectory", fake_run_loaded)

    def fake_audit(*, row, expected, wave):
        assert row["scenario_id"] == expected["scenario_id"]
        return [], {
            "scenario_id": row["scenario_id"],
            "target_canonical_completion_s": 1.0,
            "resident_overlap_service_units_per_s": 2.0,
        }

    monkeypatch.setattr(dispatcher.loaded_gate, "_audit_row", fake_audit)
    with dispatcher._continuation_wave_role(14, "calibration"):
        campaign, observations = dispatcher._run_loaded_suite(
            manifest=manifest,
            item=item,
            lease=_HealthyLease(),
            poll_s=0.0,
            clean_samples=1,
            report={"loaded_campaign": None},
            output=output,
            queue_path=queue,
        )

    assert campaign["pass"] is True
    assert len(calls) == len(observations) == 2
    assert [call["scenario"].scenario_id for call in calls] == item["loaded_suite"][
        "scenario_ids"
    ]
    assert [call["gpu"] for call in calls] == [1, 0]
    assert all(call["wave"] == 14 for call in calls)
    assert all(call["cleanup_remote_output"] is True for call in calls)
    assert all(
        row["joint95_scheduler_interval_audit"]["status"] == "PASS"
        for row in campaign["rows"]
    )


def test_execute_without_explicit_launch_refuses_before_creating_results(
    registered_manifest: tuple[Path, dict],
    tmp_path: Path,
):
    manifest_path, _ = registered_manifest
    results = tmp_path / "results"
    output = results / "unused.json"
    with pytest.raises(dispatcher.DispatchRefusal, match="explicit --allow-launch"):
        dispatcher.execute_wave(
            manifest_path=manifest_path,
            results_dir=results,
            wave=14,
            output=output,
            allow_launch=False,
        )
    assert not results.exists()
