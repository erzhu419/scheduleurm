from __future__ import annotations

import json
from pathlib import Path

import pytest

from algorithm.experiments import critical_gpu_joint95_v2_dispatcher as dispatcher
from algorithm.experiments import critical_gpu_joint95_v2_registration as campaign
from algorithm.experiments import critical_gpu_joint95_v2_supervisor as supervisor


@pytest.fixture()
def authorized_campaign(tmp_path: Path) -> tuple[Path, Path]:
    manifest_path = tmp_path / "manifest.json"
    results_dir = tmp_path / "results"
    report = campaign.build_registration(results_dir=results_dir)
    campaign.write_registration(
        report,
        output=manifest_path,
        results_dir=results_dir,
    )
    dispatcher.authorize_registered_manifest(
        manifest_path=manifest_path,
        results_dir=results_dir,
    )
    return manifest_path, results_dir


def test_registration_is_frozen_only_at_zero_result_start(
    authorized_campaign: tuple[Path, Path], tmp_path: Path
):
    manifest_path, results_dir = authorized_campaign
    queue_path = tmp_path / "queue.json"
    queue_path.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "id": "t-v1-running",
                        "status": "running",
                        "signature": (
                            "ScheduleurmBench/critical_gpu_joint95/legacy_unit"
                        ),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    protocol = supervisor.build_registration(
        manifest_path=manifest_path,
        results_dir=results_dir,
        queue_path=queue_path,
        node_status_path=tmp_path / "nodes.json",
        scheduler_path=tmp_path / "scheduler.py",
    )
    assert protocol["status"] == supervisor.REGISTRATION_STATUS
    assert protocol["campaign_contract"]["calibration_waves"] == list(range(34, 53))
    assert protocol["campaign_contract"]["holdout_wave"] == 53
    assert protocol["campaign_contract"]["finite_sample_rank"] == 19
    assert protocol["campaign_contract"]["v1_joint_units_eligible"] == 0
    assert protocol["registration_snapshot"] == {
        "kind": "ZERO_RESULT_START",
        "collection_status_read_while_controller_active": False,
        "results_entry_count": 0,
        "successful_calibration_count": 0,
        "next_wave": 34,
        "scores_read_by_supervisor_registration": 0,
    }
    assert protocol["scheduler_placement"] == {
        "require_node": "local",
        "vram_mb": 0,
        "signature": "ScheduleurmBench/joint95_v2_supervisor_v3",
        "signature_is_outside_joint95_controller_prefix": True,
    }
    assert protocol["execution_scope"][
        "terminal_controller_history_checked_before_every_submission"
    ] is True
    assert protocol["execution_scope"]["retry_authorized"] is False
    assert "joint_wave_score" not in json.dumps(protocol)

    (results_dir / "unexpected.json").write_text("{}", encoding="utf-8")
    with pytest.raises(supervisor.SupervisorRefusal, match="cannot contain v2 attempts"):
        supervisor.build_registration(
            manifest_path=manifest_path,
            results_dir=results_dir,
            queue_path=queue_path,
        )


def test_v2_registration_accepts_active_wave34_zero_measurement_repair(
    authorized_campaign: tuple[Path, Path], tmp_path: Path
):
    manifest_path, results_dir = authorized_campaign
    manifest = json.loads(manifest_path.read_text())
    item = manifest["joint_work_items"][0]
    result = results_dir / f"{item['joint_unit_id']}_attempt01.json"
    result.write_text(
        json.dumps(
            {
                "status": "WAIT_SCHEDULER_ACK",
                "terminal": False,
                "pass": False,
                "joint_wave": 34,
                "joint_unit_id": item["joint_unit_id"],
                "empty_rows": [],
                "loaded_campaign": None,
                "observations": [],
            }
        ),
        encoding="utf-8",
    )
    signature = dispatcher.COLLECTION_SIGNATURE_PREFIX + item["joint_unit_id"]
    queue = tmp_path / "queue.json"
    queue.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "id": "t-v1-running",
                        "status": "running",
                        "signature": (
                            "ScheduleurmBench/critical_gpu_joint95/legacy_unit"
                        ),
                    },
                    {
                        "id": "t-running",
                        "status": "running",
                        "signature": signature,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    report = supervisor.build_registration(
        manifest_path=manifest_path,
        results_dir=results_dir,
        queue_path=queue,
        node_status_path=tmp_path / "nodes.json",
        scheduler_path=tmp_path / "scheduler.py",
    )
    assert report["protocol"].endswith("supervisor_v3")
    assert report["registration_snapshot"] == {
        "kind": "ACTIVE_WAVE34_ZERO_MEASUREMENT_CONTROL_REPAIR",
        "collection_status_read_while_controller_active": False,
        "active_controller_id": "t-running",
        "active_controller_signature": signature,
        "attempt_path": str(result),
        "attempt_status": "WAIT_SCHEDULER_ACK",
        "attempt_terminal": False,
        "measurement_count": 0,
        "successful_calibration_count": 0,
        "scores_read_by_supervisor_registration": 0,
    }


def _protocol() -> dict:
    return {
        "source_paths": {
            "manifest": "/tmp/manifest.json",
            "results_directory": "/tmp/results",
            "queue_status": "/tmp/queue.json",
            "node_status": "/tmp/nodes.json",
            "scheduler": "/tmp/scheduler.py",
            "python_bin": "/usr/bin/python3",
        }
    }


def _patch_context(monkeypatch: pytest.MonkeyPatch) -> dict:
    manifest = {
        "manifest": "v2",
        "joint_work_items": [
            {
                "joint_wave": wave,
                "joint_unit_id": (
                    f"critical_gpu_joint95_v2_jtl110gpu2_r{wave:02d}"
                ),
            }
            for wave in range(34, 54)
        ],
    }
    monkeypatch.setattr(
        supervisor,
        "_load_registration",
        lambda _path: (_protocol(), manifest),
    )
    monkeypatch.setattr(
        supervisor,
        "_terminal_controller_history",
        lambda _path, _signature: [],
    )
    return manifest


def _forbid_mutations(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        dispatcher,
        "submit_next",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("submit_next must not be called")
        ),
    )
    monkeypatch.setattr(
        dispatcher,
        "release_holdout",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("release_holdout must not be called")
        ),
    )


def test_ready_calibration_submits_exactly_one_existing_dispatcher_step(
    monkeypatch: pytest.MonkeyPatch,
):
    manifest = _patch_context(monkeypatch)
    monkeypatch.setattr(
        dispatcher,
        "collection_status",
        lambda actual, _results: {
            "status": "READY_CALIBRATION",
            "errors": [],
            "next_wave": 34,
        }
        if actual is manifest
        else (_ for _ in ()).throw(AssertionError("manifest changed")),
    )
    monkeypatch.setattr(dispatcher, "_active_collection_tasks", lambda _path: [])
    calls = []

    def submit(**kwargs):
        calls.append(kwargs)
        return {"joint_wave": 34, "split_role": "calibration", "task_id": "t1"}

    monkeypatch.setattr(dispatcher, "submit_next", submit)
    monkeypatch.setattr(
        dispatcher,
        "release_holdout",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("calibration cannot release holdout")
        ),
    )
    report = supervisor.supervisor_step(Path("unused"), dry_run=False)
    assert report["status"] == "SUBMITTED_ONE"
    assert report["joint_wave"] == 34
    assert report["task_id"] == "t1"
    assert report["mutations"] == 1
    assert len(calls) == 1
    assert calls[0]["dry_run"] is False


def test_exact_v2_active_controller_waits_without_collection_status(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_context(monkeypatch)
    monkeypatch.setattr(
        dispatcher,
        "collection_status",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("active controller must be checked before collection status")
        ),
    )
    monkeypatch.setattr(
        dispatcher,
        "_active_collection_tasks",
        lambda _path: [
            {
                "id": "t-v1",
                "signature": "ScheduleurmBench/critical_gpu_joint95/legacy_unit",
            },
            {
                "id": "t-v20",
                "signature": "ScheduleurmBench/critical_gpu_joint95_v20/unit",
            },
            {"id": "t-missing"},
            {
                "id": "t-controller",
                "signature": dispatcher.COLLECTION_SIGNATURE_PREFIX + "unit",
            },
        ],
    )
    _forbid_mutations(monkeypatch)
    report = supervisor.supervisor_step(Path("unused"), dry_run=False)
    assert report["status"] == "WAIT_ACTIVE_CONTROLLER"
    assert report["collection_status"] == "NOT_READ_WHILE_CONTROLLER_ACTIVE"
    assert report["active_controller_ids"] == ["t-controller"]
    assert report["mutations"] == 0


def test_wrong_prefix_active_task_does_not_block_collection_status(
    monkeypatch: pytest.MonkeyPatch,
):
    manifest = _patch_context(monkeypatch)
    calls = []

    def collection_status(actual, _results):
        calls.append(actual)
        return {"status": "COMPLETE", "errors": []}

    monkeypatch.setattr(dispatcher, "collection_status", collection_status)
    monkeypatch.setattr(
        dispatcher,
        "_active_collection_tasks",
        lambda _path: [
            {
                "id": "t-v1",
                "signature": "ScheduleurmBench/critical_gpu_joint95/legacy_unit",
            },
            {
                "id": "t-v20",
                "signature": "ScheduleurmBench/critical_gpu_joint95_v20/unit",
            },
            {"id": "t-missing"},
        ],
    )
    _forbid_mutations(monkeypatch)
    report = supervisor.supervisor_step(Path("unused"), dry_run=False)
    assert report["status"] == "COMPLETE"
    assert report["mutations"] == 0
    assert calls == [manifest]


@pytest.mark.parametrize("terminal_status", ["done", "failed", "cancelled"])
def test_terminal_expected_controller_without_success_stops_reproposal(
    monkeypatch: pytest.MonkeyPatch, terminal_status: str
):
    _patch_context(monkeypatch)
    monkeypatch.setattr(
        dispatcher,
        "collection_status",
        lambda *_args: {
            "status": "READY_CALIBRATION",
            "errors": [],
            "next_wave": 34,
        },
    )
    monkeypatch.setattr(dispatcher, "_active_collection_tasks", lambda _path: [])
    seen = []

    def terminal(_path, signature):
        seen.append(signature)
        return [
            {
                "id": "t-dead",
                "status": terminal_status,
                "signature": signature,
            }
        ]

    monkeypatch.setattr(supervisor, "_terminal_controller_history", terminal)
    _forbid_mutations(monkeypatch)
    report = supervisor.supervisor_step(Path("unused"), dry_run=False)
    assert report["status"] == "TERMINATE_CONTROLLER_FAILURE"
    assert report["expected_wave"] == 34
    assert report["retry_attempted"] is False
    assert report["mutations"] == 0
    assert seen == [
        "ScheduleurmBench/critical_gpu_joint95_v2/"
        "critical_gpu_joint95_v2_jtl110gpu2_r34"
    ]


def test_completed_prior_wave_with_pass_does_not_block_next_wave(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_context(monkeypatch)
    monkeypatch.setattr(
        dispatcher,
        "collection_status",
        lambda *_args: {
            "status": "READY_CALIBRATION",
            "errors": [],
            "next_wave": 35,
        },
    )
    monkeypatch.setattr(dispatcher, "_active_collection_tasks", lambda _path: [])
    prior_signature = (
        "ScheduleurmBench/critical_gpu_joint95_v2/"
        "critical_gpu_joint95_v2_jtl110gpu2_r34"
    )
    expected_signature = (
        "ScheduleurmBench/critical_gpu_joint95_v2/"
        "critical_gpu_joint95_v2_jtl110gpu2_r35"
    )
    seen = []

    def terminal(_path, signature):
        seen.append(signature)
        return [] if signature != prior_signature else [
            {"id": "t34", "status": "done", "signature": prior_signature}
        ]

    monkeypatch.setattr(supervisor, "_terminal_controller_history", terminal)
    monkeypatch.setattr(
        dispatcher,
        "release_holdout",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("wave35 cannot release holdout")
        ),
    )
    submissions = []

    def submit(**kwargs):
        submissions.append(kwargs)
        return {"joint_wave": 35, "split_role": "calibration", "task_id": "t35"}

    monkeypatch.setattr(dispatcher, "submit_next", submit)
    report = supervisor.supervisor_step(Path("unused"), dry_run=False)
    assert report["status"] == "SUBMITTED_ONE"
    assert report["joint_wave"] == 35
    assert seen == [expected_signature]
    assert len(submissions) == 1


def test_terminal_history_uses_exact_signature_and_supported_terminal_states(
    tmp_path: Path,
):
    wanted = "ScheduleurmBench/critical_gpu_joint95_v2/unit34"
    queue = tmp_path / "queue.json"
    queue.write_text(
        json.dumps(
            {
                "tasks": [
                    {"id": "done", "status": "done", "signature": wanted},
                    {"id": "failed", "status": "failed", "signature": wanted},
                    {
                        "id": "cancelled",
                        "status": "cancelled",
                        "signature": wanted,
                    },
                    {"id": "active", "status": "running", "signature": wanted},
                    {
                        "id": "other",
                        "status": "failed",
                        "signature": wanted + "-other",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    rows = supervisor._terminal_controller_history(queue, wanted)
    assert [(row["id"], row["status"]) for row in rows] == [
        ("done", "done"),
        ("failed", "failed"),
        ("cancelled", "cancelled"),
    ]


@pytest.mark.parametrize("state", ["FAIL_STOP", "FAIL_RESULT_STATE"])
def test_failed_or_invalid_campaign_terminates_without_retry(
    monkeypatch: pytest.MonkeyPatch, state: str
):
    _patch_context(monkeypatch)
    monkeypatch.setattr(
        dispatcher,
        "collection_status",
        lambda *_args: {"status": state, "errors": ["terminal failure"]},
    )
    monkeypatch.setattr(
        dispatcher,
        "_active_collection_tasks",
        lambda _path: [],
    )
    _forbid_mutations(monkeypatch)
    report = supervisor.supervisor_step(Path("unused"), dry_run=False)
    assert report["terminal"] is True
    assert report["retry_attempted"] is False
    assert report["mutations"] == 0


def test_holdout_lock_releases_then_submits_wave53_once(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_context(monkeypatch)
    states = iter(
        [
            {
                "status": "HOLDOUT_LOCKED",
                "errors": [],
                "successful_calibration_count": 19,
                "holdout_success": False,
                "holdout_released": False,
            },
            {"status": "READY_HOLDOUT", "errors": [], "next_wave": 53},
        ]
    )
    monkeypatch.setattr(dispatcher, "collection_status", lambda *_args: next(states))
    monkeypatch.setattr(dispatcher, "_active_collection_tasks", lambda _path: [])
    calls = []
    monkeypatch.setattr(
        dispatcher,
        "release_holdout",
        lambda **kwargs: calls.append(("release", kwargs)),
    )

    def submit(**kwargs):
        calls.append(("submit", kwargs))
        return {"joint_wave": 53, "split_role": "holdout", "task_id": "t53"}

    monkeypatch.setattr(dispatcher, "submit_next", submit)
    report = supervisor.supervisor_step(Path("unused"), dry_run=False)
    assert report["status"] == "SUBMITTED_HOLDOUT"
    assert report["joint_wave"] == 53
    assert report["task_id"] == "t53"
    assert [kind for kind, _ in calls] == ["release", "submit"]
    assert calls[1][1]["dry_run"] is False


def test_holdout_dry_run_plans_both_actions_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_context(monkeypatch)
    monkeypatch.setattr(
        dispatcher,
        "collection_status",
        lambda *_args: {
            "status": "HOLDOUT_LOCKED",
            "errors": [],
            "successful_calibration_count": 19,
            "holdout_success": False,
            "holdout_released": False,
        },
    )
    monkeypatch.setattr(dispatcher, "_active_collection_tasks", lambda _path: [])
    _forbid_mutations(monkeypatch)
    report = supervisor.supervisor_step(Path("unused"), dry_run=True)
    assert report["status"] == "DRY_RUN_RELEASE_THEN_SUBMIT_HOLDOUT"
    assert report["planned_actions"] == [
        "release_holdout",
        "submit_next_wave53",
    ]
    assert report["mutations"] == 0


def test_complete_exits_without_inspecting_active_tasks_or_mutating(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_context(monkeypatch)
    monkeypatch.setattr(
        dispatcher,
        "collection_status",
        lambda *_args: {"status": "COMPLETE", "errors": []},
    )
    monkeypatch.setattr(
        dispatcher,
        "_active_collection_tasks",
        lambda _path: [],
    )
    _forbid_mutations(monkeypatch)
    report = supervisor.supervisor_step(Path("unused"), dry_run=False)
    assert report == {
        "status": "COMPLETE",
        "terminal": True,
        "collection_status": "COMPLETE",
        "mutations": 0,
    }


def test_run_dry_run_requires_once_and_never_sleeps(monkeypatch: pytest.MonkeyPatch):
    with pytest.raises(supervisor.SupervisorRefusal, match="requires --once"):
        supervisor.run_supervisor(
            Path("unused"), dry_run=True, once=False, poll_s=30.0
        )
    monkeypatch.setattr(
        supervisor,
        "supervisor_step",
        lambda *_args, **_kwargs: {
            "status": "DRY_RUN_SUBMIT_ONE",
            "terminal": False,
        },
    )
    monkeypatch.setattr(
        supervisor.time,
        "sleep",
        lambda _seconds: (_ for _ in ()).throw(
            AssertionError("--once must not sleep")
        ),
    )
    assert (
        supervisor.run_supervisor(
            Path("unused"), dry_run=True, once=True, poll_s=30.0
        )
        == 0
    )
