from __future__ import annotations

import json
from pathlib import Path

import pytest

from algorithm.experiments import aligned_hardware_joint95_v2_postcampaign_runner as runner


def _stages(tmp_path: Path) -> list[dict]:
    return [
        {
            "name": "capacity",
            "registration_snapshot": {"stage": "capacity"},
            "registration_path": str(tmp_path / "capacity_protocol.json"),
            "fixed_sources_path": None,
            "result_path": str(tmp_path / "capacity.json"),
            "decision_universe": [
                runner.CAPACITY_ADVANCE_DECISION,
                "JOINT95_NO_POSITIVE_CAPACITY_MARGIN_STOP",
            ],
            "advance_decisions": [runner.CAPACITY_ADVANCE_DECISION],
        },
        {
            "name": "canonical",
            "registration_snapshot": {"stage": "canonical"},
            "registration_path": str(tmp_path / "canonical_protocol.json"),
            "fixed_sources_path": str(tmp_path / "canonical_sources.json"),
            "result_path": str(tmp_path / "canonical.json"),
            "decision_universe": [
                runner.canonical.PASS_DECISION,
                "CANONICAL_PHYSICAL_SELECTOR_SUPPORT_FAIL_STOP",
            ],
            "advance_decisions": [runner.canonical.PASS_DECISION],
        },
        {
            "name": "baseline",
            "registration_snapshot": {"stage": "baseline"},
            "registration_path": str(tmp_path / "baseline_protocol.json"),
            "fixed_sources_path": str(tmp_path / "baseline_sources.json"),
            "result_path": str(tmp_path / "baseline.json"),
            "decision_universe": [
                runner.baseline.PASS_DECISION,
                "BASELINE_CONFIRMATION_FAIL_STOP",
            ],
            "advance_decisions": [runner.baseline.PASS_DECISION],
        },
    ]


def _payload(stage: dict, *, advances: bool) -> dict:
    name = stage["name"]
    gate, experiment = {
        "capacity": (runner.capacity.GATE, runner.capacity.EXPERIMENT),
        "canonical": (runner.canonical.GATE, runner.canonical.EXPERIMENT),
        "baseline": (runner.baseline.GATE, runner.baseline.EXPERIMENT),
    }[name]
    decision = (
        stage["advance_decisions"][0]
        if advances
        else next(
            value
            for value in stage["decision_universe"]
            if value not in stage["advance_decisions"]
        )
    )
    payload = {
        "schema_version": 1,
        "gate": gate,
        "experiment": experiment,
        "status": "PASS",
        "protocol": stage["registration_snapshot"],
        "scientific_decision": decision,
    }
    if name == "capacity":
        payload.update(
            {
                "capacity_established": advances,
                "joint95_v2_audit": {"holdout_covered": advances},
            }
        )
    elif name == "canonical":
        payload.update(
            {
                "physical_trajectory_advance": advances,
                "baseline_trajectory_advance": advances,
            }
        )
    else:
        payload["baseline_confirmation_passed"] = advances
    return payload


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    collection: dict,
) -> list[dict]:
    stages = _stages(tmp_path)
    manifest = {"results_contract": {"results_directory": str(tmp_path / "joint")}}
    monkeypatch.setattr(
        runner,
        "_load_registration",
        lambda _path: (
            {"source_paths": {"queue_status": str(tmp_path / "queue.json")}},
            manifest,
            stages,
        ),
    )
    monkeypatch.setattr(
        runner.dispatcher, "_active_collection_tasks", lambda _path: []
    )
    monkeypatch.setattr(
        runner.dispatcher,
        "collection_status",
        lambda actual, _results: collection
        if actual is manifest
        else (_ for _ in ()).throw(AssertionError("manifest changed")),
    )
    return stages


def _forbid_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runner,
        "_run_stage",
        lambda _stage: (_ for _ in ()).throw(
            AssertionError("downstream stage must not run")
        ),
    )


def test_registration_requires_zero_collection_and_downstream_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    results = tmp_path / "joint"
    results.mkdir()
    manifest = {"results_contract": {"results_directory": str(results.resolve())}}
    stages = _stages(tmp_path)
    monkeypatch.setattr(
        runner.dispatcher, "load_registered_manifest", lambda _path: manifest
    )
    monkeypatch.setattr(
        runner.dispatcher,
        "collection_status",
        lambda *_args: {
            "status": "READY_CALIBRATION",
            "next_wave": 34,
            "successful_calibration_count": 0,
            "holdout_success": False,
        },
    )
    monkeypatch.setattr(runner, "_load_downstream_protocols", lambda: stages)
    monkeypatch.setattr(
        runner.dispatcher,
        "_active_collection_tasks",
        lambda _path: [
            {
                "id": "t-v1",
                "signature": "ScheduleurmBench/critical_gpu_joint95/legacy_unit",
            }
        ],
    )
    report = runner.build_registration(
        manifest_path=tmp_path / "manifest.json",
        results_dir=results,
        queue_path=tmp_path / "queue.json",
    )
    assert report["stage_order"] == ["capacity", "canonical", "baseline"]
    assert report["registration_snapshot"] == {
        "kind": "ZERO_RESULT_START",
        "collection_status_read_while_controller_active": False,
        "joint_result_entries": [],
    }

    (results / "visit.json").write_text("{}", encoding="utf-8")
    with pytest.raises(runner.PostcampaignRefusal, match="zero v2 results"):
        runner.build_registration(
            manifest_path=tmp_path / "manifest.json",
            results_dir=results,
            queue_path=tmp_path / "queue.json",
        )


def test_exact_v2_active_controller_is_checked_before_collection_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    stages = _stages(tmp_path)
    manifest = {"results_contract": {"results_directory": str(tmp_path / "joint")}}
    monkeypatch.setattr(
        runner,
        "_load_registration",
        lambda _path: (
            {"source_paths": {"queue_status": str(tmp_path / "queue.json")}},
            manifest,
            stages,
        ),
    )
    monkeypatch.setattr(
        runner.dispatcher,
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
                "id": "t-running",
                "signature": runner.dispatcher.COLLECTION_SIGNATURE_PREFIX + "unit",
            },
        ],
    )
    monkeypatch.setattr(
        runner.dispatcher,
        "collection_status",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("collection status must not read a partial active attempt")
        ),
    )
    _forbid_run(monkeypatch)
    report = runner.postcampaign_step(Path("unused"), dry_run=False)
    assert report == {
        "status": "WAIT_COLLECTION_CONTROLLER",
        "terminal": False,
        "collection_status": "NOT_READ_WHILE_CONTROLLER_ACTIVE",
        "active_controller_ids": ["t-running"],
        "stages_run": [],
    }


def test_wrong_prefix_active_task_does_not_block_collection_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    calls = []
    stages = _stages(tmp_path)
    manifest = {"results_contract": {"results_directory": str(tmp_path / "joint")}}
    monkeypatch.setattr(
        runner,
        "_load_registration",
        lambda _path: (
            {"source_paths": {"queue_status": str(tmp_path / "queue.json")}},
            manifest,
            stages,
        ),
    )
    monkeypatch.setattr(
        runner.dispatcher,
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

    def collection_status(actual, _results):
        calls.append(actual)
        return {"status": "READY_CALIBRATION", "errors": []}

    monkeypatch.setattr(runner.dispatcher, "collection_status", collection_status)
    _forbid_run(monkeypatch)
    report = runner.postcampaign_step(Path("unused"), dry_run=False)
    assert report == {
        "status": "WAIT_COLLECTION",
        "terminal": False,
        "collection_status": "READY_CALIBRATION",
        "stages_run": [],
    }
    assert calls == [manifest]


def test_registration_snapshot_filters_mixed_active_tasks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    results = tmp_path / "joint"
    results.mkdir()
    manifest = {"results_contract": {"results_directory": str(results.resolve())}}
    monkeypatch.setattr(
        runner.dispatcher, "load_registered_manifest", lambda _path: manifest
    )
    monkeypatch.setattr(runner, "_load_downstream_protocols", lambda: _stages(tmp_path))
    monkeypatch.setattr(
        runner.dispatcher,
        "_active_collection_tasks",
        lambda _path: [
            {
                "id": "t-v1",
                "signature": "ScheduleurmBench/critical_gpu_joint95/legacy_unit",
            },
            {
                "id": "t-v2",
                "signature": runner.dispatcher.COLLECTION_SIGNATURE_PREFIX + "unit",
            },
        ],
    )
    monkeypatch.setattr(
        runner.dispatcher,
        "collection_status",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("exact v2 active must block collection status")
        ),
    )
    report = runner.build_registration(
        manifest_path=tmp_path / "manifest.json",
        results_dir=results,
        queue_path=tmp_path / "queue.json",
    )
    assert report["protocol"].endswith("postcampaign_v3")
    assert report["registration_snapshot"]["active_controller_ids"] == ["t-v2"]
    assert report["registration_snapshot"]["active_controller_signatures"] == [
        runner.dispatcher.COLLECTION_SIGNATURE_PREFIX + "unit"
    ]


@pytest.mark.parametrize(
    "state", ["READY_CALIBRATION", "HOLDOUT_LOCKED", "READY_HOLDOUT"]
)
def test_incomplete_collection_only_waits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, state: str
):
    _context(monkeypatch, tmp_path, {"status": state, "errors": []})
    _forbid_run(monkeypatch)
    report = runner.postcampaign_step(Path("unused"), dry_run=False)
    assert report == {
        "status": "WAIT_COLLECTION",
        "terminal": False,
        "collection_status": state,
        "stages_run": [],
    }


@pytest.mark.parametrize("state", ["FAIL_STOP", "FAIL_RESULT_STATE"])
def test_failed_collection_terminates_without_analysis(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, state: str
):
    _context(
        monkeypatch,
        tmp_path,
        {"status": state, "errors": ["collection failed"]},
    )
    _forbid_run(monkeypatch)
    report = runner.postcampaign_step(Path("unused"), dry_run=False)
    assert report["status"] == "TERMINATE_COLLECTION_FAILURE"
    assert report["terminal"] is True
    assert report["stages_run"] == []


def test_complete_dry_run_plans_strict_order_without_writes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    stages = _context(
        monkeypatch, tmp_path, {"status": "COMPLETE", "errors": []}
    )
    _forbid_run(monkeypatch)
    report = runner.postcampaign_step(Path("unused"), dry_run=True)
    assert report["status"] == "DRY_RUN_POSTCAMPAIGN"
    assert report["planned_stages"] == ["capacity", "canonical", "baseline"]
    assert report["stages_run"] == []
    assert all(not Path(stage["result_path"]).exists() for stage in stages)


def test_positive_pipeline_runs_capacity_canonical_baseline_in_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    stages = _context(
        monkeypatch, tmp_path, {"status": "COMPLETE", "errors": []}
    )
    calls = []

    def run_stage(stage):
        calls.append(stage["name"])
        payload = _payload(stage, advances=True)
        _write(Path(stage["result_path"]), payload)
        return payload

    monkeypatch.setattr(runner, "_run_stage", run_stage)
    report = runner.postcampaign_step(Path("unused"), dry_run=False)
    assert report["status"] == "POSTCAMPAIGN_COMPLETE"
    assert report["stages_run"] == calls == ["capacity", "canonical", "baseline"]
    assert all(Path(stage["result_path"]).is_file() for stage in stages)


@pytest.mark.parametrize("negative_index", [0, 1, 2])
def test_negative_decision_retains_result_and_blocks_all_downstream(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, negative_index: int
):
    stages = _context(
        monkeypatch, tmp_path, {"status": "COMPLETE", "errors": []}
    )
    calls = []

    def run_stage(stage):
        index = [row["name"] for row in stages].index(stage["name"])
        calls.append(stage["name"])
        payload = _payload(stage, advances=index < negative_index)
        _write(Path(stage["result_path"]), payload)
        return payload

    monkeypatch.setattr(runner, "_run_stage", run_stage)
    report = runner.postcampaign_step(Path("unused"), dry_run=False)
    assert report["status"] == (
        f"TERMINATE_{stages[negative_index]['name'].upper()}_DECISION"
    )
    assert calls == [row["name"] for row in stages[: negative_index + 1]]
    assert Path(stages[negative_index]["result_path"]).is_file()
    assert all(
        not Path(stage["result_path"]).exists()
        for stage in stages[negative_index + 1 :]
    )


def test_existing_positive_result_is_validated_and_never_rerun(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    stages = _context(
        monkeypatch, tmp_path, {"status": "COMPLETE", "errors": []}
    )
    _write(Path(stages[0]["result_path"]), _payload(stages[0], advances=True))
    calls = []

    def run_stage(stage):
        calls.append(stage["name"])
        payload = _payload(stage, advances=True)
        _write(Path(stage["result_path"]), payload)
        return payload

    monkeypatch.setattr(runner, "_run_stage", run_stage)
    report = runner.postcampaign_step(Path("unused"), dry_run=False)
    assert report["status"] == "POSTCAMPAIGN_COMPLETE"
    assert calls == ["canonical", "baseline"]
    assert report["stages_run"] == calls


def test_existing_negative_upstream_is_retained_without_rerun(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    stages = _context(
        monkeypatch, tmp_path, {"status": "COMPLETE", "errors": []}
    )
    original = _payload(stages[0], advances=False)
    _write(Path(stages[0]["result_path"]), original)
    _forbid_run(monkeypatch)
    report = runner.postcampaign_step(Path("unused"), dry_run=False)
    assert report["status"] == "TERMINATE_CAPACITY_DECISION"
    assert report["existing_result_retained"] is True
    assert json.loads(Path(stages[0]["result_path"]).read_text()) == original
    assert all(not Path(stage["result_path"]).exists() for stage in stages[1:])


def test_existing_downstream_after_missing_or_negative_upstream_is_invalid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    stages = _context(
        monkeypatch, tmp_path, {"status": "COMPLETE", "errors": []}
    )
    _write(Path(stages[1]["result_path"]), _payload(stages[1], advances=True))
    _forbid_run(monkeypatch)
    report = runner.postcampaign_step(Path("unused"), dry_run=False)
    assert report["status"] == "TERMINATE_DOWNSTREAM_ORDER_INVALID"
    assert report["stages_run"] == []


def test_dry_run_requires_once(monkeypatch: pytest.MonkeyPatch):
    with pytest.raises(runner.PostcampaignRefusal, match="requires --once"):
        runner.run_loop(Path("unused"), dry_run=True, once=False, poll_s=30.0)
    monkeypatch.setattr(
        runner,
        "postcampaign_step",
        lambda *_args, **_kwargs: {
            "status": "WAIT_COLLECTION",
            "terminal": False,
        },
    )
    monkeypatch.setattr(
        runner.time,
        "sleep",
        lambda _seconds: (_ for _ in ()).throw(
            AssertionError("once cannot sleep")
        ),
    )
    assert runner.run_loop(Path("unused"), dry_run=True, once=True, poll_s=30.0) == 0
