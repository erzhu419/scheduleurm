from __future__ import annotations

import json
from pathlib import Path

import pytest

from algorithm.experiments import aligned_hardware_joint95_v5_postcampaign_runner as runner


def _capacity_stage(tmp_path: Path) -> dict:
    return {
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
    }


def _capacity_payload(stage: dict) -> dict:
    return {
        "schema_version": 1,
        "gate": runner.capacity.GATE,
        "experiment": runner.capacity.EXPERIMENT,
        "status": "PASS",
        "protocol": stage["registration_snapshot"],
        "scientific_decision": "JOINT95_NO_POSITIVE_CAPACITY_MARGIN_STOP",
        "capacity_established": False,
        "joint95_v5_audit": {"holdout_covered": True},
        "physical_complete_frame_envelope": {
            "model": {"scales": (60.0, 40.0, 120.0)}
        },
    }


def _complete_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stage: dict
) -> None:
    manifest = {
        "results_contract": {"results_directory": str(tmp_path / "joint")}
    }
    monkeypatch.setattr(
        runner,
        "_load_registration",
        lambda _path: (
            {"source_paths": {"queue_status": str(tmp_path / "queue.json")}},
            manifest,
            [stage],
        ),
    )
    monkeypatch.setattr(runner, "_active_v5_collection_tasks", lambda _path: [])
    monkeypatch.setattr(
        runner.dispatcher,
        "collection_status",
        lambda actual, _results: {"status": "COMPLETE", "errors": []}
        if actual is manifest
        else (_ for _ in ()).throw(AssertionError("manifest changed")),
    )


def test_stage_return_is_compared_in_its_json_representation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    stage = _capacity_stage(tmp_path)
    _complete_context(monkeypatch, tmp_path, stage)
    returned = _capacity_payload(stage)

    def run_stage(_stage):
        Path(stage["result_path"]).write_text(
            json.dumps(returned), encoding="utf-8"
        )
        return returned

    monkeypatch.setattr(runner, "_run_stage", run_stage)
    report = runner.postcampaign_step(Path("unused"), dry_run=False)
    assert report["status"] == "TERMINATE_CAPACITY_DECISION"
    assert report["stages_run"] == ["capacity"]


def test_json_normalization_does_not_accept_a_material_difference(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    stage = _capacity_stage(tmp_path)
    _complete_context(monkeypatch, tmp_path, stage)
    returned = _capacity_payload(stage)
    stored = json.loads(json.dumps(returned))
    stored["physical_complete_frame_envelope"]["model"]["scales"][0] = 61.0

    def run_stage(_stage):
        Path(stage["result_path"]).write_text(json.dumps(stored), encoding="utf-8")
        return returned

    monkeypatch.setattr(runner, "_run_stage", run_stage)
    report = runner.postcampaign_step(Path("unused"), dry_run=False)
    assert report["status"] == "TERMINATE_CAPACITY_RUNNER_FAILURE"
    assert report["errors"] == [
        "capacity returned payload differs from stored result"
    ]


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("TERMINATE_CAPACITY_DECISION", 0),
        ("TERMINATE_CAPACITY_RUNNER_FAILURE", 2),
        ("POSTCAMPAIGN_COMPLETE", 0),
    ],
)
def test_run_loop_exit_code_distinguishes_scientific_stop_from_failure(
    monkeypatch: pytest.MonkeyPatch, status: str, expected: int
):
    monkeypatch.setattr(
        runner,
        "postcampaign_step",
        lambda *_args, **_kwargs: {"status": status, "terminal": True},
    )
    assert (
        runner.run_loop(Path("unused"), dry_run=False, once=True, poll_s=30.0)
        == expected
    )
