from __future__ import annotations

import json
from pathlib import Path

import pytest

import algorithm.experiments.critical_gpu_reserved_campaign as campaign_module

from algorithm.experiments.critical_gpu_reserved_campaign import (
    _active_scheduler_tasks,
    _max_command_attempts,
    _parse_waves,
    _reservation_acknowledged,
    build_commands,
)


def test_wave_parser_supports_ranges_and_deduplicates():
    assert _parse_waves("1-3,2,5", tuple(range(1, 14))) == (1, 2, 3, 5)
    with pytest.raises(ValueError):
        _parse_waves("3-1", tuple(range(1, 14)))
    with pytest.raises(ValueError):
        _parse_waves("14", tuple(range(1, 14)))


def test_command_plan_keeps_frozen_campaigns_separate():
    rows = build_commands(
        node="jtl311linux",
        empty_waves=(1, 2),
        loaded_waves=(1,),
        run_empty_gate=True,
        run_loaded_gate=True,
    )
    assert [row["phase"] for row in rows] == [
        "empty",
        "empty",
        "empty_gate",
        "loaded",
        "loaded_gate",
    ]
    assert rows[0]["argv"][2] == "algorithm.experiments.critical_gpu_completion_campaign"
    assert rows[3]["argv"][2] == "algorithm.experiments.critical_gpu_loaded_wave_runner"
    assert "--keep-remote-output" not in rows[0]["argv"]
    assert "--keep-remote-output" in rows[3]["argv"]


def test_only_measurement_waves_receive_bounded_automatic_retries():
    assert _max_command_attempts("empty", 2) == 3
    assert _max_command_attempts("loaded", 2) == 3
    assert _max_command_attempts("empty_gate", 2) == 1
    assert _max_command_attempts("loaded_gate", 2) == 1
    with pytest.raises(ValueError):
        _max_command_attempts("loaded", -1)


def test_ack_requires_new_cache_and_matching_reservation(tmp_path):
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({
        "ts": 100.0,
        "states": {
            "node007": {
                "measurement_reservation_active": True,
                "measurement_reservation": {"reservation_id": "r1"},
            }
        },
    }))
    assert _reservation_acknowledged("node007", "r1", 99.0, cache_file=cache)
    assert not _reservation_acknowledged("node007", "r2", 99.0, cache_file=cache)
    assert not _reservation_acknowledged("node007", "r1", 101.0, cache_file=cache)


def test_active_scheduler_tasks_is_node_scoped(tmp_path):
    queue = tmp_path / "queue.json"
    queue.write_text(json.dumps({
        "tasks": [
            {"id": "a", "node": "node007", "status": "running", "project": "P"},
            {"id": "b", "node": "node007", "status": "done", "project": "P"},
            {"id": "c", "node": "jtl110gpu", "status": "launching", "project": "P"},
        ]
    }))
    assert [row["id"] for row in _active_scheduler_tasks("node007", queue_file=queue)] == ["a"]


class _Lease:
    def __enter__(self):
        return {
            "reservation_id": "test-reservation",
            "created_at": 1.0,
        }

    def __exit__(self, exc_type, exc, tb):
        return None

    def ensure_healthy(self):
        return None


def test_failed_measurement_wave_retries_but_gate_does_not(monkeypatch, tmp_path):
    monkeypatch.setattr(campaign_module, "MeasurementReservation", lambda **_: _Lease())
    monkeypatch.setattr(campaign_module, "_wait_for_scheduler_ack", lambda **_: None)
    monkeypatch.setattr(
        campaign_module,
        "_wait_for_clean_node",
        lambda **_: {"gpu_idle": {"ready": True}, "active_scheduler_tasks": []},
    )
    returncodes = iter((3, 0))
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        return campaign_module.subprocess.CompletedProcess(argv, next(returncodes))

    monkeypatch.setattr(campaign_module.subprocess, "run", fake_run)
    report = campaign_module.run_reserved_campaign(
        node="jtl110gpu",
        loaded_waves=(1,),
        wave_retries=2,
        output=tmp_path / "retry.json",
    )
    assert report["pass"] is True
    assert [row["returncode"] for row in report["commands"]] == [3, 0]
    assert [row["attempt"] for row in report["commands"]] == [1, 2]
    assert len(calls) == 2

    gate_calls = []

    def failed_gate(argv, **kwargs):
        gate_calls.append(list(argv))
        return campaign_module.subprocess.CompletedProcess(argv, 3)

    monkeypatch.setattr(campaign_module.subprocess, "run", failed_gate)
    gate_report = campaign_module.run_reserved_campaign(
        node="jtl110gpu",
        run_loaded_gate=True,
        wave_retries=5,
        output=tmp_path / "gate.json",
    )
    assert gate_report["pass"] is False
    assert gate_report["failed_after_attempts"] == 1
    assert len(gate_calls) == 1
