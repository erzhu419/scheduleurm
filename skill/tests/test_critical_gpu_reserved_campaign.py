from __future__ import annotations

import json
from pathlib import Path

import pytest

from algorithm.experiments.critical_gpu_reserved_campaign import (
    _active_scheduler_tasks,
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
