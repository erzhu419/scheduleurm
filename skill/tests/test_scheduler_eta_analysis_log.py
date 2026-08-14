from __future__ import annotations

import json

from skill.scheduler_eta.analysis_log import record_eta_analysis_tasks


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_eta_analysis_records_initial_minute_samples_and_terminal_accuracy(tmp_path):
    task = {
        "id": "t1",
        "status": "running",
        "project": "BAPR",
        "signature": "bapr/sig",
        "description": "BAPR run",
        "node": "node001",
        "gpu_idx": 0,
        "started_at": 100.0,
        "eta_seconds": 300,
        "eta_source": "runtime_history",
        "eta_confidence": "low",
        "eta_updated_at": 100,
    }

    first = record_eta_analysis_tasks([task], log_dir=tmp_path, now=100.0)
    assert first["records"] == 1

    too_early = record_eta_analysis_tasks(
        [task], log_dir=tmp_path, now=159.0, include_periodic=True)
    assert too_early["records"] == 0

    task.update(
        eta_seconds=180,
        eta_source="progress_rate",
        eta_confidence="medium",
        eta_updated_at=160,
        last_progress_line="Step 4/10",
    )
    minute = record_eta_analysis_tasks(
        [task], log_dir=tmp_path, now=160.0, include_periodic=True)
    assert minute["records"] == 1

    task.update(
        status="done",
        finished_at=351.0,
        backend_finished_at=340.0,
        exit_code=0,
        eta_seconds=20,
        eta_updated_at=339,
        _diagnosis={"reason": "normal exit"},
    )
    terminal = record_eta_analysis_tasks([task], log_dir=tmp_path, now=351.0)
    assert terminal["records"] == 1
    assert terminal["terminals"] == 1

    rows = _rows(tmp_path / "t1.jsonl")
    assert [row["event"] for row in rows] == ["initial", "sample", "terminal"]
    assert rows[0]["predicted_total_s"] == 300.0
    assert rows[1]["minute_index"] == 1
    assert rows[1]["predicted_total_s"] == 240.0

    summary = rows[-1]
    assert summary["actual_duration_s"] == 240.0
    assert summary["actual_finished_at_source"] == "backend_finished_at"
    assert summary["watcher_detection_lag_s"] == 11.0
    assert summary["initial_error_s"] == 60.0
    assert summary["initial_error_pct"] == 25.0
    assert summary["eta_bias_s"] == 30.0
    assert summary["eta_mae_s"] == 30.0
    assert summary["final_error_s"] == 0.0
    assert summary["eta_source_counts"] == {
        "progress_rate": 1,
        "runtime_history": 1,
    }
    assert _rows(tmp_path / "summary.jsonl") == [summary]

    duplicate = record_eta_analysis_tasks([task], log_dir=tmp_path, now=400.0)
    assert duplicate["records"] == 0
    assert len(_rows(tmp_path / "t1.jsonl")) == 3
    assert len(_rows(tmp_path / "summary.jsonl")) == 1


def test_eta_analysis_keeps_missing_eta_and_missed_interval_evidence(tmp_path):
    task = {
        "id": "t2",
        "status": "running",
        "started_at": 100.0,
        "eta_seconds": 0,
    }
    record_eta_analysis_tasks([task], log_dir=tmp_path, now=100.0)
    task.update(eta_seconds=50, eta_source="tqdm")
    record_eta_analysis_tasks(
        [task], log_dir=tmp_path, now=280.0, include_periodic=True)
    task.update(status="failed", finished_at=300.0)
    record_eta_analysis_tasks([task], log_dir=tmp_path, now=300.0)

    rows = _rows(tmp_path / "t2.jsonl")
    assert rows[0]["eta_available"] is False
    assert rows[0]["eta_seconds"] is None
    assert rows[1]["minute_index"] == 3
    assert rows[1]["missed_intervals"] == 2
    assert rows[-1]["missing_eta_samples"] == 1
    assert rows[-1]["prediction_count"] == 1


def test_eta_analysis_separates_relaunched_attempts_for_same_task_id(tmp_path):
    first = {
        "id": "same",
        "status": "done",
        "started_at": 10.0,
        "finished_at": 20.0,
        "eta_seconds": 10,
    }
    second = {
        "id": "same",
        "status": "running",
        "started_at": 30.0,
        "eta_seconds": 15,
    }
    record_eta_analysis_tasks([first], log_dir=tmp_path, now=20.0)
    record_eta_analysis_tasks([second], log_dir=tmp_path, now=30.0)

    rows = _rows(tmp_path / "same.jsonl")
    assert [row["event"] for row in rows] == ["initial", "terminal", "initial"]
    assert len({row["attempt_id"] for row in rows}) == 2


def test_fixed_interval_trigger_records_across_same_wall_clock_minute(tmp_path):
    task = {
        "id": "clock-slew",
        "status": "running",
        "started_at": 100.0,
        "eta_seconds": 300,
    }
    record_eta_analysis_tasks([task], log_dir=tmp_path, now=120.0)
    result = record_eta_analysis_tasks(
        [task],
        log_dir=tmp_path,
        now=151.0,
        include_periodic=True,
        force_periodic=True,
    )

    rows = _rows(tmp_path / "clock-slew.jsonl")
    assert result["records"] == 1
    assert [row["minute_index"] for row in rows] == [0, 0]
    assert rows[-1]["fixed_interval_trigger"] is True
