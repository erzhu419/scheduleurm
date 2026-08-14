import json
from types import SimpleNamespace

from skill import scheduler_archive as archive


def test_archive_terminal_tasks_moves_only_old_terminal_records(tmp_path):
    archive_file = tmp_path / "queue_archive.jsonl"
    now = 1_000_000.0
    old = now - 3 * 86400
    recent = now - 60
    state = {
        "tasks": [
            {"id": "t_old_done", "status": "done", "finished_at": old},
            {"id": "t_recent_done", "status": "done", "finished_at": recent},
            {"id": "t_running", "status": "running", "submitted_at": old},
            {"id": "t_old_no_ts", "status": "failed"},
        ]
    }

    count = archive.archive_terminal_tasks(
        state,
        archive_file=archive_file,
        age_days=1,
        now_fn=lambda: now,
    )

    assert count == 1
    assert [task["id"] for task in state["tasks"]] == [
        "t_recent_done",
        "t_running",
        "t_old_no_ts",
    ]
    assert [json.loads(line)["id"] for line in archive_file.read_text().splitlines()] == ["t_old_done"]


def test_compact_age_days_from_args_uses_all_terminal_and_hour_precedence():
    assert archive.compact_age_days_from_args(
        SimpleNamespace(all_terminal=True, age_hours=5, age_days=9),
        default_age_days=3,
    ) == 0.0
    assert archive.compact_age_days_from_args(
        SimpleNamespace(all_terminal=False, age_hours=12, age_days=9),
        default_age_days=3,
    ) == 0.5
    assert archive.compact_age_days_from_args(
        SimpleNamespace(all_terminal=False, age_hours=None, age_days=-1),
        default_age_days=3,
    ) == 0.0
    assert archive.compact_age_days_from_args(
        SimpleNamespace(all_terminal=False, age_hours=None, age_days=None),
        default_age_days=3,
    ) == 3.0


def test_terminal_archive_stats_matches_archive_selection_reasons():
    now = 1_000_000.0
    state = {
        "tasks": [
            {"id": "t_old_done", "status": "done", "finished_at": now - 3 * 86400},
            {"id": "t_noise", "status": "failed", "finished_at": now - 60, "kind": "noise"},
            {"id": "t_recent_done", "status": "done", "finished_at": now - 50},
            {"id": "t_cancelled", "status": "cancelled", "cancelled_at": now - 40},
            {"id": "t_forgotten", "status": "forgotten", "finished_at": now - 30},
            {"id": "t_running", "status": "running", "submitted_at": now - 3 * 86400},
        ]
    }

    stats = archive.terminal_archive_stats(
        state,
        age_days=1,
        max_hot_terminal=2,
        extra_archive_predicate=lambda task: task.get("kind") == "noise",
        now_fn=lambda: now,
    )

    assert stats["count"] == 3
    assert stats["age_count"] == 1
    assert stats["extra_count"] == 1
    assert stats["excess_count"] == 1
    assert stats["terminal_total"] == 5
    assert stats["max_hot_terminal"] == 2
    assert stats["approx_bytes"] > 0
    assert stats["by_status"] == {"done": 2, "failed": 1}


def test_archive_terminal_tasks_caps_hot_terminal_records(tmp_path):
    archive_file = tmp_path / "queue_archive.jsonl"
    now = 1_000_000.0
    state = {
        "tasks": [
            {"id": "t_done_1", "status": "done", "finished_at": now - 50},
            {"id": "t_running", "status": "running", "submitted_at": now - 40},
            {"id": "t_done_2", "status": "done", "finished_at": now - 30},
            {"id": "t_failed_3", "status": "failed", "finished_at": now - 20},
            {"id": "t_cancelled_4", "status": "cancelled", "cancelled_at": now - 10},
        ]
    }

    count = archive.archive_terminal_tasks(
        state,
        archive_file=archive_file,
        age_days=999,
        max_hot_terminal=2,
        now_fn=lambda: now,
    )

    assert count == 2
    assert [task["id"] for task in state["tasks"]] == [
        "t_running",
        "t_failed_3",
        "t_cancelled_4",
    ]
    assert [json.loads(line)["id"] for line in archive_file.read_text().splitlines()] == [
        "t_done_1",
        "t_done_2",
    ]


def test_archive_terminal_tasks_keeps_done_results_until_sync_commits(tmp_path):
    archive_file = tmp_path / "queue_archive.jsonl"
    now = 1_000_000.0
    state = {
        "tasks": [
            {
                "id": "t_pending",
                "status": "done",
                "finished_at": now - 3 * 86400,
                "result_dir": "/remote/result",
                "local_result_dir": "/local/result",
                "result_synced_at": None,
            },
            {
                "id": "t_synced",
                "status": "done",
                "finished_at": now - 3 * 86400,
                "result_dir": "/remote/result-2",
                "result_synced_at": now - 60,
            },
            {"id": "t_recent", "status": "done", "finished_at": now - 10},
        ]
    }

    stats = archive.terminal_archive_stats(
        state,
        age_days=1,
        max_hot_terminal=0,
        now_fn=lambda: now,
    )
    count = archive.archive_terminal_tasks(
        state,
        archive_file=archive_file,
        age_days=1,
        max_hot_terminal=0,
        now_fn=lambda: now,
    )

    assert count == 2
    assert [task["id"] for task in state["tasks"]] == ["t_pending"]
    assert stats["pending_result_sync_count"] == 1
    assert stats["count"] == 2
    assert [json.loads(line)["id"] for line in archive_file.read_text().splitlines()] == [
        "t_synced",
        "t_recent",
    ]


def test_archive_terminal_tasks_accepts_extra_predicate(tmp_path):
    archive_file = tmp_path / "queue_archive.jsonl"
    now = 1_000_000.0
    state = {
        "tasks": [
            {"id": "t_noise", "status": "done", "finished_at": now - 1, "kind": "noise"},
            {"id": "t_recent_done", "status": "done", "finished_at": now - 1},
            {"id": "t_running_noise", "status": "running", "kind": "noise"},
        ]
    }

    count = archive.archive_terminal_tasks(
        state,
        archive_file=archive_file,
        age_days=999,
        extra_archive_predicate=lambda task: task.get("kind") == "noise",
        now_fn=lambda: now,
    )

    assert count == 1
    assert [task["id"] for task in state["tasks"]] == ["t_recent_done", "t_running_noise"]
    assert [json.loads(line)["id"] for line in archive_file.read_text().splitlines()] == ["t_noise"]


def test_archive_terminal_tasks_keeps_state_when_archive_write_fails(tmp_path, monkeypatch):
    archive_file = tmp_path / "archive-as-dir"
    archive_file.mkdir()
    state = {"tasks": [{"id": "t1", "status": "cancelled", "finished_at": 1.0}]}
    original_tasks = list(state["tasks"])

    count = archive.archive_terminal_tasks(
        state,
        archive_file=archive_file,
        age_days=1,
        now_fn=lambda: 10 * 86400,
    )

    assert count == 0
    assert state["tasks"] == original_tasks


def test_load_archive_tasks_skips_bad_lines_and_limits_to_newest(tmp_path):
    archive_file = tmp_path / "queue_archive.jsonl"
    archive_file.write_text(
        "\n".join([
            json.dumps({"id": "t1"}),
            "{bad json",
            json.dumps(["not", "a", "task"]),
            json.dumps({"id": "t2"}),
            json.dumps({"id": "t3"}),
        ])
    )

    assert [task["id"] for task in archive.load_archive_tasks(archive_file)] == ["t1", "t2", "t3"]
    assert [task["id"] for task in archive.load_archive_tasks(archive_file, limit=2)] == ["t2", "t3"]


def test_find_task_record_prefers_hot_queue_and_latest_archive_duplicate():
    state = {"tasks": [{"id": "t1", "status": "running"}]}
    archive_records = [
        {"id": "t1", "status": "done"},
        {"id": "t2", "status": "failed", "attempt": 1},
        {"id": "t2", "status": "done", "attempt": 2},
    ]

    task, source = archive.find_task_record(
        "t1",
        state,
        include_archive=True,
        archive_loader=lambda: archive_records,
    )
    assert source == "queue"
    assert task["status"] == "running"

    task, source = archive.find_task_record(
        "t2",
        state,
        include_archive=True,
        archive_loader=lambda: archive_records,
    )
    assert source == "archive"
    assert task["attempt"] == 2

    task, source = archive.find_task_record(
        "t2",
        state,
        include_archive=False,
        archive_loader=lambda: archive_records,
    )
    assert task is None
    assert source == ""
