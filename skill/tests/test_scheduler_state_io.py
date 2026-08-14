from __future__ import annotations

import json
import os

import pytest

from skill import scheduler_state_io as state_io


ALLOW_ENV = "SCHEDULEURM_TEST_ALLOW_EMPTY_QUEUE_RESET"


def test_load_json_quarantines_empty_and_corrupt_files(tmp_path):
    missing = tmp_path / "missing.json"
    assert state_io.load_json(missing, {"ok": True}) == {"ok": True}

    empty = tmp_path / "empty.json"
    empty.write_text("", encoding="utf-8")
    assert state_io.load_json(empty, {"empty": True}, now_fn=lambda: 1234.0) == {"empty": True}
    assert not empty.exists()
    assert (tmp_path / "empty.json.empty-1234").exists()

    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{bad", encoding="utf-8")
    with pytest.raises(RuntimeError, match="corrupt JSON"):
        state_io.load_json(corrupt, {}, now_fn=lambda: 5678.0)
    assert not corrupt.exists()
    assert (tmp_path / "corrupt.json.corrupt-5678").exists()


def test_atomic_write_json_replaces_file_and_removes_tmp(tmp_path):
    path = tmp_path / "state" / "queue.json"
    state_io.atomic_write_json(path, {"tasks": [{"id": "t1"}]})

    assert json.loads(path.read_text(encoding="utf-8")) == {"tasks": [{"id": "t1"}]}
    assert not (tmp_path / "state" / "queue.json.tmp").exists()


def test_guard_missing_queue_state_refuses_recent_backup_unless_allowed(tmp_path, monkeypatch):
    queue_file = tmp_path / "queue.json"
    backup = tmp_path / "queue.json.20260708"
    backup.write_text('{"tasks": [{"id": "t1"}]}', encoding="utf-8")
    os.utime(backup, (1000, 1000))

    with pytest.raises(RuntimeError, match="missing/empty"):
        state_io.guard_missing_queue_state(
            queue_file=queue_file,
            state_dir=tmp_path,
            allow_env=ALLOW_ENV,
            max_age_s=20,
            now_fn=lambda: 1010,
        )

    monkeypatch.setenv(ALLOW_ENV, "1")
    state_io.guard_missing_queue_state(
        queue_file=queue_file,
        state_dir=tmp_path,
        allow_env=ALLOW_ENV,
        max_age_s=20,
        now_fn=lambda: 1010,
    )


def test_guard_empty_queue_save_refuses_to_overwrite_large_or_active_queue(tmp_path, monkeypatch):
    queue_file = tmp_path / "queue.json"
    queue_file.write_text(
        json.dumps({
            "tasks": [
                {"id": "t1", "status": "done"},
                {"id": "t2", "status": "running"},
            ]
        }),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="empty task list"):
        state_io.guard_empty_queue_save(
            {"tasks": []},
            queue_file=queue_file,
            allow_env=ALLOW_ENV,
            min_tasks=10,
            min_active=1,
        )

    monkeypatch.setenv(ALLOW_ENV, "true")
    state_io.guard_empty_queue_save(
        {"tasks": []},
        queue_file=queue_file,
        allow_env=ALLOW_ENV,
        min_tasks=10,
        min_active=1,
    )


def test_load_and_save_scheduler_state_apply_hooks(tmp_path):
    queue_file = tmp_path / "queue.json"
    calls: list[str] = []

    def canonicalize(state):
        calls.append("canonicalize")
        state["canonicalized"] = True

    def compact(state):
        calls.append("compact")
        state.pop("drop_me", None)

    state_io.save_scheduler_state(
        {"tasks": [{"id": "t1"}], "drop_me": True},
        queue_file=queue_file,
        allow_env=ALLOW_ENV,
        empty_guard_min_tasks=20,
        empty_guard_min_active=5,
        canonicalize=canonicalize,
        compact=compact,
    )
    loaded = state_io.load_scheduler_state(
        queue_file=queue_file,
        state_dir=tmp_path,
        allow_env=ALLOW_ENV,
        missing_guard_max_age_s=20,
        canonicalize=canonicalize,
    )

    assert loaded["canonicalized"] is True
    assert "drop_me" not in loaded
    assert calls == ["canonicalize", "compact", "canonicalize"]
