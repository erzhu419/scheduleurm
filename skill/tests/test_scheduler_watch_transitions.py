from __future__ import annotations

from skill.scheduler_watch_transitions import classify_running_terminal_transitions


def test_classify_running_terminal_transitions_marks_new_done_and_failed():
    done = {"id": "t0001", "status": "done"}
    failed = {"id": "t0002", "status": "failed"}
    still_running = {"id": "t0003", "status": "running"}
    tasks = [done, failed, still_running]

    newly_done, newly_crashed = classify_running_terminal_transitions(
        tasks,
        {"t0001": "running", "t0002": "running", "t0003": "running"},
    )

    assert newly_done == [done]
    assert newly_crashed == [failed]
    assert done["notified_done"] is True
    assert failed["notified_done"] is True
    assert "notified_done" not in still_running


def test_classify_running_terminal_transitions_skips_non_running_previous_status():
    task = {"id": "t0004", "status": "done"}

    newly_done, newly_crashed = classify_running_terminal_transitions(
        [task],
        {"t0004": "queued"},
    )

    assert newly_done == []
    assert newly_crashed == []
    assert "notified_done" not in task


def test_classify_running_terminal_transitions_skips_already_notified_tasks():
    task = {"id": "t0005", "status": "failed", "notified_done": True}

    newly_done, newly_crashed = classify_running_terminal_transitions(
        [task],
        {"t0005": "running"},
    )

    assert newly_done == []
    assert newly_crashed == []
    assert task["notified_done"] is True
