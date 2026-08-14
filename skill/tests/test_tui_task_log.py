from contextlib import contextmanager

from skill import tui


@contextmanager
def _dummy_lock(*args, **kwargs):
    yield


def test_tui_task_log_payload_reads_scheduler_tail(monkeypatch):
    calls = []

    def fake_find(task_id, include_archive=True):
        calls.append(("find", task_id, include_archive))
        return {"id": task_id, "status": "done", "node": "node006", "log_path": "/tmp/x.log"}, "queue"

    def fake_tail(task, lines=80):
        calls.append(("tail", task["id"], lines))
        return True, "/tmp/x.log", "DONE\n"

    monkeypatch.setattr(tui.sch, "state_lock", _dummy_lock)
    monkeypatch.setattr(tui.sch, "_find_task_record", fake_find)
    monkeypatch.setattr(tui.sch, "_tail_task_log", fake_tail)

    payload = tui._read_task_log_payload("t20846", lines=7)

    assert payload == {
        "ok": True,
        "id": "t20846",
        "status": "done",
        "node": "node006",
        "log_path": "/tmp/x.log",
        "source": "queue",
        "text": "DONE\n",
    }
    assert calls == [("find", "t20846", True), ("tail", "t20846", 7)]


def test_tui_task_log_payload_reports_missing_task(monkeypatch):
    monkeypatch.setattr(tui.sch, "state_lock", _dummy_lock)
    monkeypatch.setattr(tui.sch, "_find_task_record", lambda task_id, include_archive=True: (None, ""))

    payload = tui._read_task_log_payload("t99999", lines=3)

    assert payload["ok"] is False
    assert payload["id"] == "t99999"
    assert "not found" in payload["text"]
