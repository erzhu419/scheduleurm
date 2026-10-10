import asyncio
import json
import threading
from types import SimpleNamespace

from textual.widgets import DataTable

from skill import tui


def test_display_state_reuses_unchanged_files_and_refreshes_atomic_replacement(monkeypatch, tmp_path):
    queue = tmp_path / "queue.json"
    history = tmp_path / "history.json"
    queue.write_text(json.dumps({"tasks": [{"id": "t1"}]}))
    history.write_text("{}")
    calls = []
    monkeypatch.setattr(tui, "_DISPLAY_FILE_CACHE", {})
    monkeypatch.setattr(tui.sch, "QUEUE_FILE", queue)
    monkeypatch.setattr(tui.sch, "VRAM_FILE", history)

    def load(path):
        calls.append(path.name)
        return json.loads(path.read_text())

    monkeypatch.setattr(tui.sch, "load_state", lambda: load(queue))
    monkeypatch.setattr(tui.sch, "load_history", lambda: load(history))
    first = tui._load_display_state()
    assert tui._load_display_state()[0] is first[0]
    assert calls == ["queue.json", "history.json"]
    replacement = tmp_path / "queue.tmp"
    replacement.write_text(json.dumps({"tasks": [{"id": "t200"}]}))
    replacement.replace(queue)
    assert tui._load_display_state()[0]["tasks"][0]["id"] == "t200"
    assert calls == ["queue.json", "history.json", "queue.json"]


def test_fast_refresh_does_not_start_overlapping_state_loads():
    calls = []
    app = SimpleNamespace(_fast_refreshing=True, run_worker=lambda *a, **k: calls.append(k))
    tui.SchedulerTUI._refresh_fast_state(app)
    assert calls == []


def test_initial_state_load_runs_off_ui_thread(monkeypatch):
    ui_thread = threading.get_ident()
    threads = []
    snap = {"state": {"tasks": []}, "hist": {}, "nodes": [], "ts": 0}

    def load():
        threads.append(threading.get_ident())
        return snap

    monkeypatch.setattr(tui, "_fast_snapshot", load)
    monkeypatch.setattr(tui.SchedulerTUI, "_kick_probe", lambda self: None)

    async def run():
        app = tui.SchedulerTUI()
        async with app.run_test() as pilot:
            await pilot.pause()
            await app.workers.wait_for_complete()
            assert threads and all(thread != ui_thread for thread in threads)

    asyncio.run(run())


def test_thousands_of_unchanged_rows_do_not_repaint_and_updates_preserve_cursor(monkeypatch):
    monkeypatch.setattr(tui.SchedulerTUI, "on_mount", lambda self: self._apply_table_columns(
        self.query_one(DataTable)))
    monkeypatch.setattr(tui, "_watcher_status_line", lambda: "")
    monkeypatch.setattr(tui.time, "time", lambda: 1000)
    tasks = [{"id": f"t{i:06}", "status": "queued", "project": "Thesis", "ram_mb": 1024}
             for i in range(6000)]

    async def run():
        app = tui.SchedulerTUI()
        async with app.run_test(size=(120, 30)) as pilot:
            table = app.query_one(DataTable)
            app._snap = {"state": {"tasks": tasks}, "hist": {}, "nodes": [], "ts": 1000}
            app._render_from_cache()
            assert table.row_count == 6000
            table.move_cursor(row=3500, animate=False)
            changed = []
            update = table.update_cell

            def record(*args, **kwargs):
                changed.append(args[:2])
                return update(*args, **kwargs)

            monkeypatch.setattr(table, "update_cell", record)
            app._render_from_cache()
            assert changed == []
            tasks[3500]["ram_mb"] = 2048
            app._render_from_cache()
            assert changed == [("t003500", "ram")]
            assert app._selected_task_id() == "t003500"
            changed.clear()
            tasks[3500].update(status="running", node="node001", started_at=990)
            app._render_from_cache()
            assert len(changed) == len(app.column_order)
            assert app._selected_task_id() == "t003500"
            assert table.get_cell("t003500", "status").plain == "running"
            app.action_set_filter("running")
            assert table.row_count == 1
            app.action_set_filter("queued")
            assert table.row_count == 5999
            app.action_set_filter("all")
            assert table.row_count == 6000

    asyncio.run(run())
