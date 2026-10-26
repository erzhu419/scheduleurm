import asyncio
import json
import threading
from types import SimpleNamespace

from textual.widgets import DataTable

from skill import tui
from skill.scheduler_tui_snapshot import read_active_state


def test_ram_column_distinguishes_measurement_from_startup_or_retry_budget():
    task = {"status": "running", "current_ram_mb": 0, "peak_ram_mb": 64000, "ram_mb": 24000}
    assert tui._format_task_ram(task) == "?"
    assert tui._task_ram_mb(task) == 0
    task.update(current_ram_mb=1024, current_ram_measurement="pss")
    assert tui._format_task_ram(task) == tui.sch._format_mem_gb(1024)
    task["current_ram_measurement"] = "rss"
    assert tui._format_task_ram(task) == tui.sch._format_mem_gb(1024, approx=True)
    task["status"] = "queued"
    assert tui._task_ram_mb(task) == 24000
    assert tui._format_task_ram(task) == tui.sch._format_mem_gb(24000, approx=True)


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

    monkeypatch.setattr(tui, "_read_active_display_state", lambda: load(queue))
    monkeypatch.setattr(tui.sch, "load_history", lambda: load(history))
    first = tui._load_display_state()
    assert tui._load_display_state()[0] is first[0]
    assert calls == ["queue.json", "history.json"]
    replacement = tmp_path / "queue.tmp"
    replacement.write_text(json.dumps({"tasks": [{"id": "t200"}]}))
    replacement.replace(queue)
    assert tui._load_display_state()[0]["tasks"][0]["id"] == "t200"
    assert calls == ["queue.json", "history.json", "queue.json"]


def test_display_reader_returns_active_records_without_modifying_queue(tmp_path):
    path = tmp_path / "queue.json"
    tasks = [{"id": "done", "status": "done", "payload": "completed data"}]
    tasks += [{"id": state, "status": state, "remote_pids": [123]}
              for state in ("queued", "launching", "running")]
    original = json.dumps({"next_id": 42, "tasks": tasks})
    path.write_text(original)

    state = read_active_state(path)

    assert [task["id"] for task in state["tasks"]] == ["queued", "launching", "running"]
    assert state["next_id"] == 42
    assert state["tasks"][0]["remote_pids"] == [123]
    assert path.read_text() == original


def test_display_queue_decode_runs_in_separate_interpreter(monkeypatch, tmp_path):
    captured = []
    monkeypatch.setattr(tui.sch, "QUEUE_FILE", tmp_path / "queue.json")
    monkeypatch.setattr(tui.subprocess, "run", lambda args, **kwargs:
                        captured.append((args, kwargs)) or SimpleNamespace(stdout='{"tasks": []}'))

    assert tui._read_active_display_state() == {"tasks": []}
    args, options = captured[0]
    assert args[1] == str(tui._DISPLAY_READER_SOURCE)
    assert args[2] == str(tmp_path / "queue.json")
    assert options["cwd"] == tui.Path.home()


def test_display_read_failure_keeps_previous_snapshot(monkeypatch, tmp_path):
    queue, history = tmp_path / "queue.json", tmp_path / "history.json"
    queue.write_text("{}")
    history.write_text("{}")
    monkeypatch.setattr(tui.sch, "QUEUE_FILE", queue)
    monkeypatch.setattr(tui.sch, "VRAM_FILE", history)
    monkeypatch.setattr(tui, "_DISPLAY_FILE_CACHE", {})
    monkeypatch.setattr(tui, "_read_active_display_state", lambda: {"tasks": [{"id": "t1"}]})
    monkeypatch.setattr(tui.sch, "load_history", lambda: {})
    previous = tui._load_display_state()
    queue.write_text('{"changed": true}')
    monkeypatch.setattr(tui, "_read_active_display_state", lambda: (_ for _ in ()).throw(RuntimeError("read failed")))

    assert tui._load_display_state()[0] is previous[0]


def test_fast_refresh_does_not_start_overlapping_state_loads():
    calls = []
    app = SimpleNamespace(_fast_refreshing=True, run_worker=lambda *a, **k: calls.append(k))
    tui.SchedulerTUI._refresh_fast_state(app)
    assert calls == []


def test_remote_process_rows_render_filter_and_disappear_without_queue_writes(monkeypatch):
    monkeypatch.setattr(tui.SchedulerTUI, "on_mount", lambda self: self._apply_table_columns(
        self.query_one(DataTable)))
    monkeypatch.setattr(tui, "_watcher_status_line", lambda: "")
    monkeypatch.setattr(tui.time, "time", lambda: 1000)
    state = {"tasks": []}
    nodes = [{"name": "node003", "alive": True, "process_snapshot": {
        "received_at": 1000, "processes": [{
            "pid": 101, "pgid": 100, "owner": "aiyw05", "comm": "fluent",
            "cwd": "", "cmdline": "fluent -g", "pcpu": 95, "rss_mb": 512,
            "elapsed_s": 600, "gpus": [], "ancestors": [],
        }],
    }}]

    async def run():
        app = tui.SchedulerTUI()
        async with app.run_test(size=(160, 30)):
            table = app.query_one(DataTable)
            app._snap = {"state": state, "hist": {}, "nodes": nodes, "ts": 1000}
            app._render_from_cache()
            assert table.row_count == 1
            assert table.get_cell("ext-node003-aiyw05-100", "owner").plain.startswith("aiyw05:external")
            app.action_set_filter("queued")
            assert table.row_count == 0
            app.action_set_filter("running")
            assert table.row_count == 1
            nodes[0]["process_snapshot"]["processes"] = []
            app._render_from_cache()
            assert table.row_count == 0
            assert state == {"tasks": []}

    asyncio.run(run())


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


def test_prepared_snapshot_never_reparses_processes_during_interaction(monkeypatch):
    monkeypatch.setattr(tui.SchedulerTUI, "on_mount", lambda self: self._apply_table_columns(
        self.query_one(DataTable)))
    snap = tui._prepare_display_snapshot({
        "state": {"tasks": [{"id": "t1", "status": "queued", "project": "BAPR"}]},
        "hist": {}, "nodes": [], "ts": 1000,
    })
    monkeypatch.setattr(tui, "external_display_tasks", lambda *args, **kwargs:
                        (_ for _ in ()).throw(AssertionError("process scan on UI thread")))

    async def run():
        app = tui.SchedulerTUI()
        async with app.run_test():
            app._snap = snap
            app._render_from_cache()
            app.action_sort_by("ram")
            app.text_filter = "bapr"
            app._render_from_cache()
            assert app.query_one(DataTable).row_count == 1
            app.text_filter = "missing"
            app._render_from_cache()
            assert app.query_one(DataTable).row_count == 0

    asyncio.run(run())


def test_sort_and_small_task_changes_reuse_rows_and_keep_selection(monkeypatch):
    monkeypatch.setattr(tui.SchedulerTUI, "on_mount", lambda self: self._apply_table_columns(
        self.query_one(DataTable)))
    tasks = [{"id": f"t{i}", "status": "queued", "project": "BAPR", "ram_mb": i * 1024}
             for i in (1, 2, 3)]

    async def run():
        app = tui.SchedulerTUI()
        async with app.run_test(size=(180, 30)):
            table = app.query_one(DataTable)
            app._snap = {"state": {"tasks": tasks}, "hist": {}, "nodes": [], "ts": 1000}
            app._render_from_cache()
            table.move_cursor(row=1, animate=False)
            calls = []
            clear, add, remove = table.clear, table.add_row, table.remove_row
            monkeypatch.setattr(table, "clear", lambda *args, **kwargs: calls.append("clear") or clear(*args, **kwargs))
            monkeypatch.setattr(table, "add_row", lambda *args, **kwargs: calls.append("add") or add(*args, **kwargs))
            monkeypatch.setattr(table, "remove_row", lambda *args, **kwargs: calls.append("remove") or remove(*args, **kwargs))
            app.action_sort_by("ram")
            assert [row.key.value for row in table.ordered_rows] == ["t3", "t2", "t1"]
            assert app._selected_task_id() == "t2"
            assert calls == []
            tasks.pop(0)
            tasks.append({"id": "t4", "status": "queued", "project": "BAPR", "ram_mb": 4096})
            app._render_from_cache()
            assert [row.key.value for row in table.ordered_rows] == ["t4", "t3", "t2"]
            assert app._selected_task_id() == "t2"
            assert calls == ["remove", "add"]

    asyncio.run(run())
