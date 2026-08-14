from __future__ import annotations

from skill import scheduler_task_log as task_log


def _deps(**overrides):
    deps = {
        "nodes": {},
        "node_is_windows": lambda node: False,
        "windows_path_for_task": lambda task, path: path,
        "ps_quote": lambda text: repr(text),
        "run_windows_ps": lambda *args, **kwargs: (1, "", "unexpected windows"),
        "run_on": lambda *args, **kwargs: (1, "", "unexpected remote"),
    }
    deps.update(overrides)
    return deps


def test_task_log_tail_reports_missing_log_path():
    ok, log_path, text = task_log.tail_task_log(
        {"id": "t-missing"},
        **_deps(),
    )

    assert ok is False
    assert log_path == ""
    assert "has no log_path" in text
    assert "metadata was not committed" in text


def test_task_log_tail_explains_auto_adopted_missing_log_path():
    ok, log_path, text = task_log.tail_task_log(
        {"id": "t-adopted", "auto_adopted": True, "origin": "external"},
        **_deps(),
    )

    assert ok is False
    assert log_path == ""
    assert "no scheduler-captured log_path" in text
    assert "adopted from an already-running external process" in text


def test_task_log_tail_reads_local_file(tmp_path):
    log_path = tmp_path / "task.log"
    log_path.write_text("a\nb\nc\n", encoding="utf-8")

    ok, path, text = task_log.tail_task_log(
        {"id": "t-local", "node": None, "log_path": str(log_path)},
        lines=2,
        **_deps(),
    )

    assert ok is True
    assert path == str(log_path)
    assert text == "b\nc\n"


def test_task_log_tail_routes_remote_unix_node():
    calls = []

    def fake_run_on(node, cmd, timeout=30, check=False):
        calls.append((node, cmd, timeout, check))
        return 0, "remote tail\n", ""

    ok, path, text = task_log.tail_task_log(
        {"id": "t-remote", "node": "node001", "log_path": "/tmp/sched one.log"},
        lines=5,
        **_deps(nodes={"node001": {"host": "node001"}}, run_on=fake_run_on),
    )

    assert ok is True
    assert path == "/tmp/sched one.log"
    assert text == "remote tail\n"
    assert calls == [("node001", "tail -n 5 '/tmp/sched one.log'", 30, False)]


def test_task_log_tail_routes_windows_node():
    calls = []

    def fake_run_windows_ps(node, script, timeout=30, check=False):
        calls.append((node, script, timeout, check))
        return 0, "windows tail\n", ""

    ok, path, text = task_log.tail_task_log(
        {"id": "t-win", "node": "jtl110cpu", "log_path": "/tmp/sched_t.log"},
        lines=7,
        **_deps(
            nodes={"jtl110cpu": {"host": "win-host"}},
            node_is_windows=lambda node: True,
            windows_path_for_task=lambda task, path: r"F:\logs\t.log",
            ps_quote=lambda text: f"Q({text})",
            run_windows_ps=fake_run_windows_ps,
        ),
    )

    assert ok is True
    assert path == "/tmp/sched_t.log"
    assert text == "windows tail\n"
    assert calls and calls[0][0] == "jtl110cpu"
    assert "Get-Content -LiteralPath $path -Tail 7" in calls[0][1]
    assert "$path = Q(F:\\logs\\t.log)" in calls[0][1]
