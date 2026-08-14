from __future__ import annotations

from pathlib import Path

from skill.scheduler_windows_logs import (
    WindowsLogDeps,
    WindowsTaskPathDeps,
    fetch_windows_text_tail,
    is_windows_native_path,
    scan_windows_log_for_patterns,
    windows_path_for_task,
    windows_tail_ps,
)


def _task_path_deps(tmp_path: Path):
    return WindowsTaskPathDeps(
        node_is_windows=lambda node: node == "win",
        windows_path_for_node=lambda node, path: r"F:\work" + str(path).replace(str(tmp_path), "").replace("/", "\\"),
        home=lambda: tmp_path,
    )


def test_windows_tail_ps_reads_locked_file_tail_with_share_readwrite():
    script = windows_tail_ps(r"F:\logs\t.log", max_bytes=123, ps_quote=lambda text: f"Q({text})")

    assert "$p = Q(F:\\logs\\t.log)" in script
    assert "[IO.FileShare]::ReadWrite" in script
    assert "$len - 123" in script


def test_windows_path_for_task_maps_native_home_and_relative_paths(tmp_path):
    deps = _task_path_deps(tmp_path)

    assert is_windows_native_path(r"C:\logs\t.log")
    assert is_windows_native_path(r"\\server\share\t.log")
    assert windows_path_for_task(
        {"node": "linux", "cwd": str(tmp_path / "proj")},
        str(tmp_path / "proj" / "t.log"),
        deps=deps,
    ) == str(tmp_path / "proj" / "t.log")
    assert windows_path_for_task(
        {"node": "win", "cwd": str(tmp_path / "proj")},
        r"C:/native/t.log",
        deps=deps,
    ) == r"C:\native\t.log"
    assert windows_path_for_task(
        {"node": "win", "cwd": str(tmp_path / "proj")},
        str(tmp_path / "proj" / "t.log"),
        deps=deps,
    ) == r"F:\work\proj\t.log"
    assert windows_path_for_task(
        {"node": "win", "cwd": str(tmp_path / "proj")},
        "logs/t.log",
        deps=deps,
    ) == r"F:\work\proj\logs\t.log"


def test_fetch_windows_text_tail_parses_size_marker_and_failures():
    calls = []

    def run_windows_ps(node, script, timeout=0, check=True):
        calls.append((node, script, timeout, check))
        return 0, "tail\n___SZ___\n123\n", ""

    deps = WindowsLogDeps(
        windows_path_for_task=lambda task, path: r"F:\logs\t.log",
        windows_tail_ps=lambda path, max_bytes=4096: f"TAIL({path},{max_bytes})\n",
        ps_quote=lambda text: f"Q({text})",
        run_windows_ps=run_windows_ps,
    )

    assert fetch_windows_text_tail({"node": "win"}, "/tmp/t.log", max_bytes=55, deps=deps) == ("tail\n", 123)
    assert calls == [("win", "TAIL(F:\\logs\\t.log,55)\n" + "\nWrite-Output '___SZ___'\nif (Test-Path -LiteralPath Q(F:\\logs\\t.log)) {\n  Write-Output ([int64](Get-Item -LiteralPath Q(F:\\logs\\t.log)).Length)\n} else {\n  Write-Output 0\n}\n", 10, False)]
    assert fetch_windows_text_tail({}, "/tmp/t.log", deps=deps) == ("", 0)


def test_scan_windows_log_for_patterns_parses_json_string_and_list():
    outputs = iter([('"DONE"', ""), ('["A","B"]', ""), ("not-json", "")])
    calls = []

    def run_windows_ps(node, script, timeout=0, check=True):
        calls.append((node, script, timeout, check))
        out, err = next(outputs)
        return 0, out, err

    deps = WindowsLogDeps(
        windows_path_for_task=lambda task, path: r"F:\logs\t.log",
        windows_tail_ps=lambda *args, **kwargs: "",
        ps_quote=lambda text: f"Q({text})",
        run_windows_ps=run_windows_ps,
    )
    task = {"node": "win", "log_path": "/tmp/t.log"}

    assert scan_windows_log_for_patterns(task, ["DONE"], deps=deps) == ["DONE"]
    assert scan_windows_log_for_patterns(task, ["A", "B"], deps=deps) == ["A", "B"]
    assert scan_windows_log_for_patterns(task, ["C"], deps=deps) == []
    assert "FromBase64String" in calls[0][1]
    assert calls[0][2:] == (15, False)
    assert scan_windows_log_for_patterns({}, ["DONE"], deps=deps) == []
