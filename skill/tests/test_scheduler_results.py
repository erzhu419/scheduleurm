from __future__ import annotations

from argparse import Namespace
import base64
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import struct
import threading
import time

from skill.scheduler_result.discovery import (
    ResultArtifactDeps,
    ResultsCommandDeps,
    apply_discovered_result_artifacts,
    cmd_results,
    collect_terminal_evidence,
    discover_result_artifacts,
    final_model_file_groups_from_cmd,
    result_artifacts_from_cmd,
    task_result_artifacts_for_display,
    terminal_result_artifact_success,
)


def _artifact_deps(tmp_path):
    return ResultArtifactDeps(
        node_configs={"local": {"host": None}, "node001": {"host": "node001"}},
        fetch_log_tail=lambda task: ((tmp_path / "tail.txt").read_text() if (tmp_path / "tail.txt").exists() else "", 0),
        node_is_windows=lambda node: False,
        windows_path_for_task=lambda task, path: path,
        ps_quote=repr,
        run_windows_ps=lambda *args, **kwargs: (1, "", "unexpected windows"),
        run_on=lambda *args, **kwargs: (1, "", "unexpected remote"),
        now=lambda: 123.0,
    )


def _write_int64_npy(path: Path, values: list[int]) -> None:
    header = {
        "descr": "<i8",
        "fortran_order": False,
        "shape": (len(values),),
    }
    header_text = repr(header)
    preamble_len = 10
    padding = 16 - ((preamble_len + len(header_text) + 1) % 16)
    header_bytes = (header_text + " " * padding + "\n").encode("latin1")
    with path.open("wb") as handle:
        handle.write(b"\x93NUMPY")
        handle.write(bytes([1, 0]))
        handle.write(struct.pack("<H", len(header_bytes)))
        handle.write(header_bytes)
        for value in values:
            handle.write(struct.pack("<q", int(value)))


def test_result_artifacts_from_cmd_handles_save_root_run_name_and_output_file():
    task = {
        "id": "t1",
        "node": "node001",
        "cwd": "/work/proj",
        "cmd": (
            "python train.py --save-root runs --run-name seed0 "
            "--out-json metrics/out.json"
        ),
    }

    artifacts = result_artifacts_from_cmd(task)
    paths = {(rec["kind"], rec["path"], rec["source"]) for rec in artifacts}

    assert ("file", "/work/proj/metrics/out.json", "cmd:--out-json") in paths
    assert ("dir", "/work/proj/runs/seed0", "cmd:save_root+run_name") in paths
    assert ("dir", "/work/proj/runs/model/seed0", "cmd:save_root+model+run_name") in paths
    assert ("dir", "/work/proj/runs/logs/seed0", "cmd:save_root+logs+run_name") in paths
    assert ("dir", "/work/proj/runs/pic/seed0", "cmd:save_root+pic+run_name") in paths


def test_result_artifacts_from_cmd_handles_out_dir():
    task = {
        "id": "t1",
        "node": "node001",
        "cwd": "/work/proj",
        "cmd": "python -m eval --out-dir results/final_eval",
    }

    artifacts = result_artifacts_from_cmd(task)

    assert {
        "kind": "dir",
        "path": "/work/proj/results/final_eval",
        "node": "node001",
        "source": "cmd:--out-dir",
    } in artifacts


def test_terminal_result_artifact_success_uses_fresh_summary_file(tmp_path):
    out_dir = tmp_path / "results" / "run0"
    out_dir.mkdir(parents=True)
    (out_dir / "summary.csv").write_text("metric,value\nscore,1\n", encoding="utf-8")
    task = {
        "id": "t1",
        "node": "local",
        "cwd": str(tmp_path),
        "cmd": "python -m eval --out-dir results/run0",
        "started_at": (out_dir / "summary.csv").stat().st_mtime - 10,
    }

    marker = terminal_result_artifact_success(task, deps=_artifact_deps(tmp_path))

    assert marker == "result_artifact:summary.csv"


def test_terminal_result_artifact_success_uses_fresh_summary_markdown(tmp_path):
    out_dir = tmp_path / "results" / "run0"
    out_dir.mkdir(parents=True)
    summary = out_dir / "summary.md"
    summary.write_text("# Complete report\n", encoding="utf-8")
    task = {
        "id": "t1",
        "node": "local",
        "cwd": str(tmp_path),
        "cmd": "python -m eval --out-dir results/run0",
        "started_at": summary.stat().st_mtime - 10,
    }

    marker = terminal_result_artifact_success(task, deps=_artifact_deps(tmp_path))

    assert marker == "result_artifact:summary.md"


def test_terminal_result_artifact_success_uses_fresh_success_json(tmp_path):
    out_dir = tmp_path / "results" / "run0"
    out_dir.mkdir(parents=True)
    success = out_dir / "_SUCCESS.json"
    success.write_text('{"status": "completed"}\n', encoding="utf-8")
    task = {
        "id": "t1",
        "node": "local",
        "cwd": str(tmp_path),
        "cmd": "python -m eval --out-dir results/run0",
        "started_at": success.stat().st_mtime - 10,
    }

    marker = terminal_result_artifact_success(task, deps=_artifact_deps(tmp_path))

    assert marker == "result_artifact:_SUCCESS.json"


def test_terminal_result_artifact_success_uses_fresh_group_json(tmp_path):
    out_dir = tmp_path / "results" / "run0"
    out_dir.mkdir(parents=True)
    group = out_dir / "group.json"
    group.write_text('{"status": "complete"}\n', encoding="utf-8")
    task = {
        "id": "t1",
        "node": "local",
        "cwd": str(tmp_path),
        "cmd": "python -m eval --out-dir results/run0",
        "started_at": group.stat().st_mtime - 10,
    }

    marker = terminal_result_artifact_success(task, deps=_artifact_deps(tmp_path))

    assert marker == "result_artifact:group.json"


def test_collect_terminal_evidence_reads_local_tasks_in_one_batch(tmp_path):
    out_dir = tmp_path / "results" / "run0"
    out_dir.mkdir(parents=True)
    summary = out_dir / "summary.md"
    summary.write_text("# done\n", encoding="utf-8")
    log1 = tmp_path / "t1.log"
    log2 = tmp_path / "t2.log"
    log1.write_text("work\nDONE\n", encoding="utf-8")
    log2.write_text("Traceback\n", encoding="utf-8")
    tasks = [
        {
            "id": "t1",
            "node": "local",
            "cwd": str(tmp_path),
            "cmd": "python eval.py --out-dir results/run0",
            "log_path": str(log1),
            "started_at": summary.stat().st_mtime - 10,
        },
        {
            "id": "t2",
            "node": "local",
            "cwd": str(tmp_path),
            "cmd": "python eval.py",
            "log_path": str(log2),
            "started_at": summary.stat().st_mtime - 10,
        },
    ]

    evidence = collect_terminal_evidence(tasks, deps=_artifact_deps(tmp_path))

    assert evidence["t1"]["success_marker"] == "result_artifact:summary.md"
    assert evidence["t1"]["tail"].endswith("DONE\n")
    assert evidence["t2"]["success_marker"] == ""
    assert "Traceback" in evidence["t2"]["tail"]


def test_collect_terminal_evidence_batches_remote_tasks_without_remote_python(tmp_path):
    calls = []

    def encoded(value: str) -> str:
        return base64.b64encode(value.encode("utf-8")).decode("ascii")

    def run_on(node, cmd, **kwargs):
        calls.append((node, cmd, kwargs))
        assert "python3" not in cmd
        done_tail = encoded("DONE\n")
        traceback_tail = encoded("Traceback")
        return 0, (
            f"E\tt1\t5\t{done_tail}\t\t0\n"
            f"E\tt2\t9\t{traceback_tail}\t\t0\n"
        ), ""

    deps = ResultArtifactDeps(
        node_configs={"node001": {"host": "node001"}},
        fetch_log_tail=lambda task: ("", 0),
        node_is_windows=lambda node: False,
        windows_path_for_task=lambda task, path: path,
        ps_quote=repr,
        run_windows_ps=lambda *args, **kwargs: (1, "", "unexpected windows"),
        run_on=run_on,
        now=lambda: 123.0,
    )
    tasks = [
        {"id": "t1", "node": "node001", "log_path": "/tmp/t1.log", "cmd": "work"},
        {"id": "t2", "node": "node001", "log_path": "/tmp/t2.log", "cmd": "work"},
    ]

    evidence = collect_terminal_evidence(tasks, deps=deps)

    assert len(calls) == 1
    assert evidence["t1"]["tail"] == "DONE\n"
    assert evidence["t2"]["tail"] == "Traceback"


def test_collect_terminal_evidence_serializes_chunks_per_node_and_reads_markers():
    calls = []
    active = 0
    max_active = 0
    lock = threading.Lock()

    def encoded(value: str) -> str:
        return base64.b64encode(value.encode("utf-8")).decode("ascii")

    def run_on(node, cmd, **kwargs):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            calls.append((node, cmd, kwargs))
            time.sleep(0.02)
            if "tid=t1" in cmd:
                return 0, (
                    f"E\tt1\t0\t\t{encoded('_SUCCESS.json')}\t100\t\t0\t\n"
                ), ""
            return 0, (
                f"E\tt2\t198\t{encoded('failed')}\t\t0\t"
                f"{encoded('_FAILED.exitcode')}\t101\t{encoded('7')}\n"
            ), ""
        finally:
            with lock:
                active -= 1

    deps = ResultArtifactDeps(
        node_configs={"node001": {"host": "node001"}},
        fetch_log_tail=lambda task: ("", 0),
        node_is_windows=lambda node: False,
        windows_path_for_task=lambda task, path: path,
        ps_quote=repr,
        run_windows_ps=lambda *args, **kwargs: (1, "", "unexpected windows"),
        run_on=run_on,
        now=lambda: 123.0,
    )
    tasks = [
        {
            "id": "t1",
            "node": "node001",
            "cwd": "/work",
            "cmd": "python eval.py --out-dir results/t1",
            "log_path": "/tmp/t1.log",
            "started_at": 90,
        },
        {
            "id": "t2",
            "node": "node001",
            "cwd": "/work",
            "cmd": "python eval.py --out-dir results/t2",
            "log_path": "/tmp/t2.log",
            "started_at": 90,
        },
    ]

    evidence = collect_terminal_evidence(tasks, deps=deps, batch_size=1, max_workers=8)

    assert len(calls) == 2
    assert max_active == 1
    assert evidence["t1"]["success_marker"] == "result_artifact:_SUCCESS.json"
    assert evidence["t1"]["failure_reason"] == ""
    assert evidence["t2"]["success_marker"] == ""
    assert evidence["t2"]["failure_reason"] == "result_failure:_FAILED.exitcode=7"


def test_collect_terminal_evidence_returns_at_deadline_when_remote_call_blocks(tmp_path):
    entered = threading.Event()
    release = threading.Event()

    def blocking_run_on(*_args, **_kwargs):
        entered.set()
        release.wait(timeout=2.0)
        return 1, "", "blocked"

    deps = replace(_artifact_deps(tmp_path), run_on=blocking_run_on)
    task = {
        "id": "blocked",
        "status": "running",
        "node": "node001",
        "log_path": "/tmp/blocked.log",
        "started_at": 1.0,
        "cmd": "python work.py",
    }

    started_at = time.monotonic()
    try:
        evidence = collect_terminal_evidence(
            [task],
            deps=deps,
            timeout_s=0.05,
        )
    finally:
        release.set()

    assert entered.wait(timeout=0.5)
    assert evidence == {}
    assert time.monotonic() - started_at < 0.5


def test_terminal_result_artifact_success_uses_completed_iteration_npy(tmp_path):
    out_dir = tmp_path / "results" / "run0"
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True)
    iteration_path = log_dir / "iteration.npy"
    _write_int64_npy(iteration_path, [0, 1, 2, 3, 4])
    task = {
        "id": "t1",
        "node": "local",
        "cwd": str(tmp_path),
        "cmd": "python -m train --save_root results --run_name run0 --max_iters 5",
        "started_at": iteration_path.stat().st_mtime - 10,
    }

    marker = terminal_result_artifact_success(task, deps=_artifact_deps(tmp_path))

    assert marker == "result_artifact:iteration.npy"


def test_terminal_result_artifact_success_uses_remote_completed_iteration_npy(tmp_path):
    calls = []

    def run_on(node, cmd, **kwargs):
        calls.append(cmd)
        if "python3 -c" in cmd and "logs/iteration.npy" in cmd:
            return 0, "NPY 100 5 4\n", ""
        return 0, "", ""

    task = {
        "id": "t1",
        "node": "node001",
        "cwd": "/work/proj",
        "cmd": "python -m train --save_root results --run_name run0 --max_iters 5",
        "started_at": 90,
    }
    deps = ResultArtifactDeps(
        node_configs={"node001": {"host": "node001"}},
        fetch_log_tail=lambda task: ("", 0),
        node_is_windows=lambda node: False,
        windows_path_for_task=lambda task, path: path,
        ps_quote=repr,
        run_windows_ps=lambda *args, **kwargs: (1, "", "unexpected windows"),
        run_on=run_on,
        now=lambda: 123.0,
    )

    marker = terminal_result_artifact_success(task, deps=deps)

    assert marker == "result_artifact:iteration.npy"
    assert any("summary.csv" in cmd for cmd in calls)


def test_terminal_result_artifact_success_rejects_incomplete_iteration_npy(tmp_path):
    out_dir = tmp_path / "results" / "run0"
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True)
    iteration_path = log_dir / "iteration.npy"
    _write_int64_npy(iteration_path, [0, 1, 2])
    task = {
        "id": "t1",
        "node": "local",
        "cwd": str(tmp_path),
        "cmd": "python -m train --save_root results --run_name run0 --max_iters 5",
        "started_at": iteration_path.stat().st_mtime - 10,
    }

    marker = terminal_result_artifact_success(task, deps=_artifact_deps(tmp_path))

    assert marker == ""


def test_final_model_file_groups_follow_save_root_model_layout():
    task = {
        "cwd": "/work/proj",
        "cmd": "python train.py --save_root results --name runA",
    }

    groups = final_model_file_groups_from_cmd(task)

    assert groups == [(
        "/work/proj/results/model/runA",
        [
            "/work/proj/results/model/runA/final_policy",
            "/work/proj/results/model/runA/final_q",
            "/work/proj/results/model/runA/final_norm",
        ],
    )]


def test_discover_result_artifacts_reads_tail_lines_and_deduplicates_sources(tmp_path):
    (tmp_path / "tail.txt").write_text(
        "warmup\nResults saved to outputs/final.csv\nSaved: outputs/final.csv\n",
        encoding="utf-8",
    )
    task = {
        "id": "t1",
        "node": "local",
        "cwd": str(tmp_path),
        "cmd": "python eval.py --output outputs/final.csv",
        "log_path": str(tmp_path / "run.log"),
    }

    artifacts = discover_result_artifacts(task, deps=_artifact_deps(tmp_path))

    assert len(artifacts) == 1
    assert artifacts[0]["path"] == str(tmp_path / "outputs/final.csv")
    assert artifacts[0]["kind"] == "file"
    assert artifacts[0]["source"] == "cmd:--output,log"


def test_apply_discovered_result_artifacts_preserves_stored_results_and_timestamp(tmp_path):
    task = {
        "id": "t1",
        "node": "node001",
        "result_artifacts": [{"path": "/old/out", "node": "node001", "kind": "dir", "source": "old"}],
    }

    out = apply_discovered_result_artifacts(
        task,
        [{"path": "/new/out.csv", "node": "node001", "kind": "file", "source": "new"}],
        deps=_artifact_deps(tmp_path),
    )

    assert out == task["result_artifacts"]
    assert task["result_artifacts_discovered_at"] == 123.0
    assert {rec["path"] for rec in out} == {"/old/out", "/new/out.csv"}


def test_cmd_results_filters_queue_and_archive_and_emits_json(capsys):
    queue_task = {
        "id": "t1",
        "status": "done",
        "project": "Proj",
        "signature": "Proj/run",
        "node": "node001",
        "description": "queue task",
        "submitted_at": 1,
        "result_artifacts": [{"path": "/queue/out.csv", "node": "node001", "kind": "file", "source": "stored"}],
    }
    archive_task = {
        "id": "t2",
        "status": "done",
        "project": "Proj",
        "signature": "Proj/archive",
        "node": "node002",
        "description": "archive task",
        "finished_at": 2,
        "result_artifacts": [{"path": "/archive/out.csv", "node": "node002", "kind": "file", "source": "stored"}],
    }

    @contextmanager
    def state_lock():
        yield

    args = Namespace(
        task_ids=[],
        no_archive=False,
        scan_logs=False,
        no_log_scan=True,
        status=["done"],
        project="Proj",
        signature="Proj/*",
        limit=10,
        include_empty=False,
        json=True,
    )
    cmd_results(
        args,
        deps=ResultsCommandDeps(
            state_lock=state_lock,
            load_state=lambda: {"tasks": [queue_task]},
            load_archive_tasks=lambda: [archive_task],
            task_result_artifacts_for_display=lambda task, include_log=True: task_result_artifacts_for_display(
                task,
                include_log=include_log,
                deps=_artifact_deps(Path("/tmp")),
            ),
        ),
    )

    text = capsys.readouterr().out
    assert '"matched": 2' in text
    assert '"id": "t2"' in text
    assert '"id": "t1"' in text
    assert text.index('"id": "t2"') < text.index('"id": "t1"')
