from __future__ import annotations

import datetime
from pathlib import Path

from skill.scheduler_failure.terminal_diagnosis import (
    FinalModelSuccessDeps,
    FullLogSuccessScanDeps,
    LocalOomKillDetectionDeps,
    TerminalDiagnosisDeps,
    detect_oom_kills_local,
    diagnose_terminal,
    scan_full_log_for_success,
    terminal_final_model_success,
)


def _deps(
    *,
    nodes=None,
    crash_patterns=None,
    training_markers=None,
    success_patterns=None,
    full_success=None,
    final_model_success=None,
    result_artifact_success=None,
    run_calls=None,
    run_outputs=None,
    windows_tail=None,
    eval_like=False,
    now=1000.0,
):
    run_calls = run_calls if run_calls is not None else []
    run_outputs = list(run_outputs or [])

    def run_on(node, cmd, **kwargs):
        run_calls.append((node, cmd, kwargs))
        if run_outputs:
            return run_outputs.pop(0)
        return 0, "", ""

    return TerminalDiagnosisDeps(
        node_configs=nodes or {"local": {"host": None}, "remote": {"host": "r"}},
        crash_patterns=crash_patterns or ["Traceback", "CUDA out of memory"],
        training_markers=training_markers or ["Epoch", "step", "iter"],
        early_death_seconds=30,
        short_live_seconds=120,
        node_is_windows=lambda node: node == "win",
        fetch_windows_text_tail=lambda task, path, max_bytes=4096: windows_tail or ("", 0),
        run_on=run_on,
        success_patterns_for_task=lambda task: success_patterns or ["DONE", "Training complete!"],
        cmd_looks_like_eval_or_benchmark=lambda cmd: eval_like,
        scan_full_log_for_success=lambda task: full_success or [],
        terminal_final_model_success=lambda task: final_model_success,
        terminal_result_artifact_success=lambda task: result_artifact_success or "",
        now=lambda: now,
    )


def _task(log_path="/tmp/t.log", **overrides):
    task = {
        "id": "t1",
        "node": "local",
        "log_path": log_path,
        "started_at": 0.0,
        "finished_at": 500.0,
        "cmd": "python train.py",
        "peak_vram_mb": 0,
    }
    task.update(overrides)
    return task


def test_auto_adopted_or_missing_log_is_not_called_crash():
    diag = diagnose_terminal(
        _task(log_path=None, auto_adopted=True, started_at=100.0, finished_at=160.0),
        deps=_deps(),
    )

    assert diag["is_crash"] is False
    assert diag["reason"] == "auto-adopted (no scheduler log; cannot diagnose)"
    assert diag["tail"] == "(no log)"
    assert diag["lifetime_s"] == 60
    assert diag["success_marker"] is None


def test_success_marker_classifies_normal_even_when_lifetime_short(tmp_path: Path):
    log = tmp_path / "task.log"
    log.write_text("start\nDONE\n", encoding="utf-8")

    diag = diagnose_terminal(
        _task(log_path=str(log), started_at=10.0, finished_at=20.0),
        deps=_deps(),
    )

    assert diag["is_crash"] is False
    assert diag["reason"] == "normal exit (success marker found)"
    assert diag["success_marker"] == "DONE"


def test_batched_terminal_evidence_is_decisive_without_remote_io():
    run_calls = []
    task = _task(
        node="remote",
        started_at=10.0,
        finished_at=20.0,
        _terminal_batch_probe_only=True,
        _terminal_evidence={
            "log_path": "/tmp/t.log",
            "tail": "work\nDONE\n",
            "log_size": 10,
            "success_marker": "",
            "artifact_checked": True,
        },
    )

    diag = diagnose_terminal(task, deps=_deps(run_calls=run_calls))

    assert diag["is_crash"] is False
    assert diag["success_marker"] == "DONE"
    assert run_calls == []


def test_batched_failure_marker_is_decisive_without_remote_io():
    run_calls = []
    task = _task(
        node="remote",
        started_at=10.0,
        finished_at=20.0,
        _terminal_batch_probe_only=True,
        _terminal_evidence={
            "log_path": "/tmp/t.log",
            "tail": "short log\n",
            "log_size": 10,
            "success_marker": "",
            "failure_reason": "result_failure:_FAILED.exitcode=7",
            "artifact_checked": True,
        },
    )

    diag = diagnose_terminal(task, deps=_deps(run_calls=run_calls))

    assert diag["is_crash"] is True
    assert "result_failure:_FAILED.exitcode=7" in diag["reason"]
    assert run_calls == []


def test_batched_ambiguous_evidence_defers_to_bounded_deep_diagnosis():
    run_calls = []
    task = _task(
        node="remote",
        _terminal_batch_probe_only=True,
        _terminal_evidence={
            "log_path": "/tmp/t.log",
            "tail": "work in progress\n",
            "log_size": 17,
            "success_marker": "",
            "artifact_checked": True,
        },
    )

    diag = diagnose_terminal(task, deps=_deps(run_calls=run_calls))

    assert diag["needs_deep_terminal_diagnosis"] is True
    assert run_calls == []


def test_error_pattern_overrides_success_marker(tmp_path: Path):
    log = tmp_path / "task.log"
    log.write_text("DONE\nTraceback\n", encoding="utf-8")

    diag = diagnose_terminal(_task(log_path=str(log)), deps=_deps())

    assert diag["is_crash"] is True
    assert diag["reason"].startswith("err_pattern: Traceback")
    assert diag["success_marker"] == "DONE"


def test_headroom_success_marker_does_not_match_embedded_oom(tmp_path: Path):
    log = tmp_path / "task.log"
    log.write_text("HEADROOM CONTROLLER COMPLETE\nDONE\n", encoding="utf-8")

    diag = diagnose_terminal(
        _task(log_path=str(log)),
        deps=_deps(crash_patterns=["OOM"]),
    )

    assert diag["is_crash"] is False
    assert diag["success_marker"] == "DONE"


def test_standalone_oom_still_overrides_success_marker(tmp_path: Path):
    log = tmp_path / "task.log"
    log.write_text("OOM\nDONE\n", encoding="utf-8")

    diag = diagnose_terminal(
        _task(log_path=str(log)),
        deps=_deps(crash_patterns=["OOM"]),
    )

    assert diag["is_crash"] is True
    assert diag["reason"].startswith("err_pattern: OOM")


def test_success_marker_outside_tail_uses_full_log_scan(tmp_path: Path):
    log = tmp_path / "task.log"
    log.write_text("Training complete!\n" + ("x" * 9000), encoding="utf-8")

    diag = diagnose_terminal(
        _task(log_path=str(log), peak_vram_mb=1000),
        deps=_deps(success_patterns=["Training complete!"], full_success=["Training complete!"]),
    )

    assert diag["is_crash"] is False
    assert diag["success_marker"] == "Training complete!"


def test_result_artifact_success_classifies_normal_when_log_missing(tmp_path: Path):
    missing_log = tmp_path / "missing.log"

    diag = diagnose_terminal(
        _task(
            log_path=str(missing_log),
            started_at=10.0,
            finished_at=50.0,
            cmd="python eval.py --out-dir results/run0",
        ),
        deps=_deps(result_artifact_success="result_artifact:summary.csv"),
    )

    assert diag["is_crash"] is False
    assert diag["reason"] == "normal exit (success marker found)"
    assert diag["success_marker"] == "result_artifact:summary.csv"


def test_redirected_empty_wrapper_log_reads_real_log_and_trusts_it(tmp_path: Path):
    wrapper = tmp_path / "wrapper.log"
    real = tmp_path / "real.log"
    wrapper.write_text("", encoding="utf-8")
    real.write_text("Epoch 1\nDONE\n", encoding="utf-8")

    diag = diagnose_terminal(
        _task(
            log_path=str(wrapper),
            cmd=f"python train.py > {real} 2>&1",
            peak_vram_mb=2048,
        ),
        deps=_deps(),
    )

    assert diag["is_crash"] is False
    assert diag["success_marker"] == "DONE"
    assert diag["log_size"] == len("Epoch 1\nDONE\n")


def test_small_log_after_long_lifetime_adds_disk_full_reason(tmp_path: Path):
    log = tmp_path / "task.log"
    log.write_text("tiny", encoding="utf-8")
    run_calls = []

    diag = diagnose_terminal(
        _task(log_path=str(log), node="remote", started_at=0.0, finished_at=500.0),
        deps=_deps(run_calls=run_calls, run_outputs=[(0, "tiny___SZ___4\n", ""), (0, "97\n", "")]),
    )

    assert diag["is_crash"] is True
    assert "log only 4B after 500s" in diag["reason"]
    assert "DISK_FULL" in diag["reason"]
    assert any("df -P" in call[1] for call in run_calls)


def test_training_marker_without_success_is_crash(tmp_path: Path):
    log = tmp_path / "task.log"
    log.write_text(("Epoch 1\nstep 10\nstill running\n" * 30), encoding="utf-8")

    diag = diagnose_terminal(
        _task(log_path=str(log), started_at=0.0, finished_at=500.0, peak_vram_mb=2048),
        deps=_deps(),
    )

    assert diag["is_crash"] is True
    assert "training markers present but no success marker" in diag["reason"]


def test_gpu_work_without_training_marker_is_crash_but_eval_like_is_ambiguous(tmp_path: Path):
    log = tmp_path / "task.log"
    log.write_text(("custom logger output\n" * 40), encoding="utf-8")
    task = _task(log_path=str(log), peak_vram_mb=2048, started_at=0.0, finished_at=500.0)

    diag = diagnose_terminal(task, deps=_deps())
    eval_diag = diagnose_terminal(task, deps=_deps(eval_like=True))

    assert diag["is_crash"] is True
    assert "GPU work observed" in diag["reason"]
    assert eval_diag["is_crash"] is False
    assert eval_diag["reason"] == "ambiguous; assumed normal"


def test_scan_full_log_for_success_reads_local_log(tmp_path: Path):
    log = tmp_path / "task.log"
    log.write_text("start\n" + ("x" * 5000) + "\nTraining complete!\n", encoding="utf-8")

    matched = scan_full_log_for_success(
        {"node": "local", "log_path": str(log)},
        deps=FullLogSuccessScanDeps(
            node_configs={"local": {"host": None}},
            node_is_windows=lambda node: False,
            scan_windows_log_for_patterns=lambda task, patterns: [],
            success_patterns_for_task=lambda task: ["Training complete!", "DONE"],
            run_on=lambda *args, **kwargs: (1, "", ""),
        ),
    )

    assert matched == ["Training complete!"]


def test_scan_full_log_for_success_uses_remote_grep():
    calls = []

    matched = scan_full_log_for_success(
        {"node": "remote", "log_path": "/tmp/t.log"},
        deps=FullLogSuccessScanDeps(
            node_configs={"remote": {"host": "h"}},
            node_is_windows=lambda node: False,
            scan_windows_log_for_patterns=lambda task, patterns: [],
            success_patterns_for_task=lambda task: ["DONE", "Training complete!"],
            run_on=lambda node, cmd, **kwargs: calls.append((node, cmd, kwargs)) or (0, "abc DONE xyz\n", ""),
        ),
    )

    assert matched == ["DONE"]
    assert calls and "grep -F -m 1" in calls[0][1]


def test_terminal_final_model_success_accepts_complete_local_group(tmp_path: Path):
    files = [tmp_path / "actor.pt", tmp_path / "critic.pt", tmp_path / "meta.json"]
    for file in files:
        file.write_text("x", encoding="utf-8")

    result = terminal_final_model_success(
        {"node": "local"},
        deps=FinalModelSuccessDeps(
            node_configs={"local": {"host": None}},
            final_model_file_groups_from_cmd=lambda task: [("/model", [str(file) for file in files])],
            mtimes_match_task_window=lambda task, mtimes: len(mtimes) == 3,
            node_is_windows=lambda node: False,
            windows_path_for_task=lambda task, path: path,
            ps_quote=lambda text: repr(text),
            run_windows_ps=lambda *args, **kwargs: (1, "", ""),
            bash_path_arg=lambda path: path,
            run_on=lambda *args, **kwargs: (1, "", ""),
        ),
    )

    assert result == "final_model_files"


def test_detect_oom_kills_local_flips_incomplete_done_task():
    event_dt = datetime.datetime.strptime("2026 Jul 09 12:00:00", "%Y %b %d %H:%M:%S")
    event_ts = event_dt.timestamp()
    state = {"tasks": [
        {
            "id": "victim",
            "node": "local",
            "status": "done",
            "started_at": event_ts - 10,
            "finished_at": event_ts + 10,
            "peak_vram_mb": 0,
            "_diagnosis": {"reason": "ambiguous"},
        },
        {
            "id": "success",
            "node": "local",
            "status": "done",
            "started_at": event_ts - 10,
            "finished_at": event_ts + 10,
            "_diagnosis": {"success_marker": "DONE"},
        },
    ]}

    flipped = detect_oom_kills_local(
        state,
        deps=LocalOomKillDetectionDeps(
            syslog_path="/tmp/syslog",
            path_exists=lambda path: True,
            check_output=lambda *args, **kwargs: (
                "Jul 09 12:00:00 host kernel: Out of memory: Killed process 123 python\n"
            ),
            now=lambda: event_ts + 60,
            current_year=lambda: 2026,
        ),
    )

    assert [task["id"] for task in flipped] == ["victim"]
    assert state["tasks"][0]["status"] == "failed"
    assert state["tasks"][0]["_diagnosis"]["is_crash"] is True
    assert state["tasks"][1]["status"] == "done"
