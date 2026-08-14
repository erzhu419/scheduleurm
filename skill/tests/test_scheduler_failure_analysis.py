from pathlib import Path

from skill.scheduler_failure.analysis import (
    CompletedLogCrashScanDeps,
    FailureClassificationDeps,
    LogTailDeps,
    cached_task_success_marker,
    classify_failure,
    cmd_looks_like_eval_or_benchmark,
    fetch_log_tail,
    scan_completed_log_for_crash,
    success_patterns_for_task,
)


def _tail_deps(*, node_configs=None, run_on=None, node_is_windows=None, fetch_windows=None):
    return LogTailDeps(
        node_configs=node_configs or {"local": {"host": None}},
        node_is_windows=node_is_windows or (lambda node: False),
        fetch_windows_text_tail=fetch_windows or (lambda *a, **k: ("", 0)),
        run_on=run_on or (lambda *a, **k: (1, "", "")),
    )


def _classify_deps():
    return FailureClassificationDeps(
        env_missing_patterns=["No such file or directory"],
        python_import_patterns=["ModuleNotFoundError", "No module named"],
        cuda_runtime_patterns=["CUDA driver version is insufficient"],
        invalid_flag_patterns=["unrecognized arguments"],
        disk_full_patterns=["No space left on device", "Disk quota exceeded"],
        oom_patterns=["CUDA out of memory", "MemoryError"],
    )


def test_fetch_log_tail_reads_local_tail(tmp_path: Path):
    log = tmp_path / "task.log"
    log.write_text("hello\n" + ("x" * 5000) + "\nDONE\n")

    text, size = fetch_log_tail(
        {"node": "local", "log_path": str(log)},
        deps=_tail_deps(),
    )

    assert size == log.stat().st_size
    assert "DONE" in text
    assert "hello" not in text


def test_fetch_log_tail_parses_remote_size_marker():
    calls = []

    def run_on(node, cmd, **kwargs):
        calls.append((node, cmd, kwargs))
        return 0, "tail text\n___SZ___1234\n", ""

    text, size = fetch_log_tail(
        {"node": "node001", "log_path": "/tmp/task.log"},
        deps=_tail_deps(node_configs={"node001": {"host": "node001"}}, run_on=run_on),
    )

    assert text == "tail text\n"
    assert size == 1234
    assert calls[0][0] == "node001"
    assert "tail -c 4096 /tmp/task.log" in calls[0][1]
    assert calls[0][2] == {"timeout": 10, "check": False}


def test_fetch_log_tail_uses_windows_tail_hook():
    text, size = fetch_log_tail(
        {"node": "win", "log_path": "C:/task.log"},
        deps=_tail_deps(
            node_configs={"win": {"host": "win"}},
            node_is_windows=lambda node: node == "win",
            fetch_windows=lambda task, path, **kwargs: ("win tail", 88),
        ),
    )

    assert (text, size) == ("win tail", 88)


def test_scan_completed_log_for_crash_uses_explicit_patterns_only():
    matched, reason = scan_completed_log_for_crash(
        {"id": "t1"},
        deps=CompletedLogCrashScanDeps(
            fetch_log_tail=lambda task: ("Traceback\nCUDA out of memory\n", 32),
            crash_patterns=["CUDA out of memory"],
        ),
    )

    assert matched is True
    assert "CUDA out of memory" in reason

    matched, reason = scan_completed_log_for_crash(
        {"id": "t2"},
        deps=CompletedLogCrashScanDeps(
            fetch_log_tail=lambda task: ("Epoch 1\nDONE\n", 12),
            crash_patterns=["CUDA out of memory"],
        ),
    )
    assert (matched, reason) == (False, "")


def test_scan_completed_log_does_not_match_oom_inside_headroom():
    matched, reason = scan_completed_log_for_crash(
        {"id": "t3"},
        deps=CompletedLogCrashScanDeps(
            fetch_log_tail=lambda task: ("HEADROOM CONTROLLER COMPLETE\nDONE\n", 34),
            crash_patterns=["OOM"],
        ),
    )

    assert (matched, reason) == (False, "")


def test_classify_failure_priority_and_fallbacks():
    deps = _classify_deps()

    assert classify_failure({}, deps=deps) == "NORMAL"
    assert classify_failure({"is_crash": True, "tail": "ModuleNotFoundError: jax"}, deps=deps) == "PYTHON_IMPORT"
    assert classify_failure({"is_crash": True, "tail": "python: No module named audit"}, deps=deps) == "PYTHON_IMPORT"
    assert classify_failure({"is_crash": True, "tail": "unrecognized arguments: --x CUDA out of memory"}, deps=deps) == "INVALID_FLAG"
    assert classify_failure({"is_crash": True, "tail": "OSError: No space left on device CUDA out of memory"}, deps=deps) == "DISK_FULL"
    assert classify_failure({"is_crash": True, "tail": "CUDA out of memory"}, deps=deps) == "OOM"
    assert classify_failure({"is_crash": True, "tail": "Traceback", "lifetime_s": 61}, deps=deps) == "APP_BUG"
    assert classify_failure({"is_crash": True, "tail": "unknown"}, deps=deps) == "UNKNOWN"


def test_classify_task_artifact_missing_without_poisoning_node_environment():
    deps = _classify_deps()

    assert classify_failure(
        {
            "is_crash": True,
            "tail": (
                "FileNotFoundError: [Errno 2] No such file or directory: "
                "'/work/results/checkpoints/source_snapshot.tar.gz'"
            ),
        },
        deps=deps,
    ) == "ARTIFACT_MISSING"
    assert classify_failure(
        {
            "is_crash": True,
            "tail": "/opt/missing/bin/python: No such file or directory",
        },
        deps=deps,
    ) == "ENV_MISSING"


def test_success_patterns_extend_for_eval_but_not_training():
    assert cmd_looks_like_eval_or_benchmark("python validate_performance.py") is True
    assert cmd_looks_like_eval_or_benchmark(
        "python validate_performance.py",
        cmd_looks_like_training=lambda cmd: True,
    ) is False

    eval_patterns = success_patterns_for_task({"cmd": "python benchmark_foo.py"})
    train_patterns = success_patterns_for_task({"cmd": "python train.py"})

    assert '"summary_rows"' in eval_patterns
    assert '"summary_rows"' not in train_patterns


def test_revision_sumo_evaluator_is_recognized_as_eval_with_terminal_marker():
    cmd = "python eval_revision_metrics.py --tasks_json tasks.json"
    assert cmd_looks_like_eval_or_benchmark(cmd) is True
    assert "[eval_revision_metrics] complete" in success_patterns_for_task({"cmd": cmd})


def test_success_patterns_drop_repeated_subrun_done_for_traffic_benchmark():
    patterns = success_patterns_for_task({"cmd": "python benchmark_traffic_ingolstadt21.py"})

    assert "DONE" not in patterns
    assert '"summary_paths"' in patterns


def test_v7_training_dispatch_report_path_is_a_task_specific_success_marker():
    task = {
        "cmd": "python reproducibility/run_training_dispatch_v7.py --host jtl110gpu",
        "_diagnosis": {
            "tail": "/work/dispatch_execution/core/jtl110gpu/host_execution.json",
        },
    }

    assert "/host_execution.json" in success_patterns_for_task(task)
    assert "/host_execution.json" not in success_patterns_for_task(
        {"cmd": "python train.py"}
    )
    assert cached_task_success_marker(
        task,
        success_patterns_for_task_fn=success_patterns_for_task,
    ) == "/host_execution.json"


def test_cached_task_success_marker_scans_progress_diag_and_tail():
    marker = cached_task_success_marker(
        {
            "last_progress_line": "",
            "_diagnosis": {
                "success_marker": "",
                "tail": "noise\n[Done]\n",
            },
        },
        success_patterns_for_task_fn=success_patterns_for_task,
    )

    assert marker == "[Done]"


def test_cached_task_success_marker_accepts_terminal_model_saved_message():
    marker = cached_task_success_marker(
        {"_diagnosis": {"tail": "Model saved -> outputs/model_final.pt"}},
        success_patterns_for_task_fn=success_patterns_for_task,
    )

    assert marker == "Model saved"


def test_cached_task_success_marker_accepts_pair_protocol_terminal_markers():
    for expected in (
        "PAIR COMPLETE:",
        "PAIR ALREADY COMPLETE:",
        "FINALIZE-ONLY COMPLETE:",
        "RECOVERY COMPLETE:",
    ):
        marker = cached_task_success_marker(
            {
                "last_progress_line": "",
                "_diagnosis": {
                    "success_marker": "",
                    "tail": f"validation passed\n{expected} /tmp/pair\n",
                },
            },
            success_patterns_for_task_fn=success_patterns_for_task,
        )

        assert marker == expected


def test_cached_task_success_marker_accepts_valid_specialist_audit_markers():
    for expected in (
        "INDEPENDENT SPECIALIST AUDIT COMPLETE:",
        "Complete valid specialist audit already exists:",
        "INDEPENDENT SPECIALIST ANALYSIS COMPLETE:",
    ):
        marker = cached_task_success_marker(
            {
                "_diagnosis": {
                    "tail": f"validation passed\n{expected} /results/group\n",
                },
            },
            success_patterns_for_task_fn=success_patterns_for_task,
        )

        assert marker == expected
