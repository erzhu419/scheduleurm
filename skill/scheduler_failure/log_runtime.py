"""Runtime wrappers for full-log success scanning."""

from __future__ import annotations

from typing import Any, Mapping


def _ns(namespace: Mapping[str, Any], name: str) -> Any:
    return namespace[name]


def build_failure_log_runtime_exports(namespace: dict[str, Any]) -> dict[str, Any]:
    def _cmd_looks_like_eval_or_benchmark(cmd: str) -> bool:
        return _ns(namespace, "_cmd_looks_like_eval_or_benchmark_impl")(
            cmd,
            cmd_looks_like_training=_ns(namespace, "_cmd_looks_like_training"),
        )

    def _cmd_has_repeated_subrun_done(cmd: str) -> bool:
        return _ns(namespace, "_cmd_has_repeated_subrun_done_impl")(cmd)

    def _success_patterns_for_task(task: dict) -> list[str]:
        return _ns(namespace, "_success_patterns_for_task_impl")(
            task,
            cmd_looks_like_training=_ns(namespace, "_cmd_looks_like_training"),
        )

    def _cached_task_success_marker(task: dict) -> str:
        return _ns(namespace, "_cached_task_success_marker_impl")(
            task,
            success_patterns_for_task_fn=_ns(namespace, "_success_patterns_for_task"),
        )

    def _fetch_log_tail(task) -> tuple[str, int]:
        return _ns(namespace, "_fetch_log_tail_impl")(
            task,
            deps=_ns(namespace, "_LogTailDeps")(
                node_configs=_ns(namespace, "NODES"),
                node_is_windows=_ns(namespace, "_node_is_windows"),
                fetch_windows_text_tail=_ns(namespace, "_fetch_windows_text_tail"),
                run_on=_ns(namespace, "run_on"),
            ),
        )

    def _scan_full_log_for_success(task) -> list:
        return _ns(namespace, "_scan_full_log_for_success_impl")(
            task,
            deps=_ns(namespace, "_FullLogSuccessScanDeps")(
                node_configs=_ns(namespace, "NODES"),
                node_is_windows=_ns(namespace, "_node_is_windows"),
                scan_windows_log_for_patterns=_ns(namespace, "_scan_windows_log_for_patterns"),
                success_patterns_for_task=_ns(namespace, "_success_patterns_for_task"),
                run_on=_ns(namespace, "run_on"),
            ),
        )

    def _scan_completed_log_for_crash(task) -> tuple[bool, str]:
        return _ns(namespace, "_scan_completed_log_for_crash_impl")(
            task,
            deps=_ns(namespace, "_CompletedLogCrashScanDeps")(
                fetch_log_tail=_ns(namespace, "_fetch_log_tail"),
                crash_patterns=_ns(namespace, "CRASH_PATTERNS"),
            ),
        )

    return {
        "_cmd_looks_like_eval_or_benchmark": _cmd_looks_like_eval_or_benchmark,
        "_cmd_has_repeated_subrun_done": _cmd_has_repeated_subrun_done,
        "_success_patterns_for_task": _success_patterns_for_task,
        "_cached_task_success_marker": _cached_task_success_marker,
        "_fetch_log_tail": _fetch_log_tail,
        "_scan_full_log_for_success": _scan_full_log_for_success,
        "_scan_completed_log_for_crash": _scan_completed_log_for_crash,
    }
