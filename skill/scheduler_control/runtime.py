from __future__ import annotations

import time
from typing import Any, Mapping, Optional


def _ns(namespace: Mapping[str, Any], name: str) -> Any:
    return namespace[name]


def build_control_runtime_exports(namespace: dict[str, Any]) -> dict[str, Any]:
    def _format_task_vram_usage(task):
        return _ns(namespace, "_format_task_vram_usage_impl")(task)

    def _format_task_ram_usage(task):
        return _ns(namespace, "_format_task_ram_usage_impl")(task)

    def _task_result_artifacts_for_display(task: dict, include_log: bool = True) -> list:
        return _ns(namespace, "_task_result_artifacts_for_display_impl")(
            task,
            include_log=include_log,
            deps=_ns(namespace, "_result_artifact_deps")(),
        )

    def _print_result_artifacts(task: dict, include_log: bool = True):
        return _ns(namespace, "_print_result_artifacts_impl")(
            task,
            include_log=include_log,
            deps=_ns(namespace, "_result_artifact_deps")(),
            remote_path_for_node=_ns(namespace, "_remote_path_for_node"),
        )

    def _compact_status_task(task: dict) -> dict:
        return _ns(namespace, "_compact_status_task_impl")(task)

    def _status_command_deps():
        return _ns(namespace, "_build_status_command_deps")(namespace)

    def cmd_status(args):
        return _ns(namespace, "_cmd_status_impl")(
            args,
            deps=_ns(namespace, "_status_command_deps")(),
        )

    def _compact_age_days_from_args(args) -> float:
        return _ns(namespace, "_compact_age_days_from_args_impl")(
            args,
            default_age_days=_ns(namespace, "ARCHIVE_AGE_DAYS"),
        )

    def _terminal_archive_stats(
        state: dict,
        age_days: float,
        max_hot_terminal: Optional[int] = None,
        extra_archive_predicate=None,
    ) -> dict:
        if max_hot_terminal is None:
            max_hot_terminal = _ns(namespace, "ARCHIVE_MAX_HOT_TERMINAL")
        if extra_archive_predicate is None:
            extra_archive_predicate = _ns(namespace, "_archive_control_plane_autoadopted_terminal")
        return _ns(namespace, "_terminal_archive_stats_impl")(
            state,
            age_days,
            max_hot_terminal=max_hot_terminal,
            extra_archive_predicate=extra_archive_predicate,
            now_fn=time.time,
        )

    def _control_command_deps():
        return _ns(namespace, "_ControlCommandDeps")(
            state_lock=_ns(namespace, "state_lock"),
            load_state=_ns(namespace, "load_state"),
            save_state=_ns(namespace, "save_state"),
            lock_timeout_error=_ns(namespace, "SchedulerLockTimeout"),
            queue_file=_ns(namespace, "QUEUE_FILE"),
            archive_file=_ns(namespace, "ARCHIVE_FILE"),
            archive_max_hot_terminal=_ns(namespace, "ARCHIVE_MAX_HOT_TERMINAL"),
            compact_age_days_from_args=_ns(namespace, "_compact_age_days_from_args"),
            terminal_archive_stats=_ns(namespace, "_terminal_archive_stats"),
            archive_terminal_tasks=_ns(namespace, "archive_terminal_tasks"),
            node_configs=_ns(namespace, "NODES"),
            claim_enabled_for=_ns(namespace, "_ClaimManager").enabled_for,
            claim_snapshot=_ns(namespace, "_ClaimManager").snapshot,
            format_claim_record=_ns(namespace, "_format_claim_record"),
            find_task_record=_ns(namespace, "_find_task_record"),
            tail_task_log=_ns(namespace, "_tail_task_log"),
            node_is_windows=_ns(namespace, "_node_is_windows"),
            ssh_base_args=_ns(namespace, "_ssh_base_args"),
            print_result_artifacts=_ns(namespace, "_print_result_artifacts"),
        )

    def cmd_compact(args):
        return _ns(namespace, "_cmd_compact_impl")(
            args,
            deps=_ns(namespace, "_control_command_deps")(),
        )

    def cmd_claims(args):
        return _ns(namespace, "_cmd_claims_impl")(
            args,
            deps=_ns(namespace, "_control_command_deps")(),
        )

    def _tail_task_log(task: dict, lines: int = 80) -> tuple[bool, str, str]:
        return _ns(namespace, "_tail_task_log_impl")(
            task,
            lines=lines,
            nodes=_ns(namespace, "NODES"),
            node_is_windows=_ns(namespace, "_node_is_windows"),
            windows_path_for_task=_ns(namespace, "_windows_path_for_task"),
            ps_quote=_ns(namespace, "_ps_quote"),
            run_windows_ps=_ns(namespace, "_run_windows_ps"),
            run_on=_ns(namespace, "run_on"),
        )

    def cmd_task_log(args):
        return _ns(namespace, "_cmd_task_log_impl")(
            args,
            deps=_ns(namespace, "_control_command_deps")(),
        )

    def cmd_show(args):
        return _ns(namespace, "_cmd_show_impl")(
            args,
            deps=_ns(namespace, "_control_command_deps")(),
        )

    def cmd_results(args):
        return _ns(namespace, "_cmd_results_impl")(
            args,
            deps=_ns(namespace, "_ResultsCommandDeps")(
                state_lock=_ns(namespace, "state_lock"),
                load_state=_ns(namespace, "load_state"),
                load_archive_tasks=_ns(namespace, "load_archive_tasks"),
                task_result_artifacts_for_display=_ns(
                    namespace, "_task_result_artifacts_for_display"
                ),
            ),
        )

    def _task_control_deps():
        return _ns(namespace, "_build_task_control_deps")(namespace)

    def cmd_cancel(args):
        return _ns(namespace, "_cmd_cancel_impl")(
            args,
            deps=_ns(namespace, "_task_control_deps")(),
        )

    def cmd_forget(args):
        return _ns(namespace, "_cmd_forget_impl")(
            args,
            deps=_ns(namespace, "_task_control_deps")(),
        )

    def cmd_clear_queue(args):
        return _ns(namespace, "_cmd_clear_queue_impl")(
            args,
            deps=_ns(namespace, "_task_control_deps")(),
        )

    def _adopt_command_deps():
        return _ns(namespace, "_build_adopt_command_deps")(namespace)

    def cmd_adopt(args):
        return _ns(namespace, "_cmd_adopt_impl")(
            args,
            deps=_ns(namespace, "_adopt_command_deps")(),
        )

    def cmd_record_vram(args):
        with _ns(namespace, "state_lock")():
            _ns(namespace, "history_record")(args.signature, peak_vram_mb=args.peak_vram_mb)
        after = _ns(namespace, "history_get")(args.signature) or {}
        print(
            f"recorded {args.signature} -> vram={after.get('vram_mb', 0)}MB "
            f"ram={after.get('ram_mb', 0)}MB cpu={after.get('cpu_cores', 0)}"
        )

    def cmd_tui(args):
        from tui import main as tui_main
        tui_main()

    def cmd_priority(args):
        return _ns(namespace, "_cmd_priority_impl")(
            args,
            deps=_ns(namespace, "_control_command_deps")(),
        )

    def _edit_command_deps():
        return _ns(namespace, "_build_edit_command_deps")(namespace)

    def cmd_edit(args):
        return _ns(namespace, "_cmd_edit_impl")(
            args,
            deps=_ns(namespace, "_edit_command_deps")(),
        )

    def _why_command_deps():
        return _ns(namespace, "_build_why_command_deps")(namespace)

    def _explain_node_fit(task: dict, node_state: dict) -> str:
        return _ns(namespace, "_explain_node_fit_impl")(
            task,
            node_state,
            deps=_ns(namespace, "_why_command_deps")(),
        )

    def cmd_why(args):
        return _ns(namespace, "_cmd_why_impl")(
            args,
            deps=_ns(namespace, "_why_command_deps")(),
        )

    def _history_command_deps():
        return _ns(namespace, "_build_history_command_deps")(namespace)

    def cmd_history(args):
        return _ns(namespace, "_cmd_history_impl")(
            args,
            deps=_ns(namespace, "_history_command_deps")(),
        )

    return {
        "_format_task_vram_usage": _format_task_vram_usage,
        "_format_task_ram_usage": _format_task_ram_usage,
        "_task_result_artifacts_for_display": _task_result_artifacts_for_display,
        "_print_result_artifacts": _print_result_artifacts,
        "_compact_status_task": _compact_status_task,
        "_status_command_deps": _status_command_deps,
        "cmd_status": cmd_status,
        "_compact_age_days_from_args": _compact_age_days_from_args,
        "_terminal_archive_stats": _terminal_archive_stats,
        "_control_command_deps": _control_command_deps,
        "cmd_compact": cmd_compact,
        "cmd_claims": cmd_claims,
        "_tail_task_log": _tail_task_log,
        "cmd_task_log": cmd_task_log,
        "cmd_show": cmd_show,
        "cmd_results": cmd_results,
        "_task_control_deps": _task_control_deps,
        "cmd_cancel": cmd_cancel,
        "cmd_forget": cmd_forget,
        "cmd_clear_queue": cmd_clear_queue,
        "_adopt_command_deps": _adopt_command_deps,
        "cmd_adopt": cmd_adopt,
        "cmd_record_vram": cmd_record_vram,
        "cmd_tui": cmd_tui,
        "cmd_priority": cmd_priority,
        "_edit_command_deps": _edit_command_deps,
        "cmd_edit": cmd_edit,
        "_why_command_deps": _why_command_deps,
        "_explain_node_fit": _explain_node_fit,
        "cmd_why": cmd_why,
        "_history_command_deps": _history_command_deps,
        "cmd_history": cmd_history,
    }
