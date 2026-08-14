"""Runtime-bound result artifact wrappers for scheduler.py."""

from __future__ import annotations

import os
from typing import Any, Mapping

from .discovery import (
    add_result_artifact as _add_result_artifact_impl,
    collect_terminal_evidence as _collect_terminal_evidence_impl,
    fetch_log_result_lines as _fetch_log_result_lines_impl,
    result_artifacts_from_log as _result_artifacts_from_log_impl,
    terminal_result_artifact_success as _terminal_result_artifact_success_impl,
)


def _ns(namespace: Mapping[str, Any], name: str) -> Any:
    return namespace[name]


def build_result_artifact_runtime_exports(namespace: dict[str, Any]) -> dict[str, Any]:
    def _clean_result_path(raw: str) -> str:
        return _ns(namespace, "_clean_result_path_impl")(raw)

    def _resolve_task_path(task: dict, raw_path: str) -> str:
        return _ns(namespace, "_resolve_task_path_impl")(task, raw_path)

    def _result_kind_for_path(path: str, preferred: str = "") -> str:
        return _ns(namespace, "_result_kind_for_path_impl")(path, preferred)

    def _add_result_artifact(
        artifacts: list,
        task: dict,
        raw_path: str,
        kind: str = "",
        source: str = "",
    ):
        return _add_result_artifact_impl(artifacts, task, raw_path, kind, source)

    def _cmd_flag_values(tokens: list, flags: set) -> dict:
        return _ns(namespace, "_cmd_flag_values_impl")(tokens, flags)

    def _cmd_token_variants(cmd: str) -> list:
        return _ns(namespace, "_cmd_token_variants_impl")(cmd)

    def _cmd_flag_values_from_cmd(cmd: str, flags: set) -> dict:
        return _ns(namespace, "_cmd_flag_values_from_cmd_impl")(cmd, flags)

    def _result_artifacts_from_cmd(task: dict) -> list:
        return _ns(namespace, "_result_artifacts_from_cmd_impl")(task)

    def _save_root_run_name_pairs_from_cmd(task: dict) -> list:
        return _ns(namespace, "_save_root_run_name_pairs_from_cmd_impl")(task)

    def _bash_path_arg(path: str) -> str:
        return _ns(namespace, "_bash_path_arg_impl")(path)

    def _final_model_file_groups_from_cmd(task: dict) -> list:
        return _ns(namespace, "_final_model_file_groups_from_cmd_impl")(task)

    def _mtimes_match_task_window(task: dict, mtimes: list) -> bool:
        return _ns(namespace, "_mtimes_match_task_window_impl")(task, mtimes)

    def _terminal_final_model_success(task: dict) -> str:
        return _ns(namespace, "_terminal_final_model_success_impl")(
            task,
            deps=_ns(namespace, "_FinalModelSuccessDeps")(
                node_configs=_ns(namespace, "NODES"),
                final_model_file_groups_from_cmd=_ns(namespace, "_final_model_file_groups_from_cmd"),
                mtimes_match_task_window=_ns(namespace, "_mtimes_match_task_window"),
                node_is_windows=_ns(namespace, "_node_is_windows"),
                windows_path_for_task=_ns(namespace, "_windows_path_for_task"),
                ps_quote=_ns(namespace, "_ps_quote"),
                run_windows_ps=_ns(namespace, "_run_windows_ps"),
                bash_path_arg=_ns(namespace, "_bash_path_arg"),
                run_on=_ns(namespace, "run_on"),
                expanduser=os.path.expanduser,
            ),
        )

    def _result_artifact_deps():
        return _ns(namespace, "_build_result_artifact_deps")(namespace)

    def _fetch_log_result_lines(task: dict, max_lines: int = 50) -> list:
        return _fetch_log_result_lines_impl(
            task,
            max_lines=max_lines,
            deps=_ns(namespace, "_result_artifact_deps")(),
        )

    def _result_artifacts_from_log(task: dict) -> list:
        return _result_artifacts_from_log_impl(
            task,
            deps=_ns(namespace, "_result_artifact_deps")(),
        )

    def _discover_result_artifacts(task: dict, include_log: bool = True) -> list:
        return _ns(namespace, "_discover_result_artifacts_impl")(
            task,
            include_log=include_log,
            deps=_ns(namespace, "_result_artifact_deps")(),
        )

    def _record_result_artifacts(task: dict) -> list:
        return _ns(namespace, "_record_result_artifacts_impl")(
            task,
            deps=_ns(namespace, "_result_artifact_deps")(),
        )

    def _apply_discovered_result_artifacts(task: dict, discovered: list) -> list:
        return _ns(namespace, "_apply_discovered_result_artifacts_impl")(
            task,
            discovered,
            deps=_ns(namespace, "_result_artifact_deps")(),
        )

    def _terminal_result_artifact_success(task: dict) -> str:
        return _terminal_result_artifact_success_impl(
            task,
            deps=_ns(namespace, "_result_artifact_deps")(),
        )

    def _collect_terminal_evidence(
        tasks: list[dict],
        *,
        timeout_s: float = 0.0,
    ) -> dict:
        return _collect_terminal_evidence_impl(
            tasks,
            deps=_ns(namespace, "_result_artifact_deps")(),
            max_workers=_ns(namespace, "TERMINAL_EVIDENCE_MAX_WORKERS"),
            batch_size=_ns(namespace, "TERMINAL_EVIDENCE_BATCH_SIZE"),
            timeout_s=timeout_s,
        )

    return {
        "_clean_result_path": _clean_result_path,
        "_resolve_task_path": _resolve_task_path,
        "_result_kind_for_path": _result_kind_for_path,
        "_add_result_artifact": _add_result_artifact,
        "_cmd_flag_values": _cmd_flag_values,
        "_cmd_token_variants": _cmd_token_variants,
        "_cmd_flag_values_from_cmd": _cmd_flag_values_from_cmd,
        "_result_artifacts_from_cmd": _result_artifacts_from_cmd,
        "_save_root_run_name_pairs_from_cmd": _save_root_run_name_pairs_from_cmd,
        "_bash_path_arg": _bash_path_arg,
        "_final_model_file_groups_from_cmd": _final_model_file_groups_from_cmd,
        "_mtimes_match_task_window": _mtimes_match_task_window,
        "_terminal_final_model_success": _terminal_final_model_success,
        "_result_artifact_deps": _result_artifact_deps,
        "_fetch_log_result_lines": _fetch_log_result_lines,
        "_result_artifacts_from_log": _result_artifacts_from_log,
        "_discover_result_artifacts": _discover_result_artifacts,
        "_record_result_artifacts": _record_result_artifacts,
        "_apply_discovered_result_artifacts": _apply_discovered_result_artifacts,
        "_terminal_result_artifact_success": _terminal_result_artifact_success,
        "_collect_terminal_evidence": _collect_terminal_evidence,
    }
