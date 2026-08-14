"""Runtime-bound node OS and path mapping wrappers."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from scheduler_transport.path_mapping import (
    WindowsPrepareCommandDeps,
    ps_quote as _ps_quote_impl,
    remote_path_for_node as _remote_path_for_node_impl,
    rewrite_command_paths_for_node as _rewrite_command_paths_for_node_impl,
    windows_env_spec_error as _windows_env_spec_error_impl,
    windows_path_for_node as _windows_path_for_node_impl,
    windows_prepare_command as _windows_prepare_command_impl,
    windows_rewrite_token as _windows_rewrite_token_impl,
)


def _ns(namespace: Mapping[str, Any], name: str) -> Any:
    return namespace[name]


def build_transport_path_runtime_exports(namespace: Mapping[str, Any]) -> dict[str, Any]:
    def _node_is_windows(node: str) -> bool:
        node = _ns(namespace, "_canonical_node_name")(node)
        return str((_ns(namespace, "NODES").get(node) or {}).get("os") or "").lower().startswith("win")

    def _ps_quote(s: str) -> str:
        return _ps_quote_impl(s)

    def _windows_path_for_node(node: str, path: str) -> str:
        return _windows_path_for_node_impl(
            node,
            path,
            node_configs=_ns(namespace, "NODES"),
            node_is_windows=_ns(namespace, "_node_is_windows"),
        )

    def _remote_path_for_node(node: str, path: str) -> str:
        return _remote_path_for_node_impl(
            node,
            path,
            node_configs=_ns(namespace, "NODES"),
            canonical_node_name=_ns(namespace, "_canonical_node_name"),
            node_is_windows=_ns(namespace, "_node_is_windows"),
            windows_path_mapper=_ns(namespace, "_windows_path_for_node"),
        )

    def _rewrite_command_paths_for_node(node: str, cmd: str) -> str:
        return _rewrite_command_paths_for_node_impl(
            node,
            cmd,
            node_configs=_ns(namespace, "NODES"),
            node_is_windows=_ns(namespace, "_node_is_windows"),
            remote_path_mapper=_ns(namespace, "_remote_path_for_node"),
        )

    def _windows_rewrite_token(node: str, token: str) -> str:
        return _windows_rewrite_token_impl(
            node,
            token,
            node_configs=_ns(namespace, "NODES"),
            windows_path_mapper=_ns(namespace, "_windows_path_for_node"),
        )

    def _windows_prepare_command(task: dict) -> dict:
        return _windows_prepare_command_impl(
            task,
            deps=WindowsPrepareCommandDeps(
                node_configs=_ns(namespace, "NODES"),
                inject_python_u=_ns(namespace, "_inject_python_u"),
                task_launch_cpu_mode=_ns(namespace, "_task_launch_cpu_mode"),
                windows_path_for_node=_ns(namespace, "_windows_path_for_node"),
                windows_rewrite_token=_ns(namespace, "_windows_rewrite_token"),
            ),
        )

    def _windows_env_spec_error(task: dict) -> Optional[str]:
        return _windows_env_spec_error_impl(task)

    return {
        "_node_is_windows": _node_is_windows,
        "_ps_quote": _ps_quote,
        "_windows_path_for_node": _windows_path_for_node,
        "_remote_path_for_node": _remote_path_for_node,
        "_rewrite_command_paths_for_node": _rewrite_command_paths_for_node,
        "_windows_rewrite_token": _windows_rewrite_token,
        "_windows_prepare_command": _windows_prepare_command,
        "_windows_env_spec_error": _windows_env_spec_error,
    }
