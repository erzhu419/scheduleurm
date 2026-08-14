"""Runtime-bound Windows PowerShell/logging wrappers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from scheduler_windows.logs import (
    WindowsLogDeps,
    WindowsTaskPathDeps,
    fetch_windows_text_tail as _fetch_windows_text_tail_impl,
    is_windows_native_path as _is_windows_native_path_impl,
    scan_windows_log_for_patterns as _scan_windows_log_for_patterns_impl,
    windows_path_for_task as _windows_path_for_task_impl,
    windows_tail_ps as _windows_tail_ps_impl,
)


def _ns(namespace: Mapping[str, Any], name: str) -> Any:
    return namespace[name]


def build_transport_windows_runtime_exports(namespace: Mapping[str, Any]) -> dict[str, Any]:
    def _windows_probe_error_hint(node: str, msg: str) -> str:
        text = (msg or "").strip()
        if "Permission denied" in text or "publickey,password" in text:
            ident = (_ns(namespace, "NODES").get(node, {}) or {}).get("ssh_identity") or "~/.ssh/id_ed25519"
            return (
                f"{text[:180]} | SSH key auth failed for Windows node {node}; "
                f"install {_ns(namespace, 'os').path.expanduser(str(ident))}.pub on the host. "
                "Do not store the password in scheduler.py."
            )
        if "timed out" in text.lower() or "TimeoutExpired" in text:
            return (
                f"{text[:180]} | Windows probe timed out; check SSH reachability, "
                "PowerShell startup, and host CPU pressure."
            )
        return text[:300]

    def _windows_tail_ps(path: str, max_bytes: int = 4096) -> str:
        return _windows_tail_ps_impl(path, max_bytes=max_bytes, ps_quote=_ns(namespace, "_ps_quote"))

    def _windows_path_for_task(task: dict, path: str) -> str:
        return _windows_path_for_task_impl(
            task,
            path,
            deps=WindowsTaskPathDeps(
                node_is_windows=_ns(namespace, "_node_is_windows"),
                windows_path_for_node=_ns(namespace, "_windows_path_for_node"),
                home=Path.home,
            ),
        )

    def _is_windows_native_path(path: str) -> bool:
        return _is_windows_native_path_impl(path)

    def _fetch_windows_text_tail(task: dict, path: str, max_bytes: int = 4096) -> tuple[str, int]:
        return _fetch_windows_text_tail_impl(
            task,
            path,
            max_bytes=max_bytes,
            deps=WindowsLogDeps(
                windows_path_for_task=_ns(namespace, "_windows_path_for_task"),
                windows_tail_ps=_ns(namespace, "_windows_tail_ps"),
                ps_quote=_ns(namespace, "_ps_quote"),
                run_windows_ps=_ns(namespace, "_run_windows_ps"),
            ),
        )

    def _scan_windows_log_for_patterns(task: dict, patterns: list[str]) -> list[str]:
        return _scan_windows_log_for_patterns_impl(
            task,
            patterns,
            deps=WindowsLogDeps(
                windows_path_for_task=_ns(namespace, "_windows_path_for_task"),
                windows_tail_ps=_ns(namespace, "_windows_tail_ps"),
                ps_quote=_ns(namespace, "_ps_quote"),
                run_windows_ps=_ns(namespace, "_run_windows_ps"),
            ),
        )

    return {
        "_windows_probe_error_hint": _windows_probe_error_hint,
        "_windows_tail_ps": _windows_tail_ps,
        "_windows_path_for_task": _windows_path_for_task,
        "_is_windows_native_path": _is_windows_native_path,
        "_fetch_windows_text_tail": _fetch_windows_text_tail,
        "_scan_windows_log_for_patterns": _scan_windows_log_for_patterns,
    }
