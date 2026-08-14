"""Runtime-bound SSH, route, and remote command wrappers."""

from __future__ import annotations

import subprocess as _subprocess
from typing import Any, Mapping, Optional

from scheduler_probe.running_wiring import build_remote_exec_deps as _build_remote_exec_deps
from scheduler_remote.exec import (
    cached_outer_route_failure as _cached_outer_route_failure_impl,
    cleanup_proxy_jump_children as _cleanup_proxy_jump_children_impl,
    configured_proxy_jumps as _configured_proxy_jumps_impl,
    outer_ssh_route_key as _outer_ssh_route_key_impl,
    probe_all_route_failures as _probe_all_route_failures_impl,
    probe_outer_ssh_route as _probe_outer_ssh_route_impl,
    probe_ssh_proxy_jump as _probe_ssh_proxy_jump_impl,
    remote_bash_command_for_node as _remote_bash_command_for_node_impl,
    route_list as _route_list_impl,
    run_on as _run_on_impl,
    run_ssh_subprocess as _run_ssh_subprocess_impl,
    run_windows_ps as _run_windows_ps_impl,
    select_proxy_jump as _select_proxy_jump_impl,
    ssh_arg_value as _ssh_arg_value_impl,
    ssh_base_arg_variants as _ssh_base_arg_variants_impl,
    ssh_base_args as _ssh_base_args_impl,
    ssh_base_args_with_proxy as _ssh_base_args_with_proxy_impl,
    ssh_no_stdin_arg_variants as _ssh_no_stdin_arg_variants_impl,
    ssh_no_stdin_args as _ssh_no_stdin_args_impl,
    ssh_rsync_shell_for_node as _ssh_rsync_shell_for_node_impl,
    ssh_target_for_node as _ssh_target_for_node_impl,
    ssh_target_from_args as _ssh_target_from_args_impl,
)


_ORIGINAL_SUBPROCESS_RUN = _subprocess.run


def _ns(namespace: Mapping[str, Any], name: str) -> Any:
    return namespace[name]


def build_transport_remote_runtime_exports(namespace: Mapping[str, Any]) -> dict[str, Any]:
    def _remote_exec_deps():
        return _build_remote_exec_deps(namespace)

    def _route_list(value) -> list:
        return _route_list_impl(value)

    def _configured_proxy_jumps(info: dict) -> list:
        return _configured_proxy_jumps_impl(info)

    def _run_ssh_subprocess(
        args: list,
        *,
        timeout,
        input=None,
        capture_output=False,
        stdout=None,
        stderr=None,
        text=None,
    ):
        return _run_ssh_subprocess_impl(
            args,
            timeout=timeout,
            input=input,
            capture_output=capture_output,
            stdout=stdout,
            stderr=stderr,
            text=text,
            deps=_ns(namespace, "_remote_exec_deps")(),
        )

    def _run_control_subprocess(
        args: list,
        *,
        timeout,
        input=None,
        capture_output=False,
        stdout=None,
        stderr=None,
        text=None,
        **kwargs,
    ):
        run = _ns(namespace, "subprocess").run
        if run is not _ORIGINAL_SUBPROCESS_RUN:
            return run(
                args,
                timeout=timeout,
                input=input,
                capture_output=capture_output,
                stdout=stdout,
                stderr=stderr,
                text=text,
                **kwargs,
            )
        if kwargs:
            return run(
                args,
                timeout=timeout,
                input=input,
                capture_output=capture_output,
                stdout=stdout,
                stderr=stderr,
                text=text,
                **kwargs,
            )
        return _run_ssh_subprocess(
            args,
            timeout=timeout,
            input=input,
            capture_output=capture_output,
            stdout=stdout,
            stderr=stderr,
            text=text,
        )

    def _ssh_arg_value(args: list, opt: str) -> Optional[str]:
        return _ssh_arg_value_impl(args, opt)

    def _ssh_target_from_args(args: list) -> tuple[str, str]:
        return _ssh_target_from_args_impl(args)

    def _cleanup_proxy_jump_children(args: list) -> None:
        return _cleanup_proxy_jump_children_impl(args, deps=_ns(namespace, "_remote_exec_deps")())

    def _outer_ssh_route_key(node: str):
        return _outer_ssh_route_key_impl(node, deps=_ns(namespace, "_remote_exec_deps")())

    def _probe_outer_ssh_route(node: str, timeout: Optional[int] = None) -> tuple[bool, str]:
        return _probe_outer_ssh_route_impl(
            node,
            timeout=timeout,
            deps=_ns(namespace, "_remote_exec_deps")(),
        )

    def _cached_outer_route_failure(node: str) -> str:
        return _cached_outer_route_failure_impl(node, deps=_ns(namespace, "_remote_exec_deps")())

    def _probe_all_route_failures() -> dict:
        return _probe_all_route_failures_impl(deps=_ns(namespace, "_remote_exec_deps")())

    def _ssh_base_args_with_proxy(node: str, proxy_jump: Optional[str] = None) -> list:
        return _ssh_base_args_with_proxy_impl(
            node,
            proxy_jump,
            deps=_ns(namespace, "_remote_exec_deps")(),
        )

    def _probe_ssh_proxy_jump(node: str, proxy_jump: str, timeout: int = 8) -> bool:
        return _probe_ssh_proxy_jump_impl(
            node,
            proxy_jump,
            timeout=timeout,
            deps=_ns(namespace, "_remote_exec_deps")(),
        )

    def _select_proxy_jump(node: str) -> Optional[str]:
        return _select_proxy_jump_impl(node, deps=_ns(namespace, "_remote_exec_deps")())

    def _ssh_base_arg_variants(node: str) -> list:
        return _ssh_base_arg_variants_impl(node, deps=_ns(namespace, "_remote_exec_deps")())

    def _ssh_base_args(node: str) -> list:
        return _ssh_base_args_impl(node, deps=_ns(namespace, "_remote_exec_deps")())

    def _ssh_no_stdin_args(node: str) -> list:
        return _ssh_no_stdin_args_impl(node, deps=_ns(namespace, "_remote_exec_deps")())

    def _ssh_no_stdin_arg_variants(node: str) -> list:
        return _ssh_no_stdin_arg_variants_impl(node, deps=_ns(namespace, "_remote_exec_deps")())

    def _run_windows_ps(
        node: str,
        ps_script: str,
        timeout=15,
        check=True,
        input_data: bytes | None = None,
    ):
        return _run_windows_ps_impl(
            node,
            ps_script,
            timeout=timeout,
            check=check,
            input_data=input_data,
            deps=_ns(namespace, "_remote_exec_deps")(),
        )

    def run_on(node, shell_cmd, timeout=15, check=True):
        return _run_on_impl(
            node,
            shell_cmd,
            timeout=timeout,
            check=check,
            deps=_ns(namespace, "_remote_exec_deps")(),
        )

    def _remote_bash_command_for_node(
        node: str,
        shell_cmd: str,
        timeout: Optional[int] = None,
    ) -> str:
        return _remote_bash_command_for_node_impl(
            node,
            shell_cmd,
            timeout=timeout,
            deps=_ns(namespace, "_remote_exec_deps")(),
        )

    def _ssh_rsync_shell_for_node(node: str) -> str:
        return _ssh_rsync_shell_for_node_impl(node, deps=_ns(namespace, "_remote_exec_deps")())

    def _ssh_target_for_node(node: str) -> str:
        return _ssh_target_for_node_impl(node, deps=_ns(namespace, "_remote_exec_deps")())

    return {
        "_remote_exec_deps": _remote_exec_deps,
        "_route_list": _route_list,
        "_configured_proxy_jumps": _configured_proxy_jumps,
        "_run_ssh_subprocess": _run_ssh_subprocess,
        "_run_control_subprocess": _run_control_subprocess,
        "_ssh_arg_value": _ssh_arg_value,
        "_ssh_target_from_args": _ssh_target_from_args,
        "_cleanup_proxy_jump_children": _cleanup_proxy_jump_children,
        "_outer_ssh_route_key": _outer_ssh_route_key,
        "_probe_outer_ssh_route": _probe_outer_ssh_route,
        "_cached_outer_route_failure": _cached_outer_route_failure,
        "_probe_all_route_failures": _probe_all_route_failures,
        "_ssh_base_args_with_proxy": _ssh_base_args_with_proxy,
        "_probe_ssh_proxy_jump": _probe_ssh_proxy_jump,
        "_select_proxy_jump": _select_proxy_jump,
        "_ssh_base_arg_variants": _ssh_base_arg_variants,
        "_ssh_base_args": _ssh_base_args,
        "_ssh_no_stdin_args": _ssh_no_stdin_args,
        "_ssh_no_stdin_arg_variants": _ssh_no_stdin_arg_variants,
        "_run_windows_ps": _run_windows_ps,
        "run_on": run_on,
        "_remote_bash_command_for_node": _remote_bash_command_for_node,
        "_ssh_rsync_shell_for_node": _ssh_rsync_shell_for_node,
        "_ssh_target_for_node": _ssh_target_for_node,
    }
