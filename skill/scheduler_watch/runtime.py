from __future__ import annotations

from typing import Any

from scheduler_control.runtime import build_control_runtime_exports
from scheduler_dispatch.command_runtime import build_dispatch_command_runtime_exports
from scheduler_external.runtime import build_external_runtime_exports
from scheduler_notification.runtime import build_notify_runtime_exports
from .loop_runtime import build_watch_loop_runtime_exports


def build_watch_runtime_exports(namespace: dict[str, Any]) -> dict[str, Any]:
    exports: dict[str, Any] = {}
    for builder in (
        build_dispatch_command_runtime_exports,
        build_notify_runtime_exports,
        build_external_runtime_exports,
        build_watch_loop_runtime_exports,
        build_control_runtime_exports,
    ):
        new_exports = builder(namespace)
        exports.update(new_exports)
        namespace.update(new_exports)
    return exports
