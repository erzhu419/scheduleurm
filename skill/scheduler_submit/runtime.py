from __future__ import annotations

from typing import Any

from scheduler_diagnostics.doctor_runtime import build_doctor_runtime_exports
from .command_runtime import build_submit_command_runtime_exports
from .helpers_runtime import build_submit_helpers_runtime_exports


def build_submit_runtime_exports(namespace: dict[str, Any]) -> dict[str, Any]:
    exports: dict[str, Any] = {}
    for builder in (
        build_submit_helpers_runtime_exports,
        build_doctor_runtime_exports,
        build_submit_command_runtime_exports,
    ):
        new_exports = builder(namespace)
        exports.update(new_exports)
        namespace.update(new_exports)
    return exports
