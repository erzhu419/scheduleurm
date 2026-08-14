"""Runtime-bound failure compatibility exports."""

from __future__ import annotations

from typing import Any

from .classification_runtime import build_failure_classification_runtime_exports
from .crash_runtime import build_failure_crash_runtime_exports
from .heal_runtime import build_failure_heal_runtime_exports
from .log_runtime import build_failure_log_runtime_exports
from .terminal_runtime import build_failure_terminal_runtime_exports


def build_failure_runtime_exports(namespace: dict[str, Any]) -> dict[str, Any]:
    exports: dict[str, Any] = {}
    for builder in (
        build_failure_log_runtime_exports,
        build_failure_terminal_runtime_exports,
        build_failure_classification_runtime_exports,
        build_failure_heal_runtime_exports,
        build_failure_crash_runtime_exports,
    ):
        new_exports = builder(namespace)
        exports.update(new_exports)
        namespace.update(new_exports)
    return exports
