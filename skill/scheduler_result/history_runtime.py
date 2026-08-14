"""Compatibility aggregator for archive/result/history runtime exports."""

from __future__ import annotations

from typing import Any

from scheduler_state.archive_runtime import build_archive_runtime_exports
from scheduler_log_runtime import build_log_runtime_exports
from scheduler_resource.history_runtime import build_resource_history_runtime_exports
from .artifact_runtime import build_result_artifact_runtime_exports


def build_results_history_runtime_exports(namespace: dict[str, Any]) -> dict[str, Any]:
    exports: dict[str, Any] = {}
    exports.update(build_archive_runtime_exports(namespace))
    exports.update(build_result_artifact_runtime_exports(namespace))
    exports.update(build_log_runtime_exports(namespace))
    exports.update(build_resource_history_runtime_exports(namespace))
    return exports
