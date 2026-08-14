"""Runtime-bound ETA compatibility exports."""

from __future__ import annotations

from typing import Any

from .analysis_runtime import build_eta_analysis_runtime_exports
from .fields_runtime import build_eta_fields_runtime_exports
from .probe_runtime import build_eta_probe_runtime_exports


def build_eta_runtime_exports(namespace: dict[str, Any]) -> dict[str, Any]:
    exports: dict[str, Any] = {}
    for builder in (
        build_eta_fields_runtime_exports,
        build_eta_analysis_runtime_exports,
        build_eta_probe_runtime_exports,
    ):
        new_exports = builder(namespace)
        exports.update(new_exports)
        namespace.update(new_exports)
    return exports
