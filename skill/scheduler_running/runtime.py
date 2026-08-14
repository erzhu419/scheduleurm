"""Runtime-bound running-task compatibility exports."""

from __future__ import annotations

from typing import Any

from .identity_runtime import build_running_identity_runtime_exports
from .probe_runtime import build_running_probe_runtime_exports


def build_running_runtime_exports(namespace: dict[str, Any]) -> dict[str, Any]:
    exports: dict[str, Any] = {}
    for builder in (
        build_running_identity_runtime_exports,
        build_running_probe_runtime_exports,
    ):
        new_exports = builder(namespace)
        exports.update(new_exports)
        namespace.update(new_exports)
    return exports
