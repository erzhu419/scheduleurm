"""Runtime-bound transport exports assembled by subsystem."""

from __future__ import annotations

from typing import Any

from .path_runtime import build_transport_path_runtime_exports
from .relay_runtime import build_transport_relay_runtime_exports
from .remote_runtime import build_transport_remote_runtime_exports
from .windows_runtime import build_transport_windows_runtime_exports


def build_transport_runtime_exports(namespace: dict[str, Any]) -> dict[str, Any]:
    exports: dict[str, Any] = {}
    for builder in (
        build_transport_path_runtime_exports,
        build_transport_remote_runtime_exports,
        build_transport_windows_runtime_exports,
        build_transport_relay_runtime_exports,
    ):
        new_exports = builder(namespace)
        exports.update(new_exports)
        namespace.update(new_exports)
    return exports
