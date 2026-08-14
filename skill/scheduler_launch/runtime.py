"""Runtime-bound launch exports assembled by subsystem."""

from __future__ import annotations

from typing import Any

from .backend_runtime import build_launch_backend_runtime_exports
from .env_runtime import build_launch_env_runtime_exports
from .resume_runtime import build_launch_resume_runtime_exports


def build_launch_runtime_exports(namespace: dict[str, Any]) -> dict[str, Any]:
    exports: dict[str, Any] = {}
    for builder in (
        build_launch_resume_runtime_exports,
        build_launch_env_runtime_exports,
        build_launch_backend_runtime_exports,
    ):
        new_exports = builder(namespace)
        exports.update(new_exports)
        namespace.update(new_exports)
    return exports
