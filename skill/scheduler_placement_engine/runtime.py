from __future__ import annotations

from typing import Any

from .gates_runtime import build_placement_gates_runtime_exports
from .policy_runtime import build_placement_policy_runtime_exports
from scheduler_resource.forensics_runtime import build_resource_forensics_runtime_exports


def build_placement_runtime_exports(namespace: dict[str, Any]) -> dict[str, Any]:
    exports: dict[str, Any] = {}
    for builder in (
        build_resource_forensics_runtime_exports,
        build_placement_gates_runtime_exports,
        build_placement_policy_runtime_exports,
    ):
        new_exports = builder(namespace)
        exports.update(new_exports)
        namespace.update(new_exports)
    return exports
