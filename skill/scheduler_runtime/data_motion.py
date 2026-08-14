from __future__ import annotations

from typing import Any

from scheduler_launch.staging_runtime import build_launch_staging_runtime_exports
from scheduler_migration.runtime import build_migration_runtime_exports
from scheduler_result_sync.runtime import build_result_sync_runtime_exports
from scheduler_staging.runtime import build_staging_runtime_exports
from scheduler_transport.runtime import build_transport_runtime_exports


def build_data_motion_runtime_exports(namespace: dict[str, Any]) -> dict[str, Any]:
    """Assemble runtime wrappers for code/checkpoint/result data movement.

    The data-motion surface is intentionally wired once: transport, staging
    state, result sync, checkpoint migration, and launch staging all share the
    same in-memory staging caches. Individual wrappers resolve late-bound
    dependencies through ``namespace`` when called, so this single assembly point
    can stay before backend/dispatch wiring without reintroducing import cycles.
    """
    exports: dict[str, Any] = {}
    for builder in (
        build_staging_runtime_exports,
        build_transport_runtime_exports,
        build_result_sync_runtime_exports,
        build_migration_runtime_exports,
        build_launch_staging_runtime_exports,
    ):
        new_exports = builder(namespace)
        exports.update(new_exports)
        namespace.update(new_exports)
    return exports
