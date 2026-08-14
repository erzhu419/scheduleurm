"""Data-motion runtime assembly for the scheduler facade."""

from __future__ import annotations

from typing import Any

from scheduler_launch.staging_runtime import build_launch_staging_runtime_exports
from scheduler_migration.runtime import build_migration_runtime_exports
from scheduler_result_sync.runtime import build_result_sync_runtime_exports
from scheduler_staging.runtime import build_staging_runtime_exports
from scheduler_transport.runtime import build_transport_runtime_exports


def build_data_motion_runtime_exports(namespace: dict[str, Any]) -> dict[str, Any]:
    """Install data-motion related runtime exports into ``namespace``."""
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
