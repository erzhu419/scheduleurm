"""Compatibility facade for staging state helpers."""

try:
    from scheduler_staging.state import (
        can_migrate_to,
        conda_sync_ok,
        mark_launch_stage_cap_exceeded,
        mark_launch_stage_success,
        mark_staging_cache_success,
        recently_failed,
        record_conda_sync_failed,
        record_conda_sync_ok,
        record_staging_failure,
        stage_cwd_check,
        stage_failure_reason,
        ttl_cache_hit,
    )
except ModuleNotFoundError:
    from .scheduler_staging.state import (
        can_migrate_to,
        conda_sync_ok,
        mark_launch_stage_cap_exceeded,
        mark_launch_stage_success,
        mark_staging_cache_success,
        recently_failed,
        record_conda_sync_failed,
        record_conda_sync_ok,
        record_staging_failure,
        stage_cwd_check,
        stage_failure_reason,
        ttl_cache_hit,
    )

__all__ = [
    "can_migrate_to",
    "conda_sync_ok",
    "mark_launch_stage_cap_exceeded",
    "mark_launch_stage_success",
    "mark_staging_cache_success",
    "recently_failed",
    "record_conda_sync_failed",
    "record_conda_sync_ok",
    "record_staging_failure",
    "stage_cwd_check",
    "stage_failure_reason",
    "ttl_cache_hit",
]
