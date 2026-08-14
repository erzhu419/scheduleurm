"""Compatibility facade for result-sync transport helpers."""

try:
    from scheduler_result_sync.executor import (
        ResultSyncExecutorDeps,
        WindowsResultSyncDeps,
        sync_one_result,
        sync_windows_result,
    )
except ModuleNotFoundError:
    from .scheduler_result_sync.executor import (
        ResultSyncExecutorDeps,
        WindowsResultSyncDeps,
        sync_one_result,
        sync_windows_result,
    )

__all__ = [
    "ResultSyncExecutorDeps",
    "WindowsResultSyncDeps",
    "sync_one_result",
    "sync_windows_result",
]
