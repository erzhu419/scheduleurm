"""Result-sync state and transport helpers."""

from .state import ResultSyncDeps, sync_completed_results_outside_lock

__all__ = [
    "ResultSyncDeps",
    "sync_completed_results_outside_lock",
]
