"""Compatibility facade for scheduler_recovery.queued_artifact_reconcile."""

if __package__:
    from .scheduler_recovery.queued_artifact_reconcile import *  # noqa: F401,F403
else:
    from scheduler_recovery.queued_artifact_reconcile import *  # noqa: F401,F403
