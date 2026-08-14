"""Compatibility facade for scheduler_external.reconcile."""

if __package__:
    from .scheduler_external.reconcile import *  # noqa: F401,F403
else:
    from scheduler_external.reconcile import *  # noqa: F401,F403
