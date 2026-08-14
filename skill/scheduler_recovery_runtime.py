"""Compatibility facade for scheduler_recovery.runtime."""

if __package__:
    from .scheduler_recovery.runtime import *  # noqa: F401,F403
else:
    from scheduler_recovery.runtime import *  # noqa: F401,F403
