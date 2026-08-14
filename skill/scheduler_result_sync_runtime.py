"""Compatibility facade for scheduler_result_sync.runtime."""

if __package__:
    from .scheduler_result_sync.runtime import *  # noqa: F401,F403
else:
    from scheduler_result_sync.runtime import *  # noqa: F401,F403
