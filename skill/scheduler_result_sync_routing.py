"""Compatibility facade for scheduler_result_sync.routing."""

if __package__:
    from .scheduler_result_sync.routing import *  # noqa: F401,F403
else:
    from scheduler_result_sync.routing import *  # noqa: F401,F403
