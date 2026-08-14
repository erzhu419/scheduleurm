"""Compatibility facade for scheduler history helpers."""

if __package__:
    from .scheduler_runtime.history import *  # noqa: F401,F403
else:
    from scheduler_runtime.history import *  # noqa: F401,F403
