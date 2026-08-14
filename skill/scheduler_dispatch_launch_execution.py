"""Compatibility facade for dispatch launch execution helpers."""

if __package__:
    from .scheduler_dispatch.launch_execution import *  # noqa: F401,F403
else:
    from scheduler_dispatch.launch_execution import *  # noqa: F401,F403
