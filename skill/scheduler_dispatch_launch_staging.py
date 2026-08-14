"""Compatibility facade for dispatch launch staging helpers."""

if __package__:
    from .scheduler_dispatch.launch_staging import *  # noqa: F401,F403
else:
    from scheduler_dispatch.launch_staging import *  # noqa: F401,F403
