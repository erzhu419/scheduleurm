"""Compatibility facade for launch result helpers."""

if __package__:
    from .scheduler_launch.result import *  # noqa: F401,F403
else:
    from scheduler_launch.result import *  # noqa: F401,F403
