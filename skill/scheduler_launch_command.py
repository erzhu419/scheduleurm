"""Compatibility facade for launch command helpers."""

if __package__:
    from .scheduler_launch.command import *  # noqa: F401,F403
else:
    from scheduler_launch.command import *  # noqa: F401,F403
