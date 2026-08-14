"""Compatibility facade for launch environment helpers."""

if __package__:
    from .scheduler_launch.env import *  # noqa: F401,F403
else:
    from scheduler_launch.env import *  # noqa: F401,F403
