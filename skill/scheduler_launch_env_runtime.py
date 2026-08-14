"""Compatibility facade for launch environment runtime helpers."""

if __package__:
    from .scheduler_launch.env_runtime import *  # noqa: F401,F403
else:
    from scheduler_launch.env_runtime import *  # noqa: F401,F403
