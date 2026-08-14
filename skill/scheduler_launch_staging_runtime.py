"""Compatibility facade for launch staging runtime helpers."""

if __package__:
    from .scheduler_launch.staging_runtime import *  # noqa: F401,F403
else:
    from scheduler_launch.staging_runtime import *  # noqa: F401,F403
