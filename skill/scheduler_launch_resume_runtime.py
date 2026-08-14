"""Compatibility facade for launch resume runtime helpers."""

if __package__:
    from .scheduler_launch.resume_runtime import *  # noqa: F401,F403
else:
    from scheduler_launch.resume_runtime import *  # noqa: F401,F403
