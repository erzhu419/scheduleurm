"""Compatibility facade for launch commit helpers."""

if __package__:
    from .scheduler_launch.commit import *  # noqa: F401,F403
else:
    from scheduler_launch.commit import *  # noqa: F401,F403
