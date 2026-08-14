"""Compatibility facade for dispatch loop helpers."""

if __package__:
    from .scheduler_dispatch.loop import *  # noqa: F401,F403
else:
    from scheduler_dispatch.loop import *  # noqa: F401,F403
