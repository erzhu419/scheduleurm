"""Compatibility facade for dispatch command helpers."""

if __package__:
    from .scheduler_dispatch.command import *  # noqa: F401,F403
else:
    from scheduler_dispatch.command import *  # noqa: F401,F403
