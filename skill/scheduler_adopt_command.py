"""Compatibility facade for scheduler adopt command helpers."""

if __package__:
    from .scheduler_adopt.command import *  # noqa: F401,F403
else:
    from scheduler_adopt.command import *  # noqa: F401,F403
