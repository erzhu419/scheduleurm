"""Compatibility facade for scheduler adopt probe helpers."""

if __package__:
    from .scheduler_adopt.probe import *  # noqa: F401,F403
else:
    from scheduler_adopt.probe import *  # noqa: F401,F403
