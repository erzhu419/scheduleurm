"""Compatibility facade for scheduler why command helpers."""

if __package__:
    from .scheduler_commands.why import *  # noqa: F401,F403
else:
    from scheduler_commands.why import *  # noqa: F401,F403
