"""Compatibility facade for scheduler control command helpers."""

if __package__:
    from .scheduler_control.commands import *  # noqa: F401,F403
else:
    from scheduler_control.commands import *  # noqa: F401,F403
