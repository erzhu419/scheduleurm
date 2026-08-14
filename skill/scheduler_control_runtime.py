"""Compatibility facade for scheduler control runtime helpers."""

if __package__:
    from .scheduler_control.runtime import *  # noqa: F401,F403
else:
    from scheduler_control.runtime import *  # noqa: F401,F403
