"""Compatibility facade for scheduler control-process helpers."""

if __package__:
    from .scheduler_control.process import *  # noqa: F401,F403
else:
    from scheduler_control.process import *  # noqa: F401,F403
